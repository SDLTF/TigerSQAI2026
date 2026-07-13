from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


@dataclass
class SegmentationMetrics:
    confusion: np.ndarray
    dice_per_class: np.ndarray
    iou_per_class: np.ndarray
    dice_macro_foreground: float
    iou_macro_foreground: float
    pixel_accuracy: float

    @property
    def jaccard_macro_foreground(self) -> float:
        return self.iou_macro_foreground


@dataclass
class SurfaceDistanceMetrics:
    hd95_per_class: np.ndarray
    assd_per_class: np.ndarray
    valid_pairs_per_class: np.ndarray
    one_empty_pairs_per_class: np.ndarray
    hd95_macro_foreground: float
    assd_macro_foreground: float
    unit: str
    evaluation_size: tuple[int, int] | None


def update_confusion_matrix(
    confusion: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    prediction = prediction.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + prediction[valid]
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).to(confusion.device)


def metrics_from_confusion(confusion: torch.Tensor | np.ndarray) -> SegmentationMetrics:
    matrix = confusion.detach().cpu().numpy() if isinstance(confusion, torch.Tensor) else confusion.copy()
    matrix = matrix.astype(np.float64, copy=False)

    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive

    dice_denominator = 2.0 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative

    dice = np.divide(
        2.0 * true_positive,
        dice_denominator,
        out=np.full_like(true_positive, np.nan),
        where=dice_denominator > 0,
    )
    iou = np.divide(
        true_positive,
        iou_denominator,
        out=np.full_like(true_positive, np.nan),
        where=iou_denominator > 0,
    )

    foreground_dice = dice[1:]
    foreground_iou = iou[1:]
    accuracy = float(true_positive.sum() / matrix.sum()) if matrix.sum() > 0 else float("nan")

    return SegmentationMetrics(
        confusion=matrix.astype(np.int64),
        dice_per_class=dice,
        iou_per_class=iou,
        dice_macro_foreground=float(np.nanmean(foreground_dice)),
        iou_macro_foreground=float(np.nanmean(foreground_iou)),
        pixel_accuracy=accuracy,
    )


def _letterbox_label(mask: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize an integer label map with nearest-neighbour interpolation and zero padding."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {mask.shape}")
    height, width = mask.shape
    scale = min(target_width / width, target_height / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
        (new_width, new_height), Image.Resampling.NEAREST
    )
    canvas = Image.new("L", (target_width, target_height), color=0)
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner surface using 4-connectivity."""
    structure = generate_binary_structure(rank=2, connectivity=1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def symmetric_surface_distances(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, bool]:
    """
    Compute HD95 and ASSD for one binary prediction/target pair.

    Returns ``(hd95, assd, one_empty)``. If both masks are empty, both scores are NaN.
    If exactly one mask is empty, the image diagonal is used as a finite penalty.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")

    if np.array_equal(prediction, target):
        if prediction.any():
            return 0.0, 0.0, False
        return float("nan"), float("nan"), False

    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())
    if not prediction_nonempty and not target_nonempty:
        return float("nan"), float("nan"), False

    if prediction_nonempty != target_nonempty:
        height, width = prediction.shape
        penalty = float(np.hypot(max(0, height - 1), max(0, width - 1)))
        return penalty, penalty, True

    prediction_surface = _surface(prediction)
    target_surface = _surface(target)

    distance_to_target = distance_transform_edt(~target_surface)[prediction_surface]
    distance_to_prediction = distance_transform_edt(~prediction_surface)[target_surface]
    distances = np.concatenate((distance_to_target, distance_to_prediction)).astype(np.float64, copy=False)

    hd95 = float(np.percentile(distances, 95))
    assd = float(distances.mean())
    return hd95, assd, False


class SurfaceDistanceAccumulator:
    """
    Accumulate per-image, per-class HD95 and ASSD and then macro-average by class.

    Both-empty class/image pairs are undefined and skipped. Exactly-one-empty pairs receive
    the evaluation-image diagonal as a penalty, preventing missing small structures from being
    silently rewarded. Headline scores exclude class 0 (background).
    """

    def __init__(
        self,
        num_classes: int,
        evaluation_size: tuple[int, int] | None = (640, 384),
    ) -> None:
        self.num_classes = num_classes
        self.evaluation_size = evaluation_size
        self._hd95_sum = np.zeros(num_classes, dtype=np.float64)
        self._assd_sum = np.zeros(num_classes, dtype=np.float64)
        self._valid_count = np.zeros(num_classes, dtype=np.int64)
        self._one_empty_count = np.zeros(num_classes, dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.ndim == 2:
            prediction = prediction[None, ...]
        if target.ndim == 2:
            target = target[None, ...]
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")

        for prediction_item, target_item in zip(prediction, target, strict=True):
            if self.evaluation_size is not None:
                target_width, target_height = self.evaluation_size
                prediction_item = _letterbox_label(prediction_item, target_width, target_height)
                target_item = _letterbox_label(target_item, target_width, target_height)

            for class_id in range(self.num_classes):
                hd95, assd, one_empty = symmetric_surface_distances(
                    prediction_item == class_id,
                    target_item == class_id,
                )
                if np.isnan(hd95):
                    continue
                self._hd95_sum[class_id] += hd95
                self._assd_sum[class_id] += assd
                self._valid_count[class_id] += 1
                if one_empty:
                    self._one_empty_count[class_id] += 1

    def compute(self) -> SurfaceDistanceMetrics:
        hd95 = np.divide(
            self._hd95_sum,
            self._valid_count,
            out=np.full(self.num_classes, np.nan, dtype=np.float64),
            where=self._valid_count > 0,
        )
        assd = np.divide(
            self._assd_sum,
            self._valid_count,
            out=np.full(self.num_classes, np.nan, dtype=np.float64),
            where=self._valid_count > 0,
        )
        return SurfaceDistanceMetrics(
            hd95_per_class=hd95,
            assd_per_class=assd,
            valid_pairs_per_class=self._valid_count.copy(),
            one_empty_pairs_per_class=self._one_empty_count.copy(),
            hd95_macro_foreground=float(np.nanmean(hd95[1:])),
            assd_macro_foreground=float(np.nanmean(assd[1:])),
            unit="pixel",
            evaluation_size=self.evaluation_size,
        )
