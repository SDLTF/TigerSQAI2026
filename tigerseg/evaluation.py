from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from tigerseg.metrics import (
    SurfaceDistanceAccumulator,
    metrics_from_confusion,
    update_confusion_matrix,
)


def evaluate_prediction_directory(
    prediction_dir: str | Path,
    ground_truth_dir: str | Path,
    class_names: list[str],
    surface_size: tuple[int, int] | None = (640, 384),
) -> dict[str, object]:
    prediction_dir = Path(prediction_dir)
    ground_truth_dir = Path(ground_truth_dir)
    num_classes = len(class_names)

    prediction_paths = sorted(prediction_dir.glob("*.png"))
    if not prediction_paths:
        raise FileNotFoundError(f"No PNG predictions in {prediction_dir}")

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    surface_accumulator = SurfaceDistanceAccumulator(num_classes, evaluation_size=surface_size)

    for prediction_path in tqdm(prediction_paths, desc="surface evaluation", leave=False):
        target_path = ground_truth_dir / prediction_path.name
        if not target_path.exists():
            raise FileNotFoundError(target_path)

        prediction_array = np.asarray(Image.open(prediction_path).convert("L"), dtype=np.int64)
        target_array = np.asarray(Image.open(target_path).convert("L"), dtype=np.int64)
        if prediction_array.shape != target_array.shape:
            raise ValueError(
                f"Shape mismatch for {prediction_path.name}: "
                f"{prediction_array.shape} vs {target_array.shape}"
            )
        if prediction_array.min() < 0 or prediction_array.max() >= num_classes:
            raise ValueError(
                f"Prediction {prediction_path.name} contains IDs outside 0..{num_classes - 1}"
            )

        prediction_tensor = torch.from_numpy(prediction_array.copy())
        target_tensor = torch.from_numpy(target_array.copy())
        update_confusion_matrix(confusion, prediction_tensor, target_tensor, num_classes)
        surface_accumulator.update(prediction_array, target_array)

    overlap = metrics_from_confusion(confusion)
    surface = surface_accumulator.compute()

    per_class = []
    for class_id, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "dice": overlap.dice_per_class[class_id],
                "iou": overlap.iou_per_class[class_id],
                "jaccard": overlap.iou_per_class[class_id],
                "hd95": surface.hd95_per_class[class_id],
                "assd": surface.assd_per_class[class_id],
                "surface_valid_pairs": int(surface.valid_pairs_per_class[class_id]),
                "surface_one_empty_pairs": int(surface.one_empty_pairs_per_class[class_id]),
            }
        )

    return {
        "dice": overlap.dice_macro_foreground,
        "iou": overlap.iou_macro_foreground,
        "jaccard": overlap.jaccard_macro_foreground,
        "hd95": surface.hd95_macro_foreground,
        "assd": surface.assd_macro_foreground,
        "pixel_accuracy": overlap.pixel_accuracy,
        "surface_unit": surface.unit,
        "surface_evaluation_size": (
            None
            if surface.evaluation_size is None
            else {"width": surface.evaluation_size[0], "height": surface.evaluation_size[1]}
        ),
        "surface_empty_policy": {
            "both_empty": "undefined and excluded",
            "exactly_one_empty": "penalized by evaluation-image diagonal",
        },
        "class_names": class_names,
        "dice_per_class": overlap.dice_per_class.tolist(),
        "iou_per_class": overlap.iou_per_class.tolist(),
        "hd95_per_class": surface.hd95_per_class.tolist(),
        "assd_per_class": surface.assd_per_class.tolist(),
        "surface_valid_pairs_per_class": surface.valid_pairs_per_class.tolist(),
        "surface_one_empty_pairs_per_class": surface.one_empty_pairs_per_class.tolist(),
        "confusion": overlap.confusion.tolist(),
        "per_class": per_class,
        "num_evaluated_images": len(prediction_paths),
    }
