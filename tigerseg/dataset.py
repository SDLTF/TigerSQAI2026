from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

LabelMode = Literal["coarse", "fine"]

_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_geometry(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    """Return ``new_width, new_height, left, top`` for aspect-ratio-preserving padding."""
    scale = min(target_width / original_width, target_height / original_height)
    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    return new_width, new_height, left, top


def resize_and_pad_pair(
    image: Image.Image,
    mask: Image.Image,
    target_width: int,
    target_height: int,
) -> tuple[Image.Image, Image.Image]:
    """Preserve the original geometry and pad to a fixed network input size."""
    width, height = image.size
    new_width, new_height, left, top = letterbox_geometry(
        width, height, target_width, target_height
    )

    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

    image_canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    mask_canvas = Image.new("L", (target_width, target_height), color=0)
    image_canvas.paste(image, (left, top))
    mask_canvas.paste(mask, (left, top))
    return image_canvas, mask_canvas


def restore_label_from_letterbox(
    label: np.ndarray,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """Undo dataset letterboxing and restore an integer prediction to the original image size."""
    if label.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {label.shape}")
    target_height, target_width = label.shape
    new_width, new_height, left, top = letterbox_geometry(
        original_width,
        original_height,
        target_width,
        target_height,
    )
    cropped = label[top : top + new_height, left : left + new_width]
    restored = Image.fromarray(cropped.astype(np.uint8), mode="L").resize(
        (original_width, original_height), Image.Resampling.NEAREST
    )
    return np.asarray(restored, dtype=np.uint8)


def _augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Conservative augmentations; no horizontal flip because labels encode anatomy."""
    if random.random() < 0.65:
        angle = random.uniform(-5.0, 5.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

    if random.random() < 0.75:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.90, 1.10))

    if random.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
    return image, mask


class TigerSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        data_root: str | Path,
        fold: int,
        split: Literal["train", "val"],
        label_mode: LabelMode = "coarse",
        image_width: int = 640,
        image_height: int = 384,
        augment: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.label_mode = label_mode
        self.image_width = image_width
        self.image_height = image_height
        self.augment = augment and split == "train"

        metadata = pd.read_csv(self.data_root / "metadata.csv")
        if split == "train":
            metadata = metadata[metadata["fold"] != fold]
        else:
            metadata = metadata[metadata["fold"] == fold]
        self.records = metadata.sort_values("filename").to_dict("records")

        if not self.records:
            raise ValueError(f"No samples found for split={split!r}, fold={fold}")

        self.image_dir = self.data_root / "images"
        self.mask_dir = self.data_root / f"masks_{label_mode}"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        filename = str(record["filename"])
        image = Image.open(self.image_dir / filename).convert("RGB")
        mask = Image.open(self.mask_dir / filename).convert("L")

        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch for {filename}: {image.size} vs {mask.size}")

        image, mask = resize_and_pad_pair(image, mask, self.image_width, self.image_height)
        if self.augment:
            image, mask = _augment_pair(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - _IMAGENET_MEAN) / _IMAGENET_STD
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()).long()
        return image_tensor, mask_tensor, filename
