from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

LabelMode = Literal["coarse", "fine"]


def load_class_table(data_root: str | Path, label_mode: LabelMode) -> list[dict[str, object]]:
    """Load deduplicated class definitions from labelmap.csv."""
    data_root = Path(data_root)
    rows: list[dict[str, str]] = []
    with (data_root / "labelmap.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if label_mode == "fine":
        return [
            {
                "id": int(row["fine_id"]),
                "name": row["fine_name"],
                "rgb": (int(row["fine_r"]), int(row["fine_g"]), int(row["fine_b"])),
            }
            for row in rows
        ]

    unique: dict[int, dict[str, object]] = {}
    for row in rows:
        class_id = int(row["merged_id"])
        unique[class_id] = {
            "id": class_id,
            "name": row["merged_name"],
            "rgb": (int(row["merged_r"]), int(row["merged_g"]), int(row["merged_b"])),
        }
    return [unique[i] for i in sorted(unique)]


def load_class_names(data_root: str | Path, label_mode: LabelMode) -> list[str]:
    return [str(row["name"]) for row in load_class_table(data_root, label_mode)]


def load_palette(data_root: str | Path, label_mode: LabelMode) -> list[tuple[int, int, int]]:
    return [tuple(row["rgb"]) for row in load_class_table(data_root, label_mode)]  # type: ignore[arg-type]
