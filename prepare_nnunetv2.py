from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from tigerseg.labels import load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert prepared TIGER data to nnU-Net v2 RGB-PNG format.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="TigerCoarse")
    parser.add_argument("--label-mode", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.data_root / "metadata.csv")
    mask_dir = args.data_root / f"masks_{args.label_mode}"
    for filename in metadata["filename"].astype(str):
        stem = Path(filename).stem
        shutil.copy2(args.data_root / "images" / filename, images_tr / f"{stem}_0000.png")
        shutil.copy2(mask_dir / filename, labels_tr / filename)

    class_names = load_class_names(args.data_root, args.label_mode)
    dataset_json = {
        "name": dataset_dir.name,
        "description": "TIGER SQ-AI 2026 thoracoscopic semantic segmentation",
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {name: index for index, name in enumerate(class_names)},
        "numTraining": int(len(metadata)),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    print(f"nnU-Net raw dataset written to: {dataset_dir}")
    print("Next: nnUNetv2_plan_and_preprocess -d", args.dataset_id, "--verify_dataset_integrity")


if __name__ == "__main__":
    main()
