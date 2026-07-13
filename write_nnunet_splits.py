from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write leakage-free patient-level splits_final.json for nnU-Net v2.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="TigerCoarse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.data_root / "metadata.csv")
    metadata["stem"] = metadata["filename"].map(lambda value: Path(str(value)).stem)

    splits = []
    for fold in range(5):
        train = sorted(metadata.loc[metadata["fold"] != fold, "stem"].astype(str).tolist())
        val = sorted(metadata.loc[metadata["fold"] == fold, "stem"].astype(str).tolist())
        splits.append({"train": train, "val": val})

    dataset_dir = args.nnunet_preprocessed / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output = dataset_dir / "splits_final.json"
    output.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print("Validation images per fold:", [len(item["val"]) for item in splits])


if __name__ == "__main__":
    main()
