from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Fold-0 metrics from the three algorithms.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_metrics(path: Path, method: str) -> dict[str, object]:
    if not path.exists():
        return {
            "Method": method,
            "Status": "missing",
            "Dice": float("nan"),
            "IoU": float("nan"),
            "Jaccard": float("nan"),
            "95HD": float("nan"),
            "ASSD": float("nan"),
            "MetricsPath": str(path),
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "Method": method,
        "Status": "complete",
        "Dice": float(data["dice"]),
        "IoU": float(data["iou"]),
        "Jaccard": float(data["jaccard"]),
        "95HD": float(data["hd95"]),
        "ASSD": float(data["assd"]),
        "BestEpoch": data.get("best_epoch"),
        "MetricsPath": str(path),
    }


def main() -> None:
    args = parse_args()
    starter = args.project_root / "tiger_segmentation_starter"
    output = args.output or starter / "runs" / "fold0_algorithm_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        load_metrics(starter / "runs" / "unet_coarse" / "fold_0" / "metrics.json", "U-Net"),
        load_metrics(
            starter / "runs" / "nnunet_100e_coarse" / "fold_0" / "metrics.json",
            "nnU-Net (100 epochs)",
        ),
        load_metrics(
            starter / "runs" / "pranet_mc_coarse" / "fold_0" / "metrics.json",
            "PraNet-MC",
        ),
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
