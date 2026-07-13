from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 5-fold metrics using one row for the spreadsheet.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--method", default="UNet")
    return parser.parse_args()


def metric_cell(series: pd.Series) -> str:
    return f"{series.mean():.4f} ± {series.std(ddof=1):.4f}"


def main() -> None:
    args = parse_args()
    rows = []
    for fold in range(5):
        path = args.runs_root / f"fold_{fold}" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "fold": fold,
                "Dice": float(data["dice"]),
                "IoU": float(data["iou"]),
                "Jaccard": float(data["jaccard"]),
                "95HD": float(data["hd95"]),
                "ASSD": float(data["assd"]),
            }
        )

    folds = pd.DataFrame(rows)
    folds.to_csv(args.runs_root / "fold_metrics.csv", index=False)

    summary: dict[str, object] = {"Method": args.method}
    for metric in ["Dice", "IoU", "Jaccard", "95HD", "ASSD"]:
        summary[f"{metric}_mean"] = folds[metric].mean()
        summary[f"{metric}_std"] = folds[metric].std(ddof=1)
        summary[f"{metric}_cell"] = metric_cell(folds[metric])

    pd.DataFrame([summary]).to_csv(args.runs_root / "spreadsheet_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
