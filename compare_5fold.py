from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the three five-fold summary CSV files."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    starter = args.project_root / "tiger_segmentation_starter"
    runs = starter / "runs"

    sources = [
        runs / "unet_coarse" / "spreadsheet_summary.csv",
        runs / "nnunet_100e_coarse" / "spreadsheet_summary.csv",
        runs / "pranet_mc_coarse" / "spreadsheet_summary.csv",
    ]

    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing summary files:\n" + "\n".join(str(path) for path in missing)
        )

    table = pd.concat(
        [pd.read_csv(path) for path in sources],
        ignore_index=True,
    )

    display_columns = [
        "Method",
        "Dice_cell",
        "IoU_cell",
        "Jaccard_cell",
        "95HD_cell",
        "ASSD_cell",
    ]

    output = runs / "fivefold_algorithm_comparison.csv"
    table[display_columns].to_csv(output, index=False, encoding="utf-8-sig")

    print(table[display_columns].to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
