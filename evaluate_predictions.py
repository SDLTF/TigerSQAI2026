from __future__ import annotations

import argparse
import json
from pathlib import Path

from tigerseg.evaluation import evaluate_prediction_directory
from tigerseg.labels import load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate integer-PNG predictions with Dice, IoU/Jaccard, HD95, and ASSD."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--label-mode", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--surface-width",
        type=int,
        default=640,
        help="Canonical width used for HD95/ASSD. Use 0 together with --surface-height 0 for native resolution.",
    )
    parser.add_argument(
        "--surface-height",
        type=int,
        default=384,
        help="Canonical height used for HD95/ASSD. Use 0 together with --surface-width 0 for native resolution.",
    )
    return parser.parse_args()


def resolve_surface_size(width: int, height: int) -> tuple[int, int] | None:
    if width == 0 and height == 0:
        return None
    if width <= 0 or height <= 0:
        raise ValueError("surface width and height must both be positive, or both be zero")
    return width, height


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.data_root, args.label_mode)
    result = evaluate_prediction_directory(
        prediction_dir=args.pred_dir,
        ground_truth_dir=args.data_root / f"masks_{args.label_mode}",
        class_names=class_names,
        surface_size=resolve_surface_size(args.surface_width, args.surface_height),
    )

    output = args.output or args.pred_dir / "metrics.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    headline = {
        key: result[key]
        for key in ["dice", "iou", "jaccard", "hd95", "assd", "pixel_accuracy"]
    }
    print(json.dumps(headline, indent=2))
    print(f"Surface-distance unit: {result['surface_unit']}")
    print(f"Surface evaluation size: {result['surface_evaluation_size']}")


if __name__ == "__main__":
    main()
