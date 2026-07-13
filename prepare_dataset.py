from __future__ import annotations

import argparse
import csv
import io
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_IMAGE_SUFFIX = "(2)"
_FINE_SUFFIX = "(1)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the flattened TIGER SQ-AI archive for training.")
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_labelmap(zf: zipfile.ZipFile) -> list[dict[str, str]]:
    text = zf.read("labelmap.csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def build_color_lut(rows: list[dict[str, str]], mode: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    lut = np.full(1 << 24, 255, dtype=np.uint8)
    classes: dict[int, dict[str, object]] = {}

    if mode == "fine":
        for row in rows:
            class_id = int(row["fine_id"])
            rgb = (int(row["fine_r"]), int(row["fine_g"]), int(row["fine_b"]))
            classes[class_id] = {"id": class_id, "name": row["fine_name"], "rgb": rgb}
    else:
        for row in rows:
            class_id = int(row["merged_id"])
            rgb = (int(row["merged_r"]), int(row["merged_g"]), int(row["merged_b"]))
            classes[class_id] = {"id": class_id, "name": row["merged_name"], "rgb": rgb}

    for class_id, item in classes.items():
        red, green, blue = item["rgb"]  # type: ignore[misc]
        lut[(int(red) << 16) | (int(green) << 8) | int(blue)] = class_id
    return lut, [classes[i] for i in sorted(classes)]


def rgb_mask_to_ids(image_bytes: bytes, lut: np.ndarray, filename: str) -> Image.Image:
    rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint32)
    encoded = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    ids = lut[encoded]
    if np.any(ids == 255):
        invalid = np.unique(rgb[ids == 255].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown RGB values in {filename}: {invalid[:10].tolist()}")
    return Image.fromarray(ids, mode="L")


def normalize_base_name(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    if stem.endswith(_IMAGE_SUFFIX):
        return stem[: -len(_IMAGE_SUFFIX)] + ".png", "image"
    if stem.endswith(_FINE_SUFFIX):
        return stem[: -len(_FINE_SUFFIX)] + ".png", "fine"
    return stem + ".png", "coarse"


def parse_case_and_station(filename: str) -> tuple[str, str]:
    match = re.match(r"^(center_\d+_case_\d+)_(.+)\.png$", filename)
    if not match:
        raise ValueError(f"Unexpected filename: {filename}")
    return match.group(1), match.group(2)


def make_case_folds(cases: list[str], seed: int) -> dict[str, int]:
    if len(cases) != 10:
        raise ValueError(f"Expected 10 cases, found {len(cases)}: {cases}")
    shuffled = sorted(cases)
    random.Random(seed).shuffle(shuffled)
    return {case: index // 2 for index, case in enumerate(shuffled)}


def write_distribution(
    mask_dir: Path,
    class_table: list[dict[str, object]],
    output_csv: Path,
) -> None:
    num_classes = len(class_table)
    pixel_count = np.zeros(num_classes, dtype=np.int64)
    image_count = np.zeros(num_classes, dtype=np.int64)
    mask_paths = sorted(mask_dir.glob("*.png"))

    for path in tqdm(mask_paths, desc=f"Distribution {mask_dir.name}"):
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        counts = np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]
        pixel_count += counts
        image_count += counts > 0

    total_pixels = int(pixel_count.sum())
    rows = []
    for item in class_table:
        class_id = int(item["id"])
        rows.append(
            {
                "class_id": class_id,
                "class_name": item["name"],
                "pixel_count": int(pixel_count[class_id]),
                "pixel_fraction": float(pixel_count[class_id] / total_pixels),
                "image_count": int(image_count[class_id]),
                "image_fraction": float(image_count[class_id] / len(mask_paths)),
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite.")

    image_dir = output / "images"
    coarse_dir = output / "masks_coarse"
    fine_dir = output / "masks_fine"
    for path in (image_dir, coarse_dir, fine_dir):
        path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        label_rows = read_labelmap(zf)
        coarse_lut, coarse_classes = build_color_lut(label_rows, "coarse")
        fine_lut, fine_classes = build_color_lut(label_rows, "fine")

        grouped: dict[str, dict[str, str]] = defaultdict(dict)
        for name in zf.namelist():
            if not name.lower().endswith(".png"):
                continue
            base, role = normalize_base_name(name)
            if role in grouped[base]:
                raise ValueError(f"Duplicate {role} entry for {base}")
            grouped[base][role] = name

        if len(grouped) != 140:
            raise ValueError(f"Expected 140 image triplets, found {len(grouped)}")

        metadata: list[dict[str, object]] = []
        for base in tqdm(sorted(grouped), desc="Extracting and decoding"):
            entries = grouped[base]
            if set(entries) != {"image", "fine", "coarse"}:
                raise ValueError(f"Incomplete triplet for {base}: {entries}")

            image_bytes = zf.read(entries["image"])
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image for {base}, got mode={image.mode}")
            # Preserve the original PNG bytes instead of recompressing a 1080p image.
            (image_dir / base).write_bytes(image_bytes)

            coarse = rgb_mask_to_ids(zf.read(entries["coarse"]), coarse_lut, entries["coarse"])
            fine = rgb_mask_to_ids(zf.read(entries["fine"]), fine_lut, entries["fine"])
            if image.size != coarse.size or image.size != fine.size:
                raise ValueError(f"Geometry mismatch for {base}")
            coarse.save(coarse_dir / base)
            fine.save(fine_dir / base)

            case_id, station = parse_case_and_station(base)
            metadata.append(
                {
                    "filename": base,
                    "case_id": case_id,
                    "station": station,
                    "width": image.width,
                    "height": image.height,
                }
            )

        for metadata_name in ("labelmap.csv", "lymph_node_station_visibility.csv", "README.md"):
            (output / metadata_name).write_bytes(zf.read(metadata_name))

    cases = sorted({str(row["case_id"]) for row in metadata})
    fold_by_case = make_case_folds(cases, args.seed)
    for row in metadata:
        row["fold"] = fold_by_case[str(row["case_id"])]
    metadata_df = pd.DataFrame(metadata).sort_values(["case_id", "station"])
    metadata_df.to_csv(output / "metadata.csv", index=False)

    fold_rows = [{"case_id": case, "fold": fold} for case, fold in sorted(fold_by_case.items())]
    pd.DataFrame(fold_rows).to_csv(output / "case_folds.csv", index=False)

    write_distribution(coarse_dir, coarse_classes, output / "class_distribution_coarse.csv")
    write_distribution(fine_dir, fine_classes, output / "class_distribution_fine.csv")

    print(f"Prepared dataset: {output}")
    print(metadata_df.groupby(["width", "height"]).size().rename("images"))
    print(pd.DataFrame(fold_rows).sort_values(["fold", "case_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
