from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from tigerseg.dataset import TigerSegmentationDataset, restore_label_from_letterbox
from tigerseg.evaluation import evaluate_prediction_directory
from tigerseg.labels import load_class_names, load_palette
from tigerseg.losses import CrossEntropyDiceLoss
from tigerseg.metrics import metrics_from_confusion, update_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a patient-level 5-fold U-Net baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label-mode", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_class_weights(dataset: TigerSegmentationDataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for record in tqdm(dataset.records, desc="Class weights"):
        filename = str(record["filename"])
        mask = np.asarray(Image.open(dataset.mask_dir / filename).convert("L"), dtype=np.uint8)
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]

    frequency = counts / max(1, counts.sum())
    weights = 1.0 / np.sqrt(np.maximum(frequency, 1e-12))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc="train" if training else "val", leave=False)
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

            batch_size = images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            prediction = logits.argmax(dim=1)
            update_confusion_matrix(confusion, prediction, masks, num_classes)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

    metrics = metrics_from_confusion(confusion)
    result: dict[str, object] = {
        "loss": total_loss / max(1, total_samples),
        "dice": metrics.dice_macro_foreground,
        "iou": metrics.iou_macro_foreground,
        "jaccard": metrics.jaccard_macro_foreground,
        "pixel_accuracy": metrics.pixel_accuracy,
        "dice_per_class": metrics.dice_per_class.tolist(),
        "iou_per_class": metrics.iou_per_class.tolist(),
        "confusion": metrics.confusion.tolist(),
    }
    return float(result["loss"]), result


def colorize(mask: np.ndarray, palette: list[tuple[int, int, int]]) -> Image.Image:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(palette):
        output[mask == class_id] = rgb
    return Image.fromarray(output, mode="RGB")


def save_previews(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    palette: list[tuple[int, int, int]],
    output_dir: Path,
    amp_enabled: bool,
    limit: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for images, masks, filenames in loader:
            images_gpu = images.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(images_gpu).argmax(dim=1).cpu().numpy()
            images_np = images.numpy().transpose(0, 2, 3, 1)
            images_np = np.clip((images_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
            masks_np = masks.numpy()

            for image_np, ground_truth, prediction, filename in zip(
                images_np, masks_np, predictions, filenames, strict=True
            ):
                original = Image.fromarray(image_np, mode="RGB")
                gt_color = colorize(ground_truth, palette)
                pred_color = colorize(prediction, palette)
                canvas = Image.new("RGB", (original.width * 3, original.height))
                canvas.paste(original, (0, 0))
                canvas.paste(gt_color, (original.width, 0))
                canvas.paste(pred_color, (original.width * 2, 0))
                canvas.save(output_dir / filename)
                saved += 1
                if saved >= limit:
                    return


def save_native_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_dir: Path,
    output_dir: Path,
    amp_enabled: bool,
) -> None:
    """Save predictions at each source image's original resolution for shared evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.png"):
        stale_path.unlink()

    model.eval()
    with torch.no_grad():
        progress = tqdm(loader, desc="save predictions", leave=False)
        for images, _, filenames in progress:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(images).argmax(dim=1).cpu().numpy()

            for prediction, filename in zip(predictions, filenames, strict=True):
                with Image.open(image_dir / filename) as original_image:
                    original_width, original_height = original_image.size
                restored = restore_label_from_letterbox(
                    prediction,
                    original_width=original_width,
                    original_height=original_height,
                )
                Image.fromarray(restored, mode="L").save(output_dir / filename)


def resolve_surface_size(width: int, height: int) -> tuple[int, int] | None:
    if width == 0 and height == 0:
        return None
    if width <= 0 or height <= 0:
        raise ValueError("surface width and height must both be positive, or both be zero")
    return width, height


def main() -> None:
    args = parse_args()
    set_seed(args.seed + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    surface_size = resolve_surface_size(args.surface_width, args.surface_height)

    class_names = load_class_names(args.data_root, args.label_mode)
    num_classes = len(class_names)
    output = args.output or Path("runs") / f"unet_{args.label_mode}" / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)

    train_dataset = TigerSegmentationDataset(
        args.data_root, args.fold, "train", args.label_mode, args.width, args.height, augment=True
    )
    val_dataset = TigerSegmentationDataset(
        args.data_root, args.fold, "val", args.label_mode, args.width, args.height, augment=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = estimate_class_weights(train_dataset, num_classes).to(device)
        print("Class weights:", class_weights.detach().cpu().numpy())
    criterion = CrossEntropyDiceLoss(class_weights=class_weights).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_dice = -math.inf
    stale_epochs = 0
    history: list[dict[str, object]] = []
    best_path = output / "best.pt"

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output"] = str(output)
    config["device"] = str(device)
    config["num_classes"] = num_classes
    config["class_names"] = class_names
    config["surface_size"] = surface_size
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, num_classes, optimizer, scaler, amp_enabled
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device, num_classes, None, scaler, amp_enabled
        )
        val_dice = float(val_metrics["dice"])
        scheduler.step(val_dice)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_jaccard": val_metrics["jaccard"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(row)
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} dice {float(train_metrics['dice']):.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} iou {float(val_metrics['iou']):.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": class_names,
                    "config": config,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    prediction_dir = output / "predictions"
    save_native_predictions(
        model,
        val_loader,
        device,
        args.data_root / "images",
        prediction_dir,
        amp_enabled,
    )
    final_metrics = evaluate_prediction_directory(
        prediction_dir=prediction_dir,
        ground_truth_dir=args.data_root / f"masks_{args.label_mode}",
        class_names=class_names,
        surface_size=surface_size,
    )
    final_metrics["best_epoch"] = checkpoint["epoch"]
    (output / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2, allow_nan=True), encoding="utf-8"
    )

    save_previews(
        model,
        val_loader,
        device,
        load_palette(args.data_root, args.label_mode),
        output / "previews",
        amp_enabled,
    )
    print(
        f"Best epoch {checkpoint['epoch']} | Dice {final_metrics['dice']:.4f} | "
        f"IoU {final_metrics['iou']:.4f} | HD95 {final_metrics['hd95']:.4f} px | "
        f"ASSD {final_metrics['assd']:.4f} px"
    )
    print(f"Best model, predictions, and metrics saved to {output}")


if __name__ == "__main__":
    main()
