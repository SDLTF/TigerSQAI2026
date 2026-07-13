# TIGER SQ-AI 2026 segmentation starter

This project is a reproducible starting point for **Task 1: 16-class coarse semantic segmentation**.
It prepares the unusual flattened ZIP, creates leakage-free patient-level folds, trains a U-Net baseline,
converts the same data to nnU-Net v2 format, and reports Dice, IoU/Jaccard, HD95, and ASSD.

## What was found in the uploaded ZIP

The archive contains 424 entries:

- 140 thoracoscopic RGB images
- 140 fine RGB masks
- 140 coarse RGB masks
- `README.md`, `labelmap.csv`, `lymph_node_station_visibility.csv`

The folders described by the original README were flattened. The suffixes identify the file role:

- `name(2).png`: RGB image
- `name(1).png`: fine mask with 31 label IDs encoded as RGB
- `name.png`: coarse mask with 16 label IDs encoded as RGB

`prepare_dataset.py` restores a clean directory structure and converts RGB masks into one-channel integer-ID PNG masks.

## Why the default task is coarse 16-class segmentation

The coarse masks contain all 16 classes. In the fine masks, three declared labels have zero annotated pixels in Part 1:

- Right bronchial artery
- Gastric conduit
- Omentum

The dataset is only 140 frames from 10 patients. Image-level random splitting would leak patients across training and validation. The preparation script creates five folds with 8 training patients and 2 validation patients per fold.

## 1. Environment

Install PyTorch for your CUDA environment first, then run:

```powershell
pip install -r requirements.txt
```

`scipy` is required for surface-distance calculations.

## 2. Prepare the uploaded ZIP

```powershell
python prepare_dataset.py `
  --zip "D:\path\to\TigerSQ-AI-2026.zip" `
  --output "D:\path\to\TigerSQ-AI-2026-prepared"
```

Output:

```text
TigerSQ-AI-2026-prepared/
├── images/
├── masks_coarse/              # uint8 values 0..15
├── masks_fine/                # uint8 values 0..30
├── metadata.csv               # includes patient-level fold
├── case_folds.csv
├── class_distribution_coarse.csv
├── class_distribution_fine.csv
├── labelmap.csv
└── lymph_node_station_visibility.csv
```

## 3. Train one U-Net fold

Recommended initial settings for an 8 GB laptop GPU:

```powershell
python train_unet.py `
  --data-root "D:\path\to\TigerSQ-AI-2026-prepared" `
  --label-mode coarse `
  --fold 0 `
  --encoder resnet34 `
  --epochs 80 `
  --batch-size 2 `
  --workers 4
```

The default network input is 640×384. The original aspect ratio is preserved and padded rather than distorted to a square. Horizontal flipping is disabled because the anatomy and fine labels contain left/right semantics.

Each fold produces:

```text
runs/unet_coarse/fold_0/
├── best.pt
├── config.json
├── history.csv
├── metrics.json
├── predictions/               # integer masks restored to original image sizes
└── previews/                  # input | ground truth | prediction
```

Training and early stopping use validation foreground macro-Dice. HD95 and ASSD are calculated only for the final best checkpoint because repeated surface-distance transforms at every epoch are unnecessarily expensive.

## 4. Metric protocol

### Overlap metrics

For each class:

```text
Dice = 2TP / (2TP + FP + FN)
IoU = Jaccard = TP / (TP + FP + FN)
```

The headline Dice and IoU/Jaccard are macro averages over foreground classes. Class 0 (background) is excluded.

### Surface metrics

For one binary class mask, let `S(P)` and `S(G)` denote the predicted and ground-truth surfaces. The symmetric surface-distance set contains distances in both directions:

```text
D = {d(p, S(G)) : p in S(P)} union {d(g, S(P)) : g in S(G)}
```

The project reports:

```text
HD95 = 95th percentile of D
ASSD = mean of D
```

For multiclass segmentation, HD95 and ASSD are calculated per image and per class, averaged over images for each class, and then macro-averaged over foreground classes.

Empty-mask policy:

- prediction empty and ground truth empty: undefined, excluded from that class average
- exactly one mask empty: penalized by the diagonal length of the evaluation canvas

This policy prevents a model from receiving an artificially good boundary score when it completely misses a small anatomical structure.

### Surface-distance resolution and unit

Physical pixel spacing is not provided in this natural-image dataset, so HD95 and ASSD are reported in **pixels**, not millimetres.

By default, both masks are mapped with nearest-neighbour letterboxing to a common 640×384 canvas before surface measurement. This provides one consistent scale for U-Net, nnU-Net, and PraNet predictions and substantially reduces metric-computation time.

Use native image resolution only when every method follows that same protocol:

```powershell
python evaluate_predictions.py `
  --data-root "D:\path\to\TigerSQ-AI-2026-prepared" `
  --pred-dir "D:\path\to\predictions" `
  --label-mode coarse `
  --surface-width 0 `
  --surface-height 0
```

Do not mix 640×384 HD95/ASSD values with native-resolution HD95/ASSD values in one comparison table.

## 5. Run all five U-Net folds

```powershell
.\scripts\run_unet_5fold.ps1 -DataRoot "D:\path\to\TigerSQ-AI-2026-prepared"
```

The spreadsheet-ready output is:

```text
runs/unet_coarse/spreadsheet_summary.csv
```

It contains five-fold mean ± sample standard deviation for:

- Dice ↑
- IoU ↑
- Jaccard ↑
- 95HD ↓
- ASSD ↓

A recommended table is:

| Method | Dice ↑ | IoU ↑ | Jaccard ↑ | 95HD ↓ | ASSD ↓ |
|---|---:|---:|---:|---:|---:|
| PraNet |  |  |  |  |  |
| U-Net |  |  |  |  |  |
| nnU-Net |  |  |  |  |  |

IoU and Jaccard remain mathematically identical; keep both only when the required spreadsheet explicitly contains both columns.

## 6. Evaluate predictions from another model

The prediction directory must contain one single-channel integer PNG per validation image, with the same filename and original image size as the prepared ground-truth mask.

```powershell
python evaluate_predictions.py `
  --data-root "D:\path\to\TigerSQ-AI-2026-prepared" `
  --pred-dir "D:\path\to\model_predictions" `
  --label-mode coarse `
  --surface-width 640 `
  --surface-height 384
```

The resulting `metrics.json` contains:

- foreground macro Dice, IoU/Jaccard, HD95, ASSD
- per-class Dice, IoU, HD95, ASSD
- number of valid class/image surface pairs
- number of exactly-one-empty penalties per class
- confusion matrix
- surface resolution and empty-mask policy

Use this same evaluator for U-Net, nnU-Net, and PraNet outputs so that all rows use exactly the same metric implementation.

## 7. Prepare nnU-Net v2

nnU-Net should be run through its own framework rather than reimplemented as a normal model class.

```powershell
pip install nnunetv2

$env:nnUNet_raw = "D:\tiger_nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\tiger_nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "D:\tiger_nnunet\nnUNet_results"

python prepare_nnunetv2.py `
  --data-root "D:\path\to\TigerSQ-AI-2026-prepared" `
  --nnunet-raw $env:nnUNet_raw `
  --dataset-id 501 `
  --dataset-name TigerCoarse

nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

python write_nnunet_splits.py `
  --data-root "D:\path\to\TigerSQ-AI-2026-prepared" `
  --nnunet-preprocessed $env:nnUNet_preprocessed `
  --dataset-id 501 `
  --dataset-name TigerCoarse

.\scripts\run_nnunet_5fold.ps1 -DatasetId 501
```

The custom `splits_final.json` is essential. Without it, default image-level folds can put different frames from the same patient into both training and validation.

After nnU-Net prediction export, evaluate its integer masks with `evaluate_predictions.py` using the same 640×384 surface protocol.

## 8. About PraNet

Do not silently replace PraNet's final convolution with 16 channels and call it an official multiclass PraNet. Original PraNet is a binary foreground/background polyp-segmentation architecture, and its reverse-attention mechanism assumes that setup.

For a valid experiment, choose one of these protocols:

1. **16-class task:** compare U-Net, nnU-Net, and an explicitly multiclass PraNet-V2/DSRA implementation; label it accurately in the table.
2. **Binary lymphatic-tissue task:** convert `coarse_id == 11` to foreground and compare original PraNet, U-Net, and nnU-Net as binary models.

Whichever protocol is selected, export masks and run the same evaluator for every method.
