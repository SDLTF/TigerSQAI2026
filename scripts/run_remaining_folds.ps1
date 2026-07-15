param(
    [string]$ProjectRoot = "D:\Code\TigerSQ-AI-2026",

    [ValidateRange(1, 4)]
    [int]$StartFold = 1,

    [ValidateRange(1, 4)]
    [int]$EndFold = 4,

    [ValidateRange(0, 32)]
    [int]$Workers = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($StartFold -gt $EndFold) {
    throw "StartFold cannot be greater than EndFold."
}

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------

$VenvRoot = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"
$NnUNetTrainExe = Join-Path $VenvRoot "Scripts\nnUNetv2_train.exe"

$StarterRoot = Join-Path $ProjectRoot "tiger_segmentation_starter"
$DataRoot = Join-Path $ProjectRoot "TigerSQ-AI-2026-prepared"

$TrainUNet = Join-Path $StarterRoot "train_unet.py"
$TrainPraNet = Join-Path $StarterRoot "train_pranet_multiclass.py"
$EvaluatePredictions = Join-Path $StarterRoot "evaluate_predictions.py"
$SummarizeResults = Join-Path $StarterRoot "summarize_results.py"
$CompareFiveFold = Join-Path $StarterRoot "compare_5fold.py"

$NnUNetRoot = Join-Path $ProjectRoot "nnunet"
$env:nnUNet_raw = Join-Path $NnUNetRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $NnUNetRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $NnUNetRoot "nnUNet_results"

$env:nnUNet_n_proc_DA = "$Workers"
$env:nnUNet_compile = "false"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"

$DatasetId = 501
$DatasetName = "Dataset501_TigerCoarse"
$NnUNetTrainer = "nnUNetTrainer_100epochs"
$NnUNetConfiguration = "2d"
$NnUNetPlans = "nnUNetPlans"

$UNetRunsRoot = Join-Path $StarterRoot "runs\unet_coarse"
$NnUNetRunsRoot = Join-Path $StarterRoot "runs\nnunet_100e_coarse"
$PraNetRunsRoot = Join-Path $StarterRoot "runs\pranet_mc_coarse"

$NnUNetConfigurationRoot = Join-Path `
    $env:nnUNet_results `
    "$DatasetName\$NnUNetTrainer`__$NnUNetPlans`__$NnUNetConfiguration"

$SplitsFile = Join-Path `
    $env:nnUNet_preprocessed `
    "$DatasetName\splits_final.json"

$UNetFold0ConfigPath = Join-Path $UNetRunsRoot "fold_0\config.json"
$PraNetFold0ConfigPath = Join-Path $PraNetRunsRoot "fold_0\config.json"

$LogRoot = Join-Path $StarterRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogRoot "remaining_folds_$Timestamp.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path $Path)) {
        throw "$Description does not exist: $Path"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ("Command: {0} {1}" -f $Executable, ($Arguments -join " ")) -ForegroundColor Cyan
    Write-Host ""

    & $Executable @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

function Get-ValidationPngCount {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory
    )

    if (-not (Test-Path $Directory)) {
        return 0
    }

    return @(
        Get-ChildItem `
            -Path $Directory `
            -File `
            -Filter "*.png" `
            -ErrorAction SilentlyContinue
    ).Count
}

function Add-SwitchIfTrue {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.ArrayList]$ArgumentList,

        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$SwitchName
    )

    if ($Condition) {
        [void]$ArgumentList.Add($SwitchName)
    }
}

function Enable-KeepAwake {
    if (-not ("TigerKeepAwake.NativeMethods" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace TigerKeepAwake
{
    public static class NativeMethods
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint esFlags);
    }
}
"@
    }

    [void][TigerKeepAwake.NativeMethods]::SetThreadExecutionState([uint32]2147483649)
}

function Disable-KeepAwake {
    if ("TigerKeepAwake.NativeMethods" -as [type]) {
        [void][TigerKeepAwake.NativeMethods]::SetThreadExecutionState([uint32]2147483648)
    }
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

Assert-PathExists $PythonExe "Virtual-environment Python"
Assert-PathExists $NnUNetTrainExe "nnUNet training executable"
Assert-PathExists $StarterRoot "Starter project"
Assert-PathExists $DataRoot "Prepared TIGER dataset"
Assert-PathExists $TrainUNet "U-Net training script"
Assert-PathExists $TrainPraNet "PraNet-MC training script"
Assert-PathExists $EvaluatePredictions "Unified evaluation script"
Assert-PathExists $SummarizeResults "Five-fold summary script"
Assert-PathExists $CompareFiveFold "Five-fold comparison script"
Assert-PathExists $SplitsFile "Patient-level split file"
Assert-PathExists $UNetFold0ConfigPath "U-Net Fold 0 config"
Assert-PathExists $PraNetFold0ConfigPath "PraNet-MC Fold 0 config"

$UNetConfig = Get-Content $UNetFold0ConfigPath -Raw | ConvertFrom-Json
$PraNetConfig = Get-Content $PraNetFold0ConfigPath -Raw | ConvertFrom-Json
$Splits = Get-Content $SplitsFile -Raw | ConvertFrom-Json

New-Item -ItemType Directory -Force $LogRoot | Out-Null
New-Item -ItemType Directory -Force $UNetRunsRoot | Out-Null
New-Item -ItemType Directory -Force $NnUNetRunsRoot | Out-Null
New-Item -ItemType Directory -Force $PraNetRunsRoot | Out-Null

Write-Host "============================================================" -ForegroundColor Green
Write-Host "TIGER remaining patient-level folds" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Folds             : $StartFold ... $EndFold"
Write-Host "Workers            : $Workers"
Write-Host "U-Net epochs       : $($UNetConfig.epochs)"
Write-Host "U-Net batch        : $($UNetConfig.batch_size)"
Write-Host "nnU-Net trainer    : $NnUNetTrainer"
Write-Host "PraNet epochs      : $($PraNetConfig.epochs)"
Write-Host "PraNet batch       : $($PraNetConfig.batch_size)"
Write-Host "Log                : $LogFile"
Write-Host "============================================================" -ForegroundColor Green

Invoke-Checked $PythonExe @(
    "-c",
    "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); assert torch.cuda.is_available()"
)

Enable-KeepAwake
Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Set-Location $StarterRoot

    for ($Fold = $StartFold; $Fold -le $EndFold; $Fold++) {
        Write-Host ""
        Write-Host "############################################################" -ForegroundColor Yellow
        Write-Host "Fold $Fold" -ForegroundColor Yellow
        Write-Host "############################################################" -ForegroundColor Yellow

        $ExpectedValidationCount = @($Splits[$Fold].val).Count

        # ---------------------------------------------------------------
        # 1. U-Net
        # ---------------------------------------------------------------

        $UNetOutput = Join-Path $UNetRunsRoot "fold_$Fold"
        $UNetMetrics = Join-Path $UNetOutput "metrics.json"

        if (Test-Path $UNetMetrics) {
            Write-Host "U-Net Fold $Fold already has metrics; skipping." -ForegroundColor Green
        }
        else {
            # The U-Net script does not implement checkpoint resume. Remove any
            # incomplete output directory so stale files cannot contaminate it.
            if (Test-Path $UNetOutput) {
                Remove-Item $UNetOutput -Recurse -Force
            }

            $UNetArgs = [System.Collections.ArrayList]@(
                $TrainUNet,
                "--data-root", $DataRoot,
                "--label-mode", "$($UNetConfig.label_mode)",
                "--fold", "$Fold",
                "--encoder", "$($UNetConfig.encoder)",
                "--encoder-weights", "$($UNetConfig.encoder_weights)",
                "--epochs", "$($UNetConfig.epochs)",
                "--patience", "$($UNetConfig.patience)",
                "--batch-size", "$($UNetConfig.batch_size)",
                "--workers", "$Workers",
                "--lr", "$($UNetConfig.lr)",
                "--weight-decay", "$($UNetConfig.weight_decay)",
                "--width", "$($UNetConfig.width)",
                "--height", "$($UNetConfig.height)",
                "--surface-width", "$($UNetConfig.surface_width)",
                "--surface-height", "$($UNetConfig.surface_height)",
                "--seed", "$($UNetConfig.seed)",
                "--output", $UNetOutput
            )

            Add-SwitchIfTrue $UNetArgs ([bool]$UNetConfig.no_amp) "--no-amp"
            Add-SwitchIfTrue $UNetArgs ([bool]$UNetConfig.no_class_weights) "--no-class-weights"

            Invoke-Checked $PythonExe ([string[]]$UNetArgs)
        }

        # ---------------------------------------------------------------
        # 2. nnU-Net 100 epochs
        # ---------------------------------------------------------------

        $NnUNetFoldRoot = Join-Path $NnUNetConfigurationRoot "fold_$Fold"
        $NnUNetFinalCheckpoint = Join-Path $NnUNetFoldRoot "checkpoint_final.pth"
        $NnUNetLatestCheckpoint = Join-Path $NnUNetFoldRoot "checkpoint_latest.pth"
        $NnUNetValidation = Join-Path $NnUNetFoldRoot "validation"

        $NnUNetOutput = Join-Path $NnUNetRunsRoot "fold_$Fold"
        $NnUNetMetrics = Join-Path $NnUNetOutput "metrics.json"
        New-Item -ItemType Directory -Force $NnUNetOutput | Out-Null

        if (Test-Path $NnUNetMetrics) {
            Write-Host "nnU-Net Fold $Fold already has metrics; skipping." -ForegroundColor Green
        }
        else {
            if (-not (Test-Path $NnUNetFinalCheckpoint)) {
                $NnUNetArgs = [System.Collections.ArrayList]@(
                    "$DatasetId",
                    $NnUNetConfiguration,
                    "$Fold",
                    "-tr", $NnUNetTrainer,
                    "-device", "cuda"
                )

                if (Test-Path $NnUNetLatestCheckpoint) {
                    Write-Host "Resuming nnU-Net Fold $Fold." -ForegroundColor Cyan
                    [void]$NnUNetArgs.Add("--c")
                }
                else {
                    Write-Host "Starting nnU-Net Fold $Fold." -ForegroundColor Cyan
                }

                Invoke-Checked $NnUNetTrainExe ([string[]]$NnUNetArgs)
            }
            else {
                Write-Host "nnU-Net Fold $Fold final checkpoint exists; training is skipped." -ForegroundColor Green
            }

            $PredictionCount = Get-ValidationPngCount $NnUNetValidation
            if ($PredictionCount -lt $ExpectedValidationCount) {
                Write-Host "nnU-Net validation is incomplete; running validation." -ForegroundColor Cyan

                Invoke-Checked $NnUNetTrainExe @(
                    "$DatasetId",
                    $NnUNetConfiguration,
                    "$Fold",
                    "-tr", $NnUNetTrainer,
                    "--val",
                    "-device", "cuda"
                )

                $PredictionCount = Get-ValidationPngCount $NnUNetValidation
            }

            if ($PredictionCount -ne $ExpectedValidationCount) {
                throw "nnU-Net Fold $Fold has $PredictionCount validation PNGs; expected $ExpectedValidationCount."
            }

            Invoke-Checked $PythonExe @(
                $EvaluatePredictions,
                "--data-root", $DataRoot,
                "--pred-dir", $NnUNetValidation,
                "--label-mode", "coarse",
                "--surface-width", "640",
                "--surface-height", "384",
                "--output", $NnUNetMetrics
            )
        }

        # ---------------------------------------------------------------
        # 3. PraNet-MC
        # ---------------------------------------------------------------

        $PraNetOutput = Join-Path $PraNetRunsRoot "fold_$Fold"
        $PraNetMetrics = Join-Path $PraNetOutput "metrics.json"

        if (Test-Path $PraNetMetrics) {
            Write-Host "PraNet-MC Fold $Fold already has metrics; skipping." -ForegroundColor Green
        }
        else {
            # The PraNet-MC script does not implement checkpoint resume.
            if (Test-Path $PraNetOutput) {
                Remove-Item $PraNetOutput -Recurse -Force
            }

            $PraNetArgs = [System.Collections.ArrayList]@(
                $TrainPraNet,
                "--data-root", $DataRoot,
                "--label-mode", "$($PraNetConfig.label_mode)",
                "--fold", "$Fold",
                "--backbone", "$($PraNetConfig.backbone)",
                "--backbone-weights", "$($PraNetConfig.backbone_weights)",
                "--decoder-channels", "$($PraNetConfig.decoder_channels)",
                "--ra-channels", "$($PraNetConfig.ra_channels)",
                "--epochs", "$($PraNetConfig.epochs)",
                "--patience", "$($PraNetConfig.patience)",
                "--batch-size", "$($PraNetConfig.batch_size)",
                "--workers", "$Workers",
                "--prefetch-factor", "$($PraNetConfig.prefetch_factor)",
                "--lr", "$($PraNetConfig.lr)",
                "--weight-decay", "$($PraNetConfig.weight_decay)",
                "--width", "$($PraNetConfig.width)",
                "--height", "$($PraNetConfig.height)",
                "--surface-width", "$($PraNetConfig.surface_width)",
                "--surface-height", "$($PraNetConfig.surface_height)",
                "--seed", "$($PraNetConfig.seed)",
                "--max-minutes", "0",
                "--output", $PraNetOutput
            )

            Add-SwitchIfTrue $PraNetArgs ([bool]$PraNetConfig.no_amp) "--no-amp"
            Add-SwitchIfTrue $PraNetArgs ([bool]$PraNetConfig.no_class_weights) "--no-class-weights"
            Add-SwitchIfTrue $PraNetArgs ([bool]$PraNetConfig.no_channels_last) "--no-channels-last"

            Invoke-Checked $PythonExe ([string[]]$PraNetArgs)
        }

        Write-Host "Fold $Fold is complete for all three methods." -ForegroundColor Green
    }

    # -------------------------------------------------------------------
    # Summaries. summarize_results.py requires all five folds.
    # -------------------------------------------------------------------

    $AllMetricsPresent = $true

    foreach ($RunsRoot in @($UNetRunsRoot, $NnUNetRunsRoot, $PraNetRunsRoot)) {
        for ($Fold = 0; $Fold -lt 5; $Fold++) {
            if (-not (Test-Path (Join-Path $RunsRoot "fold_$Fold\metrics.json"))) {
                $AllMetricsPresent = $false
            }
        }
    }

    if ($AllMetricsPresent) {
        Invoke-Checked $PythonExe @(
            $SummarizeResults,
            "--runs-root", $UNetRunsRoot,
            "--method", "U-Net"
        )

        Invoke-Checked $PythonExe @(
            $SummarizeResults,
            "--runs-root", $NnUNetRunsRoot,
            "--method", "nnU-Net (100 epochs)"
        )

        Invoke-Checked $PythonExe @(
            $SummarizeResults,
            "--runs-root", $PraNetRunsRoot,
            "--method", "PraNet-MC"
        )

        Invoke-Checked $PythonExe @(
            $CompareFiveFold,
            "--project-root", $ProjectRoot
        )

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "All five folds are complete." -ForegroundColor Green
        Write-Host "Final comparison:" -ForegroundColor Green
        Write-Host (Join-Path $StarterRoot "runs\fivefold_algorithm_comparison.csv")
        Write-Host "============================================================" -ForegroundColor Green
    }
    else {
        Write-Host ""
        Write-Host "The selected fold range finished, but not all 15 metrics files exist yet." -ForegroundColor Yellow
        Write-Host "The final five-fold summary was not generated." -ForegroundColor Yellow
    }
}
catch {
    Write-Host ""
    Write-Host "The run stopped because of an error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Log: $LogFile" -ForegroundColor Red
    throw
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }

    Disable-KeepAwake
}
