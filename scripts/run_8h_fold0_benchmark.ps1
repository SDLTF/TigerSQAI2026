param(
    [string]$ProjectRoot = "D:\Code\TigerSQ-AI-2026",
    [ValidateRange(1.0, 24.0)]
    [double]$Hours = 8.0,
    [ValidateRange(0, 32)]
    [int]$Workers = 6,
    [ValidateRange(1, 16)]
    [int]$UNetBatchSize = 4,
    [ValidateRange(1, 8)]
    [int]$PraNetBatchSize = 2,
    [ValidateRange(1, 200)]
    [int]$PraNetMaxEpochs = 80,
    [ValidateRange(5, 60)]
    [int]$ReserveMinutes = 20,
    [switch]$ForceUNet,
    [switch]$ForceNnUNet,
    [switch]$ForcePraNet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)

$VenvRoot = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"
$NnUNetTrainExe = Join-Path $VenvRoot "Scripts\nnUNetv2_train.exe"

$StarterRoot = Join-Path $ProjectRoot "tiger_segmentation_starter"
$DataRoot = Join-Path $ProjectRoot "TigerSQ-AI-2026-prepared"
$TrainUNet = Join-Path $StarterRoot "train_unet.py"
$TrainPraNet = Join-Path $StarterRoot "train_pranet_multiclass.py"
$EvaluatePredictions = Join-Path $StarterRoot "evaluate_predictions.py"
$CompareFold0 = Join-Path $StarterRoot "compare_fold0.py"

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

$UNetOutput = Join-Path $StarterRoot "runs\unet_coarse\fold_0"
$UNetMetrics = Join-Path $UNetOutput "metrics.json"

$NnUNetResultsRoot = Join-Path `
    $env:nnUNet_results `
    "$DatasetName\$NnUNetTrainer`__$NnUNetPlans`__$NnUNetConfiguration\fold_0"
$NnUNetFinalCheckpoint = Join-Path $NnUNetResultsRoot "checkpoint_final.pth"
$NnUNetLatestCheckpoint = Join-Path $NnUNetResultsRoot "checkpoint_latest.pth"
$NnUNetValidation = Join-Path $NnUNetResultsRoot "validation"
$NnUNetOutput = Join-Path $StarterRoot "runs\nnunet_100e_coarse\fold_0"
$NnUNetMetrics = Join-Path $NnUNetOutput "metrics.json"

$PraNetOutput = Join-Path $StarterRoot "runs\pranet_mc_coarse\fold_0"
$PraNetMetrics = Join-Path $PraNetOutput "metrics.json"

$LogRoot = Join-Path $StarterRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogRoot "fold0_8h_benchmark_$Timestamp.log"

function Assert-PathExists {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path $Path)) {
        throw "$Description does not exist: $Path"
    }
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    Write-Host ""
    Write-Host ("Command: {0} {1}" -f $Executable, ($Arguments -join " ")) -ForegroundColor Cyan
    Write-Host ""
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Get-RemainingMinutes {
    return [math]::Floor(($Deadline - (Get-Date)).TotalMinutes)
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

Assert-PathExists $PythonExe "Virtual-environment Python"
Assert-PathExists $NnUNetTrainExe "nnUNetv2_train executable"
Assert-PathExists $StarterRoot "Starter project"
Assert-PathExists $DataRoot "Prepared TIGER dataset"
Assert-PathExists $TrainUNet "U-Net training script"
Assert-PathExists $TrainPraNet "PraNet-MC training script"
Assert-PathExists $EvaluatePredictions "Unified evaluation script"
Assert-PathExists $CompareFold0 "Comparison script"
Assert-PathExists (Join-Path $env:nnUNet_preprocessed "$DatasetName\splits_final.json") "nnU-Net split file"

New-Item -ItemType Directory -Force $LogRoot | Out-Null
New-Item -ItemType Directory -Force $NnUNetOutput | Out-Null

Write-Host "============================================================" -ForegroundColor Green
Write-Host "TIGER Fold-0 three-algorithm benchmark" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Started          : $StartedAt"
Write-Host "Deadline         : $Deadline"
Write-Host "Budget           : $Hours hours"
Write-Host "Workers          : $Workers"
Write-Host "U-Net batch      : $UNetBatchSize"
Write-Host "PraNet batch     : $PraNetBatchSize"
Write-Host "nnU-Net trainer  : $NnUNetTrainer"
Write-Host "Log              : $LogFile"
Write-Host "============================================================" -ForegroundColor Green

Invoke-Checked $PythonExe @(
    "-c",
    "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); assert torch.cuda.is_available()"
)

Enable-KeepAwake
Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Set-Location $StarterRoot

    # Download/check the PraNet backbone before spending hours on nnU-Net.
    Invoke-Checked $PythonExe @(
        "-c",
        "from tigerseg.pranet_multiclass import MultiClassPraNet; m=MultiClassPraNet(16, backbone_name='res2net50_26w_4s', pretrained=True); print('PraNet-MC backbone preflight: OK')"
    )

    # ------------------------------------------------------------------
    # 1. U-Net Fold 0
    # ------------------------------------------------------------------
    if ($ForceUNet -and (Test-Path $UNetOutput)) {
        Remove-Item $UNetOutput -Recurse -Force
    }

    if (Test-Path $UNetMetrics) {
        Write-Host "U-Net Fold 0 metrics already exist; U-Net is skipped." -ForegroundColor Green
        Write-Host $UNetMetrics
    }
    else {
        Write-Host "U-Net Fold 0 metrics are missing; running the formal baseline." -ForegroundColor Yellow
        Invoke-Checked $PythonExe @(
            $TrainUNet,
            "--data-root", $DataRoot,
            "--label-mode", "coarse",
            "--fold", "0",
            "--encoder", "resnet34",
            "--epochs", "80",
            "--patience", "15",
            "--batch-size", "$UNetBatchSize",
            "--workers", "$Workers",
            "--lr", "1e-4",
            "--weight-decay", "1e-4",
            "--output", $UNetOutput
        )
    }

    # ------------------------------------------------------------------
    # 2. nnU-Net Fold 0, 100 epochs
    # ------------------------------------------------------------------
    if ($ForceNnUNet) {
        if (Test-Path $NnUNetResultsRoot) {
            Remove-Item $NnUNetResultsRoot -Recurse -Force
        }
        if (Test-Path $NnUNetOutput) {
            Remove-Item $NnUNetOutput -Recurse -Force
        }
        New-Item -ItemType Directory -Force $NnUNetOutput | Out-Null
    }

    if (-not (Test-Path $NnUNetMetrics)) {
        $TrainArgs = @(
            "$DatasetId",
            $NnUNetConfiguration,
            "0",
            "-tr", $NnUNetTrainer,
            "-device", "cuda"
        )

        if (Test-Path $NnUNetFinalCheckpoint) {
            Write-Host "nnU-Net 100e final checkpoint already exists; training is skipped." -ForegroundColor Green
        }
        else {
            if (Test-Path $NnUNetLatestCheckpoint) {
                Write-Host "Resuming nnU-Net 100e Fold 0 from checkpoint_latest.pth." -ForegroundColor Cyan
                $TrainArgs += "--c"
            }
            else {
                Write-Host "Starting nnU-Net 100e Fold 0." -ForegroundColor Cyan
            }
            Invoke-Checked $NnUNetTrainExe $TrainArgs
        }

        $PredictionCount = 0
        if (Test-Path $NnUNetValidation) {
            $PredictionCount = @(Get-ChildItem $NnUNetValidation -Filter "*.png" -File).Count
        }
        if ($PredictionCount -lt 28) {
            Write-Host "nnU-Net validation PNGs are incomplete; running validation." -ForegroundColor Cyan
            Invoke-Checked $NnUNetTrainExe @(
                "$DatasetId",
                $NnUNetConfiguration,
                "0",
                "-tr", $NnUNetTrainer,
                "--val",
                "-device", "cuda"
            )
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
    else {
        Write-Host "nnU-Net Fold 0 metrics already exist; nnU-Net is skipped." -ForegroundColor Green
        Write-Host $NnUNetMetrics
    }

    # ------------------------------------------------------------------
    # 3. PraNet-MC Fold 0. Use all remaining budget except the reserve.
    # ------------------------------------------------------------------
    if ($ForcePraNet -and (Test-Path $PraNetOutput)) {
        Remove-Item $PraNetOutput -Recurse -Force
    }

    if (Test-Path $PraNetMetrics) {
        Write-Host "PraNet-MC Fold 0 metrics already exist; PraNet-MC is skipped." -ForegroundColor Green
        Write-Host $PraNetMetrics
    }
    else {
        $RemainingMinutes = (Get-RemainingMinutes) - $ReserveMinutes
        if ($RemainingMinutes -lt 15) {
            throw "Only $RemainingMinutes minutes remain for PraNet-MC. Increase -Hours or reduce earlier work."
        }

        Write-Host "PraNet-MC wall-clock budget: $RemainingMinutes minutes." -ForegroundColor Yellow
        Invoke-Checked $PythonExe @(
            $TrainPraNet,
            "--data-root", $DataRoot,
            "--label-mode", "coarse",
            "--fold", "0",
            "--backbone", "res2net50_26w_4s",
            "--backbone-weights", "imagenet",
            "--epochs", "$PraNetMaxEpochs",
            "--patience", "12",
            "--batch-size", "$PraNetBatchSize",
            "--workers", "$Workers",
            "--prefetch-factor", "4",
            "--lr", "1e-4",
            "--weight-decay", "1e-4",
            "--max-minutes", "$RemainingMinutes",
            "--output", $PraNetOutput
        )
    }

    # ------------------------------------------------------------------
    # 4. One comparison CSV
    # ------------------------------------------------------------------
    Invoke-Checked $PythonExe @(
        $CompareFold0,
        "--project-root", $ProjectRoot
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Fold-0 benchmark finished." -ForegroundColor Green
    Write-Host "Comparison CSV:" -ForegroundColor Green
    Write-Host (Join-Path $StarterRoot "runs\fold0_algorithm_comparison.csv")
    Write-Host "Elapsed: $([math]::Round(((Get-Date) - $StartedAt).TotalHours, 2)) hours"
    Write-Host "============================================================" -ForegroundColor Green
}
catch {
    Write-Host "" 
    Write-Host "The benchmark stopped because of an error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Log: $LogFile" -ForegroundColor Red
    throw
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    Disable-KeepAwake
}
