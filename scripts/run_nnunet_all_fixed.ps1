param(
    [string]$ProjectRoot = "D:\Code\TigerSQ-AI-2026",

    [int]$DatasetId = 501,

    [string]$DatasetName = "Dataset501_TigerCoarse",

    [ValidateRange(0, 4)]
    [int]$StartFold = 0,

    [ValidateRange(0, 4)]
    [int]$EndFold = 4,

    [ValidateRange(0, 32)]
    [int]$Workers = 6,

    [switch]$RetrainCompleted
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($StartFold -gt $EndFold) {
    throw "StartFold cannot be greater than EndFold."
}

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

$VenvRoot = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"
$TrainExe = Join-Path $VenvRoot "Scripts\nnUNetv2_train.exe"

$StarterRoot = Join-Path $ProjectRoot "tiger_segmentation_starter"
$DataRoot = Join-Path $ProjectRoot "TigerSQ-AI-2026-prepared"

$NnUNetRoot = Join-Path $ProjectRoot "nnunet"
$env:nnUNet_raw = Join-Path $NnUNetRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $NnUNetRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $NnUNetRoot "nnUNet_results"

# nnU-Net data-augmentation and CPU thread settings.
$env:nnUNet_n_proc_DA = "$Workers"
$env:nnUNet_compile = "false"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$EvaluateScript = Join-Path $StarterRoot "evaluate_predictions.py"
$SummarizeScript = Join-Path $StarterRoot "summarize_results.py"
$RunsRoot = Join-Path $StarterRoot "runs\nnunet_coarse"

$PreprocessedDataset = Join-Path $env:nnUNet_preprocessed $DatasetName
$SplitsFile = Join-Path $PreprocessedDataset "splits_final.json"

$TrainerName = "nnUNetTrainer"
$PlansName = "nnUNetPlans"
$Configuration = "2d"

$ConfigurationResultsRoot = Join-Path `
    $env:nnUNet_results `
    "$DatasetName\$TrainerName`__$PlansName`__$Configuration"

$LogRoot = Join-Path $StarterRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogRoot "nnunet_5fold_$Timestamp.log"

# ---------------------------------------------------------------------------
# Helper functions
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

function Invoke-CheckedCommand {
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
        [string]$ValidationDirectory
    )

    if (-not (Test-Path $ValidationDirectory)) {
        return 0
    }

    return @(
        Get-ChildItem `
            -Path $ValidationDirectory `
            -File `
            -Filter "*.png" `
            -ErrorAction SilentlyContinue
    ).Count
}

function Enable-KeepAwake {
    if (-not ("KeepAwake.NativeMethods" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace KeepAwake
{
    public static class NativeMethods
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint esFlags);
    }
}
"@
    }

    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    [void][KeepAwake.NativeMethods]::SetThreadExecutionState([uint32]2147483649)
}

function Disable-KeepAwake {
    if ("KeepAwake.NativeMethods" -as [type]) {
        # ES_CONTINUOUS
        [void][KeepAwake.NativeMethods]::SetThreadExecutionState([uint32]2147483648)
    }
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

Assert-PathExists -Path $PythonExe -Description "Virtual-environment Python"
Assert-PathExists -Path $TrainExe -Description "nnUNetv2_train executable"
Assert-PathExists -Path $StarterRoot -Description "Starter project directory"
Assert-PathExists -Path $DataRoot -Description "Prepared TIGER dataset"
Assert-PathExists -Path $EvaluateScript -Description "Unified evaluation script"
Assert-PathExists -Path $SummarizeScript -Description "Five-fold summary script"
Assert-PathExists -Path $PreprocessedDataset -Description "nnU-Net preprocessed dataset"
Assert-PathExists -Path $SplitsFile -Description "Case-level five-fold split file"

New-Item -ItemType Directory -Force $RunsRoot | Out-Null
New-Item -ItemType Directory -Force $LogRoot | Out-Null

$Splits = Get-Content $SplitsFile -Raw | ConvertFrom-Json

Write-Host "============================================================" -ForegroundColor Green
Write-Host "TIGER nnU-Net five-fold runner" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Project root       : $ProjectRoot"
Write-Host "Dataset            : $DatasetId / $DatasetName"
Write-Host "Configuration      : $Configuration"
Write-Host "Folds              : $StartFold ... $EndFold"
Write-Host "DA workers         : $Workers"
Write-Host "nnUNet raw         : $env:nnUNet_raw"
Write-Host "nnUNet preprocessed: $env:nnUNet_preprocessed"
Write-Host "nnUNet results     : $env:nnUNet_results"
Write-Host "Unified metrics    : $RunsRoot"
Write-Host "Log                : $LogFile"
Write-Host "============================================================" -ForegroundColor Green

# Verify that the selected Python still uses CUDA 13.x and sees the GPU.
Invoke-CheckedCommand -Executable $PythonExe -Arguments @(
    "-c",
    "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); assert torch.cuda.is_available()"
)

Enable-KeepAwake
Start-Transcript -Path $LogFile -Append | Out-Null

try {
    for ($Fold = $StartFold; $Fold -le $EndFold; $Fold++) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Yellow
        Write-Host "Fold $Fold" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Yellow

        $ExpectedValidationCount = @($Splits[$Fold].val).Count

        $FoldResults = Join-Path $ConfigurationResultsRoot "fold_$Fold"
        $FinalCheckpoint = Join-Path $FoldResults "checkpoint_final.pth"
        $LatestCheckpoint = Join-Path $FoldResults "checkpoint_latest.pth"
        $ValidationDirectory = Join-Path $FoldResults "validation"

        $MetricsDirectory = Join-Path $RunsRoot "fold_$Fold"
        $MetricsFile = Join-Path $MetricsDirectory "metrics.json"

        New-Item -ItemType Directory -Force $MetricsDirectory | Out-Null

        $FinalExists = Test-Path $FinalCheckpoint
        $LatestExists = Test-Path $LatestCheckpoint
        $ValidationCount = Get-ValidationPngCount -ValidationDirectory $ValidationDirectory

        Write-Host "Expected validation PNGs: $ExpectedValidationCount"
        Write-Host "Existing validation PNGs: $ValidationCount"
        Write-Host "Latest checkpoint exists : $LatestExists"
        Write-Host "Final checkpoint exists  : $FinalExists"

        if ($RetrainCompleted) {
            Write-Host "RetrainCompleted was specified; starting fold $Fold from scratch." -ForegroundColor Magenta

            if (Test-Path $FoldResults) {
                Remove-Item $FoldResults -Recurse -Force
            }

            $FinalExists = $false
            $LatestExists = $false
            $ValidationCount = 0
        }

        if (-not $FinalExists) {
            $TrainArguments = @(
                "$DatasetId",
                $Configuration,
                "$Fold",
                "--npz",
                "-device",
                "cuda"
            )

            if ($LatestExists) {
                Write-Host "A partial checkpoint was found. Resuming fold $Fold." -ForegroundColor Cyan
                $TrainArguments += "--c"
            }
            else {
                Write-Host "Starting fold $Fold from the beginning." -ForegroundColor Cyan
            }

            Invoke-CheckedCommand `
                -Executable $TrainExe `
                -Arguments $TrainArguments
        }
        else {
            Write-Host "Fold $Fold already has checkpoint_final.pth; training is skipped." -ForegroundColor Green
        }

        # A completed training normally creates validation predictions automatically.
        # If they are missing or incomplete, rerun validation with probability export.
        $ValidationCount = Get-ValidationPngCount -ValidationDirectory $ValidationDirectory

        if ($ValidationCount -lt $ExpectedValidationCount) {
            Write-Host `
                "Validation predictions are incomplete ($ValidationCount/$ExpectedValidationCount). Rerunning validation." `
                -ForegroundColor Cyan

            Invoke-CheckedCommand `
                -Executable $TrainExe `
                -Arguments @(
                    "$DatasetId",
                    $Configuration,
                    "$Fold",
                    "--val",
                    "--npz",
                    "-device",
                    "cuda"
                )

            $ValidationCount = Get-ValidationPngCount -ValidationDirectory $ValidationDirectory
        }

        if ($ValidationCount -ne $ExpectedValidationCount) {
            throw "Fold $Fold has $ValidationCount validation PNGs; expected $ExpectedValidationCount."
        }

        # Unified evaluation protocol shared with U-Net/PraNet:
        # foreground macro average, 640x384 surface-distance space,
        # identical empty-mask handling.
        Invoke-CheckedCommand `
            -Executable $PythonExe `
            -Arguments @(
                $EvaluateScript,
                "--data-root",
                $DataRoot,
                "--pred-dir",
                $ValidationDirectory,
                "--label-mode",
                "coarse",
                "--surface-width",
                "640",
                "--surface-height",
                "384",
                "--output",
                $MetricsFile
            )

        Write-Host "Fold $Fold completed and evaluated." -ForegroundColor Green
        Write-Host "Metrics: $MetricsFile"
    }

    # Only summarize after all five fold metrics are available.
    $MissingMetrics = @()

    for ($Fold = 0; $Fold -lt 5; $Fold++) {
        $Path = Join-Path $RunsRoot "fold_$Fold\metrics.json"

        if (-not (Test-Path $Path)) {
            $MissingMetrics += $Fold
        }
    }

    if ($MissingMetrics.Count -eq 0) {
        Invoke-CheckedCommand `
            -Executable $PythonExe `
            -Arguments @(
                $SummarizeScript,
                "--runs-root",
                $RunsRoot,
                "--method",
                "nnU-Net"
            )

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "All five folds have finished." -ForegroundColor Green
        Write-Host "Fold metrics : $(Join-Path $RunsRoot 'fold_metrics.csv')" -ForegroundColor Green
        Write-Host "Final summary: $(Join-Path $RunsRoot 'spreadsheet_summary.csv')" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
    }
    else {
        Write-Host ""
        Write-Host "The selected fold range has finished." -ForegroundColor Green
        Write-Host "Five-fold summary was not generated because these folds are missing: $($MissingMetrics -join ', ')" -ForegroundColor Yellow
    }
}
catch {
    Write-Host ""
    Write-Host "The run stopped because of an error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Log file: $LogFile" -ForegroundColor Red
    throw
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Ignore transcript shutdown errors.
    }

    Disable-KeepAwake
}
