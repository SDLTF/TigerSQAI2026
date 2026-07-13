param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
for ($fold = 0; $fold -lt 5; $fold++) {
    & $Python train_unet.py --data-root $DataRoot --label-mode coarse --fold $fold --epochs 80 --batch-size 2 --workers 4
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $Python summarize_results.py --runs-root runs/unet_coarse --method UNet
