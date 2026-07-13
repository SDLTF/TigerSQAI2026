param(
    [Parameter(Mandatory=$true)][int]$DatasetId
)

$ErrorActionPreference = "Stop"
for ($fold = 0; $fold -lt 5; $fold++) {
    nnUNetv2_train $DatasetId 2d $fold
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
