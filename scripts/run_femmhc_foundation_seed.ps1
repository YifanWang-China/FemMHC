param(
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [string]$OpenMhcRoot = '',
    [string]$OpenMhcCheckpoint = '',
    [string]$ProcessedNhanes = ''
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if ([string]::IsNullOrWhiteSpace($OpenMhcRoot)) {
    $OpenMhcRoot = Join-Path $workspace 'datasets\openmhc-xs'
}
if ([string]::IsNullOrWhiteSpace($OpenMhcCheckpoint)) {
    $OpenMhcCheckpoint = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
}
if ([string]::IsNullOrWhiteSpace($ProcessedNhanes)) {
    $ProcessedNhanes = Join-Path $workspace 'processed\nhanes_female'
}
$runRoot = Join-Path $workspace ("artifacts\runs\seed-{0}" -f $Seed)
$stage1 = Join-Path $runRoot 'femmhc-openmhc-female-v4.ckpt'
$stage1Best = Join-Path $runRoot 'femmhc-openmhc-female-v4-best.ckpt'
$nhanes = Join-Path $runRoot 'femmhc-nhanes-female-v3.ckpt'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')

Write-Output ("[{0}] seed={1} participant-balanced OpenMHC female stage" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\train_femmhc_openmhc_female.py') `
    --checkpoint $OpenMhcCheckpoint `
    --openmhc-root $OpenMhcRoot `
    --output $stage1 `
    --max-steps 5000 `
    --batch-size 1 `
    --max-channels 6 `
    --preservation-weight 10 `
    --save-every 250 `
    --validation-days 64 `
    --seed $Seed `
    --resume
if ($LASTEXITCODE -ne 0) { throw "OpenMHC female stage failed" }
if (-not (Test-Path -LiteralPath $stage1Best)) { throw "Missing best stage-1 checkpoint" }

Write-Output ("[{0}] seed={1} NHANES rhythm-expert stage" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\train_femmhc_nhanes_female.py') `
    --checkpoint $OpenMhcCheckpoint `
    --initial-femmhc-checkpoint $stage1Best `
    --processed-dir $ProcessedNhanes `
    --openmhc-root $OpenMhcRoot `
    --output $nhanes `
    --max-steps 3000 `
    --batch-size 8 `
    --unfreeze-last-blocks 0 `
    --preservation-weight 10 `
    --replay-every 4 `
    --replay-weight 10 `
    --save-every 250 `
    --validation-pairs 512 `
    --validation-openmhc-days 64 `
    --selection-preservation-weight 10 `
    --seed $Seed `
    --resume
if ($LASTEXITCODE -ne 0) { throw "NHANES rhythm-expert stage failed" }

Write-Output ("[{0}] seed={1} foundation training complete" -f (Get-Date -Format o), $Seed)
