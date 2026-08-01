param(
    [int]$Seed = 42,
    [int]$OpenMhcSteps = 5000,
    [int]$McPhasesSteps = 3000,
    [int]$OpenMhcBatchSize = 4,
    [int]$McPhasesBatchSize = 2,
    [int]$SaveEvery = 250,
    [string]$OpenMhcRoot = "",
    [string]$OpenMhcCheckpoint = ""
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if ([string]::IsNullOrWhiteSpace($OpenMhcCheckpoint)) {
    $OpenMhcCheckpoint = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
}
if ([string]::IsNullOrWhiteSpace($OpenMhcRoot)) {
    $OpenMhcRoot = Join-Path $workspace 'datasets\openmhc-xs'
}
$openMhcSource = $OpenMhcCheckpoint
$openMhcRoot = $OpenMhcRoot
$mcPhasesRoot = Join-Path $workspace 'processed\mcphases'
$runRoot = Join-Path $workspace ("artifacts\runs\seed-{0}" -f $Seed)
$stage1 = Join-Path $runRoot 'femmhc-openmhc-female.ckpt'
$stage2 = Join-Path $runRoot 'femmhc-mcphases-causal.ckpt'

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')

Write-Output ("[{0}] seed={1} stage-1 started" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\train_femmhc_openmhc_female.py') `
    --checkpoint $openMhcSource `
    --openmhc-root $openMhcRoot `
    --output $stage1 `
    --max-steps $OpenMhcSteps `
    --batch-size $OpenMhcBatchSize `
    --max-channels 6 `
    --save-every $SaveEvery `
    --seed $Seed `
    --resume
if ($LASTEXITCODE -ne 0) {
    throw "FemMHC stage-1 failed with exit code $LASTEXITCODE"
}

Write-Output ("[{0}] seed={1} stage-2 started" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\train_femmhc_pretrain.py') `
    --checkpoint $openMhcSource `
    --femmhc-init $stage1 `
    --processed-dir $mcPhasesRoot `
    --output $stage2 `
    --max-steps $McPhasesSteps `
    --batch-size $McPhasesBatchSize `
    --save-every $SaveEvery `
    --keep-periodic-checkpoints `
    --self-supervised-weight 1 `
    --supervised-weight 0.5 `
    --seed $Seed `
    --resume
if ($LASTEXITCODE -ne 0) {
    throw "FemMHC stage-2 failed with exit code $LASTEXITCODE"
}

Write-Output ("[{0}] seed={1} training complete" -f (Get-Date -Format o), $Seed)
