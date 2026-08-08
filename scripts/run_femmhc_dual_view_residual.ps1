param(
    [int]$Seed = 42,
    [string]$Device = 'cuda',
    [int]$MaxSteps = 1000,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')
$baselineRoot = Join-Path $workspace 'artifacts\benchmark\femmhc-static-adapter-multicohort-v1'
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $workspace 'artifacts\benchmark\femmhc-dual-view-residual-v1'
}

$nativeOpenMHC = Join-Path $workspace 'artifacts\embeddings\openmhc-xs\openmhc-lsm2'
$staticOpenMHC = Join-Path $workspace ("artifacts\embeddings\openmhc-xs\femmhc-internal-adapter-rank32-onset2-seed{0}" -f $Seed)
$staticMcPhases = Join-Path $workspace ("artifacts\embeddings\mcphases\internal-adapter-rank32-onset2-six-seed{0}-dual.npy" -f $Seed)
$staticDepress = Join-Path $workspace ("artifacts\embeddings\depress-fitbit-internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)
$staticInPHRSym = Join-Path $workspace ("artifacts\embeddings\inphrsym-internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)
$staticHRV = Join-Path $workspace ("artifacts\embeddings\hrv-mental-female\internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)
$staticPregnancy = Join-Path $workspace ("artifacts\embeddings\pregnancy-ga-official\internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)

foreach ($required in @($python, $baselineRoot, $nativeOpenMHC, $staticOpenMHC, $staticMcPhases, $staticDepress, $staticInPHRSym, $staticHRV, $staticPregnancy)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required artifact: $required" }
}

$checkpointRoot = Join-Path $OutputRoot 'checkpoints'
$evaluationRoot = Join-Path $OutputRoot 'evaluations'
New-Item -ItemType Directory -Force -Path $checkpointRoot, $evaluationRoot | Out-Null
$checkpoint = Join-Path $checkpointRoot ("dual_view_residual-seed{0}.pt" -f $Seed)
$evaluation = Join-Path $evaluationRoot ("dual_view_residual-seed{0}-validation" -f $Seed)

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}

if (-not (Test-Path -LiteralPath $checkpoint)) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\train_femmhc_joint.py') `
        --architecture dual_view_residual_router --hidden-dim 124 --output $checkpoint `
        --max-steps $MaxSteps --batch-size 16 --learning-rate 0.0002 --weight-decay 0.01 `
        --dropout 0.0 --routing-initial-logit -2.0 --maximum-days 60 --minimum-history-days 3 `
        --validate-every 100 --validation-batches 8 --cohort-sampling-temperature 0.5 `
        --checkpoint-selection final_step --seed $Seed --device $Device `
        --openmhc-native-cache $nativeOpenMHC --openmhc-adapted-cache $staticOpenMHC `
        --mcphases-embeddings $staticMcPhases --depress-embeddings $staticDepress `
        --inphrsym-embeddings $staticInPHRSym --hrv-mental-embeddings $staticHRV `
        --pregnancy-embeddings $staticPregnancy
}
if (-not (Test-Path -LiteralPath (Join-Path $evaluation 'per_task_metrics.csv'))) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\evaluate_femmhc_joint.py') `
        --checkpoint $checkpoint --output-dir $evaluation --split validation --device $Device `
        --openmhc-native-cache $nativeOpenMHC --openmhc-adapted-cache $staticOpenMHC `
        --mcphases-embeddings $staticMcPhases --depress-embeddings $staticDepress `
        --inphrsym-embeddings $staticInPHRSym --hrv-mental-embeddings $staticHRV `
        --pregnancy-embeddings $staticPregnancy
}
Invoke-CheckedPython (Join-Path $workspace 'scripts\aggregate_femmhc_dual_view_residual.py') `
    --root $OutputRoot --baseline-root $baselineRoot --seed $Seed
