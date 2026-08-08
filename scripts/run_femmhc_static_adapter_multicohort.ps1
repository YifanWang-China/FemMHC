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
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $workspace 'artifacts\benchmark\femmhc-static-adapter-multicohort-v1'
}

$sourceCheckpoint = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
$adapterCheckpoint = Join-Path $workspace ("artifacts\checkpoints\femmhc-internal-adapter-rank32-onset2-six-seed{0}.ckpt" -f $Seed)
$nativeOpenMHC = Join-Path $workspace 'artifacts\embeddings\openmhc-xs\openmhc-lsm2'
$zeroOpenMHC = Join-Path $workspace 'artifacts\embeddings\openmhc-xs\zero-second-view-control'
$staticOpenMHC = Join-Path $workspace ("artifacts\embeddings\openmhc-xs\femmhc-internal-adapter-rank32-onset2-seed{0}" -f $Seed)

$staticMcPhases = Join-Path $workspace ("artifacts\embeddings\mcphases\internal-adapter-rank32-onset2-six-seed{0}-dual.npy" -f $Seed)
$staticDepress = Join-Path $workspace ("artifacts\embeddings\depress-fitbit-internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)
$staticInPHRSym = Join-Path $workspace ("artifacts\embeddings\inphrsym-internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)
$staticHRV = Join-Path $workspace ("artifacts\embeddings\hrv-mental-female\internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)
$staticPregnancy = Join-Path $workspace ("artifacts\embeddings\pregnancy-ga-official\internal-adapter-rank32-onset2-seed{0}.npz" -f $Seed)

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

foreach ($required in @($python, $sourceCheckpoint, $adapterCheckpoint, $nativeOpenMHC, $staticOpenMHC, $staticMcPhases)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required artifact: $required" }
}

if (-not (Test-Path -LiteralPath (Join-Path $zeroOpenMHC 'embeddings.npy'))) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\create_zero_openmhc_embedding_cache.py') `
        --native-cache $nativeOpenMHC --output-dir $zeroOpenMHC
}
if (-not (Test-Path -LiteralPath $staticDepress)) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\cache_affective_daily_embeddings.py') `
        --cohort depress_fitbit --checkpoint $sourceCheckpoint --femmhc-checkpoint $adapterCheckpoint `
        --processed-dir (Join-Path $workspace 'processed\depress_fitbit') --output $staticDepress `
        --batch-size 32 --seed $Seed --device $Device
}
if (-not (Test-Path -LiteralPath $staticInPHRSym)) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\cache_affective_daily_embeddings.py') `
        --cohort inphrsym --checkpoint $sourceCheckpoint --femmhc-checkpoint $adapterCheckpoint `
        --processed-dir (Join-Path $workspace 'processed\inphrsym') --output $staticInPHRSym `
        --batch-size 32 --seed $Seed --device $Device
}
if (-not (Test-Path -LiteralPath $staticHRV)) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\cache_hrv_mental_embeddings.py') `
        --checkpoint $sourceCheckpoint --femmhc-checkpoint $adapterCheckpoint `
        --processed-dir (Join-Path $workspace 'processed\wearable_hrv_mental_female') --output $staticHRV `
        --batch-size 32 --seed $Seed --device $Device
}
if (-not (Test-Path -LiteralPath $staticPregnancy)) {
    Invoke-CheckedPython (Join-Path $workspace 'scripts\cache_pregnancy_ga_embeddings.py') `
        --checkpoint $sourceCheckpoint --femmhc-checkpoint $adapterCheckpoint `
        --processed-dir (Join-Path $workspace 'processed\pregnancy_ga_clock_official') --output $staticPregnancy `
        --batch-size 16 --seed $Seed --device $Device
}

$representations = @{
    openmhc = @{
        OpenMHCAdapted = $zeroOpenMHC
        McPhases = Join-Path $workspace 'artifacts\embeddings\mcphases\dual-v4-seed42\openmhc-dual.npy'
        Depress = Join-Path $workspace 'artifacts\embeddings\depress-fitbit-openmhc-source.npz'
        InPHRSym = Join-Path $workspace 'artifacts\embeddings\inphrsym-openmhc-source.npz'
        HRV = Join-Path $workspace 'artifacts\embeddings\hrv-mental-female\openmhc-seed42.npz'
        Pregnancy = Join-Path $workspace 'artifacts\embeddings\pregnancy-ga-official\openmhc-init.npz'
    }
    static = @{
        OpenMHCAdapted = $staticOpenMHC
        McPhases = $staticMcPhases
        Depress = $staticDepress
        InPHRSym = $staticInPHRSym
        HRV = $staticHRV
        Pregnancy = $staticPregnancy
    }
}
$arms = @(
    @{ Name = 'openmhc_gru'; Representation = 'openmhc'; Architecture = 'shared_backbone'; Hidden = 166 },
    @{ Name = 'static_adapter_gru'; Representation = 'static'; Architecture = 'shared_backbone'; Hidden = 166 },
    @{ Name = 'static_adapter_mmoe'; Representation = 'static'; Architecture = 'mmoe'; Hidden = 136 },
    @{ Name = 'static_adapter_dual_path'; Representation = 'static'; Architecture = 'dual_path_router'; Hidden = 128 }
)

$checkpointRoot = Join-Path $OutputRoot 'checkpoints'
$evaluationRoot = Join-Path $OutputRoot 'evaluations'
New-Item -ItemType Directory -Force -Path $checkpointRoot, $evaluationRoot | Out-Null

foreach ($arm in $arms) {
    $cache = $representations[$arm.Representation]
    foreach ($required in @($cache.OpenMHCAdapted, $cache.McPhases, $cache.Depress, $cache.InPHRSym, $cache.HRV, $cache.Pregnancy)) {
        if (-not (Test-Path -LiteralPath $required)) { throw "Missing representation cache: $required" }
    }
    $checkpoint = Join-Path $checkpointRoot ("{0}-seed{1}.pt" -f $arm.Name, $Seed)
    $evaluation = Join-Path $evaluationRoot ("{0}-seed{1}-validation" -f $arm.Name, $Seed)
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Write-Output ("[{0}] train {1}" -f (Get-Date -Format o), $arm.Name)
        Invoke-CheckedPython (Join-Path $workspace 'scripts\train_femmhc_joint.py') `
            --architecture $arm.Architecture --hidden-dim $arm.Hidden --output $checkpoint `
            --max-steps $MaxSteps --batch-size 16 --learning-rate 0.0002 --weight-decay 0.01 `
            --dropout 0.0 --routing-initial-logit -2.0 --maximum-days 60 --minimum-history-days 3 `
            --validate-every 100 --validation-batches 8 --cohort-sampling-temperature 0.5 `
            --checkpoint-selection final_step --seed $Seed --device $Device `
            --openmhc-native-cache $nativeOpenMHC --openmhc-adapted-cache $cache.OpenMHCAdapted `
            --mcphases-embeddings $cache.McPhases --depress-embeddings $cache.Depress `
            --inphrsym-embeddings $cache.InPHRSym --hrv-mental-embeddings $cache.HRV `
            --pregnancy-embeddings $cache.Pregnancy
    }
    if (-not (Test-Path -LiteralPath (Join-Path $evaluation 'per_task_metrics.csv'))) {
        Write-Output ("[{0}] evaluate {1}" -f (Get-Date -Format o), $arm.Name)
        Invoke-CheckedPython (Join-Path $workspace 'scripts\evaluate_femmhc_joint.py') `
            --checkpoint $checkpoint --output-dir $evaluation --split validation --device $Device `
            --openmhc-native-cache $nativeOpenMHC --openmhc-adapted-cache $cache.OpenMHCAdapted `
            --mcphases-embeddings $cache.McPhases --depress-embeddings $cache.Depress `
            --inphrsym-embeddings $cache.InPHRSym --hrv-mental-embeddings $cache.HRV `
            --pregnancy-embeddings $cache.Pregnancy
    }
}

Invoke-CheckedPython (Join-Path $workspace 'scripts\aggregate_femmhc_static_adapter_feasibility.py') `
    --root $OutputRoot --seed $Seed
