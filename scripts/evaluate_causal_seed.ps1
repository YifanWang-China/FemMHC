param(
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [int]$BootstrapDraws = 2000
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$baseCheckpoint = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
$processed = Join-Path $workspace 'processed\mcphases'
$runRoot = Join-Path $workspace ("artifacts\runs\seed-{0}" -f $Seed)
$embeddingRoot = Join-Path $runRoot 'embeddings'
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')
New-Item -ItemType Directory -Force -Path $embeddingRoot | Out-Null

$steps = @(2000, 2500, 3000)
foreach ($step in $steps) {
    $checkpoint = Join-Path $runRoot ("femmhc-mcphases-causal-step{0:d4}.ckpt" -f $step)
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Missing causal checkpoint: $checkpoint"
    }
    $embedding = Join-Path $embeddingRoot ("causal-{0}.npy" -f $step)
    & $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
        --checkpoint $baseCheckpoint `
        --femmhc-checkpoint $checkpoint `
        --processed-dir $processed `
        --output $embedding `
        --batch-size 16
    if ($LASTEXITCODE -ne 0) { throw "Embedding cache failed at step $step" }
}

$baseline = Join-Path $runRoot 'baseline-openmhc-taskheads.ckpt'
& $python (Join-Path $workspace 'scripts\train_femmhc_pretrain.py') `
    --checkpoint $baseCheckpoint `
    --processed-dir $processed `
    --output $baseline `
    --max-steps 3000 `
    --batch-size 2 `
    --save-every 500 `
    --keep-periodic-checkpoints `
    --seed $Seed `
    --self-supervised-weight 0 `
    --supervised-weight 1 `
    --freeze-student `
    --resume
if ($LASTEXITCODE -ne 0) { throw "Matched OpenMHC head baseline failed" }

$baselineEmbedding = Join-Path $embeddingRoot 'baseline-openmhc-matched.npy'
$baselineStep2500 = Join-Path $runRoot 'baseline-openmhc-taskheads-step2500.ckpt'
& $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
    --checkpoint $baseCheckpoint `
    --femmhc-checkpoint $baselineStep2500 `
    --processed-dir $processed `
    --output $baselineEmbedding `
    --batch-size 16
if ($LASTEXITCODE -ne 0) { throw "Matched OpenMHC embedding cache failed" }

$candidateModels = foreach ($step in $steps) {
    $checkpoint = Join-Path $runRoot ("femmhc-mcphases-causal-step{0:d4}.ckpt" -f $step)
    $embedding = Join-Path $embeddingRoot ("causal-{0}.npy" -f $step)
    "Causal-$step|$checkpoint|$embedding"
}
$candidateOutput = Join-Path $runRoot 'direct-head-causal'
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_direct_heads.py') `
    --processed-dir $processed `
    --model $candidateModels[0] `
    --model $candidateModels[1] `
    --model $candidateModels[2] `
    --output-dir $candidateOutput
if ($LASTEXITCODE -ne 0) { throw "Causal direct-head evaluation failed" }

$baselineModels = foreach ($step in $steps) {
    $checkpoint = Join-Path $runRoot ("baseline-openmhc-taskheads-step{0:d4}.ckpt" -f $step)
    "OpenMHC-$step|$checkpoint|$baselineEmbedding"
}
$baselineOutput = Join-Path $runRoot 'direct-head-baseline-openmhc'
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_direct_heads.py') `
    --processed-dir $processed `
    --model $baselineModels[0] `
    --model $baselineModels[1] `
    --model $baselineModels[2] `
    --output-dir $baselineOutput
if ($LASTEXITCODE -ne 0) { throw "OpenMHC direct-head evaluation failed" }

$candidateSelection = Get-Content -LiteralPath (Join-Path $candidateOutput 'direct_head_selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$baselineSelection = Get-Content -LiteralPath (Join-Path $baselineOutput 'direct_head_selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$candidateStep = [int]([regex]::Match($candidateSelection.selected_model, '(\d+)$').Groups[1].Value)
$baselineStep = [int]([regex]::Match($baselineSelection.selected_model, '(\d+)$').Groups[1].Value)
$candidateCheckpoint = Join-Path $runRoot ("femmhc-mcphases-causal-step{0:d4}.ckpt" -f $candidateStep)
$candidateEmbedding = Join-Path $embeddingRoot ("causal-{0}.npy" -f $candidateStep)
$baselineCheckpoint = Join-Path $runRoot ("baseline-openmhc-taskheads-step{0:d4}.ckpt" -f $baselineStep)
$candidateSpec = "FemMHC|$candidateCheckpoint|$candidateEmbedding"
$baselineSpec = "OpenMHC|$baselineCheckpoint|$baselineEmbedding"

& $python (Join-Path $workspace 'scripts\compare_femmhc_direct_heads.py') `
    --processed-dir $processed `
    --model $baselineSpec `
    --model $candidateSpec `
    --candidate FemMHC `
    --output-dir (Join-Path $runRoot 'paired-bootstrap') `
    --bootstrap-draws $BootstrapDraws `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Paired bootstrap failed" }

Write-Output ("seed={0} evaluation complete; candidate_step={1}; baseline_step={2}" -f $Seed, $candidateStep, $baselineStep)
