param(
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [int]$BootstrapDraws = 2000,
    [string]$OpenMhcCheckpoint = '',
    [string]$ProcessedDir = ''
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if ([string]::IsNullOrWhiteSpace($OpenMhcCheckpoint)) {
    $OpenMhcCheckpoint = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
}
if ([string]::IsNullOrWhiteSpace($ProcessedDir)) {
    $ProcessedDir = Join-Path $workspace 'processed\mcphases'
}
$runRoot = Join-Path $workspace ("artifacts\runs\seed-{0}" -f $Seed)
$stage1 = Join-Path $runRoot 'femmhc-openmhc-female-v4-best.ckpt'
$candidate = Join-Path $runRoot 'femmhc-stage1-v4-v2-heads.ckpt'
$baseline = Join-Path $runRoot 'openmhc-v2-heads-v4protocol.ckpt'
$embeddingRoot = Join-Path $workspace ("artifacts\embeddings\mcphases\supervised-v4-seed{0}" -f $Seed)
$candidateEmbedding = Join-Path $embeddingRoot 'femmhc-stage1-v4.npy'
$baselineEmbedding = Join-Path $embeddingRoot 'openmhc.npy'
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')
New-Item -ItemType Directory -Force -Path $runRoot,$embeddingRoot | Out-Null

if (-not (Test-Path -LiteralPath $stage1)) {
    throw "Missing stage1-v4 checkpoint: $stage1"
}

Write-Output ("[{0}] seed={1} train frozen FemMHC v2 heads" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\train_femmhc_pretrain.py') `
    --checkpoint $OpenMhcCheckpoint `
    --femmhc-init $stage1 `
    --processed-dir $ProcessedDir `
    --output $candidate `
    --max-steps 3000 --batch-size 2 --save-every 500 `
    --keep-periodic-checkpoints --task-head-version v2 --task-group all `
    --self-supervised-weight 0 --supervised-weight 1 --freeze-student `
    --seed $Seed --resume
if ($LASTEXITCODE -ne 0) { throw "FemMHC v2 head training failed" }

Write-Output ("[{0}] seed={1} train matched OpenMHC v2 heads" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\train_femmhc_pretrain.py') `
    --checkpoint $OpenMhcCheckpoint `
    --processed-dir $ProcessedDir `
    --output $baseline `
    --max-steps 3000 --batch-size 2 --save-every 500 `
    --keep-periodic-checkpoints --task-head-version v2 --task-group all `
    --self-supervised-weight 0 --supervised-weight 1 --freeze-student `
    --seed $Seed --resume
if ($LASTEXITCODE -ne 0) { throw "OpenMHC v2 head training failed" }

Write-Output ("[{0}] seed={1} cache frozen representations" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
    --checkpoint $OpenMhcCheckpoint --femmhc-checkpoint $candidate `
    --processed-dir $ProcessedDir --output $candidateEmbedding --batch-size 32
if ($LASTEXITCODE -ne 0) { throw "FemMHC embedding cache failed" }
& $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
    --checkpoint $OpenMhcCheckpoint --femmhc-checkpoint $baseline `
    --processed-dir $ProcessedDir --output $baselineEmbedding --batch-size 32
if ($LASTEXITCODE -ne 0) { throw "OpenMHC embedding cache failed" }

$steps = @(2000, 2500, 3000)
$candidateModels = foreach ($step in $steps) {
    $checkpoint = Join-Path $runRoot ("femmhc-stage1-v4-v2-heads-step{0:d4}.ckpt" -f $step)
    "FemMHC-$step|$checkpoint|$candidateEmbedding"
}
$baselineModels = foreach ($step in $steps) {
    $checkpoint = Join-Path $runRoot ("openmhc-v2-heads-v4protocol-step{0:d4}.ckpt" -f $step)
    "OpenMHC-$step|$checkpoint|$baselineEmbedding"
}
$candidateSelectionDir = Join-Path $runRoot 'direct-head-supervised-v4-femmhc'
$baselineSelectionDir = Join-Path $runRoot 'direct-head-supervised-v4-openmhc'
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_direct_heads.py') `
    --processed-dir $ProcessedDir `
    --model $candidateModels[0] --model $candidateModels[1] --model $candidateModels[2] `
    --output-dir $candidateSelectionDir
if ($LASTEXITCODE -ne 0) { throw "FemMHC validation selection failed" }
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_direct_heads.py') `
    --processed-dir $ProcessedDir `
    --model $baselineModels[0] --model $baselineModels[1] --model $baselineModels[2] `
    --output-dir $baselineSelectionDir
if ($LASTEXITCODE -ne 0) { throw "OpenMHC validation selection failed" }

$candidateSelection = Get-Content -LiteralPath (Join-Path $candidateSelectionDir 'direct_head_selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$baselineSelection = Get-Content -LiteralPath (Join-Path $baselineSelectionDir 'direct_head_selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$candidateStep = [int]([regex]::Match($candidateSelection.selected_model, '(\d+)$').Groups[1].Value)
$baselineStep = [int]([regex]::Match($baselineSelection.selected_model, '(\d+)$').Groups[1].Value)
$candidateCheckpoint = Join-Path $runRoot ("femmhc-stage1-v4-v2-heads-step{0:d4}.ckpt" -f $candidateStep)
$baselineCheckpoint = Join-Path $runRoot ("openmhc-v2-heads-v4protocol-step{0:d4}.ckpt" -f $baselineStep)
$candidateSpec = "FemMHC|$candidateCheckpoint|$candidateEmbedding"
$baselineSpec = "OpenMHC|$baselineCheckpoint|$baselineEmbedding"

Write-Output ("[{0}] seed={1} paired participant bootstrap" -f (Get-Date -Format o), $Seed)
& $python (Join-Path $workspace 'scripts\compare_femmhc_direct_heads.py') `
    --processed-dir $ProcessedDir `
    --model $baselineSpec --model $candidateSpec --candidate FemMHC `
    --output-dir (Join-Path $runRoot 'paired-bootstrap-supervised-v4') `
    --bootstrap-draws $BootstrapDraws --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Paired bootstrap failed" }

$onsetBaseline = "OpenMHC-Onset|$baselineCheckpoint|$baselineEmbedding"
$onsetCandidate = "FemMHC-Onset|$candidateCheckpoint|$candidateEmbedding"
& $python (Join-Path $workspace 'scripts\evaluate_nested_onset.py') `
    --processed-dir $ProcessedDir `
    --baseline $onsetBaseline --candidate $onsetCandidate `
    --output-dir (Join-Path $runRoot 'calibrated-onset-supervised-v4') `
    --bootstrap-draws $BootstrapDraws --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Calibrated onset evaluation failed" }

Write-Output ("seed={0} complete; candidate_step={1}; baseline_step={2}" -f $Seed, $candidateStep, $baselineStep)
