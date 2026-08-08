param(
    [int]$Seed = 42,
    [int]$BatchSize = 32,
    [string]$DataRoot = '',
    [string]$SourceCheckpoint = '',
    [string]$CandidateCheckpoint = ''
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $workspace 'datasets\openmhc-xs'
}
if ([string]::IsNullOrWhiteSpace($SourceCheckpoint)) {
    $SourceCheckpoint = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
}
if ([string]::IsNullOrWhiteSpace($CandidateCheckpoint)) {
    $CandidateCheckpoint = Join-Path $workspace ("artifacts\runs\seed-{0}\femmhc-openmhc-female-v4-best.ckpt" -f $Seed)
}
$embeddingRoot = Join-Path $workspace 'artifacts\embeddings\openmhc-xs'
$candidateCache = Join-Path $embeddingRoot ("femmhc-stage1-v4-seed{0}" -f $Seed)
$baselineCache = Join-Path $embeddingRoot 'openmhc-lsm2'
$resultRoot = Join-Path $workspace ("artifacts\benchmark\openmhc-xs-seed{0}-stage1-v4" -f $Seed)

New-Item -ItemType Directory -Force -Path $candidateCache,$baselineCache,$resultRoot | Out-Null
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')
$env:MHC_DATA_DIR = $DataRoot

Write-Output ("[{0}] caching FemMHC daily embeddings" -f (Get-Date -Format o))
& $python (Join-Path $workspace 'scripts\cache_openmhc_benchmark_embeddings.py') `
    --model femmhc `
    --checkpoint $SourceCheckpoint `
    --femmhc-checkpoint $CandidateCheckpoint `
    --data-dir $DataRoot `
    --output-dir $candidateCache `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'FemMHC embedding extraction failed' }

Write-Output ("[{0}] caching OpenMHC daily embeddings" -f (Get-Date -Format o))
& $python (Join-Path $workspace 'scripts\cache_openmhc_benchmark_embeddings.py') `
    --model openmhc `
    --checkpoint $SourceCheckpoint `
    --data-dir $DataRoot `
    --output-dir $baselineCache `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'OpenMHC embedding extraction failed' }

Write-Output ("[{0}] running matched OpenMHC 32-task evaluation" -f (Get-Date -Format o))
& $python (Join-Path $workspace 'scripts\evaluate_openmhc_32_tasks.py') `
    --data-dir $DataRoot `
    --openmhc-cache $baselineCache `
    --femmhc-cache $candidateCache `
    --output-dir $resultRoot `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw 'OpenMHC 32-task evaluation failed' }

Write-Output ("[{0}] OpenMHC 32-task evaluation complete" -f (Get-Date -Format o))
