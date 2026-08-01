param(
    [int]$Seed = 42,
    [int]$PollSeconds = 30,
    [int]$TimeoutHours = 6
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$runRoot = Join-Path $workspace ("artifacts\runs\seed-{0}" -f $Seed)
$stage1 = Join-Path $runRoot 'femmhc-openmhc-female.ckpt'
$stage2 = Join-Path $runRoot 'femmhc-mcphases.ckpt'
$stage2Report = Join-Path $runRoot 'femmhc-mcphases.json'
$source = Join-Path $workspace 'artifacts\checkpoints\lsm2-daily\loss=0.2706.ckpt'
$processed = Join-Path $workspace 'processed\mcphases'
$embeddingRoot = Join-Path $runRoot 'embeddings'
$benchmarkRoot = Join-Path $runRoot 'benchmark'
$deadline = (Get-Date).AddHours($TimeoutHours)

New-Item -ItemType Directory -Force -Path $embeddingRoot,$benchmarkRoot | Out-Null
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')

Write-Output ("[{0}] waiting for seed={1} training" -f (Get-Date -Format o), $Seed)
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $stage2Report) {
        try {
            $report = Get-Content -LiteralPath $stage2Report -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($report.status -eq 'complete' -and [int]$report.steps -ge [int]$report.max_steps) {
                break
            }
        }
        catch {
            # The atomic writer may be between checkpoint and JSON replacement.
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
if (-not (Test-Path -LiteralPath $stage2Report)) {
    throw "Timed out waiting for stage-2 report"
}
$report = Get-Content -LiteralPath $stage2Report -Raw -Encoding UTF8 | ConvertFrom-Json
if ($report.status -ne 'complete') {
    throw "Stage-2 did not complete before timeout"
}

$initialEmbedding = Join-Path $embeddingRoot 'openmhc-initialization.npy'
$stage1Embedding = Join-Path $embeddingRoot 'openmhc-female.npy'
$stage2Embedding = Join-Path $embeddingRoot 'femmhc-final.npy'

Write-Output ("[{0}] caching OpenMHC initialization" -f (Get-Date -Format o))
& $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
    --checkpoint $source --processed-dir $processed --output $initialEmbedding --batch-size 16
if ($LASTEXITCODE -ne 0) { throw "Initial embedding cache failed" }

Write-Output ("[{0}] caching OpenMHC female stage" -f (Get-Date -Format o))
& $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
    --checkpoint $source --femmhc-checkpoint $stage1 `
    --processed-dir $processed --output $stage1Embedding --batch-size 16
if ($LASTEXITCODE -ne 0) { throw "Stage-1 embedding cache failed" }

Write-Output ("[{0}] caching final FemMHC" -f (Get-Date -Format o))
& $python (Join-Path $workspace 'scripts\cache_femmhc_embeddings.py') `
    --checkpoint $source --femmhc-checkpoint $stage2 `
    --processed-dir $processed --output $stage2Embedding --batch-size 16
if ($LASTEXITCODE -ne 0) { throw "Stage-2 embedding cache failed" }

Write-Output ("[{0}] running frozen probes" -f (Get-Date -Format o))
$initialArgument = "OpenMHC-init=$initialEmbedding"
$stage1Argument = "FemMHC-stage1=$stage1Embedding"
$stage2Argument = "FemMHC-final=$stage2Embedding"
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_mcphases.py') `
    --processed-dir $processed `
    --embedding $initialArgument `
    --embedding $stage1Argument `
    --embedding $stage2Argument `
    --output-dir $benchmarkRoot `
    --bootstrap-draws 1000 `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Frozen-probe benchmark failed" }

Write-Output ("[{0}] seed={1} evaluation complete" -f (Get-Date -Format o), $Seed)
