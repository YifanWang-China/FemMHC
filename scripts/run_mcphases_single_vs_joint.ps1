param(
    [int[]]$Seeds = @(17, 42, 73),
    [int]$MaxSteps = 1000,
    [int]$BatchSize = 16,
    [int]$BootstrapDraws = 2000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$outputRoot = Join-Path $projectRoot "artifacts\benchmark\mcphases-single-vs-joint-six-task"
$logRoot = Join-Path $outputRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONPATH = "$(Join-Path $projectRoot 'src');$(Join-Path $projectRoot 'third_party\OpenMHC\src')"

function Invoke-Run {
    param(
        [int]$Seed,
        [string]$Mode,
        [string]$Group
    )
    $name = if ($Mode -eq "joint") { "joint6" } else { "single-$Group" }
    $outputDir = Join-Path (Join-Path $outputRoot "seed-$Seed") $name
    $expected = Join-Path $outputDir "validation_predictions.csv"
    if (Test-Path -LiteralPath $expected) {
        Write-Output "skip seed=$Seed model=$name (complete)"
        return
    }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $log = Join-Path $logRoot ("seed-{0}-{1}.log" -f $Seed, $name)
    $arguments = @(
        (Join-Path $projectRoot "scripts\train_mcphases_single_vs_joint.py"),
        "--mode", $Mode,
        "--seed", $Seed,
        "--processed-dir", (Join-Path $projectRoot "processed\mcphases"),
        "--embeddings", (Join-Path $projectRoot "artifacts\embeddings\mcphases\dual-v4-seed42\femmhc-dual.npy"),
        "--output-dir", $outputDir,
        "--max-steps", $MaxSteps,
        "--batch-size", $BatchSize,
        "--device", "cuda"
    )
    if ($Mode -eq "single") {
        $arguments += @("--task-group", $Group)
    }
    Write-Output "start seed=$Seed model=$name"
    & $python @arguments *> $log
    if ($LASTEXITCODE -ne 0) {
        throw "run failed: seed=$Seed model=$name; see $log"
    }
    Write-Output "complete seed=$Seed model=$name"
}

foreach ($seed in $Seeds) {
    Invoke-Run -Seed $seed -Mode "joint" -Group ""
    foreach ($group in @("cycle", "onset", "cramps", "mood", "sleep")) {
        Invoke-Run -Seed $seed -Mode "single" -Group $group
    }
}

& $python (Join-Path $projectRoot "scripts\aggregate_mcphases_single_vs_joint.py") `
    --root $outputRoot `
    --seeds $Seeds `
    --bootstrap-draws $BootstrapDraws
if ($LASTEXITCODE -ne 0) {
    throw "aggregation failed"
}
Write-Output "experiment complete: $outputRoot"
