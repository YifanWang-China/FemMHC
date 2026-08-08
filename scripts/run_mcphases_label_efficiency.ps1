param(
    [ValidateSet(0, 1)]
    [int]$Batch = 0,
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = "$(Join-Path $projectRoot 'src');$(Join-Path $projectRoot 'third_party\OpenMHC\src')"
$outputRoot = Join-Path $projectRoot "artifacts\benchmark\mcphases-label-efficiency-v4-three-seed\per-task"
$logRoot = Join-Path $projectRoot "artifacts\benchmark\mcphases-label-efficiency-v4-three-seed\logs"
New-Item -ItemType Directory -Force -Path $outputRoot, $logRoot | Out-Null

$tasks = @(
    "cycle_phase",
    "cramps",
    "menstrual_onset_24h",
    "mood_swing",
    "estrogen",
    "menstrual_onset_72h"
)

for ($index = $Batch; $index -lt $tasks.Count; $index += 2) {
    $task = $tasks[$index]
    $taskOutput = Join-Path $outputRoot $task
    $log = Join-Path $logRoot "$task.log"
    $expected = Join-Path $taskOutput "summary.json"
    if (Test-Path -LiteralPath $expected) {
        "SKIP $task (complete)" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }
    & $python (Join-Path $projectRoot "scripts\evaluate_mcphases_label_efficiency.py") `
        --processed-dir (Join-Path $projectRoot "processed\mcphases") `
        --baseline "OpenMHC-dual=$(Join-Path $projectRoot 'artifacts\embeddings\mcphases\dual-v4-seed42\openmhc-dual.npy')" `
        --candidate "FemMHC-seed42=$(Join-Path $projectRoot 'artifacts\embeddings\mcphases\dual-v4-seed42\femmhc-dual.npy')" `
        --candidate "FemMHC-seed43=$(Join-Path $projectRoot 'artifacts\embeddings\mcphases\dual-v4-seed43\femmhc-dual.npy')" `
        --candidate "FemMHC-seed44=$(Join-Path $projectRoot 'artifacts\embeddings\mcphases\dual-v4-seed44\femmhc-dual.npy')" `
        --task $task `
        --output-dir $taskOutput `
        --fraction 0.01 --fraction 0.05 --fraction 0.10 --fraction 0.25 --fraction 1.0 `
        --repeats 5 --inner-folds 3 --jobs $Jobs --seed 20260803 *> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Label-efficiency run failed for $task; see $log"
    }
}
