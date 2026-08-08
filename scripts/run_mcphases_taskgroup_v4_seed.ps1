param(
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [Parameter(Mandatory = $true)]
    [ValidateSet('cycle', 'symptoms', 'onset', 'hormones')]
    [string]$TaskGroup,
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
$tasksByGroup = @{
    cycle = @('cycle_phase')
    symptoms = @('cramps', 'mood_swing', 'fatigue', 'sleep_issue', 'perceived_stress', 'bloating', 'flow_volume')
    onset = @('menstrual_onset_24h', 'menstrual_onset_72h')
    hormones = @('lh', 'estrogen', 'pdg')
}
$selectedTasks = $tasksByGroup[$TaskGroup]
$runRoot = Join-Path $workspace ("artifacts\runs\seed-{0}" -f $Seed)
$stage1 = Join-Path $runRoot 'femmhc-openmhc-female-v4-best.ckpt'
$candidate = Join-Path $runRoot ("femmhc-stage1-v4-v2-{0}-heads.ckpt" -f $TaskGroup)
$baseline = Join-Path $runRoot ("openmhc-v2-{0}-heads-v4protocol.ckpt" -f $TaskGroup)
$embeddingRoot = Join-Path $workspace ("artifacts\embeddings\mcphases\supervised-v4-seed{0}" -f $Seed)
$candidateEmbedding = Join-Path $embeddingRoot 'femmhc-stage1-v4.npy'
$baselineEmbedding = Join-Path $embeddingRoot 'openmhc.npy'
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')

foreach ($required in @($stage1, $candidateEmbedding, $baselineEmbedding)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing prerequisite: $required" }
}

Write-Output ("[{0}] seed={1} group={2} train FemMHC task-family head" -f (Get-Date -Format o), $Seed, $TaskGroup)
& $python (Join-Path $workspace 'scripts\train_femmhc_pretrain.py') `
    --checkpoint $OpenMhcCheckpoint --femmhc-init $stage1 `
    --processed-dir $ProcessedDir --output $candidate `
    --max-steps 1000 --batch-size 2 --save-every 250 --keep-periodic-checkpoints `
    --task-head-version v2 --task-group $TaskGroup `
    --self-supervised-weight 0 --supervised-weight 1 --freeze-student `
    --seed $Seed --resume
if ($LASTEXITCODE -ne 0) { throw "FemMHC $TaskGroup head training failed" }

Write-Output ("[{0}] seed={1} group={2} train OpenMHC matched head" -f (Get-Date -Format o), $Seed, $TaskGroup)
& $python (Join-Path $workspace 'scripts\train_femmhc_pretrain.py') `
    --checkpoint $OpenMhcCheckpoint `
    --processed-dir $ProcessedDir --output $baseline `
    --max-steps 1000 --batch-size 2 --save-every 250 --keep-periodic-checkpoints `
    --task-head-version v2 --task-group $TaskGroup `
    --self-supervised-weight 0 --supervised-weight 1 --freeze-student `
    --seed $Seed --resume
if ($LASTEXITCODE -ne 0) { throw "OpenMHC $TaskGroup head training failed" }

$steps = @(250, 500, 750, 1000)
$candidateArgs = @('--processed-dir', $ProcessedDir)
$baselineArgs = @('--processed-dir', $ProcessedDir)
foreach ($step in $steps) {
    $candidateCheckpoint = Join-Path $runRoot ("femmhc-stage1-v4-v2-{0}-heads-step{1:d4}.ckpt" -f $TaskGroup, $step)
    $baselineCheckpoint = Join-Path $runRoot ("openmhc-v2-{0}-heads-v4protocol-step{1:d4}.ckpt" -f $TaskGroup, $step)
    $candidateArgs += @('--model', "FemMHC-$step|$candidateCheckpoint|$candidateEmbedding")
    $baselineArgs += @('--model', "OpenMHC-$step|$baselineCheckpoint|$baselineEmbedding")
}
foreach ($task in $selectedTasks) {
    $candidateArgs += @('--selection-task', $task)
    $baselineArgs += @('--selection-task', $task)
}
$candidateSelectionDir = Join-Path $runRoot ("direct-{0}-supervised-v4-femmhc" -f $TaskGroup)
$baselineSelectionDir = Join-Path $runRoot ("direct-{0}-supervised-v4-openmhc" -f $TaskGroup)
$candidateArgs += @('--output-dir', $candidateSelectionDir)
$baselineArgs += @('--output-dir', $baselineSelectionDir)
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_direct_heads.py') @candidateArgs
if ($LASTEXITCODE -ne 0) { throw "FemMHC $TaskGroup validation selection failed" }
& $python (Join-Path $workspace 'scripts\evaluate_femmhc_direct_heads.py') @baselineArgs
if ($LASTEXITCODE -ne 0) { throw "OpenMHC $TaskGroup validation selection failed" }

$candidateSelection = Get-Content -LiteralPath (Join-Path $candidateSelectionDir 'direct_head_selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$baselineSelection = Get-Content -LiteralPath (Join-Path $baselineSelectionDir 'direct_head_selection.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$candidateStep = [int]([regex]::Match($candidateSelection.selected_model, '(\d+)$').Groups[1].Value)
$baselineStep = [int]([regex]::Match($baselineSelection.selected_model, '(\d+)$').Groups[1].Value)
$candidateCheckpoint = Join-Path $runRoot ("femmhc-stage1-v4-v2-{0}-heads-step{1:d4}.ckpt" -f $TaskGroup, $candidateStep)
$baselineCheckpoint = Join-Path $runRoot ("openmhc-v2-{0}-heads-v4protocol-step{1:d4}.ckpt" -f $TaskGroup, $baselineStep)
$compareArgs = @(
    '--processed-dir', $ProcessedDir,
    '--model', "OpenMHC|$baselineCheckpoint|$baselineEmbedding",
    '--model', "FemMHC|$candidateCheckpoint|$candidateEmbedding",
    '--candidate', 'FemMHC',
    '--output-dir', (Join-Path $runRoot ("paired-bootstrap-{0}-supervised-v4" -f $TaskGroup)),
    '--bootstrap-draws', $BootstrapDraws,
    '--seed', $Seed
)
foreach ($task in $selectedTasks) { $compareArgs += @('--task', $task) }
& $python (Join-Path $workspace 'scripts\compare_femmhc_direct_heads.py') @compareArgs
if ($LASTEXITCODE -ne 0) { throw "$TaskGroup paired bootstrap failed" }

if ($TaskGroup -eq 'onset') {
    & $python (Join-Path $workspace 'scripts\evaluate_nested_onset.py') `
        --processed-dir $ProcessedDir `
        --baseline "OpenMHC-Onset|$baselineCheckpoint|$baselineEmbedding" `
        --candidate "FemMHC-Onset|$candidateCheckpoint|$candidateEmbedding" `
        --output-dir (Join-Path $runRoot 'calibrated-onset-taskgroup-v4') `
        --bootstrap-draws $BootstrapDraws --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "Task-specific calibrated onset failed" }
}

Write-Output ("seed={0} group={1} complete; candidate_step={2}; baseline_step={3}" -f $Seed, $TaskGroup, $candidateStep, $baselineStep)
