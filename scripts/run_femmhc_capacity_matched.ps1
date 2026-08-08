param(
    [int[]]$Seeds = @(17, 42, 73),
    [string]$Device = 'cuda',
    [int]$MaxSteps = 1000,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $workspace 'artifacts\benchmark\femmhc-capacity-matched-v1'
}
$checkpointRoot = Join-Path $OutputRoot 'checkpoints'
$evaluationRoot = Join-Path $OutputRoot 'evaluations'
New-Item -ItemType Directory -Force -Path $checkpointRoot, $evaluationRoot | Out-Null
$env:PYTHONPATH = (Join-Path $workspace 'src') + ';' + (Join-Path $workspace 'third_party\OpenMHC\src')

$models = @(
    @{ Name = 'last_day_shared'; Hidden = 168 },
    @{ Name = 'shared_backbone'; Hidden = 166 },
    @{ Name = 'mmoe'; Hidden = 136 },
    @{ Name = 'dual_path_router'; Hidden = 128 }
)

foreach ($seed in $Seeds) {
    foreach ($model in $models) {
        $name = $model.Name
        $hidden = $model.Hidden
        $checkpoint = Join-Path $checkpointRoot ("{0}-seed{1}.pt" -f $name, $seed)
        $summary = [System.IO.Path]::ChangeExtension($checkpoint, '.json')
        $evaluation = Join-Path $evaluationRoot ("{0}-seed{1}-validation" -f $name, $seed)
        $metrics = Join-Path $evaluation 'per_task_metrics.csv'

        if (-not (Test-Path -LiteralPath $checkpoint) -or -not (Test-Path -LiteralPath $summary)) {
            Write-Output ("[{0}] train model={1} hidden={2} seed={3}" -f (Get-Date -Format o), $name, $hidden, $seed)
            & $python (Join-Path $workspace 'scripts\train_femmhc_joint.py') `
                --architecture $name `
                --hidden-dim $hidden `
                --output $checkpoint `
                --max-steps $MaxSteps `
                --batch-size 16 `
                --learning-rate 0.0002 `
                --weight-decay 0.01 `
                --dropout 0.0 `
                --routing-initial-logit -2.0 `
                --maximum-days 60 `
                --minimum-history-days 3 `
                --validate-every 100 `
                --validation-batches 8 `
                --cohort-sampling-temperature 0.5 `
                --checkpoint-selection final_step `
                --seed $seed `
                --device $Device
            if ($LASTEXITCODE -ne 0) { throw "Training failed: model=$name seed=$seed" }
        }

        if (-not (Test-Path -LiteralPath $metrics)) {
            Write-Output ("[{0}] evaluate model={1} seed={2}" -f (Get-Date -Format o), $name, $seed)
            & $python (Join-Path $workspace 'scripts\evaluate_femmhc_joint.py') `
                --checkpoint $checkpoint `
                --output-dir $evaluation `
                --split validation `
                --device $Device
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: model=$name seed=$seed" }
        }
    }
}

Write-Output ("[{0}] capacity-matched training and validation complete: {1}" -f (Get-Date -Format o), $OutputRoot)
