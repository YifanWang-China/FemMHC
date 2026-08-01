param(
    [string]$DatasetRoot = "datasets"
)

$ErrorActionPreference = "Stop"
$rawRoot = Join-Path $DatasetRoot "raw"

function Get-DirectorySummary {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{
            Dataset = Split-Path -Leaf $Path
            Exists = $false
            Files = 0
            SizeGB = 0
        }
    }

    # Dataset releases are flat at this stage. Ignore hidden resumable part
    # directories so only completed top-level files count toward the summary.
    $files = Get-ChildItem -File -LiteralPath $Path
    [pscustomobject]@{
        Dataset = Split-Path -Leaf $Path
        Exists = $true
        Files = $files.Count
        SizeGB = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1GB), 3)
    }
}

$paths = @(
    (Join-Path $rawRoot "lifesnaps_zenodo_6832242"),
    (Join-Path $rawRoot "ssaqs_zenodo_18706837"),
    (Join-Path $rawRoot "openmhc_xs_dvn_zymjf6"),
    (Join-Path $rawRoot "pregnancy_ga_clock_zenodo_7689724"),
    (Join-Path $rawRoot "wearable_hrv_sleep_figshare_28509740")
)

$paths | ForEach-Object { Get-DirectorySummary -Path $_ } | Format-Table -AutoSize
