param(
    [string]$DatasetRoot = "datasets",
    [switch]$IncludeLargeRawHrv
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-ResumableDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    $parent = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    & curl.exe `
        --location `
        --fail `
        --show-error `
        --retry 8 `
        --retry-delay 3 `
        --retry-all-errors `
        --continue-at - `
        --output $OutputPath `
        $Uri

    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Uri"
    }
}

function Assert-Md5 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected
    )

    $actual = (Get-FileHash -Algorithm MD5 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "MD5 mismatch for $Path. Expected $Expected, got $actual."
    }
}

$rawRoot = Join-Path $DatasetRoot "raw"
New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null

# LifeSnaps: Zenodo record 6832242, CC BY 4.0.
$lifeDir = Join-Path $rawRoot "lifesnaps_zenodo_6832242"
$lifeMetadata = Join-Path $lifeDir "zenodo_record_6832242.json"
Invoke-ResumableDownload `
    -Uri "https://zenodo.org/api/records/6832242" `
    -OutputPath $lifeMetadata
$lifeRecord = Get-Content -Raw -LiteralPath $lifeMetadata | ConvertFrom-Json
foreach ($file in $lifeRecord.files) {
    $target = Join-Path $lifeDir $file.key
    Invoke-ResumableDownload -Uri $file.links.self -OutputPath $target
    if ($file.checksum -match "^md5:(.+)$") {
        Assert-Md5 -Path $target -Expected $Matches[1]
    }
}

# SSAQS: Zenodo record 18706837. Public download; reuse license is not explicit.
$ssaqsDir = Join-Path $rawRoot "ssaqs_zenodo_18706837"
$ssaqsMetadata = Join-Path $ssaqsDir "zenodo_record_18706837.json"
Invoke-ResumableDownload `
    -Uri "https://zenodo.org/api/records/18706837" `
    -OutputPath $ssaqsMetadata
$ssaqsRecord = Get-Content -Raw -LiteralPath $ssaqsMetadata | ConvertFrom-Json
foreach ($file in $ssaqsRecord.files) {
    $target = Join-Path $ssaqsDir $file.key
    Invoke-ResumableDownload -Uri $file.links.self -OutputPath $target
    if ($file.checksum -match "^md5:(.+)$") {
        Assert-Md5 -Path $target -Expected $Matches[1]
    }
}

# OpenMHC XS: Harvard Dataverse doi:10.7910/DVN/ZYMJF6.
# Download individual files so interrupted transfers can resume independently.
$openMhcDir = Join-Path $rawRoot "openmhc_xs_dvn_zymjf6"
$openMhcMetadata = Join-Path $openMhcDir "dataverse_dataset_metadata.json"
$openMhcApi = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/ZYMJF6"
Invoke-ResumableDownload -Uri $openMhcApi -OutputPath $openMhcMetadata
$openMhcRecord = Get-Content -Raw -LiteralPath $openMhcMetadata | ConvertFrom-Json
$openMhcFiles = $openMhcRecord.data.latestVersion.files
foreach ($entry in $openMhcFiles) {
    $target = Join-Path $openMhcDir $entry.label
    $uri = "https://dataverse.harvard.edu/api/access/datafile/$($entry.dataFile.id)"
    Invoke-ResumableDownload -Uri $uri -OutputPath $target
    if ($entry.dataFile.checksum.type -eq "MD5") {
        Assert-Md5 -Path $target -Expected $entry.dataFile.checksum.value
    }
}

$marker = @{
    version = "xs"
    n_users = 593
    persistent_id = "doi:10.7910/DVN/ZYMJF6"
    downloaded_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$marker | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $openMhcDir "dataset_version.json")

# Pregnancy gestational-age clock: 1,083 pregnant participants, raw actigraphy
# plus processed sleep/activity tensors, Zenodo CC BY 4.0.
$pregnancyDir = Join-Path $rawRoot "pregnancy_ga_clock_zenodo_7689724"
$pregnancyMetadata = Join-Path $pregnancyDir "zenodo_record_7689724.json"
Invoke-ResumableDownload `
    -Uri "https://zenodo.org/api/records/7689724" `
    -OutputPath $pregnancyMetadata
$pregnancyRecord = Get-Content -Raw -LiteralPath $pregnancyMetadata | ConvertFrom-Json
foreach ($file in $pregnancyRecord.files) {
    $target = Join-Path $pregnancyDir $file.key
    Invoke-ResumableDownload -Uri $file.links.self -OutputPath $target
    if ($file.checksum -match "^md5:(.+)$") {
        Assert-Md5 -Path $target -Expected $Matches[1]
    }
}

# Continuous smartwatch HRV + sleep diaries: 49 participants, 25 women,
# Figshare CC BY 4.0. The processed files are ~50 MB. The optional raw PPG,
# motion and heart-rate archive is ~18.4 GB.
$hrvDir = Join-Path $rawRoot "wearable_hrv_sleep_figshare_28509740"
$hrvMetadata = Join-Path $hrvDir "figshare_article_28509740.json"
Invoke-ResumableDownload `
    -Uri "https://api.figshare.com/v2/articles/28509740" `
    -OutputPath $hrvMetadata
$hrvRecord = Get-Content -Raw -LiteralPath $hrvMetadata | ConvertFrom-Json
foreach ($file in $hrvRecord.files) {
    if (($file.name -eq "raw_data.zip") -and (-not $IncludeLargeRawHrv)) {
        continue
    }
    $target = Join-Path $hrvDir $file.name
    Invoke-ResumableDownload -Uri $file.download_url -OutputPath $target
    Assert-Md5 -Path $target -Expected $file.computed_md5
}

Write-Host "All directly downloadable datasets completed and checksums passed."
