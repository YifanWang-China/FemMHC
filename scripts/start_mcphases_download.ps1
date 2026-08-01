param(
    [Parameter(Mandatory = $true)]
    [string]$Username,
    [string]$Destination = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $workspace = Split-Path -Parent $PSScriptRoot
    $Destination = Join-Path $workspace "datasets\mcphases\1.0.0"
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "mcPHASES PhysioNet Download"
$form.Size = New-Object System.Drawing.Size(470, 205)
$form.StartPosition = "CenterScreen"
$form.TopMost = $true
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $false

$description = New-Object System.Windows.Forms.Label
$description.Location = New-Object System.Drawing.Point(20, 18)
$description.Size = New-Object System.Drawing.Size(420, 42)
$description.Text = "Enter the PhysioNet password for $Username.`r`nThe password is not saved to files or logs."
$form.Controls.Add($description)

$passwordBox = New-Object System.Windows.Forms.TextBox
$passwordBox.Location = New-Object System.Drawing.Point(20, 72)
$passwordBox.Size = New-Object System.Drawing.Size(420, 26)
$passwordBox.UseSystemPasswordChar = $true
$form.Controls.Add($passwordBox)

$okButton = New-Object System.Windows.Forms.Button
$okButton.Text = "Start"
$okButton.Location = New-Object System.Drawing.Point(250, 115)
$okButton.Size = New-Object System.Drawing.Size(90, 30)
$okButton.Add_Click({
    if ($passwordBox.Text.Length -gt 0) {
        $form.Tag = $passwordBox.Text
        $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $form.Close()
    }
})
$form.Controls.Add($okButton)

$cancelButton = New-Object System.Windows.Forms.Button
$cancelButton.Text = "Cancel"
$cancelButton.Location = New-Object System.Drawing.Point(350, 115)
$cancelButton.Size = New-Object System.Drawing.Size(90, 30)
$cancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.Controls.Add($cancelButton)

$form.AcceptButton = $okButton
$form.CancelButton = $cancelButton
$form.Add_Shown({ $passwordBox.Focus() })

$dialogResult = $form.ShowDialog()
if ($dialogResult -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host "Download cancelled by user."
    exit 2
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$env:PHYSIONET_PASSWORD = [string]$form.Tag
$form.Tag = $null
$passwordBox.Clear()

try {
    & $Python `
        (Join-Path $PSScriptRoot "download_physionet_recursive.py") `
        --url "https://physionet.org/files/mcphases/1.0.0/" `
        --user $Username `
        --dest $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "Downloader exit code: $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:PHYSIONET_PASSWORD -ErrorAction SilentlyContinue
}
