# Rebuilds the distributable offline bundle from scratch.
#
#   1. shinylive export app site      (the static website, from scratch)
#   2. copies site/ into offline/
#   3. zips offline/ into OnTarget-Offline-Windows.zip (~53 MB)
#
# Run it from the repository root:  powershell -File scripts\build-offline-zip.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "1/3  Exporting the ShinyLive site..." -ForegroundColor Cyan
# A fresh export every time: exporting over an older site/ leaves ShinyLive
# warning that its own runtime files differ from the ones already there.
if (Test-Path "site") { Remove-Item "site" -Recurse -Force }
shinylive export app site

Write-Host "2/3  Staging the offline bundle..." -ForegroundColor Cyan
if (Test-Path "offline\site") { Remove-Item "offline\site" -Recurse -Force }
Copy-Item "site" "offline\site" -Recurse

Write-Host "3/3  Zipping..." -ForegroundColor Cyan
if (Test-Path "OnTarget-Offline-Windows.zip") { Remove-Item "OnTarget-Offline-Windows.zip" -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
# Virus scanning and cloud sync can still hold a file that was copied a second
# ago, so give them a few seconds and try again rather than failing the build.
$zipped = $false
foreach ($attempt in 1..5) {
    try {
        [System.IO.Compression.ZipFile]::CreateFromDirectory(
            (Join-Path $root "offline"), (Join-Path $root "OnTarget-Offline-Windows.zip"))
        $zipped = $true
        break
    } catch {
        if (Test-Path "OnTarget-Offline-Windows.zip") { Remove-Item "OnTarget-Offline-Windows.zip" -Force }
        Write-Host "     a file was locked (attempt $attempt of 5) - retrying in 5s..." -ForegroundColor DarkYellow
        Start-Sleep -Seconds 5
    }
}
if (-not $zipped) { throw "Could not write the zip: a file under offline\ stayed locked." }

$mb = [math]::Round((Get-Item "OnTarget-Offline-Windows.zip").Length / 1MB, 1)
Write-Host ""
Write-Host "Done - OnTarget-Offline-Windows.zip ($mb MB)" -ForegroundColor Green
