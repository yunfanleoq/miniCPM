# Activate venv and enable UTF-8 console output
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& "$ProjectRoot\.venv\Scripts\Activate.ps1"
Write-Host "venv activated, UTF-8 enabled" -ForegroundColor Green
