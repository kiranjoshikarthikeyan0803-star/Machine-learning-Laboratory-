# Root PowerShell Launcher
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$scriptDir\Cancer Prediction"
& ".\run.ps1"
