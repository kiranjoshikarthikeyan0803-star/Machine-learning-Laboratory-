# PowerShell Launcher for OncoVision AI
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "          OncoVision AI - Cancer Prediction & ML Pipeline Studio             " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# Find python
$pythonPath = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Test-Path "$env:LOCALAPPDATA\Python\bin\python.exe") {
        $pythonPath = "$env:LOCALAPPDATA\Python\bin\python.exe"
    } else {
        Write-Error "Python was not found in PATH or standard install locations."
        exit 1
    }
}

Write-Host "[*] Using Python: $pythonPath" -ForegroundColor Green

# Generate dataset if needed
if (-not (Test-Path "data\cancer_features_dataset.csv")) {
    Write-Host "[*] Generating clinical dataset and benchmark images..." -ForegroundColor Yellow
    & $pythonPath generate_dataset.py
}

Write-Host "[*] Launching Web Application at http://localhost:8000..." -ForegroundColor Cyan
Start-Process "http://localhost:8000"
& $pythonPath app.py
