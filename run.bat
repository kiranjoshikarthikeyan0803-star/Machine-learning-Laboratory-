@echo off
title OncoVision AI - Cancer Prediction Machine Learning Studio
color 0B

echo ==============================================================================
echo           OncoVision AI - Cancer Prediction & ML Pipeline Studio
echo ==============================================================================
echo [1/3] Checking Python Environment...

:: Try specific Python paths if python is not in standard PATH
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
) else (
    if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
        set PYTHON_CMD="%LOCALAPPDATA%\Python\bin\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
        set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    ) else (
        echo [ERROR] Python not found! Please install Python 3.10+ from python.org.
        pause
        exit /b 1
    )
)

echo [*] Using Python: %PYTHON_CMD%

echo [2/3] Checking dataset and pre-trained models...
if not exist "data\cancer_features_dataset.csv" (
    echo [*] Generating clinical dataset and benchmark images...
    %PYTHON_CMD% generate_dataset.py
)

echo [3/3] Launching OncoVision AI Web Studio Server...
echo ==============================================================================
echo Server running at: http://localhost:8000
echo Opening web browser...
echo Press Ctrl+C in this terminal window to stop the server.
echo ==============================================================================

start "" http://localhost:8000
%PYTHON_CMD% app.py

pause
