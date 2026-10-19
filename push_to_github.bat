@echo off
title Push Cancer Prediction Project to GitHub
color 0A

echo ==============================================================================
echo       Push Project to: kiranjoshikarthikeyan0803-star/Machine-learning-Laboratory-
echo ==============================================================================
echo.
echo All files are staged, committed, and merged with your existing experiments.
echo.
echo Choose authentication method:
echo   [1] Interactive Web Browser Sign-In (Recommended)
echo   [2] Paste GitHub Personal Access Token (PAT)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo [*] Opening Git Credential Manager to sign in as kiranjoshikarthikeyan0803-star...
    git push origin main
) else if "%choice%"=="2" (
    echo.
    set /p token="Paste your GitHub Personal Access Token: "
    echo [*] Pushing with token...
    git push https://kiranjoshikarthikeyan0803-star:%token%@github.com/kiranjoshikarthikeyan0803-star/Machine-learning-Laboratory-.git main
) else (
    echo [!] Invalid choice.
)

echo.
echo ==============================================================================
pause
