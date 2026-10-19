# Push Project to GitHub PowerShell Script
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " Push Project to: kiranjoshikarthikeyan0803-star/Machine-learning-Laboratory- " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "All files are staged, committed, and merged with your existing experiments."
Write-Host ""
Write-Host "Choose authentication method:"
Write-Host "  [1] Interactive Web Browser Sign-In (Recommended)"
Write-Host "  [2] Paste GitHub Personal Access Token (PAT)"
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host "`n[*] Opening Git Credential Manager to sign in as kiranjoshikarthikeyan0803-star..." -ForegroundColor Yellow
    git push origin main
} elseif ($choice -eq "2") {
    $token = Read-Host "Paste your GitHub Personal Access Token"
    Write-Host "`n[*] Pushing to GitHub with token..." -ForegroundColor Yellow
    git push "https://kiranjoshikarthikeyan0803-star:${token}@github.com/kiranjoshikarthikeyan0803-star/Machine-learning-Laboratory-.git" main
} else {
    Write-Host "[!] Invalid selection." -ForegroundColor Red
}

Write-Host "`n==============================================================================" -ForegroundColor Cyan
Pause
