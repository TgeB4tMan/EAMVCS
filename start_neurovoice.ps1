# NeuroVoice System Bootstrapper
# This script starts the backend and instructions for the frontend.

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "       🚀 NEUROVOICE SYSTEM BOOTSTRAPPER      " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Environment Check
Write-Host "[1/3] Checking environment..." -ForegroundColor Yellow
$PythonPath = "python"
if (!(Get-Command $PythonPath -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Python not found in Path!" -ForegroundColor Red
    exit
}

# 2. Start Backend
Write-Host "[2/3] Starting Backend API..." -ForegroundColor Yellow
Write-Host "      (Server will run on http://localhost:8000)" -ForegroundColor Gray

# Use a separate window for the backend to keep it running
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m Backend.app"

# 3. Launch Frontend
Write-Host "[3/3] Launching Frontend Interface..." -ForegroundColor Yellow
$FrontendPath = "Frontend\index.html"
if (Test-Path $FrontendPath) {
    Start-Process $FrontendPath
} else {
    Write-Host "❌ Error: Frontend/index.html not found!" -ForegroundColor Red
}

Write-Host "-----------------------------------------------" -ForegroundColor Cyan
Write-Host "✅ System started successfully!" -ForegroundColor Green
Write-Host "-----------------------------------------------" -ForegroundColor Cyan
Write-Host "NOTES:" -ForegroundColor Gray
Write-Host "- Keep the other PowerShell window open (it's the Backend)." -ForegroundColor Gray
Write-Host "- Training is likely still running in your other terminal." -ForegroundColor Gray
Write-Host "- Once training finishes, the Backend will pick up the new model automatically." -ForegroundColor Gray
Write-Host "===============================================" -ForegroundColor Cyan
