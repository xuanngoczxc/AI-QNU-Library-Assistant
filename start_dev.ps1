# PowerShell script khởi động đồng thời Backend và Frontend
$host.UI.RawUI.WindowTitle = "QNU Library Assistant Launcher"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   KHỞI ĐỘNG HỆ THỐNG TRỢ LÝ THƯ VIỆN QNU (BE + FE)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Start Backend in separate window
Write-Host "[1/2] Đang khởi động Backend FastAPI (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootDir'; `$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload"

# 2. Start Frontend in separate window
Write-Host "[2/2] Đang khởi động Frontend React Vite (Port 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootDir\frontend'; npm run dev"

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "   HỆ THỐNG ĐÃ ĐƯỢC MỞ TRONG 2 CỬA SỔ RIÊNG BIỆT!" -ForegroundColor Green
Write-Host ""
Write-Host "   - Frontend UI : http://localhost:5173" -ForegroundColor White
Write-Host "   - Backend API : http://127.0.0.1:8000" -ForegroundColor White
Write-Host "   - Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor Green
