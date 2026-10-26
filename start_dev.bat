@echo off
title QNU Library Assistant - Launcher
chcp 65001 >nul

echo =======================================================
echo    KHỞI ĐỘNG HỆ THỐNG TRỢ LÝ THƯ VIỆN QNU (BE + FE)
echo =======================================================
echo.

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

echo [1/2] Đang khởi động Backend FastAPI (Port 8000)...
start "QNU Backend (FastAPI)" cmd /k "chcp 65001 >nul && set PYTHONIOENCODING=utf-8 && .venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Đang khởi động Frontend React Vite (Port 5173)...
start "QNU Frontend (React)" cmd /k "cd /d %ROOT_DIR%frontend && npm run dev"

echo.
echo =======================================================
echo    HỆ THỐNG ĐÃ ĐƯỢC MỞ TRONG 2 CỬA SỔ RIÊNG BIỆT!
echo.
echo    - Frontend UI : http://localhost:5173
echo    - Backend API : http://127.0.0.1:8000
echo    - Swagger Docs: http://127.0.0.1:8000/docs
echo =======================================================
echo.
pause
