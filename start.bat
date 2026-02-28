@echo off
title FinSight AI - Launcher
color 0A

echo.
echo  ============================================
echo   FinSight AI - Starting Services
echo  ============================================
echo.

REM Start Backend in a new window
echo  [1/2] Starting Backend (FastAPI + Uvicorn)...
start "FinSight AI - Backend" cmd /k "cd /d C:\Project\finsightai\backend && color 0B && echo  FinSight AI Backend && echo  ======================== && uvicorn main:app --reload"

REM Small delay so backend window appears first
timeout /t 2 /nobreak > nul

REM Start Frontend in a new window
echo  [2/2] Starting Frontend (Vite Dev Server)...
start "FinSight AI - Frontend" cmd /k "cd /d C:\Project\finsightai\frontend && color 0D && echo  FinSight AI Frontend && echo  ======================== && npm run dev"

echo.
echo  Both services are starting in separate windows.
echo  Frontend: http://localhost:5173
echo  Backend:  http://localhost:8000
echo.
echo  Opening browser in 5 seconds...
timeout /t 5 /nobreak > nul

REM Open browser
start "" "http://localhost:5173"

echo  Done! You can close this window.
timeout /t 3 /nobreak > nul
exit
