@echo off
echo =======================================================
echo    Starting Mutual Fund FAQ Assistant (Phases 1-7)
echo =======================================================

:: Try to find node
set "NODE_EXE=node"
where node >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\nodejs\node.exe" (
        set "NODE_EXE=C:\Program Files\nodejs\node.exe"
    ) else if exist "%LocalAppData%\Programs\cursor\resources\app\resources\helpers\node.exe" (
        set "NODE_EXE=%LocalAppData%\Programs\cursor\resources\app\resources\helpers\node.exe"
    ) else (
        echo [ERROR] Node.js not found in PATH or standard locations.
        echo Please install Node.js from https://nodejs.org/
        pause
        exit /b 1
    )
)

echo.
echo [1/2] Booting FastAPI Backend Orchestrator (Phases 1-6)...
start "FastAPI Backend" cmd /k "python -m uvicorn phase_6_response_delivery.api:app --host 127.0.0.1 --port 8000"

echo.
echo [2/2] Booting React UI (Phase 7)...
cd phase_7_frontend
:: Use node to run vite directly if npm is missing
where npm >nul 2>&1
if %errorlevel% equ 0 (
    start "React Frontend" cmd /k "npm run dev"
) else (
    echo [INFO] npm not found, using node fallback for vite...
    start "React Frontend" cmd /k "\"%NODE_EXE%\" node_modules\vite\bin\vite.js --host 127.0.0.1"
)

echo.
echo Both systems are now booting up!
echo Backend API will be available at: http://localhost:8000
echo Frontend UI will be available at: http://localhost:5173
echo =======================================================
