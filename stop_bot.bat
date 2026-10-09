@echo off
title Stop MT5 Algo Bot
color 0C
echo ============================================================
echo           STOPPING MT5 ALGORITHMIC TRADING TERMINAL
echo ============================================================
echo.
echo Searching for active bot processes on port 8501...
set FOUND=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    set FOUND=1
    echo Terminating PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

if "%FOUND%"=="1" (
    echo.
    echo [SUCCESS] Bot process stopped successfully!
) else (
    echo.
    echo [INFO] No bot was running on port 8501.
)
echo.
pause
