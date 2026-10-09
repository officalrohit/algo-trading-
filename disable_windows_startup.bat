@echo off
title Disable Auto-Start on Windows Boot
color 0E
echo ============================================================
echo      DISABLE AUTO-START ON WINDOWS BOOT / LOGIN
echo ============================================================
echo.
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\MT5_AlgoBot_AutoStart.vbs"

if exist "%SHORTCUT_PATH%" (
    del /F /Q "%SHORTCUT_PATH%" >nul
    echo [SUCCESS] Auto-start is now DISABLED!
    echo The bot will no longer start automatically when Windows boots.
) else (
    echo [INFO] Auto-start was not enabled.
)
echo.
pause
