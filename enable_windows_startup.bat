@echo off
title Enable Auto-Start on Windows Boot
color 0A
echo ============================================================
echo      ENABLE AUTO-START ON WINDOWS BOOT / LOGIN
echo ============================================================
echo.
set "TARGET_DIR=%~dp0"
set "VBS_PATH=%TARGET_DIR%start_background.vbs"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\MT5_AlgoBot_AutoStart.vbs"

echo Creating automatic startup launcher...
copy /Y "%VBS_PATH%" "%SHORTCUT_PATH%" >nul

if exist "%SHORTCUT_PATH%" (
    echo.
    echo [SUCCESS] Auto-start is now ENABLED!
    echo Whenever your computer boots up and you log into Windows:
    echo - The bot will automatically start in the background.
    echo - Your browser will open http://localhost:8501 automatically.
) else (
    echo.
    echo [ERROR] Could not write to Windows Startup directory.
)
echo.
pause
