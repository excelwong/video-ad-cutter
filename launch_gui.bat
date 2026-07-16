@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
set "FF=%~dp0ffmpeg\bin"

if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo Please create it, for example:
    echo   "%~dp0venv\Scripts\python.exe" -m venv "%~dp0venv"
    pause
    exit /b 1
)

set "PATH=%FF%;%PATH%"

echo ==================================================
echo   Video Ad Cutter
echo   Python : %PY%
echo   ffmpeg : %FF%
echo ==================================================
"%PY%" gui.py
set "RC=%errorlevel%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] exited with code %RC%. See crash.log / error popup.
    echo.
    pause
)
endlocal
