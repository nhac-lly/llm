@echo off
REM Setup script for ONNX Runtime GenAI Agent on Windows
REM Double-click this file to run setup

echo ============================================================
echo ONNX Runtime GenAI Agent - Setup
echo ============================================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: PowerShell not found!
    echo Please install PowerShell or run setup.ps1 manually.
    pause
    exit /b 1
)

REM Run the PowerShell setup script
echo Running PowerShell setup script...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Setup completed successfully!
    echo ============================================================
    echo.
    echo To activate the virtual environment, run:
    echo   .venv\Scripts\activate.bat
    echo.
) else (
    echo.
    echo ============================================================
    echo Setup failed!
    echo ============================================================
    echo.
    echo Please check the error messages above.
    echo.
)

pause

