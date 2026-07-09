@echo off
REM ===========================================================================
REM  Herbarium Pipeline - portable launcher (Windows)
REM
REM  Unzip anywhere and double-click. No Python, conda, or admin rights needed:
REM  uv manages its own Python and builds the (slim) environment on first run.
REM  Subsequent launches are instant. To enable local AI features (Quick ID /
REM  local Identify), use the "Enable offline AI features" button in the app,
REM  or run:  uv sync --extra local-ml
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM --- Locate uv: bundled next to this script, else on PATH, else install it --
set "UV="
if exist "%~dp0uv.exe" set "UV=%~dp0uv.exe"
if not defined UV (
    where uv >nul 2>nul && set "UV=uv"
)
if not defined UV (
    echo Installing uv ^(one-time, ~15 MB^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if not defined UV (
    echo.
    echo Could not find or install uv automatically.
    echo Install it from https://docs.astral.sh/uv/ then re-run this launcher.
    pause
    exit /b 1
)

echo.
echo === Herbarium Pipeline ===
echo Setting up environment ^(first run downloads Python + dependencies, ~150 MB^)...
"%UV%" sync
if errorlevel 1 (
    echo.
    echo Environment setup failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Launching the web UI - your browser will open at http://localhost:8765
echo Close this window to quit.
"%UV%" run python herbarium_pipeline_webui.py
pause
