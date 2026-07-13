@echo off
REM ===========================================================================
REM  Herbarium Pipeline - SECOND parallel instance (Windows)
REM
REM  Double-click this INSTEAD of start.bat when you want to work on a second
REM  project at the same time as the first (e.g. two families training on two
REM  cloud pods at once).
REM
REM    * start.bat        -> first  window, http://localhost:8765
REM    * start-second.bat -> second window, http://localhost:8766
REM
REM  Each window remembers its own project + settings independently (they use
REM  separate storage folders), so set a DIFFERENT "Project name" in each.
REM
REM  NOTE: each project runs its own cloud pod, so two at once means TWO pods
REM  billing at the same time. Close a window when you're done with it.
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM --- What makes this the "second" instance: a different port + its own store -
set "HERBARIUM_PORT=8766"
set "NICEGUI_STORAGE_PATH=%~dp0.nicegui-2"

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
echo === Herbarium Pipeline (second project) ===
echo Setting up environment ^(first run downloads Python + dependencies, ~150 MB^)...
"%UV%" sync
if errorlevel 1 (
    echo.
    echo Environment setup failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Launching a SECOND web UI - your browser will open at http://localhost:8766
echo Set a different Project name here than in the first window.
echo Close this window to quit this second project.
"%UV%" run python herbarium_pipeline_webui.py
pause
