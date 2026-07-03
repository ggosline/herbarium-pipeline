@echo off
REM ---------------------------------------------------------------------------
REM Start the Herbarium Pipeline web UI.
REM Opens automatically at http://localhost:8765
REM
REM Uses uv (see CLAUDE.md): `uv run` resolves/syncs the project's .venv from
REM pyproject.toml + uv.lock and runs inside it -- no manual activation needed.
REM
REM Runs from this script's own folder so NiceGUI state
REM (.nicegui\storage-general.json) lands next to the project as expected.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: "uv" was not found on PATH.
    echo   Install it from https://docs.astral.sh/uv/  then re-run this script.
    echo.
    pause
    exit /b 1
)

uv run python herbarium_pipeline_webui.py

REM Keep the window open after the server stops (or if it fails to start)
REM so any error message stays visible.
pause
