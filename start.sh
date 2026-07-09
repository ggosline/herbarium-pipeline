#!/usr/bin/env bash
# ===========================================================================
#  Herbarium Pipeline - portable launcher (macOS / Linux)
#
#  Unzip anywhere and run ./start.sh . No system Python or admin rights needed:
#  uv manages its own Python and builds the (slim) environment on first run.
#  To enable local AI features (Quick ID / local Identify), use the app's
#  "Enable offline AI features" button, or run:  uv sync --extra local-ml
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Locate uv: bundled next to this script, else on PATH, else install it.
if [ -x "./uv" ]; then
    UV="./uv"
elif command -v uv >/dev/null 2>&1; then
    UV="uv"
else
    echo "Installing uv (one-time)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi

echo
echo "=== Herbarium Pipeline ==="
echo "Setting up environment (first run downloads Python + dependencies, ~150 MB)..."
"$UV" sync

echo
echo "Launching the web UI - your browser will open at http://localhost:8765"
echo "Press Ctrl+C to quit."
exec "$UV" run python herbarium_pipeline_webui.py
