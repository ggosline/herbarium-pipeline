# Running the Herbarium Pipeline on any machine

The web UI is portable: it needs **no Python, conda, or admin rights** on the
target machine. [uv](https://docs.astral.sh/uv/) manages its own Python and
builds the environment on first launch.

## For end users

1. Unzip `herbarium-pipeline-portable.zip` anywhere (e.g. your Desktop).
2. **Windows:** double-click `start.bat`.
   **macOS / Linux:** run `./start.sh` in a terminal.
3. First launch downloads a private Python and the slim dependencies
   (~150 MB, ~1 minute). Later launches are instant.
4. Your browser opens at <http://localhost:8765>.

The slim install runs the whole **cloud** pipeline (train / identify / publish
on a RunPod pod). To identify specimens **on your own machine** (CPU is fine),
click **Enable offline AI features** on the Get Started tab — that adds the
heavier ML stack (`uv sync --extra local-ml`). You only need it once.

If uv isn't already installed and isn't bundled in the zip, the launcher
installs it automatically on first run (needs internet that once).

## For whoever builds the zip

```bash
# Code-only bundle (uv self-installs on the target's first run):
python make_portable.py

# Fully offline Windows bundle (also carries uv.exe):
python make_portable.py --with-uv

# Custom output path:
python make_portable.py --out dist/herbarium-2026-07.zip
```

The bundle carries the code, `pyproject.toml`, and `uv.lock` — the *recipe* for
the environment, not a prebuilt one. It excludes `.venv`, `.git`, caches, and
any downloaded data/images. The exact dependency versions come from `uv.lock`,
so every user gets the same set.
