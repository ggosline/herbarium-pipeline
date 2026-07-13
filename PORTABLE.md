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

## Working on two projects at once

If you want to run a second project **at the same time** as the first (for
example, two plant families training on two cloud pods in parallel):

1. Keep your first project running — that's the `start.bat` window at
   <http://localhost:8765>.
2. Double-click **`start-second.bat`**. A second window opens at
   <http://localhost:8766>.
3. In that second window, type a **different Project name** than the first, then
   use it normally.

The two windows remember their projects and settings separately, so they never
overwrite each other. Close either window when you're done with that project.

> **Heads-up on cost:** each project runs its own cloud pod, so two projects
> open at once means **two pods billing at the same time**. Close a window to
> stop paying for that one. Want a third project in parallel? Copy
> `start-second.bat`, and in the copy change `8766` to `8767` and `.nicegui-2`
> to `.nicegui-3`.

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
