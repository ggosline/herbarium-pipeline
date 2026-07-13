#!/usr/bin/env python3
"""Build a portable, self-contained zip of the herbarium web UI.

The zip carries the code + lockfile + launchers. On the target machine, the
user unzips and runs start.bat (Windows) / start.sh (Unix); uv fetches its own
Python and builds the slim venv on first launch - no conda or system Python
required. Heavy ML deps are opt-in (the "Enable offline AI features" button).

Usage:
    python make_portable.py                 # code-only zip (uv self-installs)
    python make_portable.py --with-uv       # also bundle uv.exe for Windows
                                            #   (fully offline uv on the target)
    python make_portable.py --out dist/foo.zip

The zip deliberately excludes the local .venv, .git, caches, and any downloaded
data/images - it's the recipe, not a prebuilt environment.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

_HERE = Path(__file__).parent.resolve()

# Top-level files always included (launchers + project metadata + docs).
_ROOT_FILES = [
    "pyproject.toml",
    "uv.lock",
    "start.bat",
    "start-second.bat",
    "start.sh",
    "README.md",
    "cloud_setup.md",
    "PORTABLE.md",
]

# Directories copied wholesale (minus the exclude patterns below).
_PKG_DIRS = ["cloud", "webui", "tools"]

# All root-level pipeline scripts (download/filter/resize/train/identify/etc.)
# are picked up by the "*.py" glob, minus this build script itself.
_EXCLUDE = [
    "*/__pycache__/*", "__pycache__/*", "*.pyc",
    ".venv/*", ".git/*", ".nicegui/*", "dist/*",
    "make_portable.py",
]

# uv standalone release asset for Windows x64 (a zip containing uv.exe).
_UV_WIN_URL = ("https://github.com/astral-sh/uv/releases/latest/download/"
               "uv-x86_64-pc-windows-msvc.zip")


def _excluded(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, pat) for pat in _EXCLUDE)


def _add_file(zf: zipfile.ZipFile, path: Path, arc: str) -> None:
    if _excluded(arc):
        return
    zf.write(path, arc)


def _fetch_uv_exe() -> bytes:
    print(f"Downloading uv (Windows x64) from {_UV_WIN_URL} ...")
    with urllib.request.urlopen(_UV_WIN_URL, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith("uv.exe"))
        return z.read(name)


def build(out: Path, with_uv: bool) -> int:
    files: list[tuple[Path, str]] = []

    # Root-level scripts + curated files.
    for p in sorted(_HERE.glob("*.py")):
        files.append((p, p.name))
    for name in _ROOT_FILES:
        p = _HERE / name
        if p.is_file():
            files.append((p, name))
        else:
            print(f"  (skip missing {name})")

    # Packages.
    for d in _PKG_DIRS:
        base = _HERE / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                files.append((p, str(p.relative_to(_HERE))))

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, arc in files:
            before = len(zf.namelist())
            _add_file(zf, path, arc)
            n += len(zf.namelist()) - before
        if with_uv:
            zf.writestr("uv.exe", _fetch_uv_exe())
            n += 1
            print("  + bundled uv.exe (offline uv on Windows targets)")

    size_mb = out.stat().st_size / (1 << 20)
    print(f"[done] Wrote {out}  ({n} files, {size_mb:.1f} MB)")
    if not with_uv:
        print("  (no uv bundled - start.bat/start.sh will install uv on first "
              "run; pass --with-uv for a fully offline Windows bundle.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=_HERE / "dist" / "herbarium-pipeline-portable.zip",
                    help="Output zip path.")
    ap.add_argument("--with-uv", action="store_true",
                    help="Download and bundle uv.exe for Windows (offline uv).")
    args = ap.parse_args()
    return build(args.out.resolve(), args.with_uv)


if __name__ == "__main__":
    sys.exit(main())
