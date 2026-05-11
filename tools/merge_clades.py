"""
Merge two (or more) per-clade specsin CSVs into a single combined specsin
and (optionally) clone the cloud-state file so a new project name reuses
the existing RunPod network volume.

Why: incrementally extending a trained dataset (e.g. adding Liliopsida to
an existing Magnoliopsida project) without re-downloading, re-filtering,
or re-cropping the original clade.

Usage:
  python tools/merge_clades.py \
      --specsin /media/ggosline/linuxdata/Magnoliopsida/specsin.csv \
      --specsin /media/ggosline/linuxdata/Liliopsida/specsin.csv    \
      --output  /media/ggosline/linuxdata/Angiosperm-families_Africa/specsin.csv \
      --project Angiosperm-families_Africa \
      --clone-from-project Magnoliopsida

If --clone-from-project is omitted, only the merged specsin is written —
no cloud-state files are touched. With it, ~/.herbarium-cloud/<source>.json
is copied to ~/.herbarium-cloud/<project>.json with volume_id +
data_center_id preserved and run-specific fields reset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


CLOUD_STATE_DIR = Path.home() / ".herbarium-cloud"


def merge_specsins(paths: list[Path], output: Path) -> None:
    frames: list[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            sys.exit(f"specsin not found: {p}")
        df = pd.read_csv(p)
        print(f"  {p}: {len(df):,} rows, {df.shape[1]} cols")
        frames.append(df)

    # Column-set sanity check — surfaces schema drift between source CSVs
    # that would otherwise silently produce all-NaN columns after concat.
    common = set.intersection(*(set(f.columns) for f in frames))
    only: list[set[str]] = []
    for f in frames:
        only.append(set(f.columns) - common)
    if any(only):
        print("  ⚠ column schema differs across inputs:")
        for path, extras in zip(paths, only):
            if extras:
                print(f"      {path.name} has extra columns: {sorted(extras)}")
        print("  → continuing with concat(sort=False); missing values become NaN")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    key = "gbifID" if "gbifID" in combined.columns else "fname"
    before = len(combined)
    combined = combined.drop_duplicates(subset=[key], keep="first")
    print(f"  combined: {before:,} → after dedupe on '{key}': {len(combined):,}")

    if "fname" in combined.columns:
        n_dup_fname = combined["fname"].duplicated().sum()
        if n_dup_fname:
            print(f"  ⚠ {n_dup_fname:,} duplicate fname values remain — these "
                  f"would collide on the pod's images/ dir. Investigate before "
                  f"running download.")

    if "family" in combined.columns and "species" in combined.columns:
        print(f"  families: {combined['family'].nunique():,}  "
              f"species: {combined['species'].nunique():,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"✓ wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def clone_cloud_state(source_project: str, new_project: str) -> None:
    src = CLOUD_STATE_DIR / f"{source_project}.json"
    dst = CLOUD_STATE_DIR / f"{new_project}.json"
    if not src.exists():
        sys.exit(f"source state file not found: {src}")
    if dst.exists():
        sys.exit(f"target state file already exists (refusing to overwrite): "
                 f"{dst}\n  Delete it manually if you really want to redo this.")
    with src.open() as f:
        state = json.load(f)
    # Preserve volume_id + data_center_id so the new project attaches to the
    # same RunPod network volume — that's the whole point of cloning.
    # Reset run-specific state so the new project doesn't think it has a
    # running pod inherited from the source.
    state["project"] = new_project
    for k in ("pod_id", "ssh_host", "ssh_port", "pod_started_at",
              "pod_hourly_rate", "current_step"):
        state[k] = None if k != "current_step" else ""
    state["completed_steps"] = []
    state["accumulated_cost_usd"] = 0.0
    with dst.open("w") as f:
        json.dump(state, f, indent=2)
    print(f"✓ cloned cloud state {src.name} → {dst.name}")
    print(f"  volume_id={state.get('volume_id')} "
          f"data_center_id={state.get('data_center_id')}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--specsin", action="append", required=True, type=Path,
                   help="Path to a per-clade specsin CSV. Repeat for each input.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the combined specsin CSV.")
    p.add_argument("--project",
                   help="New cloud project name (only needed with "
                        "--clone-from-project).")
    p.add_argument("--clone-from-project",
                   help="Existing cloud project whose state file (volume_id, "
                        "data_center_id) should seed the new project's state. "
                        "Skipped if not provided.")
    args = p.parse_args()

    if len(args.specsin) < 2:
        sys.exit("Need at least two --specsin inputs to merge.")

    merge_specsins(args.specsin, args.output.resolve())

    if args.clone_from_project:
        if not args.project:
            sys.exit("--clone-from-project requires --project to name the "
                     "new state file.")
        clone_cloud_state(args.clone_from_project, args.project)
    elif args.project:
        print("(--project given without --clone-from-project — "
              "no cloud-state file written.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
