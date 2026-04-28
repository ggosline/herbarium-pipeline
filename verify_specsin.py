"""Reconcile specsin.csv against an actual image directory.

Sets hasfile=False for any row whose fname is not present in --image-dir.
Optionally also sets hasfile=True if a previously-missing file has reappeared.

Usage:
  python verify_specsin.py --specsin /path/specsin.csv --image-dir /path/images_640
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def reconcile(specsin: Path, image_dir: Path, restore: bool = False) -> tuple[int, int]:
    df = pd.read_csv(specsin)
    if "hasfile" not in df.columns or "fname" not in df.columns:
        raise SystemExit(f"{specsin} missing required columns (fname, hasfile)")

    exists = df["fname"].apply(lambda f: (image_dir / str(f)).is_file())
    had = df["hasfile"].astype(str).str.lower().isin(("true", "1"))

    # Rows claiming hasfile=True but file is gone → set False.
    cleared = had & ~exists
    n_cleared = int(cleared.sum())
    df.loc[cleared, "hasfile"] = False

    n_restored = 0
    if restore:
        # Rows claiming hasfile=False but file is now present → set True.
        restored_mask = ~had & exists
        n_restored = int(restored_mask.sum())
        df.loc[restored_mask, "hasfile"] = True

    df.to_csv(specsin, index=False)
    return n_cleared, n_restored


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--specsin", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--restore", action="store_true",
                   help="Also flip hasfile back to True for rows whose file has reappeared")
    args = p.parse_args()

    if not args.specsin.is_file():
        raise SystemExit(f"specsin not found: {args.specsin}")
    if not args.image_dir.is_dir():
        raise SystemExit(f"image dir not found: {args.image_dir}")

    n_cleared, n_restored = reconcile(args.specsin, args.image_dir, restore=args.restore)
    print(f"  Cleared hasfile for {n_cleared} missing file(s) → {args.specsin}")
    if args.restore:
        print(f"  Restored hasfile for {n_restored} reappeared file(s)")


if __name__ == "__main__":
    main()
