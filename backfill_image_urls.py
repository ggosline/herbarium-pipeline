"""Populate image_url for existing specsin.csv rows that predate the new schema.

For each row whose gbifID is set but image_url is blank, query the GBIF
single-occurrence endpoint and pull the first StillImage URL out of the media
block. Writes back to the same CSV.

Usage:
  python backfill_image_urls.py --specsin /path/specsin.csv [--workers 16]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from download_gbif_images import get_image_urls

GBIF_OCCURRENCE_API = "https://api.gbif.org/v1/occurrence/{}"


def fetch_url(gbif_id: str, session: requests.Session, timeout: float = 15.0) -> str:
    try:
        r = session.get(GBIF_OCCURRENCE_API.format(gbif_id), timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return ""
    urls = get_image_urls(r.json())
    return urls[0] if urls else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--specsin", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    if not args.specsin.is_file():
        raise SystemExit(f"specsin not found: {args.specsin}")

    df = pd.read_csv(args.specsin)
    if "gbifID" not in df.columns:
        raise SystemExit("specsin has no gbifID column — nothing to backfill from")
    if "image_url" not in df.columns:
        df["image_url"] = ""

    needs = df[df["image_url"].fillna("").astype(str).str.strip().eq("") &
               df["gbifID"].fillna("").astype(str).str.strip().ne("")]
    n = len(needs)
    if n == 0:
        print("Nothing to backfill — all rows already have image_url.")
        return
    print(f"Backfilling {n:,} rows...")

    sess = requests.Session()
    found = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_url, str(int(float(g))), sess): idx
                for idx, g in zip(needs.index, needs["gbifID"])}
        for fut in as_completed(futs):
            idx = futs[fut]
            url = fut.result()
            if url:
                df.at[idx, "image_url"] = url
                found += 1
            else:
                failed += 1
            if (found + failed) % 100 == 0:
                print(f"  {found + failed:,}/{n:,}  found={found:,}  failed={failed:,}")

    df.to_csv(args.specsin, index=False)
    print(f"Done. Filled {found:,} URLs, {failed:,} not found → {args.specsin}")


if __name__ == "__main__":
    main()
