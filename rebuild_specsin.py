"""
Rebuild a specsin CSV from a GBIF DwC-A occurrence.txt download,
matched against the actual files in an images_cropped directory.

Usage:
  python rebuild_specsin.py \
      --occurrence  /mnt/d/Annonaceae/Annonaceae/GBIFAnnonaceaeLeiden/occurrence.txt \
      --images-dir  /mnt/d/Annonaceae/Annonaceae/imagesLeiden_cropped \
      --output      /mnt/d/Annonaceae/Annonaceae/AnnonaceaeLeiden_specsin.csv
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

SPECSIN_COLS = [
    "catalogNumber", "species", "verbatimName", "family", "genus",
    "institutionID", "institutionCode", "countryCode",
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
    "gbifID", "indet", "fname", "hasfile", "sparse", "outlier", "invalid",
]

# Column indices (1-based) found in occurrence.txt
# Resolved by header row at parse time.


def safe_name(s: str) -> str:
    return s.strip().replace(" ", "_").replace("/", "_")


def catalog_from_fname(filename: str) -> str:
    """Extract catalogNumber from image filename.

    Pattern: {verbatimName}_{catalogNumber}.{ext}
    The catalogNumber is splitext(filename.rsplit('_', 1)[-1])[0].
    """
    stem = os.path.splitext(filename)[0]   # strip extension
    # stem = e.g. "Afroguatteria_bequaertii_WAG.1587437"
    # catalogNumber is everything after the last underscore
    return stem.rsplit("_", 1)[-1]


def build_image_lookup(images_dir: Path) -> dict[str, str]:
    """Return {catalogNumber: filename} for every file in images_dir."""
    lookup: dict[str, str] = {}
    for fname in os.listdir(images_dir):
        cat = catalog_from_fname(fname)
        if cat:
            lookup[cat] = fname
    print(f"Indexed {len(lookup)} files in {images_dir}")
    return lookup


def parse_occurrence(occurrence_path: Path, image_lookup: dict[str, str]) -> dict[str, dict]:
    """Parse GBIF occurrence.txt and build specsin rows keyed by catalogNumber."""
    rows: dict[str, dict] = {}

    with open(occurrence_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for rec in reader:
            catalog = (rec.get("catalogNumber") or "").strip()
            if not catalog:
                continue

            species = (rec.get("species") or "").strip()
            genus   = (rec.get("genus")   or "").strip()
            family  = (rec.get("family")  or "").strip()
            indet   = not bool(species)
            verbatim_name = species if species else (f"{genus}_indet" if genus else f"{family}_indet")

            fname   = image_lookup.get(catalog, "")
            hasfile = bool(fname)

            if not fname:
                # Build expected fname even if file is absent
                fname = f"{safe_name(verbatim_name)}_{safe_name(catalog)}.jpg"

            rows[catalog] = {
                "catalogNumber": catalog,
                "species":       species,
                "verbatimName":  safe_name(verbatim_name),
                "family":        family,
                "genus":         genus,
                "institutionID": rec.get("institutionID", ""),
                "institutionCode": rec.get("institutionCode", ""),
                "countryCode":   rec.get("countryCode", ""),
                "decimalLatitude":  rec.get("decimalLatitude", ""),
                "decimalLongitude": rec.get("decimalLongitude", ""),
                "coordinateUncertaintyInMeters": rec.get("coordinateUncertaintyInMeters", ""),
                "gbifID":  rec.get("gbifID", ""),
                "indet":   indet,
                "fname":   fname,
                "hasfile": hasfile,
                "sparse":  False,   # filled in during save
                "outlier": False,
                "invalid": False,
            }

    print(f"Parsed {len(rows)} occurrence records")
    return rows


def save_specsin(path: Path, rows: dict[str, dict]) -> None:
    """Compute sparse flag and write specsin CSV."""
    species_counts: Counter = Counter(
        r["species"]
        for r in rows.values()
        if r.get("hasfile") and r.get("species")
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPECSIN_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            sp = row.get("species", "")
            row["sparse"] = species_counts.get(sp, 0) < 5
            writer.writerow(row)

    n_species = sum(1 for cnt in species_counts.values() if cnt >= 5)
    n_hasfile = sum(1 for r in rows.values() if r.get("hasfile"))
    print(f"Saved {len(rows)} records ({n_hasfile} with files, {n_species} non-sparse species) → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild specsin CSV from GBIF DwC-A occurrence.txt")
    parser.add_argument("--occurrence", type=Path, required=True,
                        help="Path to GBIF occurrence.txt (tab-delimited DwC-A)")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory of cropped images to match against")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output specsin CSV path")
    args = parser.parse_args()

    image_lookup = build_image_lookup(args.images_dir)
    rows = parse_occurrence(args.occurrence, image_lookup)

    unmatched = sum(1 for r in rows.values() if not r["hasfile"])
    extra_files = set(image_lookup.values()) - {r["fname"] for r in rows.values() if r["hasfile"]}
    print(f"Matched: {len(rows) - unmatched}  |  Unmatched occurrences: {unmatched}  |  Extra files (no occurrence): {len(extra_files)}")
    if extra_files:
        print("  Extra files sample:", sorted(extra_files)[:5])

    save_specsin(args.output, rows)


if __name__ == "__main__":
    main()
