"""
Download GBIF herbarium images for a family or genus, optionally filtered by continent
and/or country code.

Builds and updates a specsin-compatible CSV metadata file with coordinates.
Skips images already downloaded; verifies and updates species names from GBIF.

Usage:
  python download_gbif_images.py --family Ebenaceae --continent AFRICA
  python download_gbif_images.py --genus Diospyros --continent ASIA
  python download_gbif_images.py --family Ebenaceae --continent AFRICA --exclude-countries MG
  python download_gbif_images.py --family Ebenaceae --countries ZA NG TZ KE
  python download_gbif_images.py --family Ebenaceae --output-dir /path/to/project/images --specsin /path/to/project/specsin.csv

GBIF continent codes:
  AFRICA, ASIA, EUROPE, NORTH_AMERICA, SOUTH_AMERICA, OCEANIA, ANTARCTICA

Country codes are ISO 3166-1 alpha-2 (e.g. MG=Madagascar, ZA=South Africa, NG=Nigeria).
--countries and --exclude-countries are mutually exclusive.
"""

import csv
import io
import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
import urllib.parse
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

GBIF_SEARCH_API = "https://api.gbif.org/v1/occurrence/search"
HEADERS = {"User-Agent": "HerbariumImageDownloader/1.0 (research)"}
PAGE_SIZE = 300  # GBIF max per request
GBIF_MAX_OFFSET = 100_000  # GBIF hard cap on paged results

SPECSIN_COLS = [
    "catalogNumber", "species", "verbatimName", "family", "genus",
    "institutionID", "institutionCode", "countryCode",
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
    "gbifID", "indet", "fname", "hasfile", "sparse", "outlier", "invalid",
]

VALID_CONTINENTS = {
    "AFRICA", "ASIA", "EUROPE", "NORTH_AMERICA",
    "SOUTH_AMERICA", "OCEANIA", "ANTARCTICA",
}


# ---------------------------------------------------------------------------
# GBIF helpers
# ---------------------------------------------------------------------------

def gbif_get(url: str, retries: int = 3) -> dict:
    """HTTP GET a GBIF API URL, with retries on transient errors."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"GBIF request failed: {url}") from exc


def search_occurrences(
    family: str | None,
    genus: str | None,
    continent: str | None,
    countries: list[str] | None = None,
    exclude_countries: list[str] | None = None,
) -> list[dict]:
    """
    Page through the GBIF occurrence search API and return all records that
    have at least one StillImage.  Results come directly from the search
    response (no per-record API calls needed for media URLs).

    country filtering is done client-side so that --exclude-countries works
    cleanly alongside a continent filter.
    """
    params: dict = {"mediaType": "StillImage", "basisOfRecord": "PRESERVED_SPECIMEN",
                    "limit": PAGE_SIZE, "offset": 0}
    if family:
        params["family"] = family
    if genus:
        params["genus"] = genus
    if continent:
        params["continent"] = continent

    taxon_label = family or genus
    cont_label = continent or "all continents"
    print(f"Querying GBIF: {taxon_label} / {cont_label} (mediaType=StillImage)...")

    include_set = {c.upper() for c in countries} if countries else None
    exclude_set = {c.upper() for c in exclude_countries} if exclude_countries else None
    if include_set:
        print(f"  Keeping countries: {', '.join(sorted(include_set))}")
    if exclude_set:
        print(f"  Excluding countries: {', '.join(sorted(exclude_set))}")

    all_results: list[dict] = []
    while True:
        url = GBIF_SEARCH_API + "?" + urllib.parse.urlencode(params)
        data = gbif_get(url)
        results = data.get("results", [])

        for r in results:
            cc = (r.get("countryCode") or "").upper()
            if include_set is not None and cc not in include_set:
                continue
            if exclude_set is not None and cc in exclude_set:
                continue
            all_results.append(r)

        total = data.get("count", 0)
        print(f"  Fetched page offset={params['offset']} — kept {len(all_results)} so far...",
              end="\r", flush=True)

        if data.get("endOfRecords", True):
            break
        params["offset"] += PAGE_SIZE
        if params["offset"] >= GBIF_MAX_OFFSET:
            print(f"\n  WARNING: Hit GBIF {GBIF_MAX_OFFSET:,} record limit — results truncated.")
            break

    print(f"\n  Found {len(all_results)} occurrence records with images after country filter.")
    return all_results


# ---------------------------------------------------------------------------
# DwC-A ZIP loader  (alternative to search API)
# ---------------------------------------------------------------------------

def load_dwca(
    dwca_path: Path,
    family: str | None = None,
    genus: str | None = None,
    continent: str | None = None,
    countries: list[str] | None = None,
    exclude_countries: list[str] | None = None,
) -> list[dict]:
    """
    Parse a locally saved GBIF Darwin Core Archive ZIP and return records in
    the same format as search_occurrences(), so the rest of the pipeline is
    unchanged.

    The ZIP must contain:
      occurrence.txt  — tab-separated occurrences (standard GBIF download)
      multimedia.txt  — tab-separated image URLs (type / identifier / gbifID)

    Filters (family, genus, continent, countries, exclude_countries) are all
    applied client-side, so you can use a broad DwC-A (e.g. whole family) and
    still filter to a subset here.
    """
    include_set  = {c.upper() for c in countries}         if countries         else None
    exclude_set  = {c.upper() for c in exclude_countries} if exclude_countries else None

    print(f"Reading DwC-A: {dwca_path}")

    with zipfile.ZipFile(dwca_path) as zf:
        names = zf.namelist()

        # Locate files — GBIF names them consistently but allow variations
        def find(keyword):
            return next((n for n in names if keyword in n.lower()), None)

        occ_file   = find("occurrence")
        media_file = find("multimedia")

        if not occ_file:
            raise FileNotFoundError(f"No occurrence file found in {dwca_path}. Contents: {names}")

        # Build gbifID → [image_url, ...] from multimedia.txt
        media_map: dict[str, list[str]] = {}
        if media_file:
            print(f"  Parsing {media_file}...")
            with zf.open(media_file) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                for row in reader:
                    if row.get("type") != "StillImage":
                        continue
                    url = row.get("identifier", "").strip()
                    if not url:
                        continue
                    gid = (row.get("gbifID") or row.get("coreid") or row.get("id") or "").strip()
                    if gid:
                        media_map.setdefault(gid, []).append(url)
            print(f"  {len(media_map):,} occurrences have StillImage media")
        else:
            print("  WARNING: no multimedia.txt found — no image URLs available")

        # Parse occurrences and apply filters
        print(f"  Parsing {occ_file}...")
        records: list[dict] = []
        skipped = 0
        with zf.open(occ_file) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
            for row in reader:
                gid = row.get("gbifID", "").strip()
                urls = media_map.get(gid, [])
                if not urls:
                    skipped += 1
                    continue  # no image — skip

                # Taxon filter
                if family and row.get("family", "").lower() != family.lower():
                    continue
                if genus and row.get("genus", "").lower() != genus.lower():
                    continue

                # Basis of record filter
                if row.get("basisOfRecord") != "PRESERVED_SPECIMEN":
                    continue

                # Geography filters
                cc = (row.get("countryCode") or "").upper()
                if continent and row.get("continent", "").upper().replace(" ", "_") != continent.upper():
                    continue
                if include_set is not None and cc not in include_set:
                    continue
                if exclude_set is not None and cc in exclude_set:
                    continue

                records.append({
                    "key":             gid,
                    "species":         row.get("species", ""),
                    "genus":           row.get("genus", ""),
                    "family":          row.get("family", ""),
                    "catalogNumber":   row.get("catalogNumber", ""),
                    "countryCode":     row.get("countryCode", ""),
                    "continent":       row.get("continent", ""),
                    "decimalLatitude":  row.get("decimalLatitude", ""),
                    "decimalLongitude": row.get("decimalLongitude", ""),
                    "coordinateUncertaintyInMeters": row.get("coordinateUncertaintyInMeters", ""),
                    "publishingOrgKey": row.get("publishingOrgKey", ""),
                    "institutionCode":  row.get("institutionCode", ""),
                    "media": [{"type": "StillImage", "identifier": u} for u in urls],
                })

    print(f"  Loaded {len(records):,} records with images "
          f"({skipped:,} skipped — no media).")
    return records


# ---------------------------------------------------------------------------
# Filename / row helpers
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    """Return a filesystem-safe version of s (spaces → underscores)."""
    return s.strip().replace(" ", "_").replace("/", "_")


def make_fname(family: str, verbatim_name: str, catalog: str, suffix: str = "") -> str:
    """Build the canonical image filename matching the existing naming convention."""
    return f"{safe_name(family)}_{safe_name(verbatim_name)}_{safe_name(catalog)}{suffix}.jpg"


def get_image_urls(record: dict) -> list[str]:
    """Extract StillImage URLs from a GBIF occurrence record's media list."""
    return [
        m["identifier"]
        for m in record.get("media", [])
        if m.get("type") == "StillImage" and m.get("identifier")
    ]


def record_to_row(record: dict, fname: str, hasfile: bool) -> dict:
    """Convert a GBIF occurrence dict to a specsin-compatible row dict."""
    species = (record.get("species") or "").strip()
    genus   = (record.get("genus")   or "").strip()
    family  = (record.get("family")  or "").strip()
    indet   = not bool(species)
    verbatim_name = species if species else f"{genus}_indet"

    return {
        "catalogNumber": record.get("catalogNumber", ""),
        "species":        species,
        "verbatimName":   safe_name(verbatim_name),
        "family":         family,
        "genus":          genus,
        "institutionID":  record.get("publishingOrgKey", ""),
        "institutionCode": record.get("institutionCode", ""),
        "countryCode":    record.get("countryCode", ""),
        "decimalLatitude":  record.get("decimalLatitude", ""),
        "decimalLongitude": record.get("decimalLongitude", ""),
        "coordinateUncertaintyInMeters": record.get("coordinateUncertaintyInMeters", ""),
        "gbifID":   str(record.get("key", "")),
        "indet":    indet,
        "fname":    fname,
        "hasfile":  hasfile,
        "sparse":   False,   # computed after all records are processed
        "outlier":  False,
        "invalid":  False,
    }


# ---------------------------------------------------------------------------
# IIIF resolution upgrade
# ---------------------------------------------------------------------------

# Matches a standard IIIF Image API v2/v3 request URL.
# Structure after the identifier: /{region}/{size}/{rotation}/{quality}.{format}
# The base+identifier group is lazy so it stops at the first plausible region.
_IIIF_PARAMS_RE = re.compile(
    r'^(https?://.+?)'                                       # base URL + identifier (lazy)
    r'/(full|square|\d+,\d+,\d+,\d+|pct:[\d.,]+)'          # region
    r'/(max|full|\^?!?\d+,\d*|\^?!?\d*,\d+|pct:[\d.]+)'   # size
    r'/(!?\d+(?:\.\d+)?)'                                   # rotation
    r'/(default|color|gr[ae]y|bitonal)'                     # quality
    r'\.(jpe?g|tiff?|png|gif|jp2|webp)$',                  # format
    re.IGNORECASE,
)


def _iiif_upgrade(url: str, iiif_size: str) -> list[str]:
    """
    If url is an IIIF Image API request, return candidate URLs to try for the
    requested size, in order of preference, followed by the original URL as
    fallback.  Returns [] if the URL doesn't look like an IIIF image request.

    iiif_size: "max" | "full" | a pixel count like "2048" (→ "!2048,2048")
    """
    low = url.lower()
    if "/iiif/" not in low and "/iiif2/" not in low and "/iiif3/" not in low:
        return []
    m = _IIIF_PARAMS_RE.match(url)
    if not m:
        return []
    base, region, size, rotation, quality, ext = m.groups()

    # Resolve requested size string to IIIF size tokens
    if iiif_size.lower() in ("max", "full"):
        target_sizes = ["max", "full"]   # try v3 then v2
    else:
        px = iiif_size.strip().rstrip("px")
        target_sizes = [f"!{px},{px}", "max", "full"]

    candidates = []
    for s in target_sizes:
        candidate = f"{base}/{region}/{s}/{rotation}/{quality}.{ext}"
        if candidate != url and candidate not in candidates:
            candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str, retries: int = 3) -> bytes | None:
    """Fetch URL, retrying on transient errors. Returns None on any failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 403, 404, 410, 501):
                return None  # permanent — don't retry
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
        except (urllib.error.URLError, OSError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def download_image(img_url: str, dest: Path, max_size: int | None = None,
                   iiif_size: str | None = None, retries: int = 3) -> bool:
    """
    Download a single image to dest, optionally resizing so the longer side <= max_size.
    If iiif_size is set (e.g. "2048" or "max"), attempts IIIF size-upgrade variants
    before falling back to the original URL. Returns True on success. Skips if exists.
    """
    if dest.exists():
        return True

    candidates = (_iiif_upgrade(img_url, iiif_size) if iiif_size else []) + [img_url]

    data = None
    used_url = img_url
    for url in candidates:
        data = _fetch_bytes(url, retries)
        if data is not None:
            used_url = url
            break

    if data is None:
        print(f"  FAILED {dest.name}")
        return False

    try:
        if max_size:
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)
            img.save(dest, format="JPEG", quality=90)
        else:
            dest.write_bytes(data)
        if iiif_size and used_url != img_url:
            print(f"  [IIIF↑] {dest.name}")
        return True
    except Exception as exc:
        print(f"  FAILED {dest.name}: {exc}")
        return False


def process_record(record: dict, out_dir: Path, max_size: int | None = None,
                   iiif_size: str | None = None) -> tuple[dict, str, int, int]:
    """
    Download images for one GBIF occurrence record.

    Returns:
        (specsin_row, catalog_number, n_newly_downloaded, n_failed)

    Images that already exist on disk are counted as present (not re-downloaded).
    The returned specsin_row always reflects current GBIF species data.
    """
    catalog = record.get("catalogNumber", "")
    species = (record.get("species") or "").strip()
    genus   = (record.get("genus")   or "").strip()
    family  = (record.get("family")  or "").strip()
    verbatim_name = species if species else f"{genus}_indet"

    urls = get_image_urls(record)
    downloaded = failed = 0
    present_fnames: list[str] = []

    for i, url in enumerate(urls):
        suffix = f"_{i}" if len(urls) > 1 else ""
        fname  = make_fname(family, verbatim_name, catalog, suffix)
        dest   = out_dir / fname

        already_existed = dest.exists()
        ok = download_image(url, dest, max_size=max_size, iiif_size=iiif_size)

        if ok:
            present_fnames.append(fname)
            if not already_existed:
                downloaded += 1
        else:
            failed += 1

    if not urls:
        failed = 1

    primary_fname = present_fnames[0] if present_fnames else make_fname(family, verbatim_name, catalog)
    row = record_to_row(record, primary_fname, bool(present_fnames))
    return row, catalog, downloaded, failed


# ---------------------------------------------------------------------------
# specsin CSV I/O
# ---------------------------------------------------------------------------

def load_specsin(path: Path) -> dict[str, dict]:
    """Load an existing specsin CSV as {catalogNumber: row_dict}."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["catalogNumber"]: row for row in reader}


def save_specsin(path: Path, rows: dict[str, dict]) -> None:
    """Compute sparse flag and write specsin CSV."""
    # sparse: species with fewer than 5 images on disk
    species_counts: Counter = Counter(
        r["species"]
        for r in rows.values()
        if str(r.get("hasfile", "")).lower() in ("true", "1") and r.get("species")
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPECSIN_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            sp = row.get("species", "")
            row["sparse"] = species_counts.get(sp, 0) < 5
            writer.writerow(row)

    n_species = sum(1 for sp, cnt in species_counts.items() if cnt >= 5)
    print(f"Saved {len(rows)} records ({n_species} non-sparse species) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GBIF herbarium images and build a specsin CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dwca", type=Path, default=None, metavar="ZIP",
                        help="Path to a locally saved GBIF DwC-A ZIP file. "
                             "When provided, skips the API search entirely.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--family", metavar="FAMILY",
                       help="Plant family to query or filter (e.g. Ebenaceae)")
    group.add_argument("--genus",  metavar="GENUS",
                       help="Plant genus to query or filter (e.g. Diospyros)")

    parser.add_argument("--continent", metavar="CONTINENT", default=None,
                        help=f"GBIF continent code: {', '.join(sorted(VALID_CONTINENTS))}")

    country_group = parser.add_mutually_exclusive_group()
    country_group.add_argument("--countries", nargs="+", metavar="CC",
                               help="Only include these ISO country codes (e.g. ZA NG TZ)")
    country_group.add_argument("--exclude-countries", nargs="+", metavar="CC",
                               help="Exclude these ISO country codes (e.g. MG to skip Madagascar)")

    parser.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                        help="Directory for downloaded images (default: ./<taxon>_images/)")
    parser.add_argument("--specsin", type=Path, default=None, metavar="CSV",
                        help="specsin CSV to create/update (default: ./<taxon>_specsin.csv)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    parser.add_argument("--max-size", type=int, default=0, metavar="PX",
                        help="Resize so the longer side <= PX using PIL (default: disabled). "
                             "Prefer a separate DALI post-processing pass for bulk resizing.")
    parser.add_argument("--iiif-size", default=None, metavar="PX_OR_MAX",
                        help="For IIIF image URLs, request this size instead of the default "
                             "thumbnail. Use a pixel count (e.g. 2048 → !2048,2048 fit-box) "
                             "or 'max' for full resolution. Falls back to the original URL "
                             "if the server rejects the request. Omit to use GBIF defaults.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records to process — useful for testing")
    parser.add_argument("--max-per-species", type=int, default=0, metavar="N",
                        help="Randomly subsample to at most N records per species before "
                             "downloading (0 = no cap).  Applied after --limit.")

    args = parser.parse_args()

    if not args.dwca and not args.family and not args.genus:
        parser.error("Provide --family or --genus (for API search), or --dwca (for local ZIP).")

    if args.continent and args.continent.upper() not in VALID_CONTINENTS:
        parser.error(f"Unknown continent '{args.continent}'. Valid: {', '.join(sorted(VALID_CONTINENTS))}")

    taxon     = args.family or args.genus or args.dwca.stem
    taxon_key = safe_name(taxon)
    out_dir   = args.output_dir or Path(f"{taxon_key}_images")
    specsin_path = args.specsin or Path(f"{taxon_key}_specsin.csv")

    max_size  = args.max_size if args.max_size and args.max_size > 0 else None
    iiif_size = args.iiif_size.strip() if args.iiif_size else None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing specsin (for species verification)
    existing = load_specsin(specsin_path)
    print(f"Loaded {len(existing)} existing records from {specsin_path}")

    # Get records — from local DwC-A ZIP or live GBIF API
    kw = dict(
        family=args.family,
        genus=args.genus,
        continent=args.continent.upper() if args.continent else None,
        countries=[c.upper() for c in args.countries] if args.countries else None,
        exclude_countries=[c.upper() for c in args.exclude_countries] if args.exclude_countries else None,
    )
    if args.dwca:
        records = load_dwca(args.dwca, **kw)
    else:
        records = search_occurrences(**kw)
    if args.limit:
        records = records[: args.limit]

    if args.max_per_species and args.max_per_species > 0:
        import random as _random
        by_species: dict[str, list] = {}
        for r in records:
            sp = (r.get("species") or r.get("genus") or "unknown").strip()
            by_species.setdefault(sp, []).append(r)
        records = []
        for sp, recs in by_species.items():
            if len(recs) > args.max_per_species:
                recs = _random.sample(recs, args.max_per_species)
            records.extend(recs)
        print(f"After per-species cap ({args.max_per_species}): {len(records)} records "
              f"across {len(by_species)} species/taxa")

    total = len(records)
    print(f"\nProcessing {total} records with {args.workers} workers...")
    print(f"Output directory: {out_dir}\n")

    newly_downloaded = failed = species_updated = 0

    # Start with a copy of existing rows; new/updated rows will overwrite
    updated: dict[str, dict] = dict(existing)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_record, r, out_dir, max_size, iiif_size): r for r in records}

        if tqdm is not None:
            completed = tqdm(
                as_completed(futures), total=total,
                desc="Downloading", unit="img",
                dynamic_ncols=True,
                postfix={"new": 0, "fail": 0, "sp_upd": 0},
            )
        else:
            completed = as_completed(futures)

        for future in completed:
            row, catalog, n_ok, n_fail = future.result()
            newly_downloaded += n_ok
            failed           += n_fail

            # Verify / update species against what GBIF currently says
            if catalog in existing:
                old_species = existing[catalog].get("species", "")
                new_species = row["species"]
                if old_species != new_species:
                    msg = f"  Species update [{catalog}]: '{old_species}' → '{new_species}'"
                    if tqdm is not None:
                        tqdm.write(msg)
                    else:
                        print(msg)
                    species_updated += 1
                merged = dict(existing[catalog])
                for field in ("species", "verbatimName", "indet",
                              "decimalLatitude", "decimalLongitude",
                              "coordinateUncertaintyInMeters", "gbifID"):
                    merged[field] = row[field]
                if row["hasfile"]:
                    merged["fname"]   = row["fname"]
                    merged["hasfile"] = row["hasfile"]
                updated[catalog] = merged
            else:
                updated[catalog] = row

            if tqdm is not None:
                completed.set_postfix({"new": newly_downloaded, "fail": failed,
                                       "sp_upd": species_updated})

    print(
        f"\nDone — {newly_downloaded} new images downloaded, "
        f"{failed} failures, {species_updated} species name(s) updated."
    )

    save_specsin(specsin_path, updated)


if __name__ == "__main__":
    main()
