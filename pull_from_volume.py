"""
Read files off a RunPod network volume over its S3 gateway - no pod required.

Why
---
A RunPod network volume is a resource in its own right; it outlives the pods
that mount it. RunPod fronts it with an S3-compatible endpoint
(``s3api-<datacenter>.runpod.io``), so once a training/identify run has written
its results to the volume you can terminate the (expensive) GPU pod and still
pull ``predictions.csv``, the AUM sheets, or the images down for free.

This authenticates with the **S3 API key pair** (console -> Settings -> S3 API
Keys), which is separate from the REST API key. Store it once:

    uv run python pull_from_volume.py --save-keys ACCESS_KEY SECRET_KEY

Then list or fetch. The volume defaults to the active project's volume (read
from ``~/.herbarium-cloud/<project>.json``); its datacenter is resolved from
the RunPod REST API so you never have to hand-type the endpoint.

Usage
-----
    # What's on the volume under review/ and the images dir? (keys are relative
    # to the volume root, i.e. what was mounted at /workspace on the pod)
    uv run --with boto3 python pull_from_volume.py --list data/images/
    uv run --with boto3 python pull_from_volume.py --list data/predictions/

    # Pull one object, or everything under a prefix, into a local dir. Files
    # already present with the same size are skipped, so an interrupted pull
    # just resumes on re-run.
    uv run --with boto3 python pull_from_volume.py --get data/predictions/predictions.csv --dest ./pulled
    uv run --with boto3 python pull_from_volume.py --get data/images/ --dest /path/to/images

    # Pull only a named subset (one filename per line) — e.g. the AUM candidates.
    uv run --with boto3 python pull_from_volume.py --get data/images/ --dest ./imgs --files-from cands.txt

Uses boto3 (RunPod's documented S3 client — rclone's request signing is rejected
by their gateway). It is pulled in on the fly by ``uv run --with boto3`` so
nothing needs adding to pyproject. Read-only: list and get.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Repo root is on sys.path when run from here; cloud is a package.
from cloud import secrets as cloud_secrets

STATE_DIR = Path.home() / ".herbarium-cloud"


def _resolve_volume(project: str | None, volume: str | None) -> str:
    if volume:
        return volume
    if not project:
        sys.exit("Give --volume, or --project to read it from "
                 f"{STATE_DIR}/<project>.json")
    state = STATE_DIR / f"{project}.json"
    if not state.exists():
        sys.exit(f"No state file at {state} - pass --volume explicitly.")
    vid = json.loads(state.read_text()).get("volume_id")
    if not vid:
        sys.exit(f"{state} has no volume_id - pass --volume explicitly.")
    return vid


def _resolve_datacenter(volume: str, given: str | None) -> str:
    if given:
        return given
    key = cloud_secrets.get_runpod_api_key()
    if not key:
        sys.exit("No RunPod REST key in keyring to look up the datacenter - "
                 "pass --datacenter (e.g. EUR-IS-1).")
    import httpx
    r = httpx.get("https://rest.runpod.io/v1/networkvolumes",
                  headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    vols = data if isinstance(data, list) else data.get("networkVolumes", [])
    for v in vols:
        if v.get("id") == volume:
            if v.get("dataCenterId"):
                return v["dataCenterId"]
    sys.exit(f"Volume {volume} not found via REST - pass --datacenter explicitly.")


def _client(endpoint: str, dc: str, creds: cloud_secrets.RunPodS3Credentials):
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=dc,                       # RunPod uses the DC id as region
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        # Bucket is the volume id; path-style addressing avoids DNS games.
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "path"}),
    )


def _iter_objects(s3, bucket: str, prefix: str):
    """Yield (key, size) under prefix, paging through the listing."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024
    return f"{n} B"


def _local_for(key: str, target: str, dest: Path) -> Path:
    """Where a key lands locally: strip the prefix for a folder pull so the
    subtree mirrors under dest; a single-object pull lands as its basename."""
    rel = key[len(target):].lstrip("/") if target.endswith("/") else Path(key).name
    return dest / rel


def _do_list(s3, bucket: str, prefix: str) -> int:
    n = total = 0
    for key, size in _iter_objects(s3, bucket, prefix):
        print(f"  {_human(size):>12}  {key}")
        n += 1
        total += size
    print(f"\n{n:,} object(s), {_human(total)} under {prefix!r}")
    return 0


def _do_get(s3, bucket: str, target: str, dest: Path,
            only: set[str] | None, workers: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[str, int, Path]] = []
    skipped = skip_bytes = 0
    print(f"Listing {target!r} ...")
    for key, size in _iter_objects(s3, bucket, target):
        if only is not None and Path(key).name not in only:
            continue
        local = _local_for(key, target, dest)
        # Skip files already present at the same size — makes the pull resumable
        # and lets it complete a partial download without re-fetching the rest.
        if local.is_file() and local.stat().st_size == size:
            skipped += 1
            skip_bytes += size
            continue
        todo.append((key, size, local))

    if only is not None:
        print(f"  matched {len(todo) + skipped:,} of the {len(only):,} names requested")
    print(f"  {skipped:,} already present ({_human(skip_bytes)}), "
          f"{len(todo):,} to download ({_human(sum(s for _, s, _ in todo))})")
    if not todo:
        print("\nNothing to download — everything is already local.")
        return 0

    done = [0]
    failed: list[tuple[str, str]] = []

    def _pull(item):
        key, size, local = item
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        try:
            # get_object, not download_file: the latter issues a HeadObject first,
            # which RunPod's gateway 403s under bulk/concurrency (and 404s on keys
            # containing '='). Streaming the body is one request and sidesteps both.
            resp = s3.get_object(Bucket=bucket, Key=key)
            with open(tmp, "wb") as f:
                for chunk in resp["Body"].iter_chunks(1 << 20):
                    f.write(chunk)
            tmp.replace(local)          # atomic: a killed pull never leaves a half file
        except Exception as e:          # noqa: BLE001 - report, keep going
            tmp.unlink(missing_ok=True)
            return key, str(e)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_pull, it) for it in todo]
        for f in as_completed(futs):
            err = f.result()
            if err:
                failed.append(err)
            done[0] += 1
            if done[0] % 200 == 0 or done[0] == len(todo):
                print(f"  … {done[0]:,}/{len(todo):,} "
                      f"({len(failed)} failed)", flush=True)

    if failed:
        print(f"\n⚠ {len(failed)} file(s) failed; first few:")
        for key, err in failed[:5]:
            print(f"    {key}: {err[:100]}")
        print("Re-run the same command to retry just the ones still missing.")
        return 1
    print(f"\n✓ pulled {len(todo):,} file(s) into {dest}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-keys", nargs=2, metavar=("ACCESS_KEY", "SECRET_KEY"),
                   help="Store the RunPod S3 key pair in the OS keyring and exit.")
    p.add_argument("--project", help="Project name; volume read from its state file.")
    p.add_argument("--volume", help="Network volume id (overrides --project lookup).")
    p.add_argument("--datacenter", help="e.g. EUR-IS-1 (else resolved via REST).")
    p.add_argument("--list", dest="list_prefix", metavar="PREFIX",
                   help='List objects under PREFIX (use "" for the whole volume).')
    p.add_argument("--get", metavar="KEY_OR_PREFIX",
                   help="Download one object, or everything under a prefix.")
    p.add_argument("--dest", type=Path, default=Path("./pulled"),
                   help="Local directory for --get (default ./pulled).")
    p.add_argument("--files-from", type=Path,
                   help="Only pull objects whose basename is listed in this file "
                        "(one per line) — e.g. the AUM candidate images.")
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel downloads for --get (default 16).")
    args = p.parse_args()

    if args.save_keys:
        cloud_secrets.set_runpod_s3_credentials(
            cloud_secrets.RunPodS3Credentials(*[s.strip() for s in args.save_keys]))
        print("✓ RunPod S3 keys saved to keyring "
              f"({cloud_secrets.SERVICE_NAME}/{cloud_secrets.RUNPOD_S3_KEY}).")
        return 0

    if args.list_prefix is None and not args.get:
        p.error("nothing to do - give --list PREFIX or --get KEY (or --save-keys).")

    creds = cloud_secrets.get_runpod_s3_credentials()
    if creds is None:
        sys.exit("No RunPod S3 keys in keyring. Save them first:\n"
                 "  uv run python pull_from_volume.py --save-keys ACCESS SECRET")

    bucket = _resolve_volume(args.project, args.volume)
    dc = _resolve_datacenter(bucket, args.datacenter)
    endpoint = cloud_secrets.RunPodS3Credentials.endpoint_for(dc)
    print(f"volume {bucket} @ {dc}  ({endpoint})")
    s3 = _client(endpoint, dc, creds)

    if args.list_prefix is not None:
        return _do_list(s3, bucket, args.list_prefix)

    only: set[str] | None = None
    if args.files_from:
        only = {ln.strip() for ln in args.files_from.read_text().splitlines() if ln.strip()}
        print(f"restricting to {len(only):,} named files from {args.files_from}")
    return _do_get(s3, bucket, args.get, args.dest, only, max(1, args.workers))


if __name__ == "__main__":
    sys.exit(main())
