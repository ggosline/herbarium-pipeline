"""
Read files off a RunPod network volume over its S3 gateway - no pod required.

Why
---
A RunPod network volume is a resource in its own right; it outlives the pods
that mount it. RunPod fronts it with an S3-compatible endpoint
(``s3api-<datacenter>.runpod.io``), so once a training/identify run has written
its results to the volume you can terminate the (expensive) GPU pod and still
pull ``predictions.csv``, the AUM sheets, or a checkpoint down for free.

This authenticates with the **S3 API key pair** (console -> Settings -> S3 API
Keys), which is separate from the REST API key. Store it once:

    uv run python pull_from_volume.py --save-keys ACCESS_KEY SECRET_KEY

Then list or fetch. The volume defaults to the active project's volume (read
from ``~/.herbarium-cloud/<project>.json``); its datacenter is resolved from
the RunPod REST API so you never have to hand-type the endpoint.

Usage
-----
    # What's on the volume under review/ and the checkpoints dir?
    uv run python pull_from_volume.py --project Rubiaceae-genera --list review/
    uv run python pull_from_volume.py --list checkpoints/

    # Pull one file, or everything under a prefix, into ./pulled/
    uv run python pull_from_volume.py --get predictions/predictions.csv --dest ./pulled
    uv run python pull_from_volume.py --get review/ --dest ./review_aum

    # Point at a volume explicitly (skips the state-file lookup)
    uv run python pull_from_volume.py --volume cte9steisi --datacenter EUR-IS-1 --list ""

Transfers run through rclone (already used by this project for R2), which
handles SigV4, path-style addressing, and multipart. Only reads (list/get) are
exposed - that's what RunPod's S3 layer does reliably.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Windows consoles default to cp1252; our own output uses a few non-ASCII marks.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Repo root is on sys.path when run from here; cloud is a package.
from cloud import secrets as cloud_secrets

STATE_DIR = Path.home() / ".herbarium-cloud"


def _resolve_volume(project: str | None, volume: str | None) -> str:
    """Return the volume id, from --volume or the project's state file."""
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
    """Return the volume's datacenter, from --datacenter or the REST API."""
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
            dc = v.get("dataCenterId")
            if dc:
                return dc
    sys.exit(f"Volume {volume} not found via REST - pass --datacenter explicitly.")


def _rclone_env(endpoint: str, dc: str,
                creds: cloud_secrets.RunPodS3Credentials) -> dict[str, str]:
    """Environment for an on-the-fly ``:s3:`` remote pointed at RunPod.

    Passing the keys via ``RCLONE_S3_*`` env keeps them out of the process
    argument list (where any user could read them from ``ps``).
    """
    env = dict(os.environ)
    env.update(
        RCLONE_S3_PROVIDER="Other",
        RCLONE_S3_ACCESS_KEY_ID=creds.access_key_id,
        RCLONE_S3_SECRET_ACCESS_KEY=creds.secret_access_key,
        RCLONE_S3_ENDPOINT=endpoint,
        RCLONE_S3_REGION=dc,
        RCLONE_S3_FORCE_PATH_STYLE="true",
    )
    return env


def _rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        sys.exit("rclone is not on PATH. Install it (https://rclone.org/downloads/) "
                 "- this project already uses it for R2.")
    return exe


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024
    return f"{n} B"


def _list(rclone: str, env: dict, bucket: str, prefix: str) -> int:
    src = f":s3:{bucket}/{prefix}" if prefix else f":s3:{bucket}"
    proc = subprocess.run([rclone, "lsjson", "--recursive", src],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"rclone lsjson failed:\n{proc.stderr.strip()}")
    items = [o for o in json.loads(proc.stdout or "[]") if not o.get("IsDir")]
    items.sort(key=lambda o: o["Path"])
    total = 0
    for o in items:
        print(f"  {_human(o['Size']):>12}  {prefix.rstrip('/') + '/' if prefix else ''}{o['Path']}")
        total += o["Size"]
    print(f"\n{len(items):,} object(s), {_human(total)} under {prefix!r}")
    return 0


def _get(rclone: str, env: dict, bucket: str, target: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    src = f":s3:{bucket}/{target}"
    # `rclone copy` auto-detects a single object vs a prefix: an object lands as
    # its basename in dest; a prefix mirrors its subtree under dest.
    cmd = [rclone, "copy", src, str(dest),
           "--progress", "--stats-one-line", "--transfers", "8"]
    print(f"rclone copy {src} -> {dest}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        sys.exit(f"rclone copy failed (exit {proc.returncode}). "
                 f"If nothing exists at {target!r}, check the prefix with --list.")
    print(f"\n✓ pulled into {dest}")
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
    rclone = _rclone()
    env = _rclone_env(endpoint, dc, creds)

    if args.list_prefix is not None:
        return _list(rclone, env, bucket, args.list_prefix)
    return _get(rclone, env, bucket, args.get, args.dest)


if __name__ == "__main__":
    sys.exit(main())
