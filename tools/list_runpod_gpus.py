#!/usr/bin/env python3
"""Print the GPU type IDs RunPod's REST API will accept.

Reads them from RunPod's published OpenAPI schema (the same source the
API server validates against), so the output is authoritative — anything
listed here is a legal value for the "GPU type override" field in the
☁ Cloud tab and for ``GPU_BY_PURPOSE`` in cloud/orchestrator.py.

Usage:
    python tools/list_runpod_gpus.py            # full list
    python tools/list_runpod_gpus.py --grep L4  # filter by substring
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make the repo root importable when invoked as `python tools/list_runpod_gpus.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud import secrets as cloud_secrets
from cloud.runpod_client import RunPodClient


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grep", default="",
                    help="case-insensitive substring filter")
    args = ap.parse_args()

    api_key = cloud_secrets.get_runpod_api_key() or "openapi-spec-is-public"
    async with RunPodClient(api_key) as rp:
        ids = await rp.list_gpu_types()

    needle = args.grep.lower()
    rows = [g for g in ids if not needle or needle in g.lower()]
    rows.sort()
    for g in rows:
        print(f"  {g}")
    print(f"\n{len(rows)} of {len(ids)} GPU type(s).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
