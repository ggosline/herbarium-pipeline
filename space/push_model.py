"""
Publish a trained Lightning checkpoint to the Hugging Face Hub for use by
the herbarium-id Space. Strips optimizer state to roughly a third of the
original size so the Space loads faster.

Layout uploaded to <repo>:

    model.ckpt        — slimmed checkpoint (state_dict + hyper_parameters)
    nameslist.json    — list of class names at the model's label rank
    config.json       — { "model_name", "image_sz", "label_level" }
    README.md         — model card (only if --readme is passed)

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # one-time, with a write token

Usage:
    python push_model.py \
        --ckpt /media/ggosline/linuxdata/Magnoliopsida/cloud_results/checkpoints/<best>.ckpt \
        --nameslist /media/ggosline/linuxdata/Magnoliopsida/cloud_results/checkpoints/nameslist.json \
        --repo ggosline/herbarium-africa-magnoliopsida-family \
        --image-sz 640
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


def _slim_checkpoint(ckpt: dict, out_path: Path) -> None:
    keep = {
        "state_dict": ckpt["state_dict"],
        "hyper_parameters": ckpt.get("hyper_parameters", {}),
    }
    torch.save(keep, out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Local path to the .ckpt to publish.")
    p.add_argument("--nameslist", type=Path,
                   help="Local nameslist.json (defaults to ckpt's parent dir).")
    p.add_argument("--repo", required=True,
                   help="HF Hub model repo, e.g. ggosline/herbarium-africa-magnoliopsida-family")
    p.add_argument("--image-sz", type=int, default=640,
                   help="Image size used at training time. Default: 640")
    p.add_argument("--private", action="store_true",
                   help="Create the Hub repo as private.")
    p.add_argument("--readme", type=Path,
                   help="Optional model-card README.md to upload alongside the weights.")
    args = p.parse_args()

    nameslist_path = args.nameslist or (args.ckpt.parent / "nameslist.json")
    if not args.ckpt.exists():
        sys.exit(f"checkpoint not found: {args.ckpt}")
    if not nameslist_path.exists():
        sys.exit(f"nameslist not found: {nameslist_path}")

    print(f"→ Loading {args.ckpt}…")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    hp = ckpt.get("hyper_parameters", {}) or {}
    model_name = hp.get("model_name") or hp.get("backbone")
    label_level = hp.get("label_level", "family")
    if not model_name:
        sys.exit("hyper_parameters has no 'model_name' — pass it manually if needed.")
    print(f"   model_name={model_name}  label_level={label_level}")

    with open(nameslist_path) as f:
        nameslist = json.load(f)
    print(f"   {len(nameslist)} classes (first: {nameslist[:3]})")

    config = {
        "model_name": model_name,
        "image_sz":   args.image_sz,
        "label_level": label_level,
    }

    workdir = args.ckpt.parent / "_hub_upload"
    workdir.mkdir(exist_ok=True)
    slim_path = workdir / "model.ckpt"
    print(f"→ Slimming checkpoint → {slim_path}")
    _slim_checkpoint(ckpt, slim_path)
    orig_mb = args.ckpt.stat().st_size / 1e6
    slim_mb = slim_path.stat().st_size / 1e6
    print(f"   {orig_mb:,.0f} MB → {slim_mb:,.0f} MB")

    (workdir / "nameslist.json").write_text(json.dumps(nameslist, indent=2))
    (workdir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"→ Creating/ensuring repo {args.repo}")
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    api = HfApi()
    print("→ Uploading model.ckpt, nameslist.json, config.json")
    for fname in ("model.ckpt", "nameslist.json", "config.json"):
        api.upload_file(
            path_or_fileobj=str(workdir / fname),
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="model",
        )
    if args.readme and args.readme.exists():
        print(f"→ Uploading {args.readme.name} as README.md")
        api.upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
        )

    print(f"✓ Done — https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
