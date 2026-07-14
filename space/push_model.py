"""
Publish a trained Lightning checkpoint to the Hugging Face Hub for use by
the herbarium-id Space. Strips optimizer state to roughly a third of the
original size so the Space loads faster.

Source-agnostic: ``--ckpt`` accepts a single ``.ckpt`` file *or* a
directory of checkpoints (the best by ``valid_loss`` in the filename is
chosen, falling back to the most recent). This means the same command
publishes from a local ``runs/`` dir, from a pod's
``/workspace/data/checkpoints/``, or from a directory you rclone'd out of
the R2 backup — the caller just stages the checkpoint to a path.

The class names list is read from the checkpoint itself (training embeds
it under the top-level ``nameslist`` key), so a sidecar ``nameslist.json``
is optional. This matters for the R2 path, where ``backup()`` does not
archive ``nameslist.json`` separately.

Layout uploaded to <repo>:

    model.ckpt        — slimmed checkpoint (state_dict + hyper_parameters
                        + embedded nameslist)
    nameslist.json    — list of class names at the model's label rank
    config.json       — { model_name, image_sz, label_level, family,
                          region, display_name, num_classes, valid_loss }
    README.md         — model card with discovery tags (auto-generated
                        unless --readme is passed)

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # one-time, with a write token
    # …or pass --token, set HF_TOKEN, or drop the token at
    # /workspace/.hf_token (chmod 600) for unattended pod-side runs.

Usage:
    # From a checkpoints directory (auto-pick best), deriving the repo
    # name from the family:
    python push_model.py \
        --ckpt /workspace/data/checkpoints \
        --family Menispermaceae --region Africa --label-level species \
        --hf-user ggosline

    # Explicit repo + single checkpoint (back-compatible):
    python push_model.py \
        --ckpt .../checkpoints/epoch=07-valid_loss=0.41.ckpt \
        --repo ggosline/herbarium-africa-magnoliopsida-family \
        --image-sz 640
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


# Training saves both a loss-best and an accuracy-best checkpoint per stage,
# embedding the metric in the filename, e.g.:
#   epoch=07-valid_loss=0.4072.ckpt       (stage 2, loss-best)
#   acc-07-val_Accuracy=0.9123.ckpt       (stage 2, accuracy-best)
#   s1-03-valid_loss=0.68.ckpt / s1-acc-03-val_Accuracy=0.88.ckpt
# We publish the accuracy-best by default — accuracy is the deployment
# objective for an ID tool — falling back to loss when no acc checkpoint
# exists (older runs).
_LOSS_RE = re.compile(r"valid_loss=(\d+\.\d+)")
_ACC_RE  = re.compile(r"val_Accuracy=(\d+\.\d+)")


def _metrics_from_name(name: str) -> dict[str, float | None]:
    lm = _LOSS_RE.search(name)
    am = _ACC_RE.search(name)
    return {
        "valid_loss":   float(lm.group(1)) if lm else None,
        "val_accuracy": float(am.group(1)) if am else None,
    }


def _pick_checkpoint(path: Path, select_by: str = "accuracy",
                     ) -> tuple[Path, dict[str, float | None]]:
    """Resolve ``path`` to a single checkpoint.

    A file is returned as-is. For a directory, the checkpoint that best
    satisfies ``select_by`` ("accuracy" → max val_Accuracy, "loss" → min
    valid_loss) wins; if no filename carries the preferred metric we fall
    back to the other, then to the most recently modified ``.ckpt``.
    Returns (chosen_path, metrics_parsed_from_its_name).
    """
    if path.is_file():
        return path, _metrics_from_name(path.name)
    if not path.is_dir():
        sys.exit(f"checkpoint path not found: {path}")

    ckpts = list(path.glob("*.ckpt"))
    if not ckpts:
        sys.exit(f"no .ckpt files found in {path}")

    # (metric_key, want_max, pretty_label)
    by_acc  = ("val_accuracy", True,  "val_Accuracy")
    by_loss = ("valid_loss",   False, "valid_loss")
    order = [by_acc, by_loss] if select_by == "accuracy" else [by_loss, by_acc]

    for i, (key, want_max, label) in enumerate(order):
        cand = [(m[key], p) for p in ckpts
                if (m := _metrics_from_name(p.name))[key] is not None]
        if not cand:
            continue
        cand.sort(key=lambda x: x[0], reverse=want_max)
        best_val, best = cand[0]
        print(f"   best-by-{label}: {best.name} ({label}={best_val:.4f})")
        if i != 0:
            primary = order[0][2]
            print(f"   (no {primary} in any filename — fell back to {label})")
        return best, _metrics_from_name(best.name)

    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    print(f"   no parseable metric — using most recent: {latest.name}")
    return latest, _metrics_from_name(latest.name)


def _extract_nameslist(ckpt: dict, fallback: Path | None) -> list[str]:
    """Class names at the model's label rank.

    Preferred source is the checkpoint's embedded ``nameslist`` (written by
    training's ``on_save_checkpoint``); it may be a flat list or a dict of
    per-rank lists. Falls back to a sidecar JSON file when absent.
    """
    embedded = ckpt.get("nameslist")
    if embedded:
        if isinstance(embedded, dict):
            names = embedded.get("species") or max(embedded.values(), key=len)
            print(f"   nameslist: embedded in checkpoint (hierarchical; "
                  f"{len(names)} classes)")
        else:
            names = embedded
            print(f"   nameslist: embedded in checkpoint ({len(names)} classes)")
        return list(names)

    if fallback and fallback.exists():
        raw = json.loads(fallback.read_text())
        names = raw.get("species") or max(raw.values(), key=len) if isinstance(raw, dict) else raw
        print(f"   nameslist: {fallback} ({len(names)} classes)")
        return list(names)

    sys.exit("no nameslist found — checkpoint has no embedded 'nameslist' and "
             "no --nameslist file was given/found.")


def _extract_genus_nameslist(ckpt: dict) -> list[str]:
    """Genus class names, present only on hierarchical checkpoints.

    The embedded nameslist is a dict of per-rank lists there; a flat list means
    a single-rank model, which has no genus head to publish.
    """
    embedded = ckpt.get("nameslist")
    if isinstance(embedded, dict):
        return list(embedded.get("genus") or [])
    return []


def _resolve_token(explicit: str | None, token_file: Path | None) -> str | None:
    """HF write token: --token > $HF_TOKEN > --token-file > /workspace/.hf_token.

    Returns None to fall back on a cached ``huggingface-cli login``.
    """
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    candidates = [token_file] if token_file else []
    candidates.append(Path("/workspace/.hf_token"))
    for c in candidates:
        if c and c.exists():
            tok = c.read_text().strip()
            if tok:
                print(f"   token: read from {c}")
                return tok
    return None


def _default_repo(hf_user: str | None, family: str | None, region: str | None,
                  label_level: str) -> str:
    """Convention-based repo id when --repo is omitted:
    ``<user>/herbarium-<region>-<family>-<rank>`` (lowercased)."""
    if not (hf_user and family):
        sys.exit("either --repo, or both --hf-user and --family, must be given.")
    parts = ["herbarium", region or "", family, label_level]
    slug = "-".join(p for p in parts if p).lower().replace(" ", "-")
    return f"{hf_user}/{slug}"


def _default_display_name(family: str | None, region: str | None,
                          label_level: str, repo: str) -> str:
    if family and region:
        return f"{family} ({region}) — {label_level} rank"
    if family:
        return f"{family} — {label_level} rank"
    return f"{repo.split('/')[-1]} — {label_level} rank"


def _slim_checkpoint(ckpt: dict, out_path: Path) -> None:
    keep = {
        "state_dict": ckpt["state_dict"],
        "hyper_parameters": ckpt.get("hyper_parameters", {}),
    }
    # Preserve the embedded nameslist so the published .ckpt stays
    # self-describing even if nameslist.json is later separated from it.
    if "nameslist" in ckpt:
        keep["nameslist"] = ckpt["nameslist"]
    torch.save(keep, out_path)


def _write_model_card(path: Path, *, display_name: str, repo: str,
                      config: dict, nameslist: list[str]) -> None:
    """Generate a model card whose YAML frontmatter tags the repo for the
    Space's ``discover_models()`` (filters on the ``herbarium-pipeline``
    library/tag)."""
    family = config.get("family") or ""
    tags = ["herbarium-pipeline", "image-classification", "biology", "plants"]
    if family:
        tags.append(family.lower())
    fm_tags = "\n".join(f"- {t}" for t in tags)
    sample = ", ".join(nameslist[:8]) + (" …" if len(nameslist) > 8 else "")
    genus_line = (f"\n- **Genus head:** {config['genus_classes']} genera "
                  f"(predicted directly, not derived from the species name)"
                  if config.get("genus_classes") else "")
    metric = ""
    if config.get("val_accuracy") is not None:
        metric += f"\n- **Validation accuracy:** {config['val_accuracy']:.4f}"
    if config.get("valid_loss") is not None:
        metric += f"\n- **Validation loss:** {config['valid_loss']:.4f}"
    path.write_text(
        f"""---
library_name: herbarium-pipeline
pipeline_tag: image-classification
tags:
{fm_tags}
---

# {display_name}

Herbarium specimen classifier published by the
[herbarium-pipeline](https://github.com/ggosline/herbarium-pipeline) project.

- **Backbone:** `{config['model_name']}`
- **Label rank:** {config['label_level']}
- **Classes:** {config['num_classes']}{genus_line}
- **Input size:** {config['image_sz']} px{metric}

Classes (sample): {sample}

Loaded automatically by the herbarium-id Space — no code change needed.
"""
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path,
                   help="A .ckpt file OR a directory of checkpoints "
                        "(best by --select-by is chosen, else most recent).")
    p.add_argument("--select-by", choices=("accuracy", "loss"), default="accuracy",
                   help="When --ckpt is a directory, pick the best checkpoint by "
                        "validation accuracy (default) or loss.")
    p.add_argument("--nameslist", type=Path,
                   help="Optional sidecar nameslist.json. Only used if the "
                        "checkpoint has no embedded nameslist.")
    p.add_argument("--repo",
                   help="HF Hub model repo, e.g. ggosline/herbarium-africa-"
                        "menispermaceae-species. If omitted, derived from "
                        "--hf-user/--family/--region/--label-level.")
    p.add_argument("--hf-user", help="HF username, used to build --repo when omitted.")
    p.add_argument("--family", help="Family this model classifies within / at.")
    p.add_argument("--region", help="Geographic scope, e.g. Africa.")
    p.add_argument("--label-level", choices=("species", "genus", "family"),
                   help="Override the checkpoint's label_level if needed.")
    p.add_argument("--display-name", help="Human-readable name for the Space dropdown.")
    p.add_argument("--image-sz", type=int, default=None,
                   help="Image size used at training time. Defaults to the "
                        "checkpoint's hyper_parameters, then 640.")
    p.add_argument("--token", help="HF write token (else $HF_TOKEN / token file).")
    p.add_argument("--token-file", type=Path,
                   help="File holding an HF write token (default /workspace/.hf_token).")
    p.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    p.add_argument("--readme", type=Path,
                   help="Custom model-card README.md (else one is generated with "
                        "discovery tags).")
    args = p.parse_args()

    # 1. Resolve the checkpoint (file or directory).
    print(f"→ Resolving checkpoint from {args.ckpt} (select_by={args.select_by})…")
    ckpt_path, metrics = _pick_checkpoint(args.ckpt, args.select_by)
    print(f"   using {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2. Metadata from hyper_parameters.
    hp = ckpt.get("hyper_parameters", {}) or {}
    model_name = hp.get("model_name") or hp.get("backbone")
    if not model_name:
        sys.exit("hyper_parameters has no 'model_name' — cannot determine architecture.")
    label_level = args.label_level or hp.get("label_level", "family")
    image_sz = args.image_sz or hp.get("image_sz") or 640
    print(f"   model_name={model_name}  label_level={label_level}  image_sz={image_sz}")

    # 3. Class names (embedded → sidecar fallback).
    fallback = args.nameslist or (ckpt_path.parent / "nameslist.json")
    nameslist = _extract_nameslist(ckpt, fallback)

    # A hierarchical checkpoint also carries a trained genus head, which is the
    # model's most accurate output (~97% vs ~89% at species). Publish its class
    # names too so the Space can offer genus as a first-class answer instead of
    # guessing one from the first word of the species prediction.
    genus_nameslist = _extract_genus_nameslist(ckpt)
    has_genus_head = bool(
        genus_nameslist
        and any(k.startswith(("model.head_genus.", "head_genus."))
                for k in ckpt.get("state_dict", {}))
    )
    if genus_nameslist and not has_genus_head:
        print("   [warn] genus names present but no head_genus.* weights — "
              "publishing species only")
        genus_nameslist = []
    if has_genus_head:
        print(f"   genus head: {len(genus_nameslist)} genera")

    # 4. Repo + display name.
    repo = args.repo or _default_repo(args.hf_user, args.family, args.region, label_level)
    display_name = args.display_name or _default_display_name(
        args.family, args.region, label_level, repo)
    print(f"   repo={repo}")
    print(f"   display_name={display_name}")

    # Calibration temperature fitted at end of training (train_herbarium
    # embeds it in the checkpoint). The Space applies softmax(logits / T);
    # 1.0 for older checkpoints trained before calibration existed.
    try:
        temperature = round(float(ckpt.get("temperature", 1.0)) or 1.0, 4)
    except (TypeError, ValueError):
        temperature = 1.0
    print(f"   temperature={temperature}")

    config = {
        "model_name":   model_name,
        "image_sz":     int(image_sz),
        "label_level":  label_level,
        "family":       args.family or "",
        "region":       args.region or "",
        "display_name": display_name,
        "num_classes":  len(nameslist),
        "genus_classes": len(genus_nameslist) if has_genus_head else 0,
        "temperature":  temperature,
        "valid_loss":   round(metrics["valid_loss"], 4) if metrics.get("valid_loss") is not None else None,
        "val_accuracy": round(metrics["val_accuracy"], 4) if metrics.get("val_accuracy") is not None else None,
    }

    # 5. Stage upload artifacts.
    workdir = ckpt_path.parent / "_hub_upload"
    workdir.mkdir(exist_ok=True)
    slim_path = workdir / "model.ckpt"
    print(f"→ Slimming checkpoint → {slim_path}")
    _slim_checkpoint(ckpt, slim_path)
    orig_mb = ckpt_path.stat().st_size / 1e6
    slim_mb = slim_path.stat().st_size / 1e6
    print(f"   {orig_mb:,.0f} MB → {slim_mb:,.0f} MB")

    (workdir / "nameslist.json").write_text(json.dumps(nameslist, indent=2))
    (workdir / "config.json").write_text(json.dumps(config, indent=2))
    upload_files = ["model.ckpt", "nameslist.json", "config.json", "README.md"]
    if has_genus_head:
        (workdir / "genus_nameslist.json").write_text(
            json.dumps(genus_nameslist, indent=2))
        upload_files.append("genus_nameslist.json")

    readme_path = workdir / "README.md"
    if args.readme and args.readme.exists():
        readme_path.write_text(args.readme.read_text())
    else:
        _write_model_card(readme_path, display_name=display_name, repo=repo,
                          config=config, nameslist=nameslist)

    # 6. Publish.
    token = _resolve_token(args.token, args.token_file)
    print(f"→ Creating/ensuring repo {repo}")
    create_repo(repo, repo_type="model", private=args.private, exist_ok=True, token=token)

    api = HfApi(token=token)
    print(f"→ Uploading {', '.join(upload_files)}")
    for fname in upload_files:
        api.upload_file(
            path_or_fileobj=str(workdir / fname),
            path_in_repo=fname,
            repo_id=repo,
            repo_type="model",
        )

    print(f"✓ Done — https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
