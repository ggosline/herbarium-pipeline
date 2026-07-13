"""
Fit a softmax calibration temperature for an ALREADY-TRAINED checkpoint and
(optionally) patch it onto a published Hugging Face model — no retraining.

Why this exists
---------------
Models trained with plain cross-entropy are over-confident: the top-1 softmax
probability sits near 100% even when the model is only ~85% accurate. Temperature
scaling (Guo et al. 2017, "On Calibration of Modern Neural Networks") is a
post-hoc fix: divide the logits by a single scalar T (fit on held-out data)
before softmax. It changes only the reported probabilities, never the argmax,
so accuracy is unchanged.

This script fits T for an existing checkpoint and writes it into the model's
`config.json` on the Hub. The Space (`space/app.py`) reads `config["temperature"]`
and applies `softmax(logits / T)`. Only the tiny config.json is re-uploaded —
the multi-hundred-MB `model.ckpt` is untouched.

How the held-out split is recovered
-----------------------------------
train_herbarium splits data with a deterministic, seeded, stratified
`train_test_split`. Given the SAME specsin rows, seed, and split fraction, the
exact validation set is reproducible. All those parameters (seed,
sparse_threshold, train_val_split, label_level, hierarchical, max_per_species)
are embedded in the checkpoint's hyper_parameters, so we reuse the real
`HerbariumData` class to rebuild the identical split. As an integrity check we
compare the reconstructed class list against the checkpoint's embedded
nameslist: if they match, the underlying `combined` set matched, so the split
matches. If your specsin.csv has changed since training (rows added/removed),
the check will fail — see --allow-nameslist-mismatch.

Logits are collected with the SAME torchvision preprocessing the Space uses
(Resize + CenterCrop + ImageNet normalize), so T is calibrated for exactly how
the model runs in production.

Usage
-----
    # Fit T from local data and patch the Hub config.json:
    python calibrate_temperature.py \
        --checkpoint C:/AIProjects/africa_angiosperms/runs/checkpoints \
        --sources C:/AIProjects/africa_angiosperms/specsin.csv:C:/AIProjects/africa_angiosperms/images_1024 \
        --repo ggosline/herbarium-africa-angiosperms-family

    # Just compute T, don't upload (inspect first):
    python calibrate_temperature.py --checkpoint ... --sources ... --dry-run

    # Skip fitting; write a manual heuristic T (no data needed):
    python calibrate_temperature.py --repo ggosline/... --set-temperature 2.5

Prerequisites: run in the training env (p12) so torch/timm/DALI import. HF
write token via --token, $HF_TOKEN, or a cached `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# NB: torch / identify_herbarium / train_herbarium are imported lazily inside
# the fitting path of main() so the --set-temperature heuristic (which needs
# no data, no model, no GPU) runs with only huggingface_hub installed.


# ---------------------------------------------------------------------------
# Temperature fit + calibration metrics
# ---------------------------------------------------------------------------

def fit_temperature(logits, targets, max_iter: int = 100) -> float:
    """Minimise validation NLL over a single scalar T (parameterised as
    exp(log_T) so it stays positive). Returns the fitted temperature."""
    import torch
    import torch.nn as nn
    logits  = logits.double()
    targets = targets.long()
    log_T = torch.zeros(1, dtype=torch.double, requires_grad=True)
    nll   = nn.CrossEntropyLoss()
    opt   = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_T.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.detach().exp().item())


def calibration_report(logits, targets, T: float, n_bins: int = 15) -> dict:
    """NLL, mean top-1 confidence, accuracy, and expected calibration error
    (ECE) at temperature T. Accuracy is T-invariant (argmax unchanged)."""
    import torch
    import torch.nn as nn
    with torch.no_grad():
        probs = torch.softmax(logits.double() / T, dim=1)
        conf, pred = probs.max(dim=1)
        acc_vec = pred.eq(targets)
        nll = nn.CrossEntropyLoss()(logits.double() / T, targets.long()).item()

        # Expected calibration error: |acc - confidence| averaged over bins,
        # weighted by bin population.
        ece = 0.0
        edges = torch.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += (m.float().mean() *
                        (acc_vec[m].float().mean() - conf[m].mean()).abs()).item()
    return {
        "nll":       nll,
        "mean_conf": conf.mean().item(),
        "accuracy":  acc_vec.float().mean().item(),
        "ece":       ece,
    }


# ---------------------------------------------------------------------------
# Checkpoint hyper-parameter access (config may be flat or nested)
# ---------------------------------------------------------------------------

def _hp_get(hp: dict, key: str, default):
    if key in hp:
        return hp[key]
    cfg = hp.get("config", {})
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


# ---------------------------------------------------------------------------
# Hub patching
# ---------------------------------------------------------------------------

def _resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)


def patch_hub_temperature(repo: str, temperature: float, token: str | None) -> None:
    """Download the model's config.json, set temperature, re-upload just that
    file. Leaves model.ckpt / nameslist.json / README.md untouched."""
    from huggingface_hub import hf_hub_download, HfApi

    cfg_path = hf_hub_download(repo_id=repo, filename="config.json",
                               repo_type="model", token=token)
    with open(cfg_path) as f:
        cfg = json.load(f)
    old = cfg.get("temperature", 1.0)
    cfg["temperature"] = round(float(temperature), 4)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "config.json"
        out.write_text(json.dumps(cfg, indent=2))
        HfApi(token=token).upload_file(
            path_or_fileobj=str(out), path_in_repo="config.json",
            repo_id=repo, repo_type="model",
            commit_message=f"Set calibration temperature {cfg['temperature']} "
                           f"(was {old})",
        )
    print(f"✓ Patched {repo} config.json: temperature {old} → {cfg['temperature']}")
    print(f"  https://huggingface.co/{repo}/blob/main/config.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Windows consoles default to cp1252 and crash on the arrows/checkmarks
    # below; force UTF-8 so the script runs the same everywhere.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   help="Published .ckpt file, or a directory (auto-picks latest). "
                        "Required unless --set-temperature is used.")
    p.add_argument("--sources", nargs="+", metavar="CSV:DIR",
                   help="specsin.csv:images_dir pairs (same as train/identify). "
                        "Required unless --set-temperature is used.")
    p.add_argument("--repo", help="HF model repo to patch, e.g. ggosline/herbarium-"
                                  "africa-angiosperms-family. Omit with --dry-run.")
    p.add_argument("--set-temperature", type=float, default=None,
                   help="Skip fitting and write this fixed T to the Hub config "
                        "(heuristic fallback when validation data isn't available).")
    p.add_argument("--dry-run", action="store_true",
                   help="Fit/print the temperature but do not upload anything.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--token", help="HF write token (else $HF_TOKEN / cached login).")
    p.add_argument("--allow-nameslist-mismatch", action="store_true",
                   help="Proceed even if the reconstructed class list disagrees "
                        "with the checkpoint (split may not match training; T will "
                        "be approximate).")
    # Split-parameter overrides. Defaults come from the checkpoint's
    # hyper_parameters; only pass these if the checkpoint predates them.
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sparse-threshold", type=int, default=None)
    p.add_argument("--train-val-split", type=float, default=None)
    p.add_argument("--label-level", choices=("species", "genus", "family"), default=None)
    p.add_argument("--max-per-species", type=int, default=None)
    p.add_argument("--image-sz", type=int, default=None)
    args = p.parse_args()

    token = _resolve_token(args.token)

    # --- Heuristic shortcut: write a fixed T, no data / no fitting ----------
    if args.set_temperature is not None:
        if not args.repo:
            sys.exit("--set-temperature requires --repo.")
        if args.set_temperature <= 0:
            sys.exit("--set-temperature must be > 0.")
        print(f"→ Writing heuristic temperature {args.set_temperature} to {args.repo} "
              f"(no fitting).")
        patch_hub_temperature(args.repo, args.set_temperature, token)
        return 0

    if not args.checkpoint or not args.sources:
        sys.exit("--checkpoint and --sources are required (unless --set-temperature).")

    # Heavy deps only needed for fitting. Imported here so --set-temperature
    # above stays torch/DALI-free. train_herbarium is imported further down.
    import torch
    from torch.utils.data import DataLoader
    from identify_herbarium import (
        resolve_checkpoint, load_model, build_model_from_state, InferenceDataset,
    )

    # --- 1. Load checkpoint + its split hyper-parameters -------------------
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"→ Checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {}) or {}

    seed         = args.seed            if args.seed            is not None else int(_hp_get(hp, "seed", 42))
    sparse       = args.sparse_threshold if args.sparse_threshold is not None else int(_hp_get(hp, "sparse_threshold", 5))
    val_split    = args.train_val_split if args.train_val_split is not None else float(_hp_get(hp, "train_val_split", 0.2))
    hierarchical = bool(_hp_get(hp, "hierarchical", False))
    label_level  = args.label_level     if args.label_level     is not None else str(_hp_get(hp, "label_level", "species"))
    max_per_sp   = args.max_per_species if args.max_per_species is not None else int(_hp_get(hp, "max_per_species", 0))
    print(f"   split params: seed={seed} sparse_threshold={sparse} "
          f"train_val_split={val_split} label_level={label_level} "
          f"hierarchical={hierarchical} max_per_species={max_per_sp}")

    # --- 2. Reconstruct the model (weights + geo/plain arch) ---------------
    (state_dict, ckpt_model_name, num_classes, ck_nameslist, geo_dim, _lvl, cur_T,
     _excluded, _class_counts) = load_model(ckpt_path, [], 640)
    image_sz = args.image_sz or int(_hp_get(hp, "image_sz", 640))
    use_location = geo_dim > 0
    print(f"   model={ckpt_model_name} classes={num_classes} geo_dim={geo_dim} "
          f"image_sz={image_sz} current_T={cur_T}")
    model = build_model_from_state(state_dict, ckpt_model_name, num_classes, geo_dim)

    # --- 3. Rebuild the exact held-out validation split --------------------
    # Imported here (not at top) so --set-temperature stays DALI-free.
    from train_herbarium import HerbariumData

    sources = []
    for s in args.sources:
        csv, img = s.split(":", 1)
        sources.append((Path(csv), Path(img)))
    print(f"→ Rebuilding validation split from {len(sources)} source(s)…")
    data = HerbariumData(sources, label_level=label_level, hierarchical=hierarchical,
                         sparse_threshold=sparse, train_val_split=val_split,
                         seed=seed, max_per_species=max_per_sp)

    # Integrity check: same classes ⇒ same `combined` ⇒ same split.
    if list(data.nameslist) != list(ck_nameslist):
        msg = (f"Reconstructed class list ({len(data.nameslist)}) does not match "
               f"the checkpoint's embedded nameslist ({len(ck_nameslist)}). The "
               f"specsin data has likely changed since training, so the held-out "
               f"split cannot be reproduced exactly.")
        if not args.allow_nameslist_mismatch:
            sys.exit("ERROR: " + msg + "\n  Re-run with --allow-nameslist-mismatch to "
                     "fit on an approximate split anyway (T may be biased low).")
        print("  WARNING: " + msg)
    else:
        print(f"   integrity OK: {len(data.nameslist)} classes match the checkpoint.")

    valid_paths  = [Path(p) for p in data.valid_files]
    valid_labels = torch.tensor(data.valid_labels, dtype=torch.long)
    geo_coords   = data.valid_coords if use_location else None
    print(f"   held-out validation images: {len(valid_paths):,}")

    # --- 4. Collect raw logits (Space preprocessing) -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    ds = InferenceDataset(valid_paths, image_sz, geo_coords)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4,
                        pin_memory=True, shuffle=False)
    print(f"→ Running inference on {device}…")
    logits_all = []
    from tqdm import tqdm
    with torch.inference_mode():
        for x, _, geo in tqdm(loader, desc="val", unit="batch"):
            x = x.to(device)
            out = model(x, geo.to(device)) if use_location else model(x)
            logits_all.append(out.float().cpu())
    logits = torch.cat(logits_all)

    # --- 5. Fit T + report -------------------------------------------------
    before = calibration_report(logits, valid_labels, T=1.0)
    T = fit_temperature(logits, valid_labels)
    after = calibration_report(logits, valid_labels, T=T)

    print(f"\n{'='*56}\nCALIBRATION RESULT")
    print(f"  Fitted temperature : {T:.3f}")
    print(f"  {'metric':<12}{'T=1.0':>12}{'T=%.3f' % T:>12}")
    for k, label in [("mean_conf", "mean top-1 p"), ("ece", "ECE"),
                     ("nll", "NLL"), ("accuracy", "accuracy")]:
        print(f"  {label:<12}{before[k]:>12.4f}{after[k]:>12.4f}")
    print(f"  (accuracy is identical by construction — T never changes argmax)")
    print(f"{'='*56}")

    if not (0.05 < T < 100.0):
        print(f"  WARNING: fitted T={T:.3f} is out of the sane range; not uploading.")
        return 1

    # --- 6. Patch the Hub config.json --------------------------------------
    if args.dry_run or not args.repo:
        print("  (dry-run / no --repo — not uploading. Re-run with --repo to patch.)")
        return 0
    patch_hub_temperature(args.repo, T, token)
    print("  The Space applies it automatically on next model load "
          "(tap ⟳ or restart the Space).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
