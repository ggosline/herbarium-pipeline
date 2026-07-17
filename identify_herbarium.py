"""
Run inference with a trained herbarium checkpoint.

Two modes:
  1. Indets  — images where indet=True in specsin.  Sorts them into
               output_dir/indets/{predicted_species}/ by top prediction.
  2. Flagged — images where indet=False but the model's top prediction
               disagrees with the recorded label OR confidence < threshold.
               Marked as flagged=True in predictions.csv (no image copies).

A predictions CSV is saved to output_dir/predictions.csv.

Usage:
  python identify_herbarium.py \\
      --checkpoint  runs/ebenaceae/checkpoints/last.ckpt \\
      --nameslist   runs/ebenaceae/nameslist.json \\
      --sources     specsin.csv:images/  specsinAsia.csv:imagesAsia/ \\
      --output-dir  runs/ebenaceae/review/ \\
      --threshold   0.7 \\
      --image-sz    640
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset for inference
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class InferenceDataset(Dataset):
    """Flat list of image paths → (tensor, path_str, geo_vec).

    geo_coords: optional float32 Tensor of shape [N, 4] (sphere-encoded lat/lon).
    If None, a zero vector is returned for every sample.
    """

    def __init__(self, paths: list[Path], image_sz: int,
                 geo_coords: torch.Tensor | None = None):
        self.paths = paths
        self.geo   = geo_coords  # [N, 4] or None
        self.image_sz = image_sz
        self.transform = transforms.Compose([
            transforms.Resize(image_sz),
            transforms.CenterCrop(image_sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        geo  = self.geo[idx] if self.geo is not None else torch.zeros(4)
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), str(path), geo
        except Exception:
            blank = torch.zeros(3, self.image_sz, self.image_sz)
            return blank, str(path), geo


# ---------------------------------------------------------------------------
# Geo encoding + geo-capable model wrapper
# ---------------------------------------------------------------------------

def encode_coords(lat_vals, lon_vals) -> torch.Tensor:
    """Encode lat/lon sequences as (N, 4) sphere coordinates matching training encoding.

    Encoding: (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat), has_location)
    Invalid / missing coords produce an all-zero row (model trained to ignore them).
    """
    lat = pd.to_numeric(pd.Series(lat_vals), errors="coerce").values
    lon = pd.to_numeric(pd.Series(lon_vals), errors="coerce").values
    valid = np.isfinite(lat) & np.isfinite(lon)
    lat_r = np.where(valid, np.radians(lat), 0.0)
    lon_r = np.where(valid, np.radians(lon), 0.0)
    coords = np.stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
        valid.astype(np.float32),
    ], axis=1)
    return torch.from_numpy(coords.astype(np.float32))


def build_geo_index(df: pd.DataFrame, nameslist: list[str]) -> dict[int, np.ndarray]:
    """Build a per-species occurrence index from specsin lat/lon data.

    Returns a dict mapping class_index → float32 array [N, 2] of
    (lat_radians, lon_radians) for every georeferenced occurrence.
    Used for post-hoc geographic reranking of model predictions.
    """
    if "decimalLatitude" not in df.columns or "decimalLongitude" not in df.columns:
        return {}
    if "species" not in df.columns:
        return {}
    sp_to_idx = {sp: i for i, sp in enumerate(nameslist)}
    lat_num = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    lon_num = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    valid = lat_num.notna() & lon_num.notna()
    dv = df[valid].copy()
    dv["_lat"] = lat_num[valid].values
    dv["_lon"] = lon_num[valid].values
    geo_index: dict[int, np.ndarray] = {}
    for sp, grp in dv.groupby("species"):
        idx = sp_to_idx.get(str(sp))
        if idx is None:
            continue
        geo_index[idx] = np.radians(grp[["_lat", "_lon"]].values.astype(np.float32))
    return geo_index


def geo_rerank(
    topk_preds: list[list[int]],
    topk_probs: list[list[float]],
    df: pd.DataFrame,
    geo_index: dict[int, np.ndarray],
    geo_weight: float = 0.3,
    sigma_km: float = 500.0,
) -> tuple[list[list[int]], list[list[float]]]:
    """Rerank top-k predictions by blending model probability with a geographic prior.

    For each query specimen with valid lat/lon, computes a kernel density geo
    score for each candidate species based on its known occurrences in the
    training data:

        geo_score = mean(exp(-distance_km / sigma_km))

    Final score: (1 - geo_weight) * model_prob + geo_weight * geo_score

    Specimens with missing/invalid coordinates are returned unchanged.
    Setting geo_weight=0 or passing an empty geo_index is a no-op.
    """
    if not geo_index or geo_weight <= 0:
        return topk_preds, topk_probs

    lat_col = pd.to_numeric(df["decimalLatitude"], errors="coerce").values
    lon_col = pd.to_numeric(df["decimalLongitude"], errors="coerce").values

    new_preds: list[list[int]]   = []
    new_probs: list[list[float]] = []

    for i, (preds_k, probs_k) in enumerate(zip(topk_preds, topk_probs)):
        lat, lon = float(lat_col[i]), float(lon_col[i])
        if not (np.isfinite(lat) and np.isfinite(lon)):
            new_preds.append(preds_k)
            new_probs.append(probs_k)
            continue

        lat_r = np.radians(lat)
        lon_r = np.radians(lon)

        scores = []
        for pi, pr in zip(preds_k, probs_k):
            occ = geo_index.get(pi)
            if occ is not None and len(occ) > 0:
                dlat = occ[:, 0] - lat_r
                dlon = occ[:, 1] - lon_r
                a = (np.sin(dlat / 2) ** 2
                     + np.cos(lat_r) * np.cos(occ[:, 0]) * np.sin(dlon / 2) ** 2)
                d_km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                gs = float(np.mean(np.exp(-d_km / sigma_km)))
            else:
                gs = 0.0
            scores.append((1.0 - geo_weight) * pr + geo_weight * gs)

        order = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        new_preds.append([preds_k[o] for o in order])
        new_probs.append([scores[o] for o in order])

    return new_preds, new_probs


class _GeoModel(nn.Module):
    """Backbone + geo MLP + head assembled from separately loaded weights."""

    def __init__(self, backbone: nn.Module, geo_mlp: nn.Module,
                 head: nn.Module, geo_dim: int):
        super().__init__()
        self.backbone = backbone
        self.geo_mlp  = geo_mlp
        self.head     = head
        self.geo_dim  = geo_dim

    def forward(self, x, geo=None):
        feats = self.backbone(x)
        if geo is None:
            geo = torch.zeros(feats.shape[0], 4, device=feats.device)
        geo_feats = self.geo_mlp(geo)
        return self.head(torch.cat([feats, geo_feats], dim=1))


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def resolve_checkpoint(path: Path) -> Path:
    """If path is a directory, pick the best checkpoint in it.

    Preference order:
      1. highest val_Accuracy among ``acc-*`` checkpoints (what we train for);
      2. lowest valid_loss among ``epoch=*`` checkpoints;
      3. most recently modified, as a last resort.

    Stage-1 checkpoints (``s1-*``) are never auto-selected: they are
    frozen-backbone intermediates at roughly half the final accuracy.

    This used to be "most recently modified", which is unsafe — mtime records
    when a file was last *written*, not how good it is. Anything that rewrites a
    checkpoint in place (embedding a temperature, say) reorders the ranking, and
    a stage-1 checkpoint could win. Nothing in the output would have said so.
    """
    if not path.is_dir():
        return path

    ckpts = [p for p in path.glob("*.ckpt") if not p.name.startswith("s1-")]
    if not ckpts:
        raise FileNotFoundError(f"No usable .ckpt files found in {path}")

    def _metric(p: Path, key: str) -> float | None:
        m = re.search(rf"{key}=([0-9.]+)", p.name)
        if not m:
            return None
        try:
            return float(m.group(1).rstrip("."))
        except ValueError:
            return None

    by_acc = [(v, p) for p in ckpts if (v := _metric(p, "val_Accuracy")) is not None]
    if by_acc:
        v, chosen = max(by_acc, key=lambda t: t[0])
        print(f"  Auto-selected checkpoint: {chosen.name} (best val_Accuracy={v})")
        return chosen

    by_loss = [(v, p) for p in ckpts if (v := _metric(p, "valid_loss")) is not None]
    if by_loss:
        v, chosen = min(by_loss, key=lambda t: t[0])
        print(f"  Auto-selected checkpoint: {chosen.name} (lowest valid_loss={v})")
        return chosen

    chosen = max(ckpts, key=lambda p: p.stat().st_mtime)
    print(f"  Auto-selected checkpoint: {chosen.name} (most recent; no metric in "
          f"any filename)")
    return chosen


def derive_class_counts(df_all: pd.DataFrame, nameslist: list[str]) -> list[int]:
    """Rebuild the per-class training image counts, aligned with nameslist.

    Only needed for checkpoints saved before train_herbarium embedded them.
    Reproduces HerbariumData's trainable-row mask (not indet, has a file, not
    outlier/invalid) and then counts at whichever rank the nameslist is in —
    detected by seeing which column's values actually cover the class names,
    rather than trusting the checkpoint's label_level, which is unreliable on
    hierarchical runs.

    Returns [] if the rank cannot be identified, in which case the caller skips
    the adjustment rather than applying a wrong correction.
    """
    if not nameslist or "species" not in df_all.columns:
        return []

    d = df_all
    if "indet" in d.columns:
        d = d[~d["indet"].astype(str).str.lower().isin(("true", "1"))]
    for col in ("outlier", "invalid"):
        if col in d.columns:
            d = d[~d[col].astype(str).str.lower().isin(("true", "1"))]

    sp = d["species"].astype(str)
    candidates = {"species": sp, "genus": sp.str.split().str[0]}
    if "family" in d.columns:
        candidates["family"] = d["family"].astype(str)

    want = set(nameslist)
    best_rank, best_cov, best_counts = None, 0.0, None
    for rank, series in candidates.items():
        vc = series.value_counts()
        cov = sum(1 for n in nameslist if n in vc.index) / len(nameslist)
        if cov > best_cov:
            best_rank, best_cov, best_counts = rank, cov, vc

    # Require near-total coverage: a partial match means we guessed the rank wrong,
    # and a wrong correction is worse than none.
    if best_counts is None or best_cov < 0.95:
        print(f"  [warn] could not match nameslist to a rank in specsin "
              f"(best: {best_rank} at {100 * best_cov:.0f}% coverage)")
        return []
    missing = len(want) - sum(1 for n in nameslist if n in best_counts.index)
    print(f"  Matched nameslist to '{best_rank}' ({100 * best_cov:.1f}% coverage"
          + (f", {missing} classes absent from specsin → count 1" if missing else "") + ")")
    # Absent classes get 1 (not 0): log(1)=0, i.e. no adjustment, and it keeps log() finite.
    return [int(best_counts.get(n, 1)) for n in nameslist]


def _infer_label_level(nameslist: list[str], hierarchical: bool, stored: str) -> str:
    """The rank the class indices actually refer to.

    The checkpoint's stored label_level cannot be trusted. In hierarchical mode
    train_herbarium always indexes by species *regardless of --label-level*, so a
    run configured with `--label-level family --hierarchical` still produces a
    species nameslist while recording "family". Believing that writes species
    binomials into pred_family and leaves pred_genus empty — the whole
    predictions CSV is then garbage.

    The nameslist is the ground truth, so read it: binomials contain a space.
    """
    if hierarchical:
        return "species"
    if nameslist and sum(" " in str(n) for n in nameslist) > len(nameslist) / 2:
        return "species"
    if stored in ("genus", "family"):
        return stored
    return "species"


class _DualHead(nn.Module):
    """Wraps an inference model so it also emits genus logits from the same features.

    The genus head is a separately trained classifier, not a projection of the
    species prediction, and it is materially better at genus: 93% vs ~88% on the
    Rubiaceae run, because it is optimised for genus directly and does not have to
    get the species right first. identify used to throw it away.
    """

    def __init__(self, base: nn.Module, genus_head: nn.Module):
        super().__init__()
        self.base = base
        self.genus_head = genus_head

    def _shared_features(self, x, geo):
        if isinstance(self.base, _GeoModel):
            feats = self.base.backbone(x)
            if geo is None:
                geo = torch.zeros(feats.shape[0], 4, device=feats.device)
            return torch.cat([feats, self.base.geo_mlp(geo)], dim=1)
        # Plain timm classifier: pooled pre-logit features.
        return self.base.forward_head(self.base.forward_features(x), pre_logits=True)

    def forward(self, x, geo=None):
        z = self._shared_features(x, geo)
        head = self.base.head if isinstance(self.base, _GeoModel) else self.base.get_classifier()
        return head(z), self.genus_head(z)


def build_genus_head(genus_head_sd: dict, feat_dim: int) -> nn.Module:
    """Rebuild the trained genus classifier from its saved weights."""
    n_genus, in_dim = genus_head_sd["weight"].shape
    head = nn.Linear(in_dim, n_genus)
    head.load_state_dict(genus_head_sd)
    if in_dim != feat_dim:
        raise SystemExit(
            f"ERROR: genus head expects {in_dim} input features but the model "
            f"produces {feat_dim}. Checkpoint is inconsistent.")
    return head.eval()


def load_model(checkpoint_path: Path, nameslist: list[str], image_sz: int):
    """Load a TimmModel from a Lightning checkpoint.

    Returns (state_dict, model_name, num_classes, nameslist, geo_dim,
    label_level, temperature, excluded, class_counts, genus_head, split).
    nameslist may be updated from the checkpoint if embedded there.
    excluded is {"rank": str, "taxa": {name: n_images}} — taxa the training
    run dropped as too sparse (empty for older checkpoints).
    geo_dim > 0 indicates the checkpoint uses a geo MLP; state_dict will
    contain geo_mlp.* keys in addition to backbone internals and head.*.
    temperature is the fitted softmax calibration temperature (Guo et al.
    2017); 1.0 for older checkpoints that were never calibrated.
    split is {"train": [fname...], "valid": [fname...]} — which specimens the
    model actually saw. Empty for checkpoints written before this was recorded;
    consumers must then treat every metric as train-contaminated.
    """
    num_classes = len(nameslist)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Which specimens the model trained on. It memorises them (100% accurate,
    # mean confidence 0.9985, vs 88.7% / 0.887 held out), so any metric that
    # averages them in reads far too well.
    split = ckpt.get("split") or {}
    if split.get("valid"):
        print(f"  Split embedded: {len(split.get('train', [])):,} train / "
              f"{len(split['valid']):,} held-out")

    # Extract nameslist embedded by on_save_checkpoint (preferred over external file)
    genus_nameslist: list[str] = []
    if "nameslist" in ckpt:
        embedded = ckpt["nameslist"]
        if isinstance(embedded, dict):
            nameslist = embedded.get("species") or max(embedded.values(), key=len)
            genus_nameslist = list(embedded.get("genus") or [])
        else:
            nameslist = embedded
        num_classes = len(nameslist)
        print(f"  Nameslist loaded from checkpoint ({num_classes} classes)")
    # Taxa the training run dropped as too sparse — embedded by on_save_checkpoint.
    # Older checkpoints won't have it; default to an empty payload.
    excluded = ckpt.get("excluded_species") or {"rank": "species", "taxa": {}}
    # Training images per class, aligned with nameslist (empty for older
    # checkpoints). Used by --logit-adjust to undo class weighting post-hoc.
    class_counts = list(ckpt.get("class_counts") or [])
    if class_counts and len(class_counts) != num_classes:
        print(f"  [warn] class_counts length {len(class_counts)} != {num_classes} classes "
              f"— ignoring (cannot align for logit adjustment)")
        class_counts = []
    state_dict = ckpt["state_dict"]

    # Strip Lightning / torch.compile prefixes.
    # TimmModel wraps timm as self.model, so non-hierarchical keys are
    # model.model.* (or model._orig_mod.model.*) — check longer prefixes first.
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ("model._orig_mod.model.", "model._orig_mod.",
                       "model.model.",            "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        cleaned[key] = v

    # Hierarchical / geo model: backbone.* + head_species.* / head_genus.* / head_family.*
    # Remap so it matches a standard single-head timm model:
    #   backbone.*     → *        (backbone weights match timm internals)
    #   head_species.* → head.*   (species head maps to timm's classification head)
    #   geo_mlp.*      → geo_mlp.* (preserved for geo-capable inference)
    # A head_species/head_genus pair means the run was --hierarchical, which pins
    # the class index to species no matter what --label-level said.
    hierarchical = any(k.startswith(("head_species.", "head_genus.")) for k in cleaned)
    genus_head_sd: dict = {}
    if any(k.startswith("backbone.") for k in cleaned):
        print("  (hierarchical/geo checkpoint detected; remapping backbone/head_species keys)")
        remapped = {}
        for k, v in cleaned.items():
            if k.startswith("backbone."):
                remapped[k[len("backbone."):]] = v
            elif k.startswith("head_species."):
                remapped["head." + k[len("head_species."):]] = v
            elif k.startswith("geo_mlp."):
                remapped[k] = v  # preserve geo MLP weights
            elif k.startswith("head_genus."):
                # Kept, not discarded: a head trained directly on genus beats
                # taking the first word of the species head's guess.
                genus_head_sd[k[len("head_genus."):]] = v
            elif k.startswith("head."):
                # Non-hierarchical geo checkpoint: head.* is already the
                # species classifier, no rename needed.
                remapped[k] = v
            # head_family still dropped — it is a coarser view of the same
            # features, and pred_family is looked up from the predicted taxon.
        cleaned = remapped

    # Detect geo_dim from geo_mlp weights (geo_mlp.0 is Linear(4, geo_dim))
    geo_dim = 0
    if "geo_mlp.0.weight" in cleaned:
        geo_dim = cleaned["geo_mlp.0.weight"].shape[0]
        print(f"  Geo MLP detected (geo_dim={geo_dim})")

    # Detect num_classes from the checkpoint head weight (ground truth)
    ckpt_num_classes = None
    for k, v in cleaned.items():
        if k in ("head.weight", "head_species.weight"):
            ckpt_num_classes = v.shape[0]
            break
    if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
        print(f"  WARNING: nameslist has {num_classes} classes but checkpoint head has "
              f"{ckpt_num_classes} — using checkpoint size. "
              f"Ensure your nameslist matches the training run.")
        num_classes = ckpt_num_classes

    # Detect model name from checkpoint hyper-params if available
    hparams = ckpt.get("hyper_parameters", {})
    model_name = hparams.get("model_name") or hparams.get("config", {}).get("model_name")
    if not model_name:
        model_name = None  # caller must pass --model if needed

    # What rank does this model classify at? Drives which columns get filled in
    # the predictions CSV. Inferred from the nameslist, NOT taken on trust — see
    # _infer_label_level for why the stored value lies on hierarchical runs.
    stored_level = (hparams.get("label_level")
                    or hparams.get("config", {}).get("label_level")
                    or "species")
    label_level = _infer_label_level(nameslist, hierarchical, stored_level)
    if label_level != stored_level:
        print(f"  NOTE: checkpoint records label_level='{stored_level}', but its class "
              f"names are {label_level} — trusting the names. "
              + ("(--hierarchical always indexes by species, whatever --label-level said.)"
                 if hierarchical else ""))

    genus_head = {"nameslist": genus_nameslist, "state_dict": genus_head_sd} \
        if (genus_nameslist and genus_head_sd) else None
    if genus_head:
        print(f"  Genus head found ({len(genus_nameslist)} genera) — will predict genus "
              f"directly instead of deriving it from the species head")

    # Softmax calibration temperature fitted at the end of training.
    # Absent on older checkpoints → 1.0 (no rescaling, original behaviour).
    try:
        temperature = float(ckpt.get("temperature", 1.0)) or 1.0
    except (TypeError, ValueError):
        temperature = 1.0
    if temperature != 1.0:
        print(f"  Calibration temperature: {temperature:.3f}")

    return (cleaned, model_name, num_classes, nameslist, geo_dim, label_level,
            temperature, excluded, class_counts, genus_head, split)


def _ckpt_embed_dim(state_dict: dict) -> int | None:
    """Backbone embedding dim recorded in the checkpoint, or None if unknown.

    Read from a transformer LayerNorm (shape == embed_dim). Lets us catch a
    wrong --model before load_state_dict throws a wall of shape mismatches.
    """
    for key in ("norm.weight", "blocks.0.norm1.weight",
                "backbone.norm.weight", "backbone.blocks.0.norm1.weight"):
        t = state_dict.get(key)
        if t is not None and t.ndim == 1:
            return int(t.shape[0])
    return None


def _check_arch(state_dict: dict, model_name: str, feat_dim: int) -> None:
    """Fail fast with a readable message when --model disagrees with the
    checkpoint's actual backbone (e.g. vit_base checkpoint, vit_large --model)."""
    ckpt_dim = _ckpt_embed_dim(state_dict)
    if ckpt_dim is not None and ckpt_dim != feat_dim:
        raise SystemExit(
            f"ERROR: architecture mismatch — the checkpoint's backbone embed "
            f"dim is {ckpt_dim}, but --model '{model_name}' has {feat_dim}. "
            f"This checkpoint was trained with a different backbone (e.g. a "
            f"vit_base checkpoint can't load into vit_large). Pass the --model "
            f"that matches this checkpoint, or point --checkpoint at one built "
            f"with '{model_name}'."
        )


def build_model_from_state(state_dict: dict, model_name: str, num_classes: int,
                           geo_dim: int) -> nn.Module:
    """Reconstruct the inference model from a cleaned state_dict.

    Mirrors the architecture in train_herbarium: a plain timm classifier, or a
    backbone + geo_mlp + head geo model when geo_dim > 0. Returns an eval()
    model on CPU. Shared by identify() and the calibration script so both
    reconstruct weights identically.
    """
    if geo_dim:
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feat_dim = backbone.num_features
        _check_arch(state_dict, model_name, feat_dim)
        geo_mlp = nn.Sequential(
            nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim)
        )
        head = nn.Linear(feat_dim + geo_dim, num_classes)

        backbone_sd = {k: v for k, v in state_dict.items()
                       if not k.startswith(("geo_mlp.", "head."))}
        geo_mlp_sd  = {k[len("geo_mlp."):]: v for k, v in state_dict.items()
                       if k.startswith("geo_mlp.")}
        head_sd     = {k[len("head."):]: v for k, v in state_dict.items()
                       if k.startswith("head.")}

        missing, _ = backbone.load_state_dict(backbone_sd, strict=False)
        if missing:
            print(f"  WARNING: backbone missing keys: {missing[:5]}")
        geo_mlp.load_state_dict(geo_mlp_sd)
        head.load_state_dict(head_sd)
        model = _GeoModel(backbone, geo_mlp, head, geo_dim)
        print(f"  Geo-capable model built (feat_dim={feat_dim}, geo_dim={geo_dim})")
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        _check_arch(state_dict, model_name, model.num_features)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}")
        if missing:
            print(f"  WARNING: missing keys — weights not loaded for: {missing[:5]}")
    return model.eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(
    model: nn.Module,
    paths: list[Path],
    image_sz: int,
    batch_size: int,
    device: torch.device,
    top_k: int = 5,
    geo_coords: torch.Tensor | None = None,
    temperature: float = 1.0,
    logit_adjust: float = 0.0,
    class_counts: list[int] | None = None,
) -> tuple[list[list[int]], list[list[float]], list[int], list[float]]:
    """Return (top_k_indices, top_k_probs, genus_idx, genus_prob) for each path.

    The last two are empty lists unless the model is a _DualHead carrying a
    trained genus head, in which case they are that head's top-1 genus and its
    probability — more accurate than taking the first word of the species guess.

    geo_coords: optional float32 Tensor [N, 4] aligned with paths.
    Passed to the model when provided (geo-capable checkpoints).
    temperature: divides the logits before softmax (temperature scaling,
    Guo et al. 2017). T>1 softens over-confident predictions; T=1 is a
    no-op. Does not change the ranking, only the reported probabilities.

    logit_adjust (tau): adds tau * log(class_counts) to the logits, which DOES
    change the ranking. Training with class weights w_c makes the network learn
        p_model(c|x) ∝ w_c · p_true(c|x)
    so with w_c ∝ (1/n_c)**beta the fitted logits carry a -beta*log(n_c) bias
    toward rare classes. Adding tau*log(n_c) back cancels it exactly at
    tau = beta, restoring predictions under the true class prior. Set tau to the
    --class-weight-beta the model was trained with (1.0 for checkpoints from
    before the weighting was made configurable — those were hardcoded to full
    inverse-frequency). tau = 0 leaves the model as trained.
    """
    ds = InferenceDataset(paths, image_sz, geo_coords)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=False)
    model.eval().to(device)

    log_counts = None
    if logit_adjust and class_counts:
        log_counts = torch.log(
            torch.tensor(class_counts, dtype=torch.float32, device=device).clamp(min=1.0)
        )

    all_topk_preds, all_topk_probs = [], []
    all_genus_preds, all_genus_probs = [], []
    for batch_tensors, _, batch_geo in tqdm(loader, desc="Inferring", unit="batch"):
        batch_tensors = batch_tensors.to(device)
        geo_arg = batch_geo.to(device) if geo_coords is not None else None
        out = model(batch_tensors, geo_arg) if geo_coords is not None else model(batch_tensors)
        # _DualHead returns (species_logits, genus_logits); a plain model returns logits.
        genus_logits = None
        if isinstance(out, tuple):
            logits, genus_logits = out
        else:
            logits = out
        # Undo the training-time class weighting before any calibration, since
        # that bias lives in the raw logits.
        if log_counts is not None:
            logits = logits + logit_adjust * log_counts
        if temperature != 1.0:
            logits = logits / temperature
        probs  = torch.softmax(logits, dim=1)
        k = min(top_k, probs.shape[1])
        topk_probs, topk_preds = torch.topk(probs, k=k, dim=1)
        all_topk_preds.extend(topk_preds.cpu().tolist())
        all_topk_probs.extend(topk_probs.cpu().tolist())

        if genus_logits is not None:
            # The genus head is trained without class weighting (train_herbarium
            # only weights the species criterion), so no logit adjustment here.
            if temperature != 1.0:
                genus_logits = genus_logits / temperature
            g_probs = torch.softmax(genus_logits, dim=1)
            # Top-k, not top-1: the genus head is the model's most accurate
            # output (~97% vs ~89% at species), so it deserves a ranked list of
            # its own rather than being reduced to a single label. Note there
            # are far fewer genera than species, so k may be clamped.
            gk = min(top_k, g_probs.shape[1])
            g_top_p, g_top_i = torch.topk(g_probs, k=gk, dim=1)
            all_genus_preds.extend(g_top_i.cpu().tolist())
            all_genus_probs.extend(g_top_p.cpu().tolist())

    # Genus lists are empty unless the model carries a genus head. When present
    # each element is a top-k list, parallel to all_topk_preds/probs.
    return all_topk_preds, all_topk_probs, all_genus_preds, all_genus_probs


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------

def copy_image(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def identify(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint (auto-pick latest if a directory was given)
    checkpoint_path = resolve_checkpoint(Path(args.checkpoint))

    # Load nameslist — from external file if provided, else will be read from checkpoint
    nameslist: list[str] = []
    if args.nameslist:
        nameslist_raw = json.loads(Path(args.nameslist).read_text())
        if isinstance(nameslist_raw, dict):
            nameslist = nameslist_raw.get("species", []) or max(nameslist_raw.values(), key=len)
            print(f"  (hierarchical nameslist detected; using species list)")
        else:
            nameslist = nameslist_raw
        print(f"Loaded {len(nameslist)} class names from {args.nameslist}")

    # Load model weights (may update nameslist + num_classes from embedded data)
    (state_dict, ckpt_model_name, num_classes, nameslist, geo_dim, label_level,
     ckpt_temperature, excluded, class_counts, genus_head, ckpt_split) = load_model(
        checkpoint_path, nameslist, args.image_sz
    )
    print(f"  Model rank: {label_level}")
    if genus_head and args.no_genus_head:
        print("  --no-genus-head: ignoring the trained genus head, "
              "deriving genus from the species prediction instead")
        genus_head = None

    # Cancel the class weighting the model was trained with (see run_inference).
    # class_counts may be empty on checkpoints predating their being embedded —
    # they are re-derived from specsin below, once df_all is loaded.
    logit_adjust = float(args.logit_adjust or 0.0)

    # Tell the end user which taxa the model can't predict (dropped as too
    # sparse at train time). Write a sidecar into the review dir so the webui /
    # Space can show it, and print a short banner here.
    excluded_taxa = (excluded or {}).get("taxa", {})
    if excluded_taxa:
        excl_rank = (excluded or {}).get("rank", "species")
        out_json = output_dir / "excluded_species.json"
        out_json.write_text(json.dumps(excluded, indent=2))
        # Human-readable CSV alongside predictions.csv (rarest first).
        excl_rows = sorted(excluded_taxa.items(), key=lambda kv: kv[1])
        pd.DataFrame(excl_rows, columns=[excl_rank, "n_images"]).to_csv(
            output_dir / "excluded_species.csv", index=False)
        preview = ", ".join(n for n, _ in excl_rows[:10])
        print(f"\n  NOTE: {len(excluded_taxa)} {excl_rank} had too few images to "
              f"train and are NOT in this model — specimens of these will be "
              f"mis-assigned to the nearest trained class.")
        print(f"        {preview}{' …' if len(excluded_taxa) > 10 else ''}")
        print(f"        Full list → {out_json.name} / excluded_species.csv\n")

    # CLI --temperature overrides the value fitted at training time; otherwise
    # use the checkpoint's (1.0 for uncalibrated checkpoints).
    temperature = args.temperature if args.temperature is not None else ckpt_temperature
    if temperature <= 0:
        temperature = 1.0
    if temperature != 1.0:
        print(f"  Applying softmax temperature: {temperature:.3f}")
    if not nameslist:
        print("ERROR: no nameslist found. Pass --nameslist or use a checkpoint from a recent run.")
        sys.exit(1)

    model_name = args.model or ckpt_model_name
    if not model_name:
        print("ERROR: cannot determine model architecture. Pass --model <timm_model_name>")
        sys.exit(1)

    print(f"Building model: {model_name}  ({num_classes} classes)")

    base_model = build_model_from_state(state_dict, model_name, num_classes, geo_dim)

    # Attach the trained genus head so genus is predicted directly rather than
    # read off the first word of the species guess.
    genus_nameslist: list[str] = []
    if genus_head:
        feat_dim = (base_model.head.in_features if isinstance(base_model, _GeoModel)
                    else base_model.get_classifier().in_features)
        base_model = _DualHead(base_model,
                               build_genus_head(genus_head["state_dict"], feat_dim)).eval()
        genus_nameslist = genus_head["nameslist"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and merge specsin sources
    frames = []
    for src in args.sources:
        specsin_path, img_dir = src.split(":", 1)
        specsin_path = Path(specsin_path)
        img_dir      = Path(img_dir)
        df = pd.read_csv(specsin_path)
        df["abs_path"]     = df["fname"].apply(lambda f: str(img_dir / f))
        df["img_dir"]      = str(img_dir)
        df["specsin_file"] = str(specsin_path)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["hasfile"].astype(str).str.lower().isin(("true", "1"))]
    # Defensive: hasfile may be stale (resize failures, manual deletes). Drop rows
    # whose image is actually missing on disk so workers don't crash mid-batch.
    exists = df_all["abs_path"].apply(lambda p: Path(p).is_file())
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"  WARNING: {n_missing} rows have hasfile=True but file is missing on disk — skipping")
    df_all = df_all[exists].copy()
    print(f"Total images with files: {len(df_all):,}")

    # Logit adjustment needs one training-image count per class. Newer checkpoints
    # embed them; for older ones, rebuild from specsin (same rows train_herbarium
    # would have used) so the fix works without retraining.
    if logit_adjust and not class_counts:
        class_counts = derive_class_counts(df_all, nameslist)
        if class_counts:
            print(f"  Class counts re-derived from specsin ({len(class_counts)} classes) "
                  f"— checkpoint predates embedded counts")
    if logit_adjust and not class_counts:
        print(f"  [warn] --logit-adjust {logit_adjust} requested but per-class counts could not "
              f"be determined — skipping adjustment.")
        logit_adjust = 0.0
    elif logit_adjust:
        lo, hi = min(class_counts), max(class_counts)
        print(f"  Logit adjustment tau={logit_adjust}: rebalancing toward the true class prior "
              f"(class sizes {lo}–{hi}; up to "
              f"{logit_adjust * math.log(max(hi, 1) / max(lo, 1)):.1f} logits of correction)")

    # Build species → family and genus → family lookups from specsin metadata
    # (may be absent in older CSVs). Used to fill pred_family when the model
    # only outputs species or genus directly.
    species_to_family: dict[str, str] = {}
    genus_to_family:   dict[str, str] = {}
    if "family" in df_all.columns and "species" in df_all.columns:
        for sp, fam in zip(df_all["species"], df_all["family"]):
            if sp and fam and str(sp) not in ("nan", "") and str(fam) not in ("nan", ""):
                species_to_family[str(sp)] = str(fam)
                genus_to_family.setdefault(str(sp).split()[0], str(fam))

    def _genus_name(idx: int | None) -> str:
        if idx is None or not genus_nameslist or idx >= len(genus_nameslist):
            return ""
        return genus_nameslist[idx]

    def _level_columns(pred_name: str, genus_k: list | None = None,
                       genus_p: list | None = None) -> dict:
        """Map a class-index name (whatever rank the model predicts at) to
        the right pred_* / true_*-companion columns.

        When the checkpoint carries a trained genus head, pred_genus comes from
        that head rather than from the first word of the species guess — it is
        optimised for genus directly and is measurably better at it. The two can
        disagree; genus_agrees records that, and a disagreement is a useful
        "this specimen is odd" signal.

        genus_k / genus_p are the genus head's top-k index / probability lists;
        this uses only the head of each.
        """
        if label_level == "family":
            return {"pred_species": "", "pred_genus": "",
                    "pred_family":  pred_name}
        if label_level == "genus":
            return {"pred_species": "", "pred_genus": pred_name,
                    "pred_family":  genus_to_family.get(pred_name, "")}
        # species
        from_species = pred_name.split()[0] if pred_name else ""
        cols = {"pred_species": pred_name,
                "pred_genus":   from_species,
                "pred_family":  species_to_family.get(pred_name, "")}
        if genus_nameslist and genus_k:
            g = _genus_name(genus_k[0])
            cols["pred_genus"]    = g
            cols["genus_conf"]    = round(float((genus_p or [0.0])[0]), 4)
            cols["genus_agrees"]  = (g == from_species)
            cols["pred_family"]   = (species_to_family.get(pred_name, "")
                                     or genus_to_family.get(g, ""))
        return cols

    def _genus_topk_columns(genus_k: list | None, genus_p: list | None) -> dict:
        """gtop{k}_name / gtop{k}_prob — the genus head's own ranked list.

        Distinct from top{k}_genus, which is the genus *of* the k-th species
        guess. The two answer different questions: top{k}_genus marginalises the
        species head (and only over its top 5), whereas these come from the head
        actually trained to predict genus.
        """
        if not (genus_nameslist and genus_k):
            return {}
        out: dict = {}
        for k, (gi, gp) in enumerate(zip(genus_k, genus_p or []), 1):
            out[f"gtop{k}_name"] = _genus_name(gi)
            out[f"gtop{k}_prob"] = round(float(gp), 4)
        return out

    def _topk_columns(preds_k, probs_k) -> dict:
        """Per-rank top-k columns mirror the pred_* convention.
        - top{k}_name kept for back-compat (always the class-index name).
        - top{k}_family / top{k}_genus added so Analysis can compute top-5
          accuracy at the model's actual rank.
        """
        out: dict = {}
        for k, (pi, pr) in enumerate(zip(preds_k, probs_k), 1):
            name = nameslist[pi] if pi < len(nameslist) else "unknown"
            out[f"top{k}_name"] = name
            out[f"top{k}_prob"] = round(pr, 4)
            if label_level == "family":
                out[f"top{k}_family"] = name
            elif label_level == "genus":
                out[f"top{k}_genus"]  = name
                out[f"top{k}_family"] = genus_to_family.get(name, "")
            else:
                out[f"top{k}_family"] = species_to_family.get(name, "")
                out[f"top{k}_genus"]  = name.split()[0] if name else ""
        return out

    # Build geographic occurrence index for post-hoc reranking
    geo_index: dict[int, np.ndarray] = {}
    if args.geo_weight > 0:
        geo_index = build_geo_index(df_all, nameslist)
        if geo_index:
            n_spp = len(geo_index)
            n_occ = sum(len(v) for v in geo_index.values())
            print(f"  Geo index built: {n_spp} species, {n_occ:,} occurrences "
                  f"(weight={args.geo_weight}, sigma={args.geo_sigma} km)")
        else:
            print("  NOTE: --geo-weight > 0 but no lat/lon data found in specsin — skipping geo reranking")

    # Split into indets and identified
    df_indet = df_all[df_all["indet"].astype(str).str.lower().isin(("true", "1"))].copy()
    df_ident = df_all[~df_all["indet"].astype(str).str.lower().isin(("true", "1"))].copy()
    print(f"  Indeterminate: {len(df_indet):,}")
    print(f"  Identified:    {len(df_ident):,}")

    # Encode lat/lon from specsin when the model was trained with geo features
    has_coords = (geo_dim > 0
                  and "decimalLatitude"  in df_all.columns
                  and "decimalLongitude" in df_all.columns)
    if geo_dim and not has_coords:
        print("  NOTE: geo checkpoint but no lat/lon columns in specsin — inference without geo")

    def _geo_for(df) -> torch.Tensor | None:
        if not has_coords:
            return None
        return encode_coords(df["decimalLatitude"].values, df["decimalLongitude"].values)

    results = []

    # --- Indets ---
    if len(df_indet) > 0:
        print(f"\nRunning inference on {len(df_indet):,} indeterminate specimens...")
        indet_paths = [Path(p) for p in df_indet["abs_path"]]
        topk_preds, topk_probs, g_preds, g_probs = run_inference(
            base_model, indet_paths, args.image_sz,
            args.batch_size, device,
            geo_coords=_geo_for(df_indet),
            temperature=temperature,
            logit_adjust=logit_adjust,
            class_counts=class_counts)
        topk_preds, topk_probs = geo_rerank(topk_preds, topk_probs, df_indet,
                                             geo_index, args.geo_weight, args.geo_sigma)
        g_preds = g_preds or [None] * len(topk_preds)
        g_probs = g_probs or [None] * len(topk_preds)
        for row, preds_k, probs_k, gi, gp in zip(df_indet.itertuples(), topk_preds,
                                                 topk_probs, g_preds, g_probs):
            pred_name = nameslist[preds_k[0]] if preds_k[0] < len(nameslist) else "unknown"
            entry = {
                "fname":          row.fname,
                "abs_path":       row.abs_path,
                "specsin_file":   row.specsin_file,
                "source":         row.img_dir,
                "gbifID":         str(getattr(row, "gbifID", "") or ""),
                "image_url":      str(getattr(row, "image_url", "") or ""),
                "decimalLatitude":  getattr(row, "decimalLatitude",  ""),
                "decimalLongitude": getattr(row, "decimalLongitude", ""),
                "true_species":   "",
                "true_genus":     "",
                "true_family":    getattr(row, "family", "") or "",
                "sparse":         str(getattr(row, "sparse", "")).lower() in ("true", "1"),
                **_level_columns(pred_name, gi, gp),
                "confidence":     round(probs_k[0], 4),
                "indet":          True,
                "flagged":        False,
                **_topk_columns(preds_k, probs_k),
                **_genus_topk_columns(gi, gp),
            }
            results.append(entry)
        print(f"  → Wrote {len(df_indet):,} indet predictions to predictions.csv")

    # --- Identified: flag disagreements ---
    if len(df_ident) > 0:
        print(f"\nRunning inference on {len(df_ident):,} identified specimens...")
        ident_paths = [Path(p) for p in df_ident["abs_path"]]
        topk_preds, topk_probs, g_preds, g_probs = run_inference(
            base_model, ident_paths, args.image_sz,
            args.batch_size, device,
            geo_coords=_geo_for(df_ident),
            temperature=temperature,
            logit_adjust=logit_adjust,
            class_counts=class_counts)
        topk_preds, topk_probs = geo_rerank(topk_preds, topk_probs, df_ident,
                                             geo_index, args.geo_weight, args.geo_sigma)
        g_preds = g_preds or [None] * len(topk_preds)
        g_probs = g_probs or [None] * len(topk_preds)
        flagged_count = 0
        for row, preds_k, probs_k, gi, gp in zip(df_ident.itertuples(), topk_preds,
                                                 topk_probs, g_preds, g_probs):
            pred_name     = nameslist[preds_k[0]] if preds_k[0] < len(nameslist) else "unknown"
            conf          = probs_k[0]
            true_species  = getattr(row, "species", "")
            true_family   = getattr(row, "family",  "") or ""
            true_genus    = (true_species.split()[0] if true_species else "")
            # Mismatch is evaluated at the rank the model actually predicts:
            # for a family model, "Rosaceae" vs true_family — comparing it
            # against true_species would always flag everything.
            if label_level == "family":
                true_at_rank = true_family
            elif label_level == "genus":
                true_at_rank = true_genus
            else:
                true_at_rank = true_species
            mismatch = (pred_name != true_at_rank) and (conf >= args.threshold)
            low_conf = conf < args.low_conf_threshold if args.low_conf_threshold > 0 else False
            flagged  = mismatch or low_conf

            if flagged:
                flagged_count += 1

            entry = {
                "fname":          row.fname,
                "abs_path":       row.abs_path,
                "specsin_file":   row.specsin_file,
                "source":         row.img_dir,
                "gbifID":         str(getattr(row, "gbifID", "") or ""),
                "image_url":      str(getattr(row, "image_url", "") or ""),
                "decimalLatitude":  getattr(row, "decimalLatitude",  ""),
                "decimalLongitude": getattr(row, "decimalLongitude", ""),
                "true_species":   true_species,
                "true_genus":     true_genus,
                "true_family":    true_family,
                "sparse":         str(getattr(row, "sparse", "")).lower() in ("true", "1"),
                **_level_columns(pred_name, gi, gp),
                "confidence":     round(conf, 4),
                "indet":          False,
                "flagged":        flagged,
                **_topk_columns(preds_k, probs_k),
                **_genus_topk_columns(gi, gp),
            }
            results.append(entry)

        print(f"  → Flagged {flagged_count:,} specimens (see flagged=True in predictions.csv)")

    # Save predictions CSV
    results_df = pd.DataFrame(results)

    # Mark which specimens the model trained on, so nobody has to reconstruct the
    # split to get an honest number. A model scores 100% on its own training
    # images; averaging those in with held-out ones inflates every accuracy and
    # every novelty score. Anything not in the split was dropped by the sparse
    # filter — it is neither train nor held-out, it is a taxon the model cannot
    # predict at all, which is a third thing again.
    train_f = set(ckpt_split.get("train") or [])
    valid_f = set(ckpt_split.get("valid") or [])
    if valid_f:
        base = results_df["fname"].map(lambda p: Path(str(p)).name)
        results_df["split"] = np.where(
            base.isin(train_f), "train",
            np.where(base.isin(valid_f), "held-out", "excluded"))
        counts = results_df["split"].value_counts()
        print(f"\n  split: " + "  ".join(f"{k}={v:,}" for k, v in counts.items()))
        print("         (report accuracy on held-out only — the model scores "
              "100% on its own training images)")
    else:
        results_df["split"] = "unknown"
        print("\n  NOTE: this checkpoint predates split recording, so predictions.csv "
              "cannot tell\n        train from held-out. Any accuracy computed over "
              "all rows is inflated.")

    csv_path = output_dir / "predictions.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved → {csv_path}")

    # Summary
    if len(results_df) > 0:
        print(f"\nSummary:")
        print(f"  Total processed : {len(results_df):,}")
        print(f"  Indets sorted   : {results_df['indet'].sum():,}")
        print(f"  Flagged (uncertain/misID): {results_df['flagged'].sum():,}")
        print(f"  Mean confidence : {results_df['confidence'].mean():.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="Identify indets and flag misidentified herbarium images.")
    p.add_argument("--checkpoint", required=True, metavar="CKPT|DIR",
                   help="checkpoint .ckpt file, or a directory — auto-picks the most recent .ckpt")
    p.add_argument("--nameslist",  default=None, metavar="JSON",
                   help="nameslist.json (optional if checkpoint was saved by a recent training run)")
    p.add_argument("--sources", nargs="+", required=True, metavar="CSV:DIR",
                   help="specsin.csv:images_dir pairs (same format as train_herbarium.py)")
    p.add_argument("--output-dir", required=True, metavar="DIR")
    p.add_argument("--model", default=None, metavar="TIMM_MODEL",
                   help="timm model name (only needed if not in checkpoint)")
    p.add_argument("--image-sz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Confidence threshold for flagging mismatch (default 0.7)")
    p.add_argument("--low-conf-threshold", type=float, default=0.0,
                   help="Flag identified images below this confidence regardless of label "
                        "(0=disabled, e.g. 0.3 flags anything the model is unsure about)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override softmax temperature for calibration (divides logits before "
                        "softmax). Default: use the value fitted during training and stored in "
                        "the checkpoint (1.0 if none). >1 softens over-confident predictions; "
                        "try 2-4 on an uncalibrated checkpoint to spread probability into the top-5.")
    p.add_argument("--no-genus-head", action="store_true",
                   help="Ignore the trained genus head on a hierarchical checkpoint and derive "
                        "pred_genus from the first word of the species prediction (the old "
                        "behaviour). The genus head is normally more accurate at genus, since "
                        "it is trained for it directly and doesn't have to get the species right "
                        "first — 93%% vs ~88%% on Rubiaceae.")
    p.add_argument("--logit-adjust", type=float, default=0.0, metavar="TAU",
                   help="Shift predictions along the common/rare axis by adding "
                        "TAU*log(class_count) to each logit. Two-way dial: "
                        "TAU > 0 favours commoner taxa — set it to the --class-weight-beta the "
                        "model was trained with to cancel that weighting exactly (use 1.0 for "
                        "checkpoints predating --class-weight-beta, which hardcoded full "
                        "inverse-frequency and let near-empty classes swamp the commonest taxa). "
                        "TAU < 0 favours rarer taxa — this is the cheap, tunable way to get a "
                        "rare-class boost: train with --class-weight-beta 0 and sweep TAU here, "
                        "rather than baking a guess into the loss and paying for a GPU run per "
                        "guess. TAU=-0.25 ≈ training at beta=0.25. 0 = model as trained. "
                        "Counts come from the checkpoint, or are re-derived from specsin.")
    p.add_argument("--geo-weight", type=float, default=0.0,
                   help="Weight for geographic reranking (0=off, 0.3 is a good starting point). "
                        "Blends model probability with a kernel density score from training "
                        "occurrence data: final = (1-w)*model_prob + w*geo_score")
    p.add_argument("--geo-sigma", type=float, default=500.0,
                   help="Bandwidth in km for the geographic kernel (default 500). "
                        "Larger values give a broader, more permissive range influence.")
    return p.parse_args()


if __name__ == "__main__":
    identify(parse_args())
