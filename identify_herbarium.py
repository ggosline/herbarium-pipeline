"""
Run inference with a trained herbarium checkpoint.

Two modes:
  1. Indets  — images where indet=True in specsin.  Sorts them into
               output_dir/indets/{predicted_species}/ by top prediction.
  2. Flagged — images where indet=False but the model's top prediction
               disagrees with the recorded label OR confidence < threshold.
               Sorted into output_dir/uncertain/{true_species}__pred_{predicted}/

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
            blank = torch.zeros(3, self.transform.transforms[1].size,
                                self.transform.transforms[1].size)
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
    """If path is a directory, return the most recently modified .ckpt inside it."""
    if path.is_dir():
        ckpts = sorted(path.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {path}")
        chosen = ckpts[0]
        print(f"  Auto-selected checkpoint: {chosen.name}")
        return chosen
    return path


def load_model(checkpoint_path: Path, nameslist: list[str], image_sz: int):
    """Load a TimmModel from a Lightning checkpoint.

    Returns (state_dict, model_name, num_classes, nameslist, geo_dim).
    nameslist may be updated from the checkpoint if embedded there.
    geo_dim > 0 indicates the checkpoint uses a geo MLP; state_dict will
    contain geo_mlp.* keys in addition to backbone internals and head.*.
    """
    num_classes = len(nameslist)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract nameslist embedded by on_save_checkpoint (preferred over external file)
    if "nameslist" in ckpt:
        embedded = ckpt["nameslist"]
        if isinstance(embedded, dict):
            nameslist = embedded.get("species") or max(embedded.values(), key=len)
        else:
            nameslist = embedded
        num_classes = len(nameslist)
        print(f"  Nameslist loaded from checkpoint ({num_classes} classes)")
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
            # head_genus / head_family discarded — species head is sufficient
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

    return cleaned, model_name, num_classes, nameslist, geo_dim


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
) -> tuple[list[list[int]], list[list[float]]]:
    """Return (top_k_indices, top_k_probs) for each path.

    geo_coords: optional float32 Tensor [N, 4] aligned with paths.
    Passed to the model when provided (geo-capable checkpoints).
    """
    ds = InferenceDataset(paths, image_sz, geo_coords)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=False)
    model.eval().to(device)

    all_topk_preds, all_topk_probs = [], []
    for batch_tensors, _, batch_geo in tqdm(loader, desc="Inferring", unit="batch"):
        batch_tensors = batch_tensors.to(device)
        if geo_coords is not None:
            logits = model(batch_tensors, batch_geo.to(device))
        else:
            logits = model(batch_tensors)
        probs  = torch.softmax(logits, dim=1)
        k = min(top_k, probs.shape[1])
        topk_probs, topk_preds = torch.topk(probs, k=k, dim=1)
        all_topk_preds.extend(topk_preds.cpu().tolist())
        all_topk_probs.extend(topk_probs.cpu().tolist())

    return all_topk_preds, all_topk_probs


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
    state_dict, ckpt_model_name, num_classes, nameslist, geo_dim = load_model(
        checkpoint_path, nameslist, args.image_sz
    )
    if not nameslist:
        print("ERROR: no nameslist found. Pass --nameslist or use a checkpoint from a recent run.")
        sys.exit(1)

    model_name = args.model or ckpt_model_name
    if not model_name:
        print("ERROR: cannot determine model architecture. Pass --model <timm_model_name>")
        sys.exit(1)

    print(f"Building model: {model_name}  ({num_classes} classes)")

    if geo_dim:
        # Geo checkpoint: build backbone + geo_mlp + head separately so all
        # weights load correctly and geo features can be used at inference.
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feat_dim = backbone.num_features
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
        base_model = _GeoModel(backbone, geo_mlp, head, geo_dim)
        print(f"  Geo-capable model built (feat_dim={feat_dim}, geo_dim={geo_dim})")
    else:
        base_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}")
        if missing:
            print(f"  WARNING: missing keys — weights not loaded for: {missing[:5]}")

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
    print(f"Total images with files: {len(df_all):,}")

    # Build species → family lookup from specsin metadata (may be absent in older CSVs)
    species_to_family: dict[str, str] = {}
    if "family" in df_all.columns and "species" in df_all.columns:
        for sp, fam in zip(df_all["species"], df_all["family"]):
            if sp and fam and str(sp) not in ("nan", "") and str(fam) not in ("nan", ""):
                species_to_family[str(sp)] = str(fam)

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
        topk_preds, topk_probs = run_inference(base_model, indet_paths, args.image_sz,
                                               args.batch_size, device,
                                               geo_coords=_geo_for(df_indet))
        topk_preds, topk_probs = geo_rerank(topk_preds, topk_probs, df_indet,
                                             geo_index, args.geo_weight, args.geo_sigma)
        for row, preds_k, probs_k in zip(df_indet.itertuples(), topk_preds, topk_probs):
            pred_species = nameslist[preds_k[0]] if preds_k[0] < len(nameslist) else "unknown"
            dest = output_dir / "indets" / pred_species
            copy_image(Path(row.abs_path), dest)
            entry = {
                "fname":          row.fname,
                "abs_path":       row.abs_path,
                "specsin_file":   row.specsin_file,
                "source":         row.img_dir,
                "decimalLatitude":  getattr(row, "decimalLatitude",  ""),
                "decimalLongitude": getattr(row, "decimalLongitude", ""),
                "true_species":   "",
                "true_family":    getattr(row, "family", "") or "",
                "pred_species":   pred_species,
                "pred_family":    species_to_family.get(pred_species, ""),
                "confidence":     round(probs_k[0], 4),
                "indet":          True,
                "flagged":        False,
            }
            for k, (pi, pr) in enumerate(zip(preds_k, probs_k), 1):
                entry[f"top{k}_name"] = nameslist[pi] if pi < len(nameslist) else "unknown"
                entry[f"top{k}_prob"] = round(pr, 4)
            results.append(entry)
        print(f"  → Sorted into {output_dir / 'indets'}/")

    # --- Identified: flag disagreements ---
    if len(df_ident) > 0:
        print(f"\nRunning inference on {len(df_ident):,} identified specimens...")
        ident_paths = [Path(p) for p in df_ident["abs_path"]]
        topk_preds, topk_probs = run_inference(base_model, ident_paths, args.image_sz,
                                               args.batch_size, device,
                                               geo_coords=_geo_for(df_ident))
        topk_preds, topk_probs = geo_rerank(topk_preds, topk_probs, df_ident,
                                             geo_index, args.geo_weight, args.geo_sigma)
        flagged_count = 0
        for row, preds_k, probs_k in zip(df_ident.itertuples(), topk_preds, topk_probs):
            pred_species  = nameslist[preds_k[0]] if preds_k[0] < len(nameslist) else "unknown"
            conf          = probs_k[0]
            true_species  = getattr(row, "species", "")
            # Flag if model strongly disagrees OR if confidence is below threshold
            mismatch = (pred_species != true_species) and (conf >= args.threshold)
            low_conf = conf < args.low_conf_threshold if args.low_conf_threshold > 0 else False
            flagged  = mismatch or low_conf

            if flagged:
                folder_name = f"{true_species}__pred_{pred_species}"
                dest = output_dir / "uncertain" / folder_name
                copy_image(Path(row.abs_path), dest)
                flagged_count += 1

            entry = {
                "fname":          row.fname,
                "abs_path":       row.abs_path,
                "specsin_file":   row.specsin_file,
                "source":         row.img_dir,
                "decimalLatitude":  getattr(row, "decimalLatitude",  ""),
                "decimalLongitude": getattr(row, "decimalLongitude", ""),
                "true_species":   true_species,
                "true_family":    getattr(row, "family", "") or "",
                "pred_species":   pred_species,
                "pred_family":    species_to_family.get(pred_species, ""),
                "confidence":     round(conf, 4),
                "indet":          False,
                "flagged":        flagged,
            }
            for k, (pi, pr) in enumerate(zip(preds_k, probs_k), 1):
                entry[f"top{k}_name"] = nameslist[pi] if pi < len(nameslist) else "unknown"
                entry[f"top{k}_prob"] = round(pr, 4)
            results.append(entry)

        print(f"  → Flagged {flagged_count:,} specimens → {output_dir / 'uncertain'}/")

    # Save predictions CSV
    results_df = pd.DataFrame(results)
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
