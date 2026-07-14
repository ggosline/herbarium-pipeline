#!/usr/bin/env python
"""Novelty (out-of-distribution) scoring for herbarium specimens.

Which specimens is the model quietly guessing at? Softmax sums to 1 over the
trained classes, so a taxon the model has never seen still gets filed somewhere,
often confidently. This script scores every specimen by how *unlike* the training
distribution it looks, so a curator can triage the ones worth a second look.

Two questions are kept deliberately apart, because they are not the same problem:

  NOVEL GENUS  — the genus was never trained. Genuinely out-of-distribution.
  NOVEL SPECIES in a KNOWN genus — a new Psychotria IS a Psychotria. The genus
      head is *right* to be confident; only the species head has nowhere to put
      it. No genus-level signal can detect this, even in principle — which is why
      the embedding scores below matter more than the softmax ones.

Scores, per specimen (higher = more novel):

  msp     1 - max softmax                        Hendrycks & Gimpel 2017
  energy  -logsumexp(logits)                     Liu et al. 2020
  maha    Mahalanobis distance to the nearest class centroid, tied covariance
                                                 Lee et al. 2018
  knn     cosine distance to the k-th nearest training specimen
                                                 Sun et al. 2022

Centroids and covariance are fitted on the TRAIN split ONLY, and evaluation uses
the HELD-OUT split as the in-distribution negatives. This matters: the model
memorises its training images (mean top-1 confidence 0.9985 on train vs 0.8870
held out), so scoring novelty against training images inflates AUROC by ~8 points.
The split is reconstructed deterministically from the checkpoint's own
hyperparameters and verified against its stored class_counts before use.

Usage:
    python score_ood.py \
        --checkpoint runs/checkpoints/acc-epoch=09-val_Accuracy=0.8870.ckpt \
        --sources specsin.csv:images/ \
        --output-dir runs/ood/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from identify_herbarium import (
    InferenceDataset,
    _GeoModel,
    build_genus_head,
    build_model_from_state,
    encode_coords,
    load_model,
    resolve_checkpoint,
)

DEFAULT_KNN_K = 10
# Ridge added to the covariance diagonal before inversion. The empirical
# covariance of a 1024-d feature space fitted on ~68k samples is full rank but
# badly conditioned; without this the inverse amplifies noise directions and the
# Mahalanobis score degenerates.
DEFAULT_SHRINKAGE = 0.01


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _Featurizer(torch.nn.Module):
    """Runs the model and returns (image features, species logits, genus logits).

    The features are the backbone's pooled output — the IMAGE embedding only,
    deliberately excluding the geo vector. A specimen collected somewhere unusual
    is not morphologically novel, and fusing geo in here would conflate the two.
    The classifier heads still see the geo-fused vector, exactly as in training.
    """

    def __init__(self, base: torch.nn.Module, genus_head: torch.nn.Module | None):
        super().__init__()
        self.base = base
        self.genus_head = genus_head

    def forward(self, x, geo=None):
        if isinstance(self.base, _GeoModel):
            img = self.base.backbone(x)
            if geo is None:
                geo = torch.zeros(img.shape[0], 4, device=img.device)
            z = torch.cat([img, self.base.geo_mlp(geo)], dim=1)
            logits = self.base.head(z)
        else:
            img = self.base.forward_head(
                self.base.forward_features(x), pre_logits=True)
            z = img
            logits = self.base.get_classifier()(z)
        glogits = self.genus_head(z) if self.genus_head is not None else None
        return img, logits, glogits


@torch.inference_mode()
def extract(model, paths, image_sz, batch_size, device, geo_coords=None):
    ds = InferenceDataset(paths, image_sz, geo_coords)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=False)
    model.eval().to(device)

    feats, msp, energy, gconf = [], [], [], []
    for imgs, _, geo in tqdm(loader, desc="Embedding", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        geo_arg = geo.to(device) if geo_coords is not None else None
        f, logits, glogits = model(imgs, geo_arg)

        feats.append(f.float().cpu())
        # Raw logits — NOT temperature-scaled. Temperature is a monotone rescale
        # per specimen but not across specimens, so it perturbs the ranking these
        # scores are judged on; energy in particular is defined on raw logits.
        p = logits.float().softmax(dim=1)
        msp.append(p.max(dim=1).values.cpu())
        energy.append((-torch.logsumexp(logits.float(), dim=1)).cpu())
        if glogits is not None:
            gconf.append(glogits.float().softmax(dim=1).max(dim=1).values.cpu())

    return (torch.cat(feats).numpy(),
            torch.cat(msp).numpy(),
            torch.cat(energy).numpy(),
            torch.cat(gconf).numpy() if gconf else None)


# ---------------------------------------------------------------------------
# Distance scores
# ---------------------------------------------------------------------------

def fit_mahalanobis(train_feats: np.ndarray, train_labels: np.ndarray,
                    n_classes: int, shrinkage: float, device):
    """Per-class centroids + one shared (tied) covariance, fitted on TRAIN only.

    A tied covariance is the standard choice (Lee et al. 2018): a per-class
    covariance would need ~1024² samples per class, and the rarest trained class
    here has 20.
    """
    X = torch.from_numpy(train_feats).to(device)
    y = torch.from_numpy(train_labels).to(device)
    d = X.shape[1]

    mu = torch.zeros(n_classes, d, device=device)
    for c in range(n_classes):
        m = y == c
        if m.any():
            mu[c] = X[m].mean(0)

    centered = X - mu[y]                       # subtract each sample's own centroid
    sigma = (centered.T @ centered) / len(X)
    sigma += shrinkage * torch.eye(d, device=device) * sigma.diagonal().mean()

    # Whitening transform: maha(f, mu) == ||L^T f - L^T mu||^2 with L from the
    # Cholesky of the precision. Turns a quadratic form into a plain euclidean
    # distance, so scoring is one cdist instead of 1341 quadratic forms.
    prec = torch.linalg.inv(sigma)
    prec = (prec + prec.T) / 2                 # re-symmetrise after inversion
    L = torch.linalg.cholesky(prec)
    return mu @ L, L


def mahalanobis_score(feats, mu_w, L, device, chunk=2048):
    out = np.empty(len(feats), dtype=np.float32)
    for i in range(0, len(feats), chunk):
        f = torch.from_numpy(feats[i:i + chunk]).to(device) @ L
        out[i:i + chunk] = torch.cdist(f, mu_w).min(dim=1).values.cpu().numpy()
    return out


def fit_background(train_feats: np.ndarray, shrinkage: float, device):
    """One class-agnostic Gaussian over ALL training features.

    This is the fix for NEAR-OOD (Ren et al. 2021). Plain Mahalanobis measures
    total distance from a class centroid, which is dominated by how generically
    "Rubiaceae-like" a sheet is — pressed, mounted, a herbarium sheet at all.
    Every novel taxon here IS another Rubiaceae, so that shared component swamps
    the bit we care about. Subtracting the background distance cancels it and
    leaves only the part that is class-specific.
    """
    X = torch.from_numpy(train_feats).to(device)
    mu0 = X.mean(0, keepdim=True)
    c = X - mu0
    sigma0 = (c.T @ c) / len(X)
    sigma0 += shrinkage * torch.eye(X.shape[1], device=device) * sigma0.diagonal().mean()
    prec = torch.linalg.inv(sigma0)
    prec = (prec + prec.T) / 2
    return mu0 @ torch.linalg.cholesky(prec), torch.linalg.cholesky(prec)


def background_score(feats, mu0_w, L0, device, chunk=2048):
    out = np.empty(len(feats), dtype=np.float32)
    for i in range(0, len(feats), chunk):
        f = torch.from_numpy(feats[i:i + chunk]).to(device) @ L0
        out[i:i + chunk] = torch.cdist(f, mu0_w).squeeze(1).cpu().numpy()
    return out


def knn_score(feats, train_feats, k, device, chunk=1024):
    """Cosine distance to the k-th nearest TRAIN specimen."""
    T = F.normalize(torch.from_numpy(train_feats).to(device), dim=1)
    out = np.empty(len(feats), dtype=np.float32)
    for i in range(0, len(feats), chunk):
        q = F.normalize(torch.from_numpy(feats[i:i + chunk]).to(device), dim=1)
        sim = q @ T.T
        kth = sim.topk(k, dim=1).values[:, -1]
        out[i:i + chunk] = (1.0 - kth).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def auroc(score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    a, b = score[pos], score[neg]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan")
    r = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def flag_rate(score, pos, neg, recall=0.80) -> float:
    """Fraction of the reviewed pool you must flag to catch `recall` of the novel."""
    thr = np.nanquantile(score[pos], 1 - recall)
    return float(np.nanmean(np.concatenate([score[pos], score[neg]]) >= thr))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="specsin.csv:image_dir/ (repeatable)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-sz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    ap.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    ap.add_argument("--reuse-features", action="store_true",
                    help="Skip the GPU pass and reload features.npy from --output-dir.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = resolve_checkpoint(args.checkpoint)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {})
    hp = hp.get("config", hp)
    nl = raw["nameslist"]
    if not isinstance(nl, dict):
        sys.exit("This script needs a hierarchical checkpoint (nameslist must be a dict).")
    species_names = list(nl["species"])
    genus_names = list(nl["genus"])
    print(f"Checkpoint: {ckpt.name}")
    print(f"  {len(species_names):,} species, {len(genus_names)} genera")

    # ── the specimen table ───────────────────────────────────────────────────
    sources = []
    for s in args.sources:
        csv_s, _, img_s = s.partition(":")
        sources.append((Path(csv_s), Path(img_s)))

    frames = []
    for csv_p, img_d in sources:
        df = pd.read_csv(csv_p, low_memory=False)
        m = df["hasfile"].fillna(False).astype(bool)
        for col in ("outlier", "invalid"):
            if col in df.columns:
                m &= ~df[col].fillna(False).astype(bool)
        df = df.loc[m].copy()
        df["abs_path"] = df["fname"].map(lambda f: str(img_d / f))
        df = df[df["abs_path"].map(lambda p: Path(p).exists())]
        frames.append(df)
    spec = pd.concat(frames, ignore_index=True)
    print(f"  {len(spec):,} specimens with images")

    # ── features ─────────────────────────────────────────────────────────────
    fpath = args.output_dir / "features.npy"
    spath = args.output_dir / "softmax_scores.npz"
    if args.reuse_features and fpath.exists():
        feats = np.load(fpath)
        z = np.load(spath)
        msp, energy, gconf = z["msp"], z["energy"], z["gconf"]
        print(f"  reused features {feats.shape} from {fpath}")
    else:
        (state, model_name, num_classes, _nl, geo_dim, _lvl,
         _temp, _exc, _cc, genus_head_sd, _split) = load_model(
            ckpt, species_names, args.image_sz)
        base = build_model_from_state(state, model_name, num_classes, geo_dim)
        ghead = None
        if genus_head_sd:
            # load_model hands back {"nameslist": [...], "state_dict": {...}},
            # not a bare state dict. The head is fed the geo-FUSED vector, so its
            # input width is feat_dim + geo_dim — read it off the species head.
            feat_dim = (base.head.in_features if isinstance(base, _GeoModel)
                        else base.get_classifier().in_features)
            ghead = build_genus_head(genus_head_sd["state_dict"], feat_dim)
        model = _Featurizer(base, ghead)

        geo = None
        if geo_dim and {"decimalLatitude", "decimalLongitude"} <= set(spec.columns):
            geo = encode_coords(spec["decimalLatitude"], spec["decimalLongitude"])

        feats, msp, energy, gconf = extract(
            model, [Path(p) for p in spec["abs_path"]],
            args.image_sz, args.batch_size, device, geo)
        np.save(fpath, feats)
        np.savez(spath, msp=msp, energy=energy,
                 gconf=gconf if gconf is not None else np.zeros(0))
        print(f"  saved features {feats.shape} → {fpath}")

    # ── which specimens the model actually saw ───────────────────────────────
    # Every number below depends on getting this right. Prefer the split recorded
    # in the checkpoint; fall back to reconstructing it for older checkpoints, and
    # in that case PROVE the reconstruction before trusting it.
    embedded = raw.get("split") or {}
    if embedded.get("valid"):
        train_f = set(embedded.get("train") or [])
        val_f = set(embedded["valid"])
        print(f"\nSplit read from checkpoint: {len(train_f):,} train / "
              f"{len(val_f):,} held-out")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_herbarium import HerbariumData

        print("\nCheckpoint has no embedded split — reconstructing "
              f"(seed={hp['seed']}, sparse>={hp['sparse_threshold']}, "
              f"cap={hp['max_per_class']}) ...")
        data = HerbariumData(
            sources=sources,
            label_level=hp["label_level"],
            hierarchical=hp["hierarchical"],
            sparse_threshold=hp["sparse_threshold"],
            seed=hp["seed"],
            max_per_class=hp["max_per_class"],
            class_weight_beta=hp["class_weight_beta"],
        )
        if list(data.class_counts) != list(raw["class_counts"]):
            sys.exit("ABORT: reconstructed split does not match the checkpoint's "
                     "class_counts — the held-out set would be wrong, and every "
                     "AUROC below would be meaningless.")
        print("  split reproduced exactly (class_counts match the checkpoint)")
        train_f = {Path(p).name for p in data.train_files}
        val_f = {Path(p).name for p in data.valid_files}

    spec["_f"] = spec["fname"].map(lambda p: Path(str(p)).name)
    spec["in_train"] = spec["_f"].isin(train_f)
    spec["in_val"] = spec["_f"].isin(val_f)

    sp_set, ge_set = set(species_names), set(genus_names)
    spec["_sp"] = spec["species"].astype(str)
    spec["_ge"] = spec["_sp"].str.split().str[0]

    # An INDET has no identification at all. It is scored (it is exactly who the
    # curator wants triaged) but it must never enter the EVALUATION: it has no
    # true taxon, so it cannot be called novel or in-distribution. Counting indets
    # as "novel genus" — which is what a naive `species` lookup does, since NaN
    # is in no genus — silently added 6,237 ordinary specimens of known genera to
    # the novel set and pushed max-softmax AUROC from 0.88 down to 0.69.
    spec["is_indet"] = (
        spec["indet"].fillna(False).astype(str).str.lower().isin(("true", "1"))
        if "indet" in spec.columns else False
    )
    labelled = (~spec["is_indet"]
                & spec["species"].notna()
                & (spec["_sp"].str.strip() != "")
                & (spec["_sp"].str.lower() != "nan"))
    spec["_labelled"] = labelled

    spec["novel_genus"] = labelled & ~spec["_ge"].isin(ge_set)
    spec["novel_species"] = labelled & ~spec["_sp"].isin(sp_set)
    spec["novel_sp_known_genus"] = spec["novel_species"] & ~spec["novel_genus"]
    print(f"  {int(spec['is_indet'].sum()):,} indets (scored, but excluded from evaluation)")

    # ── fit on TRAIN only, score everything ──────────────────────────────────
    sp_index = {n: i for i, n in enumerate(species_names)}
    tr_mask = spec["in_train"].to_numpy()
    tr_lab = spec.loc[tr_mask, "_sp"].map(sp_index).to_numpy(dtype=np.int64)

    print(f"\nFitting on {tr_mask.sum():,} TRAIN specimens "
          f"(shrinkage={args.shrinkage}, knn k={args.knn_k}) ...")
    mu_w, L = fit_mahalanobis(feats[tr_mask], tr_lab, len(species_names),
                              args.shrinkage, device)
    maha = mahalanobis_score(feats, mu_w, L, device)
    knn = knn_score(feats, feats[tr_mask], args.knn_k, device)

    mu0_w, L0 = fit_background(feats[tr_mask], args.shrinkage, device)
    bg = background_score(feats, mu0_w, L0, device)
    rmd = maha - bg                       # relative Mahalanobis (Ren et al. 2021)

    scores = {"msp": 1.0 - msp, "energy": energy, "maha": maha,
              "rmd": rmd, "knn": knn}
    if gconf is not None and len(gconf):
        scores["1-genus_conf"] = 1.0 - gconf

    # Softmax and embedding distance fail on different specimens, so their sum
    # (each standardised on the held-out set, which is the only distribution we
    # are entitled to calibrate against) can beat either alone.
    def _z(x):
        ref = x[spec["in_val"].to_numpy()]
        return (x - np.nanmean(ref)) / (np.nanstd(ref) + 1e-9)
    scores["msp+rmd"] = _z(scores["msp"]) + _z(rmd)

    for name, s in scores.items():
        spec[f"ood_{name}"] = s

    # ── evaluate against the HELD-OUT negatives ──────────────────────────────
    neg = spec["in_val"].to_numpy()
    print(f"\nEvaluating vs {neg.sum():,} HELD-OUT in-distribution specimens\n")
    report = {}
    for label, pos in (("NOVEL GENUS (true OOD)", spec["novel_genus"].to_numpy()),
                       ("NOVEL SPECIES, known genus",
                        spec["novel_sp_known_genus"].to_numpy())):
        print("=" * 68)
        print(f"{label}   (n={pos.sum():,})")
        print("=" * 68)
        print(f"  {'score':<16} {'AUROC':>7}   flag-rate to catch 80% of them")
        rows = sorted(((auroc(s, pos, neg), n, flag_rate(s, pos, neg))
                       for n, s in scores.items()), reverse=True)
        for a, n, fr in rows:
            print(f"  {n:<16} {a:7.3f}   {fr:6.1%}")
        report[label] = {n: {"auroc": a, "flag_rate_at_80": fr}
                         for a, n, fr in rows}
        print()

    out_csv = args.output_dir / "ood_scores.csv"
    cols = (["fname", "species", "_ge", "in_train", "in_val", "novel_genus",
             "novel_species", "novel_sp_known_genus"]
            + [f"ood_{n}" for n in scores])
    spec[cols].rename(columns={"_ge": "genus"}).to_csv(out_csv, index=False)
    (args.output_dir / "ood_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.output_dir / 'ood_report.json'}")


if __name__ == "__main__":
    main()
