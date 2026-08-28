"""Step 2 of the interpretability study: is the classifier reading the plant?

Three controls, all on the held-out split:

  mask   — blank the plant, or everything but the plant, and re-run the real
           species head. The plant mask is thresholded from PC1 of the patch
           tokens (probe_embeddings showed PC1 is a foreground detector), with
           a crude bottom-quadrant version as an assumption-free cross-check.
  geo    — zero / permute / permute-within-country the geo vector and watch
           what happens to accuracy, globally and for well-sampled regions.
  probe  — linear probe from the cached embedding to institutionCode, against
           a genus-only baseline. The gap is the answer: an embedding that
           barely beats genus-one-hot was never carrying herbarium identity,
           it was carrying taxon.

Accuracy is always measured on the trained species head, never on embedding
geometry — the question is whether predictions survive, not whether clusters
move. Reads the feature cache written by `probe_embeddings.py extract`.

Usage:
  python probe_confounds.py mask  --checkpoint CKPT --out DIR
  python probe_confounds.py geo   --checkpoint CKPT --out DIR
  python probe_confounds.py probe --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from identify_herbarium import InferenceDataset, encode_coords, load_model, resolve_checkpoint
import probe_embeddings as pe
from probe_embeddings import META_NAME, FEAT_NAME, extract, grid_shape, load_backbone

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Plant masks from PC1
# ---------------------------------------------------------------------------

def otsu(values: np.ndarray, bins: int = 64) -> float:
    """Otsu threshold over a small 1-D array (one image's 1,600 PC1 scores)."""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.cumsum()
    total = w[-1]
    if total == 0:
        return float(values.mean())
    m = (hist * centers).cumsum()
    mu_t = m[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mu_t * w / total - m) ** 2 / (w * (total - w) / total**2 + 1e-12)
    between = np.nan_to_num(between)
    return float(centers[int(np.argmax(between))])


def pc1_masks(backbone, paths: list[str], args, device) -> np.ndarray:
    """Boolean [N, gh, gw] plant masks, True where PC1 says 'specimen'.

    PCA component signs are arbitrary, so 'plant' is pinned by geometry rather
    than by sign: the sheet dominates the border of a herbarium scan, so the
    component is flipped if its border mean exceeds its interior mean. Getting
    this wrong silently inverts every mask condition below.
    """
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(args.seed)
    fit_paths = list(np.array(paths)[rng.permutation(len(paths))[:args.fit_images]])
    bag: list[np.ndarray] = []
    extract(backbone, fit_paths, args.image_sz, args.batch_size, device,
            args.num_workers,
            token_cb=lambda t: bag.append(t.reshape(-1, t.shape[-1]).astype(np.float32)),
            tokens_per_image=args.tokens_per_image, seed=args.seed)
    tokens = np.concatenate(bag, axis=0)
    mean = tokens.mean(axis=0, keepdims=True)
    pca = PCA(n_components=1, svd_solver="randomized", random_state=args.seed)
    pca.fit(tokens - mean)

    gh, gw = grid_shape(backbone, args.image_sz)
    maps: list[np.ndarray] = []
    extract(backbone, paths, args.image_sz, args.batch_size, device, args.num_workers,
            token_cb=lambda t: maps.extend(
                list(pca.transform(t.reshape(-1, t.shape[-1]).astype(np.float32) - mean)
                     .reshape(t.shape[0], gh, gw))),
            tokens_per_image=0)
    pc1 = np.stack(maps, axis=0)

    border = np.concatenate([pc1[:, 0, :], pc1[:, -1, :], pc1[:, :, 0], pc1[:, :, -1]], axis=1)
    interior = pc1[:, gh // 4:3 * gh // 4, gw // 4:3 * gw // 4].reshape(len(pc1), -1)
    if border.mean() > interior.mean():
        pc1 = -pc1
        print("  PC1 sign flipped so that bright == specimen")

    masks = np.stack([m > otsu(m.ravel()) for m in pc1], axis=0)
    frac = masks.mean(axis=(1, 2))
    print(f"  Plant mask covers {frac.mean():.1%} of the sheet on average "
          f"(p10 {np.percentile(frac, 10):.1%}, p90 {np.percentile(frac, 90):.1%})")
    return masks


def quadrant_mask(gh: int, gw: int) -> np.ndarray:
    """Bottom-right quadrant — where herbarium labels overwhelmingly sit."""
    m = np.zeros((gh, gw), dtype=bool)
    m[gh // 2:, gw // 2:] = True
    return m


# ---------------------------------------------------------------------------
# Masked inference
# ---------------------------------------------------------------------------

class MaskedDataset(InferenceDataset):
    """InferenceDataset that blanks part of the sheet before normalisation.

    Fill colour is the per-image median of the *unmasked* region, so a blanked
    area reads as empty mounting sheet rather than as a grey rectangle the
    model has never seen on any specimen.
    """

    def __init__(self, paths, image_sz, geo_coords, masks: np.ndarray | None,
                 invert: bool = False):
        super().__init__(paths, image_sz, geo_coords)
        self.masks = masks
        self.invert = invert

    def __getitem__(self, idx):
        path = self.paths[idx]
        geo = self.geo[idx] if self.geo is not None else torch.zeros(4)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return torch.zeros(3, self.image_sz, self.image_sz), str(path), geo
        for t in self.transform.transforms[:2]:      # Resize, CenterCrop only
            img = t(img)
        if self.masks is not None:
            arr = np.asarray(img).copy()
            grid = self.masks[idx]
            m = np.array(Image.fromarray(grid.astype(np.uint8) * 255)
                         .resize((self.image_sz, self.image_sz), Image.NEAREST)) > 127
            if self.invert:
                m = ~m
            keep = arr[~m]
            fill = np.median(keep, axis=0) if keep.size else np.array([200, 200, 190])
            arr[m] = fill.astype(arr.dtype)
            img = Image.fromarray(arr)
        for t in self.transform.transforms[2:]:      # ToTensor, Normalize
            img = t(img)
        return img, str(path), geo


def build_full_model(ckpt_path: Path, device):
    """The trained model exactly as identify runs it, plus its metadata.

    `label_level` comes back with the rest because a probe cannot score a model
    without knowing which rank its class indices refer to. The family-level
    Angiosperm run indexes 235 families, so scoring it against specsin's
    `species` column matches nothing and every accuracy silently reads 0.0 —
    quieter and more misleading than a crash. identify_herbarium infers the rank
    from the nameslist itself (see _infer_label_level), so it is trustworthy
    even when the checkpoint's stored label_level is not.
    """
    from identify_herbarium import build_model_from_state
    ckpt = resolve_checkpoint(Path(ckpt_path))
    (state, model_name, num_classes, nameslist, geo_dim, label_level,
     temperature, excluded, class_counts, genus_head, split) = load_model(ckpt, [], 0)
    model = build_model_from_state(state, model_name, num_classes, geo_dim).to(device).eval()
    return model, nameslist, temperature, geo_dim, label_level


def truth_labels(df: pd.DataFrame, nameslist: list[str], label_level: str) -> np.ndarray:
    """Class index per row, -1 where the taxon is not one the model can predict.

    Sparse taxa dropped at training time land at -1 by design; a *wholesale* miss
    means the wrong column was read, so say so rather than reporting 0.0.
    """
    names = {n: i for i, n in enumerate(nameslist)}
    col = label_level if label_level in df.columns else "species"
    truth = df[col].map(lambda s: names.get(s, -1)).values
    if len(df) and (truth >= 0).sum() == 0:
        raise SystemExit(
            f"ERROR: none of the {len(df)} rows' {col!r} values are in the "
            f"checkpoint's {len(nameslist)}-class nameslist (e.g. {nameslist[:3]}). "
            "The probe would score every condition at 0.0.")
    return truth


@torch.inference_mode()
def predict(model, paths, geo, image_sz, batch_size, device, workers,
            masks=None, invert=False, temperature=1.0):
    ds = MaskedDataset([Path(p) for p in paths], image_sz, geo, masks, invert)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        pin_memory=True, shuffle=False)
    top1, top5, conf = [], [], []
    for batch, _, batch_geo in tqdm(loader, desc="Inferring", unit="batch",
                                    leave=False, disable=pe.QUIET):
        batch = batch.to(device, non_blocking=True)
        g = batch_geo.to(device) if geo is not None else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(batch, g) if geo is not None else model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits.float() / temperature, dim=1)
        p5 = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        top1.extend(p5.indices[:, 0].cpu().tolist())
        top5.extend(p5.indices.cpu().tolist())
        conf.extend(p5.values[:, 0].cpu().tolist())
    return np.array(top1), top5, np.array(conf)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(name: str, truth: np.ndarray, top1: np.ndarray, top5: list,
          conf: np.ndarray, base_top1: np.ndarray | None) -> dict:
    known = truth >= 0                       # taxa not in the nameslist score as nothing
    row = {
        "condition": name,
        "n": int(known.sum()),
        "top1": round(float((top1[known] == truth[known]).mean()), 4),
        "top5": round(float(np.mean([truth[i] in top5[i] for i in np.where(known)[0]])), 4),
        "mean_conf": round(float(conf.mean()), 4),
    }
    if base_top1 is not None:
        row["flip_rate"] = round(float((top1 != base_top1).mean()), 4)
    return row


def stage_mask(args) -> None:
    device = torch.device(args.device)
    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)

    backbone, _, _ = load_backbone(args.checkpoint, device)
    gh, gw = grid_shape(backbone, args.image_sz)
    masks = pc1_masks(backbone, df["path"].tolist(), args, device)
    np.savez_compressed(out / "plant_masks.npz", masks=masks)
    del backbone
    torch.cuda.empty_cache()

    model, nameslist, temperature, geo_dim, label_level = build_full_model(
        args.checkpoint, device)
    truth = truth_labels(df, nameslist, label_level)
    geo = encode_coords(df["decimalLatitude"], df["decimalLongitude"]) if geo_dim else None

    # Same mask, wrong place: preserves area and shape exactly, so any accuracy
    # lost to "a hole of this size exists" is subtracted rather than attributed
    # to the plant.
    rng = np.random.RandomState(args.seed)
    shifted = np.stack([np.roll(m, (rng.randint(gh // 4, 3 * gh // 4),
                                    rng.randint(gw // 4, 3 * gw // 4)), axis=(0, 1))
                        for m in masks])
    quad = np.repeat(quadrant_mask(gh, gw)[None], len(df), axis=0)

    conditions = [
        ("baseline",         None,     False),
        ("plant_removed",    masks,    False),
        ("plant_only",       masks,    True),
        ("shift_control",    shifted,  False),
        ("quadrant_removed", quad,     False),
        ("quadrant_only",    quad,     True),
    ]
    rows, base_top1, preds = [], None, {}
    for name, m, inv in conditions:
        print(f"  {name} ...")
        t1, t5, c = predict(model, df["path"].tolist(), geo, args.image_sz,
                            args.batch_size, device, args.num_workers, m, inv, temperature)
        rows.append(score(name, truth, t1, t5, c, base_top1))
        preds[name] = t1
        if base_top1 is None:
            base_top1 = t1
        print("   ", rows[-1])

    res = pd.DataFrame(rows)
    base = res.loc[res.condition == "baseline", "top1"].iloc[0]
    ctrl = res.loc[res.condition == "shift_control", "top1"].iloc[0]
    res["delta_vs_baseline"] = (res.top1 - base).round(4)
    # The number that means something: cost beyond an equal-area hole.
    res["delta_vs_control"] = (res.top1 - ctrl).round(4)
    res.to_csv(out / "mask_conditions.csv", index=False)
    print("\n" + res.to_string(index=False))

    per = pd.DataFrame({"species": df["species"], "true_idx": truth})
    for name, t1 in preds.items():
        per[name] = (t1 == truth)
    per = per[per.true_idx >= 0].groupby("species").mean(numeric_only=True).drop(columns="true_idx")
    per["n"] = df[truth >= 0].groupby("species").size()
    per.sort_values("plant_removed").to_csv(out / "mask_per_species.csv")
    print(f"  Wrote mask_conditions.csv and mask_per_species.csv")


# ---------------------------------------------------------------------------
# Geo ablation — cached image features, so no backbone pass is needed
# ---------------------------------------------------------------------------

def stage_geo(args) -> None:
    device = torch.device(args.device)
    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    feats = np.load(out / FEAT_NAME)["pooled"].astype(np.float32)

    ckpt = resolve_checkpoint(Path(args.checkpoint))
    (state, model_name, num_classes, nameslist, geo_dim, label_level,
     temperature, excluded, class_counts, genus_head, split) = load_model(ckpt, [], 0)
    if not geo_dim:
        raise SystemExit("ERROR: this checkpoint has no geo MLP — nothing to ablate.")
    geo_mlp = nn.Sequential(nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim))
    geo_mlp.load_state_dict({k[len("geo_mlp."):]: v for k, v in state.items()
                             if k.startswith("geo_mlp.")})
    head = nn.Linear(feats.shape[1] + geo_dim, num_classes)
    head.load_state_dict({k[len("head."):]: v for k, v in state.items()
                          if k.startswith("head.")})
    geo_mlp, head = geo_mlp.eval().to(device), head.eval().to(device)

    truth = truth_labels(df, nameslist, label_level)
    x = torch.from_numpy(feats).to(device)
    real = encode_coords(df["decimalLatitude"], df["decimalLongitude"])
    rng = np.random.RandomState(args.seed)

    def run(geo_vec):
        with torch.inference_mode():
            g = geo_mlp(geo_vec.to(device))
            logits = head(torch.cat([x, g], dim=1))
            probs = torch.softmax(logits.float() / temperature, dim=1)
            p5 = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        return (p5.indices[:, 0].cpu().numpy(), p5.indices.cpu().tolist(),
                p5.values[:, 0].cpu().numpy())

    # Permuting inside a country keeps the coarse region and destroys only the
    # within-country detail, separating 'knows the region' from 'knows the spot'.
    within = real.clone().numpy()
    country = df["countryCode"].fillna("(missing)").values
    for c in np.unique(country):
        pos = np.where(country == c)[0]
        within[pos] = within[pos][rng.permutation(len(pos))]

    # A missing coordinate does NOT encode as zeros: encode_coords sets lat/lon
    # to 0 and takes cos(0)*cos(0) = 1, giving [1, 0, 0, 0] with the fourth
    # element flagging 'no location'. Feeding all-zeros instead is an input the
    # model never saw in training, and it craters accuracy for that reason
    # alone — kept below only as an explicitly labelled OOD control.
    nan = pd.Series([np.nan] * len(df))
    variants = {
        "real": real,
        "no_location": encode_coords(nan, nan),
        "permuted": real[torch.from_numpy(rng.permutation(len(real)))],
        "permuted_within_country": torch.from_numpy(within),
        "all_zeros_OOD": torch.zeros_like(real),
    }
    rows, base = [], None
    for name, gv in variants.items():
        t1, t5, c = run(gv)
        rows.append(score(name, truth, t1, t5, c, base))
        if base is None:
            base = t1
        # Whether geo is load-bearing should be read per region, not globally:
        # a model can lean on coordinates heavily where sampling is dense and
        # not at all elsewhere, and the two average out to "no effect".
        has = df["decimalLatitude"].notna().values
        ok = (t1 == truth)
        rows[-1]["top1_with_coords"] = round(float(ok[has & (truth >= 0)].mean()), 4)
        rows[-1]["top1_no_coords"] = round(float(ok[~has & (truth >= 0)].mean()), 4)
    res = pd.DataFrame(rows)
    res.to_csv(out / "geo_ablation.csv", index=False)
    print(res.to_string(index=False))

    perm_top1 = run(variants["permuted"])[0]
    per_country = pd.DataFrame({"country": df["countryCode"].fillna("(missing)"),
                                "real": base == truth,
                                "permuted": perm_top1 == truth})
    agg = per_country.groupby("country").agg(n=("real", "size"), real=("real", "mean"),
                                             permuted=("permuted", "mean"))
    agg["drop"] = (agg.real - agg.permuted).round(4)
    agg[agg.n >= args.min_country].sort_values("drop", ascending=False).to_csv(
        out / "geo_by_country.csv")
    print(f"  Wrote geo_ablation.csv and geo_by_country.csv")


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

def stage_probe(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import OneHotEncoder

    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    feats = np.load(out / FEAT_NAME)["pooled"].astype(np.float32)
    df["institutionCode"] = df["institutionCode"].fillna("(missing)")

    # Group by species: a probe that splits at random can win by recognising
    # individual taxa it has already seen, which is the very confound under test.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    tr, te = next(gss.split(feats, df["institutionCode"], groups=df["species"]))
    print(f"  {len(tr):,} train / {len(te):,} test, split by species "
          f"({df.species.iloc[tr].nunique()} vs {df.species.iloc[te].nunique()} species)")

    genus_oh = OneHotEncoder(handle_unknown="ignore").fit(df[["genus"]])
    rng = np.random.RandomState(args.seed)

    def fit(X, y, label, train=None, test=None):
        train = tr if train is None else train
        test = te if test is None else test
        clf = LogisticRegression(max_iter=args.max_iter, C=args.C)
        clf.fit(X[train], y[train])
        return {"probe": label, "test_accuracy": round(float(clf.score(X[test], y[test])), 4)}

    # The species reference needs a RANDOM split, not the species-disjoint one:
    # under a grouped split no test species is ever seen in training, so the
    # score is 0 by construction and says nothing about the embedding.
    rand = rng.permutation(len(df))
    r_tr, r_te = rand[:int(0.7 * len(rand))], rand[int(0.7 * len(rand)):]

    inst = df["institutionCode"].values
    rows = [
        fit(feats, inst, "embedding -> institution"),
        fit(genus_oh.transform(df[["genus"]]).toarray(), inst, "genus one-hot -> institution"),
        fit(feats, df["species"].values,
            "embedding -> species (reference, random split)", r_tr, r_te),
        # Family survives the species-disjoint split (unlike species), so it is
        # the one reference measured on exactly the same rows as the institution
        # probe — and on a family-level model it is the rank the head predicts.
        fit(feats, df["family"].values,
            "embedding -> family (reference, species-disjoint)"),
    ]
    # Majority class and a label permutation: 28 institutions this imbalanced
    # make raw accuracy look impressive on its own.
    maj = df["institutionCode"].iloc[te].value_counts(normalize=True).iloc[0]
    rows.append({"probe": "majority class (institution)", "test_accuracy": round(float(maj), 4)})
    perm = inst.copy()
    perm[tr] = perm[tr][rng.permutation(len(tr))]
    rows.append(fit(feats, perm, "embedding -> shuffled institution"))

    res = pd.DataFrame(rows)
    res.to_csv(out / "linear_probes.csv", index=False)
    print(res.to_string(index=False))
    emb = res.loc[res.probe == "embedding -> institution", "test_accuracy"].iloc[0]
    gen = res.loc[res.probe == "genus one-hot -> institution", "test_accuracy"].iloc[0]
    print(f"\n  Embedding beats genus-only by {emb - gen:+.3f}. A small gap means the "
          f"retrieval institution lift was taxon confounding; a large one means the "
          f"sheet itself carries herbarium identity.")
    print(f"  Wrote linear_probes.csv")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp, needs_ckpt: bool):
        sp.add_argument("--out", required=True, help="Directory holding the feature cache.")
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--seed", type=int, default=42)
        if needs_ckpt:
            sp.add_argument("--checkpoint", required=True)
            sp.add_argument("--image-sz", type=int, default=640)
            sp.add_argument("--batch-size", type=int, default=12)
            sp.add_argument("--num-workers", type=int, default=4)

    m = sub.add_parser("mask", help="Blank the plant / the sheet; re-run the head.")
    common(m, True)
    m.add_argument("--fit-images", type=int, default=200)
    m.add_argument("--tokens-per-image", type=int, default=256)
    m.add_argument("--limit", type=int, default=0)
    m.set_defaults(func=stage_mask)

    g = sub.add_parser("geo", help="Zero / permute the geo vector.")
    common(g, True)
    g.add_argument("--min-country", type=int, default=30)
    g.set_defaults(func=stage_geo)

    b = sub.add_parser("probe", help="Linear probe to institutionCode.")
    common(b, False)
    b.add_argument("--max-iter", type=int, default=2000)
    b.add_argument("--C", type=float, default=1.0)
    b.set_defaults(func=stage_probe)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
