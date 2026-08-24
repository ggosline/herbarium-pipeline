"""Probe what a trained herbarium classifier actually encodes.

Three views of one frozen backbone, sharing a single feature-extraction pass:

  1. pca-maps  — PCA of *patch tokens* (not weights) rendered as RGB over the
                 40x40 grid, beside the image the model saw. Shows where the
                 backbone puts its structure: plant vs sheet vs label.
  2. umap      — 2-D map of the pooled embedding the head sees, coloured by
                 genus / species / institutionCode / train-vs-held-out. The
                 institution panel is the cheap confound check: if herbaria
                 separate more cleanly than genera, the model is partly reading
                 the sheet, not the plant.
  3. retrieve  — cosine nearest neighbours, with species/genus/institution
                 agreement rates against their chance baselines.

Everything is read-only with respect to the project: no checkpoint, specsin or
image is modified.

Preprocessing is imported from identify_herbarium (InferenceDataset), so what
the probe sees is exactly what inference sees. That matters — a probe run
through a different resize is measuring a different model.

Usage:
  # one pass over the held-out split, cached to <out>/features.npz + meta.csv
  python probe_embeddings.py extract \
      --checkpoint /path/acc-epoch=10-val_Accuracy=0.8143.ckpt \
      --images /path/images --specsin /path/specsin.csv \
      --out ./runs/interpretability

  python probe_embeddings.py pca-maps  --out ./runs/interpretability --genus Uvaria
  python probe_embeddings.py umap      --out ./runs/interpretability
  python probe_embeddings.py retrieve  --out ./runs/interpretability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from identify_herbarium import (
    InferenceDataset,
    build_model_from_state,
    encode_coords,
    load_model,
    resolve_checkpoint,
)

Image.MAX_IMAGE_PIXELS = None

# tqdm draws with carriage returns, which turn a redirected log into one
# unreadable line — the stdout/TTY problem this project hits repeatedly. Default
# to quiet whenever stdout is not a terminal; --quiet forces it either way.
# Read through the module (probe_embeddings.QUIET), never `from ... import QUIET`,
# so a runtime override is visible to every caller.
QUIET = not sys.stdout.isatty()

META_NAME = "meta.csv"
FEAT_NAME = "features.npz"


# ---------------------------------------------------------------------------
# Selection: which specimens to probe
# ---------------------------------------------------------------------------

def index_images(dirs: list[Path]) -> dict[str, Path]:
    """basename → path, first directory wins.

    The split is recorded by *filename* (not path), because training ran on a
    pod where the images lived somewhere else entirely.
    """
    found: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"ERROR: not a directory: {d}")
        for p in sorted(d.glob("*.jpg")):
            found.setdefault(p.name, p)
    if not found:
        raise SystemExit(f"ERROR: no .jpg found in {[str(d) for d in dirs]}")
    return found


def select(df: pd.DataFrame, images: dict[str, Path], split_payload: dict,
           want_split: str, per_class: int, limit: int, seed: int) -> pd.DataFrame:
    """Join specsin to files on disk, tag train/valid, subsample."""
    valid = set(split_payload.get("valid") or [])
    train = set(split_payload.get("train") or [])
    if not valid:
        print("  [warn] checkpoint records no split — every metric below mixes in "
              "images the model memorised. Treat cluster tightness and retrieval "
              "accuracy as upper bounds, not estimates.")

    df = df[df["fname"].isin(images)].copy()
    df["path"] = df["fname"].map(lambda n: str(images[n]))
    df["split"] = np.where(df["fname"].isin(valid), "valid",
                  np.where(df["fname"].isin(train), "train", "unknown"))
    if want_split != "all":
        df = df[df["split"] == want_split]
        if df.empty:
            raise SystemExit(f"ERROR: no images on disk for split '{want_split}'.")

    rng = np.random.RandomState(seed)
    if per_class > 0:
        df = (df.groupby("species", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed)))
    if limit > 0 and len(df) > limit:
        df = df.iloc[rng.permutation(len(df))[:limit]]
    return df.sort_values("fname").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model + features
# ---------------------------------------------------------------------------

def load_backbone(ckpt_path: Path, device: torch.device):
    """Rebuild the trained model and hand back its backbone plus metadata."""
    ckpt = resolve_checkpoint(Path(ckpt_path))
    (state, model_name, num_classes, nameslist, geo_dim, label_level,
     temperature, excluded, class_counts, genus_head, split) = load_model(ckpt, [], 0)
    if not model_name:
        raise SystemExit("ERROR: checkpoint records no model_name; pass --model.")
    model = build_model_from_state(state, model_name, num_classes, geo_dim)
    backbone = getattr(model, "backbone", model)
    geo_mlp = getattr(model, "geo_mlp", None)
    backbone = backbone.eval().to(device)
    if geo_mlp is not None:
        geo_mlp = geo_mlp.eval().to(device)
    meta = {"model_name": model_name, "num_classes": num_classes,
            "nameslist": nameslist, "geo_dim": geo_dim, "label_level": label_level,
            "split": split, "checkpoint": str(ckpt)}
    return backbone, geo_mlp, meta


def grid_shape(backbone, image_sz: int) -> tuple[int, int]:
    patch = backbone.patch_embed.patch_size[0]
    if image_sz % patch:
        raise SystemExit(f"ERROR: image-sz {image_sz} is not a multiple of the "
                         f"patch size {patch}; the token grid would not be square.")
    return image_sz // patch, image_sz // patch


def split_tokens(backbone, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """Drop CLS + register tokens, leaving [B, gh*gw, D] in raster order.

    DINOv3 carries 4 register tokens on top of CLS (num_prefix_tokens == 5).
    Slicing off only CLS shears the map by one column — the failure is subtle
    enough to look like a real diagonal structure, so assert instead.
    """
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    patches = tokens[:, prefix:, :]
    if patches.shape[1] != gh * gw:
        raise SystemExit(
            f"ERROR: {tokens.shape[1]} tokens - {prefix} prefix = {patches.shape[1]}, "
            f"but the grid is {gh}x{gw} = {gh * gw}. Wrong --image-sz for this model?")
    return patches


@torch.inference_mode()
def extract(backbone, paths: list[str], image_sz: int, batch_size: int,
            device: torch.device, workers: int = 4,
            token_cb=None, tokens_per_image: int = 0, seed: int = 42):
    """Pooled embeddings for every path; optionally stream patch tokens out.

    Patch tokens are never all held at once — 2,000 images x 1,600 tokens x
    1,024 dims is 6 GB in fp16. `token_cb` receives each batch so callers can
    subsample (PCA fitting) or render (maps) and drop them.
    """
    ds = InferenceDataset([Path(p) for p in paths], image_sz, None)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        pin_memory=True, shuffle=False)
    gh, gw = grid_shape(backbone, image_sz)
    rng = np.random.RandomState(seed)
    pooled_all = []
    for batch, _, _ in tqdm(loader, desc="Extracting", unit="batch", disable=QUIET):
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            tokens = backbone.forward_features(batch)
            pooled = backbone.forward_head(tokens, pre_logits=True)
        pooled_all.append(pooled.float().cpu().numpy().astype(np.float16))
        if token_cb is not None:
            patches = split_tokens(backbone, tokens, gh, gw).float()
            if tokens_per_image:
                idx = rng.choice(gh * gw, size=min(tokens_per_image, gh * gw),
                                 replace=False)
                patches = patches[:, idx, :]
            token_cb(patches.cpu().numpy())
    return np.concatenate(pooled_all, axis=0)


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------

def stage_extract(args) -> None:
    device = torch.device(args.device)
    backbone, geo_mlp, meta = load_backbone(args.checkpoint, device)

    df_all = pd.read_csv(args.specsin, low_memory=False)
    images = index_images([Path(d) for d in args.images])
    df = select(df_all, images, meta["split"], args.split,
                args.per_class, args.limit, args.seed)
    print(f"  Probing {len(df):,} specimens "
          f"({df['species'].nunique()} species, {df['split'].value_counts().to_dict()})")

    feats = extract(backbone, df["path"].tolist(), args.image_sz,
                    args.batch_size, device, args.num_workers)

    # Geo features are computed straight from coordinates — no image pass — so
    # both the image-only and the head's-eye (fused) view are available later
    # without re-running the backbone.
    geo_feats = np.zeros((len(df), 0), dtype=np.float16)
    if geo_mlp is not None:
        geo_vec = encode_coords(df["decimalLatitude"], df["decimalLongitude"])
        with torch.inference_mode():
            geo_feats = geo_mlp(geo_vec.to(device)).float().cpu().numpy().astype(np.float16)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / FEAT_NAME, pooled=feats, geo=geo_feats)
    cols = [c for c in ("fname", "path", "species", "genus", "family",
                        "institutionCode", "countryCode", "split",
                        "decimalLatitude", "decimalLongitude") if c in df.columns]
    df[cols].to_csv(out / META_NAME, index=False)
    (out / "probe_meta.json").write_text(json.dumps(
        {k: v for k, v in meta.items() if k not in ("nameslist", "split")}
        | {"n": len(df), "image_sz": args.image_sz, "split": args.split}, indent=2))
    print(f"  Wrote {out / FEAT_NAME} ({feats.shape}) and {out / META_NAME}")


# ---------------------------------------------------------------------------
# Stage: patch-token PCA rendered as RGB
# ---------------------------------------------------------------------------

# Needs matplotlib >= 3.10. On 3.9.x under Python 3.14, Path.__deepcopy__ deep-
# copies a super() object and recurses forever whenever a tick marker is copied,
# which every layout engine triggers through get_tightbbox — so figures here die
# in savefig rather than anywhere near the code that drew them.


def model_view(path: str, image_sz: int) -> Image.Image:
    """The crop the model actually saw, for display beside its token map."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = image_sz / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.BILINEAR)
    w, h = img.size
    left, top = (w - image_sz) // 2, (h - image_sz) // 2
    return img.crop((left, top, left + image_sz, top + image_sz))


def stage_pca_maps(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    df["institutionCode"] = df["institutionCode"].fillna("(no institution)")
    if args.genus:
        df = df[df["genus"] == args.genus]
    if args.species:
        df = df[df["species"] == args.species]
    if df.empty:
        raise SystemExit("ERROR: filter left no specimens.")

    device = torch.device(args.device)
    backbone, _, meta = load_backbone(args.checkpoint, device)
    gh, gw = grid_shape(backbone, args.image_sz)

    rng = np.random.RandomState(args.seed)
    fit_df = df.iloc[rng.permutation(len(df))[:args.fit_images]]
    bag: list[np.ndarray] = []
    extract(backbone, fit_df["path"].tolist(), args.image_sz, args.batch_size,
            device, args.num_workers, token_cb=lambda t: bag.append(
                t.reshape(-1, t.shape[-1]).astype(np.float32)),
            tokens_per_image=args.tokens_per_image, seed=args.seed)
    tokens = np.concatenate(bag, axis=0)
    print(f"  PCA fit on {tokens.shape[0]:,} patch tokens x {tokens.shape[1]} dims")

    mean = tokens.mean(axis=0, keepdims=True)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=args.seed)
    pca.fit(tokens - mean)
    print(f"  Explained variance (PC1-3): "
          f"{np.round(pca.explained_variance_ratio_[:3], 3).tolist()}")

    # Scale each component by global percentiles, not per-image ones: per-image
    # scaling makes every sheet look equally structured and destroys any
    # comparison between them.
    proj = pca.transform(tokens - mean)
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)

    render_df = df.iloc[rng.permutation(len(df))[:args.render_images]]
    maps: list[np.ndarray] = []
    extract(backbone, render_df["path"].tolist(), args.image_sz, args.batch_size,
            device, args.num_workers,
            token_cb=lambda t: maps.extend(list(t)), tokens_per_image=0)

    ncol = 4
    nrow = int(np.ceil(len(render_df) / (ncol // 2)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3.1))
    axes = np.atleast_2d(axes)
    for i, (_, row) in enumerate(render_df.reset_index(drop=True).iterrows()):
        r, c = divmod(i, ncol // 2)
        rgb = (pca.transform(maps[i].astype(np.float32) - mean) - lo) / (hi - lo)
        rgb = np.clip(rgb, 0, 1).reshape(gh, gw, 3)
        axes[r][c * 2].imshow(model_view(row["path"], args.image_sz))
        axes[r][c * 2].set_title(f"{row['species']}\n{row['institutionCode']}",
                                 fontsize=6)
        axes[r][c * 2 + 1].imshow(rgb, interpolation="nearest")
        axes[r][c * 2 + 1].set_title("PC1-3 patch tokens", fontsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    name = args.genus or args.species or "sample"
    dest = out / f"pca_maps_{name.replace(' ', '_')}.png"
    fig.savefig(dest, dpi=args.dpi)
    plt.close(fig)
    print(f"  Wrote {dest}")

    # PC1 alone, as a grayscale panel: the clearest read on whether the backbone
    # separates specimen from sheet at all.
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3.1))
    axes = np.atleast_2d(axes)
    for i, (_, row) in enumerate(render_df.reset_index(drop=True).iterrows()):
        r, c = divmod(i, ncol // 2)
        p1 = (pca.transform(maps[i].astype(np.float32) - mean)[:, 0] - lo[0]) / (hi[0] - lo[0])
        axes[r][c * 2].imshow(model_view(row["path"], args.image_sz))
        axes[r][c * 2].set_title(row["species"], fontsize=6)
        axes[r][c * 2 + 1].imshow(np.clip(p1, 0, 1).reshape(gh, gw),
                                  cmap="magma", interpolation="nearest")
        axes[r][c * 2 + 1].set_title("PC1", fontsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    dest = out / f"pca_pc1_{name.replace(' ', '_')}.png"
    fig.savefig(dest, dpi=args.dpi)
    plt.close(fig)
    print(f"  Wrote {dest}")


# ---------------------------------------------------------------------------
# Stage: UMAP of the pooled embedding
# ---------------------------------------------------------------------------

def load_features(out: Path, fused: bool) -> tuple[np.ndarray, pd.DataFrame]:
    npz = np.load(out / FEAT_NAME)
    feats = npz["pooled"].astype(np.float32)
    if fused:
        geo = npz["geo"].astype(np.float32)
        if geo.shape[1] == 0:
            raise SystemExit("ERROR: --fused needs a geo-capable checkpoint; "
                             "this one stored no geo features.")
        feats = np.hstack([feats, geo])
    return feats, pd.read_csv(out / META_NAME)


def project_2d(feats: np.ndarray, seed: int, neighbors: int, min_dist: float):
    """UMAP if installed, t-SNE otherwise. Cosine metric either way."""
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist,
                            metric="cosine", random_state=seed)
        return reducer.fit_transform(feats), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        print("  [note] umap-learn not installed — falling back to t-SNE "
              "(`pip install umap-learn` for the intended projection).")
        ts = TSNE(n_components=2, metric="cosine", init="pca",
                  perplexity=min(30, max(5, len(feats) // 100)), random_state=seed)
        return ts.fit_transform(feats), "t-SNE"


def scatter_by(ax, xy: np.ndarray, labels: pd.Series, top: int, title: str) -> None:
    """Colour the top-N most frequent labels; everything else stays grey.

    With 261 species a full legend is unreadable and every colour is reused
    several times — which invents structure that isn't there.
    """
    import matplotlib.pyplot as plt
    counts = labels.value_counts()
    keep = list(counts.index[:top])
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="0.85", linewidths=0)
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(keep):
        m = (labels == name).values
        ax.scatter(xy[m, 0], xy[m, 1], s=5, color=cmap(i % 20),
                   label=f"{name} ({counts[name]})", linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=4, markerscale=1.6, loc="best", frameon=False)


def stage_umap(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    feats, df = load_features(out, args.fused)
    print(f"  Projecting {feats.shape[0]:,} x {feats.shape[1]} "
          f"({'image+geo' if args.fused else 'image only'})")
    xy, how = project_2d(feats, args.seed, args.neighbors, args.min_dist)

    tag = "fused" if args.fused else "image"
    coords = df.copy()
    coords["x"], coords["y"] = xy[:, 0], xy[:, 1]
    coords.to_csv(out / f"projection_{tag}.csv", index=False)

    # A panel whose column holds one value everywhere says nothing — the split
    # panel is blank whenever extract ran on a single split, which is the norm.
    panels = [("genus", args.top_labels), ("species", args.top_labels),
              ("institutionCode", args.top_labels), ("split", 3),
              ("countryCode", args.top_labels)]
    panels = [(c, n) for c, n in panels
              if c in df.columns and df[c].fillna("(missing)").nunique() > 1][:4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (col, top) in zip(axes.ravel(), panels):
        scatter_by(ax, xy, df[col].fillna("(missing)").astype(str), top,
                   f"{how} of pooled embedding — {col}")
    fig.tight_layout()
    dest = out / f"projection_{tag}.png"
    fig.savefig(dest, dpi=args.dpi)
    plt.close(fig)
    print(f"  Wrote {dest} and projection_{tag}.csv")


# ---------------------------------------------------------------------------
# Stage: nearest-neighbour retrieval
# ---------------------------------------------------------------------------

def neighbour_table(feats: np.ndarray, k: int, device: torch.device) -> np.ndarray:
    """Cosine top-k neighbour indices, self excluded."""
    x = torch.from_numpy(feats).to(device)
    x = torch.nn.functional.normalize(x, dim=1)
    sims = x @ x.T
    sims.fill_diagonal_(-2.0)
    return torch.topk(sims, k=k, dim=1).indices.cpu().numpy()


def agreement(df: pd.DataFrame, nn: np.ndarray, col: str) -> tuple[float, float]:
    """(top-1 neighbour agreement, chance baseline) for a metadata column."""
    vals = df[col].fillna("(missing)").astype(str).values
    hit = float((vals[nn[:, 0]] == vals).mean())
    p = df[col].fillna("(missing)").astype(str).value_counts(normalize=True).values
    return hit, float((p ** 2).sum())


def stage_retrieve(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    feats, df = load_features(out, args.fused)
    df["institutionCode"] = df["institutionCode"].fillna("(no institution)")
    device = torch.device(args.device)
    nn = neighbour_table(feats, args.k, device)

    rows = []
    for col in ("species", "genus", "institutionCode", "countryCode"):
        if col in df.columns:
            hit, chance = agreement(df, nn, col)
            rows.append({"field": col, "top1_agreement": round(hit, 4),
                         "chance": round(chance, 4),
                         "lift": round(hit / chance, 1) if chance else float("nan")})
    # Variant-named, like the figure: a --fused run must not clobber the
    # image-only numbers it is meant to be compared against.
    tag = "fused" if args.fused else "image"
    stats = pd.DataFrame(rows)
    stats.to_csv(out / f"retrieval_agreement_{tag}.csv", index=False)
    print(stats.to_string(index=False))
    print("  Lift is agreement over chance. Institution lift well above 1 means "
          "the embedding carries which herbarium mounted the sheet — read the "
          "species numbers with that in mind.")

    rng = np.random.RandomState(args.seed)
    picks = rng.permutation(len(df))[:args.n_queries]
    ncol = args.k + 1
    fig, axes = plt.subplots(len(picks), ncol, figsize=(ncol * 2.1, len(picks) * 2.3))
    axes = np.atleast_2d(axes)
    for r, qi in enumerate(picks):
        q = df.iloc[qi]
        axes[r][0].imshow(model_view(q["path"], args.thumb))
        axes[r][0].set_title(f"query\n{q['species']}\n{q['institutionCode']}", fontsize=5)
        for c, ni in enumerate(nn[qi], start=1):
            n = df.iloc[ni]
            same = "species" if n["species"] == q["species"] else (
                   "genus" if n["genus"] == q["genus"] else "-")
            inst = "same inst" if n["institutionCode"] == q["institutionCode"] else ""
            axes[r][c].imshow(model_view(n["path"], args.thumb))
            axes[r][c].set_title(f"{n['species']}\n{same} {inst}", fontsize=5)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    dest = out / f"retrieval_{tag}.png"
    fig.savefig(dest, dpi=args.dpi)
    plt.close(fig)
    print(f"  Wrote {dest} and retrieval_agreement_{tag}.csv")


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp, needs_ckpt: bool):
        sp.add_argument("--out", required=True, help="Output/cache directory.")
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--dpi", type=int, default=160)
        if needs_ckpt:
            sp.add_argument("--checkpoint", required=True)
            sp.add_argument("--image-sz", type=int, default=640)
            sp.add_argument("--batch-size", type=int, default=8)
            sp.add_argument("--num-workers", type=int, default=4)

    e = sub.add_parser("extract", help="One backbone pass; caches embeddings.")
    common(e, True)
    e.add_argument("--images", nargs="+", required=True)
    e.add_argument("--specsin", required=True)
    e.add_argument("--split", default="valid", choices=["valid", "train", "all"],
                   help="Which specimens to probe. Default valid (held out).")
    e.add_argument("--per-class", type=int, default=0,
                   help="Cap images per species (0 = no cap).")
    e.add_argument("--limit", type=int, default=0, help="Cap total images.")
    e.set_defaults(func=stage_extract)

    m = sub.add_parser("pca-maps", help="Patch-token PCA rendered as RGB.")
    common(m, True)
    m.add_argument("--genus", help="Restrict to one genus.")
    m.add_argument("--species", help="Restrict to one species.")
    m.add_argument("--fit-images", type=int, default=200)
    m.add_argument("--tokens-per-image", type=int, default=256)
    m.add_argument("--render-images", type=int, default=12)
    m.set_defaults(func=stage_pca_maps)

    u = sub.add_parser("umap", help="2-D projection of the pooled embedding.")
    common(u, False)
    u.add_argument("--fused", action="store_true",
                   help="Append geo features — the head's-eye view.")
    u.add_argument("--neighbors", type=int, default=15)
    u.add_argument("--min-dist", type=float, default=0.1)
    u.add_argument("--top-labels", type=int, default=12)
    u.set_defaults(func=stage_umap)

    r = sub.add_parser("retrieve", help="Cosine nearest-neighbour retrieval.")
    common(r, False)
    r.add_argument("--fused", action="store_true")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--n-queries", type=int, default=8)
    r.add_argument("--thumb", type=int, default=320)
    r.set_defaults(func=stage_retrieve)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
