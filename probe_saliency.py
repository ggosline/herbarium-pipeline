"""Step 4 of the interpretability study: where on the sheet the evidence is.

Two spatial views, for figures:

  occlusion  — slide an occluder over the sheet and record how far the true
               class's probability falls. Model-agnostic and it measures the
               thing we actually care about (the prediction), but it costs one
               forward pass per position.
  rollout    — attention rollout (Abnar & Zuidema 2020): head-averaged
               attention per block, plus the residual identity, multiplied
               through all 24 blocks, read off the CLS row. Cheap and detailed,
               but it shows where the backbone *looks*, which is not the same
               as what the classifier *uses* — read it alongside occlusion,
               never instead of it.

Note on DINOv3: timm implements it as EvaAttention with fused SDPA, which never
materialises the attention matrix, so `fused_attn = False` must be set on every
block before hooking or the hook silently captures nothing. Its 4 register
tokens sit alongside CLS (num_prefix_tokens == 5) and are dropped before the
map is reshaped to the 40x40 grid.

Usage:
  python probe_saliency.py --checkpoint CKPT --out DIR --n 6
  python probe_saliency.py --checkpoint CKPT --out DIR --contrast edges
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import probe_embeddings as pe
from identify_herbarium import IMAGENET_MEAN, IMAGENET_STD, encode_coords
from probe_confounds import build_full_model
from probe_embeddings import META_NAME, model_view

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Occlusion sensitivity
# ---------------------------------------------------------------------------

@torch.inference_mode()
def occlusion_map(model, img: np.ndarray, geo: torch.Tensor | None, true_idx: int,
                  size: int, stride: int, batch_size: int, device) -> tuple[np.ndarray, float]:
    """Drop in p(true class) for an occluder centred at each position.

    Positive values mean 'covering this hurt the prediction', i.e. the evidence
    lives there. The occluder is filled with the sheet's own median colour so a
    covered region reads as blank mounting paper rather than as a grey box.
    """
    H, W, _ = img.shape
    fill = np.median(img.reshape(-1, 3), axis=0).astype(img.dtype)
    ys = list(range(0, H - size + 1, stride))
    xs = list(range(0, W - size + 1, stride))

    # The occluded array is already in the model's framing (resized and centre
    # cropped by model_view), so only the tensor half of the pipeline is left.
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def to_tensor(a):
        return tf(Image.fromarray(a))

    variants = [img]
    for y in ys:
        for x in xs:
            v = img.copy()
            v[y:y + size, x:x + size] = fill
            variants.append(v)

    probs = []
    for i in range(0, len(variants), batch_size):
        batch = torch.stack([to_tensor(v) for v in variants[i:i + batch_size]]).to(device)
        g = geo.repeat(len(batch), 1).to(device) if geo is not None else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(batch, g) if g is not None else model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        probs.append(torch.softmax(logits.float(), dim=1)[:, true_idx].cpu().numpy())
    probs = np.concatenate(probs)

    base = float(probs[0])
    grid = (base - probs[1:]).reshape(len(ys), len(xs))
    return grid, base


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

@torch.inference_mode()
def rollout_map(backbone, tensor: torch.Tensor, device) -> np.ndarray:
    """CLS-to-patch attention rolled through every block."""
    caught: list[torch.Tensor] = []

    def hook(_m, _i, out):
        # Head-averaged immediately: keeping [16, 1605, 1605] per block for 24
        # blocks would be ~4 GB at 640px. Returning nothing leaves the module's
        # own output untouched — a hook that returns a value replaces it.
        caught.append(out.detach().float().mean(dim=1)[0])

    for blk in backbone.blocks:
        blk.attn.fused_attn = False
    handles = [blk.attn.attn_drop.register_forward_hook(hook) for blk in backbone.blocks]
    try:
        backbone.forward_features(tensor.unsqueeze(0).to(device))
    finally:
        for h in handles:
            h.remove()
        for blk in backbone.blocks:
            blk.attn.fused_attn = True
    if not caught:
        raise SystemExit("ERROR: no attention captured — the block layout changed.")

    n = caught[0].shape[-1]
    eye = torch.eye(n, device=caught[0].device)
    roll = eye
    for a in caught:
        a = a + eye                       # residual stream carries tokens forward
        a = a / a.sum(dim=-1, keepdim=True)
        roll = a @ roll
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    cls_row = roll[0, prefix:].cpu().numpy()
    g = int(round(np.sqrt(cls_row.size)))
    return cls_row.reshape(g, g)


# ---------------------------------------------------------------------------
# Selection + rendering
# ---------------------------------------------------------------------------

def pick(df: pd.DataFrame, args, level_col: str) -> pd.DataFrame:
    """Which specimens to draw: random, one taxon, or a step-1 contrast.

    `level_col` is the rank the model predicts — species for the Annonaceae run,
    family for the Angiosperm one. Everything selected or grouped here has to
    follow it, or a family-level model gets contrasted on taxa it never scores.
    """
    rng = np.random.RandomState(args.seed)
    taxon = args.taxon or args.species
    if taxon:
        sub = df[df[level_col] == taxon]
        if sub.empty:
            raise SystemExit(f"ERROR: no held-out specimen of {level_col} '{taxon}'.")
        return sub.head(args.n)
    if args.contrast:
        per_file = Path(args.out) / "perturbation_per_taxon.csv"
        if not per_file.exists():
            raise SystemExit("ERROR: --contrast needs perturbation_per_taxon.csv "
                             "(run probe_perturbations.py first).")
        per = pd.read_csv(per_file)
        if args.contrast not in per.columns:
            raise SystemExit(f"ERROR: '{args.contrast}' is not a condition in "
                             f"perturbation_per_taxon.csv. Options: "
                             f"{', '.join(c for c in per.columns if c not in ('taxon', 'n'))}")
        per = per[per["n"] >= 8]
        half = max(1, args.n // 2)
        # Half the figure from taxa that survive the condition, half from
        # taxa it destroys — the point is to see whether the maps differ.
        names = (list(per.nlargest(half, args.contrast).taxon)
                 + list(per.nsmallest(half, args.contrast).taxon))
        # Keep several candidates per taxon; the caller narrows them to the
        # most confident one. An occlusion map is meaningless on a specimen the
        # model was never confident about — there is no probability to lose.
        rows = [df[df[level_col] == s].head(args.candidates) for s in names]
        out = pd.concat([r for r in rows if not r.empty])
        out["contrast_group"] = np.where(out[level_col].isin(
            per.nlargest(half, args.contrast).taxon), "robust", "fragile")
        return out
    return df.iloc[rng.permutation(len(df))[:args.n]]


@torch.inference_mode()
def most_confident(model, sel: pd.DataFrame, geo_all, names: dict, args, device,
                   level_col: str) -> pd.DataFrame:
    """One specimen per taxon: the one the model is most sure of."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    scores = []
    for i, row in sel.reset_index(drop=True).iterrows():
        img = model_view(row["path"], args.image_sz)
        batch = tf(img).unsqueeze(0).to(device)
        g = geo_all[i:i + 1].to(device) if geo_all is not None else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(batch, g) if g is not None else model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        idx = names.get(row[level_col], -1)
        p = torch.softmax(logits.float(), dim=1)[0, idx].item() if idx >= 0 else 0.0
        scores.append(p)
    sel = sel.reset_index(drop=True).assign(p_true=scores)
    kept = sel.sort_values("p_true", ascending=False).groupby(level_col, as_index=False).head(1)
    print(f"  Picked {len(kept)} specimens, p(true) "
          f"{kept.p_true.min():.2f}-{kept.p_true.max():.2f}")
    return kept.sort_values("contrast_group", ascending=False)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-sz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=6, help="Specimens to draw.")
    p.add_argument("--taxon", help="Restrict to one taxon at the model's own rank "
                                   "(species or family, depending on the checkpoint).")
    p.add_argument("--species", help="Alias for --taxon, kept for existing commands.")
    p.add_argument("--contrast", help="Split the figure by a step-1 condition, "
                                      "e.g. 'edges': robust species vs fragile ones.")
    p.add_argument("--candidates", type=int, default=4,
                   help="Specimens per taxon to score before keeping the most "
                        "confident one (--contrast only).")
    p.add_argument("--occ-size", type=int, default=96)
    p.add_argument("--occ-stride", type=int, default=48)
    p.add_argument("--mode", default="both", choices=["both", "occlusion", "rollout"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--rollout-clip", type=float, default=99.0,
                   help="Percentile to clip the rollout map at. DINOv3 grows a few "
                        "very high-norm outlier tokens that otherwise own the whole "
                        "colour range and leave the rest of the sheet flat.")
    args = p.parse_args(argv)

    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    device = torch.device(args.device)
    model, nameslist, temperature, geo_dim, label_level = build_full_model(
        args.checkpoint, device)
    backbone = getattr(model, "backbone", model)
    names = {n: i for i, n in enumerate(nameslist)}
    level_col = label_level if label_level in df.columns else "species"
    sel = pick(df, args, level_col).reset_index(drop=True)

    geo_all = (encode_coords(sel["decimalLatitude"], sel["decimalLongitude"])
               if geo_dim else None)

    if "contrast_group" in sel.columns and args.candidates > 1:
        sel = most_confident(model, sel, geo_all, names, args, device,
                             level_col).reset_index(drop=True)
        geo_all = (encode_coords(sel["decimalLatitude"], sel["decimalLongitude"])
                   if geo_dim else None)

    ncol = 1 + (args.mode in ("both", "occlusion")) + (args.mode in ("both", "rollout"))
    fig, axes = plt.subplots(len(sel), ncol, figsize=(ncol * 3.2, len(sel) * 3.4))
    axes = np.atleast_2d(axes)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    # Collected so the maps survive as numbers, not only as rendered pixels.
    # The figure normalises each occlusion map to its own maximum, which makes
    # colours incomparable between specimens and sizes unreadable anywhere —
    # so any claim about how large a negative cell is has to come from here.
    saved = {"fname": [], "taxon": [], "p_true": [], "occlusion": [], "rollout": []}

    for i, row in sel.iterrows():
        img = np.asarray(model_view(row["path"], args.image_sz))
        true_idx = names.get(row[level_col], -1)
        saved["fname"].append(row["fname"])
        saved["taxon"].append(row[level_col])
        geo = geo_all[i:i + 1] if geo_all is not None else None
        col = 0
        group = f" [{row['contrast_group']}]" if "contrast_group" in sel.columns else ""
        axes[i][col].imshow(img)
        axes[i][col].set_title(f"{row[level_col]}{group}", fontsize=6)
        col += 1

        if args.mode in ("both", "occlusion"):
            if true_idx < 0:
                axes[i][col].set_title("species not in nameslist", fontsize=6)
            else:
                grid, base = occlusion_map(model, img, geo, true_idx, args.occ_size,
                                           args.occ_stride, args.batch_size, device)
                axes[i][col].imshow(img, alpha=0.45)
                lim = max(abs(grid).max(), 1e-6)
                axes[i][col].imshow(np.kron(grid, np.ones((args.occ_stride, args.occ_stride))),
                                    cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55,
                                    extent=(0, args.image_sz, args.image_sz, 0))
                axes[i][col].set_title(f"occlusion  p(true)={base:.2f}", fontsize=6)
                saved["occlusion"].append(grid)
                saved["p_true"].append(base)
            col += 1

        if args.mode in ("both", "rollout"):
            r = rollout_map(backbone, tf(Image.fromarray(img)), device)
            flat = np.sort(r.ravel())[::-1]
            concentration = float(flat[:5].sum() / max(r.sum(), 1e-9))
            hi = np.percentile(r, args.rollout_clip)
            axes[i][col].imshow(img, alpha=0.45)
            axes[i][col].imshow(np.clip(r, 0, hi), cmap="inferno", alpha=0.55,
                                interpolation="bilinear",
                                extent=(0, args.image_sz, args.image_sz, 0))
            axes[i][col].set_title(f"attention rollout (top-5 tokens hold "
                                   f"{concentration:.0%})", fontsize=6)
            saved["rollout"].append(r)
        print(f"  {row['species']} done", flush=True)

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    tag = args.contrast or args.species or "sample"
    dest = out / f"saliency_{tag.replace(' ', '_')}.png"
    fig.savefig(dest, dpi=args.dpi)
    plt.close(fig)

    maps = dest.with_name(dest.stem + "_maps.npz")
    np.savez_compressed(
        maps,
        fname=np.array(saved["fname"]),
        taxon=np.array(saved["taxon"]),
        p_true=np.array(saved["p_true"], dtype=np.float32),
        # Stacked only when every specimen produced one; a row skipped for a
        # missing true class would otherwise silently shift the alignment.
        occlusion=(np.stack(saved["occlusion"]).astype(np.float32)
                   if len(saved["occlusion"]) == len(saved["fname"])
                   else np.array([], dtype=np.float32)),
        rollout=(np.stack(saved["rollout"]).astype(np.float32)
                 if len(saved["rollout"]) == len(saved["fname"])
                 else np.array([], dtype=np.float32)),
        occ_size=args.occ_size, occ_stride=args.occ_stride, image_sz=args.image_sz,
    )
    print(f"  Wrote {dest}")
    print(f"  Wrote {maps} (occlusion values are drops in p(true): "
          f"positive = evidence, negative = covering it RAISED p(true))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
