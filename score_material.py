#!/usr/bin/env python
"""Score how much plant tissue each herbarium sheet carries, and flag the sheets
that carry too little to identify ("just a stick").

    tissue_frac   share of the visible sheet covered by plant tissue
    low_material  tissue_frac < --cutoff

Training skips low_material rows the way it skips invalid ones, so they also drop
out of the held-out split; identify still predicts them but marks them. On Diospyros
the 2% of sheets under 5.8% were identified at 0.71 against 0.93 for the rest.

How a sheet is scored (both steps are needed):
  1. Specimen region. PC1 of the model's patch tokens is a foreground axis: it
     lights up the plant and stays dark on labels, barcodes, rulers and colour
     charts. One threshold (Otsu over a sample of sheets) is used for every sheet,
     because a per-sheet threshold always finds a "plant" half, even on a bare
     stick. On its own PC1 is a halo: patch tokens carry their neighbours'
     context, so paper between the leaves counts as specimen too.
  2. Tissue. Inside that region, a pixel counts only if its CIELAB colour is more
     than --delta-e from the sheet's own paper colour (the median over the
     central 60% of the sheet, outside the region). The share is taken of the
     visible sheet, paper plus tissue, so scanner background and the ruler and
     colour-chart strip beside the sheet do not count.

The whole sheet is scored, upright and uncropped (identify's square centre crop
loses the top and bottom of a portrait sheet).

LIMITS. This is an AREA score, so it suits broad-leaved trees and shrubs. A
complete sedge, grass or small herb also covers little of its sheet and would be
flagged. The cutoff is tied to the model that computed PC1: 5.8% was calibrated
with the Diospyros ViT-L checkpoint, so rescore and recheck it on any other model.
Brown fragment packets partly count as tissue.

Usage:
    python score_material.py \\
        --checkpoint runs/checkpoints/acc-epoch=04-val_Accuracy=0.9321.ckpt \\
        --sources specsin.csv:images_cropped/ \\
        --cutoff 0.058 --write
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from identify_herbarium import (IMAGENET_MEAN, IMAGENET_STD, build_model_from_state,
                                load_model, resolve_checkpoint)

Image.MAX_IMAGE_PIXELS = None
H, W = 928, 640                       # portrait, both multiples of the 16 px patch
GH, GW = H // 16, W // 16


def _upright(path: str, max_side: int) -> Image.Image:
    im = Image.open(path)
    im.draft("RGB", (max_side, max_side))   # JPEG DCT downscale: never decode the full scan
    im = im.convert("RGB")
    return im.rotate(90, expand=True) if im.width > im.height else im


class _Sheets(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.tf = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(_upright(self.paths[i], 2 * H))


def _otsu(v: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(v, bins=bins)
    c = (edges[:-1] + edges[1:]) / 2
    w = hist.cumsum()
    m = (hist * c).cumsum()
    tot = w[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = np.nan_to_num((m[-1] * w / tot - m) ** 2 / (w * (tot - w) / tot ** 2 + 1e-12))
    return float(c[int(np.argmax(between))])


@torch.inference_mode()
def _patch_tokens(model, paths, args, device, callback) -> None:
    loader = DataLoader(_Sheets(paths), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    prefix = int(getattr(model, "num_prefix_tokens", 1))
    for k, x in enumerate(loader):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            t = model.forward_features(x.to(device, non_blocking=True))[:, prefix:]
        callback(t.float().cpu().numpy())
        if k % 50 == 0:
            print(f"  batch {k}/{len(loader)}", flush=True)


def pc1_maps(model, paths, args, device):
    """[N, GH, GW] PC1 scores (high = specimen) and the one threshold for all sheets."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(args.seed)
    fit = [paths[i] for i in rng.choice(len(paths), min(args.fit_images, len(paths)), replace=False)]
    bag: list[np.ndarray] = []

    def sample(t):
        flat = t.reshape(-1, t.shape[-1])
        bag.append(flat[rng.choice(len(flat), 300 * t.shape[0], replace=False)])

    print(f"Fitting PC1 on {len(fit)} sheets...")
    _patch_tokens(model, fit, args, device, sample)
    tokens = np.concatenate(bag)
    mean = tokens.mean(0, keepdims=True)
    pca = PCA(1, svd_solver="randomized", random_state=args.seed).fit(tokens - mean)
    del bag, tokens

    print(f"Projecting {len(paths):,} sheets...")
    maps: list[np.ndarray] = []
    _patch_tokens(model, paths, args, device, lambda t: maps.extend(
        pca.transform(t.reshape(-1, t.shape[-1]) - mean).reshape(t.shape[0], GH, GW)))
    pc1 = np.stack(maps).astype(np.float32)
    # PCA signs are arbitrary; the paper dominates a scan's border, the specimen its middle.
    border = np.concatenate([pc1[:, 0], pc1[:, -1], pc1[:, :, 0], pc1[:, :, -1]], axis=1).mean()
    interior = pc1[:, GH // 4:3 * GH // 4, GW // 4:3 * GW // 4].mean()
    if border > interior:
        pc1 = -pc1
    thr = _otsu(pc1[rng.choice(len(pc1), min(1000, len(pc1)), replace=False)].ravel())
    return pc1, thr


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIELAB (D65), the same conversion as skimage.color.rgb2lab."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def _tissue(job) -> float:
    path, region_patches, delta_e = job
    im = _upright(path, 2 * H).resize((W, H), Image.BILINEAR)
    lab = _srgb_to_lab(np.asarray(im, dtype=np.float32) / 255.0)
    region = np.kron(region_patches, np.ones((16, 16), bool))
    centre = np.zeros((H, W), bool)
    centre[int(H * .2):int(H * .8), int(W * .2):int(W * .8)] = True
    sample = lab[centre & ~region]
    if len(sample) < 1000:
        sample = lab[~region] if (~region).sum() > 1000 else lab.reshape(-1, 3)
    dist = np.linalg.norm(lab - np.median(sample, axis=0), axis=2)
    tissue = (dist > delta_e) & region
    sheet = (dist <= delta_e) | tissue
    return float(tissue.sum() / max(sheet.sum(), 1))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None, metavar="CKPT|DIR",
                   help="Trained checkpoint whose backbone supplies the PC1 foreground axis.")
    p.add_argument("--sources", nargs="+", default=None, metavar="CSV:DIR")
    p.add_argument("--cutoff", type=float, default=0.058,
                   help="low_material below this tissue share (default 0.058, calibrated on "
                        "Diospyros with the ViT-L checkpoint)")
    p.add_argument("--delta-e", type=float, default=20.0,
                   help="CIELAB distance from the paper colour that counts as tissue")
    p.add_argument("--write", action="store_true",
                   help="Write tissue_frac and low_material into each specsin (backed up first). "
                        "Without it, only the report is written.")
    p.add_argument("--from-report", type=Path, default=None, metavar="CSV",
                   help="Skip scoring and apply an earlier report at --cutoff (e.g. to move the "
                        "cutoff without rescoring)")
    p.add_argument("--report", type=Path, default=None,
                   help="Per-sheet CSV report (default: material_scores.csv beside the first specsin)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=16, help="DataLoader and tissue-pass processes")
    p.add_argument("--fit-images", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def write_specsin(sheets: pd.DataFrame, cutoff: float) -> None:
    """Add tissue_frac / low_material to each specsin, leaving every existing value
    byte-for-byte as it was (read as text: no lost leading zeros, no 1234.0 IDs)."""
    for csv_path, g in sheets.groupby("specsin"):
        csv_path = Path(csv_path)
        backup = csv_path.with_name(f"{csv_path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(csv_path, backup)
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        score = dict(zip(g["fname"], g["tissue_frac"]))
        frac = df["fname"].map(score)
        df["tissue_frac"] = frac.map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        # Rows without a score (no file) are never low_material: they are already
        # excluded by hasfile, and a missing score must not read as "a stick".
        df["low_material"] = np.where(frac.lt(cutoff).fillna(False), "True", "False")
        df.to_csv(csv_path, index=False)
        print(f"specsin updated → {csv_path}  ({int((df['low_material'] == 'True').sum()):,} "
              f"low_material at cutoff {cutoff}; backup {backup.name})")


def main():
    args = parse_args()
    if args.from_report:
        sheets = pd.read_csv(args.from_report)
        print(f"Applying {args.from_report} at cutoff {args.cutoff}: "
              f"{int((sheets['tissue_frac'] < args.cutoff).sum()):,} of {len(sheets):,} low_material")
        if args.write:
            write_specsin(sheets, args.cutoff)
        else:
            print("Dry run: pass --write to record tissue_frac / low_material in specsin.")
        return
    if args.checkpoint is None or not args.sources:
        sys.exit("--checkpoint and --sources are required unless --from-report is given.")
    frames = []
    for src in args.sources:
        csv_path, img_dir = (Path(s) for s in src.split(":", 1))
        df = pd.read_csv(csv_path)
        df = df[df["hasfile"].astype(str).str.lower().isin(("true", "1"))]
        df = df.assign(abs_path=[str(img_dir / f) for f in df["fname"]], specsin=str(csv_path))
        exists = df["abs_path"].map(lambda q: Path(q).is_file())
        if (~exists).any():
            print(f"  [warn] {int((~exists).sum())} rows in {csv_path.name} have hasfile=True "
                  f"but no file — not scored")
        frames.append(df[exists][["fname", "abs_path", "specsin"]])
    sheets = pd.concat(frames, ignore_index=True)
    if sheets.empty:
        sys.exit("No sheets with files to score.")
    print(f"Scoring {len(sheets):,} sheets")

    ckpt = resolve_checkpoint(args.checkpoint)
    state, name, n_cls, _names, geo_dim, *_ = load_model(ckpt, [], W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_state(state, name, n_cls, geo_dim, img_size=W)
    model = (model.backbone if hasattr(model, "backbone") else model).eval().to(device)

    pc1, thr = pc1_maps(model, sheets["abs_path"].tolist(), args, device)
    print(f"Measuring tissue on {len(sheets):,} sheets...")
    jobs = [(q, pc1[i] > thr, args.delta_e) for i, q in enumerate(sheets["abs_path"])]
    with Pool(args.workers) as pool:
        sheets["tissue_frac"] = pool.map(_tissue, jobs, chunksize=16)
    sheets["low_material"] = sheets["tissue_frac"] < args.cutoff

    q = sheets["tissue_frac"].quantile([.01, .05, .1, .5, .9])
    print(f"tissue share: p1 {q[.01]:.3f}  p5 {q[.05]:.3f}  p10 {q[.1]:.3f}  "
          f"median {q[.5]:.3f}  p90 {q[.9]:.3f}")
    print(f"low_material (< {args.cutoff}): {int(sheets['low_material'].sum()):,} of "
          f"{len(sheets):,} ({sheets['low_material'].mean():.1%})")

    first = Path(args.sources[0].split(":", 1)[0])
    report = args.report or first.with_name("material_scores.csv")
    sheets.drop(columns="abs_path").to_csv(report, index=False)
    meta = dict(checkpoint=str(ckpt), model=name, pc1_threshold=thr, cutoff=args.cutoff,
                delta_e=args.delta_e, scored=len(sheets), date=datetime.now().isoformat(timespec="seconds"))
    report.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Report → {report}")

    if not args.write:
        print("Dry run: pass --write to record tissue_frac / low_material in specsin.")
        return
    write_specsin(sheets, args.cutoff)


if __name__ == "__main__":
    main()
