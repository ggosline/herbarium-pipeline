"""Step 1 of the interpretability study: which image cues the classifier needs.

A perturbation battery over the held-out split. Each condition destroys one
kind of information and leaves the rest, and is scored on the trained species
head against the unperturbed baseline:

  colour      grayscale, hue rotation, saturation scaling
  scale       Gaussian blur ladder, resolution ladder
  texture     phase scramble (keeps the amplitude spectrum, destroys structure)
  layout      patch shuffle at several block sizes
  shape       Sobel edges, and the PC1 silhouette alone

Perturbations are applied AFTER the resize/centre-crop that inference uses and
BEFORE normalisation, so every condition sees the same framing the model sees
and differs only in the cue under test. Geo is held at its real values
throughout — the geo channel is step 2's subject, not this one.

Usage:
  python probe_perturbations.py --checkpoint CKPT --out DIR
  python probe_perturbations.py --checkpoint CKPT --out DIR --only blur,phase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from tqdm import tqdm

from identify_herbarium import InferenceDataset, encode_coords
from probe_confounds import build_full_model, score, truth_labels
import probe_embeddings as pe
from probe_embeddings import META_NAME

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Perturbations: each takes/returns a uint8 HxWx3 RGB array
# ---------------------------------------------------------------------------

def to_gray(a, _):
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[:, :, None], 3, axis=2)


def hue_shift(a, deg):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + int(deg / 2)) % 180   # OpenCV hue is 0-179
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def saturate(a, factor):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def blur(a, sigma):
    return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(sigma)))


def resample(a, size):
    """Down to `size`, back to full: destroys detail finer than the round trip."""
    img = Image.fromarray(a)
    full = img.size
    return np.asarray(img.resize((size, size), Image.BILINEAR).resize(full, Image.BILINEAR))


def phase_scramble(a, seed):
    """Randomise the Fourier phase, keep the amplitude spectrum.

    One shared random phase field across channels, not three independent ones:
    per-channel scrambling also destroys the colour correlations, which would
    confound this with the colour conditions.
    """
    rng = np.random.RandomState(seed)
    h, w, _ = a.shape
    noise = rng.uniform(0, 2 * np.pi, size=(h, w))
    noise = (noise - noise[::-1, ::-1]) / 2          # antisymmetric -> real output
    out = np.empty_like(a, dtype=np.float32)
    for c in range(3):
        f = np.fft.fft2(a[:, :, c].astype(np.float32))
        out[:, :, c] = np.real(np.fft.ifft2(np.abs(f) * np.exp(1j * (np.angle(f) + noise))))
    lo, hi = out.min(), out.max()
    return ((out - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)


def patch_shuffle(a, block, seed=0):
    """Permute square blocks: keeps local texture, destroys global arrangement."""
    rng = np.random.RandomState(seed)
    h, w, c = a.shape
    gh, gw = h // block, w // block
    tiles = (a[:gh * block, :gw * block]
             .reshape(gh, block, gw, block, c).swapaxes(1, 2).reshape(gh * gw, block, block, c))
    tiles = tiles[rng.permutation(len(tiles))]
    out = tiles.reshape(gh, gw, block, block, c).swapaxes(1, 2).reshape(gh * block, gw * block, c)
    canvas = a.copy()
    canvas[:gh * block, :gw * block] = out
    return canvas


def edges(a, _):
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(sx, sy)
    m = 255 - np.clip(m / max(m.max(), 1e-6) * 255 * 3, 0, 255)   # dark lines on white
    return np.repeat(m.astype(np.uint8)[:, :, None], 3, axis=2)


PERTURBATIONS = {
    "grayscale":   (to_gray, [None]),
    "hue":         (hue_shift, [60, 120]),
    "saturation":  (saturate, [0.5, 1.5]),
    "blur":        (blur, [1, 2, 4, 8, 16]),
    "resolution":  (resample, [320, 224, 160, 112, 64]),
    "phase":       (phase_scramble, [0]),
    "shuffle":     (patch_shuffle, [16, 32, 64, 128]),
    "edges":       (edges, [None]),
}


# ---------------------------------------------------------------------------
# Dataset + runner
# ---------------------------------------------------------------------------

class PerturbedDataset(InferenceDataset):
    """InferenceDataset with one perturbation applied inside the model's framing."""

    def __init__(self, paths, image_sz, geo_coords, fn=None, param=None,
                 silhouette: np.ndarray | None = None):
        super().__init__(paths, image_sz, geo_coords)
        self.fn = fn
        self.param = param
        self.silhouette = silhouette

    def __getitem__(self, idx):
        path = self.paths[idx]
        geo = self.geo[idx] if self.geo is not None else torch.zeros(4)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return torch.zeros(3, self.image_sz, self.image_sz), str(path), geo
        for t in self.transform.transforms[:2]:      # Resize, CenterCrop
            img = t(img)
        if self.silhouette is not None:
            m = np.array(Image.fromarray(self.silhouette[idx].astype(np.uint8) * 255)
                         .resize((self.image_sz, self.image_sz), Image.NEAREST)) > 127
            arr = np.full((self.image_sz, self.image_sz, 3), 255, dtype=np.uint8)
            arr[m] = 0
            img = Image.fromarray(arr)
        elif self.fn is not None:
            arr = np.asarray(img)
            # Seeded per image so the condition is reproducible but not identical
            # across specimens (a single fixed permutation would be a systematic
            # rearrangement the model could partly undo).
            if self.fn is patch_shuffle:
                arr = patch_shuffle(arr, self.param, idx)
            elif self.fn is phase_scramble:
                arr = phase_scramble(arr, idx)
            else:
                arr = self.fn(arr, self.param)
            img = Image.fromarray(arr)
        for t in self.transform.transforms[2:]:      # ToTensor, Normalize
            img = t(img)
        return img, str(path), geo


@torch.inference_mode()
def run_condition(model, paths, geo, args, device, fn=None, param=None,
                  silhouette=None, temperature=1.0):
    ds = PerturbedDataset([Path(p) for p in paths], args.image_sz, geo, fn, param, silhouette)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)
    top1, top5, conf = [], [], []
    for batch, _, batch_geo in tqdm(loader, desc="  inferring", unit="batch",
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-sz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="Cap specimens (0 = all).")
    p.add_argument("--only", help="Comma-separated families to run.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress bars (automatic when stdout is not a TTY).")
    args = p.parse_args(argv)
    if args.quiet:
        pe.QUIET = True

    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    device = torch.device(args.device)
    model, nameslist, temperature, geo_dim, label_level = build_full_model(
        args.checkpoint, device)
    truth = truth_labels(df, nameslist, label_level)
    geo = encode_coords(df["decimalLatitude"], df["decimalLongitude"]) if geo_dim else None
    paths = df["path"].tolist()

    conditions = [("baseline", None, None, None)]
    families = args.only.split(",") if args.only else list(PERTURBATIONS)
    for fam in families:
        if fam not in PERTURBATIONS:
            raise SystemExit(f"ERROR: unknown family '{fam}'. "
                             f"Known: {', '.join(PERTURBATIONS)}")
        fn, params = PERTURBATIONS[fam]
        for prm in params:
            conditions.append((fam if prm is None else f"{fam}_{prm}", fn, prm, None))

    # The silhouette reuses the PC1 plant mask from probe_confounds rather than
    # re-thresholding: same mask, so shape-only and plant-removed are comparable.
    mask_file = out / "plant_masks.npz"
    if (not args.only or "silhouette" in args.only) and mask_file.exists():
        sil = np.load(mask_file)["masks"]
        if len(sil) == len(df):
            conditions.append(("silhouette", None, None, sil))
        else:
            print(f"  [warn] plant_masks.npz has {len(sil)} rows for {len(df)} specimens "
                  f"— skipping silhouette (re-run probe_confounds mask with the same --limit).")
    elif not mask_file.exists():
        print("  [warn] no plant_masks.npz — run `probe_confounds.py mask` first for silhouette.")

    rows, base_top1, preds = [], None, {}
    for name, fn, prm, sil in conditions:
        print(f"  {name} ...", flush=True)
        t1, t5, c = run_condition(model, paths, geo, args, device, fn, prm, sil, temperature)
        rows.append(score(name, truth, t1, t5, c, base_top1))
        rows[-1]["family"] = name.split("_")[0]
        preds[name] = t1
        if base_top1 is None:
            base_top1 = t1
        print("   ", {k: v for k, v in rows[-1].items() if k != "family"}, flush=True)

    res = pd.DataFrame(rows)
    base = res.loc[res.condition == "baseline", "top1"].iloc[0]
    res["delta_vs_baseline"] = (res.top1 - base).round(4)
    res.to_csv(out / "perturbation_conditions.csv", index=False)
    print("\n" + res.to_string(index=False))

    # Broken down by the rank the model predicts, not always species: a
    # family-level model has no per-species accuracy to report. The column is
    # called `taxon` so downstream readers need not know which rank it holds.
    level_col = label_level if label_level in df.columns else "species"
    per = pd.DataFrame({"taxon": df[level_col], "true_idx": truth})
    for name, t1 in preds.items():
        per[name] = (t1 == truth)
    per = per[per.true_idx >= 0].groupby("taxon").mean(numeric_only=True).drop(columns="true_idx")
    per["n"] = df[truth >= 0].groupby(df[level_col]).size()
    per.to_csv(out / "perturbation_per_taxon.csv")
    print(f"  Wrote perturbation_conditions.csv and perturbation_per_taxon.csv "
          f"(taxon = {level_col})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
