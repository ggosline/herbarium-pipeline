"""
Rank training specimens by Area Under the Margin (AUM) — lowest first.

Why
---
A trained model scores ~100% on its own training set: it has memorised it. So
"prediction disagrees with the label" finds exactly zero mis-determinations
there, and the errors it memorised get confidently confirmed back to the curator.

AUM (Pleiss et al. 2020, NeurIPS) exploits *when* an example is learned rather
than whether it is. Networks fit generalisable structure first and memorise noise
last, so during training we record, for each specimen and each epoch:

    margin = logit(recorded label) - max(logit over all other classes)

and average it over the run. A correctly labelled sheet is supported by every
other specimen of its class: its margin goes positive early and stays there. A
mis-labelled sheet is *opposed* by every correctly labelled specimen of the class
it was assigned to — the only way to fit it is to memorise it individually, which
happens late. Its mean margin is low or negative.

So the lowest-AUM specimens are the candidate mis-determinations. train_herbarium
accumulates this during training (free: no gradient, no loss term) and embeds it
in every checkpoint, so this script needs only the checkpoint.

Usage
-----
    python aum_candidates.py \
        --checkpoint runs/checkpoints/acc-epoch=06-val_Accuracy=0.8800.ckpt \
        --specsin project/specsin.csv \
        --out review/aum_candidates.csv \
        --top 200

The output carries catalogNumber / institutionCode / image_url alongside the
score, so each candidate can be pulled up and adjudicated against the sheet.
That adjudication — not a synthetic label-noise benchmark — is the real test of
whether this is finding anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

# Columns worth carrying through to the review sheet, if specsin has them.
CARRY = ("species", "genus", "family", "catalogNumber", "institutionCode",
         "countryCode", "gbifID", "image_url")


def load_aum(ckpt_path: Path) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    aum = ckpt.get("aum") or {}
    if not aum.get("aum"):
        sys.exit(
            f"{ckpt_path.name} carries no AUM. Only checkpoints from a run with "
            f"AUM enabled (the default; --no-aum disables it) have it."
        )
    df = pd.DataFrame({
        "fname": aum["fname"],
        "aum":   [float(v) for v in aum["aum"]],
    })
    seen = aum.get("epochs_seen")
    if seen:
        df["epochs_seen"] = [int(v) for v in seen]
        # A specimen the sampler never drew has no margin history and a mean of
        # 0.0, which would sit misleadingly mid-distribution rather than flagging
        # itself. Drop it rather than let it look like an ordinary specimen.
        never = df["epochs_seen"] == 0
        if never.any():
            print(f"  {int(never.sum())} specimen(s) never sampled during training "
                  f"— excluded (no margin history)")
            df = df[~never]
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Checkpoint with an embedded AUM payload.")
    p.add_argument("--specsin", type=Path, required=True,
                   help="specsin.csv, to attach determinations and catalogue numbers.")
    p.add_argument("--out", type=Path, required=True,
                   help="Where to write the ranked candidates CSV.")
    p.add_argument("--top", type=int, default=200,
                   help="How many of the lowest-AUM specimens to write (0 = all).")
    args = p.parse_args()

    df = load_aum(args.checkpoint)
    print(f"AUM over {len(df):,} training specimens")

    sp = pd.read_csv(args.specsin, low_memory=False)
    sp["_base"] = sp["fname"].map(lambda f: Path(str(f)).name)
    df["_base"] = df["fname"].map(lambda f: Path(str(f)).name)
    carry = [c for c in CARRY if c in sp.columns]
    df = df.merge(sp[["_base", *carry]].drop_duplicates("_base"),
                  on="_base", how="left").drop(columns="_base")

    df = df.sort_values("aum").reset_index(drop=True)

    print("\nAUM distribution")
    for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0):
        print(f"   {q:>6.1%}  {df['aum'].quantile(q):>8.3f}")
    neg = int((df["aum"] < 0).sum())
    print(f"\n   negative AUM: {neg:,} of {len(df):,} ({neg / max(len(df), 1):.2%}) "
          f"— candidate mis-determinations")

    out = df if args.top <= 0 else df.head(args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n→ wrote {len(out):,} lowest-AUM specimens to {args.out}")

    print(f"\n{'AUM':>8}  {'recorded determination':<38} catalogNumber")
    for _, r in df.head(20).iterrows():
        print(f"{r['aum']:>8.3f}  {str(r.get('species', ''))[:37]:<38} "
              f"{str(r.get('catalogNumber', ''))[:22]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
