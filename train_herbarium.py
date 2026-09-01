"""
Standalone herbarium image classifier training script.
Generalised from Herbarium_lightning_2GPU_DALI.ipynb.

Sources are specified as colon-separated pairs: specsin_csv:image_dir
Multiple sources are supported and combined automatically.

Usage:
  python train_herbarium.py \\
      --sources specsin.csv:images/ specsinAsia.csv:imagesAsia/ \\
      --output-dir ./runs/ebenaceae/ \\
      --model vit_large_patch16_dinov3.lvd1689m \\
      --image-sz 640 --batch-size 4 \\
      --stage1-epochs 4  --stage1-lr 0.005 \\
      --stage2-epochs 15 --stage2-lr 0.0001 \\
      --num-gpus 2 --wandb-project HerbariumPipeline

Resume after crash:
  python train_herbarium.py ... --resume ./runs/ebenaceae/checkpoints/last.ckpt
"""

import sys as _sys
_print = _sys.stderr.write  # stderr is unbuffered; visible before torch swallows stdout

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

# Must be set before torch initializes the CUDA allocator. Reduces fragmentation
# from variable-size activations (DALI + gradient checkpointing), letting batch=12
# fit on 24 GB cards at ViT-Large / 640px.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_print("Loading numpy/pandas...\n")
import numpy as np
import pandas as pd
_print("Loading torch...\n")
import torch
import torch.nn as nn
_print("Loading timm...\n")
import timm
import timm.optim
_print("Loading pytorch_lightning...\n")
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import train_test_split
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

# Pre-flight image validation, shared with filter_and_crop_herbarium.py (which is
# where bad files are meant to be caught and marked in specsin). Kept here as a
# guard for sources that never went through the filter step: DALI's decoder has
# no per-sample error tolerance, so one bad file kills the run mid-epoch.
from image_checks import scan_images as _scan_images

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug import trivial_augment
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _encode_coords(df: pd.DataFrame) -> torch.Tensor:
    """
    Encode decimalLatitude / decimalLongitude as a (N, 4) float32 tensor.
    Encoding: (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat), has_location)
    Missing / invalid coords → all-zero row (model learns to ignore them).
    """
    lat = pd.to_numeric(df.get("decimalLatitude",  pd.Series(dtype=float)), errors="coerce").values
    lon = pd.to_numeric(df.get("decimalLongitude", pd.Series(dtype=float)), errors="coerce").values
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


def make_early_stop(config: dict, patience: int) -> EarlyStopping:
    """Early stopping on the metric we actually select checkpoints by.

    It used to monitor valid_loss, which never fired: late in training the model
    grows more confident and the loss drifts up on the few hard examples while
    accuracy is flat or still improving. So the loss keeps twitching, patience
    keeps resetting, and the run goes to max_epochs. On the 16-epoch Rubiaceae run
    it never fired once — and epochs 12-15 bought 0.0 accuracy.

    Monitoring val_Accuracy instead (mode="max") tracks what identify actually
    picks (the acc-*.ckpt), and min_delta stops noise-level wobbles from resetting
    the counter. Across past runs (Rubiaceae, Opiliaceae, Icacinales) accuracy
    reaches ~99% of its peak by roughly epoch 8-9 and then flatlines, so this
    typically saves 3-5 epochs.

    Set --early-stop-metric valid_loss to restore the old behaviour.
    """
    metric = config.get("early_stop_metric") or "val_Accuracy"
    mode = "min" if "loss" in metric else "max"
    # 0 = use this stage's own default patience; anything else overrides it.
    pat = int(config.get("early_stop_patience", 0) or 0) or patience
    return EarlyStopping(
        monitor=metric,
        patience=pat,
        min_delta=float(config.get("early_stop_min_delta", 0.005)),
        mode=mode,
        verbose=True,
    )


def getweights(series, beta: float = 0.0):
    """Per-class loss weights, ordered to match the sorted nameslist.

        w_c  ∝  (1 / n_c) ** beta,  normalised to mean 1

    beta = 0.0  no weighting (plain cross-entropy). Optimises overall accuracy
                against the true class prior — for curation this is usually what
                you want: a specimen that looks like the commonest genus most
                likely *is* the commonest genus.
    beta = 0.5  square-root inverse frequency — a mild nudge toward rare taxa.
    beta = 1.0  full inverse frequency. Optimises *balanced* accuracy, and on a
                long-tailed herbarium it is a trap.

    Weighted CE makes the model learn p(c|x) ∝ w_c · p_true(c|x), so the ratio
    between the largest and smallest weight is a straight logit bias at
    prediction time. At beta=1 on Rubiaceae that ratio was 11050/5 = 2210x: the
    loss valued one image of a 5-image genus as much as 2210 images of
    Psychotria. The tail ate the head — Psychotria recall fell to 11% while six
    5-image genera absorbed 12% of all predictions. Keep the ratio small.

    Normalising to mean 1 also keeps the loss value comparable across beta and
    stops a single rare, high-weight sample from dominating a batch's gradient.
    """
    namescount = Counter(series)
    counts = torch.tensor(
        [v for (_, v) in sorted(namescount.items(), key=lambda x: x[0])],
        dtype=torch.float32,
    )
    if beta <= 0:
        return torch.ones_like(counts)
    w = counts.pow(-float(beta))
    return w / w.mean()


class HerbariumData:
    """
    Loads one or more (specsin_csv, image_dir) source pairs and combines them
    into a single stratified train/val split.

    label_level: "species" | "genus" | "family"  — used in single-head mode.
    hierarchical: if True, always indexes by species and builds genus/family
                  mapping tensors for use with multi-head models.
    """

    def __init__(self, sources: list[tuple[Path, Path]],
                 label_level: str = "species",
                 hierarchical: bool = False,
                 sparse_threshold: int = 5, train_val_split: float = 0.2, seed: int = 42,
                 max_per_class: int = 0, class_weight_beta: float = 0.0,
                 split_seed: int | None = None, check_images: bool = True,
                 test_split: float = 0.0):
        # `seed` drives initialisation and data order; `split_seed` decides WHICH
        # specimens land in train vs held-out. They default to the same value, so
        # every existing invocation — and every checkpoint written before this
        # split existed — behaves exactly as before.
        #
        # Separating them is what makes an init-only rerun possible. With a single
        # seed, changing it reshuffles the held-out set at the same time as the
        # starting weights, so a difference in final accuracy cannot be attributed
        # to either one. Hold split_seed fixed and vary seed to measure run-to-run
        # variance on a constant evaluation set.
        if split_seed is None:
            split_seed = seed
        self.split_seed = split_seed
        # path → why it is unusable; written to a sidecar by train() so the bad
        # rows can be re-downloaded or marked hasfile=False in specsin.
        self.bad_images: dict[str, str] = {}
        keep_cols = {"fname", "species"}
        coord_cols = {"decimalLatitude", "decimalLongitude"}
        frames = []
        for specsin_path, img_dir in sources:
            df = pd.read_csv(specsin_path)
            mask = ~df["indet"].fillna(False).astype(bool) & df["hasfile"].fillna(False).astype(bool)
            for col in ("outlier", "invalid"):
                if col in df.columns:
                    mask &= ~df[col].fillna(False).astype(bool)
            available = keep_cols | ({"family"} if "family" in df.columns else set())
            available |= coord_cols & set(df.columns)
            df = df.loc[mask, list(available)].copy()
            df["abs_path"] = df["fname"].apply(lambda f: str(img_dir / f))
            defects = _scan_images(list(df["abs_path"]), check_headers=check_images)
            if defects:
                bad = df["abs_path"].isin(defects)
                n_missing = sum(1 for r in defects.values() if r == "missing")
                if n_missing:
                    print(f"  [warn] {n_missing} files listed in CSV but missing on disk — skipping")
                if len(defects) > n_missing:
                    print(f"  [warn] {len(defects) - n_missing} files present but not decodable "
                          f"— skipping (they would abort the DALI pipeline mid-run)")
                    shown = [(p_, r) for p_, r in defects.items() if r != "missing"]
                    for p_, r in shown[:5]:
                        print(f"           {Path(p_).name}: {r}")
                    if len(shown) > 5:
                        print(f"           ... and {len(shown) - 5} more")
                self.bad_images.update(defects)
                df = df[~bad]
            frames.append(df)
            print(f"  {str(img_dir)[-30:]:>30s}: {len(df):5,} specimens")

        combined = pd.concat(frames, ignore_index=True)

        # Genus-only determinations ("Psychotria sp.") name a genus but no
        # species, and must never become a species class. Review writes them with
        # indet=True so the mask above already drops them; this is an explicit
        # guard so the invariant does not depend on that flag surviving.
        sp_only = combined["species"].astype(str).str.match(
            r"^\s*\S+\s+spp?\.?\s*$", case=False, na=False)
        if sp_only.any():
            print(f"  Dropping {int(sp_only.sum()):,} genus-only determinations "
                  f"(e.g. 'Psychotria sp.') — no species to learn")
            combined = combined[~sp_only].reset_index(drop=True)

        # Derive genus from species (first word)
        combined["genus"] = combined["species"].str.split().str[0]

        # Filter by sparse_threshold at the rank we'll actually train on.
        # Hierarchical models always index by species, so use species there.
        rank_col = "species" if hierarchical else label_level
        if rank_col not in combined.columns:
            raise ValueError(
                f"Cannot apply sparse filter: column '{rank_col}' not in specsin "
                f"(available: {list(combined.columns)})")
        rank_counts = combined[rank_col].value_counts()
        # Taxa that have images but fewer than the threshold get dropped here:
        # the model never sees enough of them to learn them, so it can't predict
        # them and will force their specimens into some other class. Record
        # name → image count so the identify step can tell the end user which
        # taxa the model omits. Sorted by count (rarest first) for a readable list.
        dropped = rank_counts[rank_counts < sparse_threshold].sort_values()
        self.excluded_rank = rank_col
        self.excluded_species = {str(n): int(c) for n, c in dropped.items()}
        combined = combined[combined[rank_col].isin(
            rank_counts[rank_counts >= sparse_threshold].index
        )].copy()
        print(f"  Sparse filter @ {rank_col} (≥{sparse_threshold}): "
              f"{len(combined):,} rows, {combined[rank_col].nunique():,} classes "
              f"({len(self.excluded_species)} taxa dropped as too sparse)")

        # Classes that clear the threshold but are still far too sparse to learn.
        # They cost a class slot and — if class weighting is on — become attractors
        # that soak up predictions from the commonest taxa.
        kept_counts = rank_counts[rank_counts >= sparse_threshold]
        tiny = kept_counts[kept_counts < 20]
        if len(tiny):
            print(f"  [warn] {len(tiny)} {rank_col} classes kept with <20 images "
                  f"(smallest: {int(kept_counts.min())}). Too sparse to actually learn, "
                  f"but they still occupy a class slot. Raise --sparse-threshold to drop them.")

        # Cap images per CLASS, at the rank actually being trained (applied before
        # the split so stratification still works).
        #
        # This used to group by "species" unconditionally, which quietly did almost
        # nothing on a genus- or family-level model: capping each of Psychotria's
        # 344 species at 100 still admits ~10,700 Psychotria images. Capping at the
        # training rank is what you meant, and it is the cleanest way to deal with a
        # long tail — it fixes the imbalance in the *data* (Rubiaceae genera: 552x →
        # 15x at 300/genus) instead of lying to the loss about the class prior, and
        # it cuts epoch time in proportion.
        #
        # Within an over-full class, sample round-robin across the next rank down so
        # the class keeps its morphological breadth: a flat random 300 of Psychotria
        # covers ~152 of its 344 species, round-robin covers 300.
        if max_per_class and max_per_class > 0:
            sub_col = {"family": "genus", "genus": "species"}.get(rank_col)
            shuffled = combined.sample(frac=1.0, random_state=split_seed)
            if sub_col and sub_col in shuffled.columns:
                # cumcount gives each row its rank within its sub-taxon; a stable sort
                # on it interleaves the sub-taxa, so head(cap) takes one from each in
                # turn before taking a second from any.
                order = shuffled.groupby([rank_col, sub_col]).cumcount()
                shuffled = shuffled.assign(_rr=order).sort_values("_rr", kind="stable")
            kept = shuffled.groupby(rank_col).head(max_per_class)
            combined = (kept.drop(columns="_rr", errors="ignore")
                            .sort_index().reset_index(drop=True))
            per = combined[rank_col].value_counts()
            print(f"  Per-{rank_col} cap ({max_per_class}): {len(combined):,} specimens remain, "
                  f"imbalance now {per.max() / per.min():.1f}x"
                  + (f" (sampled round-robin across {sub_col})" if sub_col else ""))

        # Choose indexing column (same rank the filter and cap used)
        index_col = rank_col
        if index_col not in combined.columns:
            raise ValueError(f"Column '{index_col}' not found. Available: {list(combined.columns)}")

        self.nameslist = sorted(combined[index_col].unique())
        self.namesdict = {n: i for i, n in enumerate(self.nameslist)}
        self.num_classes = len(self.nameslist)
        self.weights = getweights(combined[index_col], class_weight_beta)

        w_max, w_min = float(self.weights.max()), float(self.weights.min())
        ratio = w_max / max(w_min, 1e-12)
        print(f"  Class weights: beta={class_weight_beta} "
              f"(ratio rarest:commonest = {ratio:,.0f}x)")
        if ratio > 20:
            print(f"  [warn] A {ratio:,.0f}x weight ratio is a {math.log(ratio):.1f}-logit "
                  f"bias toward the rarest class at prediction time. On long-tailed data "
                  f"this makes the tail swamp the head. Consider --class-weight-beta 0.")

        self.imagepath = ""  # DALI uses absolute paths; file_root must be ""
        self.hierarchical = hierarchical
        self.label_level = label_level

        # Hierarchical: build genus/family mapping tensors (species_idx → genus/family idx)
        self.species_to_genus: torch.Tensor | None = None
        self.species_to_family: torch.Tensor | None = None
        self.genus_nameslist: list[str] = []
        self.family_nameslist: list[str] = []
        self.num_genus = 0
        self.num_family = 0
        if hierarchical:
            genus_names = sorted(combined["genus"].unique())
            genus_dict = {g: i for i, g in enumerate(genus_names)}
            self.genus_nameslist = genus_names
            self.num_genus = len(genus_names)
            self.species_to_genus = torch.tensor(
                [genus_dict[s.split()[0]] for s in self.nameslist], dtype=torch.long
            )
            if "family" in combined.columns:
                family_names = sorted(combined["family"].unique())
                family_dict = {f: i for i, f in enumerate(family_names)}
                self.family_nameslist = family_names
                self.num_family = len(family_names)
                # .first() silently picks a winner when sources disagree about a
                # species' family — a real data problem (a merge of two specsin
                # files with different taxonomies) that would otherwise train a
                # family head against labels that are wrong for some specimens.
                fam_per_species = combined.groupby("species")["family"].nunique()
                conflicted = fam_per_species[fam_per_species > 1]
                if len(conflicted):
                    print(f"  [warn] {len(conflicted)} species map to MORE THAN ONE family "
                          f"across the sources; taking the first seen. Fix the taxonomy in "
                          f"specsin — the family head is being taught both.")
                    for sp_name in list(conflicted.index)[:5]:
                        fams = sorted(combined.loc[combined["species"] == sp_name,
                                                   "family"].unique())
                        print(f"           {sp_name}: {', '.join(map(str, fams))}")
                species_family = combined.groupby("species")["family"].first()
                self.species_to_family = torch.tensor(
                    [family_dict[species_family[s]] for s in self.nameslist], dtype=torch.long
                )

        # Optional held-out TEST set, carved off before anything else sees the data.
        #
        # Checkpoint selection, early stopping and the calibration temperature all
        # read the validation split, so val_Accuracy is chosen-on-and-reported-from
        # the same specimens and is an optimistically biased estimate of what the
        # model does on unseen material. That is the number that ends up in write-
        # ups. A test set touched by nothing gives an honest one.
        #
        # Split off with `split_seed`, so it is stable across runs that vary only
        # --seed: two checkpoints with the same split_seed are scored on identical
        # held-out specimens and are directly comparable.
        df_test = None
        n_all = len(combined)          # before the test set is carved off
        if test_split and test_split > 0:
            try:
                combined, df_test = train_test_split(
                    combined, test_size=test_split,
                    stratify=combined[index_col], random_state=split_seed)
            except ValueError as exc:
                # Raised when a class has too few members to appear in both parts.
                print(f"  [warn] stratified test split failed ({exc}); "
                      f"falling back to an unstratified split")
                combined, df_test = train_test_split(
                    combined, test_size=test_split, random_state=split_seed)
            combined = combined.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

        df_train, df_valid = train_test_split(
            combined, test_size=train_val_split,
            stratify=combined[index_col], random_state=split_seed
        )
        self.train_files  = list(df_train["abs_path"])
        self.valid_files  = list(df_valid["abs_path"])
        self.train_labels = [self.namesdict[n] for n in df_train[index_col]]
        self.valid_labels = [self.namesdict[n] for n in df_valid[index_col]]

        # Per-class image counts aligned with nameslist. Embedded in every
        # checkpoint so identify can undo the training-time class weighting
        # post-hoc (--logit-adjust) without re-deriving counts from specsin.
        #
        # Counted over the TRAINING split alone. It used to count `combined`,
        # i.e. train + val (+ test), which is not what the name or the
        # --logit-adjust help text says: that flag treats these as the class
        # prior the model was fitted to, and the model is fitted to train only.
        # The two are near-proportional so the correction barely moves, but a
        # number labelled "training images per class" should be one.
        _idx_counts = df_train[index_col].value_counts()
        self.class_counts = [int(_idx_counts.get(n, 0)) for n in self.nameslist]

        self.train_coords = _encode_coords(df_train.reset_index(drop=True))
        self.valid_coords = _encode_coords(df_valid.reset_index(drop=True))

        self.test_files:  list[str] = []
        self.test_labels: list[int] = []
        self.test_coords = torch.zeros(0, 4)
        if df_test is not None and len(df_test):
            self.test_files  = list(df_test["abs_path"])
            self.test_labels = [self.namesdict[n] for n in df_test[index_col]]
            self.test_coords = _encode_coords(df_test)

        # Store per-category counts for downstream logging
        self.species_counts = dict(combined["species"].value_counts().sort_index())
        self.genus_counts   = dict(combined["genus"].value_counts().sort_index())
        self.family_counts  = (dict(combined["family"].value_counts().sort_index())
                               if "family" in combined.columns else {})

        # n_all, not len(combined): `combined` has had the test set removed by now,
        # so printing it would under-report the data set by exactly the held-out
        # fraction.
        print(f"\n  Combined: {n_all:,} specimens, {self.num_classes} {index_col} classes")
        if hierarchical:
            print(f"  Hierarchical heads: {self.num_genus} genera"
                  + (f", {self.num_family} families" if self.num_family else ""))
        print(f"  Train: {len(self.train_files):,}  |  Valid: {len(self.valid_files):,}"
              + (f"  |  Test (held out, untouched): {len(self.test_files):,}"
                 if self.test_files else ""))


# ---------------------------------------------------------------------------
# DALI pipeline  (identical to notebook)
# ---------------------------------------------------------------------------

@pipeline_def(enable_conditionals=True)
def create_dali_pipeline(files, labels, crop, size, shard_id, num_shards,
                         file_root="", dali_cpu=False, is_training=True,
                         use_mmap=False, scale_jitter=1.0, random_crop=False):
    images, labels = fn.readers.file(
        files=files, labels=labels, file_root=file_root,
        shard_id=shard_id, num_shards=num_shards,
        pad_last_batch=True, name="Reader",
        # mmap is faster on local NVMe / tmpfs but fails on some network FS
        # (MooseFS, NFS without locking). Default off; set use_mmap=True after
        # staging images to local storage.
        dont_use_mmap=not use_mmap,
        random_shuffle=is_training, # stagger I/O between ranks; also improves generalisation
    )
    dali_device  = "cpu" if dali_cpu else "gpu"
    decoder_device = "cpu" if dali_cpu else "mixed"
    pw = size if decoder_device == "mixed" else 0
    ph = size if decoder_device == "mixed" else 0

    if is_training:
        images = fn.decoders.image(
            images, device=decoder_device, output_type=types.RGB,
            preallocate_width_hint=pw, preallocate_height_hint=ph,
        )
        # Scale jitter. `resize_shorter` accepts a random DataNode, so the short
        # side lands somewhere in [size, size*scale_jitter] and the fixed-size
        # crop below therefore covers a randomly varying FRACTION of the sheet.
        # scale_jitter=1.0 reproduces the old `size=size, mode="not_smaller"`
        # exactly, which is why it is the disable value.
        if scale_jitter > 1.0:
            images = fn.resize(
                images, device=dali_device,
                resize_shorter=fn.random.uniform(range=[float(size),
                                                        float(size) * scale_jitter]),
                interp_type=types.INTERP_TRIANGULAR)
        else:
            images = fn.resize(images, device=dali_device, size=size,
                               mode="not_smaller", interp_type=types.INTERP_TRIANGULAR)
        images = images.gpu()
        images = trivial_augment.trivial_augment_wide(images)
        mirror  = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(
            images, device=decoder_device, output_type=types.RGB,
            preallocate_width_hint=pw, preallocate_height_hint=ph,
        )
        images = fn.resize(images, device=dali_device, size=size,
                           mode="not_smaller", interp_type=types.INTERP_TRIANGULAR)
        images = images.gpu()
        mirror  = False

    # crop_mirror_normalize centres the crop by default (crop_pos = 0.5), so
    # before this the training and validation branches produced the SAME
    # deterministic centre crop — the only geometric augmentation in the whole
    # pipeline was the mirror. On herbarium sheets, where the specimen sits at a
    # variable position and scale within a fixed sheet, that leaves the model
    # free to key on absolute position within the frame. Randomising the crop
    # window (translation) alongside the scale jitter above is the missing
    # augmentation. Validation stays centred and deterministic.
    if is_training and random_crop:
        crop_pos_x = fn.random.uniform(range=[0.0, 1.0])
        crop_pos_y = fn.random.uniform(range=[0.0, 1.0])
    else:
        crop_pos_x = crop_pos_y = 0.5

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT, output_layout="CHW",
        crop=(crop, crop),
        crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std =[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    labels = labels.gpu()
    return images, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _create_timm(model_name: str, img_size: int = 0, **kwargs):
    """timm.create_model, forwarding img_size only to backbones that accept it.

    A ViT with a learned position-embedding grid bakes that grid in at
    construction, so a backbone whose pretrained cfg is smaller than our
    --image-sz must be told the real input size or it rejects the input.
    RoPE ViTs need no such help (both vit_large_patch16_dinov3 and
    vit_large_patch16_lingbot report pos_embed=None, dynamic_img_size=True,
    strict_img_size=False and run at any size), so for them img_size is a
    verified no-op and the 640 px baseline is unchanged. CNNs (efficientnet
    et al.) take no img_size argument at all and raise TypeError, so fall
    back to plain creation.
    """
    if img_size:
        try:
            return timm.create_model(model_name, img_size=img_size, **kwargs)
        except TypeError:
            pass
    return timm.create_model(model_name, **kwargs)


class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True,
                 geo_dim: int = 0, img_size: int = 0):
        super().__init__()
        self.geo_dim = geo_dim
        if geo_dim:
            self.backbone = _create_timm(model_name, img_size, pretrained=pretrained, num_classes=0)
            feat_dim = self.backbone.num_features
            self.geo_mlp = nn.Sequential(
                nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim)
            )
            self.head = nn.Linear(feat_dim + geo_dim, num_classes)
        else:
            self.model = _create_timm(model_name, img_size, pretrained=pretrained,
                                      num_classes=num_classes)

    def forward(self, x, geo=None):
        if self.geo_dim:
            feats = self.backbone(x)
            if geo is not None:
                geo_feats = self.geo_mlp(geo)
            else:
                geo_feats = torch.zeros(feats.shape[0], self.geo_dim, device=feats.device)
            return self.head(torch.cat([feats, geo_feats], dim=1))
        return self.model(x)

    def freeze_backbone(self):
        if self.geo_dim:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen; geo_mlp + head remain trainable")
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if "head" in name:
                    param.requires_grad = True
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            print(f"Trainable params: {trainable}")

    def set_grad_checkpointing(self, enable=True):
        if self.geo_dim:
            self.backbone.set_grad_checkpointing(enable)
        else:
            self.model.set_grad_checkpointing(enable)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")


class TimmModelHierarchical(nn.Module):
    """
    Shared backbone with separate linear classification heads per taxonomic level.
    forward() returns a dict: {"species": logits, "genus": logits, "family": logits}
    Keys are omitted when the corresponding head is None.
    """

    def __init__(self, model_name: str,
                 num_species: int,
                 num_genus: int = 0,
                 num_family: int = 0,
                 pretrained: bool = True,
                 geo_dim: int = 0,
                 img_size: int = 0):
        super().__init__()
        self.backbone = _create_timm(model_name, img_size, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.geo_dim = geo_dim
        if geo_dim:
            self.geo_mlp = nn.Sequential(
                nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim)
            )
        in_dim = feat_dim + geo_dim
        self.head_species = nn.Linear(in_dim, num_species)
        self.head_genus  = nn.Linear(in_dim, num_genus)  if num_genus  else None
        self.head_family = nn.Linear(in_dim, num_family) if num_family else None

    def forward(self, x, geo=None):
        feats = self.backbone(x)
        if self.geo_dim:
            if geo is not None:
                geo_feats = self.geo_mlp(geo)
            else:
                geo_feats = torch.zeros(feats.shape[0], self.geo_dim, device=feats.device)
            feats = torch.cat([feats, geo_feats], dim=1)
        out = {"species": self.head_species(feats)}
        if self.head_genus  is not None: out["genus"]  = self.head_genus(feats)
        if self.head_family is not None: out["family"] = self.head_family(feats)
        return out

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen; heads remain trainable")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LitHerbarium(pl.LightningModule):
    def __init__(self, model, data: HerbariumData, config: dict):
        super().__init__()
        # Embed config (model_name etc.) into the checkpoint so identify_herbarium.py
        # can autodetect the architecture without --model.
        # ignore non-serializable members; "model" and "data" are not hyperparams.
        self.save_hyperparameters(config, ignore=[])
        self.model  = model
        self.data   = data
        self.config = config
        self.lr     = config["stage1_lr"]
        self.t_max  = config["stage1_epochs"]
        self.hierarchical = config.get("hierarchical", False)
        self.use_location = config.get("use_location", False)
        self.aum = bool(config.get("aum", True))

        # DALI hands us the sample INDEX rather than the class whenever anything
        # needs to be attributed back to an individual specimen — the geo lookup,
        # or AUM. _step() then reads the real label out of train_labels_t.
        self._index_batches = self.use_location or self.aum

        if self._index_batches:
            # persistent=False: these are per-example lookups sized to THIS run's
            # split, not learned weights. Saving them bloated every checkpoint and
            # was the only reason a cross-run resume hit shape mismatches — the
            # freshly built ones are always the correct ones to use.
            self.register_buffer("train_labels_t",
                                 torch.tensor(data.train_labels, dtype=torch.long),
                                 persistent=False)
            self.register_buffer("valid_labels_t",
                                 torch.tensor(data.valid_labels, dtype=torch.long),
                                 persistent=False)
            self.register_buffer("test_labels_t",
                                 torch.tensor(data.test_labels or [0], dtype=torch.long),
                                 persistent=False)
        if self.use_location:
            self.register_buffer("train_coords_t", data.train_coords, persistent=False)
            self.register_buffer("valid_coords_t", data.valid_coords, persistent=False)
            self.register_buffer("test_coords_t", data.test_coords, persistent=False)

        if self.aum:
            # Running total of each training specimen's margin, and how many
            # epochs it has been seen. persistent=False: the per-epoch running
            # state does not belong in state_dict; the finished AUM is embedded
            # by on_save_checkpoint instead.
            #
            # _aum_active gates accumulation by stage — see reset_aum().
            self._aum_active = True
            n_train = len(data.train_files)
            self.register_buffer("_aum_sum", torch.zeros(n_train), persistent=False)
            self.register_buffer("_aum_n",
                                 torch.zeros(n_train, dtype=torch.long),
                                 persistent=False)

        # Store nameslist so it can be embedded in every checkpoint
        if data.hierarchical:
            self._nameslist_payload = {"species": data.nameslist,
                                       "genus":   data.genus_nameslist,
                                       "family":  data.family_nameslist}
        else:
            self._nameslist_payload = data.nameslist

        # Taxa dropped by the sparse filter — embedded in every checkpoint so the
        # "not in this model" list travels with the weights (e.g. to the HF Space).
        self._excluded_payload = {
            "rank": getattr(data, "excluded_rank", "species"),
            "taxa": getattr(data, "excluded_species", {}),
        }

        # Training image count per class, aligned with nameslist. Lets identify
        # cancel the training-time class weighting at inference (--logit-adjust)
        # and lets any consumer see how thin a predicted class actually is.
        self._class_counts_payload = list(getattr(data, "class_counts", []))
        self._class_weight_beta = float(config.get("class_weight_beta", 0.0))

        # WHICH SPECIMENS THE MODEL ACTUALLY SAW. Without this, every downstream
        # metric silently mixes in training images — and the model memorises them
        # (100% accuracy, mean confidence 0.9985, vs 88.7% / 0.887 held out), so
        # any accuracy or novelty score computed over "all specimens" is inflated.
        #
        # It lives HERE and not in specsin.csv on purpose: the split is a function
        # of the training config (seed, sparse_threshold, max_per_class,
        # hierarchical, and the exact set of files present on disk), not a property
        # of the specimen. A column in the shared specsin would be silently stale
        # after any retrain with different settings — worse than absent, because
        # the numbers derived from it still look plausible.
        self._split_payload = {
            "train": sorted(Path(p).name for p in getattr(data, "train_files", [])),
            "valid": sorted(Path(p).name for p in getattr(data, "valid_files", [])),
            "seed": int(config.get("seed", 42)),
            # Recorded separately so a later comparison can tell whether two
            # checkpoints share an evaluation set (same split_seed) even when
            # their initialisation differed (different seed).
            "split_seed": int(config.get("split_seed") or config.get("seed", 42)),
            # Empty unless --test-split was used. Recorded so a later analysis can
            # prove a number came from specimens nothing in training ever saw.
            "test": sorted(Path(p).name for p in getattr(data, "test_files", [])),
        }

        num_classes = data.num_classes

        def _metrics(n, prefix):
            k = min(5, n)
            m = {"Accuracy": Accuracy(task="multiclass", num_classes=n),
                 "Precision": Precision(task="multiclass", average="weighted", num_classes=n),
                 "Recall":    Recall(task="multiclass", average="weighted", num_classes=n),
                 "F1":        F1Score(task="multiclass", num_classes=n, average="weighted")}
            if k > 1:
                m["Top5"] = Accuracy(task="multiclass", num_classes=n, top_k=k)
            # compute_groups=False is load-bearing, not a micro-optimisation to
            # undo. torchmetrics' default groups metrics whose states match after
            # the first batch and then computes ONE of them, copying the result to
            # the rest. Accuracy, Top5, Precision, Recall and F1 all carry the same
            # (tp, fp, tn, fn) state, so whenever the first validation batch makes
            # two of them agree they are merged for the whole epoch and report the
            # same number thereafter.
            #
            # Observed: a run logged val_Top5 bit-identical to val_Accuracy
            # (0.714286) for every epoch while an otherwise identical run on a
            # different backbone reported a healthy 0.9158. It depends on the first
            # batch, so it strikes one run and not another and looks like a model
            # property rather than an instrumentation fault.
            return MetricCollection(m, compute_groups=False).clone(prefix=prefix)

        self.train_metrics = _metrics(num_classes, "train_")
        self.valid_metrics = _metrics(num_classes, "val_")
        # Label smoothing caps the target probability below 1.0 so the network
        # never learns to drive the correct logit to +inf — the main cause of
        # over-confident (near-100%) softmax outputs. 0.0 = classic behaviour.
        self.label_smoothing = float(config.get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(weight=data.weights,
                                             label_smoothing=self.label_smoothing)

        # Post-hoc temperature scaling (Guo et al. 2017): filled in after
        # training by fit_temperature() and embedded in every checkpoint.
        # _cal_* accumulate the most recent validation epoch's logits/targets.
        self.temperature = 1.0
        self._cal_logits: list = []
        self._cal_targets: list = []

        if self.hierarchical:
            # Register lookup tensors as buffers so they move to GPU automatically
            if data.species_to_genus is not None and data.num_genus >= 2:
                self.register_buffer("species_to_genus", data.species_to_genus)
                self.criterion_genus = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                self.train_metrics_genus = _metrics(data.num_genus, "train_genus_")
                self.valid_metrics_genus = _metrics(data.num_genus, "val_genus_")
            else:
                self.species_to_genus = None
            if data.species_to_family is not None and data.num_family >= 2:
                self.register_buffer("species_to_family", data.species_to_family)
                self.criterion_family = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                self.train_metrics_family = _metrics(data.num_family, "train_family_")
                self.valid_metrics_family = _metrics(data.num_family, "val_family_")
            else:
                self.species_to_family = None

    def reset_aum(self, active: bool = True) -> None:
        """Restart AUM accumulation, and choose whether it runs at all.

        AUM separates mislabels from clean labels by how EARLY a specimen's
        margin goes positive, so it is only meaningful over the stage that
        actually learns the classes. Pooling stage 1 into the average dilutes
        exactly that signal: with the backbone frozen, margins are near-noise for
        every specimen alike, and 4 such epochs averaged against ~10 stage-2 ones
        pull every score toward the same middle — mislabels included. The
        cool-down is the mirror problem at the other end: by then the network has
        memorised the training set, so every specimen (right or wrong) carries a
        large positive margin and the extra epochs compress the ranking.

        So: reset at the start of stage 2, and switch off for the cool-down.
        """
        if not self.aum:
            return
        self._aum_active = bool(active)
        self._aum_sum.zero_()
        self._aum_n.zero_()

    def _aum_payload(self) -> dict:
        """Mean margin per training specimen, for mislabel triage.

        Under DDP each rank only ever sees its own shard, so the running totals
        must be summed across ranks before they mean anything.
        """
        if not self.aum:
            return {}
        s = self._aum_sum.clone()
        n = self._aum_n.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(s)
            torch.distributed.all_reduce(n)
        seen = n.clamp(min=1)
        aum = (s / seen).cpu()
        return {
            "fname": [Path(p).name for p in self.data.train_files],
            "aum": [round(float(v), 4) for v in aum],
            "epochs_seen": n.cpu().tolist(),
            "note": "mean margin over stage-2 epochs (stage 1 and the cool-down are "
                    "excluded); LOW/NEGATIVE = candidate mislabel. Ranks training "
                    "specimens only.",
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["nameslist"] = self._nameslist_payload
        checkpoint["excluded_species"] = self._excluded_payload
        checkpoint["class_counts"] = self._class_counts_payload
        checkpoint["class_weight_beta"] = self._class_weight_beta
        checkpoint["split"] = self._split_payload
        checkpoint["aum"] = self._aum_payload()
        checkpoint["use_location"] = self.use_location
        checkpoint["geo_dim"] = self.config.get("geo_dim", 0)
        # 1.0 while training; patched to the fitted value after the run so
        # checkpoints written mid-training also get the temperature (see the
        # post-training embed loop in train()).
        checkpoint["temperature"] = getattr(self, "temperature", 1.0)

    def on_load_checkpoint(self, checkpoint):
        # Same _orig_mod. reconciliation as the manual weight loads take, so a
        # ckpt_path= resume and a --reset-optimizer load behave identically.
        # (The old inline version only rewrote "model.model.", which is the
        # non-hierarchical layout; a hierarchical checkpoint is "model.backbone."
        # and went unfixed.)
        checkpoint["state_dict"] = _align_compile_prefix(self.state_dict(),
                                                         checkpoint["state_dict"])

    def _hierarchical_loss(self, outputs: dict, species_target: torch.Tensor):
        """Compute weighted multi-head loss; return total_loss and species logits."""
        w_sp = self.config.get("species_weight", 1.0)
        w_ge = self.config.get("genus_weight", 0.5)
        w_fa = self.config.get("family_weight", 0.0)

        loss = w_sp * self.criterion(outputs["species"], species_target)
        if "genus" in outputs and self.species_to_genus is not None and w_ge > 0:
            genus_target = self.species_to_genus[species_target]
            loss = loss + w_ge * self.criterion_genus(outputs["genus"], genus_target)
        if "family" in outputs and self.species_to_family is not None and w_fa > 0:
            family_target = self.species_to_family[species_target]
            loss = loss + w_fa * self.criterion_family(outputs["family"], family_target)
        return loss

    def _update_hierarchical_metrics(self, outputs: dict, species_target: torch.Tensor,
                                     metrics_sp, metrics_ge, metrics_fa):
        metrics_sp.update(outputs["species"], species_target)
        if metrics_ge is not None and "genus" in outputs and self.species_to_genus is not None:
            metrics_ge.update(outputs["genus"], self.species_to_genus[species_target])
        if metrics_fa is not None and "family" in outputs and self.species_to_family is not None:
            metrics_fa.update(outputs["family"], self.species_to_family[species_target])

    def _log(self, msg: str) -> None:
        """Print with rank prefix and immediate flush so it appears in the webui log."""
        print(f"[rank {self.global_rank}] {msg}", flush=True)

    def setup(self, stage=None):
        # Always rebuild DALI pipelines — stale pipelines from a previous Trainer
        # (or process) will deadlock in DDP.
        for attr in ("train_loader", "val_loader", "test_loader"):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).reset()
                except Exception:
                    pass
                delattr(self, attr)

        self.criterion = nn.CrossEntropyLoss(weight=self.data.weights.to(self.device),
                                             label_smoothing=self.label_smoothing)
        cfg   = self.config
        sz    = cfg["image_sz"]
        batch = cfg["batch_size"]
        world = self.trainer.world_size

        # Truncate training data so total is divisible by batch_size × world_size.
        # This guarantees every DDP rank's DALI shard has exactly the same number
        # of complete batches — unequal batch counts cause one rank to exit the
        # epoch early while the other is still in a gradient all-reduce, hanging.
        #
        # It costs up to batch×world−1 specimens for the whole run. That is not a
        # biased loss: train_files comes out of train_test_split, which shuffles,
        # so the discarded tail is already a random subset (seeded by split_seed)
        # rather than whatever sat at the end of the CSV. The one imperfection is
        # that it is the SAME random subset every epoch instead of being resampled
        # — not worth trading the DDP hang guarantee for.
        total = len(self.data.train_files)
        keep  = (total // (batch * world)) * (batch * world)
        train_files  = self.data.train_files[:keep]
        # Whenever _step() indexes by sample (geo lookup OR AUM), DALI must hand it
        # the sample's ROW INDEX as the "label"; _step() then reads the real class
        # out of *_labels_t. This must match _step()'s own condition exactly —
        # gating on use_location alone silently fed class indices while _step()
        # treated them as row indices (AUM-on/location-off), scrambling the
        # class↔image mapping and collapsing validation accuracy to chance.
        if self._index_batches:
            train_dali_labels = list(range(keep))
            valid_dali_labels = list(range(len(self.data.valid_files)))
        else:
            train_dali_labels = self.data.train_labels[:keep]
            valid_dali_labels = self.data.valid_labels

        prefetch = max(1, int(cfg.get("prefetch_queue", 2)))
        use_mmap = bool(cfg.get("use_mmap", False))
        # DALI defaults to seed=-1, i.e. a fresh random seed per process: without
        # this, shuffle order and every TrivialAugment draw differ between runs no
        # matter what seed_everything() was given. That silently breaks the one
        # thing --split-seed exists for — holding the evaluation set fixed and
        # varying ONLY initialisation, to measure run-to-run variance. Each rank
        # gets its own offset so the shards don't augment in lockstep.
        dali_seed = int(cfg.get("seed", 42)) + 1000 * self.global_rank
        try:
            pipe = create_dali_pipeline(
                batch_size=batch, num_threads=cfg["num_workers"],
                device_id=self.local_rank, shard_id=self.global_rank,
                num_shards=world, file_root=self.data.imagepath,
                files=train_files, labels=train_dali_labels,
                crop=sz, size=sz, dali_cpu=False, is_training=True,
                use_mmap=use_mmap, seed=dali_seed,
                scale_jitter=float(cfg.get("aug_scale_jitter", 1.0)),
                random_crop=bool(cfg.get("aug_random_crop", False)),
                prefetch_queue_depth=prefetch,
            )
            self.train_loader = DALIClassificationIterator(pipe, reader_name="Reader",
                                                           last_batch_policy=LastBatchPolicy.DROP,
                                                           auto_reset=True)
        except Exception as exc:
            self._log(f"ERROR building train DALI pipeline: {exc}")
            raise

        try:
            pipe = create_dali_pipeline(
                batch_size=batch, num_threads=cfg["num_workers"],
                device_id=self.local_rank, shard_id=self.global_rank,
                num_shards=world, file_root=self.data.imagepath,
                files=self.data.valid_files, labels=valid_dali_labels,
                crop=sz, size=sz, dali_cpu=False, is_training=False,
                use_mmap=use_mmap, seed=dali_seed,
                prefetch_queue_depth=prefetch,
            )
            # PARTIAL, not FILL. The reader pads the final batch (pad_last_batch=True
            # equalises shard sizes so DDP ranks stay in step), and FILL hands those
            # padded duplicates to the iterator, which counts them in the metrics —
            # so val_Accuracy was computed over a handful of samples scored twice,
            # once per rank. PARTIAL trims them and yields a short final batch.
            self.val_loader = DALIClassificationIterator(pipe, reader_name="Reader",
                                                         last_batch_policy=LastBatchPolicy.PARTIAL,
                                                         auto_reset=True)
        except Exception as exc:
            self._log(f"ERROR building val DALI pipeline: {exc}")
            raise

        # Held-out test pipeline (--test-split). Built here so it shares the DALI
        # device/seed setup, but it is never handed to the Trainer: no callback,
        # no early stopping and no temperature fit can see it. PARTIAL rather than
        # FILL because a reported test number must not double-count the samples
        # DALI pads the final batch with.
        self.test_loader = None
        if getattr(self.data, "test_files", None):
            test_dali_labels = (list(range(len(self.data.test_files)))
                                if self._index_batches else self.data.test_labels)
            try:
                pipe = create_dali_pipeline(
                    batch_size=batch, num_threads=cfg["num_workers"],
                    device_id=self.local_rank, shard_id=self.global_rank,
                    num_shards=world, file_root=self.data.imagepath,
                    files=self.data.test_files, labels=test_dali_labels,
                    crop=sz, size=sz, dali_cpu=False, is_training=False,
                    use_mmap=use_mmap, seed=dali_seed,
                    prefetch_queue_depth=prefetch,
                )
                self.test_loader = DALIClassificationIterator(
                    pipe, reader_name="Reader",
                    last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
            except Exception as exc:
                self._log(f"ERROR building test DALI pipeline: {exc}")
                raise

        n_train = keep // (batch * world)
        n_val   = len(self.data.valid_files)
        self._log(f"ready — {n_train} train batches/rank, {n_val} val files, "
                  f"device {self.local_rank}")

    def on_validation_model_eval(self):
        # Gradient checkpointing holds all block inputs live during no_grad forward
        # passes, preventing layer-by-layer freeing and causing a large memory spike.
        # Disable it for validation; no_grad already prevents grad computation.
        super().on_validation_model_eval()
        if self.config.get("grad_checkpoint", True):
            self.model.set_grad_checkpointing(False)

    def on_validation_model_train(self):
        # Called by Lightning to restore train mode after validation.
        # Must set model back to train() ourselves since we overrode this hook.
        self.train()
        if self.config.get("grad_checkpoint", True):
            self.model.set_grad_checkpointing(True)

    def on_validation_epoch_start(self):
        # Free gradient buffers before validation (set_to_none releases the memory
        # rather than just zeroing it; they'll be re-created at the next backward).
        # Do NOT call empty_cache() here — returning the allocator cache to CUDA
        # then re-requesting it next epoch causes fragmentation-driven growth.
        opt = self.optimizers()
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        # Start fresh so _cal_* holds only this (most recent) epoch's logits,
        # used to fit the calibration temperature after training.
        self._cal_logits = []
        self._cal_targets = []

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.train_metrics.reset()
        if self.hierarchical:
            if hasattr(self, "train_metrics_genus"):
                self.log_dict(self.train_metrics_genus.compute(), logger=True, sync_dist=True)
                self.train_metrics_genus.reset()
            if hasattr(self, "train_metrics_family"):
                self.log_dict(self.train_metrics_family.compute(), logger=True, sync_dist=True)
                self.train_metrics_family.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.valid_metrics.reset()
        if self.hierarchical:
            if hasattr(self, "valid_metrics_genus"):
                self.log_dict(self.valid_metrics_genus.compute(), prog_bar=True, logger=True, sync_dist=True)
                self.valid_metrics_genus.reset()
            if hasattr(self, "valid_metrics_family"):
                self.log_dict(self.valid_metrics_family.compute(), logger=True, sync_dist=True)
                self.valid_metrics_family.reset()

    def train_dataloader(self): return self.train_loader
    def val_dataloader(self):   return self.val_loader

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = timm.optim.create_optimizer_v2(trainable, opt="nadamuon", lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.t_max, eta_min=self.config["min_lr"])
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def set_stage2(self, lr, t_max):
        self.lr    = lr
        self.t_max = t_max

    def _step(self, batch, split: str = "train"):
        image = batch[0]["data"]
        raw   = batch[0]["label"].squeeze(-1).long()
        idx   = None
        if self._index_batches:
            # Buffers are named <split>_labels_t / <split>_coords_t. The coords
            # lookup stays behind use_location: those buffers are only registered
            # when geo fusion is on.
            target = getattr(self, f"{split}_labels_t")[raw]
            idx    = raw                       # the specimen's row in train_files
            geo    = (getattr(self, f"{split}_coords_t")[raw]
                      if self.use_location else None)
        else:
            target = raw
            geo    = None
        outputs = self.model(image, geo) if geo is not None else self.model(image)
        if self.hierarchical:
            loss = self._hierarchical_loss(outputs, target)
        else:
            loss = self.criterion(outputs, target)
        return loss, outputs, target, idx

    @torch.no_grad()
    def _accumulate_aum(self, outputs, target, idx):
        """Area Under the Margin (Pleiss et al. 2020) — a MISLABEL detector.

            margin = logit[assigned label] - max(logit over every other class)

        Purely observational. Detached, no gradient, no term in the loss: the run
        is bit-for-bit what it would have been without this.

        The idea: a correctly-labelled specimen is *supported* by every other
        specimen of its class, so its margin goes positive early and stays there.
        A MISLABELLED specimen is fought by all of them — the only way the network
        can fit it is to memorise it individually, which happens late. Its margin
        therefore sits low or negative for many epochs first. Averaging the margin
        over training separates the two; the final epoch does not, because by then
        memorisation has won and train accuracy is 100% (every specimen, right or
        wrong, is confidently "correct"). That is exactly why the mismatch flag in
        predictions.csv finds no mis-IDs among training specimens, and why this is
        measured across epochs rather than at the end.
        """
        logits = outputs["species"] if isinstance(outputs, dict) else outputs
        logits = logits.detach().float()
        tgt = target.view(-1, 1)
        assigned = logits.gather(1, tgt).squeeze(1)
        # Largest logit among the OTHER classes (mask the assigned one out).
        other = logits.scatter(1, tgt, float("-inf")).max(dim=1).values
        self._aum_sum.index_add_(0, idx, assigned - other)
        self._aum_n.index_add_(0, idx, torch.ones_like(idx))

    def training_step(self, batch, batch_idx):
        try:
            loss, outputs, target, idx = self._step(batch, split="train")
        except Exception as exc:
            self._log(f"ERROR in training_step epoch {self.current_epoch} "
                      f"batch {batch_idx}: {exc}")
            raise
        if self.aum and getattr(self, "_aum_active", True) and idx is not None:
            self._accumulate_aum(outputs, target, idx)
        if self.hierarchical:
            self._update_hierarchical_metrics(outputs, target,
                self.train_metrics,
                getattr(self, "train_metrics_genus", None),
                getattr(self, "train_metrics_family", None))
        else:
            self.train_metrics.update(outputs, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        try:
            loss, outputs, target, _idx = self._step(batch, split="valid")
        except Exception as exc:
            self._log(f"ERROR in validation_step epoch {self.current_epoch} "
                      f"batch {batch_idx}: {exc}")
            raise
        if self.hierarchical:
            self._update_hierarchical_metrics(outputs, target,
                self.valid_metrics,
                getattr(self, "valid_metrics_genus", None),
                getattr(self, "valid_metrics_family", None))
        else:
            self.valid_metrics.update(outputs, target)
        # Stash species logits + targets (CPU) to fit the temperature later.
        species_logits = outputs["species"] if isinstance(outputs, dict) else outputs
        self._cal_logits.append(species_logits.detach().float().cpu())
        self._cal_targets.append(target.detach().cpu())
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @torch.no_grad()
    def collect_val_logits(self, split: str = "valid") -> int:
        """Re-run a split with the CURRENT weights, refilling _cal_*.

        Needed because the logits stashed during training belong to whichever
        epoch happened to run last, and early stopping guarantees that is not the
        epoch whose weights got selected — patience 2 means it is typically two
        epochs past the peak. Fitting a temperature on one model and stamping it
        into another is exactly the mismatch this avoids: T is refitted against
        each checkpoint's own weights before it is embedded.

        Runs the DALI val pipeline directly rather than trainer.validate() so it
        stays out of Lightning's logging (no phantom metric points after the run)
        and involves no collectives, which is what makes it safe to call on rank 0
        alone. Returns the number of samples seen.

        split="test" runs the held-out set instead (--test-split). Nothing during
        training ever touches that loader, which is the entire point of it.
        """
        loader = {"valid": lambda: self.val_loader,
                  "test":  lambda: getattr(self, "test_loader", None)}[split]()
        if loader is None:
            return 0
        self._cal_logits = []
        self._cal_targets = []
        was_training = self.training
        self.eval()
        if self.config.get("grad_checkpoint", True):
            try:
                self.model.set_grad_checkpointing(False)
            except Exception:
                pass
        n = 0
        try:
            for batch in loader:
                _loss, outputs, target, _idx = self._step(batch, split=split)
                logits = outputs["species"] if isinstance(outputs, dict) else outputs
                self._cal_logits.append(logits.detach().float().cpu())
                self._cal_targets.append(target.detach().cpu())
                n += int(target.numel())
        finally:
            if self.config.get("grad_checkpoint", True):
                try:
                    self.model.set_grad_checkpointing(True)
                except Exception:
                    pass
            if was_training:
                self.train()
        return n

    def fit_temperature(self, max_iter: int = 100) -> float:
        """Fit a single softmax temperature on the logits currently in _cal_*
        (temperature scaling, Guo et al. 2017).

        Those logits come from collect_val_logits(), which re-runs validation
        under a specific checkpoint's weights — so the T returned belongs to that
        checkpoint, not to whichever epoch happened to finish the run.

        Minimises validation NLL over a scalar T, applied as softmax(logits / T).
        T > 1 means the raw model was over-confident and probabilities get
        softened; the argmax (and therefore accuracy) is unchanged. Returns 1.0
        if no validation logits were collected. Runs on whatever logits this
        rank saw — a subset under DDP, which is plenty for a single scalar.
        """
        if not self._cal_logits:
            return 1.0
        logits  = torch.cat(self._cal_logits).double()
        targets = torch.cat(self._cal_targets).long()
        if logits.numel() == 0:
            return 1.0

        log_T = torch.zeros(1, dtype=torch.double, requires_grad=True)  # T = exp(log_T) > 0
        nll   = nn.CrossEntropyLoss()
        opt   = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(logits / log_T.exp(), targets)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            T = float(log_T.detach().exp().item())
        except Exception as exc:
            print(f"  Temperature fit failed ({exc}); falling back to T=1.0")
            return 1.0
        # Guard against degenerate fits; keep within a sane range.
        if not (0.05 < T < 100.0):
            print(f"  Temperature fit out of range (T={T:.3f}); falling back to T=1.0")
            return 1.0
        return T


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _score_logits(logits: torch.Tensor, targets: torch.Tensor,
                  temperature: float = 1.0) -> dict:
    """Top-1 / top-5 / mean confidence for a finished set of logits."""
    if logits.numel() == 0:
        return {}
    k = min(5, logits.shape[1])
    top1 = (logits.argmax(1) == targets).float().mean().item()
    out = {"n": int(targets.numel()), "top1": top1}
    if k > 1:
        topk = logits.topk(k, dim=1).indices
        out[f"top{k}"] = (topk == targets.view(-1, 1)).any(1).float().mean().item()
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
    out["mean_confidence"] = probs.max(1).values.mean().item()
    return out


class _TextProgressCallback(pl.Callback):
    """Plain newline-terminated progress for pipe / webui output (no tqdm)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = trainer.num_training_batches
        interval = max(1, n // 10)          # ~10% progress updates
        if (batch_idx + 1) % interval == 0 or batch_idx + 1 == n:
            pct = 100 * (batch_idx + 1) // n
            try:
                loss = float(outputs["loss"] if isinstance(outputs, dict) else outputs)
                print(f"  train  epoch {trainer.current_epoch}  "
                      f"{pct:3d}%  loss={loss:.4f}", flush=True)
            except Exception:
                print(f"  train  epoch {trainer.current_epoch}  {pct:3d}%", flush=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        n = trainer.num_val_batches[0] if trainer.num_val_batches else 0
        interval = max(1, n // 5) if n else 1   # fewer val updates
        if (batch_idx + 1) % interval == 0 or batch_idx + 1 == n:
            pct = 100 * (batch_idx + 1) // n if n else 0
            print(f"  val    epoch {trainer.current_epoch}  {pct:3d}%", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        parts = [f"epoch {trainer.current_epoch} train complete"]
        if "train_loss_epoch" in m:
            parts.append(f"loss={float(m['train_loss_epoch']):.4f}")
        if "train_Accuracy" in m:
            parts.append(f"acc={float(m['train_Accuracy']):.3f}")
        print("  " + "  ".join(parts), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        parts = [f"epoch {trainer.current_epoch} val complete"]
        if "valid_loss" in m:
            parts.append(f"val_loss={float(m['valid_loss']):.4f}")
        if "val_Accuracy" in m:
            parts.append(f"val_acc={float(m['val_Accuracy']):.3f}")
        print("  " + "  ".join(parts), flush=True)


def build_trainer(config: dict, output_dir: Path, logger, callbacks: list,
                  epochs: int, num_gpus: int) -> Trainer:
    # When launched via torchrun each rank is an independent process that
    # initialises its own CUDA context cleanly — no fork-inherited state,
    # no shared DALI CUDA handles, no stdout pipe contention.
    # torch.compile is disabled for multi-GPU (no AccumulateGrad stream mismatch).
    hierarchical = config.get("hierarchical", False)
    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=hierarchical)
    else:
        strategy = "auto"
    # Disable tqdm when stdout is not a TTY (e.g. webui subprocess pipe).
    # tqdm writes \r-terminated lines; pipe readers split on \n so updates
    # never arrive.  Use our _TextProgressCallback for plain-text progress.
    enable_pb = sys.stdout.isatty()
    if not enable_pb:
        callbacks = list(callbacks) + [_TextProgressCallback()]
    return Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        accumulate_grad_batches=max(1, config.get("accum", 2)),
        precision=config.get("precision", "bf16-mixed"),
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        enable_progress_bar=enable_pb,
        default_root_dir=str(output_dir),
    )


def _align_compile_prefix(live: dict, state_dict: dict) -> dict:
    """Reconcile the `_orig_mod.` prefix torch.compile inserts into key names.

    A run with --no-compile writes `model.model.blocks...`; a compiled run wants
    `model._orig_mod.model.blocks...`. The names differ only by that segment, so
    load_state_dict(strict=False) matches nothing at all and reports success.
    """
    ck_compiled   = any("._orig_mod." in k for k in state_dict)
    live_compiled = any("._orig_mod." in k for k in live)
    if ck_compiled and not live_compiled:
        print("  checkpoint was written by a compiled run; stripping _orig_mod. prefix")
        return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    if live_compiled and not ck_compiled:
        print("  checkpoint predates torch.compile wrapping; inserting _orig_mod. prefix")
        return {k.replace("model.", "model._orig_mod.", 1) if k.startswith("model.") else k: v
                for k, v in state_dict.items()}
    return state_dict


def _load_compatible_state_dict(module, state_dict: dict) -> None:
    """load_state_dict(..., strict=False), but also drop any key whose shape
    doesn't match the live module.

    strict=False only forgives keys that are missing/unexpected by NAME — a
    key present in both dicts with a mismatched shape still raises. That bites
    every cross-run resume (--resume --reset-optimizer) once the dataset size
    or class count differs from the checkpoint's run: train_labels_t/
    valid_labels_t are per-example label lookups sized to THIS run's
    train/valid split, train_coords_t/valid_coords_t (when use_location) are
    the matching per-example coordinates, and criterion.weight is the
    per-class loss weighting vector — none of them are transferable learned
    weights, they're just data-shaped buffers that happen to live in
    state_dict. Filtering by shape here is also the semantically right call:
    the freshly-constructed buffers from the current data are exactly what
    should be used, not stale ones from a different dataset.
    """
    live = module.state_dict()
    state_dict = _align_compile_prefix(live, state_dict)
    compatible = {}
    shape_skipped = set()
    for k, v in state_dict.items():
        if k in live and live[k].shape != v.shape:
            print(f"  skipping {k}: checkpoint shape {tuple(v.shape)} != "
                  f"current shape {tuple(live[k].shape)}")
            shape_skipped.add(k)
            continue
        compatible[k] = v
    result = module.load_state_dict(compatible, strict=False)

    # strict=False is silent by design, and that silence is dangerous here: if the
    # names don't line up (the classic case is a checkpoint written by an
    # uncompiled run being loaded into a compiled one, or vice versa — every
    # backbone key gains or loses an `_orig_mod.`), NOTHING loads and the run
    # continues from pretrained init while the log still says "loading weights
    # from <ckpt>". _align_compile_prefix above fixes the known case; this counts
    # what actually landed so an unknown one can't pass unnoticed.
    # Shape-skipped keys are excluded from the denominator: dropping the head
    # because the class count changed is intended, and must not count against
    # the guard. What is left measures purely whether the NAMES line up.
    weight_keys  = [k for k in live if k.startswith("model.") and k not in shape_skipped]
    loaded_keys  = [k for k in weight_keys if k in compatible]
    missing      = [k for k in getattr(result, "missing_keys", [])
                    if k.startswith("model.") and k not in shape_skipped]
    unexpected   = list(getattr(result, "unexpected_keys", []))
    frac = len(loaded_keys) / max(len(weight_keys), 1)
    print(f"  loaded {len(loaded_keys)}/{len(weight_keys)} model tensors "
          f"({frac:.0%}) from checkpoint"
          + (f", {len(shape_skipped)} skipped on shape" if shape_skipped else ""))
    if missing:
        head = ", ".join(missing[:5])
        print(f"  [warn] {len(missing)} model tensors NOT in the checkpoint: {head}"
              + (" ..." if len(missing) > 5 else ""))
    if unexpected:
        print(f"  [warn] {len(unexpected)} checkpoint tensors had no home in this model "
              f"(first: {unexpected[0]})")
    if weight_keys and frac < 0.5:
        raise RuntimeError(
            f"Refusing to continue: only {len(loaded_keys)} of {len(weight_keys)} model "
            f"tensors matched the checkpoint ({frac:.0%}). The weights did not load — "
            f"training would silently restart from the pretrained initialisation. "
            f"Check that the checkpoint matches --model / --hierarchical / --use-location.")


def _fit(trainer: Trainer, lit, **kwargs) -> None:
    """trainer.fit with a decode failure translated into something actionable.

    The pre-flight scan catches failed downloads, but it is a header check, so
    corruption in the middle of a well-formed file still reaches the decoder —
    and DALI's abort message names the pipeline, never the file. Point at the
    one tool that can find it.
    """
    try:
        trainer.fit(lit, **kwargs)
    except RuntimeError as exc:
        text = str(exc)
        if "nvImageCodec" not in text and "decoders.image" not in text:
            raise
        raise RuntimeError(
            "DALI aborted: an image could not be decoded. The pre-flight header "
            "check passed for every file, so this is corruption inside an "
            "otherwise well-formed image rather than a failed download (or the "
            "check was disabled with --no-image-check).\n"
            "Find it with a full decode pass — this stops at the first bad file "
            "and names it in the traceback:\n"
            "  python -c \"import sys, glob; from PIL import Image; "
            "Image.MAX_IMAGE_PIXELS = None; "
            "[Image.open(f).load() for f in sorted(glob.glob(sys.argv[1] + '/*'))]\" "
            "<image_dir>\n"
            "Then set hasfile=False for it in specsin.csv, or re-download it."
        ) from exc


def train(config: dict):
    import os as _os
    local_rank = int(_os.environ.get("LOCAL_RANK", 0))
    print(f"[rank LOCAL={local_rank}] train() starting — "
          f"GPUs={config.get('num_gpus')}, "
          f"batch={config.get('batch_size')}, "
          f"model={config.get('model_name')}", flush=True)

    seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse sources
    sources = []
    for src in config["sources"]:
        specsin, img_dir = src.split(":", 1)
        sources.append((Path(specsin), Path(img_dir)))

    print("Loading data...")
    data = HerbariumData(
        sources=sources,
        label_level=config.get("label_level", "species"),
        hierarchical=config.get("hierarchical", False),
        sparse_threshold=config.get("sparse_threshold", 5),
        train_val_split=config.get("train_val_split", 0.2),
        seed=config["seed"],
        split_seed=config.get("split_seed"),
        max_per_class=config.get("max_per_class", 0),
        class_weight_beta=config.get("class_weight_beta", 0.0),
        check_images=config.get("check_images", True),
        test_split=config.get("test_split", 0.0),
    )
    config["num_classes"] = data.num_classes

    # Sidecar listing every file dropped as unusable, so a bad download can be
    # re-fetched (or marked hasfile=False) instead of being rediscovered by the
    # next run. Written only when there is something to report.
    if getattr(data, "bad_images", None):
        bad_path = output_dir / "unreadable_images.json"
        bad_path.write_text(json.dumps(
            {Path(k).name: v for k, v in sorted(data.bad_images.items())}, indent=2))
        print(f"Saved unreadable-image list ({len(data.bad_images)}) → {bad_path}")

    # Save nameslist for the identify step
    nameslist_path = output_dir / "nameslist.json"
    if data.hierarchical:
        nameslist_data = {"species": data.nameslist,
                          "genus": data.genus_nameslist,
                          "family": data.family_nameslist}
    else:
        nameslist_data = data.nameslist
    nameslist_path.write_text(json.dumps(nameslist_data, indent=2))
    print(f"Saved nameslist ({data.num_classes} classes) → {nameslist_path}")

    # Sidecar: taxa present in the data but dropped by the sparse filter. Lets
    # the identify step / UI tell the end user which species the model can't
    # predict (same info is embedded in each checkpoint too).
    excluded_path = output_dir / "excluded_species.json"
    excluded_path.write_text(json.dumps({
        "rank": getattr(data, "excluded_rank", "species"),
        "threshold": config.get("sparse_threshold", 5),
        "taxa": getattr(data, "excluded_species", {}),
    }, indent=2))
    print(f"Saved excluded list ({len(getattr(data, 'excluded_species', {}))} "
          f"taxa below threshold) → {excluded_path}")

    # Build model
    print(f"Building model: {config['model_name']}")
    geo_dim = config.get("geo_dim", 0) if config.get("use_location", False) else 0
    if data.hierarchical:
        model = TimmModelHierarchical(
            config["model_name"],
            num_species=data.num_classes,
            num_genus=data.num_genus,
            num_family=data.num_family,
            pretrained=config.get("pretrained", True),
            geo_dim=geo_dim,
            img_size=config.get("image_sz", 0),
        )
    else:
        model = TimmModel(config["model_name"], num_classes=data.num_classes,
                          pretrained=config.get("pretrained", True),
                          geo_dim=geo_dim,
                          img_size=config.get("image_sz", 0))
    # Gradient checkpointing trades ~30% compute for activation memory.
    # Required on 24 GB cards (3090) at ViT-L 640px batch=4. On A100s with
    # plenty of VRAM headroom, disabling it gives back the compute and lifts
    # GPU utilisation. Default on for safety; webui passes NO_GRAD_CKPT=1
    # / --no-grad-checkpoint to opt out.
    if config.get("grad_checkpoint", True) and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        print("Gradient checkpointing: enabled")
    else:
        print("Gradient checkpointing: disabled")
    num_gpus = config.get("num_gpus", 1)
    # torch.compile triggers AccumulateGrad stream mismatch with DDP backward hooks;
    # only use it for single-GPU runs where there are no cross-rank sync concerns.
    if config.get("compile_model", True) and num_gpus == 1:
        model = torch.compile(model)
    elif config.get("compile_model", True) and num_gpus > 1:
        print("Note: torch.compile skipped for multi-GPU (stream mismatch with DDP).")

    # Logger
    use_wandb = config.get("wandb_project") and not config.get("no_wandb", False)
    wandb_id_file     = output_dir / "wandb_run_id.txt"
    wandb_offset_file = output_dir / "wandb_step_offset.txt"
    _step_offset: list[int] = [0]   # accumulated global_step across stages for WandB x-axis
    if use_wandb:
        try:
            import wandb
            wandb.login()
            # Resume the previous WandB run only when also resuming from a checkpoint
            is_resume = bool(config.get("resume"))
            saved_id = (wandb_id_file.read_text().strip()
                        if is_resume and wandb_id_file.exists() else None)
            if is_resume and wandb_offset_file.exists():
                # Restore the step offset so resumed steps continue the WandB x-axis
                # forward rather than going backward (which WandB silently discards).
                try:
                    _step_offset[0] = int(wandb_offset_file.read_text().strip())
                except ValueError:
                    pass
            if not is_resume:
                # Fresh run — delete any stale run ID/offset so next resume picks up the new one
                wandb_id_file.unlink(missing_ok=True)
                wandb_offset_file.unlink(missing_ok=True)
            logger = WandbLogger(project=config["wandb_project"],
                                 name=config.get("wandb_run_name", "herbarium_run"),
                                 id=saved_id,
                                 resume="allow" if saved_id else None,
                                 config=config)
            # Persist run ID for future resumes
            wandb_id_file.write_text(logger.experiment.id)
            # Log per-category counts as tables (only on fresh runs)
            if not saved_id:
                for level, counts in [
                    ("species", data.species_counts),
                    ("genus",   data.genus_counts),
                    ("family",  data.family_counts),
                ]:
                    if counts:
                        table = wandb.Table(
                            columns=["name", "count"],
                            data=[[k, v] for k, v in counts.items()],
                        )
                        logger.experiment.log({f"{level}_counts": table}, commit=False)
            # Each new Trainer resets global_step to 0, but WandB uses
            # trainer/global_step as the x-axis (via define_metric).  Patch
            # log_metrics to add a cumulative offset so stages appear as one
            # continuous run in the WandB chart instead of restarting at 0.
            _orig_log = logger.log_metrics
            def _offset_log(metrics, step=None, _orig=_orig_log, _off=_step_offset):
                _orig(metrics, step=(step + _off[0]) if step is not None else None)
            logger.log_metrics = _offset_log
        except Exception as e:
            print(f"WandB unavailable ({e}), falling back to CSV logger.")
            logger = CSVLogger(str(output_dir), name="logs")
    else:
        logger = CSVLogger(str(output_dir), name="logs")

    resume          = config.get("resume")
    stage1_epochs   = config.get("stage1_epochs", 4)
    stage2_epochs   = config.get("stage2_epochs", 15)
    cooldown_epochs = config.get("cooldown_epochs", 0)

    model_module = model._orig_mod if hasattr(model, "_orig_mod") else model
    do_stage1    = stage1_epochs > 0 and not resume

    lit = LitHerbarium(model, data, config)

    # ── Stage 1: frozen backbone ──────────────────────────────────────────────
    # Run as a separate trainer.fit() so that stage 2 gets its own fresh DDP
    # context.  DDP registers gradient-reduction hooks only for parameters that
    # have requires_grad=True at initialisation; unfreezing mid-run means the
    # backbone gradients are never all-reduced across GPUs and training goes flat.
    checkpoint_cb = None
    acc_ckpt_cb   = None
    # Track whichever trainer ran last so the final summary can pull the
    # last-epoch metrics from its callback_metrics dict.
    last_trainer = None
    if do_stage1:
        if hasattr(model_module, "freeze_backbone"):
            model_module.freeze_backbone()

        # enable_version_counter=False on every save_last=True callback below.
        # Each stage builds its OWN ModelCheckpoint into the same directory, and
        # Lightning's version counter refuses to reuse a name that already exists
        # — so the three of them produced last.ckpt (stage 1), last-v1.ckpt
        # (stage 2) and last-v2.ckpt (cool-down). "last.ckpt" therefore held the
        # FROZEN-BACKBONE warm-up model, which is the one thing it must never be:
        # identify falls back to it, push_model.py can publish it, and it is
        # documented as "most recent". Overwriting is the intended behaviour.
        s1_ckpt_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="s1-{epoch:02d}-{valid_loss:.4f}",
            monitor="valid_loss", save_top_k=1, save_last=True, mode="min",
            enable_version_counter=False,
        )
        # Parallel best-by-accuracy tracker. Lightning keeps both top files
        # independently in the same dir; loss-best and acc-best can diverge
        # late in training (model becomes more confident → fewer-but-bigger
        # losses on hard examples) and we want both available for identify.
        s1_acc_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="s1-acc-{epoch:02d}-{val_Accuracy:.4f}",
            monitor="val_Accuracy", save_top_k=1, save_last=False, mode="max",
        )
        print(f"\n{'='*50}\n"
              f"STAGE 1: frozen backbone, {stage1_epochs} epochs, lr={config['stage1_lr']}\n"
              f"{'='*50}")
        # Stage 1 is a short frozen-backbone warm-up; patience 5 means this
        # effectively never fires, which is what we want — the big jump comes at
        # the first stage-2 epoch, right after this.
        s1_trainer = build_trainer(config, output_dir, logger,
                                   [s1_ckpt_cb, s1_acc_cb,
                                    make_early_stop(config, patience=5)],
                                   stage1_epochs, num_gpus)
        _fit(s1_trainer, lit)
        last_trainer = s1_trainer
        if use_wandb:
            _step_offset[0] += s1_trainer.global_step
            wandb_offset_file.write_text(str(_step_offset[0]))

        # Load best stage-1 weights only — fresh optimizer/DDP in stage 2
        best_s1 = s1_ckpt_cb.best_model_path or str(output_dir / "checkpoints" / "last.ckpt")
        print(f"Stage 1 complete. Loading weights from {best_s1} for stage 2.")
        ckpt_data = torch.load(best_s1, map_location="cpu", weights_only=False)
        _load_compatible_state_dict(lit, ckpt_data["state_dict"])

        if hasattr(model_module, "unfreeze_all"):
            model_module.unfreeze_all()

    # ── Stage 2: full fine-tune ───────────────────────────────────────────────
    if stage2_epochs > 0:
        s2_batch = config.get("stage2_batch_size") or 0
        if s2_batch > 0 and s2_batch != config["batch_size"]:
            print(f"  Stage 2 batch size overridden: {config['batch_size']} → {s2_batch}")
            config["batch_size"] = s2_batch
            # save_hyperparameters() copied config at construction, so mutating the
            # dict alone leaves every checkpoint from here on claiming the stage-1
            # batch size. Keep the recorded value honest.
            lit.hparams["batch_size"] = s2_batch
        lit.set_stage2(lr=config["stage2_lr"], t_max=stage2_epochs)
        print(f"\n{'='*50}\n"
              f"STAGE 2: full fine-tune, {stage2_epochs} epochs, lr={config['stage2_lr']}, "
              f"batch={config['batch_size']}\n"
              f"{'='*50}")

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="{epoch:02d}-{valid_loss:.4f}",
            monitor="valid_loss", save_top_k=1, save_last=True, mode="min",
            enable_version_counter=False,
        )
        acc_ckpt_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="acc-{epoch:02d}-{val_Accuracy:.4f}",
            monitor="val_Accuracy", save_top_k=1, save_last=False, mode="max",
        )

        # When resuming, optionally discard saved optimizer state (e.g. resuming
        # from a stage-1 checkpoint whose LR has decayed to min_lr).
        fit_ckpt = resume if not do_stage1 else None
        if fit_ckpt and config.get("reset_optimizer"):
            print(f"Loading weights only from {fit_ckpt} (optimizer state discarded)")
            ckpt_data = torch.load(fit_ckpt, map_location="cpu", weights_only=False)
            _load_compatible_state_dict(lit, ckpt_data["state_dict"])
            fit_ckpt = None

        if hasattr(model_module, "unfreeze_all"):
            model_module.unfreeze_all()

        # torch.compile traced stage 1 with the backbone frozen, so its backward
        # graph treats those parameters as constants. Flipping requires_grad does
        # NOT invalidate that graph on torch 2.11: stage 2 then silently trains
        # the head alone — same loss curve shape, ~3x too fast, and the backbone
        # comes out bit-identical to the stage-1 checkpoint.
        #
        # Re-wrapping in torch.compile() does not help; the cache is keyed on the
        # underlying code, so a fresh wrapper hits the same stale graph. Only
        # clearing dynamo's cache forces the retrace that picks up the unfrozen
        # parameters. Verified both ways on 2.11.0+cu130.
        # torch.compiler.reset(), not `import torch._dynamo` — a function-local
        # import of a torch submodule rebinds the name `torch` for this whole
        # scope, shadowing the module-level import.
        if config.get("compile_model", True) and num_gpus == 1:
            torch.compiler.reset()
            print("compile cache reset — stage 2 retraces with the backbone unfrozen")

        # Discard any margins accumulated during the frozen-backbone warm-up:
        # they are noise for every specimen equally and only blur the ranking.
        lit.reset_aum(active=True)

        # Patience 2 on val_Accuracy. Replayed against the real curves of the last
        # three runs (Rubiaceae, Opiliaceae, Icacinales) this stops 2-3 epochs
        # early and still captures each run's exact peak — the tail buys nothing.
        s2_trainer = build_trainer(config, output_dir, logger,
                                   [checkpoint_cb, acc_ckpt_cb,
                                    make_early_stop(config, patience=2)],
                                   stage2_epochs, num_gpus)
        _fit(s2_trainer, lit, ckpt_path=fit_ckpt)
        last_trainer = s2_trainer
        if use_wandb:
            _step_offset[0] += s2_trainer.global_step
            wandb_offset_file.write_text(str(_step_offset[0]))

    # ── Stage 3: cool-down (smaller batch + lower LR) ────────────────────────
    if cooldown_epochs > 0:
        cooldown_batch = config.get("cooldown_batch_size", config["batch_size"])
        cooldown_lr    = config.get("cooldown_lr",         config["stage2_lr"])
        cooldown_accum = config.get("cooldown_accum",      config["accum"])
        # Resume the cool-down from the ACCURACY-best, not the loss-best. Every
        # other selection in this run keys on val_Accuracy — early stopping, and
        # identify's checkpoint auto-select — for the reason in make_early_stop:
        # late on, the model grows more confident and valid_loss drifts up on hard
        # examples while accuracy holds or improves. This was the last place still
        # steering by the old metric, so the cool-down could start from a strictly
        # worse model than the one the run would go on to publish.
        best_so_far = ((acc_ckpt_cb.best_model_path if acc_ckpt_cb else None)
                       or (checkpoint_cb.best_model_path if checkpoint_cb else None)
                       or str(output_dir / "checkpoints" / "last.ckpt"))

        print(f"\n{'='*50}\n"
              f"COOL-DOWN: {cooldown_epochs} epochs, "
              f"batch={cooldown_batch}, accum={cooldown_accum}, lr={cooldown_lr}\n"
              f"resuming from {best_so_far}\n"
              f"{'='*50}")

        config["batch_size"] = cooldown_batch
        config["accum"]      = cooldown_accum
        lit.hparams["batch_size"] = cooldown_batch
        lit.hparams["accum"]      = cooldown_accum
        lit.set_stage2(lr=cooldown_lr, t_max=cooldown_epochs)

        # Load model weights only — do NOT use ckpt_path on the new trainer, because
        # that would restore the epoch counter from the checkpoint and the trainer
        # would see current_epoch >= max_epochs and exit immediately without training.
        ckpt_data = torch.load(best_so_far, map_location="cpu", weights_only=False)
        _load_compatible_state_dict(lit, ckpt_data["state_dict"])
        print(f"Loaded weights from {best_so_far}")

        cooldown_ckpt_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="cd-{epoch:02d}-{valid_loss:.4f}",
            monitor="valid_loss", save_top_k=1, save_last=True, mode="min",
            enable_version_counter=False,
        )
        cooldown_acc_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="cd-acc-{epoch:02d}-{val_Accuracy:.4f}",
            monitor="val_Accuracy", save_top_k=1, save_last=False, mode="max",
        )
        cooldown_trainer = build_trainer(
            config, output_dir, logger,
            [cooldown_ckpt_cb, cooldown_acc_cb,
             make_early_stop(config, patience=2)],
            cooldown_epochs, num_gpus,
        )
        # Keep the stage-2 AUM as the reported score; post-memorisation margins
        # would only compress the ranking (see LitHerbarium.reset_aum).
        if lit.aum:
            lit._aum_active = False
        _fit(cooldown_trainer, lit)
        if lit.aum:
            lit._aum_active = True
        last_trainer = cooldown_trainer
        if use_wandb:
            _step_offset[0] += cooldown_trainer.global_step
            wandb_offset_file.write_text(str(_step_offset[0]))
        checkpoint_cb = cooldown_ckpt_cb
        acc_ckpt_cb   = cooldown_acc_cb

    # wandb.finish() prints its own (long) Run summary table, which can be
    # truncated by output viewers (NiceGUI's log panel collapses long
    # contiguous bursts into "... + N lines"). Run it FIRST so our compact
    # summary below is the very last thing on stdout — guaranteed visible
    # even if wandb's table got folded.
    if use_wandb:
        import wandb
        wandb.finish()

    best_ckpt = (checkpoint_cb.best_model_path if checkpoint_cb
                 else str(output_dir / "checkpoints" / "last.ckpt"))

    # ── Calibration: fit softmax temperature and embed it in the checkpoints ──
    # Raw cross-entropy makes the network over-confident (top-1 ≈ 100%);
    # temperature scaling divides logits by a single T fitted on validation so
    # the reported probabilities are honest. Argmax/accuracy are unchanged.
    #
    # EACH checkpoint gets its OWN T, fitted against its OWN weights. The logits
    # stashed during training belong to whichever epoch ran last, and early
    # stopping (patience 2 on val_Accuracy) guarantees that is not the epoch the
    # acc-best checkpoint came from — so the previous single-fit version stamped
    # one model's calibration into a different model's file. T is sensitive to
    # exactly the thing that differs between those epochs (how confident the
    # network has become), so it was wrong in the direction that matters.
    #
    # Only rank 0 fits and rewrites files (avoids DDP write races). It therefore
    # fits on its own validation shard — plenty for a single scalar.
    temperature = 1.0
    if last_trainer is None or last_trainer.is_global_zero:
        ckpt_paths: list[str] = []
        for cb in (checkpoint_cb, acc_ckpt_cb):
            if cb is not None and getattr(cb, "best_model_path", ""):
                if cb.best_model_path not in ckpt_paths:
                    ckpt_paths.append(cb.best_model_path)
        last_ck = output_dir / "checkpoints" / "last.ckpt"
        if last_ck.exists() and str(last_ck) not in ckpt_paths:
            ckpt_paths.append(str(last_ck))

        # Lightning's strategy teardown moves the module to CPU when fit() ends,
        # but the DALI val pipeline still emits on cuda:local_rank.
        cal_device = (torch.device(f"cuda:{local_rank}") if torch.cuda.is_available()
                      else torch.device("cpu"))
        can_revalidate = hasattr(lit, "val_loader")
        if can_revalidate:
            lit.to(cal_device)

        temps: dict[str, float] = {}
        print(f"\n  Calibrating {len(ckpt_paths)} checkpoint(s) — "
              f"refitting T on each one's own weights")
        for ck_path in ckpt_paths:
            try:
                cd = torch.load(ck_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"  WARNING: could not read {ck_path}: {exc}")
                continue
            T = 1.0
            try:
                if can_revalidate:
                    _load_compatible_state_dict(lit, cd["state_dict"])
                    n_seen = lit.collect_val_logits()
                    if n_seen == 0:
                        print(f"  WARNING: no validation samples for {Path(ck_path).name}; T=1.0")
                # If the val pipeline is unavailable, fall back to whatever the
                # last validation epoch stashed — the old behaviour, and clearly
                # labelled as approximate.
                T = lit.fit_temperature()
            except Exception as exc:
                print(f"  WARNING: calibration failed for {Path(ck_path).name} "
                      f"({exc}); T=1.0")
                T = 1.0
            temps[ck_path] = T
            print(f"    {Path(ck_path).name:<40s} T={T:.3f}")
            try:
                cd["temperature"] = T
                torch.save(cd, ck_path)
            except Exception as exc:
                print(f"  WARNING: could not embed temperature into {ck_path}: {exc}")

        # Headline T is the one belonging to the checkpoint identify will pick.
        head_path = (acc_ckpt_cb.best_model_path
                     if acc_ckpt_cb is not None and acc_ckpt_cb.best_model_path
                     else best_ckpt)
        temperature = temps.get(head_path, temps.get(best_ckpt, 1.0))
        lit.temperature = temperature
        print(f"  Calibration temp   : {temperature:.3f}  (identify uses softmax(logits / T))")

        # Sidecar for quick reference. "temperature" stays the scalar older
        # consumers (push_model.py, calibrate_temperature.py) expect; the
        # per-checkpoint map is additive.
        try:
            (output_dir / "temperature.json").write_text(json.dumps({
                "temperature": temperature,
                "checkpoint": head_path,
                "per_checkpoint": {Path(k).name: v for k, v in temps.items()},
            }, indent=2))
        except Exception as exc:
            print(f"  WARNING: could not write temperature.json: {exc}")

    # ── Held-out test evaluation (--test-split) ──────────────────────────────
    # Scored on the checkpoint identify will actually pick, with that checkpoint's
    # own fitted temperature, on specimens no callback ever saw. This is the number
    # to quote: val_Accuracy above is what checkpoint selection and early stopping
    # optimised against, so it is biased upward by construction.
    test_report: dict = {}
    if (last_trainer is None or last_trainer.is_global_zero) and getattr(lit, "test_loader", None):
        head_path = (acc_ckpt_cb.best_model_path
                     if acc_ckpt_cb is not None and acc_ckpt_cb.best_model_path
                     else best_ckpt)
        try:
            cd = torch.load(head_path, map_location="cpu", weights_only=False)
            _load_compatible_state_dict(lit, cd["state_dict"])
            n_seen = lit.collect_val_logits(split="test")
            if n_seen:
                test_report = _score_logits(torch.cat(lit._cal_logits),
                                            torch.cat(lit._cal_targets),
                                            temperature=temperature)
                test_report["checkpoint"] = Path(head_path).name
                (output_dir / "test_metrics.json").write_text(
                    json.dumps(test_report, indent=2))
        except Exception as exc:
            print(f"  WARNING: held-out test evaluation failed: {exc}")

    # Pull the last-epoch metrics off the trainer that actually ran most
    # recently. callback_metrics is populated as logs flow through the
    # Lightning logger, so it has whatever the last validation epoch saw.
    final_metrics: dict = (last_trainer.callback_metrics
                           if last_trainer is not None else {})

    def _fmt(name: str, fmt: str = ".4f") -> str:
        v = final_metrics.get(name)
        if v is None:
            return "n/a"
        try:
            return f"{float(v):{fmt}}"
        except (TypeError, ValueError):
            return str(v)

    # Best valid_loss across the whole run: the active checkpoint callback
    # tracks `monitor='valid_loss', mode='min'`, so its best_model_score is
    # exactly that. (For hierarchical models we still monitor valid_loss,
    # so this stays meaningful.)
    best_val_loss_str = "n/a"
    if checkpoint_cb is not None and checkpoint_cb.best_model_score is not None:
        best_val_loss_str = f"{float(checkpoint_cb.best_model_score):.4f}"

    print(f"\n{'='*50}\nTRAINING COMPLETE")
    print(f"  Final val_Accuracy : {_fmt('val_Accuracy', '.4f')}")
    print(f"  Final valid_loss   : {_fmt('valid_loss', '.4f')}")
    print(f"  Best valid_loss    : {best_val_loss_str}")
    # Hierarchical heads, when enabled, also report their own per-rank
    # accuracies. Print them only when present so single-head runs stay
    # uncluttered.
    if "val_genus_Accuracy" in final_metrics:
        print(f"  Final val_genus_Accuracy  : {_fmt('val_genus_Accuracy', '.4f')}")
    if "val_family_Accuracy" in final_metrics:
        print(f"  Final val_family_Accuracy : {_fmt('val_family_Accuracy', '.4f')}")
    print(f"  Best checkpoint    : {best_ckpt}")
    if acc_ckpt_cb is not None and acc_ckpt_cb.best_model_path:
        score = acc_ckpt_cb.best_model_score
        score_str = f"{float(score):.4f}" if score is not None else "n/a"
        print(f"  Best by val_Accuracy ({score_str}): {acc_ckpt_cb.best_model_path}")
    if test_report:
        k5 = next((k for k in test_report if k.startswith("top") and k != "top1"), None)
        print(f"  --- Held-out test set ({test_report['n']:,} specimens, seen by nothing) ---")
        print(f"  Test accuracy      : {test_report['top1']:.4f}"
              + (f"   (top-{k5[3:]}: {test_report[k5]:.4f})" if k5 else ""))
        print(f"  Mean confidence    : {test_report['mean_confidence']:.4f}  (at T={temperature:.3f})")
        print(f"  Scored checkpoint  : {test_report['checkpoint']}")
    print(f"  Nameslist          : {nameslist_path}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    seed=42,
    split_seed=None,      # None → follow --seed (previous behaviour)
    train_val_split=0.2,
    sparse_threshold=5,
    model_name="vit_large_patch16_dinov3.lvd1689m",
    pretrained=True,
    compile_model=True,
    image_sz=640,
    stage1_lr=0.005,
    stage1_epochs=4,
    stage2_lr=0.0001,
    stage2_epochs=20,
    min_lr=1e-6,
    label_smoothing=0.1,
    batch_size=12,
    accum=1,
    cooldown_epochs=0,
    cooldown_batch_size=5,
    cooldown_lr=0.0001,
    cooldown_accum=2,
    precision="bf16-mixed",
    num_workers=8,
    num_gpus=2,
    wandb_project=None,
    wandb_run_name="herbarium_run",
    no_wandb=False,
    resume=None,
    max_per_class=0,
    class_weight_beta=0.0,
    check_images=True,           # pre-flight header scan; one bad file kills a run
    # Geometric augmentation. Both were absent: train and val saw the identical
    # deterministic centre crop, so the mirror was the only geometric transform.
    test_split=0.0,              # >0 holds out a set no callback ever sees
    aug_scale_jitter=1.15,       # 1.0 = the old fixed resize
    aug_random_crop=True,        # False = centre crop, as validation uses
    aum=True,                    # free, and it is the only way to find mis-IDs in train
    early_stop_metric="val_Accuracy",
    early_stop_patience=0,       # 0 = per-stage default (s1 5, s2 2, cool-down 2)
    early_stop_min_delta=0.005,  # swept on real curves: saves ~2.3 epochs, costs 0.0
)


def parse_args():
    p = argparse.ArgumentParser(description="Train herbarium image classifier.")
    p.add_argument("--sources", nargs="+", required=True, metavar="CSV:DIR",
                   help="Source pairs: specsin.csv:images_dir/  (repeat for multiple)")
    p.add_argument("--output-dir", required=True, metavar="DIR")
    p.add_argument("--model", default=DEFAULT_CONFIG["model_name"], metavar="TIMM_MODEL")
    p.add_argument("--image-sz", type=int, default=DEFAULT_CONFIG["image_sz"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--accum", type=int, default=DEFAULT_CONFIG["accum"],
                   help="Gradient accumulation steps (0 or 1 = no accumulation)")
    p.add_argument("--stage1-epochs", type=int, default=DEFAULT_CONFIG["stage1_epochs"])
    p.add_argument("--stage1-lr", type=float, default=DEFAULT_CONFIG["stage1_lr"])
    p.add_argument("--stage2-epochs", type=int, default=DEFAULT_CONFIG["stage2_epochs"])
    p.add_argument("--stage2-lr", type=float, default=DEFAULT_CONFIG["stage2_lr"])
    p.add_argument("--cooldown-epochs", type=int, default=DEFAULT_CONFIG["cooldown_epochs"],
                   help="Extra cool-down epochs after stage 2 (0 = disabled)")
    p.add_argument("--stage2-batch-size", type=int, default=0,
                   help="Batch size for stage 2 (0 = same as --batch-size). "
                        "Use a smaller value than stage 1 to save VRAM when the full backbone is unfrozen.")
    p.add_argument("--cooldown-batch-size", type=int, default=DEFAULT_CONFIG["cooldown_batch_size"])
    p.add_argument("--cooldown-lr", type=float, default=DEFAULT_CONFIG["cooldown_lr"])
    p.add_argument("--cooldown-accum", type=int, default=DEFAULT_CONFIG["cooldown_accum"])
    p.add_argument("--min-lr", type=float, default=DEFAULT_CONFIG["min_lr"])
    p.add_argument("--label-smoothing", type=float, default=DEFAULT_CONFIG["label_smoothing"],
                   help="Cross-entropy label smoothing (default 0.1). Caps the target below 1.0 "
                        "so the model doesn't become over-confident; 0.0 = classic behaviour. "
                        "Also improves calibration and robustness to noisy labels.")
    p.add_argument("--num-gpus", type=int, default=DEFAULT_CONFIG["num_gpus"])
    p.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                   help="Seeds initialisation, data order and augmentation.")
    p.add_argument("--split-seed", type=int, default=DEFAULT_CONFIG["split_seed"],
                   metavar="N",
                   help="Seed for the train/held-out split (and the --max-per-class "
                        "subsample). Defaults to --seed. Pin it across runs to vary "
                        "only the starting weights while every run is scored on the "
                        "same held-out specimens.")
    p.add_argument("--sparse-threshold", type=int, default=DEFAULT_CONFIG["sparse_threshold"])
    p.add_argument("--class-weight-beta", type=float,
                   default=DEFAULT_CONFIG["class_weight_beta"], metavar="BETA",
                   help="Class-weight exponent: w_c ∝ (1/count_c)**BETA, normalised to mean 1. "
                        "0 = no weighting (plain CE; best overall accuracy — the default). "
                        "0.5 = sqrt inverse-frequency, a mild nudge toward rare taxa. "
                        "1.0 = full inverse-frequency (the old hardcoded behaviour): on "
                        "long-tailed data this lets 5-image classes swamp the commonest ones.")
    p.add_argument("--early-stop-metric", default=DEFAULT_CONFIG["early_stop_metric"],
                   choices=["val_Accuracy", "valid_loss"],
                   help="Metric early stopping watches. Default val_Accuracy — the same metric "
                        "identify selects the best checkpoint by. valid_loss (the old default) "
                        "barely ever fires: late on, the model gets more confident and the loss "
                        "drifts up on hard examples while accuracy holds, so patience keeps "
                        "resetting and the run goes to max_epochs.")
    p.add_argument("--early-stop-patience", type=int,
                   default=DEFAULT_CONFIG["early_stop_patience"], metavar="N",
                   help="Epochs without improvement before stopping. 0 = per-stage default "
                        "(stage1 5, stage2 2, cool-down 2). Raise it if you'd rather pay for a "
                        "few extra epochs than risk stopping on a plateau that would have broken.")
    p.add_argument("--early-stop-min-delta", type=float,
                   default=DEFAULT_CONFIG["early_stop_min_delta"], metavar="D",
                   help="Improvement smaller than this doesn't count (default 0.005). Stops "
                        "noise-level wobble from resetting patience forever. Swept against past "
                        "runs: 0.005 saves ~2.3 epochs at zero accuracy cost; 0.010 starts costing "
                        "real accuracy.")
    p.add_argument("--max-per-class", type=int, default=0, metavar="N",
                   help="Cap training images per CLASS, at the rank being trained (0 = no cap). "
                        "On a genus model this caps each genus; on a species (or hierarchical) "
                        "model, each species. Over-full classes are sampled round-robin across "
                        "the next rank down, so a capped genus keeps its species diversity. "
                        "This is the cheapest way to tame a long tail: it fixes the imbalance in "
                        "the data rather than in the loss, and cuts epoch time in proportion.")
    p.add_argument("--max-per-species", type=int, default=0, metavar="N",
                   help="Deprecated alias for --max-per-class. NOTE: it used to group by species "
                        "even on a genus/family model, where it barely reduced anything; it now "
                        "caps at the training rank.")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=DEFAULT_CONFIG["wandb_run_name"])
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="Resume from checkpoint (stage 2)")
    p.add_argument("--reset-optimizer", action="store_true",
                   help="Load only model weights from the resume checkpoint; discard "
                        "optimizer/scheduler state. Use when starting a fresh stage 2 "
                        "from a stage-1 checkpoint.")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Disable activation gradient checkpointing. Saves ~30%% compute "
                        "but increases VRAM. Recommended on A100/H100; do not use on 24 GB cards.")
    p.add_argument("--prefetch-queue", type=int, default=2, metavar="N",
                   help="DALI prefetch queue depth (default 2). Higher hides I/O latency at "
                        "the cost of VRAM; 4 helps on slow / network filesystems.")
    p.add_argument("--test-split", type=float, default=DEFAULT_CONFIG["test_split"],
                   metavar="F",
                   help="Hold out this fraction as a TEST set that nothing in training "
                        "touches — no checkpoint selection, no early stopping, no "
                        "temperature fit (0 = off, the default). Everything else is scored "
                        "on the validation split, which those three all read, so "
                        "val_Accuracy is optimistically biased; this reports an honest "
                        "number for the selected checkpoint. Split by --split-seed, so runs "
                        "sharing it are scored on identical specimens. Try 0.1.")
    p.add_argument("--aug-scale-jitter", type=float, default=DEFAULT_CONFIG["aug_scale_jitter"],
                   metavar="F",
                   help="Random scale augmentation: the short side is resized into "
                        "[image-sz, image-sz*F] before the fixed-size crop, so each epoch "
                        "sees the sheet at a slightly different zoom. 1.0 disables it and "
                        "restores the previous fixed resize. Keep it mild (~1.15): a large "
                        "value crops the specimen out of the frame.")
    p.add_argument("--no-crop-aug", dest="aug_random_crop", action="store_false",
                   help="Take the training crop from the centre, as validation does. "
                        "By default the crop WINDOW is placed at random, which together "
                        "with --aug-scale-jitter is the only geometric augmentation "
                        "besides the mirror. Before this existed, train and val saw the "
                        "identical deterministic centre crop, leaving the model free to "
                        "key on absolute position within the sheet.")
    p.add_argument("--no-image-check", dest="check_images", action="store_false",
                   help="Skip the pre-flight image header check. The check exists because "
                        "DALI's GPU decoder has no per-sample error tolerance: a single "
                        "undecodable file (an HTML error page saved as .jpg by a failed GBIF "
                        "download, or a truncated image) aborts the pipeline mid-epoch and "
                        "kills the run after the GPU time has been spent. It reads 16 bytes "
                        "from the front and 2 KB from the back of each file, in place of the "
                        "stat() it replaces. Skip it only on a data set you have already "
                        "trained on cleanly.")
    p.add_argument("--use-mmap", action="store_true",
                   help="Enable mmap in DALI file reader. Faster on local NVMe / tmpfs; "
                        "do not use on MooseFS / NFS.")

    # Hierarchical / label-level options
    p.add_argument("--label-level", default="species",
                   choices=["species", "genus", "family"],
                   help="Classification target for single-head training (default: species)")
    p.add_argument("--hierarchical", action="store_true",
                   help="Multi-head: shared backbone with species+genus[+family] heads and joint loss")
    p.add_argument("--species-weight", type=float, default=1.0,
                   help="Loss weight for species head (hierarchical mode, default: 1.0)")
    p.add_argument("--genus-weight", type=float, default=0.5,
                   help="Loss weight for genus head (hierarchical mode, default: 0.5)")
    p.add_argument("--family-weight", type=float, default=0.0,
                   help="Loss weight for family head (hierarchical mode, default: 0.0)")
    p.add_argument("--no-aum", dest="aum", action="store_false",
                   help="Disable AUM (Area Under the Margin) mislabel scoring. AUM is on by "
                        "default: it is purely observational — detached, no gradient, no loss "
                        "term, so the run is unchanged — and it is the ONLY thing that can find "
                        "mis-identified TRAINING specimens. The model memorises its training set "
                        "(100%% accuracy), so it agrees with a wrong label rather than flagging "
                        "it; predictions.csv therefore reports zero mis-IDs among training rows. "
                        "AUM works because a mislabelled sheet is fought by every correctly "
                        "labelled sheet of its class, so its margin stays low for many epochs "
                        "before memorisation wins. Cost: one 4-byte float per training image.")
    p.add_argument("--use-location", action="store_true",
                   help="Fuse lat/lon (sphere-encoded) with image features during training")
    p.add_argument("--geo-dim", type=int, default=64, metavar="N",
                   help="Hidden size of the geo MLP (default: 64, only used with --use-location)")
    return p.parse_args()


if __name__ == "__main__":
    # When launched via torchrun from the webui, all ranks share the same stdout
    # pipe (64 KB kernel buffer).  If rank > 0 writes freely, the buffer fills
    # before the async reader drains it; the blocked write() stalls that rank
    # while the other hits an NCCL barrier → deadlock.  Suppress rank > 0 stdout
    # HERE, before any imports or prints produce output.
    import os as _os
    if int(_os.environ.get("LOCAL_RANK", "0")) != 0:
        sys.stdout = open(_os.devnull, "w")
        # stderr stays open so errors from non-zero ranks remain visible
    else:
        print("Starting train_herbarium.py (imports may take ~30s for torch/timm/lightning)...",
              flush=True)

    args = parse_args()
    config = dict(
        sources=args.sources,
        output_dir=args.output_dir,
        model_name=args.model,
        image_sz=args.image_sz,
        batch_size=args.batch_size,
        accum=args.accum,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
        cooldown_epochs=args.cooldown_epochs,
        stage2_batch_size=args.stage2_batch_size,
        cooldown_batch_size=args.cooldown_batch_size,
        cooldown_lr=args.cooldown_lr,
        cooldown_accum=args.cooldown_accum,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        seed=args.seed,
        split_seed=args.split_seed,
        sparse_threshold=args.sparse_threshold,
        class_weight_beta=args.class_weight_beta,
        aum=args.aum,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        max_per_class=args.max_per_class or args.max_per_species,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        no_wandb=args.no_wandb,
        resume=args.resume,
        reset_optimizer=args.reset_optimizer,
        compile_model=not args.no_compile,
        grad_checkpoint=not args.no_grad_checkpoint,
        prefetch_queue=args.prefetch_queue,
        use_mmap=args.use_mmap,
        check_images=args.check_images,
        test_split=args.test_split,
        aug_scale_jitter=args.aug_scale_jitter,
        aug_random_crop=args.aug_random_crop,
        pretrained=True,
        precision="bf16-mixed",
        label_level=args.label_level,
        hierarchical=args.hierarchical,
        species_weight=args.species_weight,
        genus_weight=args.genus_weight,
        family_weight=args.family_weight,
        use_location=args.use_location,
        geo_dim=args.geo_dim,
    )
    train(config)
