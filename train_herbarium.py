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

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import timm.optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import train_test_split
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

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


def getweights(series):
    namescount = Counter(series)
    counts = [v for (n, v) in sorted(namescount.items(), key=lambda x: x[0])]
    weights = torch.ones(1) / torch.FloatTensor(counts)
    return weights


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
                 max_per_species: int = 0):
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
            missing = ~df["abs_path"].apply(lambda p: Path(p).exists())
            if missing.any():
                print(f"  [warn] {missing.sum()} files listed in CSV but missing on disk — skipping")
                df = df[~missing]
            frames.append(df)
            print(f"  {str(img_dir)[-30:]:>30s}: {len(df):5,} specimens")

        combined = pd.concat(frames, ignore_index=True)

        # Derive genus from species (first word)
        combined["genus"] = combined["species"].str.split().str[0]

        # Filter by sparse_threshold at species level (always)
        species_counts = combined["species"].value_counts()
        combined = combined[combined["species"].isin(
            species_counts[species_counts >= sparse_threshold].index
        )].copy()

        # Cap images per species (applied before split so stratification still works)
        if max_per_species and max_per_species > 0:
            combined = (combined.groupby("species", group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), max_per_species),
                                                   random_state=seed))
                        .reset_index(drop=True))
            print(f"  Per-species cap ({max_per_species}): {len(combined):,} specimens remain")

        # Choose indexing column
        index_col = "species" if hierarchical else label_level
        if index_col not in combined.columns:
            raise ValueError(f"Column '{index_col}' not found. Available: {list(combined.columns)}")

        self.nameslist = sorted(combined[index_col].unique())
        self.namesdict = {n: i for i, n in enumerate(self.nameslist)}
        self.num_classes = len(self.nameslist)
        self.weights = getweights(combined[index_col])
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
                species_family = combined.groupby("species")["family"].first()
                self.species_to_family = torch.tensor(
                    [family_dict[species_family[s]] for s in self.nameslist], dtype=torch.long
                )

        df_train, df_valid = train_test_split(
            combined, test_size=train_val_split,
            stratify=combined[index_col], random_state=seed
        )
        self.train_files  = list(df_train["abs_path"])
        self.valid_files  = list(df_valid["abs_path"])
        self.train_labels = [self.namesdict[n] for n in df_train[index_col]]
        self.valid_labels = [self.namesdict[n] for n in df_valid[index_col]]

        self.train_coords = _encode_coords(df_train.reset_index(drop=True))
        self.valid_coords = _encode_coords(df_valid.reset_index(drop=True))

        # Store per-category counts for downstream logging
        self.species_counts = dict(combined["species"].value_counts().sort_index())
        self.genus_counts   = dict(combined["genus"].value_counts().sort_index())
        self.family_counts  = (dict(combined["family"].value_counts().sort_index())
                               if "family" in combined.columns else {})

        print(f"\n  Combined: {len(combined):,} specimens, {self.num_classes} {index_col} classes")
        if hierarchical:
            print(f"  Hierarchical heads: {self.num_genus} genera"
                  + (f", {self.num_family} families" if self.num_family else ""))
        print(f"  Train: {len(self.train_files):,}  |  Valid: {len(self.valid_files):,}")


# ---------------------------------------------------------------------------
# DALI pipeline  (identical to notebook)
# ---------------------------------------------------------------------------

@pipeline_def(enable_conditionals=True)
def create_dali_pipeline(files, labels, crop, size, shard_id, num_shards,
                         file_root="", dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(
        files=files, labels=labels, file_root=file_root,
        shard_id=shard_id, num_shards=num_shards,
        pad_last_batch=True, name="Reader",
        dont_use_mmap=True,        # avoids mmap failures on some mounted filesystems
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

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT, output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std =[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    labels = labels.gpu()
    return images, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True,
                 geo_dim: int = 0):
        super().__init__()
        self.geo_dim = geo_dim
        if geo_dim:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            feat_dim = self.backbone.num_features
            self.backbone.set_grad_checkpointing(True)
            self.geo_mlp = nn.Sequential(
                nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim)
            )
            self.head = nn.Linear(feat_dim + geo_dim, num_classes)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            self.model.set_grad_checkpointing(True)

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
                 geo_dim: int = 0):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.backbone.set_grad_checkpointing(True)
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
        self.model  = model
        self.data   = data
        self.config = config
        self.lr     = config["stage1_lr"]
        self.t_max  = config["stage1_epochs"]
        self.hierarchical = config.get("hierarchical", False)
        self.use_location = config.get("use_location", False)

        if self.use_location:
            self.register_buffer("train_labels_t",
                                 torch.tensor(data.train_labels, dtype=torch.long))
            self.register_buffer("valid_labels_t",
                                 torch.tensor(data.valid_labels, dtype=torch.long))
            self.register_buffer("train_coords_t", data.train_coords)
            self.register_buffer("valid_coords_t", data.valid_coords)

        # Store nameslist so it can be embedded in every checkpoint
        if data.hierarchical:
            self._nameslist_payload = {"species": data.nameslist,
                                       "genus":   data.genus_nameslist,
                                       "family":  data.family_nameslist}
        else:
            self._nameslist_payload = data.nameslist

        num_classes = data.num_classes

        def _metrics(n, prefix):
            k = min(5, n)
            m = {"Accuracy": Accuracy(task="multiclass", num_classes=n),
                 "Precision": Precision(task="multiclass", average="weighted", num_classes=n),
                 "Recall":    Recall(task="multiclass", average="weighted", num_classes=n),
                 "F1":        F1Score(task="multiclass", num_classes=n, average="weighted")}
            if k > 1:
                m["Top5"] = Accuracy(task="multiclass", num_classes=n, top_k=k)
            return MetricCollection(m).clone(prefix=prefix)

        self.train_metrics = _metrics(num_classes, "train_")
        self.valid_metrics = _metrics(num_classes, "val_")
        self.criterion = nn.CrossEntropyLoss(weight=data.weights)

        if self.hierarchical:
            # Register lookup tensors as buffers so they move to GPU automatically
            if data.species_to_genus is not None and data.num_genus >= 2:
                self.register_buffer("species_to_genus", data.species_to_genus)
                self.criterion_genus = nn.CrossEntropyLoss()
                self.train_metrics_genus = _metrics(data.num_genus, "train_genus_")
                self.valid_metrics_genus = _metrics(data.num_genus, "val_genus_")
            else:
                self.species_to_genus = None
            if data.species_to_family is not None and data.num_family >= 2:
                self.register_buffer("species_to_family", data.species_to_family)
                self.criterion_family = nn.CrossEntropyLoss()
                self.train_metrics_family = _metrics(data.num_family, "train_family_")
                self.valid_metrics_family = _metrics(data.num_family, "val_family_")
            else:
                self.species_to_family = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint["nameslist"] = self._nameslist_payload
        checkpoint["use_location"] = self.use_location
        checkpoint["geo_dim"] = self.config.get("geo_dim", 0)

    def on_load_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"]
        if any("._orig_mod" in k for k in sd) and not any("._orig_mod" in k for k in self.state_dict()):
            checkpoint["state_dict"] = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
        elif not any("._orig_mod" in k for k in sd) and any("._orig_mod" in k for k in self.state_dict()):
            checkpoint["state_dict"] = {k.replace("model.model.", "model._orig_mod.model."): v for k, v in sd.items()}

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
        for attr in ("train_loader", "val_loader"):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).reset()
                except Exception:
                    pass
                delattr(self, attr)

        self.criterion = nn.CrossEntropyLoss(weight=self.data.weights.to(self.device))
        cfg   = self.config
        sz    = cfg["image_sz"]
        batch = cfg["batch_size"]
        world = self.trainer.world_size

        # Truncate training data so total is divisible by batch_size × world_size.
        # This guarantees every DDP rank's DALI shard has exactly the same number
        # of complete batches — unequal batch counts cause one rank to exit the
        # epoch early while the other is still in a gradient all-reduce, hanging.
        total = len(self.data.train_files)
        keep  = (total // (batch * world)) * (batch * world)
        train_files  = self.data.train_files[:keep]
        # When use_location is on, pass sample indices as DALI "labels" so _step()
        # can look up the real class label and coordinates from the registered buffers.
        if self.use_location:
            train_dali_labels = list(range(keep))
            valid_dali_labels = list(range(len(self.data.valid_files)))
        else:
            train_dali_labels = self.data.train_labels[:keep]
            valid_dali_labels = self.data.valid_labels

        try:
            pipe = create_dali_pipeline(
                batch_size=batch, num_threads=cfg["num_workers"],
                device_id=self.local_rank, shard_id=self.global_rank,
                num_shards=world, file_root=self.data.imagepath,
                files=train_files, labels=train_dali_labels,
                crop=sz, size=sz, dali_cpu=False, is_training=True,
                prefetch_queue_depth=1,
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
                prefetch_queue_depth=1,
            )
            self.val_loader = DALIClassificationIterator(pipe, reader_name="Reader",
                                                         last_batch_policy=LastBatchPolicy.FILL,
                                                         auto_reset=True)
        except Exception as exc:
            self._log(f"ERROR building val DALI pipeline: {exc}")
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
        self.model.set_grad_checkpointing(False)

    def on_validation_model_train(self):
        # Called by Lightning to restore train mode after validation.
        # Must set model back to train() ourselves since we overrode this hook.
        self.train()
        self.model.set_grad_checkpointing(True)

    def on_validation_epoch_start(self):
        # Free gradient buffers before validation (set_to_none releases the memory
        # rather than just zeroing it; they'll be re-created at the next backward).
        # Do NOT call empty_cache() here — returning the allocator cache to CUDA
        # then re-requesting it next epoch causes fragmentation-driven growth.
        opt = self.optimizers()
        if opt is not None:
            opt.zero_grad(set_to_none=True)

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

    def _step(self, batch, is_train: bool = True):
        image = batch[0]["data"]
        raw   = batch[0]["label"].squeeze(-1).long()
        if self.use_location:
            labels_t = self.train_labels_t if is_train else self.valid_labels_t
            coords_t = self.train_coords_t if is_train else self.valid_coords_t
            target = labels_t[raw]
            geo    = coords_t[raw]
        else:
            target = raw
            geo    = None
        outputs = self.model(image, geo) if geo is not None else self.model(image)
        if self.hierarchical:
            loss = self._hierarchical_loss(outputs, target)
            return loss, outputs, target
        else:
            loss = self.criterion(outputs, target)
            return loss, outputs, target

    def training_step(self, batch, batch_idx):
        try:
            loss, outputs, target = self._step(batch, is_train=True)
        except Exception as exc:
            self._log(f"ERROR in training_step epoch {self.current_epoch} "
                      f"batch {batch_idx}: {exc}")
            raise
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
            loss, outputs, target = self._step(batch, is_train=False)
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
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
        accumulate_grad_batches=config.get("accum", 2),
        precision=config.get("precision", "16-mixed"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        enable_progress_bar=enable_pb,
        default_root_dir=str(output_dir),
    )


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
        max_per_species=config.get("max_per_species", 0),
    )
    config["num_classes"] = data.num_classes

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
        )
    else:
        model = TimmModel(config["model_name"], num_classes=data.num_classes,
                          pretrained=config.get("pretrained", True),
                          geo_dim=geo_dim)
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
    if do_stage1:
        if hasattr(model_module, "freeze_backbone"):
            model_module.freeze_backbone()

        s1_ckpt_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="s1-epoch={epoch:02d}-val_loss={valid_loss:.4f}",
            monitor="valid_loss", save_top_k=1, save_last=True, mode="min",
        )
        print(f"\n{'='*50}\n"
              f"STAGE 1: frozen backbone, {stage1_epochs} epochs, lr={config['stage1_lr']}\n"
              f"{'='*50}")
        s1_trainer = build_trainer(config, output_dir, logger,
                                   [s1_ckpt_cb,
                                    EarlyStopping(monitor="valid_loss", patience=5, mode="min")],
                                   stage1_epochs, num_gpus)
        s1_trainer.fit(lit)
        if use_wandb:
            _step_offset[0] += s1_trainer.global_step
            wandb_offset_file.write_text(str(_step_offset[0]))

        # Load best stage-1 weights only — fresh optimizer/DDP in stage 2
        best_s1 = s1_ckpt_cb.best_model_path or str(output_dir / "checkpoints" / "last.ckpt")
        print(f"Stage 1 complete. Loading weights from {best_s1} for stage 2.")
        ckpt_data = torch.load(best_s1, map_location="cpu", weights_only=False)
        lit.load_state_dict(ckpt_data["state_dict"], strict=False)

        if hasattr(model_module, "unfreeze_all"):
            model_module.unfreeze_all()

    # ── Stage 2: full fine-tune ───────────────────────────────────────────────
    if stage2_epochs > 0:
        s2_batch = config.get("stage2_batch_size") or 0
        if s2_batch > 0 and s2_batch != config["batch_size"]:
            print(f"  Stage 2 batch size overridden: {config['batch_size']} → {s2_batch}")
            config["batch_size"] = s2_batch
        lit.set_stage2(lr=config["stage2_lr"], t_max=stage2_epochs)
        print(f"\n{'='*50}\n"
              f"STAGE 2: full fine-tune, {stage2_epochs} epochs, lr={config['stage2_lr']}, "
              f"batch={config['batch_size']}\n"
              f"{'='*50}")

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="epoch={epoch:02d}-val_loss={valid_loss:.4f}",
            monitor="valid_loss", save_top_k=1, save_last=True, mode="min",
        )

        # When resuming, optionally discard saved optimizer state (e.g. resuming
        # from a stage-1 checkpoint whose LR has decayed to min_lr).
        fit_ckpt = resume if not do_stage1 else None
        if fit_ckpt and config.get("reset_optimizer"):
            print(f"Loading weights only from {fit_ckpt} (optimizer state discarded)")
            ckpt_data = torch.load(fit_ckpt, map_location="cpu", weights_only=False)
            lit.load_state_dict(ckpt_data["state_dict"], strict=False)
            fit_ckpt = None

        if hasattr(model_module, "unfreeze_all"):
            model_module.unfreeze_all()

        s2_trainer = build_trainer(config, output_dir, logger,
                                   [checkpoint_cb,
                                    EarlyStopping(monitor="valid_loss", patience=5, mode="min")],
                                   stage2_epochs, num_gpus)
        s2_trainer.fit(lit, ckpt_path=fit_ckpt)
        if use_wandb:
            _step_offset[0] += s2_trainer.global_step
            wandb_offset_file.write_text(str(_step_offset[0]))

    # ── Stage 3: cool-down (smaller batch + lower LR) ────────────────────────
    if cooldown_epochs > 0:
        cooldown_batch = config.get("cooldown_batch_size", config["batch_size"])
        cooldown_lr    = config.get("cooldown_lr",         config["stage2_lr"])
        cooldown_accum = config.get("cooldown_accum",      config["accum"])
        best_so_far    = (checkpoint_cb.best_model_path if checkpoint_cb else None) or str(output_dir / "checkpoints" / "last.ckpt")

        print(f"\n{'='*50}\n"
              f"COOL-DOWN: {cooldown_epochs} epochs, "
              f"batch={cooldown_batch}, accum={cooldown_accum}, lr={cooldown_lr}\n"
              f"resuming from {best_so_far}\n"
              f"{'='*50}")

        config["batch_size"] = cooldown_batch
        config["accum"]      = cooldown_accum
        lit.set_stage2(lr=cooldown_lr, t_max=cooldown_epochs)

        # Load model weights only — do NOT use ckpt_path on the new trainer, because
        # that would restore the epoch counter from the checkpoint and the trainer
        # would see current_epoch >= max_epochs and exit immediately without training.
        ckpt_data = torch.load(best_so_far, map_location="cpu", weights_only=False)
        lit.load_state_dict(ckpt_data["state_dict"], strict=False)
        print(f"Loaded weights from {best_so_far}")

        cooldown_ckpt_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="cd-epoch={epoch:02d}-val_loss={valid_loss:.4f}",
            monitor="valid_loss", save_top_k=1, save_last=True, mode="min",
        )
        cooldown_trainer = build_trainer(
            config, output_dir, logger,
            [cooldown_ckpt_cb, EarlyStopping(monitor="valid_loss", patience=3, mode="min")],
            cooldown_epochs, num_gpus,
        )
        cooldown_trainer.fit(lit)
        if use_wandb:
            _step_offset[0] += cooldown_trainer.global_step
            wandb_offset_file.write_text(str(_step_offset[0]))
        checkpoint_cb = cooldown_ckpt_cb

    best_ckpt = checkpoint_cb.best_model_path if checkpoint_cb else str(output_dir / "checkpoints" / "last.ckpt")
    print(f"\n{'='*50}\nTRAINING COMPLETE")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Nameslist       : {nameslist_path}")
    print(f"{'='*50}")

    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    seed=42,
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
    batch_size=4,
    accum=2,
    cooldown_epochs=0,
    cooldown_batch_size=5,
    cooldown_lr=0.0001,
    cooldown_accum=2,
    precision="16-mixed",
    num_workers=8,
    num_gpus=2,
    wandb_project=None,
    wandb_run_name="herbarium_run",
    no_wandb=False,
    resume=None,
    max_per_species=0,
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
                   help="Gradient accumulation steps")
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
    p.add_argument("--num-gpus", type=int, default=DEFAULT_CONFIG["num_gpus"])
    p.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--sparse-threshold", type=int, default=DEFAULT_CONFIG["sparse_threshold"])
    p.add_argument("--max-per-species", type=int, default=0, metavar="N",
                   help="Cap training images per species (0 = no cap)")
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
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        seed=args.seed,
        sparse_threshold=args.sparse_threshold,
        max_per_species=args.max_per_species,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        no_wandb=args.no_wandb,
        resume=args.resume,
        reset_optimizer=args.reset_optimizer,
        compile_model=not args.no_compile,
        pretrained=True,
        precision="16-mixed",
        label_level=args.label_level,
        hierarchical=args.hierarchical,
        species_weight=args.species_weight,
        genus_weight=args.genus_weight,
        family_weight=args.family_weight,
        use_location=args.use_location,
        geo_dim=args.geo_dim,
    )
    train(config)
