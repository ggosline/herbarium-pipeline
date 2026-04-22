# Herbarium Classification Pipeline — User Guide

This pipeline lets you build an AI model that can identify plant species from herbarium sheet photographs. You download images from the global GBIF database, clean them, train a model, and then use that model to check or sort new images — all through a browser-based interface.

---

## What you will need

- A Windows or Linux PC with an NVIDIA GPU (required for training; the more VRAM the better)
- The conda environment `p12` already set up on this machine
- An internet connection for the download step
- Disk space: plan for roughly 1–2 GB per 1,000 images at default GBIF thumbnail size; 5–15 GB per 1,000 if downloading at IIIF size 2048; much more for `max` resolution

---

## Starting the application

Open a terminal, activate the environment, and launch the web UI:

```
conda activate p12
python /path/to/Pipeline/herbarium_pipeline_webui.py
```

Your browser will open automatically at `http://localhost:8765`. All settings are **saved automatically** as you type — if you close and reopen the application, every field is restored exactly where you left off.

---

## Projects root and Project name

At the very top of the window are two fields and an image folder selector:

- **Projects root** — the parent folder where all your projects live (e.g. `/mnt/e` or `/data/herbarium`).
- **Project name** — a short name for the current project (e.g. `Sapindales` or `AfricanEbenaceae`).
- **Image folder** — which subfolder of the project holds the images (`images`, `images_cropped`, or `images_filtered`). You can type a custom name.

Click **Apply paths** and the application fills in sensible file paths for every tab, all stored under `<Projects root>/<ProjectName>/`. You can still change any individual path afterwards.

---

## Tab 1 — Download

Downloads specimen images and metadata from [GBIF](https://www.gbif.org), the Global Biodiversity Information Facility.

| Field | What to enter |
|---|---|
| Taxon rank | Choose Family, Genus, or Order depending on what you are searching |
| Taxon name | e.g. `Ebenaceae`, `Diospyros`, `Sapindales` |
| Continent | Optional. Limits results to one continent (e.g. `AFRICA`) |
| Include countries | Optional. Space-separated ISO-2 codes, e.g. `ZA NG TZ` — only these countries |
| Exclude countries | Optional. Space-separated ISO-2 codes to leave out, e.g. `MG` |
| Local DwC-A ZIP | Optional. If you already downloaded a GBIF archive ZIP file, select it here to skip the live API |
| Output images dir | Where to save the downloaded images |
| specsin CSV path | Where to save (or update) the metadata spreadsheet |
| Workers | How many images to download in parallel — 8 is a good default |
| Limit | Maximum number of images to download (0 = no limit) |
| IIIF image size | Optional. Request a larger image from institutions that support IIIF. Enter a pixel count such as `2048` or `max` for the full scan. Leave blank for the GBIF default. |
| Resize on download | Optional. Shrink images to at most N pixels on the longer side immediately after downloading, saving disk space. 0 = off. |
| Max per species | Optional. Randomly subsample each species to at most N images (0 = no cap). |

Click **Run Download**. Progress appears in the log panel. The script skips images already downloaded, so re-running safely picks up additions.

**IIIF note:** Many herbaria (Naturalis/Leiden, Meise, Kew, and others) serve scans through the IIIF standard, allowing the client to request a specific resolution. Setting IIIF size to `2048` retrieves a much larger version than the GBIF default thumbnail. `max` requests the full archival scan — useful for inspection but very large (10–150 MB per image).

**Adding a second family to an existing project:** Run Download again pointing at the same specsin CSV and images folder with a different taxon name. New records are appended; nothing is overwritten.

---

## Tab 2 — Filter & Crop

Removes non-herbarium images (field photographs of living plants, microscope slides) and trims the dark scanning-bed border many institutional scanners leave around sheets.

| Field | What to enter |
|---|---|
| Input images dir | The folder containing your downloaded images |
| Output images dir | Where to write the cleaned images. Set the same as input to clean in place |
| specsin CSV | Optional. If provided, rejected images are flagged in the metadata and excluded from training |

**Steps to run**

- **Filter non-herbarium images** — an AI classifier separates herbarium sheets from living-plant photos and slides
- **Crop white borders** — removes the dark scanner bed visible around many scanned sheets

**Filter options**

- *Method* — `clip` uses an AI vision model (requires GPU, more accurate); `hsv` uses colour statistics (faster, CPU only)
- *Confidence* — how certain the classifier must be to keep an image (0.6 = 60%). Lower keeps more; higher is stricter

Rejected images go into a `rejected/` subfolder; living-plant field photographs go into `live/`.

Click **Run Filter & Crop**.

---

## Tab 3 — Resize

Scales all images so their longest side is at most 1,024 pixels (or another size you choose). Smaller uniform images train faster and use less disk space.

| Field | What to enter |
|---|---|
| Input images dir | Folder of images to resize |
| Output images dir | Leave blank to resize in place, or enter a different folder to keep originals |
| Max size (px) | Longest-side limit in pixels (default 1,024) |
| No upscale | Ticked by default — small images are left as-is rather than enlarged |
| Force PIL (no DALI) | Tick if NVIDIA DALI is not installed or causes errors |

Click **Run Resize**.

---

## Tab 4 — Train

Trains the AI model. Expect hours depending on the number of images and GPU speed.

**Data sources**

Click **Add Source…** and select the specsin CSV and images folder for your project. You can add multiple sources (e.g. images from different institutions or a second family) — they are combined automatically.

**Output / run dir** — where training results, checkpoints, and the species list are saved.

**Model** — the neural network architecture. `vit_large_patch16_dinov3.lvd1689m` is the most accurate option. Use a smaller model like `efficientnet_b4` if you have limited GPU memory.

**Key settings**

| Setting | Meaning |
|---|---|
| Image size (px) | Resolution images are fed to the model (640 is a good balance) |
| Batch size | Images processed at once — reduce if you run out of GPU memory |
| Grad accum | Effectively multiplies the batch size without using more memory |
| GPUs | Number of GPUs to use |
| NCCL_P2P_DISABLE | Tick only for multi-GPU setups **without** NVLink (e.g. two cards in separate PCIe slots). Do not tick if NVLink is present. |
| Max per species | Cap training images per species (0 = no cap) |

**Training stages**

Training is divided into up to three stages:

| Stage | What happens |
|---|---|
| Stage 1 (warm-up) | Backbone is frozen; only the classification head trains. Typically 4 epochs at a higher LR (0.005). Skip by setting epochs to 0 or when resuming. |
| Stage 2 (fine-tune) | All layers unlock and train together. The main training phase — 15–50 epochs depending on dataset size, LR ~0.0001. |
| Cool-down | Optional final phase at **reduced batch size and LR**, run immediately after Stage 2. Best results have been achieved with batch 5, accum 2, LR 0.0001 for a few extra epochs. Set epochs to 0 to skip. |

**Classification target**

- *Species* — the model learns to distinguish individual species (default)
- *Genus* — the model learns genera only (useful if species labels are unreliable)
- *Family* — the model learns families only
- *Hierarchical multi-head* — the model learns all levels simultaneously using a combined loss. This often improves species-level accuracy. Set loss weights (e.g. Species 1.0, Genus 0.5, Family 0.0) to control each level's contribution.

**Location features**

Tick **Use lat/lon during training** to fuse geographic coordinates with image features. When enabled, the model learns that a species from West Africa looks slightly different from the same species scanned in Europe, and can use collection locality to disambiguate similar-looking taxa. The *Geo MLP dim* controls the size of the geographic feature branch (64 is a good default).

**WandB logging**

Enter a Weights & Biases project name to get live training charts. When a run is started fresh, per-species/genus/family image counts are uploaded as tables so you can review dataset composition in the WandB UI. If training is **resumed from a checkpoint**, the run automatically continues logging to the same WandB run rather than creating a new one. Leave the project field blank to use a local CSV log instead.

**Resume checkpoint**

To continue an interrupted training run, select `<project>/runs/checkpoints/last.ckpt` and click Run Training. Stage 1 is skipped automatically on resume.

**Remote training**

Expand the *Remote training* section to run training on a remote GPU server:

- Enter the **SSH host** (`user@hostname`), **local project root**, and **remote project root**
- Click **Sync data →** to copy your dataset to the remote machine (via rsync or rclone to a cloud bucket)
- Click **Run Training** — the command runs on the remote machine over SSH
- Click **← Sync checkpoint** to retrieve the finished checkpoint

Click **Run Training**.

---

## Tab 5 — Identify

Runs the trained model over your images to sort unidentified specimens and flag possible mis-identifications.

**Model section**

| Field | What to enter |
|---|---|
| Checkpoint (.ckpt) | The trained model file — usually `<project>/runs/checkpoints/last.ckpt`. You can also point at the `checkpoints/` directory and click **Latest** to auto-select the most recently modified checkpoint. |
| nameslist.json | The species list saved during training — embedded in recent checkpoints, so this field can usually be left blank |
| timm model override | Leave blank to use the architecture stored in the checkpoint |

**Data sources** — same format as the Train tab.

**Output / Review dir** — results are written here.

**Thresholds**

| Setting | Meaning |
|---|---|
| Mismatch threshold | Confidence level above which a disagreement between the model and the recorded label is flagged (0.7 = 70%) |
| Low-conf flag | Images where the top prediction confidence is below this value are also set aside (0 = off) |

A `predictions.csv` file is written to the output directory with the full top-5 results for every image.

Click **Run Identify**.

---

## Review tab

Loads a `predictions.csv` file and lets you browse predictions image by image. Filter by category (all, indets, flagged, mis-ID, high confidence) and sort by confidence or species name. You can correct individual determinations directly in the browser and save changes back to the CSV.

---

## Analysis tab

Loads a `predictions.csv` and produces three outputs from a single **Load & Plot** click:

- **Overall metrics** — Accuracy, Precision (macro), Recall (macro), and F1 (macro) displayed as summary cards, at species, genus, or family level.
- **Confusion matrix** — heatmap of the top N most-confused true species (Y-axis) against the top N most-predicted species for those rows (X-axis). Tooltip shows True / Predicted and Recall %.
- **Per-species accuracy** — horizontal bar chart, sorted worst-first, coloured red→green.
- **Most confused pairs** — plain table of True → Predicted pairs that occur at least *Min confusions* times, sorted by count.

---

## Distribution tab

Shows the image count per species as a bar chart. Enter a specsin CSV and images directory, then optionally cap the number of images per species and filter to only species that have image files on disk. Use the export button to save a filtered CSV.

---

## Run All tab

Chains all five pipeline steps together automatically. Tick only the steps you want to run, then click **Run Full Pipeline**. Each step uses the settings entered in its own tab, so configure those first.

Typical use:
- First time: tick all five steps
- Re-training after adding more images: tick Download, Filter & Crop, and Train only
- Re-running identification after improving the model: tick Identify only

---

## Project folder layout

After a full run your project folder (`<Projects root>/<ProjectName>/`) will contain:

```
<Projects root>/Sapindales/
    specsin.csv              — metadata for every specimen
    images_cropped/          — downloaded and cleaned images
        rejected/            — slides and unidentifiable images (not used in training)
        live/                — field photographs of living plants (not used in training)
    runs/
        nameslist.json       — list of species the model knows
        wandb_run_id.txt     — WandB run ID for resume continuity (if WandB is enabled)
        checkpoints/
            last.ckpt        — most recent model checkpoint
            epoch=xx-...ckpt — best stage-2 checkpoint
            cd-epoch=xx-...  — best cool-down checkpoint (if cool-down was used)
        logs/                — training metrics (CSV or WandB)
    review/
        predictions.csv      — full prediction results
        indets/              — unidentified specimens sorted by prediction
        uncertain/           — flagged possible mis-identifications
```

---

## Frequently asked questions

**Can I add images from a second institution or country later?**
Yes. Run the Download tab again with the same specsin CSV and images folder. The script detects what is already downloaded and only fetches new records.

**Can I train on two plant families in one model?**
Yes. Download each family separately into the same specsin CSV and images folder, then add both as data sources in the Train tab.

**Training stopped halfway through. Do I have to start again?**
No. Browse to `<project>/runs/checkpoints/last.ckpt` in the **Resume checkpoint** field and click Run Training. WandB logging will automatically continue in the same run.

**The model keeps predicting the same few species. What is wrong?**
Usually caused by class imbalance. The pipeline compensates with inverse-frequency weighting, but very extreme imbalances can still cause problems. Try downloading more images for rare species, or use *Max per species* to cap the dominant ones.

**Images are downloading but many are failing. Is that normal?**
Some GBIF links are broken or the host server is slow. A failure rate below 20% is typical and not a problem — the script logs failures and continues.

**What does "sparse" mean in the metadata?**
A species is marked sparse if it has fewer than 5 images with confirmed files on disk. Sparse species are excluded from training because there is not enough data to learn them reliably.

**The GPU runs out of memory during training. What can I do?**
Reduce **Batch size** (try 2 or 1), increase **Grad accum** by the same factor to compensate, or choose a smaller model such as `efficientnet_b4`.

**I get an error about NCCL unused parameters when using Hierarchical multi-head.**
This is handled automatically — the DDP strategy sets `find_unused_parameters=True` whenever hierarchical mode is enabled.

**My settings are gone after restarting the app.**
Settings are persisted to `~/.nicegui/storage-general.json`. If this file is missing or corrupted, fields will revert to defaults. Use **Apply paths** to quickly restore all project paths from the Projects root and Project name fields.

---

## Stopping a running step

Click the **Stop** button in the top-right corner of the window at any time. The current process will be terminated.

---

## Getting help

The full command that was run is shown in the log panel, along with all output from the script. Copy this text when reporting a problem.
