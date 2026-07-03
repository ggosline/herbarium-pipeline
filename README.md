# Herbarium Classification Pipeline — User Guide

This pipeline lets you build an AI model that can identify plant species from herbarium sheet photographs. You download images from the global GBIF database, clean them, train a model, and then use that model to check or sort new images — all through a browser-based interface.

The application has **two modes**, switchable from the toggle in the top-right of the window:

- **☁ Cloud** *(default)* — orchestrates a rented GPU pod on RunPod from this UI. Recommended for everyone: training is faster on a 4090 than most local GPUs, costs a few US dollars per run, and needs no local GPU at all. GPU compute becomes a disposable, ephemeral utility — the pod is provisioned on demand and auto-terminated, while your data and trained models persist on Cloudflare R2 and your local disk. Setup is one-time (see [cloud_setup.md](cloud_setup.md)).
- **💻 Local** — runs the scripts on this machine. Requires an NVIDIA GPU with ≥20 GB VRAM for training the recommended ViT-Large model.

This guide covers both modes. The five processing tabs (Download / Filter & Crop / Resize / Train / Identify) work the same way in either; what differs is *where* the work runs and which fields apply.

---

## What you will need

**For Cloud mode (default, recommended):**
- A RunPod account with billing set up, plus an SSH keypair — see [cloud_setup.md](cloud_setup.md). Optional: WandB for live training graphs, Cloudflare R2 for project archives and a shared wheel cache.
- An internet connection.
- Local disk space only for downloaded results (checkpoints, predictions, optionally the resized image set for the Review tab).

**For Local mode:**
- A Windows or Linux PC with an NVIDIA GPU (required for training; ~20 GB VRAM minimum for the default ViT-Large recipe).
- The conda environment `p12` already set up on this machine.
- An internet connection for the download step.
- Disk space: plan for roughly 1–2 GB per 1,000 images at default GBIF thumbnail size; 5–15 GB per 1,000 at IIIF size 2048.

---

## Starting the application

Open a terminal, activate the environment, and launch the web UI:

```
conda activate p12
python /path/to/Pipeline/herbarium_pipeline_webui.py
```

Your browser will open automatically at `http://localhost:8765`. All settings — including the Local/Cloud toggle — are **saved automatically** as you type. If you close and reopen the application, every field is restored exactly where you left off.

### Header at a glance

- **Top-right toggle** — switches Local ↔ Cloud. Sticky across restarts; defaults to Cloud on first launch.
- **Cloud pod strip** *(visible in Cloud mode)* — pod ID, hourly rate, running cost, current step. Buttons: Provision, Upload DwC-A, Download results, Cancel step, Terminate. The Purpose dropdown (`light` / `train`) decides which GPU type a fresh Provision asks for; the next section explains why.

### Tabs

The same five processing tabs appear in either mode (1 Download → 5 Identify). In Cloud mode the Run button on each tab dispatches the work to the pod; in Local mode it runs the script as a subprocess on this machine. Path / output fields that don't apply are hidden depending on the mode you're in.

- **⚙ Setup** — one-time configuration: cloud credentials (RunPod / WandB / R2), SSH key, local environment check. Once everything is green here you should not need to come back.
- **1 Download → 5 Identify** — the five-step pipeline.
- **Run All** — sequences the five steps end-to-end. In Cloud mode this also handles provisioning and the auto-upgrade from a light pod to a train pod (see "Train tab" below).
- **Review, Analysis, Quick ID, Distribution** — work with results once produced.
- **☁ Cloud Tools** *(Cloud mode only, last tab)* — advanced and rare actions: GPU/datacenter overrides, download caps, R2 archive/restore, pull tarred image set, wipe pod-side directories.

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

Think of it as a funnel: your inputs (taxon name, optional geographic filter, or a local DwC-A ZIP) feed a pool of parallel workers that fetch records and images, and out the bottom come two aligned outputs — a folder of JPEGs on disk and `specsin.csv`, one row per specimen.

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

Trains the AI model. Expect hours depending on the number of images and GPU speed. Settings are organised into accordion sections — **Model & batch size** and **Schedule (epochs, learning rates, cool-down)** are open by default; the rest (Loss & hierarchy, Geo features, Logging & resume) are collapsed because their defaults work for most runs.

**Data sources** *(always visible)*

Click **Add Source…** and select the specsin CSV and images folder for your project. You can add multiple sources (e.g. images from different institutions or a second family) — they are combined automatically.

**Output / run dir** — where training results, checkpoints, and the species list are saved.

**Model & batch size** *(accordion, open by default)*

`vit_large_patch16_dinov3.lvd1689m` is the most accurate model. Use a smaller model like `efficientnet_b4` if you have limited GPU memory.

| Setting | Meaning |
|---|---|
| Image size (px) | Resolution images are fed to the model (640 is a good balance) |
| Batch size | Images processed at once — reduce if you run out of GPU memory |
| Stage 2 batch | Override batch size for stage 2 (full fine-tune); 0 = same as stage 1 |
| Grad accum | Effectively multiplies the batch size without using more memory |
| GPUs | Number of GPUs to use |
| NCCL_P2P_DISABLE | Tick only for multi-GPU setups **without** NVLink (e.g. two cards in separate PCIe slots). Do not tick if NVLink is present. |
| Max per species | Cap training images per species (0 = no cap) |

**Schedule (epochs, learning rates, cool-down)** *(accordion, open by default)*

Training is divided into up to three stages:

| Stage | What happens |
|---|---|
| Stage 1 (warm-up) | Backbone is frozen; only the classification head trains. Typically 4 epochs at a higher LR (0.005). Skip by setting epochs to 0 or when resuming. |
| Stage 2 (fine-tune) | All layers unlock and train together. The main training phase — 15–50 epochs depending on dataset size, LR ~0.0001. |
| Cool-down | Optional final phase at **reduced batch size and LR**, run immediately after Stage 2. Best results have been achieved with batch 5, accum 2, LR 0.0001 for a few extra epochs. Set epochs to 0 to skip. |

**Loss & hierarchy** *(accordion, collapsed)*

- *Species / Genus / Family* — which rank the model is trained to distinguish (default: species)
- *Hierarchical multi-head* — the model learns all three ranks simultaneously using a combined loss. Often improves species-level accuracy. Set loss weights (e.g. Species 1.0, Genus 0.5, Family 0.0) to control each rank's contribution.

**Geo features** *(accordion, collapsed)*

Tick **Use lat/lon during training** to fuse geographic coordinates with image features. When enabled, the model learns that a species from West Africa looks slightly different from the same species scanned in Europe, and can use collection locality to disambiguate similar-looking taxa. *Geo MLP dim* controls the size of the geographic feature branch (64 is a good default).

**Logging & resume** *(accordion, collapsed)*

- *WandB project* — enter a Weights & Biases project name to get live training charts. Leave blank for local CSV logs only. When training is **resumed from a checkpoint**, the run continues logging to the same WandB run rather than creating a new one.
- *Resume checkpoint* — to continue an interrupted training run, select `<project>/runs/checkpoints/last.ckpt`. Stage 1 is skipped automatically on resume.
- *Reset optimizer* — load weights only. Tick when starting a fresh stage 2 from a stage-1 checkpoint with a different LR.

**WandB run name** *(always visible — just above the Run Training button)*

The one logging field worth changing per experiment. Use descriptive names like `stage2_lr1e-4_geo` so the WandB sidebar stays scannable when you have a dozen runs in one project. The Apply paths button at the top of the page resets it to the project name, so set it after Apply.

Click **Run Training**.

---

## Tab 5 — Identify

Runs the trained model over your images along two paths:

- **Sorting the unknown** — specimens marked `indet=True` (no recorded species) are copied into `review/indets/<predicted species>/`, physically sorted by the model's top-1 prediction.
- **Auditing the known** — specimens that already carry a label are checked against the model: where the top prediction disagrees with the recorded label (above the mismatch threshold), or confidence falls below the low-confidence threshold, the row is set `flagged=True` in `predictions.csv`.

Every image gets a full top-5 row in `predictions.csv` either way.

**Model section**

| Field | What to enter |
|---|---|
| Checkpoint (.ckpt) | The trained model file — usually `<project>/runs/checkpoints/last.ckpt`. You can also point at the `checkpoints/` directory and click **Latest** to auto-select the most recently modified checkpoint. |
| nameslist.json | The species list saved during training — embedded in recent checkpoints, so this field can usually be left blank |
| timm model override | Leave blank to use the architecture stored in the checkpoint |

**Data sources** — same format as the Train tab.

**Output / Review dir** — results are written here.

**Advanced — thresholds, image size, geo re-rank** *(expander, collapsed)*

Defaults work for most runs. Open this section to adjust:

| Setting | Meaning |
|---|---|
| Mismatch threshold | Confidence level above which a disagreement between the model and the recorded label is flagged (0.7 = 70%) |
| Low-conf flag | Images where the top prediction confidence is below this value are also set aside (0 = off) |
| Image size (px) | Inference resolution; should match training (640 by default) |
| Batch size | Reduce for VRAM-constrained inference |
| Geo rerank weight | Blend model probability with geographic range from training occurrences. 0 = off, 0.3 is a good starting point. Only applied when lat/lon is present. |
| Geo sigma (km) | Bandwidth for geographic scoring. 500 km suits most plant families; use 200–300 for highly localised taxa. |

A `predictions.csv` file is written to the output directory with the full top-5 results for every image.

Click **Run Identify**.

---

## Publishing a model & identifying on a phone

Once you have a trained checkpoint you can publish it to the [Hugging Face Hub](https://huggingface.co) so anyone — including you, from a phone in the field or the herbarium — can identify specimens through a web page, with no installation and no GPU.

**How it works**

- **Publish** — `space/push_model.py` slims the checkpoint (drops optimizer state, roughly a third of the size) and uploads three small files to a Hugging Face *model* repo: `model.ckpt`, `nameslist.json`, and a `config.json` (backbone name, image size, label rank, and a calibration temperature so confidences aren't over-stated). Example:

  ```
  python space/push_model.py \
      --ckpt <project>/runs/checkpoints \
      --family Ebenaceae --region Africa --label-level species \
      --hf-user <your-hf-username>
  ```

  A one-time `huggingface-cli login` (or `HF_TOKEN`) with a write token is required.

- **Discover** — every published repo is tagged `herbarium-pipeline`, so the public **herbarium-id** Space finds it automatically. No redeploy is needed — tap **⟳** in the Space to refresh the model list after publishing.

- **Identify from a phone** — open the Space in any phone browser (**https://huggingface.co/spaces/ggosline/herbarium-id**), pick a model, tap the image box to photograph a herbarium sheet with the rear camera, and get the top-5 predictions with confidence in a second or two. For geo-aware models you can optionally enter latitude/longitude to sharpen the result.

**Calibrated confidence:** the Space applies each model's `config.json` temperature (`softmax(logits / T)`) so the top prediction shows an honest probability instead of a near-100% value. New models get a fitted temperature automatically at publish time; existing models can be calibrated without retraining via `calibrate_temperature.py`.

**Hosting note:** the Space itself runs on Hugging Face (free CPU tier is fine for occasional use; ZeroGPU is faster). The models can equally be served from other scale-to-zero platforms (RunPod Serverless, Google Cloud Run, Modal) since the weights live on the Hub and any host can pull them.

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

In **Cloud mode** Run All also handles pod lifecycle: it provisions a light pod, uploads the DwC-A, runs Setup → Download → Prep, then auto-terminates the light pod (volume preserved), provisions a train pod, and runs Train → Identify → Download results. The pod stays alive after — terminate from the header strip when finished. If a step fails the sequencer stops and leaves the current pod running so you can investigate.

In **Local mode** the steps run as subprocesses on this machine in the order shown.

Typical use:
- First time: tick all five steps
- Re-training after adding more images: tick Download, Filter & Crop, and Train only
- Re-running identification after improving the model: tick Identify only

## Train tab — auto upgrade from light to train pod (Cloud mode)

Cloud mode uses two GPU sizes:
- **light** — cheap L4 — fine for Download, Prep, Identify
- **train** — RTX 4090 — needed for the actual training run

When you click **Run Training** in Cloud mode while the active pod is `light`, you get a one-time confirmation: *"Switch to a train pod?"*. Confirming terminates the light pod (the network volume + downloaded images are preserved), provisions a train pod attached to the same volume, syncs your code, and runs train. Tick *"Don't ask again"* in that dialog to make this automatic for all future trainings. Identify can run on either pod size, so the train pod is reused; downsize manually from Cloud Tools if you want to save a few cents.

The `Purpose` dropdown in the header pod strip controls what a manual *Provision* call asks for — it doesn't trigger an upgrade by itself.

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
Settings are persisted to `<Pipeline repo>/.nicegui/storage-general.json` (relative to where you launched the script). If this file is missing or corrupted, fields will revert to defaults. Use **Apply paths** to quickly restore all project paths from the Projects root and Project name fields. Cloud credentials live in the OS keyring (not this JSON), so they survive even if the storage file is wiped.

---

## Stopping a running step

Click the **Stop** button in the top-right corner of the window at any time. The current process will be terminated.

---

## Getting help

The full command that was run is shown in the log panel, along with all output from the script. Copy this text when reporting a problem.
