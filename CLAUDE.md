# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Herbarium Classification Pipeline** is a full-stack plant species classification system that downloads herbarium specimen images from GBIF, filters and prepares them, trains a deep learning model (using PyTorch/Lightning), and runs inference to identify or verify specimens. It supports two deployment modes:

- **Cloud mode**: Orchestrates GPU training on RunPod instances (paid, recommended for speed).
- **Local mode**: Runs scripts directly on a machine with an NVIDIA GPU (requires ~20 GB VRAM for training).

The application exposes a browser-based web UI built with NiceGUI that lets users configure and execute the five-stage pipeline without command-line knowledge.

## Architecture

### High-Level Data Flow

1. **Download** (`download_gbif_images.py`) — queries GBIF API, fetches specimen images and metadata by taxon (family/genus/order) with optional geographic filtering, writes to a specsin.csv and image directory.
2. **Filter & Crop** (`filter_and_crop_herbarium.py`) — removes non-herbarium images (field photos, slides) via CLIP zero-shot classification or HSV heuristics; crops dark scanner borders.
3. **Resize** (`resize_images.py`) — scales images to a target max dimension (1024 px default) using GPU-accelerated NVIDIA DALI or PIL fallback.
4. **Train** (`train_herbarium.py`) — PyTorch/Lightning multi-stage fine-tuning on a pretrained vision model (ViT-Large, EfficientNet, etc.). Supports hierarchical multi-head training (species/genus/family), geographic feature fusion, resume-from-checkpoint, and WandB logging. Produces checkpoints and a nameslist.json.
5. **Identify** (`identify_herbarium.py`) — runs inference on new images, saves a predictions.csv with top-5 predictions per image, flags mismatches with recorded labels, sorts unidentified specimens by prediction.

### Code Organization

**Root-level scripts**: Five main pipeline stages plus utility scripts (backfill URLs, rebuild specsin, verify data, restore local state).

**`cloud/` package** — RunPod orchestration layer (not needed for local-only use):
- `runpod_client.py` — REST API wrapper (provision/terminate/query pods).
- `pod_session.py` — async SSH/SFTP session for command execution and file transfer.
- `orchestrator.py` — high-level state machine composing the above, manages pod lifecycle and step sequencing.
- `state.py` — per-project JSON state (pod ID, volume ID, DwC-A hash, running cost).
- `secrets.py` — OS keyring integration for API keys and credentials.

**`webui/` package** — NiceGUI presentation layer:
- `widgets.py` — reusable low-level widgets (buttons, pills, accordions, file picker, data-source list).
- Main UI lives in `herbarium_pipeline_webui.py`.

**`space/` directory** — Hugging Face Spaces integration (model hosting; separate from main pipeline).

**`tools/`** — one-off utilities (list RunPod GPU types, merge taxonomic clades).

### Data Model

**specsin.csv** — the project's metadata backbone. Rows are specimens; columns include:
- Core: catalogNumber, species, family, genus, verbatimName, institutionCode, countryCode
- Coordinates: decimalLatitude, decimalLongitude, coordinateUncertaintyInMeters
- Tracking: gbifID, image_url, indet (unknown species), fname, hasfile, sparse (< 5 images), outlier, invalid

Updated incrementally during Download (new rows), Filter (hasfile, rejected flags), and Identify (predictions added).

**Project folder layout**:
```
<ProjectRoot>/<ProjectName>/
  specsin.csv                      # Metadata
  images_cropped/                  # Downloaded + filtered images
    rejected/                      # Non-herbarium images
    live/                          # Field photos of living plants
  runs/
    nameslist.json                 # Species list from training
    checkpoints/
      last.ckpt                    # Most recent checkpoint
      epoch=XX-valid_loss=X.ckpt   # Best stage-2 checkpoint
      cd-epoch=XX-...              # Cool-down best (if used)
    logs/                          # Training metrics (CSV or WandB)
  review/
    predictions.csv                # Full inference results
    indets/                        # Sorted unidentified specimens
    uncertain/                     # Flagged possible mis-IDs
```

### Key Design Decisions

**Multi-stage training** — Most runs use:
1. **Stage 1** (warm-up): backbone frozen, only head trains, 4 epochs, LR ~0.005
2. **Stage 2** (fine-tune): all layers train, 15–50 epochs, LR ~0.0001
3. **Cool-down** (optional): reduced batch and LR for final polishing

**Hierarchical multi-head** — Model can simultaneously predict species, genus, and family with separate loss weights. Improves species accuracy by providing taxonomic context.

**Geographic feature fusion** — When lat/lon is available, coordinates are encoded as unit sphere vectors and concatenated with image features, helping the model learn regional variation within a species.

**Checkpointing and resume** — Each checkpoint embeds the nameslist and all config; resuming from a checkpoint auto-skips Stage 1.

**Cloud pod upgrade** — Light L4 pods (cheap, fast for download/prep) auto-provision to RTX 4090 (expensive, needed for training) on demand. The network volume and all data persist.

## Running the Application

### Local Setup

```bash
# Activate the conda environment (assumes it exists)
conda activate p12

# Install/update dependencies (uses uv, faster than pip)
uv sync

# Run the web UI
python herbarium_pipeline_webui.py
```

Opens automatically at `http://localhost:8765`. Settings persist to `~/.config/herbarium_pipeline.json` and OS keyring.

### Cloud Mode Setup

See `cloud_setup.md` for detailed account creation (RunPod, WandB, Cloudflare R2). Credentials are stored in the OS keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service).

### Running Pipeline Steps from Command Line

Each script accepts `--help`. Example:

```bash
# Download from GBIF
python download_gbif_images.py \
  --family Ebenaceae \
  --continent AFRICA \
  --output-dir ./project/images \
  --specsin ./project/specsin.csv

# Filter + crop
python filter_and_crop_herbarium.py \
  --input-dir ./project/images \
  --output-dir ./project/images_filtered \
  --specsin ./project/specsin.csv

# Resize
python resize_images.py \
  --input-dir ./project/images_filtered \
  --output-dir ./project/images_1024

# Train
python train_herbarium.py \
  --sources specsin.csv:images_1024/ \
  --output-dir ./runs/my_project/ \
  --model vit_large_patch16_dinov3.lvd1689m \
  --image-sz 640 \
  --batch-size 4 \
  --stage1-epochs 4 \
  --stage2-epochs 15 \
  --num-gpus 1

# Identify
python identify_herbarium.py \
  --checkpoint ./runs/my_project/checkpoints/last.ckpt \
  --sources specsin.csv:images_1024/ \
  --output-dir ./runs/my_project/review/
```

## Dependencies and Build

**Python**: 3.12+

**Package manager**: `uv` (faster, more efficient than pip; `uv.lock` is the lockfile).

**Key dependencies**:
- **ML**: torch, torchvision, pytorch-lightning, timm, transformers (CLIP)
- **Data**: numpy, pandas, scikit-learn
- **Image I/O**: Pillow, opencv-python-headless
- **NVIDIA DALI**: installed separately (not in `pyproject.toml`) because the wheel name depends on CUDA version; see `pod_bootstrap.sh`.
- **Web UI**: nicegui
- **Cloud**: httpx, paramiko, keyring
- **Tracking**: wandb, tqdm

**Optional**: WandB for live training graphs (free for academic use, credential optional).

**GPU support**: NVIDIA CUDA 12.8 (hardcoded in `uv.lock` source URLs). Wheels come from pytorch.org CDN for speed on RunPod.

## Testing and Validation

No dedicated test suite exists. Validation occurs via:

1. **Download step** — GBIF API health checks, image URL verification, failure logging.
2. **Filter step** — logging of rejected images into `rejected/` and `live/` folders.
3. **Verify specsin** (`verify_specsin.py`) — checks CSV integrity, file presence, missing coordinates.
4. **Local test run** — use a small GBIF download (e.g., 50 images, one family) and train for 1 epoch to verify the full pipeline.

To debug a single pipeline stage, run its script with `--help` and test with a small subset (use `--limit 50` on download, mock data for training, etc.).

## Common Development Tasks

**Add a new training hyperparameter**:
1. Add the argument to `train_herbarium.py`'s argparse block.
2. Pass it through to the Lightning Trainer or model constructor.
3. If it appears in the web UI, add a field to the Train tab in `herbarium_pipeline_webui.py` and forward its value as an env var when dispatching to the pod (Cloud mode) or subprocess (Local mode).

**Add a new preprocessing step**:
1. Write a new script in the root directory following the naming pattern `<verb>_<noun>.py`.
2. Add it to the SCRIPTS dict in `herbarium_pipeline_webui.py`.
3. Wire up a new tab with controls and add it to the Run All sequencer.

**Debug a Cloud pod issue**:
1. Pod state and logs are persisted in `~/.herbarium-cloud/<project>.json`.
2. SSH directly: `ssh root@<pod-ip>:22001 -i ~/.ssh/id_ed25519_herbarium`.
3. Logs on the pod are in `/workspace/logs/`.
4. The running step can be interrupted with the **Cancel step** button; the pod stays alive for inspection.

**Modify training loss or metrics**:
1. `train_herbarium.py` defines the Lightning module class; loss is computed in its `training_step()` method.
2. Hierarchical multi-head mode combines species/genus/family losses via separate branches; edit the class accordingly.
3. Metrics are computed via `torchmetrics.MetricCollection` (accuracy, precision, recall, F1).

**Inspect predictions**:
1. `predictions.csv` is written by `identify_herbarium.py` with columns: path, true_label, top_pred, top_confidence, pred_2, conf_2, ..., flagged, mismatch.
2. The Review tab lets you browse and correct predictions interactively; changes save back to the CSV.
3. The Analysis tab loads the CSV and plots confusion matrices, per-species accuracy, top confusions.

## Development Practices

**Env vars on Cloud pods**: The orchestrator and the UI forward settings to the pod via environment variables, which the bootstrap script (`pod_bootstrap.sh`) reads. This avoids the need to edit pod-side code; all configuration is declarative from the UI.

**Async/await in webui**: The web UI uses asyncio for non-blocking subprocess dispatch and cloud API calls. Most tab logic is in async handlers connected to button clicks. NiceGUI manages the event loop; no need to call `asyncio.run()`.

**Reproducibility**: DALI pipelines, batch order, and RNG seeding are deterministic within a single run. Cross-machine reproducibility requires matching PyTorch version and CUDA. Seeds are set via `pytorch_lightning.seed_everything()`.

**Large image handling**: Herbarium scans can exceed PIL's default 128 MP limit. Both image scripts set `Image.MAX_IMAGE_PIXELS = None`.

**Multiprocessing in CPU-bound steps**: Filter & Crop and Resize use `multiprocessing.Pool` for parallel worker processes. Disk I/O stays in the main process (sequential) to avoid contention; workers do pure compute (decode, classify/resize, encode).

## IDE and Linting

No linting or formatting tools are currently configured in the repository. For consistency with the existing codebase:

- **Code style**: Follow PEP 8; match the style of surrounding code.
- **Type hints**: Used sparsely in existing code. Add them for clarity in new functions, especially in the cloud and webui packages.
- **Docstrings**: Main script docstrings include usage examples and brief function docs; no strict format enforced.

## Troubleshooting Tips

**Pod stuck provisioning**: Check the RunPod console for pod status. If it's RUNNING but SSH times out, the image may not have sshd; use the default base image.

**uv sync extremely slow**: PyPI's CDN can be slow from certain RunPod datacenters. This is a first-time cost; the warm R2 cache (disabled by default, enabled in `cloud_setup.md`) avoids it on subsequent pods.

**NCCL errors in multi-GPU mode**: If using hierarchical multi-head training without NVLink, set `NCCL_P2P_DISABLE=1` to avoid peer-to-peer comms. The DDP strategy automatically detects and enables `find_unused_parameters=True` when hierarchical mode is on.

**Filter step is slow**: CLIP classification (default) uses a GPU. Use `--filter-method hsv` for CPU-only filtering; it's faster and works on most temperate herbaria (fails on unusually colored or tropical specimens).

**Out of GPU memory during training**: Reduce `--batch-size`, increase `--grad-accum` to compensate, or use a smaller model (e.g., `efficientnet_b4` instead of ViT-Large).

**Settings disappeared after restart**: UI state is stored in `<repo>/.nicegui/storage-general.json` (relative to the launch directory) and the main config in `~/.config/herbarium_pipeline.json`. Cloud credentials live in the OS keyring. If the storage file is missing or corrupted, use **Apply paths** to restore project paths from the Projects root and Project name fields.
