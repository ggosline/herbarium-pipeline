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
5. **Identify** (`identify_herbarium.py`) — runs inference on new images, saves a predictions.csv with top-5 predictions per image, flags mismatches with recorded labels, sorts unidentified specimens by prediction. Also writes `excluded_species.json`/`.csv` listing species dropped from training as too sparse (the model can't predict them). Auto-selects the best checkpoint by metric (see below).

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
    excluded_species.json          # Taxa dropped as too sparse (also embedded in each ckpt)
    checkpoints/
      last.ckpt                    # Most recent checkpoint (overwritten every run)
      epoch=XX-valid_loss=X.ckpt   # Best stage-2 checkpoint (lowest val loss)
      acc-epoch=XX-val_Accuracy=X.ckpt  # Best by val accuracy (what identify/publish pick)
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

**Prebaked pod image** — Pods boot from a Docker image built by GitHub Actions (`.github/workflows/build-image.yml`) and published to GHCR (`ghcr.io/ggosline/herbarium-pipeline:latest`, set by `DEFAULT_IMAGE` in `cloud/orchestrator.py`; override per session with `HERBARIUM_POD_IMAGE`). It bakes the full env (torch, DALI, locked deps) into `/opt/venv`, so `pod_bootstrap.sh setup` is near-instant and skips `uv sync`. The image uses zstd layers on a slim (non-cuDNN) CUDA base — torch ships its own cuDNN. Nobody builds Docker locally; the workflow rebuilds only when `pyproject.toml`/`uv.lock`/`Dockerfile`/the workflow change. The R2 wheel/venv cache remains only as a fallback for non-prebaked pods.

## Running the Application

### Local Setup

```bash
# Activate the conda environment (assumes it exists)
conda activate p12

# Slim install — web UI + cloud orchestration, no torch (~150 MB).
# Enough to drive the whole CLOUD pipeline from any machine.
uv sync

# Full install — adds the ML stack (torch, timm, transformers, opencv, wandb)
# needed to run training / identify / filter / Quick ID ON THIS MACHINE.
uv sync --extra local-ml

# Run the web UI
python herbarium_pipeline_webui.py
```

**Dependency split:** heavy ML deps live in the optional `local-ml` extra so a
plain `uv sync` stays slim and portable — the web UI imports no torch at module
load (Quick ID / analysis import it lazily). If you run any pipeline step
locally (not on a pod), you need `uv sync --extra local-ml`. The pod image and
`pod_bootstrap.sh` always install the extra; the UI exposes it as the
"Enable offline AI features" button.

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

**Key dependencies** — split into a slim core and an optional `local-ml` extra:

*Core (always installed by `uv sync`):*
- **Data**: numpy, pandas, scikit-learn
- **Image I/O**: Pillow (review display + PIL resize fallback)
- **Web UI**: nicegui (charts use its built-in ECharts — no matplotlib/plotly)
- **Cloud**: httpx, paramiko, keyring
- **CLI**: tqdm

*`local-ml` extra (`uv sync --extra local-ml`) — only to run pipeline steps on this machine:*
- **ML**: torch, torchvision, pytorch-lightning, torchmetrics, timm, transformers (CLIP)
- **Image**: opencv-python-headless (crop in filter_and_crop)
- **Tracking**: wandb
- **NVIDIA DALI**: installed separately (not in `pyproject.toml`) because the wheel name depends on CUDA version; see `pod_bootstrap.sh`.

The web UI imports none of the extra at module load, so the slim install runs
the UI and the entire *cloud* pipeline. The pod image / `pod_bootstrap.sh`
always install `--extra local-ml`.

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
4. The running step can be interrupted with the **Cancel step** button; the pod stays alive for inspection. Because steps are detached, Cancel must signal the pod (`cancel_step` → `kill -TERM -<pgid>` of the `setsid` session, escalating to `-KILL`), not merely cancel the local asyncio task — cancelling locally only stops *watching* the step, which then keeps running and gets re-attached by the next Run. No `.rc` is written, so the next Run starts fresh.
5. Editing a pipeline script and re-syncing does **not** affect a step already running: Python compiled the module at import. A code change only takes effect on the next Run.
6. Steps run detached on the pod (`spawn_step`: `setsid` + `nohup`, log to `/workspace/logs/<step>.log`, exit code to `<step>.rc`), so they survive a dropped web UI. On reconnect, **Attach** detects a live step via `running_step()` and re-tails its log from the top, and skips `sync_code` while a step runs. (`sync_code` finishes with `sed -i` on `pod_bootstrap.sh`, which replaces the inode rather than truncating, so a running bash keeps reading its original file — but don't rely on that; nothing else guarantees it.) Pressing the step's own Run button also re-attaches rather than launching a duplicate (`run_step` pgreps first).

**Modify training loss or metrics**:
1. `train_herbarium.py` defines the Lightning module class; loss is computed in its `training_step()` method.
2. Hierarchical multi-head mode combines species/genus/family losses via separate branches; edit the class accordingly.
3. Metrics are computed via `torchmetrics.MetricCollection` (accuracy, precision, recall, F1).

**Inspect predictions**:
1. `predictions.csv` is written by `identify_herbarium.py`. Key columns: `true_species`/`true_genus`/`true_family` (recorded labels, blank for indets), `pred_species`/`pred_genus`/`pred_family` with `confidence` (top-1), `indet`, `flagged`, `sparse`, and `top1_name`…`top5_name` each with matching `top{k}_prob`/`top{k}_genus`/`top{k}_family`.
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

**"error creating container" / ghcr.io pull timeout**: RunPod rents the machine, then its Docker daemon fails to reach ghcr.io (`net/http: request canceled while waiting for connection`). This is egress failure on the *host*, not a problem with the image — verify with an anonymous pull from your own machine (`curl` a token from `ghcr.io/token?scope=repository:ggosline/herbarium-pipeline:pull`, then HEAD the manifest). Note that the REST API reports `desiredStatus: RUNNING` for such a pod because that field is the *requested* state; the real tell is an empty `publicIp` and no `runtime` block. RunPod usually reaps the pod on its own within minutes. Terminate it (the network volume is a separate resource and survives), clear the stale `pod_id` from `~/.herbarium-cloud/<project>.json`, and re-provision to land on a different host. If it recurs across several attempts the whole DC's egress is suspect — and since a network volume pins the project to its DC and can't move, escape via the Docker Hub fallback: `HERBARIUM_POD_IMAGE=runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` (RunPod mirrors Docker Hub locally). `pod_bootstrap.sh` then finds no `/opt/venv` and falls through to the R2 venv pull. Unset it to restore the fast path.

**Same bad host on every re-provision**: Compare `machineId` across attempts. RunPod will happily place a replacement pod back on the machine that just failed, and a project's network volume pins it to one datacenter, so it cannot escape sideways. `provision()` now runs an egress preflight (`_egress_ok`) as soon as SSH is up: it curls `EGRESS_PROBE_URLS` (ghcr, GBIF, a major image host) and recycles the pod if fewer than `EGRESS_MIN_OK` answer, retrying up to `PROVISION_ATTEMPTS` times. That catches a bad host in seconds. If all attempts fail it raises with the machine IDs — file a RunPod ticket citing them, or build a pod in the console and use Attach. Note the light-tier GPU list leads with A4000/A5000 rather than the cheaper L4 precisely because EUR-IS-1 has few L4 hosts, so "first available L4" kept collapsing onto one bad machine.

**Symptoms of a broken-egress host** (all observed on the same machine): container never starts (ghcr pull timeout); `setup` hangs 300 s on a `curl`; `download` runs at ~0.3 img/s instead of ~9 img/s and marks 70–80% of images `FAILED` when the true dud rate is ~4%. That last one is the dangerous one — those spurious `hasfile=False` rows are what `--skip-failed` later drops, so a download that ran on a bad host will silently discard good specimens on the next resume. Re-run without `--skip-failed` to correct them before relying on it.

**`setup` hangs ~300s then dies on a curl timeout**: Any unreachable host in `pod_bootstrap.sh`'s install steps stalls for curl's default connect timeout and, under `set -euo pipefail`, kills setup. The installers now pass `--connect-timeout 15 --retry 3`, and `uv` is installed lazily by `ensure_uv()` at its first real use rather than up front — the prebaked image ships a complete venv and no `uv` on purpose, so the fast path must never depend on astral.sh being up. Keep it that way: if you add a network dependency to `setup`, put it behind the `prebaked_venv()` check, not before it.

**uv sync extremely slow**: Only relevant on the fallback path. Cloud pods boot from a prebaked image (see below) whose venv is baked at `/opt/venv`, so `setup` skips `uv sync` entirely. If a pod is *not* started from the prebaked image, it falls back to pulling a venv tarball from R2, or as a last resort `uv sync` from PyPI wheels — which can be slow from certain datacenters.

**NCCL errors in multi-GPU mode**: If using hierarchical multi-head training without NVLink, set `NCCL_P2P_DISABLE=1` to avoid peer-to-peer comms. The DDP strategy automatically detects and enables `find_unused_parameters=True` when hierarchical mode is on.

**Filter step is slow**: CLIP classification (default) uses a GPU. Use `--filter-method hsv` for CPU-only filtering; it's faster and works on most temperate herbaria (fails on unusually colored or tropical specimens).

**Out of GPU memory during training**: Reduce `--batch-size`, increase `--grad-accum` to compensate, or use a smaller model (e.g., `efficientnet_b4` instead of ViT-Large).

**Settings disappeared after restart**: UI state is stored in `<repo>/.nicegui/storage-general.json` (relative to the launch directory) and the main config in `~/.config/herbarium_pipeline.json`. Cloud credentials live in the OS keyring. If the storage file is missing or corrupted, use **Apply paths** to restore project paths from the Projects root and Project name fields.
