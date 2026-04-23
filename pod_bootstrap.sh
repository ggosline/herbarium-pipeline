#!/usr/bin/env bash
# pod_bootstrap.sh — RunPod bootstrap for the herbarium pipeline.
# Usage: bash pod_bootstrap.sh {setup|download|prep|train|identify|backup|restore}
#
# Expected layout (Network Volume mounted at /workspace):
#   /workspace/Pipeline/        <- this repo
#   /workspace/data/            <- images, checkpoints, predictions
#   /workspace/.wandb_key       <- one-line wandb API key (chmod 600)
#   /workspace/.config/rclone/rclone.conf  <- R2 credentials

set -euo pipefail

# ─── paths on the Network Volume ──────────────────────────────────────────
WS=/workspace
# REPO = directory containing this script, so the name on disk doesn't matter
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=$WS/data
IMG_RAW=$DATA/images_raw
IMG_FILT=$DATA/images_filtered
IMG_640=$DATA/images_640
CKPT=$DATA/checkpoints
SPECSIN=$DATA/specsin.csv

TAXON_FAMILY="Rubiaceae"          # edit per run
WANDB_PROJECT="herbarium"
R2_REMOTE="r2:herbarium-backup"
REPO_URL="https://github.com/ggosline/herbarium-pipeline.git"

export RCLONE_CONFIG="$WS/.config/rclone/rclone.conf"

# ─── keep cache + venv on ephemeral container disk, NOT the volume ────────
# Container disk is fast local NVMe, free, and wiped on pod stop — perfect
# for caches. The network volume is paid storage + slower I/O.
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=/root/venv

mkdir -p "$IMG_RAW" "$IMG_FILT" "$IMG_640" "$CKPT"

# ─── one-time per pod: env setup ──────────────────────────────────────────
setup() {
  # 1. Clone / update code on the volume
  if [ ! -d "$REPO/.git" ]; then
    git clone "$REPO_URL" "$REPO"
  else
    git -C "$REPO" pull --ff-only
  fi

  # 2. Install uv if missing
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  # 3. Create venv + install locked deps. uv.lock is committed so this is
  #    deterministic across pods. --frozen refuses to update the lock.
  cd "$REPO"
  uv sync --frozen

  # 4. DALI — installed outside the lock because the wheel name depends on
  #    the pod's CUDA version. Skip on CPU-only pods (no nvcc, no GPU wheel).
  if command -v nvcc >/dev/null; then
    CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+')
    echo "Detected CUDA ${CUDA_MAJOR}.x — installing nvidia-dali-cuda${CUDA_MAJOR}0"
    uv pip install "nvidia-dali-cuda${CUDA_MAJOR}0"
  else
    echo "No CUDA detected — skipping DALI (CPU pod). Training will not work here."
  fi

  # 5. wandb login
  if [ -f "$WS/.wandb_key" ]; then
    uv run wandb login "$(cat "$WS/.wandb_key")"
  fi

  # 6. rclone for R2 backup
  if ! command -v rclone >/dev/null; then
    curl -fsSL https://rclone.org/install.sh | bash
  fi

  echo "Setup complete. Activate with: source /root/venv/bin/activate"
}

activate() { source /root/venv/bin/activate; }

# ─── step 1: download (runs fine on a CPU pod) ────────────────────────────
download() {
  activate
  python "$REPO/download_gbif_images.py" \
    --family "$TAXON_FAMILY" \
    --output-dir "$IMG_RAW" \
    --specsin "$SPECSIN" \
    --workers 16
}

# ─── step 2: filter + crop + resize ───────────────────────────────────────
prep() {
  activate
  python "$REPO/filter_and_crop_herbarium.py" \
    --input-dir "$IMG_RAW" \
    --output-dir "$IMG_FILT" \
    --specsin "$SPECSIN" \
    --batch-size 32 --workers 8

  python "$REPO/resize_images.py" \
    --input-dir "$IMG_FILT" \
    --output-dir "$IMG_640" \
    --max-size 640 --no-upscale \
    --batch-size 16 --workers 8
}

# ─── step 3: train (needs GPU pod, DALI installed) ────────────────────────
train() {
  activate
  cd "$REPO"
  python -u train_herbarium.py \
    --sources "$SPECSIN:$IMG_640" \
    --output-dir "$CKPT" \
    --model vit_large_patch16_dinov3.lvd1689m \
    --image-sz 640 \
    --batch-size 4 --accum 2 \
    --stage1-epochs 4 --stage2-epochs 15 \
    --cooldown-epochs 2 --cooldown-batch-size 5 \
    --cooldown-accum 2 --cooldown-lr 0.0001 \
    --num-gpus 1 --num-workers 8 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "runpod-$(date +%Y%m%d-%H%M)"
}

# ─── step 4: identify ─────────────────────────────────────────────────────
identify() {
  activate
  CKPT_FILE=$(ls -t "$CKPT"/*.ckpt | head -1)
  python -u "$REPO/identify_herbarium.py" \
    --checkpoint "$CKPT_FILE" \
    --sources "$SPECSIN:$IMG_640" \
    --output-dir "$DATA/predictions" \
    --batch-size 32
}

# ─── backup: push checkpoints + predictions to R2 ─────────────────────────
backup() {
  CKPT_FILE=$(ls -t "$CKPT"/*.ckpt 2>/dev/null | head -1)
  if [ -n "$CKPT_FILE" ]; then
    echo "Uploading $(basename "$CKPT_FILE") to R2..."
    rclone copy "$CKPT_FILE" "$R2_REMOTE/checkpoints/" \
      --progress --transfers 4 --s3-chunk-size 64M
  fi
  # Class-names JSON next to ckpt — tiny, critical
  rclone copy "$CKPT/" "$R2_REMOTE/checkpoints/" \
    --include "*.json" --progress
  if [ -d "$DATA/predictions" ]; then
    rclone copy "$DATA/predictions" "$R2_REMOTE/predictions/" --progress
  fi
  echo "Backup done: $R2_REMOTE"
}

# ─── restore: pull checkpoints back (fresh volume recovery) ───────────────
restore() {
  mkdir -p "$CKPT"
  rclone copy "$R2_REMOTE/checkpoints/" "$CKPT/" --progress
  echo "Restored checkpoints to $CKPT"
}

case "${1:?usage: $0 [setup|download|prep|train|identify|backup|restore]}" in
  setup)    setup ;;
  download) download ;;
  prep)     prep ;;
  train)    train ;;
  identify) identify ;;
  backup)   backup ;;
  restore)  restore ;;
  *)        echo "unknown step: $1"; exit 1 ;;
esac
