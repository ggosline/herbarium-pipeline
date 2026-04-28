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
IMG_1024=$DATA/images_1024
CKPT=$DATA/checkpoints
SPECSIN=$DATA/specsin.csv
DWCA=$DATA/gbif.zip               # set to "" to use API (--family) instead

TAXON_FAMILY="Rubiaceae"          # used only when DWCA is empty
WANDB_PROJECT="herbarium"
R2_REMOTE="r2:herbarium-backup"
REPO_URL="https://github.com/ggosline/herbarium-pipeline.git"

export RCLONE_CONFIG="$WS/.config/rclone/rclone.conf"

# ─── idle watchdog ────────────────────────────────────────────────────────
# Self-terminate the pod if no bootstrap step has run for IDLE_LIMIT_SECONDS.
# Activity timestamp is refreshed on every script entry/exit (trap below).
ACTIVITY_FILE="$WS/.last_activity"
IDLE_LIMIT_SECONDS="${IDLE_LIMIT_SECONDS:-3600}"
mkdir -p "$WS"
touch "$ACTIVITY_FILE"
trap 'touch "$ACTIVITY_FILE" 2>/dev/null || true' EXIT

# ─── keep cache + venv on ephemeral container disk, NOT the volume ────────
# Container disk is fast local NVMe, free, and wiped on pod stop — perfect
# for caches. The network volume is paid storage + slower I/O.
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=/root/venv

mkdir -p "$IMG_RAW" "$IMG_FILT" "$IMG_1024" "$CKPT"

# ─── one-time per pod: env setup ──────────────────────────────────────────
setup() {
  # 1. Clone / update code on the volume.
  #    When run by the cloud orchestrator the code is SFTP-pushed (no .git),
  #    so we only fast-forward an existing git checkout. A fresh manual
  #    bootstrap can still clone if neither code nor .git is present.
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
  elif [ ! -f "$REPO/pod_bootstrap.sh" ]; then
    git clone "$REPO_URL" "$REPO"
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
  #    the pod's CUDA version. Detect via nvidia-smi (always in PATH on a GPU
  #    pod) rather than nvcc — nvcc reports the container toolkit, nvidia-smi
  #    reports what the host driver supports, which is what DALI needs to match.
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    # CUDA Version appears in the plain `nvidia-smi` header (e.g. "CUDA Version: 13.0").
    # It is NOT exposed via --query-gpu on most driver versions, so parse the header.
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=${CUDA_VER%%.*}
    : "${CUDA_MAJOR:=12}"   # fallback if header format ever changes
    echo "Detected CUDA $CUDA_VER (driver-supported) — installing nvidia-dali-cuda${CUDA_MAJOR}0"
    # DALI wheels for newer CUDA/Python combos live on NVIDIA's index, not PyPI.
    # --only-binary refuses source builds (we don't have cmake/CUDA toolkit).
    DALI_ARGS=(--python "$UV_PROJECT_ENVIRONMENT/bin/python" \
               --extra-index-url https://developer.download.nvidia.com/compute/redist \
               --only-binary=:all:)
    uv pip install "${DALI_ARGS[@]}" "nvidia-dali-cuda${CUDA_MAJOR}0" \
      || { echo "DALI cuda${CUDA_MAJOR}0 wheel unavailable; falling back to cuda120"; \
           uv pip install "${DALI_ARGS[@]}" nvidia-dali-cuda120; }
  else
    echo "No GPU detected — skipping DALI (CPU pod). Training will not work here."
  fi

  # 5. wandb login
  if [ -f "$WS/.wandb_key" ]; then
    uv run wandb login "$(cat "$WS/.wandb_key")"
  fi

  # 6. rclone for R2 backup
  if ! command -v rclone >/dev/null; then
    curl -fsSL https://rclone.org/install.sh | bash
  fi

  start_watchdog

  echo "Setup complete. Activate with: source /root/venv/bin/activate"
}

# Background watchdog that polls $ACTIVITY_FILE and self-terminates the pod
# after IDLE_LIMIT_SECONDS of no bootstrap-step activity. Survives ssh
# disconnect (setsid + nohup) and is idempotent (skips if already running).
start_watchdog() {
  if pgrep -f herbarium-watchdog >/dev/null 2>&1; then
    echo "Watchdog already running."
    return
  fi
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "RUNPOD_POD_ID not set — watchdog disabled."
    return
  fi
  cat >/usr/local/bin/herbarium-watchdog <<'EOF'
#!/usr/bin/env bash
# Polls $ACTIVITY_FILE; runpodctl-removes self when idle exceeds limit.
ACTIVITY_FILE="${ACTIVITY_FILE:-/workspace/.last_activity}"
IDLE_LIMIT_SECONDS="${IDLE_LIMIT_SECONDS:-3600}"
[ -f "$ACTIVITY_FILE" ] || touch "$ACTIVITY_FILE"
while :; do
  sleep 60
  age=$(( $(date +%s) - $(stat -c %Y "$ACTIVITY_FILE" 2>/dev/null || echo 0) ))
  if [ "$age" -gt "$IDLE_LIMIT_SECONDS" ]; then
    echo "[watchdog $(date -Iseconds)] idle ${age}s > ${IDLE_LIMIT_SECONDS}s — terminating pod $RUNPOD_POD_ID"
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl remove pod "$RUNPOD_POD_ID" || echo "[watchdog] runpodctl failed (rc=$?)"
    else
      echo "[watchdog] runpodctl not on PATH — cannot self-terminate"
    fi
    exit 0
  fi
done
EOF
  chmod +x /usr/local/bin/herbarium-watchdog
  ACTIVITY_FILE="$ACTIVITY_FILE" \
  IDLE_LIMIT_SECONDS="$IDLE_LIMIT_SECONDS" \
  RUNPOD_POD_ID="$RUNPOD_POD_ID" \
    nohup setsid /usr/local/bin/herbarium-watchdog \
    >>"$WS/watchdog.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
  echo "Watchdog started (idle limit ${IDLE_LIMIT_SECONDS}s, log $WS/watchdog.log)"
}

activate() { source /root/venv/bin/activate; }

# ─── step 1: download (runs fine on a CPU pod) ────────────────────────────
download() {
  activate
  # IIIF size is requested from the server; many institutions ignore it and
  # serve the full archival scan, so MAX_SIZE re-shrinks locally with PIL
  # right after each download. 1024 / 1200 are good defaults — keeps headroom
  # over the 640px training size while cutting disk ~10× vs full scans.
  # Override any of these via env vars from the orchestrator / Cloud tab:
  #   IIIF=2048  MAX_SIZE=1200  LIMIT=500  MAX_PER_SP=30  bash pod_bootstrap.sh download
  IIIF="${IIIF:-1200}"
  EXTRA=()
  if [ -n "${MAX_SIZE:-}" ];   then EXTRA+=(--max-size "$MAX_SIZE"); fi
  if [ -n "${LIMIT:-}" ];      then EXTRA+=(--limit "$LIMIT"); fi
  if [ -n "${MAX_PER_SP:-}" ]; then EXTRA+=(--max-per-species "$MAX_PER_SP"); fi
  if [ -n "$DWCA" ] && [ -f "$DWCA" ]; then
    echo "Using local DwC-A: $DWCA (iiif-size=$IIIF${MAX_SIZE:+ max-size=$MAX_SIZE}${LIMIT:+ limit=$LIMIT}${MAX_PER_SP:+ max-per-sp=$MAX_PER_SP})"
    python -u "$REPO/download_gbif_images.py" \
      --dwca "$DWCA" \
      --output-dir "$IMG_RAW" \
      --specsin "$SPECSIN" \
      --iiif-size "$IIIF" \
      --workers 16 \
      "${EXTRA[@]}"
  else
    echo "No DWCA zip at $DWCA — falling back to GBIF API (--family $TAXON_FAMILY)"
    python -u "$REPO/download_gbif_images.py" \
      --family "$TAXON_FAMILY" \
      --output-dir "$IMG_RAW" \
      --specsin "$SPECSIN" \
      --iiif-size "$IIIF" \
      --workers 16 \
      "${EXTRA[@]}"
  fi
}

# ─── step 2: filter + crop + resize ───────────────────────────────────────
prep() {
  activate
  # -u unbuffers stdout so tqdm bars stream live to the orchestrator's log
  # panel; without it Python buffers in chunks of ~4 KB and progress only
  # appears every few minutes on a slow CPU pod.
  python -u "$REPO/filter_and_crop_herbarium.py" \
    --input-dir "$IMG_RAW" \
    --output-dir "$IMG_FILT" \
    --specsin "$SPECSIN" \
    --batch-size 32 --workers 8

  python -u "$REPO/resize_images.py" \
    --input-dir "$IMG_FILT" \
    --output-dir "$IMG_1024" \
    --max-size 1024 --no-upscale \
    --batch-size 16 --workers 8

  # Reconcile hasfile against what actually landed in IMG_1024 (resize failures
  # / decode errors silently leave specsin out of sync with disk).
  python -u "$REPO/verify_specsin.py" \
    --specsin "$SPECSIN" \
    --image-dir "$IMG_1024"
}

# ─── step 3: train (needs GPU pod, DALI installed) ────────────────────────
# All hyperparameters can be overridden via env vars; defaults match the
# pinned recipe in project_training_recipe.md.  Pass via the orchestrator's
# `env=` dict from the Cloud tab, or export them before calling this script.
train() {
  activate
  cd "$REPO"

  MODEL="${MODEL:-vit_large_patch16_dinov3.lvd1689m}"
  IMAGE_SZ="${IMAGE_SZ:-640}"
  BATCH_SIZE="${BATCH_SIZE:-12}"
  ACCUM="${ACCUM:-1}"
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-4}"
  STAGE1_LR="${STAGE1_LR:-0.005}"
  STAGE2_EPOCHS="${STAGE2_EPOCHS:-15}"
  STAGE2_LR="${STAGE2_LR:-0.0001}"
  STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-0}"
  COOLDOWN_EPOCHS="${COOLDOWN_EPOCHS:-2}"
  COOLDOWN_BATCH_SIZE="${COOLDOWN_BATCH_SIZE:-5}"
  COOLDOWN_LR="${COOLDOWN_LR:-0.0001}"
  COOLDOWN_ACCUM="${COOLDOWN_ACCUM:-2}"
  NUM_GPUS="${NUM_GPUS:-1}"
  NUM_WORKERS="${NUM_WORKERS:-8}"
  MAX_PER_SP="${MAX_PER_SP:-0}"
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-runpod-$(date +%Y%m%d-%H%M)}"

  EXTRA=()
  [ "${HIERARCHICAL:-0}" = "1" ] && EXTRA+=(--hierarchical)
  [ "${USE_LOCATION:-0}" = "1" ] && EXTRA+=(--use-location --geo-dim "${GEO_DIM:-64}")
  [ -n "${LABEL_LEVEL:-}" ] && [ "${HIERARCHICAL:-0}" != "1" ] && EXTRA+=(--label-level "$LABEL_LEVEL")
  [ -n "${SPECIES_WEIGHT:-}" ] && EXTRA+=(--species-weight "$SPECIES_WEIGHT")
  [ -n "${GENUS_WEIGHT:-}" ]   && EXTRA+=(--genus-weight   "$GENUS_WEIGHT")
  [ -n "${FAMILY_WEIGHT:-}" ]  && EXTRA+=(--family-weight  "$FAMILY_WEIGHT")
  [ -n "${RESUME:-}" ] && EXTRA+=(--resume "$RESUME")
  [ "${RESET_OPTIMIZER:-0}" = "1" ] && EXTRA+=(--reset-optimizer)
  [ "${MAX_PER_SP}" != "0" ] && EXTRA+=(--max-per-species "$MAX_PER_SP")

  echo "Train recipe: model=$MODEL  batch=$BATCH_SIZE×accum=$ACCUM  "\
       "stages=${STAGE1_EPOCHS}+${STAGE2_EPOCHS}+${COOLDOWN_EPOCHS}  "\
       "lr=${STAGE1_LR}/${STAGE2_LR}/${COOLDOWN_LR}  gpus=$NUM_GPUS"

  python -u train_herbarium.py \
    --sources "$SPECSIN:$IMG_1024" \
    --output-dir "$DATA" \
    --model "$MODEL" \
    --image-sz "$IMAGE_SZ" \
    --batch-size "$BATCH_SIZE" --accum "$ACCUM" \
    --stage2-batch-size "$STAGE2_BATCH_SIZE" \
    --stage1-epochs "$STAGE1_EPOCHS" --stage1-lr "$STAGE1_LR" \
    --stage2-epochs "$STAGE2_EPOCHS" --stage2-lr "$STAGE2_LR" \
    --cooldown-epochs "$COOLDOWN_EPOCHS" --cooldown-batch-size "$COOLDOWN_BATCH_SIZE" \
    --cooldown-accum "$COOLDOWN_ACCUM" --cooldown-lr "$COOLDOWN_LR" \
    --num-gpus "$NUM_GPUS" --num-workers "$NUM_WORKERS" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    "${EXTRA[@]}"
}

# ─── step 4: identify ─────────────────────────────────────────────────────
identify() {
  activate
  CKPT_FILE=$(ls -t "$CKPT"/*.ckpt | head -1)
  python -u "$REPO/identify_herbarium.py" \
    --checkpoint "$CKPT_FILE" \
    --model vit_large_patch16_dinov3.lvd1689m \
    --sources "$SPECSIN:$IMG_1024" \
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
