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
# Single image directory — download resizes, filter marks specsin, crop optional.
# Override with IMAGES_DIR if you want a custom path or the old three-dir layout.
IMAGES_DIR="${IMAGES_DIR:-$DATA/images}"
IMG_RAW="$IMAGES_DIR"
IMG_FILT="$IMAGES_DIR"
IMG_1024="$IMAGES_DIR"
CKPT=$DATA/checkpoints
SPECSIN=$DATA/specsin.csv
DWCA=$DATA/gbif.zip               # set to "" to use API (--family) instead

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

# ─── caches: split between volume (persistent) and container disk (fast) ──
# Wheels (UV_CACHE_DIR) and HF model weights (HF_HOME) live on the volume
# so every fresh pod reuses them. EUR-IS-1's egress to PyPI has been
# observed at <3 Mbps — cold uv sync took ~1.5 hr. Volume I/O beats that
# by ~50×.
#
# UV_PROJECT_ENVIRONMENT (the resolved venv) lives on the Network Volume so
# a fresh pod attached to the same volume reuses it across terminate/
# provision cycles — `uv sync --frozen` becomes a no-op on a warm volume,
# saving the ~1 min link step that would otherwise run every time.
# Trade-off: the venv is read on every Python invocation, and RunPod's
# network volume is slower than container-disk NVMe. In practice the
# import-time penalty is sub-second per process, far less than the
# cold-cache `uv sync` cost it replaces.
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PROJECT_ENVIRONMENT=/workspace/venv
export HF_HOME=/workspace/.cache/huggingface
# uv installs its own Python interpreter when the project's requires-python
# doesn't match what's on the image. Keep that interpreter on the volume so
# the venv's bin/python symlink stays valid across terminate / re-provision.
# Without this, uv defaults to /root/.local/share/uv/python — container disk,
# wiped on terminate, leaves every cached venv with a dangling python link.
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$UV_PYTHON_INSTALL_DIR"

# Shared R2 cache — one bucket serves every project, every user. R2 has
# no egress fees and pulls into RunPod fast (~50–100 Mbps typical), so
# wheels + HF weights round-trip in seconds instead of being re-fetched
# from PyPI/HF over the slow EUR-IS-1 path. Override via env if you keep
# multiple cache buckets, e.g. for different python versions.
CACHE_REMOTE="${CACHE_REMOTE:-r2:herbarium-cache}"

mkdir -p "$IMAGES_DIR" "$CKPT"

# ─── shared cache push/pull (R2) ──────────────────────────────────────────
# Both functions are best-effort: if rclone isn't installed yet, the R2
# remote isn't configured, or the bucket is empty/unreachable, they log
# and continue rather than failing the surrounding step. The whole point
# is to *speed up* setup — never to block it.
# Shared rclone flags for cache transfers:
#   --copy-links     : follow symlinks (HF dedups via blobs/ + snapshots/
#                      symlinks; uv build envs use python symlinks). Without
#                      this rclone skips them with a NOTICE and the cache
#                      structure is incomplete on restore.
#   --exclude builds-v*/** : uv's transient build environments. Per-resolution,
#                      no value cross-pod, often contain absolute symlinks
#                      that wouldn't resolve elsewhere anyway.
RCLONE_CACHE_FLAGS=(
  --transfers 16 --checkers 16 --fast-list --stats=10s
  --copy-links
  --exclude 'builds-v*/**'
)

cache_pull() {
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed yet — skipping cache pull"; return 0
  fi
  if ! rclone lsd "$CACHE_REMOTE" >/dev/null 2>&1; then
    echo "Cache remote $CACHE_REMOTE not accessible — skipping pull"; return 0
  fi
  echo "→ Pulling shared cache from $CACHE_REMOTE..."
  rclone copy "$CACHE_REMOTE/uv/"          "$UV_CACHE_DIR/" \
    "${RCLONE_CACHE_FLAGS[@]}" 2>&1 | tail -3 || true
  rclone copy "$CACHE_REMOTE/huggingface/" "$HF_HOME/" \
    "${RCLONE_CACHE_FLAGS[@]}" 2>&1 | tail -3 || true
  echo "✓ Cache pull done ($(du -sh "$UV_CACHE_DIR" "$HF_HOME" 2>/dev/null | tr '\n' ' '))"
}

_rclone_writable() {
  rclone lsd "$CACHE_REMOTE" >/dev/null 2>&1 \
    || rclone mkdir "$CACHE_REMOTE" 2>/dev/null
}

# Full push: wheel cache + HF weights. Use after a lockfile bump to seed R2.
cache_push() {
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed — skipping cache push"; return 0
  fi
  if ! _rclone_writable; then
    echo "Cache remote $CACHE_REMOTE not writable — skipping push"; return 0
  fi
  echo "→ Pushing shared cache to $CACHE_REMOTE (diff only)..."
  echo "  local sizes:"
  du -sh "$UV_CACHE_DIR" "$HF_HOME" 2>/dev/null | sed 's/^/    /'
  local rc_uv=0 rc_hf=0
  rclone copy "$UV_CACHE_DIR/" "$CACHE_REMOTE/uv/" \
    "${RCLONE_CACHE_FLAGS[@]}" --stats-one-line 2>&1 | tail -3 || rc_uv=$?
  rclone copy "$HF_HOME/"      "$CACHE_REMOTE/huggingface/" \
    "${RCLONE_CACHE_FLAGS[@]}" --stats-one-line 2>&1 | tail -3 || rc_hf=$?
  if [ "$rc_uv" -ne 0 ] || [ "$rc_hf" -ne 0 ]; then
    echo "⚠ Cache push had errors (uv rc=$rc_uv, hf rc=$rc_hf)" >&2
  else
    echo "✓ Cache push done"
  fi
}

# HF-only push: used after prep/train where model weights may have been
# downloaded for the first time. Skips the 25 GB wheel cache.
# Uses a byte-size stamp so the push is a no-op on every run after the
# first — CLIP and DINOv3 weights don't change unless the model changes.
hf_cache_push() {
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed — skipping HF cache push"; return 0
  fi
  local stamp="$WS/.hf_cache_stamp"
  local current_size
  current_size=$(du -sb "$HF_HOME" 2>/dev/null | cut -f1)
  if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$current_size" ]; then
    echo "HF cache unchanged since last push — skipping"
    return 0
  fi
  if ! _rclone_writable; then
    echo "Cache remote $CACHE_REMOTE not writable — skipping push"; return 0
  fi
  local size
  size=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)
  echo "→ Pushing HF model cache ($size) to $CACHE_REMOTE/huggingface/ (diff only)..."
  rclone copy "$HF_HOME/" "$CACHE_REMOTE/huggingface/" \
    "${RCLONE_CACHE_FLAGS[@]}" --stats-one-line 2>&1 | tail -3 \
    && { echo "✓ HF cache push done"; echo "$current_size" > "$stamp"; } \
    || echo "⚠ HF cache push had errors" >&2
}

# ─── full-venv cache (R2) ─────────────────────────────────────────────────
# The assembled venv is project-independent: same lockfile → identical bits.
# We tar it once and pull it directly on fresh pods so `uv sync --frozen`
# doesn't have to re-link wheels into a venv (~1 min per pod) and DALI doesn't
# have to be re-installed (~30 s).
#
# Cache key = venv-<hw>-<lockhash> (see venv_cache_key for the full rationale):
#   - hw: "cuda120" for any GPU pod, "cpu" otherwise. NOT the driver's CUDA
#     version — torch is cu12x and DALI is pinned to cuda120, so a GPU venv is
#     driver-version-independent.
#   - lockhash: CRLF-normalised sha256 of pyproject.toml + uv.lock *contents*
#     (path-independent), covering Python version via requires-python.
#
# Stored at $CACHE_REMOTE/venvs/<key>.tar (uncompressed — wheels are already
# binary, gzip would burn CPU for ~5% saving). Stream with rclone cat / rcat
# so we don't double-up I/O on the network volume with a staging tarball.

venv_cache_key() {
  # Cache key components:
  #   - hw: "cuda120" for any GPU pod, "cpu" otherwise. We deliberately do
  #     NOT stamp the driver's CUDA version. torch is pinned to a cu12x build
  #     in the lock, and DALI is pinned to the cuda120 wheel (which runs on
  #     ANY CUDA-12+ driver via backward compatibility) — so the venv is
  #     identical regardless of whether the pod's driver reports CUDA 12, 13,
  #     … Stamping nvidia-smi's version fragmented the cache and forced a full
  #     rebuild on every driver bump. The only hardware axis that changes the
  #     venv bytes is GPU (has DALI) vs CPU (no DALI).
  #   - lock_hash: hashes pyproject.toml + uv.lock *contents*, with CRLF
  #     stripped so a Windows (CRLF) and a Linux (LF) checkout of the same
  #     lockfile produce the same key — and NOT via `sha256sum <path>`, whose
  #     output embeds the file path ($REPO) and so changed the key if the repo
  #     ever moved. requires-python lives in pyproject.toml, so the Python
  #     version dimension is implicitly covered.
  #
  # We deliberately do NOT stamp `python3 --version` because uv may install
  # its own python (when system python is too old) which differs from
  # /usr/bin/python3 and was producing misleading keys.
  #
  # Memoised via VENV_CACHE_KEY env var: setup() exports it once, the
  # backgrounded venv_push subcommand inherits it, and venv_pull/venv_push
  # within the same run skip the double sha256.
  if [ -n "${VENV_CACHE_KEY:-}" ]; then
    echo "$VENV_CACHE_KEY"
    return
  fi
  # GPU vs CPU is the only hardware dimension that matters (see above).
  local hw="cuda120"
  local cm="${CUDA_MAJOR:-}"
  if [ -z "$cm" ]; then
    if ! { command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; }; then
      cm="cpu"
    fi
  fi
  [ "$cm" = "cpu" ] && hw="cpu"
  local lock_hash
  if [ -f "$REPO/uv.lock" ] && [ -f "$REPO/pyproject.toml" ]; then
    lock_hash=$(cat "$REPO/pyproject.toml" "$REPO/uv.lock" | tr -d '\r' \
                  | sha256sum | cut -c1-12)
  else
    lock_hash="nolock"
  fi
  VENV_CACHE_KEY="venv-${hw}-${lock_hash}"
  export VENV_CACHE_KEY
  echo "$VENV_CACHE_KEY"
}

# ─── uv, installed on demand ──────────────────────────────────────────────
# Only the slow paths (`uv sync`, `uv pip install`) need uv; the prebaked
# image ships a complete venv without it. Call this immediately before the
# first uv invocation, never up front — an unreachable astral.sh must not be
# able to abort a setup that was never going to run uv at all.
#
# The connect timeout matters: curl's default lets a blackholed host hang for
# ~5 min, and under `set -e` that failure is fatal.
ensure_uv() {
  if command -v uv >/dev/null; then
    return 0
  fi
  echo "Installing uv..."
  if ! curl -LsSf --connect-timeout 15 --retry 3 https://astral.sh/uv/install.sh | sh; then
    # Best-effort fallback. The CUDA base image ships no system python3 (uv
    # installs its own CPython), so this only helps on the runpod/pytorch
    # fallback image — hence the guard and the `|| true`.
    echo "⚠ astral.sh unreachable — trying 'pip install uv'" >&2
    if command -v python3 >/dev/null; then
      python3 -m pip install --quiet uv || true
    fi
  fi
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null; then
    echo "✗ Could not install uv, and this pod needs the slow venv path." >&2
    echo "  Check the pod's outbound network (astral.sh, pypi.org)." >&2
    return 1
  fi
}

# ─── prebaked image detection ─────────────────────────────────────────────
# On the prebaked Docker image (see /Dockerfile) the fully assembled venv is
# already at $HERBARIUM_PREBAKED_VENV (/opt/venv) — on the container's LOCAL
# NVMe, deliberately NOT under /workspace (the network-volume mount would
# shadow it). Its .cache_key was stamped at build time with the same
# venv-<hw>-<lockhash> key venv_cache_key() computes here.
#
# Echoes the baked venv path when it exists AND its key matches this repo's
# current lockfile; empty otherwise. A mismatch means the image predates a
# lockfile bump (CI hasn't rebuilt yet) — we ignore the baked venv and let the
# caller fall through to the R2/uv slow path. This is what makes the fast path
# work identically on an auto-provisioned pod and on a pod you allocated by
# hand in the console: it keys purely off what's on disk, not how the pod was
# created. Memoised so the double sha256 (via venv_cache_key) runs once.
prebaked_venv() {
  if [ -n "${PREBAKED_VENV_CHECKED:-}" ]; then
    echo "${PREBAKED_VENV:-}"; return
  fi
  PREBAKED_VENV=""
  local baked="${HERBARIUM_PREBAKED_VENV:-/opt/venv}"
  if [ -x "$baked/bin/python" ] && [ -f "$baked/.cache_key" ]; then
    local want have
    want=$(venv_cache_key)
    have=$(cat "$baked/.cache_key" 2>/dev/null)
    if [ "$want" = "$have" ]; then
      PREBAKED_VENV="$baked"
    else
      echo "⚠ prebaked venv key '$have' != needed '$want' — image predates a" >&2
      echo "  lockfile bump (CI not rebuilt yet?); using slow path instead." >&2
    fi
  fi
  PREBAKED_VENV_CHECKED=1
  export PREBAKED_VENV PREBAKED_VENV_CHECKED
  echo "$PREBAKED_VENV"
}

venv_pull() {
  # 0. Local short-circuit: reuse the on-volume venv only when its stamped
  #    cache key EXACTLY matches what this pod needs. The key is lockfile+hw
  #    only, so code-only changes don't change it — a matching venv is safe to
  #    reuse without re-pulling. But a mismatch must NOT be reused: a venv
  #    built on a different pod can carry a CUDA-major-specific DALI wheel
  #    (nvidia-dali-cudaXY0) that the current driver can't run. The old code
  #    reused "regardless of key" and re-stamped it, which silently ran a
  #    cuda130-DALI venv on a CUDA-12.8 driver → cudaErrorInsufficientDriver
  #    (35). On mismatch we now drop the stale venv and re-pull the correct one.
  local key
  key=$(venv_cache_key)
  if [ -f /workspace/venv/bin/python ]; then
    local have=""
    [ -f /workspace/venv/.cache_key ] && have=$(cat /workspace/venv/.cache_key 2>/dev/null)
    if [ "$have" = "$key" ]; then
      echo "✓ /workspace/venv matches $key — skipping pull"
      return 0
    fi
    echo "⚠ /workspace/venv has key '${have:-none}' but this pod needs '$key' "
    echo "  (may carry a CUDA/DALI build the driver can't run) — replacing it."
    rm -rf /workspace/venv
  fi
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed yet — skipping venv pull"
    return 1
  fi
  if ! rclone lsd "$CACHE_REMOTE" >/dev/null 2>&1; then
    echo "Cache remote $CACHE_REMOTE not accessible — skipping venv pull"
    return 1
  fi
  # New format: zstd-compressed tarball at venvs/<key>.tar.zst.
  # Older pods may have pushed plain .tar files — try both, prefer .zst.
  local remote_zst="$CACHE_REMOTE/venvs/${key}.tar.zst"
  local remote_tar="$CACHE_REMOTE/venvs/${key}.tar"
  local remote="" decompress=""
  if rclone lsf "$remote_zst" >/dev/null 2>&1; then
    remote="$remote_zst"
    decompress="zstd -d -T0"
  elif rclone lsf "$remote_tar" >/dev/null 2>&1; then
    remote="$remote_tar"
    decompress="cat"
  else
    echo "Venv cache miss for $key (will build + push)"
    return 1
  fi
  echo "→ Pulling cached venv $key from $remote..."
  rm -rf /workspace/venv.new
  mkdir -p /workspace/venv.new
  # Stream R2 → (optional zstd -d) → tar -x. Single network pass.
  if ! rclone cat "$remote" | $decompress | tar -xf - --strip-components=1 -C /workspace/venv.new; then
    echo "⚠ Venv pull/extract failed — falling through to slow path"
    rm -rf /workspace/venv.new
    # The object is corrupt (truncated/interrupted upload). Delete it now so
    # the next pod doesn't trip on the same poison — this run's later
    # venv_push re-creates a good one at the same key, but if this pod dies
    # before that push, the bad object would otherwise linger.
    echo "  removing corrupt cache object $remote"
    rclone deletefile "$remote" 2>/dev/null \
      || echo "  (couldn't delete $remote — remove it manually if it persists)"
    return 1
  fi
  # Sanity-check before swapping — catches push/pull format mismatches early.
  if [ ! -f /workspace/venv.new/bin/python ]; then
    echo "⚠ Extracted venv missing bin/python — tarball corrupt or wrong format"
    rm -rf /workspace/venv.new
    echo "  removing corrupt cache object $remote"
    rclone deletefile "$remote" 2>/dev/null \
      || echo "  (couldn't delete $remote — remove it manually if it persists)"
    return 1
  fi
  # Atomic swap. If a previous venv exists, replace it.
  rm -rf /workspace/venv.old
  if [ -d /workspace/venv ]; then
    mv /workspace/venv /workspace/venv.old
  fi
  mv /workspace/venv.new /workspace/venv
  rm -rf /workspace/venv.old
  echo "$key" > /workspace/venv/.cache_key
  echo "✓ Venv ready ($(du -sh /workspace/venv 2>/dev/null | cut -f1))"
  return 0
}

# Multipart upload tuning for the large (5–10 GB) venv tarball.
#   --s3-chunk-size 64M       — default 5MiB → 2000 parts for a 10GB blob;
#                               64MiB → ~160 parts. Fewer round-trips, better
#                               throughput on R2's per-part overhead.
#   --s3-upload-concurrency 8 — default 4. Most RunPod NICs comfortably push
#                               8 parallel chunks. The bottleneck is bandwidth
#                               not CPU.
#   --no-traverse             — skip rclone's plan-the-upload listing pass;
#                               we know the destination is a single object.
RCLONE_VENV_FLAGS=(
  --s3-chunk-size 64M
  --s3-upload-concurrency 8
  --no-traverse
  --stats=15s
)

venv_push() {
  # Synchronous push — blocks until R2 acks. Used by the explicit
  # `bash pod_bootstrap.sh venv_push` subcommand. setup() invokes
  # venv_push_bg instead to avoid blocking the user on a 5–15 min upload.
  if [ ! -d /workspace/venv ]; then
    echo "No /workspace/venv to push — skipping"
    return 0
  fi
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed — skipping venv push"
    return 0
  fi
  if ! rclone lsd "$CACHE_REMOTE" >/dev/null 2>&1 \
       && ! rclone mkdir "$CACHE_REMOTE" 2>/dev/null; then
    echo "Cache remote $CACHE_REMOTE not writable — skipping venv push"
    return 0
  fi
  local key
  key=$(venv_cache_key)
  echo "$key" > /workspace/venv/.cache_key
  # Compressed tarball. zstd -1 -T0 streams at ~500 MB/s (CPU-bound) with
  # 30–50% size reduction on the binary content (libtorch, nvidia-cudnn,
  # DALI .so files). Wheels themselves are zip-compressed and won't shrink
  # further but they're a small fraction of the venv.
  local compressor="cat" suffix="tar"
  if command -v zstd >/dev/null; then
    compressor="zstd -1 -T0 --long=27"
    suffix="tar.zst"
  fi
  local remote="$CACHE_REMOTE/venvs/${key}.${suffix}"
  local remote_tmp="${remote}.uploading"
  local size
  size=$(du -sh /workspace/venv 2>/dev/null | cut -f1)
  echo "→ Pushing /workspace/venv ($size on disk, $compressor compression) → $remote..."
  # Stream tar → zstd → R2, but to a temporary object first, then atomically
  # move it onto the real key. rcat writes straight to its destination, so an
  # interrupted push (pod reaped, watchdog, network blip) would otherwise
  # leave the *real* key truncated — poisoning every future fast path and
  # forcing the slow wheel-rebuild (exactly the 10-min corruption we hit).
  # A dead push now leaves only an orphan .uploading object, which the next
  # push overwrites; the real key only ever appears fully-formed.
  # pipefail makes the pipeline's status reflect rcat (not the trailing tail),
  # so the move only runs when the upload genuinely completed.
  if tar -cf - -C /workspace venv | $compressor | rclone rcat "$remote_tmp" \
       "${RCLONE_VENV_FLAGS[@]}" 2>&1 | tail -5 \
     && rclone moveto "$remote_tmp" "$remote" 2>&1 | tail -2; then
    echo "✓ Venv push done"
  else
    echo "⚠ Venv push had errors — non-fatal (real cache key untouched)" >&2
    rclone deletefile "$remote_tmp" 2>/dev/null || true
  fi
}

# Backgrounded variants. setup() invokes these so the user isn't blocked
# on a 5–15 min upload of a 5–10 GB venv — they can move on to download
# / prep / etc. while the tarball uploads. The watchdog's idle clock is
# reset by the next bootstrap step the user runs, so the pod won't be
# reaped under an in-flight upload as long as the user is interacting
# with it. Worst case: pod terminated mid-push → next setup hits the
# slow path again and retries.
#
# The upload runs in a fresh `bash pod_bootstrap.sh <subcommand>` invocation
# rather than a subshell of the current process — that avoids capturing
# stale function/variable state and survives ssh-session teardown via
# setsid (new session) + < /dev/null (closed stdin). Output goes to a
# log file the user can `tail -f` from another session.
_run_bg() {
  # Re-invoke this script in a detached session: setsid for a new process
  # group (immune to ctrl-C and parent exit), < /dev/null so it doesn't
  # share our stdin, log to /workspace/.last_<subcmd>.log so the user can
  # `tail -f` from another ssh.
  local subcmd=$1
  local log=/workspace/.last_${subcmd}.log
  : > "$log"
  setsid bash "$REPO/pod_bootstrap.sh" "$subcmd" >> "$log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

venv_push_bg() {
  if [ ! -d /workspace/venv ]; then
    echo "No /workspace/venv to push — skipping background push"
    return 0
  fi
  local size
  size=$(du -sh /workspace/venv 2>/dev/null | cut -f1)
  echo "→ Backgrounding venv push ($size) — tail /workspace/.last_venv_push.log to follow."
  _run_bg venv_push
}

cache_push_bg() {
  command -v rclone >/dev/null || return 0
  echo "→ Backgrounding wheel + HF cache push — tail /workspace/.last_cache_push.log to follow."
  _run_bg cache_push
}

# ─── one-time per pod: env setup ──────────────────────────────────────────
setup() {
  # Guard against concurrent runs (e.g. two SSH sessions both triggering setup).
  exec 9>/workspace/.setup.lock
  flock -n 9 || { echo "Another setup is already running — skipping"; exit 0; }
  # 1. Clone / update code on the volume.
  #    When run by the cloud orchestrator the code is SFTP-pushed (no .git),
  #    so we only fast-forward an existing git checkout. A fresh manual
  #    bootstrap can still clone if neither code nor .git is present.
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
  elif [ ! -f "$REPO/pod_bootstrap.sh" ]; then
    git clone "$REPO_URL" "$REPO"
  fi

  # 2. System packages we need before anything else can run. Idempotent —
  #    apt is a no-op when these are present (e.g. on the bigger runpod
  #    pytorch image they're prebaked).
  if ! command -v curl >/dev/null || ! command -v git >/dev/null \
       || ! command -v zstd >/dev/null || ! command -v rsync >/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends curl ca-certificates git zstd rsync
  fi

  # 3. rclone — installed BEFORE the venv pull so we can stream the tarball
  #    from R2. Without it venv_pull short-circuits and we fall through to
  #    the slow path.
  if ! command -v rclone >/dev/null; then
    curl -fsSL --connect-timeout 15 --retry 3 https://rclone.org/install.sh | bash
  fi

  # 4. uv is NOT installed here. It is only ever invoked on the slow path
  #    (`uv sync` / `uv pip install` below), and the prebaked image ships a
  #    complete venv with no uv on purpose (see Dockerfile). Installing it up
  #    front made every prebaked pod reach out to astral.sh for a tool it never
  #    ran — and under `set -e` an unreachable astral.sh killed setup outright.
  #    It is now installed lazily, inside the slow path that needs it.

  # 5. GPU-vs-CPU detection. CUDA_MAJOR only distinguishes a GPU pod ("12",
  #    or any number → GPU venv with DALI) from a CPU pod ("cpu" → no DALI);
  #    the specific driver CUDA version no longer feeds the cache key (torch
  #    is cu12x, DALI is pinned to cuda120 — see venv_cache_key). We still log
  #    the driver-reported version for diagnostics.
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=${CUDA_VER%%.*}
    : "${CUDA_MAJOR:=12}"   # fallback if header format ever changes
    echo "Detected CUDA $CUDA_VER (driver-supported) → GPU venv (DALI cuda120)"
  else
    CUDA_MAJOR="cpu"
    echo "No GPU detected — using CPU venv key. Training will not work here."
  fi
  export CUDA_MAJOR

  cd "$REPO"

  # 6. Fastest path — prebaked image. The venv is baked into the container at
  #    /opt/venv; nothing to pull, sync, or install. Falls through to the R2
  #    pull if the image is stale (key mismatch) or this isn't a prebaked pod.
  local baked
  baked=$(prebaked_venv)
  if [ -n "$baked" ]; then
    echo "✓ Prebaked image venv at $baked (key $(cat "$baked/.cache_key"))."
    echo "  Skipping R2 pull / uv sync / DALI install entirely."
  # 6′. Next-fastest — pull the assembled venv tarball from R2. On a hit we skip
  #    cache_pull / uv sync / DALI install — the assembled venv is what we want.
  elif venv_pull; then
    echo "Setup fast path complete — venv pulled from R2."
  else
    # ── Slow path: build the venv from wheels. ──
    # 6a′. uv is only needed from here on. See ensure_uv().
    ensure_uv

    # 6a. Pull shared wheel + HF caches from R2. Best-effort: never fails.
    cache_pull

    # 6b. Create venv + install locked deps. With a populated UV_CACHE_DIR
    #     this is link-only (no downloads) and finishes in ~1 min. Cold
    #     cache can take an hour on a slow PyPI path.
    #     --extra local-ml pulls the full ML stack (torch/timm/transformers/…),
    #     which moved to an optional group so plain `uv sync` stays slim for
    #     local UI installs. The pod always needs it.
    uv sync --frozen --extra local-ml

    # 6c. DALI — installed outside the lock (it's not in pyproject.toml).
    #     Pinned to the cuda120 wheel regardless of the pod's driver: torch is
    #     already a cu12x build, and cuda120 DALI runs on ANY CUDA-12+ driver
    #     (backward compatibility). Pinning keeps ONE canonical GPU venv that
    #     every pod reuses, instead of one venv per driver CUDA version — the
    #     over-fragmentation that used to force needless slow-path rebuilds.
    if [ "$CUDA_MAJOR" != "cpu" ]; then
      echo "Installing nvidia-dali-cuda120..."
      DALI_ARGS=(--python "$UV_PROJECT_ENVIRONMENT/bin/python" \
                 --extra-index-url https://developer.download.nvidia.com/compute/redist \
                 --only-binary=:all:)
      uv pip install "${DALI_ARGS[@]}" nvidia-dali-cuda120
    else
      echo "Skipping DALI install on CPU pod."
    fi

    # 6d. Push the wheels + HF cache and then the assembled venv back to R2
    #     so the next pod (any project) gets the fast path.
    #
    #     NOTE: do NOT run `uv cache prune --ci` here. Its purpose is the
    #     opposite of ours — it strips out the pre-built wheel binaries
    #     ("reduces cache size by ~90%") on the assumption that
    #     re-downloading from a fast PyPI is cheaper than persisting them.
    #     For our EUR-IS-1 → PyPI path that assumption inverts:
    #     re-downloading 8 GB of wheels takes ~1 hr, pulling from R2
    #     takes ~1 min. Pruning leaves R2 with metadata-only and forces
    #     every future pod to re-fetch.
    # Background the venv push so setup() returns in seconds. The user can
    # move on to download/prep while the venv tarball uploads to R2.
    #
    # We deliberately do NOT push the wheel cache (UV_CACHE_DIR) or HF
    # cache to R2. Reasoning: with the venv tarball on R2, the fast path
    # (every subsequent pod) doesn't need wheels — the venv is complete.
    # Wheels are only useful for the slow-path rebuild that happens when
    # the lockfile changes, which is rare. Pushing 14 GB of wheels +
    # 5 GB of HF models on every setup was costing more time than the
    # actual training. Manual `bash pod_bootstrap.sh cache_push` still
    # works if you want to seed the wheel cache after a lockfile bump.
    venv_push_bg
    echo "Venv push is running in the background; subsequent pods (and"
    echo "re-provisions) will hit the fast path once it finishes."
  fi

  # 6e. Fix perms — some wheels ship binaries without execute bits
  # (triton/ptxas, wandb-core, etc.). Runs regardless of fast/slow path.
  # Note: do NOT try to glob to site-packages/ in a variable — bash skips
  # pathname expansion in assignments, so find gets a literal "*" path and
  # silently fails. Search the whole lib/ subtree instead.
  find "$UV_PROJECT_ENVIRONMENT/lib" -type f -path '*/bin/*' \
    -exec chmod +x {} + 2>/dev/null || true

  # 6f. Venv mirror moved to train() / identify() — those are the only steps
  # whose cold-import cost (torch, lightning, DALI ~30–60 s on MooseFS)
  # outweighs the 5+ min mirror itself. download / prep have minimal
  # imports and don't benefit, so we skip the mirror at setup time.

  # 7. wandb login. Resolve the venv that's actually active: the prebaked image
  #    puts it at /opt/venv, the slow path at $UV_PROJECT_ENVIRONMENT
  #    (/workspace/venv). Guard on the binary existing so a missing wandb never
  #    aborts setup under `set -e` (which is what killed the train: rc 127).
  if [ -f "$WS/.wandb_key" ]; then
    local venv_base
    venv_base="$(prebaked_venv)"; venv_base="${venv_base:-$UV_PROJECT_ENVIRONMENT}"
    if [ -x "$venv_base/bin/wandb" ]; then
      "$venv_base/bin/wandb" login "$(cat "$WS/.wandb_key")"
    else
      echo "wandb not found in $venv_base/bin — skipping wandb login (non-fatal)."
    fi
  fi

  # Resolve the venv that's actually active — the prebaked image's /opt/venv or
  # the slow path's /workspace/venv — so the interactive env + final message
  # point an SSH-in user at a venv that really exists.
  local active_venv
  active_venv="$(prebaked_venv)"; active_venv="${active_venv:-/workspace/venv}"

  # 8. Make the cache + venv env vars sticky for interactive SSH sessions,
  #     so manually running `uv sync` / `uv pip install` from a shell
  #     hits the volume cache (15 GB) instead of the empty default at
  #     ~/.cache/uv. Idempotent — the marker prevents duplicate appends.
  if ! grep -q "# herbarium-pipeline env" /root/.bashrc 2>/dev/null; then
    cat >> /root/.bashrc <<'BASHRC'

# herbarium-pipeline env — written by pod_bootstrap.sh setup
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
export HF_HOME=/workspace/.cache/huggingface
export RCLONE_CONFIG=/workspace/.config/rclone/rclone.conf
export PATH="$HOME/.local/bin:$PATH"
# Send .pyc writes to tmpfs — saves repeated MFS round-trips on every import.
export PYTHONPYCACHEPREFIX=/dev/shm/pycache
BASHRC
    # Written after the quoted heredoc so the resolved path (prebaked vs slow)
    # is interpolated rather than the literal /workspace/venv.
    echo "export UV_PROJECT_ENVIRONMENT=$active_venv" >> /root/.bashrc
    echo "Added herbarium env exports to /root/.bashrc"
  fi

  start_watchdog

  echo "Setup complete. Activate with: source $active_venv/bin/activate"
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

# ─── detached step runner ─────────────────────────────────────────────────
# `spawn <step>` launches a pipeline step fully detached from the ssh session
# (setsid + nohup) so it survives a dropped connection, logging to
# $WS/logs/<step>.log and writing its exit code to <step>.rc when it finishes.
# While the step runs, $ACTIVITY_FILE is refreshed every 60s so the idle
# watchdog neither kills a long-running step (e.g. a multi-hour download) nor
# lingers on a finished, abandoned one. The orchestrator tails the log until
# the .rc file appears; on reconnect it re-tails the same log rather than
# launching a duplicate. Returns immediately after backgrounding.
spawn_step() {
  local step="${1:?spawn: step required}"
  local logdir="$WS/logs" log rc
  log="$logdir/$step.log"; rc="$logdir/$step.rc"
  mkdir -p "$logdir"
  rm -f "$rc"
  : > "$log"
  # Re-invoke ourselves for the actual run so the detach logic and the step's
  # env (inherited from this process) both carry through, without nested
  # bash -c quoting.
  setsid nohup bash "$REPO/pod_bootstrap.sh" __run_detached "$step" "$log" "$rc" \
    >/dev/null 2>&1 </dev/null &
  disown 2>/dev/null || true
  echo "spawned '$step' detached — log $log"
}

# Internal target of spawn_step (not for direct invocation). Runs the step,
# heartbeats the activity file while it runs, then records the exit code
# atomically so a reader never sees a half-written .rc.
run_detached() {
  local step="${1:?}" log="${2:?}" rc="${3:?}"
  bash "$REPO/pod_bootstrap.sh" "$step" >"$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    touch "$ACTIVITY_FILE" 2>/dev/null || true
    sleep 60
  done
  wait "$pid"; local code=$?
  touch "$ACTIVITY_FILE" 2>/dev/null || true
  echo "$code" >"$rc.tmp" && mv "$rc.tmp" "$rc"
}

# ─── local venv mirror (escape MooseFS for python imports) ────────────────
# /workspace/venv is on MooseFS; cold `import torch` walks hundreds of
# .py / .so files and each lookup is a network round-trip — ~30–60 s on
# MFS vs ~5–8 s on local NVMe. Mirror the venv to container-disk NVMe
# once, then `activate` prefers the local copy. Local copy dies with the
# pod, but R2 → /workspace/venv is the source of truth so re-mirror is
# cheap on a fresh pod.
#
# Shebangs in bin/* are rewritten to point at the local python so `wandb`,
# `pytest`, etc. don't fall back to the network venv.
_ensure_rsync() {
  command -v rsync >/dev/null && return 0
  echo "Installing rsync..."
  apt-get update -qq
  apt-get install -y --no-install-recommends rsync
}

# Run rsync with output adapted to where stdout is going.
# Interactive TTY → live --info=progress2 (\r-updated single line).
# Non-TTY (webui pipe) → quiet rsync + a periodic `du -sh` tick so the
#   log panel gets one line every 30 s instead of thousands per second.
# Args: <src/> <dst/>
_rsync_piped() {
  local src=$1 dst=$2
  if [ -t 1 ]; then
    rsync -a --info=progress2 "$src" "$dst"
    return $?
  fi
  # Non-TTY: silent rsync in the background, periodic size ticks in the foreground.
  rsync -aq "$src" "$dst" &
  local rs_pid=$!
  local started=$(date +%s)
  while kill -0 "$rs_pid" 2>/dev/null; do
    sleep 30
    kill -0 "$rs_pid" 2>/dev/null || break
    local size files elapsed
    size=$(du -sh "$dst" 2>/dev/null | cut -f1)
    files=$(find "$dst" -type f 2>/dev/null | wc -l)
    elapsed=$(( $(date +%s) - started ))
    echo "  rsync: ${elapsed}s elapsed, ${files} files, ${size} copied"
  done
  wait "$rs_pid"
}

LOCAL_VENV=/root/venv
LOCAL_UV_PY_DIR=/root/.uv-python

# Mirror the uv-managed CPython interpreter referenced by $src_venv to local
# NVMe and return the local interpreter path on stdout.
# The venv's python is a symlink to /workspace/.uv-python/<cpython-X>/bin/python3.X
# whose stdlib (~hundreds of .py + .so files) lives on MooseFS — that's the
# main source of slow imports, even after the venv tree itself is mirrored.
_mirror_uv_python() {
  local src_venv=$1
  local cfg="$src_venv/pyvenv.cfg"
  [ -f "$cfg" ] || { echo "no pyvenv.cfg in $src_venv" >&2; return 1; }
  local home_dir
  home_dir=$(awk -F' *= *' '/^home/{print $2}' "$cfg")    # .../bin
  local uv_py_root="${home_dir%/bin}"                     # .../cpython-X.Y.Z-...
  if [ ! -d "$uv_py_root" ]; then
    echo "uv-python root $uv_py_root not present — skipping" >&2
    return 1
  fi
  case "$uv_py_root" in
    /workspace/*) : ;;                                    # on MooseFS, mirror it
    *) echo "$uv_py_root" ; return 0 ;;                   # already local-ish
  esac
  local name dst
  name=$(basename "$uv_py_root")
  dst="$LOCAL_UV_PY_DIR/$name"
  if [ ! -x "$dst/bin/python3" ] && [ ! -x "$dst/bin/python3."* ]; then
    echo "→ Mirroring python interpreter $uv_py_root → $dst..." >&2
    mkdir -p "$LOCAL_UV_PY_DIR"
    rm -rf "$dst.new"
    _rsync_piped "$uv_py_root/" "$dst.new/" >&2
    rm -rf "$dst.old"
    [ -d "$dst" ] && mv "$dst" "$dst.old"
    mv "$dst.new" "$dst"
    rm -rf "$dst.old"
  fi
  echo "$dst"
}

# Populate $1 (a fresh /root/venv.new) by streaming the venv tarball straight
# from R2 to local NVMe, decompressing + untarring in one pass. This bypasses
# the per-file rsync off MooseFS — reading 30k+ small site-packages files over
# MFS runs ~25 MB/s (metadata-bound), whereas the 3.6 GB compressed R2 object
# is one sequential stream. Returns 0 on success; any miss/error → 1 so the
# caller falls back to the rsync path. Post-processing (shebang/interpreter
# rewrite in mirror_venv_local) is identical either way — the tarball carries
# the same /workspace-relative paths the volume venv has.
_local_venv_from_r2() {
  local dst_new=$1
  command -v rclone >/dev/null || return 1
  local key; key=$(venv_cache_key)
  local remote_zst="$CACHE_REMOTE/venvs/${key}.tar.zst"
  local remote_tar="$CACHE_REMOTE/venvs/${key}.tar"
  local remote="" decompress=""
  if rclone lsf "$remote_zst" >/dev/null 2>&1; then
    command -v zstd >/dev/null || return 1
    remote="$remote_zst"; decompress="zstd -d -T0"
  elif rclone lsf "$remote_tar" >/dev/null 2>&1; then
    remote="$remote_tar"; decompress="cat"
  else
    return 1
  fi
  echo "→ Fetching local venv from R2 $remote (skips MooseFS rsync)..."
  rm -rf "$dst_new"; mkdir -p "$dst_new"
  if ! rclone cat "$remote" | $decompress | tar -xf - --strip-components=1 -C "$dst_new"; then
    echo "  ⚠ R2 venv fetch failed — falling back to MooseFS rsync"
    rm -rf "$dst_new"
    # Same poison as venv_pull(): a truncated/corrupt object at this key
    # would otherwise fail every future pod's local mirror forever, since
    # nothing else ever re-validates it. Delete now; mirror_venv_local's
    # caller backfills a good copy from the (already-valid) /workspace/venv
    # once the rsync fallback below completes.
    echo "  removing corrupt cache object $remote"
    rclone deletefile "$remote" 2>/dev/null \
      || echo "  (couldn't delete $remote — remove it manually if it persists)"
    return 1
  fi
  if [ ! -f "$dst_new/bin/activate" ]; then
    echo "  ⚠ R2 venv incomplete (no bin/activate) — falling back to rsync"
    rm -rf "$dst_new"
    echo "  removing corrupt cache object $remote"
    rclone deletefile "$remote" 2>/dev/null \
      || echo "  (couldn't delete $remote — remove it manually if it persists)"
    return 1
  fi
  echo "  ✓ Local venv extracted from R2"
  return 0
}

mirror_venv_local() {
  # Prebaked image: the venv is already at /opt/venv on local NVMe — the whole
  # reason this mirror exists (escaping slow MooseFS imports) doesn't apply.
  if [ -n "$(prebaked_venv)" ]; then
    echo "Prebaked venv on local NVMe — no MooseFS mirror needed."
    return 0
  fi
  _ensure_rsync
  local src=/workspace/venv
  local dst="$LOCAL_VENV"
  [ -d "$src" ] || { echo "mirror_venv_local: $src missing — skipping"; return 1; }
  # Skip if a previous mirror is current (compare the cache_key stamp set
  # by venv_pull / venv_push). No stamp = always re-mirror to be safe.
  # Also re-mirror if the local interpreter is missing (e.g. an older mirror
  # that pre-dated the uv-python relocation).
  if [ -f "$src/.cache_key" ] && [ -f "$dst/.cache_key" ] \
     && [ "$(cat "$src/.cache_key")" = "$(cat "$dst/.cache_key")" ] \
     && [ -x "$(readlink -f "$dst/bin/python" 2>/dev/null)" ] \
     && case "$(readlink -f "$dst/bin/python" 2>/dev/null)" in /root/*) true;; *) false;; esac
  then
    echo "Local venv at $dst is current ($(cat "$dst/.cache_key"))"
    return 0
  fi
  local size
  size=$(du -sh "$src" 2>/dev/null | cut -f1)
  echo "→ Mirroring venv $src → $dst ($size)..."
  # Fast path: stream the compressed venv from R2 straight to local NVMe.
  # Fall back to the per-file MooseFS rsync only if R2 is missing/unusable.
  if ! _local_venv_from_r2 "$dst.new"; then
    rm -rf "$dst.new"
    mkdir -p "$dst.new"
    _rsync_piped "$src/" "$dst.new/"
    # $src (/workspace/venv) just proved itself good enough to rsync from, so
    # if R2 was missing or corrupt for this key, backfill it now — otherwise
    # every future pod repeats this same slow rsync until something else
    # happens to trigger a push. Backgrounded: don't block training on it.
    echo "→ Backfilling R2 venv cache for this key (background)..."
    venv_push_bg
  fi
  # uv writes absolute shebangs into bin/ entry-point scripts. Rewrite so
  # `wandb` etc. invoke the local python and import from local site-packages.
  find "$dst.new/bin" -maxdepth 1 -type f \
    -exec sed -i "1s|^#!$src/bin/|#!$dst/bin/|" {} + 2>/dev/null || true

  # Mirror the uv-managed Python interpreter (its stdlib is on MooseFS too)
  # and re-point pyvenv.cfg + python symlinks at the local copy.
  local local_py_root
  if local_py_root=$(_mirror_uv_python "$src"); then
    local local_py_bin="$local_py_root/bin"
    # Find the python3.X executable in the interpreter dir.
    local local_py
    local_py=$(ls "$local_py_bin"/python3.* 2>/dev/null | grep -E '/python3\.[0-9]+$' | head -1)
    if [ -z "$local_py" ]; then
      local_py="$local_py_bin/python3"
    fi
    # Rewrite pyvenv.cfg → local interpreter.
    sed -i "s|^home = .*|home = $local_py_bin|" "$dst.new/pyvenv.cfg"
    # Replace the python symlinks in the venv's bin/ with links to the local interpreter.
    for n in python python3; do
      if [ -L "$dst.new/bin/$n" ] || [ -e "$dst.new/bin/$n" ]; then
        ln -snf "$local_py" "$dst.new/bin/$n"
      fi
    done
    # python3.X — keep the same minor-version name if present.
    for f in "$dst.new"/bin/python3.*; do
      [ -e "$f" ] || continue
      ln -snf "$local_py" "$f"
    done
  else
    echo "⚠ Could not mirror uv-python; venv will still load stdlib from MooseFS." >&2
  fi

  rm -rf "$dst.old"
  [ -d "$dst" ] && mv "$dst" "$dst.old"
  mv "$dst.new" "$dst"
  rm -rf "$dst.old"
  # Some wheels (triton/ptxas, wandb-core) ship without execute bits.
  # Fix here too — rsync preserves the bad perms from the source venv.
  find "$dst/lib" -type f \( -path '*/bin/*' -o -name '*.so' \) \
    -exec chmod +x {} + 2>/dev/null || true
  echo "✓ Local venv ready at $dst (python → $(readlink -f "$dst/bin/python"))"
}

activate() {
  # Preference order: prebaked image venv (/opt/venv, already local NVMe) →
  # local-NVMe mirror of the network venv → network venv itself.
  local baked
  baked=$(prebaked_venv)
  if [ -n "$baked" ]; then
    source "$baked/bin/activate"
  elif [ -d "$LOCAL_VENV/bin" ]; then
    source "$LOCAL_VENV/bin/activate"
  else
    source /workspace/venv/bin/activate
  fi
  # Send .pyc writes to tmpfs so import-time bytecode regeneration doesn't
  # touch the (slow) network FS or churn container disk.
  export PYTHONPYCACHEPREFIX=/dev/shm/pycache
  mkdir -p "$PYTHONPYCACHEPREFIX" 2>/dev/null || true
}

# Fail fast (with a plain-English message) when the active venv's DALI build
# needs a newer CUDA driver than this pod has. DALI wheels are CUDA-major
# specific (nvidia-dali-cudaN0): a cudaN build runs on any driver whose max
# CUDA >= N, and fails with the cryptic cudaErrorInsufficientDriver (35) deep
# inside DALI pipeline construction when N > the driver's CUDA major. Because
# the venv is shared across pods via the network volume, a venv built on a
# CUDA-13 host can land on a CUDA-12 host — this catches that up front.
# Call after activate(). Returns non-zero on a hard mismatch.
_check_cuda_compat() {
  # Prebaked image ships nvidia-dali-cuda120, which runs on ANY CUDA-12+ driver
  # (backward compat) — the mismatch this guard catches (a venv built for a
  # newer CUDA than the driver) can't happen when the venv is pinned at build.
  if [ -n "$(prebaked_venv)" ]; then
    echo "  Prebaked cuda120 venv — CUDA/DALI driver compat guaranteed at build."
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local drv_cuda drv_major
  drv_cuda=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
  drv_major=${drv_cuda%%.*}
  case "$drv_major" in ''|*[!0-9]*) return 0 ;; esac        # unparseable → skip
  local site dali_info dali_num dali_major
  site=$(ls -d "$LOCAL_VENV"/lib/python*/site-packages 2>/dev/null | head -1)
  [ -d "$site" ] || site=$(ls -d /workspace/venv/lib/python*/site-packages 2>/dev/null | head -1)
  [ -d "$site" ] || return 0
  dali_info=$(ls -d "$site"/nvidia_dali_cuda*.dist-info 2>/dev/null | head -1)
  [ -n "$dali_info" ] || { echo "  (no nvidia-dali in venv — skipping DALI/driver check)"; return 0; }
  dali_num=$(basename "$dali_info" | grep -oP 'cuda\K[0-9]+' | head -1)
  case "$dali_num" in ''|*[!0-9]*) return 0 ;; esac
  dali_major=${dali_num%0}                                  # 130 → 13, 120 → 12
  if [ "$dali_major" -gt "$drv_major" ]; then
    echo "✗ CUDA mismatch: this venv has nvidia-dali-cuda${dali_num} (needs a CUDA"
    echo "  ${dali_major}.x driver) but the pod's driver supports only CUDA ${drv_cuda}."
    echo "  DALI pipeline construction would fail with cudaErrorInsufficientDriver (35)."
    echo "  Fix on the pod, then retry:"
    echo "    rm -rf /workspace/venv /root/venv && bash $REPO/pod_bootstrap.sh setup"
    echo "  (re-pulls the cuda120 venv, which runs on any CUDA-12+ driver)."
    return 1
  fi
  echo "  CUDA compat OK: nvidia-dali-cuda${dali_num} on driver CUDA ${drv_cuda}."
  return 0
}

# ─── step 1: download (runs fine on a CPU pod) ────────────────────────────
download() {
  activate
  # IIIF size is requested from the server; many institutions ignore it and
  # serve the full archival scan, so MAX_SIZE re-shrinks locally with PIL
  # right after each download. 1024 / 1200 are good defaults — keeps headroom
  # over the 640px training size while cutting disk ~10× vs full scans.
  # Override any of these via env vars from the orchestrator / Cloud tab:
  #   IIIF=2048  MAX_SIZE=1200  LIMIT=500  MAX_PER_SP=30  \
  #   MAX_PER_GENUS=50  MAX_PER_FAMILY=750  FROM_SPECSIN=/workspace/data/my.csv  \
  #   bash pod_bootstrap.sh download
  IIIF="${IIIF:-1200}"
  # Default MAX_SIZE to 1200: training is 640px, so a 1200-px longest-side
  # JPEG keeps headroom for review/inspection without ballooning disk.
  # Without a default, full archival scans (5–15 MB each) accumulate.
  MAX_SIZE="${MAX_SIZE:-1200}"
  WORKERS="${WORKERS:-16}"
  # --workers is a global pool. Without a per-host cap all 16 can pile
  # onto one provider; that is how images.mobot.org came to firewall a
  # pod mid-run. 4 is polite and costs ~nothing (traffic spreads over
  # dozens of hosts).
  PER_HOST_WORKERS="${PER_HOST_WORKERS:-4}"
  EXTRA=()
  if [ -n "${MAX_SIZE:-}" ];       then EXTRA+=(--max-size "$MAX_SIZE"); fi
  if [ -n "${LIMIT:-}" ];          then EXTRA+=(--limit "$LIMIT"); fi
  if [ -n "${MAX_PER_SP:-}" ];     then EXTRA+=(--max-per-species "$MAX_PER_SP"); fi
  if [ -n "${MAX_PER_GENUS:-}" ];  then EXTRA+=(--max-per-genus "$MAX_PER_GENUS"); fi
  if [ -n "${MAX_PER_FAMILY:-}" ]; then EXTRA+=(--max-per-family "$MAX_PER_FAMILY"); fi
  if [ "${SPECSIN_ONLY:-}" = "True" ]; then EXTRA+=(--specsin-only); fi
  if [ "${SKIP_FAILED:-0}" = "1" ];   then EXTRA+=(--skip-failed); fi
  if [ -n "${FROM_SPECSIN:-}" ] && [ -f "$FROM_SPECSIN" ]; then
    echo "Using specsin CSV: $FROM_SPECSIN (iiif-size=$IIIF${MAX_SIZE:+ max-size=$MAX_SIZE}${LIMIT:+ limit=$LIMIT}${MAX_PER_SP:+ max-per-sp=$MAX_PER_SP}${MAX_PER_GENUS:+ max-per-genus=$MAX_PER_GENUS}${MAX_PER_FAMILY:+ max-per-family=$MAX_PER_FAMILY})"
    python -u "$REPO/download_gbif_images.py" \
      --from-specsin "$FROM_SPECSIN" \
      --output-dir "$IMG_RAW" \
      --specsin "$SPECSIN" \
      --iiif-size "$IIIF" \
      --workers "$WORKERS" \
      --per-host-workers "$PER_HOST_WORKERS" \
      "${EXTRA[@]}"
  elif [ -n "$DWCA" ] && [ -f "$DWCA" ]; then
    echo "Using local DwC-A: $DWCA (iiif-size=$IIIF${MAX_SIZE:+ max-size=$MAX_SIZE}${LIMIT:+ limit=$LIMIT}${MAX_PER_SP:+ max-per-sp=$MAX_PER_SP}${MAX_PER_GENUS:+ max-per-genus=$MAX_PER_GENUS}${MAX_PER_FAMILY:+ max-per-family=$MAX_PER_FAMILY})"
    python -u "$REPO/download_gbif_images.py" \
      --dwca "$DWCA" \
      --output-dir "$IMG_RAW" \
      --specsin "$SPECSIN" \
      --iiif-size "$IIIF" \
      --workers "$WORKERS" \
      --per-host-workers "$PER_HOST_WORKERS" \
      "${EXTRA[@]}"
  elif [ -n "${TAXON_FAMILIES:-}" ]; then
    # Multiple families (e.g. split clade like old Olacaceae).
    # Submits one GBIF bulk download job, waits for it, then downloads the zip.
    # Requires GBIF_USER and GBIF_PASSWORD env vars (forwarded by orchestrator).
    GEO=()
    [ -n "${CONTINENT:-}" ]         && GEO+=(--continent "$CONTINENT")
    [ -n "${EXCLUDE_COUNTRIES:-}" ] && GEO+=(--exclude-countries $EXCLUDE_COUNTRIES)
    [ -n "${COUNTRIES:-}" ]         && GEO+=(--countries $COUNTRIES)
    echo "Requesting GBIF bulk download for: $TAXON_FAMILIES"
    # shellcheck disable=SC2086  # word-split on TAXON_FAMILIES is intentional
    python -u "$REPO/download_gbif_images.py" \
      --families $TAXON_FAMILIES \
      --dwca-out "$DWCA" \
      --output-dir "$IMG_RAW" \
      --specsin "$SPECSIN" \
      --iiif-size "$IIIF" \
      --workers "$WORKERS" \
      --per-host-workers "$PER_HOST_WORKERS" \
      "${GEO[@]}" \
      "${EXTRA[@]}"
  elif [ -n "${TAXON:-}" ]; then
    # Single taxon from the Download tab's rank + name box.
    #   RANK ∈ family | genus | order   (defaults to family)
    # family/genus query the live GBIF occurrence API; order resolves the
    # order's taxon key and pulls a bulk DwC-A (needs GBIF creds, forwarded
    # by the orchestrator). The download script writes order DwC-As to a
    # taxon-named zip next to the specsin, so re-runs of the same order reuse
    # it without clobbering an uploaded gbif.zip.
    RANK="${RANK:-family}"
    GEO=()
    [ -n "${CONTINENT:-}" ]         && GEO+=(--continent "$CONTINENT")
    [ -n "${EXCLUDE_COUNTRIES:-}" ] && GEO+=(--exclude-countries $EXCLUDE_COUNTRIES)
    [ -n "${COUNTRIES:-}" ]         && GEO+=(--countries $COUNTRIES)
    echo "GBIF download for $RANK: $TAXON"
    # shellcheck disable=SC2086  # word-split on country lists is intentional
    python -u "$REPO/download_gbif_images.py" \
      --"$RANK" "$TAXON" \
      --output-dir "$IMG_RAW" \
      --specsin "$SPECSIN" \
      --iiif-size "$IIIF" \
      --workers "$WORKERS" \
      --per-host-workers "$PER_HOST_WORKERS" \
      "${GEO[@]}" \
      "${EXTRA[@]}"
  else
    echo "ERROR: nothing to download. Set a taxon (rank + name) or a families" >&2
    echo "list in the Download tab, upload a DwC-A, or provide a specsin." >&2
    exit 2
  fi
}

# ─── step 2: filter + crop + resize ───────────────────────────────────────
prep() {
  activate
  # Override via env: FILTER_METHOD=hsv (default clip), NO_FILTER=1, NO_CROP=1
  FILTER_METHOD="${FILTER_METHOD:-clip}"
  FILTER_EXTRA=()
  if [ "${NO_FILTER:-0}" = "1" ]; then FILTER_EXTRA+=(--no-filter); fi
  if [ "${NO_CROP:-0}"   = "1" ]; then FILTER_EXTRA+=(--no-crop); fi
  # Single-dir layout: input == output, so every image looks "already done."
  # Force re-processing to ensure the specsin gets classification labels.
  if [ "$IMG_RAW" = "$IMG_FILT" ]; then FILTER_EXTRA+=(--force); fi
  # -u unbuffers stdout so tqdm bars stream live to the orchestrator's log
  # panel; without it Python buffers in chunks of ~4 KB and progress only
  # appears every few minutes on a slow CPU pod.
  python -u "$REPO/filter_and_crop_herbarium.py" \
    --input-dir "$IMG_RAW" \
    --output-dir "$IMG_FILT" \
    --specsin "$SPECSIN" \
    --filter-method "$FILTER_METHOD" \
    --batch-size 32 --workers 8 \
    "${FILTER_EXTRA[@]}"

  # Resize anything still over RESIZE_MAX (default 1200). Idempotent —
  # resize_images.py with --no-upscale skips files already within bounds.
  # Catches images downloaded before MAX_SIZE was defaulted, and any that
  # were filtered/cropped into a larger frame. Single-dir layout uses
  # --in-place since IMG_FILT and IMG_1024 are the same dir.
  RESIZE_MAX="${RESIZE_MAX:-1200}"
  RESIZE_OUT=(--in-place)
  if [ "$IMG_FILT" != "$IMG_1024" ]; then
    RESIZE_OUT=(--output-dir "$IMG_1024")
  fi
  python -u "$REPO/resize_images.py" \
    --input-dir "$IMG_FILT" \
    "${RESIZE_OUT[@]}" \
    --max-size "$RESIZE_MAX" \
    --no-upscale \
    --workers 8

  # Reconcile hasfile against what actually landed on disk.
  python -u "$REPO/verify_specsin.py" \
    --specsin "$SPECSIN" \
    --image-dir "$IMG_1024" --restore --rebuild-fnames

  # Push HF weights (CLIP model) downloaded during CLIP filter.
  # Skips the wheel cache — only model weights change here.
  # HSV filter downloads nothing, so skip. NO_CACHE_PUSH=1 to disable.
  if [ "${NO_CACHE_PUSH:-0}" != "1" ] && [ "$FILTER_METHOD" != "hsv" ]; then
    hf_cache_push
  fi

  # Build + cache the staged-training tarball now, right after verify_specsin
  # fixed hasfile/fname against what's actually on disk — this is the exact
  # point the image set becomes final for this data cycle. Backgrounded so
  # prep() doesn't block on the upload; train()'s stage_images() picks it up
  # from R2 instead of tar-ing 100k+ files off MooseFS itself.
  if [ "${NO_CACHE_PUSH:-0}" != "1" ]; then
    image_set_push_bg
  fi
}

# ─── image-set cache (R2) ──────────────────────────────────────────────────
# Same idea as the venv cache: build the staging tarball once, right after
# prep() finalises the image set, and let every pod that trains against this
# exact data pull one sequential R2 object instead of re-tar-ing 100k+ files
# off MooseFS itself. Uncompressed — the images are already-compressed JPEGs,
# so zstd would burn CPU for near-zero size reduction; the win here is one
# streamed object, not bandwidth.
IMAGE_SET_REMOTE_DIR="image-sets"

image_set_cache_key() {
  # specsin.csv's fname/hasfile columns are exactly what verify_specsin.py
  # just reconciled against the real files on disk, so hashing the whole
  # file is a cheap (single sequential read, no per-file MooseFS metadata
  # storm) stand-in for "what image set is this". Later steps (identify)
  # rewrite unrelated columns and will change this hash too — that only
  # costs an extra rebuild on the next prep, never risks stale/wrong images
  # being used for training.
  sha256sum "$SPECSIN" 2>/dev/null | cut -d' ' -f1
}

image_set_push() {
  if [ ! -d "$IMAGES_DIR" ]; then
    echo "No $IMAGES_DIR to push — skipping image-set cache push"
    return 0
  fi
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed — skipping image-set cache push"
    return 0
  fi
  local key
  key=$(image_set_cache_key)
  if [ -z "$key" ]; then
    echo "No specsin at $SPECSIN — skipping image-set cache push"
    return 0
  fi
  if ! _rclone_writable; then
    echo "Cache remote $CACHE_REMOTE not writable — skipping image-set push"
    return 0
  fi
  local remote="$CACHE_REMOTE/$IMAGE_SET_REMOTE_DIR/${key}.tar"
  local remote_tmp="${remote}.uploading"
  local size
  size=$(du -sh "$IMAGES_DIR" 2>/dev/null | cut -f1)
  echo "→ Pushing $IMAGES_DIR ($size) → $remote..."
  # Same atomic-temp-then-move pattern as venv_push: an interrupted upload
  # (pod reaped, watchdog, network blip) leaves only an orphan .uploading
  # object, never a truncated real key that would poison every future pull.
  if tar -cf - -C "$DATA" images | rclone rcat "$remote_tmp" \
       "${RCLONE_VENV_FLAGS[@]}" 2>&1 | tail -5 \
     && rclone moveto "$remote_tmp" "$remote" 2>&1 | tail -2; then
    echo "✓ Image-set push done"
  else
    echo "⚠ Image-set push had errors — non-fatal (real cache key untouched)" >&2
    rclone deletefile "$remote_tmp" 2>/dev/null || true
  fi
}

image_set_push_bg() {
  [ -d "$IMAGES_DIR" ] || return 0
  echo "→ Backgrounding image-set cache push — tail /workspace/.last_image_set_push.log to follow."
  _run_bg image_set_push
}

# Pull the cached image-set tarball straight from R2 to local NVMe/tmpfs,
# bypassing MooseFS entirely. Mirrors _local_venv_from_r2's structure,
# including deleting a corrupt/incomplete object on failure so a poisoned
# cache doesn't fail every future pod (see mirror_venv_local's fix history).
# Returns 0 on success; 1 on any miss/corruption so the caller falls back
# to _tar_stage_piped.
_image_set_from_r2() {
  local dst_new=$1
  command -v rclone >/dev/null || return 1
  local key
  key=$(image_set_cache_key)
  [ -n "$key" ] || return 1
  local remote="$CACHE_REMOTE/$IMAGE_SET_REMOTE_DIR/${key}.tar"
  rclone lsf "$remote" >/dev/null 2>&1 || return 1
  echo "→ Fetching staged image set from R2 $remote (skips MooseFS entirely)..."
  rm -rf "$dst_new"; mkdir -p "$dst_new"
  if ! rclone cat "$remote" | tar -xf - --strip-components=1 -C "$dst_new"; then
    echo "  ⚠ R2 image-set fetch failed — falling back to local tar-from-MooseFS"
    rm -rf "$dst_new"
    echo "  removing corrupt cache object $remote"
    rclone deletefile "$remote" 2>/dev/null \
      || echo "  (couldn't delete $remote — remove it manually if it persists)"
    return 1
  fi
  local n
  n=$(find "$dst_new" -type f | wc -l)
  if [ "$n" -lt 1 ]; then
    echo "  ⚠ R2 image-set empty after extract — falling back to local tar-from-MooseFS"
    rm -rf "$dst_new"
    echo "  removing corrupt cache object $remote"
    rclone deletefile "$remote" 2>/dev/null \
      || echo "  (couldn't delete $remote — remove it manually if it persists)"
    return 1
  fi
  echo "  ✓ Staged $n files from R2"
  return 0
}

# Tar-pipe copy: one continuous sequential read of $src over MooseFS instead
# of rsync's per-file protocol (its own stat/compare/checksum round trips on
# top of the open() itself). At ~114k small images the open() latency alone
# dominates (~25ms/file observed — same metadata-bound cost documented on
# mirror_venv_local's R2 fast path), so cutting rsync's per-file overhead is
# the win, not compression or bandwidth. Used as the fallback when the R2
# image-set cache misses. Not incremental — a crash mid-copy just leaves an
# orphan $dst (never exposed; stage_images only swaps it in after this
# returns 0), and the next run redoes the full copy.
# Args: <src dir> <dst dir, must already exist>
_tar_stage_piped() {
  local src=$1 dst=$2
  ( tar -cf - -C "$src" . | tar -xf - -C "$dst" ) &
  local tar_pid=$!
  local started
  started=$(date +%s)
  while kill -0 "$tar_pid" 2>/dev/null; do
    sleep 30
    kill -0 "$tar_pid" 2>/dev/null || break
    local size files elapsed
    size=$(du -sh "$dst" 2>/dev/null | cut -f1)
    files=$(find "$dst" -type f 2>/dev/null | wc -l)
    elapsed=$(( $(date +%s) - started ))
    echo "  tar-stage: ${elapsed}s elapsed, ${files} files, ${size} copied"
  done
  wait "$tar_pid"
}

# ─── stage images to local storage (escape MooseFS for training) ──────────
# /workspace is a MooseFS network volume on RunPod (mfs#…runpod.net:9421).
# Per-file read latency is ~ms, which starves DALI and pins the A100 at
# ~20% util during training. Copy the dataset to local NVMe (or tmpfs if
# it fits) once at the start of a run and DALI feeds the GPU at line rate.
#
# Picks tmpfs (/dev/shm) when the dataset comfortably fits with 25% RAM
# headroom; falls back to container-disk NVMe at /root/staged_images.
# Skips the copy entirely when $dest already holds a matching-size stage
# from an earlier call this pod's lifetime (see .stage_size_kb below) —
# tar isn't incremental like rsync was, so re-running train() on the same
# pod needs its own explicit "already done" check instead of relying on
# rsync's diff to make a repeat call cheap.
#
# Sets STAGED_IMAGES_DIR for downstream steps (consumed by train()).
stage_images() {
  local src="${IMAGES_DIR}"
  if [ ! -d "$src" ]; then
    echo "stage_images: source $src does not exist"; return 1
  fi
  local src_kb shm_free_kb need_kb
  src_kb=$(du -sk "$src" | cut -f1)
  shm_free_kb=$(df -k --output=avail /dev/shm 2>/dev/null | tail -1)
  # Need 1.25× source size to leave RAM headroom for DALI buffers + the
  # python/torch process. Without this guard a tight tmpfs fills, OOM-kills
  # the trainer mid-epoch.
  need_kb=$(( src_kb * 5 / 4 ))
  local dest
  if [ -n "$shm_free_kb" ] && [ "$shm_free_kb" -gt "$need_kb" ]; then
    dest=/dev/shm/staged_images
    echo "stage_images: tmpfs has $((shm_free_kb/1024)) MB free, source $((src_kb/1024)) MB → $dest"
  else
    dest=/root/staged_images
    echo "stage_images: tmpfs too small ($((shm_free_kb/1024)) MB free, need $((need_kb/1024)) MB) → $dest"
  fi
  if [ -f "$dest/.stage_size_kb" ] && [ "$(cat "$dest/.stage_size_kb")" = "$src_kb" ]; then
    echo "stage_images: $dest already matches source ($((src_kb/1024)) MB) — skipping copy"
    export STAGED_IMAGES_DIR="$dest"
    echo "$dest" > "$WS/.staged_images_dir"
    return 0
  fi
  rm -rf "$dest.new"
  mkdir -p "$dest.new"
  # Fast path: pull the prebuilt tarball straight from R2 (image_set_push_bg,
  # kicked off at the end of prep()). Fall back to tar-ing MooseFS directly
  # only on a cache miss/corruption, then backfill R2 so the next pod gets
  # the fast path — same pattern as mirror_venv_local's R2 fast path.
  if ! _image_set_from_r2 "$dest.new"; then
    rm -rf "$dest.new"
    mkdir -p "$dest.new"
    _tar_stage_piped "$src/" "$dest.new/"
    echo "→ Backfilling R2 image-set cache for this key (background)..."
    image_set_push_bg
  fi
  echo "$src_kb" > "$dest.new/.stage_size_kb"
  rm -rf "$dest.old"
  [ -d "$dest" ] && mv "$dest" "$dest.old"
  mv "$dest.new" "$dest"
  rm -rf "$dest.old"
  echo "stage_images: staged $(find "$dest" -type f | wc -l) files at $dest"
  export STAGED_IMAGES_DIR="$dest"
  # Persist for follow-up commands (train() in a separate invocation).
  echo "$dest" > "$WS/.staged_images_dir"
}

# ─── step 3: train (needs GPU pod, DALI installed) ────────────────────────
# All hyperparameters can be overridden via env vars; defaults match the
# pinned recipe in project_training_recipe.md.  Pass via the orchestrator's
# `env=` dict from the Cloud tab, or export them before calling this script.
train() {
  # Mirror venv to local NVMe — torch/DALI/lightning cold imports dominate
  # the first ~minute on MooseFS otherwise. Idempotent: re-mirroring after
  # the first run is a sub-second cache_key check.
  mirror_venv_local || echo "⚠ venv mirror skipped — imports will be slower."
  activate
  _check_cuda_compat || exit 3
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
  [ "${NO_GRAD_CKPT:-0}" = "1" ] && EXTRA+=(--no-grad-checkpoint)
  [ -n "${PREFETCH_QUEUE:-}" ] && EXTRA+=(--prefetch-queue "$PREFETCH_QUEUE")

  # Stage images to local storage if requested (escapes MooseFS I/O bottleneck).
  # STAGE_IMAGES=1 → run stage_images now and override the source dir.
  # If a previous step already staged, reuse $WS/.staged_images_dir.
  TRAIN_IMG_DIR="$IMG_1024"
  if [ "${STAGE_IMAGES:-0}" = "1" ]; then
    stage_images
    TRAIN_IMG_DIR="$STAGED_IMAGES_DIR"
    EXTRA+=(--use-mmap)        # safe on local FS, faster than non-mmap
  elif [ -f "$WS/.staged_images_dir" ]; then
    _staged=$(cat "$WS/.staged_images_dir")
    if [ -d "$_staged" ]; then
      echo "Reusing previously staged images: $_staged"
      TRAIN_IMG_DIR="$_staged"
      EXTRA+=(--use-mmap)
    fi
  fi

  echo "Train recipe: model=$MODEL  batch=$BATCH_SIZE×accum=$ACCUM  "\
       "stages=${STAGE1_EPOCHS}+${STAGE2_EPOCHS}+${COOLDOWN_EPOCHS}  "\
       "lr=${STAGE1_LR}/${STAGE2_LR}/${COOLDOWN_LR}  gpus=$NUM_GPUS"

  # Capture full output to a log on the network volume so SSH-channel reaps
  # (the webui losing the stream) don't lose the training output.
  TRAIN_LOG="$WS/train-$(date +%Y%m%d-%H%M%S).log"
  echo "Logging to $TRAIN_LOG"
  python -u train_herbarium.py \
    --sources "$SPECSIN:$TRAIN_IMG_DIR" \
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
    "${EXTRA[@]}" 2>&1 | tee -a "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "✗ training exited with rc=$rc — see $TRAIN_LOG"
    return "$rc"
  fi

  # Push the timm/DINOv3 backbone weights (~1.2 GB) downloaded on first
  # train, so other projects skip the download on their first train.
  hf_cache_push
}

# ─── step 4: identify ─────────────────────────────────────────────────────
identify() {
  # Same rationale as train(): only steps importing torch/timm benefit
  # from the venv mirror. Cached after first run.
  mirror_venv_local || echo "⚠ venv mirror skipped — imports will be slower."
  activate
  _check_cuda_compat || exit 3
  # Pick the checkpoint unless the caller passed CKPT_FILE explicitly. Prefer
  # the best-accuracy checkpoint (highest val_Accuracy in the filename), then
  # best valid_loss (lowest), then last.ckpt, then newest by mtime. Picking by
  # mtime alone is unsafe in a REUSED workspace: `last.ckpt` is overwritten in
  # place by every run — including one with a different backbone — so it can
  # mismatch $MODEL and blow up on load. The metric-named checkpoints carry
  # their own arch and survive reruns (this matches what `publish` selects).
  if [ -z "${CKPT_FILE:-}" ]; then
    CKPT_FILE=$(ls "$CKPT"/*val_Accuracy=*.ckpt 2>/dev/null \
      | sed -E 's/.*val_Accuracy=([0-9]+\.[0-9]+).*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
    [ -z "$CKPT_FILE" ] && CKPT_FILE=$(ls "$CKPT"/*valid_loss=*.ckpt 2>/dev/null \
      | sed -E 's/.*valid_loss=([0-9]+\.[0-9]+).*/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
    [ -z "$CKPT_FILE" ] && [ -f "$CKPT/last.ckpt" ] && CKPT_FILE="$CKPT/last.ckpt"
    [ -z "$CKPT_FILE" ] && CKPT_FILE=$(ls -t "$CKPT"/*.ckpt 2>/dev/null | head -1)
  fi
  if [ -z "${CKPT_FILE:-}" ]; then
    echo "✗ no checkpoint found in $CKPT" >&2; return 1
  fi
  echo "Selected checkpoint: $CKPT_FILE"
  # Tunables — override via env (the webui's Identify-tab Run button ships
  # these from the matching gs[...] keys).
  : "${MODEL:=vit_large_patch16_dinov3.lvd1689m}"
  : "${IMAGE_SZ:=640}"
  : "${BATCH_SIZE:=32}"
  : "${THRESHOLD:=}"
  : "${LOW_CONF_THRESHOLD:=}"
  : "${GEO_WEIGHT:=}"
  : "${GEO_SIGMA:=}"

  args=(--checkpoint "$CKPT_FILE"
        --model      "$MODEL"
        --sources    "$SPECSIN:$IMG_1024"
        --output-dir "$DATA/predictions"
        --image-sz   "$IMAGE_SZ"
        --batch-size "$BATCH_SIZE")
  [ -n "$THRESHOLD" ]          && args+=(--threshold          "$THRESHOLD")
  [ -n "$LOW_CONF_THRESHOLD" ] && args+=(--low-conf-threshold "$LOW_CONF_THRESHOLD")
  [ -n "$GEO_WEIGHT" ]         && args+=(--geo-weight         "$GEO_WEIGHT")
  [ -n "$GEO_SIGMA" ]          && args+=(--geo-sigma          "$GEO_SIGMA")

  echo "Identify: model=$MODEL image_sz=$IMAGE_SZ batch=$BATCH_SIZE ckpt=$CKPT_FILE"
  python -u "$REPO/identify_herbarium.py" "${args[@]}"
}

# ─── publish: push the best checkpoint to the Hugging Face Hub ─────────────
# push_model.py points at the whole checkpoints dir, auto-picks the best by
# valid_loss, and reads the embedded nameslist — so this just supplies the
# metadata (family/region/repo) via env. The HF write token is read from
# /workspace/.hf_token (pushed by the orchestrator) or $HF_TOKEN.
publish() {
  mirror_venv_local || echo "⚠ venv mirror skipped — imports will be slower."
  activate
  cd "$REPO"

  if [ ! -f "$WS/.hf_token" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "✗ no HF token at $WS/.hf_token and HF_TOKEN unset — set one in the Cloud tab."
    return 1
  fi

  # Default the family to the project name when not given explicitly.
  FAMILY="${FAMILY:-${PROJECT:-}}"

  args=(--ckpt "$CKPT")
  [ -n "${SELECT_BY:-}" ]    && args+=(--select-by    "$SELECT_BY")
  [ -n "${HF_REPO:-}" ]      && args+=(--repo         "$HF_REPO")
  [ -n "${HF_USER:-}" ]      && args+=(--hf-user      "$HF_USER")
  [ -n "${FAMILY:-}" ]       && args+=(--family       "$FAMILY")
  [ -n "${REGION:-}" ]       && args+=(--region       "$REGION")
  [ -n "${LABEL_LEVEL:-}" ]  && args+=(--label-level  "$LABEL_LEVEL")
  [ -n "${DISPLAY_NAME:-}" ] && args+=(--display-name "$DISPLAY_NAME")
  [ -n "${IMAGE_SZ:-}" ]     && args+=(--image-sz     "$IMAGE_SZ")

  echo "Publish: ckpt-dir=$CKPT family=${FAMILY:-?} repo=${HF_REPO:-<derived from family>}"
  python -u "$REPO/push_model.py" "${args[@]}"
}

# ─── backup: full project archive to R2 ───────────────────────────────────
# Pushes everything needed to delete the network volume and rebuild later
# without re-downloading from GBIF: latest ckpt, nameslist, specsin, the
# DwC-A snapshot, predictions/, and the resized image set tarred for fast
# transfer. Requires PROJECT env var so multiple projects can coexist
# under the same R2 bucket (e.g. r2:herbarium-backup/menispermaceae/).
backup() {
  : "${PROJECT:?PROJECT env var required (e.g. PROJECT=menispermaceae)}"
  REMOTE="$R2_REMOTE/$PROJECT"
  echo "→ Archiving project '$PROJECT' to $REMOTE"

  # 1. Checkpoints to archive: the most recent (irreplaceable) AND the best
  #    by valid_loss. Often the same file, but pushing both means publishing
  #    from R2 later (push_model.py) isn't stuck with whatever happened to be
  #    latest. Dedupe so identical paths aren't uploaded twice.
  LATEST_CKPT=$(ls -t "$CKPT"/*.ckpt 2>/dev/null | head -1)
  BEST_CKPT=$(ls "$CKPT"/*valid_loss=*.ckpt 2>/dev/null \
    | sed -E 's/.*valid_loss=([0-9.]+).*/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
  # Accuracy-best — highest val_Accuracy. This is what `publish` selects by
  # default, so R2 must carry it for a later publish-from-R2 to honour that.
  BEST_ACC=$(ls "$CKPT"/*val_Accuracy=*.ckpt 2>/dev/null \
    | sed -E 's/.*val_Accuracy=([0-9.]+).*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
  copied=""
  for f in "$LATEST_CKPT" "$BEST_CKPT" "$BEST_ACC"; do
    [ -n "$f" ] || continue
    case " $copied " in *" $f "*) continue ;; esac
    copied="$copied $f"
    echo "  ckpt: $(basename "$f")"
    rclone copy "$f" "$REMOTE/checkpoints/" \
      --progress --transfers 4 --s3-chunk-size 64M
  done
  # nameslist.json lives in $DATA (not $CKPT); copy it explicitly so the R2
  # archive is self-sufficient, plus any ckpt-side *.json metadata.
  [ -f "$DATA/nameslist.json" ] && \
    rclone copy "$DATA/nameslist.json" "$REMOTE/checkpoints/" --progress
  rclone copy "$CKPT/" "$REMOTE/checkpoints/" --include "*.json" --progress

  # 2. Per-project state
  [ -f "$SPECSIN" ] && rclone copy "$SPECSIN" "$REMOTE/" --progress
  [ -f "$DWCA" ]    && rclone copy "$DWCA"    "$REMOTE/" --progress

  # 3. Predictions output (small)
  if [ -d "$DATA/predictions" ]; then
    rclone copy "$DATA/predictions" "$REMOTE/predictions/" --progress
  fi

  # 4. Images — tar first so it's one large sequential upload
  #    instead of N small PUTs (faster + cheaper at R2's per-op pricing).
  #    Reuse an existing tar (left by download_images) if it's newer than
  #    every file in $IMAGES_DIR — saves ~5–15 minutes of re-taring on
  #    MooseFS for projects we just pulled images from.
  IMG_BASENAME="$(basename "$IMAGES_DIR")"
  if [ -d "$IMAGES_DIR" ]; then
    IMG_TAR="$DATA/${IMG_BASENAME}.tar"
    if [ -f "$IMG_TAR" ] \
       && [ -z "$(find "$IMAGES_DIR" -newer "$IMG_TAR" -print -quit 2>/dev/null)" ]; then
      echo "  reusing existing $(basename "$IMG_TAR") ($(du -h "$IMG_TAR" | cut -f1))"
    else
      echo "  bundling $IMAGES_DIR → $(basename "$IMG_TAR")"
      tar cf "$IMG_TAR" -C "$DATA" "$IMG_BASENAME"
    fi
    echo "  uploading $(du -h "$IMG_TAR" | cut -f1)"
    rclone copy "$IMG_TAR" "$REMOTE/" \
      --progress --transfers 4 --s3-chunk-size 64M
    # Leave the tar in place — symmetric with download_images, so a
    # follow-up download_results doesn't have to re-tar (~5–15 min on
    # MooseFS). Disk gets reclaimed when the pod terminates anyway.
    # Set BACKUP_REMOVE_TAR=1 to delete it explicitly if disk is tight.
    if [ "${BACKUP_REMOVE_TAR:-0}" = "1" ]; then
      rm -f "$IMG_TAR"
    fi
  fi

  echo "✓ Backup complete: $REMOTE"
  echo "  Safe to delete the network volume — restore with PROJECT=$PROJECT bash $0 restore"
}

# ─── restore: pull a project archive back onto a fresh volume ─────────────
restore() {
  : "${PROJECT:?PROJECT env var required (e.g. PROJECT=menispermaceae)}"
  REMOTE="$R2_REMOTE/$PROJECT"
  echo "→ Restoring project '$PROJECT' from $REMOTE"

  mkdir -p "$CKPT" "$DATA/predictions"

  # Checkpoints + metadata
  rclone copy "$REMOTE/checkpoints/" "$CKPT/" --progress

  # Per-project state files (specsin + DwC-A) land directly in $DATA
  rclone copy "$REMOTE/" "$DATA/" --include "specsin.csv" --include "gbif.zip" --progress

  # Predictions
  rclone copy "$REMOTE/predictions/" "$DATA/predictions/" --progress 2>/dev/null || true

  # Images tarball — pull then unpack on-pod (tar lives only briefly)
  IMG_BASENAME="$(basename "$IMAGES_DIR")"
  TAR_NAME="${IMG_BASENAME}.tar"
  # Also check legacy name for projects migrated from the three-dir layout
  for _try in "$TAR_NAME" "images_1024.tar"; do
    if rclone lsf "$REMOTE/$_try" >/dev/null 2>&1; then
      IMG_TAR="$DATA/$_try"
      echo "  pulling images tar ($_try)..."
      rclone copy "$REMOTE/$_try" "$DATA/" --progress \
        --transfers 4 --s3-chunk-size 64M
      echo "  extracting..."
      rm -rf "$IMAGES_DIR"
      tar xf "$IMG_TAR" -C "$DATA"
      rm -f "$IMG_TAR"
      break
    fi
  done

  echo "✓ Restore complete. Skip download/prep — go straight to identify or further training."
}

# ─── one-shot: rebuild the wheel cache from PyPI/CDN, push to R2 ──────────
# Used after a pod survived an earlier bootstrap that prune'd the wheel
# binaries (the historical `uv cache prune --ci` bug). Forces uv to
# re-download every wheel into UV_CACHE_DIR, then ships them to R2 so
# every future pod skips the slow PyPI fetch.
repair_cache() {
  if [ ! -d "$REPO" ]; then
    echo "Repo missing at $REPO — run setup first."; exit 1
  fi
  cd "$REPO"
  ensure_uv
  echo "→ Forcing re-download of all wheels into $UV_CACHE_DIR..."
  echo "  (uv sync --frozen --reinstall — this is the slow PyPI step we"
  echo "   want to do exactly once, then never again on any future pod.)"
  uv sync --frozen --reinstall --extra local-ml
  echo
  echo "→ Wheel cache size after redownload:"
  du -sh "$UV_CACHE_DIR" 2>/dev/null
  echo
  cache_push
}

# Dispatcher only runs when the script is executed directly, not when sourced
# (the webui sources this file to call individual functions like start_watchdog,
# and an unguarded ${1:?...} aborts the source with no positional args).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:?usage: $0 [setup|download|prep|stage_images|train|identify|publish|backup|restore|cache_pull|cache_push|cache_push_bg|venv_pull|venv_push|venv_push_bg|mirror_venv_local|image_set_push|image_set_push_bg|repair_cache|start_watchdog]}" in
    setup)              setup ;;
    download)           download ;;
    prep)               prep ;;
    stage_images)       stage_images ;;
    train)              train ;;
    identify)           identify ;;
    publish)            publish ;;
    backup)             backup ;;
    restore)            restore ;;
    cache_pull)         cache_pull ;;
    cache_push)         cache_push ;;
    cache_push_bg)      cache_push_bg ;;
    venv_pull)          venv_pull ;;
    venv_push)          venv_push ;;
    venv_push_bg)       venv_push_bg ;;
    mirror_venv_local)  mirror_venv_local ;;
    image_set_push)     image_set_push ;;
    image_set_push_bg)  image_set_push_bg ;;
    repair_cache)       repair_cache ;;
    start_watchdog)     start_watchdog ;;
    spawn)              spawn_step "$2" ;;
    __run_detached)     run_detached "$2" "$3" "$4" ;;
    *)            echo "unknown step: $1"; exit 1 ;;
  esac
fi
