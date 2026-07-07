# Dockerfile — prebaked environment for the herbarium pipeline on RunPod.
#
# WHY THIS EXISTS
#   Cold-start (env assembly) dominated short training runs and the venv-cache/
#   R2/MooseFS-mirror machinery in pod_bootstrap.sh was fragile (the source of
#   the CUDA error-35 failures). This image bakes torch + DALI + all locked deps
#   once, so a fresh pod already has the environment and `setup` is a near-noop.
#
# BUILT BY CI, NOT LOCALLY
#   .github/workflows/build-image.yml builds and pushes this to GHCR on any
#   change to pyproject.toml / uv.lock / Dockerfile. Nobody runs `docker build`
#   by hand.
#
# KEY CONSTRAINT — /workspace is the network-volume mount at runtime, so ANY
#   path under /workspace baked here is shadowed by the mount and vanishes.
#   The venv therefore lives at /opt/venv (container-local NVMe), which also
#   sidesteps the slow-MooseFS-import problem that mirror_venv_local works
#   around: /opt/venv is already local, so no mirror step is needed.
#
# The pipeline CODE is NOT copied in — pod_bootstrap.sh SFTP-pushes the repo at
# runtime, so code edits never require an image rebuild. We install only the
# locked *dependencies* (uv sync --no-install-project).

# CUDA 12.8 runtime. torch's cu128 wheels bundle their own CUDA libs; this base
# supplies the cuDNN/CUDA userspace that DALI links against. Ubuntu 24.04 ships
# Python 3.12 natively, matching requires-python (>=3.12,<3.14).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Fail the whole RUN pipeline if any stage errors (esp. the piped installers).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# System packages the bootstrap and transfers need at runtime: git (code pull),
# rclone + zstd (R2 fallback path is retained), curl/ca-certificates (installers),
# rsync (kept for the fallback mirror path), openssh-server (RunPod connects via
# direct SSH to port 22 — the stock runpod/pytorch base ships sshd, nvidia/cuda
# does NOT, so we install and wire it ourselves; see /start.sh below).
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates git zstd rsync unzip openssh-server \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && ssh-keygen -A

# uv — pinned interpreter + venv locations OUTSIDE /workspace so the runtime
# volume mount can't shadow them. These match the env vars pod_bootstrap.sh
# exports, except the venv is /opt/venv here (baked) vs /workspace/venv (slow
# path).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Only the resolution inputs — copying these (not the source) keeps the heavy
# `uv sync` layer cached across code changes; it re-runs only when the lockfile
# or manifest actually changes.
COPY pyproject.toml uv.lock ./

# Install the LOCKED dependency set into /opt/venv. --no-install-project skips
# building the herbarium-pipeline package itself (the code is run by path at
# runtime, never imported as an installed dist), so we don't need the source
# tree here. --frozen forbids lockfile drift — a mismatch fails the build.
RUN uv sync --frozen --no-install-project

# DALI — installed out-of-lock exactly as the slow path does. Pinned to the
# cuda120 wheel: torch is a cu12x build and cuda120 DALI runs on ANY CUDA-12+
# driver (backward compat), so this one image serves the whole GPU fallback
# list (12.8, CUDA-13, Blackwell — all verified).
RUN uv pip install \
        --python /opt/venv/bin/python \
        --extra-index-url https://developer.download.nvidia.com/compute/redist \
        --only-binary=:all: \
        nvidia-dali-cuda120

# Some wheels ship binaries without execute bits (triton/ptxas, wandb-core).
# The slow path fixes this at runtime; do it once here instead.
RUN find /opt/venv/lib -type f \( -path '*/bin/*' -o -name '*.so' \) \
        -exec chmod +x {} + 2>/dev/null || true

# Stamp the venv with the SAME cache key pod_bootstrap.sh computes
# (venv-cuda120-<12-char sha256 of CRLF-stripped pyproject.toml + uv.lock>).
# setup() compares this against the current lockfile: a match means the baked
# env is authoritative (fast path); a mismatch — e.g. the lock was bumped but
# CI hasn't rebuilt yet — makes setup() fall back to the R2/uv slow path.
RUN lock_hash="$(cat pyproject.toml uv.lock | tr -d '\r' | sha256sum | cut -c1-12)" \
    && echo "venv-cuda120-${lock_hash}" > /opt/venv/.cache_key \
    && echo "baked venv cache key: $(cat /opt/venv/.cache_key)"

# Marker the bootstrap greps for to know it's on a prebaked image.
ENV HERBARIUM_PREBAKED_VENV=/opt/venv

# ── SSH entrypoint ────────────────────────────────────────────────────────
# RunPod injects the pod's SSH public key via the PUBLIC_KEY env var and expects
# the container to run sshd on port 22 (this is what the stock runpod images do
# in their own start scripts). We replicate the minimum: authorize PUBLIC_KEY,
# persist RunPod's env vars so non-interactive SSH sessions (the orchestrator's
# command channel) still see RUNPOD_POD_ID etc., then hand PID 1 to sshd.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'mkdir -p /root/.ssh && chmod 700 /root/.ssh' \
    'if [ -n "${PUBLIC_KEY:-}" ]; then' \
    '  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys' \
    '  chmod 600 /root/.ssh/authorized_keys' \
    'fi' \
    '# Persist pod env so `ssh root@pod some-command` (no login shell) inherits it.' \
    'printenv | grep -E "^(RUNPOD_|PUBLIC_KEY=|HERBARIUM_)" > /etc/rp_environment || true' \
    'grep -q rp_environment /root/.bashrc 2>/dev/null || echo "set -a; . /etc/rp_environment 2>/dev/null; set +a" >> /root/.bashrc' \
    'exec /usr/sbin/sshd -D -e' \
    > /start.sh \
    && chmod +x /start.sh

WORKDIR /workspace

# sshd as PID 1 keeps the container alive and serves the orchestrator's SSH.
CMD ["/start.sh"]
