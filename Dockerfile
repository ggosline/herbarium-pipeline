# Dockerfile — prebaked TRAIN environment for the herbarium pipeline on RunPod.
#
# WHY THIS EXISTS
#   Cold-start (env assembly) dominated short training runs and the venv-cache/
#   R2/MooseFS-mirror machinery in pod_bootstrap.sh was fragile (the source of
#   the CUDA error-35 failures). This image bakes torch + DALI + all locked deps
#   once, so a fresh pod already has the environment and `setup` is a near-noop.
#
#   For the cheap download/filter/resize pods (no GPU compute), use the much
#   smaller Dockerfile.light instead — this image carries the full CUDA/torch/
#   DALI stack those steps don't need.
#
# BUILT BY CI, NOT LOCALLY
#   .github/workflows/build-image.yml builds and pushes this to GHCR on any
#   change to pyproject.toml / uv.lock / Dockerfile. Nobody runs `docker build`.
#
# MULTI-STAGE — the builder assembles the venv (which drags in a ~3–4 GB uv
#   wheel cache under UV_LINK_MODE=copy: every wheel exists once cached and once
#   unpacked). The final stage copies ONLY the finished venv + its interpreter,
#   so that duplicate cache, the uv binary, and build cruft never ship in the
#   layers RunPod has to pull.
#
# KEY CONSTRAINT — /workspace is the network-volume mount at runtime, so ANY
#   path under /workspace baked here is shadowed by the mount and vanishes. The
#   venv therefore lives at /opt/venv (container-local NVMe), which also
#   sidesteps the slow-MooseFS-import problem mirror_venv_local works around.
#
# The pipeline CODE is NOT baked in — pod_bootstrap.sh SFTP-pushes the repo at
# runtime, so code edits never require an image rebuild. We install only the
# locked *dependencies* (uv sync --no-install-project).

# ── builder ────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy

# only-managed above forces uv to install its own CPython under
# /opt/uv-python (the CUDA base ships no python), so the interpreter the venv
# symlinks to lives at a known path we can copy into the final stage.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Locked deps into /opt/venv. --no-install-project skips building the
# herbarium-pipeline package (code runs by path, never imported as a dist).
RUN uv sync --frozen --no-install-project

# DALI, out-of-lock, exactly as the slow path does. cuda120 wheel runs on ANY
# CUDA-12+ driver, so one image serves the whole GPU fallback list.
RUN uv pip install \
        --python /opt/venv/bin/python \
        --extra-index-url https://developer.download.nvidia.com/compute/redist \
        --only-binary=:all: \
        nvidia-dali-cuda120

# Some wheels ship binaries without execute bits (triton/ptxas, wandb-core).
RUN find /opt/venv/lib -type f \( -path '*/bin/*' -o -name '*.so' \) \
        -exec chmod +x {} + 2>/dev/null || true

# Stamp the venv with the SAME key pod_bootstrap.sh computes
# (venv-cuda120-<12-char sha256 of CRLF-stripped pyproject.toml + uv.lock>).
# On the pod, setup() only trusts the baked venv when this matches the current
# lockfile; a mismatch (lock bumped, CI not rebuilt yet) falls back to R2/uv.
RUN lock_hash="$(cat pyproject.toml uv.lock | tr -d '\r' | sha256sum | cut -c1-12)" \
    && echo "venv-cuda120-${lock_hash}" > /opt/venv/.cache_key \
    && echo "baked venv cache key: $(cat /opt/venv/.cache_key)"

# ── final ──────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    HERBARIUM_PREBAKED_VENV=/opt/venv

# Runtime-only system packages: git (code pull), rclone + zstd + rsync (R2 /
# mirror fallback path), unzip (rclone installer), openssh-server (RunPod
# connects via direct SSH to port 22 — the CUDA base has no sshd, so we wire it
# ourselves in /start.sh below). No uv / build tools — the venv is complete;
# if the fast path is ever missed, pod_bootstrap.sh installs uv itself.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates git zstd rsync unzip openssh-server \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && ssh-keygen -A

# Copy ONLY the finished venv + its interpreter — not the uv wheel cache.
COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /opt/venv /opt/venv

# ── SSH entrypoint ────────────────────────────────────────────────────────
# RunPod injects the pod's SSH public key via PUBLIC_KEY and expects sshd on
# port 22. Replicate the minimum: authorize PUBLIC_KEY, persist RunPod's env
# vars so non-interactive SSH (the orchestrator's command channel) inherits
# RUNPOD_POD_ID etc., then hand PID 1 to sshd.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'mkdir -p /root/.ssh && chmod 700 /root/.ssh' \
    'if [ -n "${PUBLIC_KEY:-}" ]; then' \
    '  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys' \
    '  chmod 600 /root/.ssh/authorized_keys' \
    'fi' \
    'printenv | grep -E "^(RUNPOD_|PUBLIC_KEY=|HERBARIUM_)" > /etc/rp_environment || true' \
    'grep -q rp_environment /root/.bashrc 2>/dev/null || echo "set -a; . /etc/rp_environment 2>/dev/null; set +a" >> /root/.bashrc' \
    'exec /usr/sbin/sshd -D -e' \
    > /start.sh \
    && chmod +x /start.sh

WORKDIR /workspace

# sshd as PID 1 keeps the container alive and serves the orchestrator's SSH.
CMD ["/start.sh"]
