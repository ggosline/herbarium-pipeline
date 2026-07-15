"""High-level cloud orchestrator for the herbarium pipeline.

Composes the four primitives in this package:

    runpod_client.py  — REST API (provision, terminate, query)
    pod_session.py    — SSH/SFTP (exec, file transfer, hashing)
    state.py          — per-project JSON (pod_id, volume_id, dwca hash, cost)
    secrets.py        — keyring (API key, R2 creds)

…into a small surface the webui's Cloud tab can call:

    orch = CloudOrchestrator(api_key, project="Sapindales")
    pod  = await orch.provision(purpose="light", on_log=log)
    await orch.upload_dwca(pod, Path("AfricanRubiaceae.zip"), on_log=log)
    await orch.run_step(pod, "setup",    on_log=log)
    await orch.run_step(pod, "download", on_log=log)
    await orch.run_step(pod, "prep",     on_log=log)
    await orch.run_step(pod, "train",    on_log=log)   # heavy GPU pod
    await orch.download_results(pod, Path("/local/project"), on_log=log)
    await orch.terminate(pod, on_log=log)              # keeps the volume

State persists in ``~/.herbarium-cloud/<project>.json``: closing and
relaunching the desktop app picks the same pod back up if it's still alive.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko

from . import secrets, state
from .pod_session import PodSession
from .runpod_client import PodInfo, RunPodAPIError, RunPodClient

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


# Parses a numeric val_loss from filenames like:
#   epoch=07-valid_loss=0.4072.ckpt   (stage 2 best)
#   s1-epoch=03-valid_loss=0.6797.ckpt
#   cd-epoch=02-valid_loss=0.39.ckpt
# Returns None for files without a parseable score (e.g. last.ckpt, last-v3.ckpt).
_LOSS_RE = re.compile(r"valid_loss=(\d+\.\d+)")


def _select_ckpts(
    ckpts: list[tuple[float, str]],
    mode: str,
    *,
    on_log: LogFn = print,
) -> list[str]:
    """Pick which remote checkpoint paths to include in a results download.

    ``ckpts`` is a list of (mtime, remote_path). ``mode`` is one of
    ``"all" | "latest" | "best+latest"``; unknown values fall back to "latest".
    """
    if not ckpts:
        return []
    if mode == "all":
        return [p for _, p in ckpts]

    latest = max(ckpts, key=lambda x: x[0])[1]
    if mode == "latest" or mode not in ("best+latest",):
        on_log(f"  ckpt filter=latest → {Path(latest).name}")
        return [latest]

    # best+latest
    scored: list[tuple[float, str]] = []
    for _, p in ckpts:
        m = _LOSS_RE.search(Path(p).name)
        if m:
            try:
                scored.append((float(m.group(1)), p))
            except ValueError:
                pass
    chosen = {latest}
    if scored:
        scored.sort(key=lambda x: x[0])
        best_loss, best_path = scored[0]
        chosen.add(best_path)
        on_log(f"  ckpt filter=best+latest → {Path(best_path).name} "
               f"(valid_loss={best_loss:.4f}), {Path(latest).name}")
    else:
        on_log(f"  ckpt filter=best+latest → {Path(latest).name} "
               f"(no parseable valid_loss in any filename — kept latest only)")
    return list(chosen)

# ── defaults ─────────────────────────────────────────────────────────────

# Prebaked pipeline image (see /Dockerfile + .github/workflows/build-image.yml).
# Bakes torch + nvidia-dali-cuda120 + all locked deps into /opt/venv, so a fresh
# pod skips the whole venv-assembly cold start (pod_bootstrap.sh detects the
# baked venv and no-ops setup). It also carries its own sshd (the nvidia/cuda
# base has none — the image's /start.sh wires RunPod's PUBLIC_KEY flow).
#
# GHCR anonymous pulls are NOT subject to Docker Hub's ~100/6h rate limit, so
# this also retires the "toomanyrequests at create_pod" failure mode the old
# Docker Hub base occasionally hit. The GHCR package must be public (one-time
# toggle after the first CI build) for RunPod to pull it without a token.
#
# Pin to a specific :sha-<short> tag instead of :latest when you need a pod to
# match an exact build while debugging. HERBARIUM_POD_IMAGE overrides it for a
# session (e.g. testing a branch build like :zstd-layers) without a code edit;
# unset it to fall back to :latest.
DEFAULT_IMAGE = os.environ.get(
    "HERBARIUM_POD_IMAGE", "ghcr.io/ggosline/herbarium-pipeline:latest"
)
DEFAULT_DATACENTER = "EUR-IS-1"
DEFAULT_VOLUME_GB = 80
DEFAULT_CONTAINER_DISK_GB = 40

# ── Egress preflight ──────────────────────────────────────────────────────
# A RunPod host can rent, boot, and serve SSH while its outbound connectivity
# is broken. Nothing in the pod lifecycle notices: `setup` hangs on whatever
# it curls first, and `download` grinds at a fraction of normal throughput
# while marking perfectly good image URLs as permanently failed. Probe the
# hosts every later step depends on, as soon as SSH is up, and recycle the pod
# if it can't reach them — a bad host costs seconds instead of hours.
#
# We test *connectivity*, not HTTP status: ghcr.io/v2/ answers 401 unauthed
# and some image servers 403 a bare GET. Any complete response proves egress.
EGRESS_PROBE_URLS: tuple[str, ...] = (
    "https://ghcr.io/v2/",            # pod image + prebaked venv fast path
    "https://api.gbif.org/v1/",       # download step: occurrence search
    "https://images.mobot.org/",      # download step: a major image provider
)
EGRESS_PROBE_TIMEOUT = 8   # seconds, connect and total, per probe
EGRESS_MIN_OK = 2          # probes that must answer; one flaky provider is OK
PROVISION_ATTEMPTS = 3     # fresh pods to try before giving up on the DC

# When a fresh project has no network volume yet, provisioning may place the
# pod (and its new volume) in whichever of these datacenters actually has a
# GPU free — tried in order after the requested/default DC. A volume can't
# move between DCs, so once a project has data this list is *not* used: the
# volume's DC wins (see provision()).
DEFAULT_DATACENTERS: list[str] = [
    "EUR-IS-1", "EUR-IS-2", "EU-RO-1", "EU-SE-1", "EU-NL-1",
    "US-KS-2", "US-CA-2", "CA-MTL-1",
]

# GPU preference, in priority order, per purpose. create_pod accepts the whole
# list and RunPod places the pod on the *first available* one — so a busy
# top-choice GPU no longer means a failed provision. RunPod honours list order,
# so these are ranked by *preference* (reliable + good value first), NOT raw
# power: a free A6000 beats waiting on a contested A100. The chosen GPU + its
# $/hr is logged at provision time.
#
# Light tier handles download / prep / identify — any cheap reliable card.
# Train tier needs ≥24 GB VRAM. Default config (ViT-Large @ 640px, batch 4,
# gradient checkpointing on) fits comfortably in 24 GB, so 48 GB cards are
# roomy. The 48 GB RTX A6000 has trained well in practice and is usually far
# easier to get than an A100 — hence it leads. A100/H100 (80 GB) stay in the
# list as fallbacks for when they happen to be free; the 24 GB 4090 is the
# cheap last resort.
GPU_BY_PURPOSE: dict[str, list[str]] = {
    # L4 is the cheapest card here but EUR-IS-1 carries very few L4 hosts, so
    # "first available L4" collapsed onto one machine — and when that machine
    # had broken egress, every provision landed back on it (three in a row:
    # ghcr pull timeout, then astral.sh timeout, then 76% image-fetch failures
    # at 0.3 img/s). The card was never at fault; the thin host pool was. Lead
    # with the A4000/A5000 for a deeper pool, keep L4 as a cheap fallback.
    # The egress preflight below is the real guard — this is just odds.
    "light": [
        "NVIDIA RTX A4000",
        "NVIDIA RTX A5000",
        "NVIDIA L4",
        "NVIDIA GeForce RTX 4090",
    ],
    "train": [
        # Blackwell (sm_120), 96 GB. Verified: the shared cuda120 DALI builds
        # and runs on an RTX PRO 6000 via PTX JIT (no native sm_120 cubins, so
        # a small first-use JIT cost) — confirmed working in practice. Led
        # to the top since EUR-IS-1 runs deep availability on this card
        # (deeper than A6000/L40S), giving faster, more reliable provisioning.
        # If a future DALI/driver combo regresses on it, give Blackwell pods
        # a pod-local cuda130 DALI rather than touching the shared cuda120 venv.
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",       # 96 GB
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",  # 96 GB
        "NVIDIA RTX A6000",               # 48 GB — proven, usually available
        "NVIDIA RTX 6000 Ada Generation", # 48 GB — Ada sibling of the A6000
        "NVIDIA L40S",                    # 48 GB — Ada datacenter card
        "NVIDIA L40",                     # 48 GB — non-S variant, more supply
        "NVIDIA A100 80GB PCIe",          # 80 GB — fallback when free
        "NVIDIA A100-SXM4-80GB",          # 80 GB — the old default; contested
        "NVIDIA H100 PCIe",               # 80 GB — overkill but fine if free
        "NVIDIA H100 NVL",                # 94 GB — H100 variant
        "NVIDIA H100 80GB HBM3",          # 80 GB — SXM H100 variant
        "NVIDIA GeForce RTX 4090",        # 24 GB — cheap last resort
    ],
}

# Pod-side layout, mirrored in pod_bootstrap.sh.
REMOTE_REPO = "/workspace/Pipeline"
REMOTE_DATA = "/workspace/data"
REMOTE_DWCA = f"{REMOTE_DATA}/gbif.zip"
REMOTE_LOGS = "/workspace/logs"   # detached-step logs + .rc files (see spawn_step)

# Files to push from the local Pipeline checkout to the pod. Anything not
# in this list (caches, wandb runs, dot-dirs) stays local.
SYNC_FILE_PATTERNS = ("*.py", "*.sh", "pyproject.toml", "uv.lock", "README.md")


# ── handle returned to callers ───────────────────────────────────────────

@dataclass(frozen=True)
class PodHandle:
    """Everything the UI needs to know about a running pod.

    Returned by :meth:`CloudOrchestrator.provision` and accepted by every
    other method. Cheap to copy; the orchestrator caches the underlying
    SSH session by ``pod_id`` separately.
    """
    pod_id: str
    ssh_host: str
    ssh_port: int
    cost_per_hr: float
    network_volume_id: str | None
    started_at: float


# ── orchestrator ─────────────────────────────────────────────────────────

class CloudOrchestrator:
    """One instance per (project, api_key). Not thread-safe."""

    def __init__(
        self,
        api_key: str,
        project: str,
        *,
        state_root: Path | None = None,
        key_filename: str | Path | None = None,
        local_pipeline_dir: Path | None = None,
    ):
        self._api_key = api_key
        self._project = project
        self._state_root = state_root
        self._key_filename = key_filename
        # Where on the local machine the .py / .sh files live. Defaults to
        # the parent of the cloud/ package, i.e. the Pipeline checkout.
        self._local_pipeline = (
            Path(local_pipeline_dir) if local_pipeline_dir
            else Path(__file__).resolve().parent.parent
        )
        self._client: RunPodClient | None = None
        self._sessions: dict[str, PodSession] = {}
        self._state = state.load(project, root=state_root)

    @property
    def state(self) -> state.ProjectState:
        return self._state

    @property
    def project(self) -> str:
        return self._project

    @property
    def key_filename(self) -> str | Path | None:
        return self._key_filename

    def current_cost_usd(self) -> float:
        """USD spent across all pods for this project, including the active one."""
        return state.current_run_cost(self._state)

    async def aclose(self) -> None:
        for s in list(self._sessions.values()):
            try:
                await s.aclose()
            except Exception:
                pass
        self._sessions.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CloudOrchestrator":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ── internals ─────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        state.save(self._state, root=self._state_root)

    def _rp(self) -> RunPodClient:
        if self._client is None:
            self._client = RunPodClient(self._api_key)
        return self._client

    def _handle_from_pod(self, pod: PodInfo) -> PodHandle:
        ep = pod.ssh_endpoint
        if ep is None:
            raise RuntimeError(f"pod {pod.id} has no SSH endpoint")
        host, port = ep
        return PodHandle(
            pod_id=pod.id,
            ssh_host=host,
            ssh_port=port,
            cost_per_hr=pod.cost_per_hr,
            network_volume_id=pod.network_volume_id,
            started_at=self._state.pod_started_at or time.time(),
        )

    async def _egress_ok(self, handle: PodHandle, *, on_log: LogFn) -> bool:
        """True when the pod can reach the outside world.

        Runs one bounded curl per probe URL and counts complete responses.
        ``curl`` is asked only to connect and read — no ``-f`` — so a 401 or
        403 still counts as reachable. A host that cannot answer at least
        ``EGRESS_MIN_OK`` of them is not worth keeping.
        """
        urls = " ".join(f"'{u}'" for u in EGRESS_PROBE_URLS)
        cmd = (
            f"ok=0; for u in {urls}; do "
            f'if curl -s -o /dev/null --connect-timeout {EGRESS_PROBE_TIMEOUT} '
            f'-m {EGRESS_PROBE_TIMEOUT} "$u"; then ok=$((ok+1)); '
            f'else echo "  unreachable: $u"; fi; done; echo "EGRESS_OK=$ok"'
        )
        # Opening the session is inside the try on purpose: a host broken
        # enough to fail the probe is a host whose SSH may not come up at
        # all, and that must recycle the pod rather than abort provisioning.
        try:
            session = await self._ensure_session(handle, on_log=on_log)
            _, out = await session.exec_capture(cmd)
        except Exception as e:
            on_log(f"  egress probe could not run: {e}")
            return False

        n_ok = 0
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("EGRESS_OK="):
                n_ok = int(line.partition("=")[2] or 0)
            elif line.startswith("unreachable:"):
                on_log(f"  {line}")
        total = len(EGRESS_PROBE_URLS)
        on_log(f"  egress probe: {n_ok}/{total} hosts reachable "
               f"(need {EGRESS_MIN_OK})")
        return n_ok >= EGRESS_MIN_OK

    async def _ensure_volume(
        self, *, size_gb: int, data_center_id: str, on_log: LogFn,
    ) -> tuple[str, str]:
        """Return ``(volume_id, data_center_id)`` for the project's network
        volume, creating one if absent.

        Returns the *actual* DC of the volume — which may differ from the
        requested one when a saved volume exists in another region. Network
        volumes can't move between DCs; rather than silently delete the
        volume to honour a DC change, we honour the saved volume's DC and
        warn. To switch regions, the user must delete the saved volume
        explicitly.
        """
        if self._state.volume_id:
            try:
                vols = await self._rp().list_volumes()
            except Exception as e:
                on_log(f"  ⚠ couldn't list volumes ({e}) — trusting saved id")
                return self._state.volume_id, (self._state.data_center_id
                                                or data_center_id)
            match = next((v for v in vols if v.id == self._state.volume_id), None)
            if match is None:
                on_log(f"  ⚠ saved volume {self._state.volume_id} not found "
                       f"in account — clearing and creating fresh")
                self._state.volume_id = None
                self._state.data_center_id = None
            elif match.data_center_id != data_center_id:
                on_log(f"  ⚠ saved volume {match.id} is in "
                       f"{match.data_center_id}; provision asked for "
                       f"{data_center_id}. Honouring the volume's DC to keep "
                       f"data — pod will be placed in {match.data_center_id}. "
                       f"Delete the volume manually to switch regions.")
                self._state.data_center_id = match.data_center_id
                self._save_state()
                return match.id, match.data_center_id
            else:
                return self._state.volume_id, match.data_center_id
        on_log(f"Creating network volume ({size_gb} GB, {data_center_id})...")
        vol = await self._rp().create_volume(
            name=f"herb-{self._project}",
            size_gb=size_gb,
            data_center_id=data_center_id,
        )
        on_log(f"  volume {vol.id} created")
        self._state.volume_id = vol.id
        self._state.data_center_id = vol.data_center_id
        self._save_state()
        return vol.id, vol.data_center_id

    async def _ensure_session(
        self, handle: PodHandle, *, on_log: LogFn,
        attempts: int = 10, delay: float = 3.0,
    ) -> PodSession:
        """Return a connected SSH session for the pod, creating + caching one
        on first call. Retries through the brief gap between RunPod
        publishing the SSH port and sshd accepting connections.

        EOFError is caught alongside paramiko.SSHException / OSError because
        paramiko raises bare EOFError when the server closes the TCP
        connection during the SSH banner exchange — common while sshd is
        still starting up, or when the pod was reaped between provision
        and connect (in which case all retries will fail and the final
        error message points at the real problem).
        """
        cached = self._sessions.get(handle.pod_id)
        if cached is not None:
            if cached.is_alive():
                return cached
            on_log("Cached SSH session is dead — rebuilding")
            await cached.aclose()
            self._sessions.pop(handle.pod_id, None)
        last_err: Exception | None = None
        for i in range(attempts):
            session = PodSession(
                handle.ssh_host, handle.ssh_port, key_filename=self._key_filename,
            )
            try:
                await session.connect()
                self._sessions[handle.pod_id] = session
                return session
            except (paramiko.SSHException, OSError, EOFError) as e:
                last_err = e
                await session.aclose()
                on_log(f"SSH attempt {i + 1}/{attempts}: {type(e).__name__}: {e}")
                await asyncio.sleep(delay)
        # All retries failed — re-check pod status so the error message
        # tells the user whether the pod is actually still alive or got
        # reaped behind RunPod's status endpoint.
        try:
            current = await self._rp().get_pod(handle.pod_id)
            status_hint = (f" pod status now: {current.desired_status}, "
                           f"ssh_endpoint={current.ssh_endpoint}")
        except Exception:
            status_hint = " (couldn't re-query pod status)"
        raise RuntimeError(
            f"SSH never came up after {attempts} attempts: {last_err!r}.{status_hint}"
        )

    # ── public API ────────────────────────────────────────────────────────

    def _gpu_candidates(
        self, purpose: str, gpu_type: str | None,
    ) -> list[str]:
        """Return the ordered list of GPU type ids to offer RunPod.

        An explicit ``gpu_type`` override wins (single-element list). Otherwise
        the purpose's whole preference list is returned so RunPod can place on
        the first available card.
        """
        if gpu_type:
            return [gpu_type]
        val = GPU_BY_PURPOSE.get(purpose)
        if not val:
            raise ValueError(
                f"unknown purpose {purpose!r}; pass gpu_type explicitly")
        return list(val)

    async def provision(
        self,
        *,
        purpose: str = "light",
        gpu_type: str | None = None,
        image: str = DEFAULT_IMAGE,
        volume_gb: int = DEFAULT_VOLUME_GB,
        container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
        data_center_id: str | None = None,
        allow_other_datacenters: bool = True,
        on_log: LogFn = print,
    ) -> PodHandle:
        """Get a running pod for this project.

        Reuses the existing pod if state.json points at one that's still
        ``RUNNING`` with an SSH endpoint. Otherwise creates a fresh pod
        attached to the project's network volume.

        GPU selection offers RunPod the whole ``GPU_BY_PURPOSE[purpose]``
        preference list (unless ``gpu_type`` overrides it); RunPod places on
        the first available card. For a brand-new project with no volume yet,
        and ``allow_other_datacenters=True``, provisioning also tries the
        datacenters in ``DEFAULT_DATACENTERS`` until one has a GPU free —
        the volume is created in whichever DC wins. Once a project has a
        volume, its DC is fixed (volumes can't migrate) and only the GPU
        list is used to dodge a busy card.
        """
        # 1) Try to reuse an existing pod.
        if self._state.pod_id:
            try:
                p = await self._rp().get_pod(self._state.pod_id)
                if p.desired_status == "RUNNING" and p.ssh_endpoint is not None:
                    on_log(f"Reusing pod {p.id} ({p.ssh_endpoint[0]}:{p.ssh_endpoint[1]})")
                    handle = self._handle_from_pod(p)
                    await self._ensure_session(handle, on_log=on_log)
                    return handle
                on_log(f"Stale pod {p.id} status={p.desired_status}, will recreate")
            except RunPodAPIError as e:
                if e.status == 404:
                    on_log(f"Previous pod {self._state.pod_id} is gone")
                else:
                    raise
            # Pod is no longer usable — drop the stale pointer + roll cost.
            state.close_active_pod(self._state)
            self._save_state()

        # 2) Provision a fresh pod, retrying past hosts with broken egress.
        #    RunPod will happily hand back the same bad machine, so each retry
        #    is a fresh roll of the dice rather than a targeted exclusion —
        #    the API has no "not this machineId" parameter.
        gpus = self._gpu_candidates(purpose, gpu_type)
        on_log(f"GPU preference (first available wins): {', '.join(gpus)}")
        bad_machines: list[str] = []

        for attempt in range(1, PROVISION_ATTEMPTS + 1):
            requested_dc = (data_center_id or self._state.data_center_id
                            or DEFAULT_DATACENTER)

            if self._state.volume_id:
                # Existing project — the volume pins the DC (volumes can't move).
                # _ensure_volume may override `requested_dc` to the volume's real
                # region so we don't silently destroy data on a GPU-only override.
                volume_id, dc = await self._ensure_volume(
                    size_gb=volume_gb, data_center_id=requested_dc, on_log=on_log,
                )
                pod = await self._create_pod_in_dc(
                    gpus=gpus, dc=dc, volume_id=volume_id, purpose=purpose,
                    image=image, container_disk_gb=container_disk_gb, on_log=on_log,
                )
                if pod is None:
                    raise RuntimeError(
                        f"None of the {len(gpus)} candidate GPUs were available in "
                        f"{dc} (where this project's volume lives — it can't move "
                        f"to another region). Options: wait and retry, widen the "
                        f"GPU list, or create a pod in the RunPod console and paste "
                        f"its Pod ID into Attach."
                    )
            else:
                # Fresh project — no data yet, so we can chase a free GPU across
                # datacenters and create the volume wherever one lands. On a
                # retry the volume now exists, so we take the branch above.
                dc_list = [requested_dc]
                if allow_other_datacenters:
                    dc_list += [d for d in DEFAULT_DATACENTERS if d != requested_dc]
                pod, volume_id, dc = await self._provision_fresh(
                    gpus=gpus, dc_list=dc_list, volume_gb=volume_gb, purpose=purpose,
                    image=image, container_disk_gb=container_disk_gb, on_log=on_log,
                )
            on_log(f"  pod {pod.id} created (${pod.cost_per_hr}/hr), waiting for SSH...")
            # Persist the pod_id NOW, before any wait that might fail. If
            # wait_until_ready times out we still know which pod is paid-for so
            # the next provision call can reuse / clean it up instead of
            # silently spawning a second pod.
            self._state.pod_id = pod.id
            self._state.pod_started_at = time.time()
            self._state.pod_hourly_rate = pod.cost_per_hr
            self._save_state()

            # NGC's pytorch image (~10 GB) + cold-host scheduling can push first-
            # boot past the old 300s budget. 900s covers the worst case we've seen.
            ready = await self._rp().wait_until_ready(pod.id, timeout=900)
            host, port = ready.ssh_endpoint  # type: ignore[misc]
            machine = str(ready.raw.get("machineId") or "?")
            on_log(f"  pod ready @ {host}:{port} (machine {machine})")

            self._state.ssh_host = host
            self._state.ssh_port = port
            self._save_state()

            handle = self._handle_from_pod(ready)
            if await self._egress_ok(handle, on_log=on_log):
                return handle

            # Bad host. Recycling costs a couple of minutes; keeping it costs
            # a silently broken download. terminate() closes the session,
            # rolls the accrued cost into state, and keeps the volume.
            bad_machines.append(machine)
            on_log(f"  ⚠ pod {pod.id} (machine {machine}) has broken egress — "
                   f"recycling (attempt {attempt}/{PROVISION_ATTEMPTS})")
            await self.terminate(handle, keep_volume=True, on_log=on_log)

        raise RuntimeError(
            f"{PROVISION_ATTEMPTS} pods in a row could not reach the internet "
            f"(machines: {', '.join(bad_machines)}). This project's volume pins "
            f"it to one datacenter, so retrying may keep landing on the same "
            f"bad hosts. Check RunPod status, try again later, or create a pod "
            f"in the console and paste its Pod ID into Attach."
        )

    async def _create_pod_in_dc(
        self,
        *,
        gpus: list[str],
        dc: str,
        volume_id: str | None,
        purpose: str,
        image: str,
        container_disk_gb: int,
        on_log: LogFn,
    ) -> PodInfo | None:
        """Try to create a pod in one datacenter, offering the whole GPU list.

        RunPod places on the first available card. Returns the ``PodInfo`` on
        success, or ``None`` when the API reports no capacity (so the caller
        can try another DC or surface a clear message). A malformed request
        (e.g. a bad GPU id) still raises — those are the user's to fix.
        """
        on_log(f"Requesting a {purpose} pod in {dc} "
               f"(trying {len(gpus)} GPU option(s))…")
        try:
            return await self._rp().create_pod(
                name=f"herb-{self._project}-{purpose}",
                image_name=image,
                gpu_type_ids=gpus,
                container_disk_gb=container_disk_gb,
                volume_mount_path="/workspace",
                network_volume_id=volume_id,
                data_center_ids=[dc],
                ports=("22/tcp",),
            )
        except RunPodAPIError as e:
            # 4xx here is overwhelmingly "no instances available" for the
            # requested GPU/DC combo. Treat as a soft miss; the caller decides
            # whether to try elsewhere. (Bad-GPU-id is caught earlier in
            # create_pod and raised as ValueError, not RunPodAPIError.)
            on_log(f"  ⚠ no capacity in {dc} (HTTP {e.status})")
            return None

    async def _provision_fresh(
        self,
        *,
        gpus: list[str],
        dc_list: list[str],
        volume_gb: int,
        purpose: str,
        image: str,
        container_disk_gb: int,
        on_log: LogFn,
    ) -> tuple[PodInfo, str, str]:
        """Create the project's first volume + pod, chasing a free GPU across
        ``dc_list``. Returns ``(pod, volume_id, data_center_id)``.

        For each DC: create a volume there, try the pod; on capacity miss,
        delete the just-created volume and move on. The volume is only kept
        for the DC that yields a running pod, so we never orphan storage.
        """
        for dc in dc_list:
            try:
                on_log(f"Creating network volume ({volume_gb} GB, {dc})…")
                vol = await self._rp().create_volume(
                    name=f"herb-{self._project}",
                    size_gb=volume_gb,
                    data_center_id=dc,
                )
            except RunPodAPIError as e:
                on_log(f"  ⚠ can't create a volume in {dc} (HTTP {e.status}) — "
                       f"skipping this DC")
                continue
            pod = await self._create_pod_in_dc(
                gpus=gpus, dc=vol.data_center_id, volume_id=vol.id,
                purpose=purpose, image=image,
                container_disk_gb=container_disk_gb, on_log=on_log,
            )
            if pod is not None:
                self._state.volume_id = vol.id
                self._state.data_center_id = vol.data_center_id
                self._save_state()
                return pod, vol.id, vol.data_center_id
            # No GPU here — don't leave an empty volume lying around.
            on_log(f"  cleaning up unused volume in {dc}…")
            try:
                await self._rp().delete_volume(vol.id)
            except RunPodAPIError:
                on_log(f"  (couldn't delete volume {vol.id} — remove it "
                       f"manually if it lingers)")
        raise RuntimeError(
            f"No {purpose} GPU was available in any datacenter tried "
            f"({', '.join(dc_list)}). The whole preference list "
            f"({', '.join(gpus)}) is busy right now. Options: wait a few "
            f"minutes and retry, or create a pod in the RunPod console and "
            f"paste its Pod ID into Attach."
        )

    async def running_step(
        self, handle: PodHandle, *, on_log: LogFn = print,
    ) -> str | None:
        """Name of the detached step currently running on the pod, or None.

        A step is running when its log exists, its ``.rc`` does not (spawn_step
        removes the .rc before launching), and the step process is alive. Only
        one step runs at a time, so the first match wins.
        """
        probe = (
            f'for f in {REMOTE_LOGS}/*.log; do '
            f'[ -e "$f" ] || continue; '
            f's=${{f##*/}}; s=${{s%.log}}; '
            f'[ -f "{REMOTE_LOGS}/$s.rc" ] && continue; '
            f'if pgrep -f "pod_bootstrap.sh ${{s}}\\$" >/dev/null 2>&1; then '
            f'echo "RUNNING_STEP=$s"; break; fi; '
            f'done'
        )
        try:
            session = await self._ensure_session(handle, on_log=on_log)
            _, out = await session.exec_capture(probe)
        except Exception as e:
            on_log(f"  could not check for a running step: {e}")
            return None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("RUNNING_STEP="):
                step = line.partition("=")[2].strip()
                if step:
                    return step
        return None

    async def cancel_step(
        self, handle: PodHandle, step: str, *, on_log: LogFn = print,
    ) -> bool:
        """Kill a detached step running on the pod. Returns True if it died.

        ``spawn_step`` launches under ``setsid``, so the ``__run_detached``
        parent is a session leader and its process group contains the step's
        bash and that bash's python child. Signalling the whole group is the
        only way to stop the work — cancelling the local asyncio task that
        tails the log merely stops watching it.

        Killing the group means ``run_detached`` dies before it writes the
        ``.rc``. We must write one ourselves (130, "terminated by Ctrl-C"):
        ``_follow_step`` blocks in ``while [ ! -f <rc> ]``, and ``exec_streaming``
        reads that channel on a thread, which ``task.cancel()`` cannot
        interrupt. Without an ``.rc`` the follower waits forever and the UI's
        in-flight task never completes, so every later click reports "a cloud
        step is already running". ``spawn_step`` removes the ``.rc`` before it
        launches, so a stale 130 never blocks the next Run.
        """
        rcf = f"{REMOTE_LOGS}/{step}.rc"
        cmd = (
            f'pid=$(pgrep -f "pod_bootstrap.sh __run_detached {step} " | head -1); '
            f'[ -n "$pid" ] || pid=$(pgrep -f "pod_bootstrap.sh {step}\\$" | head -1); '
            f'if [ -z "$pid" ]; then echo "CANCEL=not_running"; exit 0; fi; '
            f'pgid=$(ps -o pgid= -p "$pid" | tr -d " "); '
            f'kill -TERM -"$pgid" 2>/dev/null; '
            f'for i in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; '
            f'if kill -0 "$pid" 2>/dev/null; then '
            f'kill -KILL -"$pgid" 2>/dev/null; verdict=killed; '
            f'else verdict=terminated; fi; '
            # Release the log follower, which is blocked on this file.
            f'echo 130 > {rcf}.tmp && mv {rcf}.tmp {rcf}; '
            f'echo "CANCEL=$verdict"'
        )
        # Session open is inside the try: if SSH is gone we report failure
        # rather than raising out of a UI button handler.
        try:
            session = await self._ensure_session(handle, on_log=on_log)
            _, out = await session.exec_capture(cmd)
        except Exception as e:
            on_log(f"  cancel failed: {e}")
            return False

        verdict = ""
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("CANCEL="):
                verdict = line.partition("=")[2].strip()
        if verdict == "not_running":
            on_log(f"  '{step}' was not running on the pod.")
            return False
        if verdict in ("terminated", "killed"):
            how = "SIGTERM" if verdict == "terminated" else "SIGKILL (it ignored SIGTERM)"
            on_log(f"  ✖ '{step}' stopped on the pod via {how}.")
            self._state.current_step = ""
            self._save_state()
            return True
        on_log(f"  cancel gave an unexpected result: {out.strip()[:200]}")
        return False

    async def follow_running_step(
        self, handle: PodHandle, step: str, *, on_log: LogFn = print,
    ) -> int:
        """Re-attach to an already-running detached step and stream its log
        until it finishes. Returns the step's exit code.

        The log is re-tailed from line 1, so the whole run history replays.
        Safe to call repeatedly — it never spawns anything.
        """
        session = await self._ensure_session(handle, on_log=on_log)
        self._state.current_step = step
        self._save_state()
        rc = await self._follow_step(
            session,
            f"{REMOTE_LOGS}/{step}.log",
            f"{REMOTE_LOGS}/{step}.rc",
            on_log=on_log,
        )
        if rc == 0 and step not in self._state.completed_steps:
            self._state.completed_steps.append(step)
        self._state.current_step = ""
        self._save_state()
        return rc

    async def attach(self, pod_id: str, *, on_log: LogFn = print) -> PodHandle:
        """Connect to a manually-created pod by its RunPod ID.

        Stores the pod ID in state so subsequent ``provision()`` calls reuse it.
        """
        p = await self._rp().get_pod(pod_id)
        if p.desired_status != "RUNNING" or p.ssh_endpoint is None:
            raise RuntimeError(
                f"Pod {pod_id} is not RUNNING or has no SSH endpoint "
                f"(status={p.desired_status})"
            )
        on_log(f"Attaching to pod {p.id} ({p.ssh_endpoint[0]}:{p.ssh_endpoint[1]})")
        handle = self._handle_from_pod(p)
        await self._ensure_session(handle, on_log=on_log)
        self._state.pod_id = pod_id
        self._state.ssh_host = p.ssh_endpoint[0]
        self._state.ssh_port = p.ssh_endpoint[1]
        self._state.pod_hourly_rate = p.cost_per_hr
        # Capture the pod's volume + region so a later provision() doesn't
        # reuse stale state from a different datacenter. The RunPod API has
        # moved these fields around between schema versions, so check a few
        # plausible locations.
        raw = p.raw or {}
        machine = raw.get("machine") or {}
        nv_obj = raw.get("networkVolume") or {}
        vol_id = (
            p.network_volume_id
            or raw.get("networkVolumeId")
            or nv_obj.get("id")
            or raw.get("volumeId")
        )
        if vol_id:
            prev_vol = self._state.volume_id
            self._state.volume_id = vol_id
            if prev_vol and prev_vol != vol_id:
                on_log(f"  volume_id updated: {prev_vol} → {vol_id}")
        else:
            # Help future-us figure out which key holds the volume id.
            keys = sorted(raw.keys())
            on_log(f"  ⚠ no volume id in pod payload — top-level keys: {keys}")
            self._state.volume_id = None
        dc = (
            raw.get("dataCenterId")
            or machine.get("dataCenterId")
            or (machine.get("dataCenter") or {}).get("id")
            or nv_obj.get("dataCenterId")
        )
        if dc:
            prev_dc = self._state.data_center_id
            self._state.data_center_id = dc
            if prev_dc and prev_dc != dc:
                on_log(f"  data_center_id updated: {prev_dc} → {dc}")
        else:
            self._state.data_center_id = None
        self._save_state()

        # A step spawned before the UI restarted is still running (setsid +
        # nohup). Say so — the pod looks idle otherwise. Callers should then
        # follow_running_step() rather than sync_code(), which would SFTP over
        # a pod_bootstrap.sh that a live bash is still reading by offset.
        step = await self.running_step(handle, on_log=on_log)
        if step:
            on_log(f"↻ '{step}' is still running on this pod.")
        return handle

    async def ensure_pod_template(
        self,
        *,
        name: str = "herbarium-prebaked",
        image: str = DEFAULT_IMAGE,
        container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
        on_log: LogFn = print,
    ) -> dict[str, Any]:
        """Idempotently create a RunPod template pinned to the prebaked image.

        Ergonomic helper for the manual-allocation path: when every GPU in the
        fallback list is busy and you make a pod in the console by hand, picking
        this saved template selects the prebaked image (and /workspace mount,
        port 22) in one click — no GHCR string to paste. Safe to call repeatedly;
        if a template with ``name`` already exists it's returned as-is rather
        than duplicated.
        """
        rp = self._rp()
        try:
            existing = await rp.list_templates()
        except RunPodAPIError as e:
            on_log(f"  ⚠ couldn't list templates (HTTP {e.status}) — trying create anyway")
            existing = []
        for t in existing:
            if t.get("name") == name:
                on_log(f"Template '{name}' already exists (id {t.get('id')}) — reusing.")
                return t
        on_log(f"Creating RunPod template '{name}' → {image}")
        t = await rp.create_template(
            name=name,
            image_name=image,
            container_disk_gb=container_disk_gb,
            readme=(
                "Herbarium pipeline prebaked env. Select this when allocating a "
                "pod by hand, then attach a network volume mounted at /workspace."
            ),
        )
        on_log(f"✓ Template created (id {t.get('id')}). It'll appear in the "
               f"console's template picker when you deploy a pod.")
        return t

    async def sync_code(self, handle: PodHandle, *, on_log: LogFn = print) -> None:
        """Push local pipeline scripts to ``/workspace/Pipeline`` on the pod.

        Run before ``setup`` (or ``download``/``prep``/etc., which all expect
        the script to be present). This is what makes the pod execute the
        code the user is actually running locally rather than whatever's
        on origin/main.
        """
        session = await self._ensure_session(handle, on_log=on_log)
        await session.exec_capture(f"mkdir -p {REMOTE_REPO} {REMOTE_DATA}")
        files = sorted({
            f for pat in SYNC_FILE_PATTERNS
            for f in self._local_pipeline.glob(pat)
            if f.is_file()
        })
        # space/push_model.py is standalone (no relative imports) but lives in
        # a subdir the top-level glob misses. Push it flat so `publish` can run
        # the user's current version, not whatever's on origin/main.
        extra = self._local_pipeline / "space" / "push_model.py"
        files = list(files)
        if extra.is_file():
            files.append(extra)
        on_log(f"Syncing {len(files)} files → {REMOTE_REPO}")
        for f in files:
            await session.sftp_put(f, f"{REMOTE_REPO}/{f.name}")
        await session.exec_capture(
            f"chmod +x {REMOTE_REPO}/pod_bootstrap.sh && "
            f"sed -i 's/\\r//' {REMOTE_REPO}/pod_bootstrap.sh"
        )
        await self.push_wandb_key(handle, on_log=on_log)
        await self.push_runpod_key(handle, on_log=on_log)
        await self.push_r2_config(handle, on_log=on_log)

    async def push_runpod_key(
        self, handle: PodHandle, *, on_log: LogFn = print,
    ) -> bool:
        """Write the RunPod API key to ``/workspace/.runpod_key`` (chmod 600).

        The idle watchdog needs it to DELETE its own pod: RunPod has no
        server-side idle timeout for pods (``idleTimeout`` exists only on
        serverless endpoints), and our prebaked CUDA image carries no
        ``runpodctl``, so the REST API is the only kill path available from
        inside the container.

        Note this is an account-scoped key sitting on rented hardware — the
        same trust already extended to the R2 credentials and HF write token,
        but the blast radius is larger (it can create and destroy pods). The
        alternative is a pod that bills until a human notices, which is what
        happened before this existed.
        """
        key = secrets.get_runpod_api_key()
        if not key:
            on_log("No RunPod key in keyring — watchdog cannot self-terminate the pod!")
            return False
        session = await self._ensure_session(handle, on_log=on_log)
        await session.sftp_put_bytes(key.encode("utf-8"), "/workspace/.runpod_key")
        await session.exec_capture("chmod 600 /workspace/.runpod_key")
        on_log("Pushed RunPod key → /workspace/.runpod_key (chmod 600, for the idle watchdog)")
        return True

    async def push_wandb_key(
        self, handle: PodHandle, *, on_log: LogFn = print,
    ) -> bool:
        """Write the wandb API key from the OS keyring to ``/workspace/.wandb_key``.

        Returns True if pushed, False if no key is configured (in which case
        pod_bootstrap.sh's ``wandb login`` step will be skipped and training
        will fall back to CSV logging).
        """
        key = secrets.get_wandb_api_key()
        if not key:
            on_log("No wandb key in keyring — skipping push (set one in the Cloud tab to enable wandb).")
            return False
        session = await self._ensure_session(handle, on_log=on_log)
        await session.sftp_put_bytes(key.encode("utf-8"), "/workspace/.wandb_key")
        on_log("Pushed wandb key → /workspace/.wandb_key (chmod 600)")
        return True

    async def push_hf_token(
        self, handle: PodHandle, *, on_log: LogFn = print,
    ) -> bool:
        """Write the Hugging Face write token from the keyring to
        ``/workspace/.hf_token`` (chmod 600). push_model.py reads it from
        there for unattended pod-side publishing.

        Returns True if pushed, False if no token is configured.
        """
        token = secrets.get_hf_token()
        if not token:
            on_log("No Hugging Face token in keyring — skipping push "
                   "(set one in the Cloud tab to enable publishing to the Hub).")
            return False
        session = await self._ensure_session(handle, on_log=on_log)
        await session.sftp_put_bytes(token.encode("utf-8"), "/workspace/.hf_token")
        await session.exec_capture("chmod 600 /workspace/.hf_token")
        on_log("Pushed HF token → /workspace/.hf_token (chmod 600)")
        return True

    async def publish_model(
        self,
        handle: PodHandle,
        *,
        family: str,
        hf_user: str,
        region: str = "",
        label_level: str = "",
        repo: str = "",
        display_name: str = "",
        select_by: str = "",
        on_log: LogFn = print,
    ) -> int:
        """Publish the pod's best checkpoint to the Hugging Face Hub.

        Runs the ``publish`` bootstrap step, which invokes push_model.py
        against ``/workspace/data/checkpoints`` (auto-picks the best
        checkpoint by accuracy and reads the embedded nameslist). The HF
        token is staged on the pod by ``run_step`` itself. Either ``repo``
        or (``hf_user`` + ``family``) must be set.
        Returns the step exit code (0 = success).
        """
        if not secrets.get_hf_token():
            on_log("No Hugging Face token in keyring — set one before publishing.")
            return 1
        env = {
            "FAMILY": family,
            "HF_USER": hf_user,
            "REGION": region,
            "LABEL_LEVEL": label_level,
            "HF_REPO": repo,  # not REPO — pod_bootstrap uses REPO for the code dir
            "DISPLAY_NAME": display_name,
            "SELECT_BY": select_by,  # "" → push_model default (accuracy)
        }
        return await self.run_step(handle, "publish", env=env, on_log=on_log)

    async def push_r2_config(
        self, handle: PodHandle, *, on_log: LogFn = print,
    ) -> bool:
        """Write an rclone config for Cloudflare R2 to the pod.

        Reads R2 credentials from the OS keyring and renders an
        ``rclone.conf`` at ``/workspace/.config/rclone/rclone.conf`` (the
        path bootstrap exports as ``RCLONE_CONFIG``). The remote is named
        ``r2`` — the same name the bootstrap's ``CACHE_REMOTE`` and
        backup paths assume.

        Returns True if pushed, False if no creds are configured (cache
        push/pull and backup/restore will then no-op gracefully).
        """
        creds = secrets.get_r2_credentials()
        if not creds:
            on_log("No R2 credentials in keyring — skipping rclone config push "
                   "(set them in the Cloud tab to enable shared cache + R2 backup).")
            return False
        # provider=Other rather than provider=Cloudflare — the latter is a
        # newer rclone alias that some pre-v1.62 builds don't recognize
        # (NOTICE "s3 provider 'Cloudflare' not known"). "Other" is the
        # generic S3-compatible provider; behaves the same for R2.
        # no_check_bucket avoids an extra HEAD per upload — R2 returns 200
        # on bucket existence checks slowly enough that batch uploads
        # noticeably slow without it.
        conf = (
            "[r2]\n"
            "type = s3\n"
            "provider = Other\n"
            f"access_key_id = {creds.access_key_id}\n"
            f"secret_access_key = {creds.secret_access_key}\n"
            f"endpoint = {creds.endpoint}\n"
            "acl = private\n"
            "no_check_bucket = true\n"
        )
        session = await self._ensure_session(handle, on_log=on_log)
        await session.exec_capture("mkdir -p /workspace/.config/rclone")
        await session.sftp_put_bytes(
            conf.encode("utf-8"),
            "/workspace/.config/rclone/rclone.conf",
        )
        await session.exec_capture("chmod 600 /workspace/.config/rclone/rclone.conf")
        on_log(f"Pushed rclone.conf → /workspace/.config/rclone/ "
               f"(R2 endpoint {creds.endpoint})")
        return True

    async def upload_dwca(
        self,
        handle: PodHandle,
        local_zip: str | Path,
        *,
        on_log: LogFn = print,
        on_progress: ProgressFn | None = None,
    ) -> bool:
        """Upload the DwC-A zip to the pod's volume.

        Skips the upload when the local SHA-256 matches the file already on
        the volume (fast restarts of the same project). Returns True if a
        transfer happened, False if it was skipped.
        """
        local_zip = Path(local_zip)
        if not local_zip.is_file():
            raise FileNotFoundError(local_zip)

        h = hashlib.sha256()
        with local_zip.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        local_sha = h.hexdigest()

        session = await self._ensure_session(handle, on_log=on_log)
        remote_sha = await session.remote_sha256(REMOTE_DWCA)
        if remote_sha == local_sha:
            on_log(f"DwC-A unchanged on volume (sha256 {local_sha[:16]}…), skipping")
            self._state.dwca_sha256 = local_sha
            self._save_state()
            return False

        size = local_zip.stat().st_size
        on_log(f"Uploading {local_zip.name} ({size:,} bytes)…")
        await session.sftp_put(local_zip, REMOTE_DWCA, on_progress=on_progress)
        self._state.dwca_sha256 = local_sha
        self._save_state()
        return True

    async def upload_file(
        self,
        handle: PodHandle,
        local: str | Path,
        remote: str,
        *,
        on_log: LogFn = print,
        on_progress: ProgressFn | None = None,
    ) -> None:
        """Upload a local file to an arbitrary path on the pod via SFTP."""
        local_path = Path(local)
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        size = local_path.stat().st_size
        session = await self._ensure_session(handle, on_log=on_log)
        # Ensure parent directory exists on the pod
        remote_parent = "/".join(remote.rstrip("/").split("/")[:-1]) or "/"
        await session.exec_capture(f"mkdir -p {remote_parent}")
        on_log(f"Uploading {local_path.name} → {remote} ({size:,} bytes)")
        await session.sftp_put(local_path, remote, on_progress=on_progress)

    async def run_step(
        self,
        handle: PodHandle,
        step: str,
        *,
        env: dict[str, str] | None = None,
        on_log: LogFn = print,
    ) -> int:
        """Run a ``pod_bootstrap.sh`` step, streaming logs back live.

        Valid steps: setup, download, prep, train, identify, ood, backup, restore.
        ``env`` (e.g. ``{"LIMIT": "500", "MAX_PER_SP": "30"}``) is exported
        for the bash invocation so the bootstrap script's per-step overrides
        kick in. Returns the script's exit code (0 = success).
        """
        valid = {"setup", "download", "prep", "train", "identify", "ood",
                 "backup", "restore", "publish"}
        if step not in valid:
            raise ValueError(f"unknown step {step!r}; expected one of {sorted(valid)}")

        session = await self._ensure_session(handle, on_log=on_log)
        self._state.current_step = step
        self._save_state()
        # Auto-inject GBIF credentials for the download step so --families works on-pod.
        if step == "download":
            gbif = secrets.get_gbif_credentials()
            if gbif:
                env = {"GBIF_USER": gbif.username, "GBIF_PASSWORD": gbif.password, **(env or {})}
        # Publish needs the HF write token staged on the pod (sftp, like the wandb key).
        if step == "publish":
            await self.push_hf_token(handle, on_log=on_log)
        env_prefix = ""
        if env:
            env_prefix = " ".join(
                f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v
            )
            if env_prefix:
                env_prefix += " "

        log = f"{REMOTE_LOGS}/{step}.log"
        rcf = f"{REMOTE_LOGS}/{step}.rc"
        # Steps run DETACHED on the pod (spawn_step: setsid + nohup, logging to
        # <step>.log, exit code to <step>.rc). This survives a dropped ssh
        # connection, and the pod-side heartbeat keeps the idle watchdog from
        # killing a long step. We then just tail the log until .rc appears.
        # On reconnect, if the step is still running we re-attach to its log
        # instead of launching a duplicate.
        _, chk = await session.exec_capture(
            f"if [ ! -f {shlex.quote(rcf)} ] && "
            f"pgrep -f 'pod_bootstrap.sh {step}$' >/dev/null 2>&1; "
            f"then echo RUNNING; else echo IDLE; fi"
        )
        if "RUNNING" in chk:
            on_log(f"↻ '{step}' is already running on the pod — re-attaching to its log.")
        else:
            spawn = f"{env_prefix}bash {REMOTE_REPO}/pod_bootstrap.sh spawn {step}"
            on_log(f"$ pod_bootstrap.sh {step}  (detached — survives disconnect)")
            _, spawn_out = await session.exec_capture(spawn)
            if spawn_out.strip():
                on_log(spawn_out.strip())

        rc = await self._follow_step(session, log, rcf, on_log=on_log)
        if rc == 0:
            if step not in self._state.completed_steps:
                self._state.completed_steps.append(step)
        self._state.current_step = ""
        self._save_state()
        return rc

    async def _follow_step(
        self, session: PodSession, log: str, rcf: str, *, on_log: LogFn,
    ) -> int:
        """Stream a detached step's log until it writes its .rc file, then
        return the recorded exit code. Re-runnable: on reconnect it re-tails
        from the top so the whole run history streams again."""
        follow = (
            f"touch {shlex.quote(log)}; "
            f"tail -n +1 -F {shlex.quote(log)} & TP=$!; "
            f"while [ ! -f {shlex.quote(rcf)} ]; do sleep 2; done; "
            f"sleep 1; kill $TP 2>/dev/null; true"
        )
        await session.exec_streaming(follow, on_log=on_log)
        _, out = await session.exec_capture(
            f"cat {shlex.quote(rcf)} 2>/dev/null || echo 999")
        for tok in reversed(out.split()):
            try:
                return int(tok)
            except ValueError:
                continue
        return 999

    async def download_results(
        self,
        handle: PodHandle,
        local_dir: str | Path,
        *,
        on_log: LogFn = print,
        on_progress: ProgressFn | None = None,
        ckpt_filter: str = "latest",
    ) -> list[Path]:
        """Pull checkpoints, the names list, predictions, and specsin back
        to ``local_dir``.

        ``ckpt_filter`` selects which checkpoints to include:
          - ``"latest"`` : the single most recent ``.ckpt`` by remote mtime
            (e.g. ``last-v3.ckpt``). Cheapest — usually all the local
            Identify run needs.
          - ``"best+latest"`` : the lowest-valid_loss checkpoint (parsed
            from the filename) plus the most recent one. ~2× the bytes.
          - ``"all"`` : every ``.ckpt`` under ``checkpoints/`` (legacy
            behaviour). Multi-GB; only useful when comparing checkpoints.

        Quietly skips files that don't exist on the pod (e.g. ``predictions.csv``
        before the identify step has run). Returns the list of files written.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        session = await self._ensure_session(handle, on_log=on_log)

        # Fixed-path artefacts.
        # nameslist.json lives at /workspace/data/nameslist.json (Lightning
        # writes it next to output_dir, not under checkpoints/) — easy bug
        # to introduce because the README's project layout puts it under
        # runs/ when training locally.
        wishlist: list[tuple[str, Path]] = [
            (f"{REMOTE_DATA}/nameslist.json",            local_dir / "nameslist.json"),
            (f"{REMOTE_DATA}/predictions/predictions.csv", local_dir / "predictions.csv"),
            # AUM ranking sidecar (aum_candidates.py --top 0). Sits next to
            # predictions.csv so the Review tab merges it for the "possible
            # mislabels" view; quietly skipped on runs that didn't produce one.
            (f"{REMOTE_DATA}/predictions/aum.csv",       local_dir / "aum.csv"),
            # Taxa dropped as too sparse to train — the Review tab lists them
            # (any specimen of these is forced to the nearest trained class).
            (f"{REMOTE_DATA}/predictions/excluded_species.json",
             local_dir / "excluded_species.json"),
            (f"{REMOTE_DATA}/predictions/excluded_species.csv",
             local_dir / "excluded_species.csv"),
            (f"{REMOTE_DATA}/specsin.csv",               local_dir / "specsin.csv"),
        ]

        # Discover checkpoints AND their mtimes in one shell call so we can
        # filter without paying per-file SFTP listing. `stat -c '%Y\t%n'`
        # prints "<unix_mtime>\t<path>" per line — POSIX coreutils format.
        rc, ls_out = await session.exec_capture(
            f"stat -c '%Y\t%n' {REMOTE_DATA}/checkpoints/*.ckpt 2>/dev/null"
        )
        ckpts: list[tuple[float, str]] = []
        if rc == 0 and ls_out.strip():
            for line in ls_out.strip().splitlines():
                if "\t" not in line:
                    continue
                m_str, path = line.split("\t", 1)
                try:
                    ckpts.append((float(m_str), path.strip()))
                except ValueError:
                    continue
        if not ckpts:
            on_log(f"  no checkpoints under {REMOTE_DATA}/checkpoints/")
        else:
            chosen = _select_ckpts(ckpts, ckpt_filter, on_log=on_log)
            for remote in chosen:
                wishlist.append((remote, local_dir / Path(remote).name))

        written: list[Path] = []
        for remote, local in wishlist:
            stat = await session.remote_stat(remote)
            if stat is None:
                on_log(f"  skip {remote} (not present)")
                continue
            r_size, r_mtime = stat
            if local.exists():
                l = local.stat()
                # Skip only when local size matches AND local was modified
                # at or after remote. paramiko's sftp_get does NOT preserve
                # remote mtime, so local mtime is the wall-clock time of
                # the previous download — which is reliably ≥ the remote
                # mtime that existed at that time, but < any newer remote
                # mtime produced by a re-train.
                if l.st_size == r_size and l.st_mtime >= r_mtime - 1:
                    on_log(f"  up-to-date {local.name} ({r_size:,} bytes)")
                    written.append(local)
                    continue
                on_log(f"  stale {local.name} (local {l.st_size:,}b @"
                       f" {int(l.st_mtime)}, remote {r_size:,}b @"
                       f" {int(r_mtime)}) — re-downloading")
            on_log(f"  download {remote} → {local} ({r_size:,} bytes)")
            await session.sftp_get(remote, local, on_progress=on_progress)
            written.append(local)
        return written

    async def download_images(
        self,
        handle: PodHandle,
        local_dir: str | Path,
        *,
        on_log: LogFn = print,
        on_progress: ProgressFn | None = None,
        images_dirname: str | None = None,
    ) -> Path:
        """Pull ``/workspace/data/<images_dirname>/`` to
        ``local_dir/<images_dirname>/``.

        ``images_dirname`` defaults to whichever of these exists on the pod
        (in priority order): ``images`` (current unified layout), then
        ``images_1024`` (legacy three-dir layout). Pass an explicit name to
        override.

        Bundles the directory into an uncompressed tar on the pod first so the
        transfer is one large sequential SFTP read instead of ~15k file
        round-trips. Images are JPEG already; gzip would burn CPU for nothing.
        Returns the local image directory path.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        session = await self._ensure_session(handle, on_log=on_log)

        # Resolve the directory name to use.
        if images_dirname:
            candidates = [images_dirname]
        else:
            candidates = ["images", "images_1024"]
        chosen: str | None = None
        for name in candidates:
            if await session.remote_exists(f"{REMOTE_DATA}/{name}"):
                chosen = name
                break
        if chosen is None:
            tried = ", ".join(f"{REMOTE_DATA}/{n}" for n in candidates)
            raise FileNotFoundError(f"No images dir on pod (tried {tried})")

        remote_dir = f"{REMOTE_DATA}/{chosen}"
        remote_tar = f"{REMOTE_DATA}/{chosen}.tar"
        # Reuse an existing tar (left by a prior download_images / backup)
        # if no file in the source dir is newer than the tar — saves
        # ~5–15 min of re-taring on MooseFS for the common case where
        # the user pulls images, runs backup, pulls again.
        rc_chk, out_chk = await session.exec_capture(
            f"[ -f {remote_tar} ] && "
            f"[ -z \"$(find {remote_dir} -newer {remote_tar} -print -quit 2>/dev/null)\" ] "
            f"&& echo reuse || echo rebuild"
        )
        if "reuse" in out_chk:
            on_log(f"Reusing existing {remote_tar.rsplit('/', 1)[-1]}")
        else:
            on_log(f"Bundling {remote_dir} into tar...")
            rc, out = await session.exec_capture(
                f"tar cf {remote_tar} -C {REMOTE_DATA} {chosen}"
            )
            if rc != 0:
                raise RuntimeError(f"remote tar failed (rc={rc}): {out.strip()}")

        local_tar = local_dir / f"{chosen}.tar"
        on_log(f"Downloading {local_tar.name}...")
        try:
            await session.sftp_get(remote_tar, local_tar, on_progress=on_progress)
            on_log("Extracting locally...")
            import tarfile
            with tarfile.open(local_tar) as tf:
                tf.extractall(local_dir)
        finally:
            local_tar.unlink(missing_ok=True)
            # Leave the remote tar in place so a follow-up `backup` can reuse
            # it instead of re-tarring (~5–15 GB, slow on MooseFS). backup()
            # cleans up after upload.

        out_dir = local_dir / chosen
        on_log(f"Done → {out_dir}")
        return out_dir

    async def terminate(
        self,
        handle: PodHandle,
        *,
        keep_volume: bool = True,
        on_log: LogFn = print,
    ) -> None:
        """Terminate the pod. By default the network volume is preserved.

        Pass ``keep_volume=False`` only when the user has explicitly asked
        to delete the project — losing a volume loses all downloaded images
        and trained checkpoints not previously copied off-pod.
        """
        sess = self._sessions.pop(handle.pod_id, None)
        if sess is not None:
            await sess.aclose()

        try:
            await self._rp().terminate_pod(handle.pod_id)
            on_log(f"Terminated pod {handle.pod_id}")
        except RunPodAPIError as e:
            if e.status != 404:
                raise
            on_log(f"Pod {handle.pod_id} already gone")

        state.close_active_pod(self._state)

        if not keep_volume and self._state.volume_id:
            on_log(f"Deleting volume {self._state.volume_id}")
            try:
                await self._rp().delete_volume(self._state.volume_id)
            except RunPodAPIError as e:
                if e.status != 404:
                    raise
            self._state.volume_id = None
            self._state.data_center_id = None

        self._save_state()
