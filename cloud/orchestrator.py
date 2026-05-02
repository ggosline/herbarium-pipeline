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
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import paramiko

from . import secrets, state
from .pod_session import PodSession
from .runpod_client import PodInfo, RunPodAPIError, RunPodClient

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]

# ── defaults ─────────────────────────────────────────────────────────────

# RunPod's own pytorch image on Docker Hub. Pulls are fast (RunPod hosts
# cache it aggressively — it's their most popular base) and crucially it
# ships sshd preconfigured for RunPod's SSH-over-public-IP flow.
#
# Trade-off: Docker Hub anonymous pulls are rate-limited (~100 / 6h per
# host IP). When RunPod's host IP gets unlucky you can hit
# "toomanyrequests" at create_pod time. If that becomes chronic, the
# durable fix is wiring containerRegistryAuthId through create_pod —
# but that adds a third credential for naive users so we accept the
# rare failure for now.
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_DATACENTER = "EUR-IS-1"
DEFAULT_VOLUME_GB = 80
DEFAULT_CONTAINER_DISK_GB = 40

# Light tier handles download / prep / identify on a cheap reliable GPU.
# Train tier is what the user actually picks for the long run.
GPU_BY_PURPOSE: dict[str, str] = {
    "light": "NVIDIA L4",
    "train": "NVIDIA GeForce RTX 4090",
}

# Pod-side layout, mirrored in pod_bootstrap.sh.
REMOTE_REPO = "/workspace/Pipeline"
REMOTE_DATA = "/workspace/data"
REMOTE_DWCA = f"{REMOTE_DATA}/gbif.zip"

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

    async def _ensure_volume(
        self, *, size_gb: int, data_center_id: str, on_log: LogFn,
    ) -> str:
        """Return the project's network volume id, creating one if absent."""
        if self._state.volume_id:
            return self._state.volume_id
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
        return vol.id

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

    async def provision(
        self,
        *,
        purpose: str = "light",
        gpu_type: str | None = None,
        image: str = DEFAULT_IMAGE,
        volume_gb: int = DEFAULT_VOLUME_GB,
        container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
        data_center_id: str | None = None,
        on_log: LogFn = print,
    ) -> PodHandle:
        """Get a running pod for this project.

        Reuses the existing pod if state.json points at one that's still
        ``RUNNING`` with an SSH endpoint. Otherwise creates a fresh pod
        attached to the project's network volume.
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

        # 2) Provision a fresh pod.
        gpu = gpu_type or GPU_BY_PURPOSE.get(purpose)
        if not gpu:
            raise ValueError(f"unknown purpose {purpose!r}; pass gpu_type explicitly")
        dc = data_center_id or self._state.data_center_id or DEFAULT_DATACENTER
        volume_id = await self._ensure_volume(
            size_gb=volume_gb, data_center_id=dc, on_log=on_log,
        )

        on_log(f"Creating {gpu} pod in {dc}...")
        pod = await self._rp().create_pod(
            name=f"herb-{self._project}-{purpose}",
            image_name=image,
            gpu_type_ids=[gpu],
            container_disk_gb=container_disk_gb,
            volume_mount_path="/workspace",
            network_volume_id=volume_id,
            data_center_ids=[dc],
            ports=("22/tcp",),
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
        on_log(f"  pod ready @ {host}:{port}")

        self._state.ssh_host = host
        self._state.ssh_port = port
        self._save_state()

        handle = self._handle_from_pod(ready)
        await self._ensure_session(handle, on_log=on_log)
        return handle

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
        on_log(f"Syncing {len(files)} files → {REMOTE_REPO}")
        for f in files:
            await session.sftp_put(f, f"{REMOTE_REPO}/{f.name}")
        await session.exec_capture(f"chmod +x {REMOTE_REPO}/pod_bootstrap.sh")
        await self.push_wandb_key(handle, on_log=on_log)
        await self.push_r2_config(handle, on_log=on_log)

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

    async def run_step(
        self,
        handle: PodHandle,
        step: str,
        *,
        env: dict[str, str] | None = None,
        on_log: LogFn = print,
    ) -> int:
        """Run a ``pod_bootstrap.sh`` step, streaming logs back live.

        Valid steps: setup, download, prep, train, identify, backup, restore.
        ``env`` (e.g. ``{"LIMIT": "500", "MAX_PER_SP": "30"}``) is exported
        for the bash invocation so the bootstrap script's per-step overrides
        kick in. Returns the script's exit code (0 = success).
        """
        valid = {"setup", "download", "prep", "train", "identify", "backup", "restore"}
        if step not in valid:
            raise ValueError(f"unknown step {step!r}; expected one of {sorted(valid)}")

        session = await self._ensure_session(handle, on_log=on_log)
        self._state.current_step = step
        self._save_state()
        env_prefix = ""
        if env:
            import shlex as _shlex
            env_prefix = " ".join(
                f"{k}={_shlex.quote(str(v))}" for k, v in env.items() if v
            )
            if env_prefix:
                env_prefix += " "
        cmd = f"{env_prefix}bash {REMOTE_REPO}/pod_bootstrap.sh {step}"
        on_log(f"$ {cmd}")
        rc = await session.exec_streaming(cmd, on_log=on_log)
        if rc == 0:
            if step not in self._state.completed_steps:
                self._state.completed_steps.append(step)
        self._state.current_step = ""
        self._save_state()
        return rc

    async def download_results(
        self,
        handle: PodHandle,
        local_dir: str | Path,
        *,
        on_log: LogFn = print,
        on_progress: ProgressFn | None = None,
    ) -> list[Path]:
        """Pull every checkpoint, the names list, predictions, and specsin
        back to ``local_dir``.

        Quietly skips files that don't exist on the pod (e.g. ``predictions.csv``
        before the identify step has run). Returns the list of files written.

        Checkpoints are pulled by globbing ``checkpoints/*.ckpt`` rather than
        a hardcoded list because Lightning emits one ``last.ckpt`` plus per-
        stage best-of-run files like ``epoch=17-valid_loss=0.40.ckpt`` and
        ``cd-epoch=03-valid_loss=0.39.ckpt``, and the user usually wants the
        best one (lowest valid_loss) as well as last.
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
            (f"{REMOTE_DATA}/specsin.csv",               local_dir / "specsin.csv"),
        ]

        # Discover every checkpoint file rather than hardcoding names.
        # Run as a single shell command so we don't pay per-file SFTP listing.
        rc, ls_out = await session.exec_capture(
            f"ls -1 {REMOTE_DATA}/checkpoints/*.ckpt 2>/dev/null"
        )
        if rc == 0 and ls_out.strip():
            for line in ls_out.strip().splitlines():
                remote = line.strip()
                if remote:
                    wishlist.append((remote, local_dir / Path(remote).name))
        else:
            on_log(f"  no checkpoints under {REMOTE_DATA}/checkpoints/")

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
    ) -> Path:
        """Pull ``/workspace/data/images_1024/`` to ``local_dir/images_1024/``.

        Bundles the directory into an uncompressed tar on the pod first so the
        transfer is one large sequential SFTP read instead of ~15k file
        round-trips. Images are JPEG already; gzip would burn CPU for nothing.
        Returns the local image directory path.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        session = await self._ensure_session(handle, on_log=on_log)

        remote_dir = f"{REMOTE_DATA}/images_1024"
        if not await session.remote_exists(remote_dir):
            raise FileNotFoundError(f"{remote_dir} not present on pod")

        remote_tar = f"{REMOTE_DATA}/images_1024.tar"
        on_log(f"Bundling {remote_dir} into tar...")
        rc, out = await session.exec_capture(
            f"tar cf {remote_tar} -C {REMOTE_DATA} images_1024"
        )
        if rc != 0:
            raise RuntimeError(f"remote tar failed (rc={rc}): {out.strip()}")

        local_tar = local_dir / "images_1024.tar"
        on_log(f"Downloading {local_tar.name}...")
        try:
            await session.sftp_get(remote_tar, local_tar, on_progress=on_progress)
            on_log("Extracting locally...")
            import tarfile
            with tarfile.open(local_tar) as tf:
                tf.extractall(local_dir)
        finally:
            local_tar.unlink(missing_ok=True)
            await session.exec_capture(f"rm -f {remote_tar}")

        out_dir = local_dir / "images_1024"
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
