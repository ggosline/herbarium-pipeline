"""SSH/SFTP session against a running RunPod pod.

Bridges blocking paramiko I/O to asyncio: ``exec_streaming`` runs a command
on the pod and emits each stdout/stderr line via callback as it arrives,
which is exactly what the webui's log panel wants. ``sftp_put``/``sftp_get``
move files (DwC-A up, checkpoints/predictions back). ``remote_sha256`` lets
the orchestrator skip re-uploading an unchanged DwC-A.

Auth: RunPod auto-injects whatever public keys are registered on the user's
account into every pod they create. Locally we hand paramiko the user's
default private keys (``~/.ssh/id_ed25519``, ``id_rsa``, ``id_ecdsa``) plus
ssh-agent if it's running. An explicit ``key_filename`` overrides discovery.

Host-key policy: RunPod's SSH endpoints are ephemeral (new host:port and
new host key per pod), so we use ``AutoAddPolicy``. The integrity guarantee
comes from the RunPod REST API (TLS) telling us which host:port to dial,
not from a known_hosts entry.
"""

from __future__ import annotations

import asyncio
import shlex
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import paramiko

DEFAULT_USERNAME = "root"
DEFAULT_KEY_NAMES = ("id_ed25519", "id_rsa", "id_ecdsa")


def _discover_key_files(explicit: str | Path | None) -> list[str]:
    """Return private-key paths to hand paramiko, in priority order.

    Explicit path wins. Otherwise scan ``~/.ssh`` for the standard names.
    Returning an empty list is fine — paramiko will then rely solely on
    ssh-agent (``allow_agent=True``, ``look_for_keys=True``).
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"SSH key not found: {p}")
        return [str(p)]
    home_ssh = Path.home() / ".ssh"
    return [str(home_ssh / n) for n in DEFAULT_KEY_NAMES if (home_ssh / n).exists()]


class PodSession:
    """Async SSH/SFTP session. Use as an async context manager.

        async with PodSession(host, port) as s:
            await s.exec_streaming("bash pod_bootstrap.sh download", on_log=print)
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        username: str = DEFAULT_USERNAME,
        key_filename: str | Path | None = None,
        connect_timeout: float = 30.0,
    ):
        self.host = host
        self.port = int(port)
        self.username = username
        self.key_filename = key_filename
        self.connect_timeout = connect_timeout
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None
        # paramiko's SFTPClient is NOT thread-safe — concurrent put/get
        # from different asyncio tasks (e.g. sync_code racing with an
        # aux-slot download_images) corrupt the SSH transport state and
        # raise "Garbage packet received". Serialize SFTP work behind a
        # lock so the channel stays coherent.
        self._sftp_lock: asyncio.Lock | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def __aenter__(self) -> "PodSession":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def connect(self) -> None:
        """Open the SSH transport. Raises on auth failure or unreachable host."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        keys = _discover_key_files(self.key_filename)
        # When the caller pinned an explicit key path we use *only* that key —
        # disabling agent + key auto-discovery prevents paramiko from falling
        # back to a passphrased default key and prompting interactively.
        explicit = self.key_filename is not None
        await asyncio.to_thread(
            client.connect,
            hostname=self.host,
            port=self.port,
            username=self.username,
            key_filename=keys or None,
            look_for_keys=not explicit,
            allow_agent=not explicit,
            timeout=self.connect_timeout,
            banner_timeout=self.connect_timeout,
            auth_timeout=self.connect_timeout,
        )
        # Send a keepalive every 30s so RunPod's NAT / idle timers don't
        # silently drop long-running channels (training, large rsync,
        # tarball pulls). Without this the webui log goes quiet with no
        # error and there's no way to tell what's running.
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        self._client = client

    async def aclose(self) -> None:
        if self._sftp is not None:
            try:
                await asyncio.to_thread(self._sftp.close)
            except Exception:
                pass
            self._sftp = None
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
            self._client = None

    def _require_client(self) -> paramiko.SSHClient:
        if self._client is None:
            raise RuntimeError("PodSession is not connected; call connect() first")
        return self._client

    def is_alive(self) -> bool:
        """Cheap liveness check on the underlying transport.

        Returns False if the SSH transport was never opened or is no
        longer active (e.g. the server-side ssh closed the connection
        after a long idle period or a network blip). Callers should
        evict-and-rebuild on False rather than reusing the session.
        """
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    # ── command execution ─────────────────────────────────────────────────

    async def exec_streaming(
        self,
        cmd: str,
        on_log: Callable[[str], None],
        *,
        combine_stderr: bool = True,
    ) -> int:
        """Run ``cmd`` on the pod, calling ``on_log(line)`` per output line.

        Returns the command's exit status. On asyncio cancellation the SSH
        channel is closed which kills the remote process tree.

        Implementation: paramiko's Channel is blocking, so a worker thread
        reads bytes, splits on newlines, and pushes lines into an
        asyncio.Queue via ``call_soon_threadsafe``. The main coroutine
        drains the queue and invokes ``on_log``.
        """
        client = self._require_client()
        transport = client.get_transport()
        if transport is None:
            raise RuntimeError("SSH transport is gone (connection closed)")

        chan = transport.open_session()
        chan.set_combine_stderr(combine_stderr)
        chan.exec_command(cmd)

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        SENTINEL: str | None = None  # marks end of stream

        def reader() -> None:
            # Treat both '\n' and '\r' as line terminators so tqdm-style
            # carriage-return progress bars surface live instead of buffering
            # until the bar finishes. '\r\n' is normalised first to avoid
            # emitting a spurious empty line between the two characters.
            buf = bytearray()
            try:
                while True:
                    data = chan.recv(8192)
                    if not data:
                        break
                    buf.extend(data.replace(b"\r\n", b"\n"))
                    while True:
                        nl_n = buf.find(b"\n")
                        nl_r = buf.find(b"\r")
                        if nl_n < 0 and nl_r < 0:
                            break
                        if nl_n < 0:
                            nl = nl_r
                        elif nl_r < 0:
                            nl = nl_n
                        else:
                            nl = min(nl_n, nl_r)
                        line = bytes(buf[:nl]).decode("utf-8", errors="replace")
                        del buf[: nl + 1]
                        if line:
                            loop.call_soon_threadsafe(queue.put_nowait, line)
                if buf:
                    line = bytes(buf).decode("utf-8", errors="replace")
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        thread = threading.Thread(
            target=reader, name=f"pod-exec-{self.host}", daemon=True
        )
        thread.start()
        # Heartbeat: if no output arrives for HEARTBEAT_S seconds, emit a
        # synthetic line so the user can tell the channel is alive vs hung.
        # Cold imports, DALI compile, and big rsync runs can all go silent
        # for a minute or more.
        HEARTBEAT_S = 60.0
        silent_since = 0.0
        try:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
                except asyncio.TimeoutError:
                    silent_since += HEARTBEAT_S
                    on_log(f"  (no output for {int(silent_since)}s — still running)")
                    continue
                if line is SENTINEL:
                    break
                silent_since = 0.0
                on_log(line)
        except asyncio.CancelledError:
            chan.close()
            raise
        finally:
            # recv_exit_status is blocking; channel is closed by the time
            # the reader returns, so this returns promptly.
            await asyncio.to_thread(thread.join, 5.0)

        return await asyncio.to_thread(chan.recv_exit_status)

    async def exec_capture(
        self, cmd: str, *, combine_stderr: bool = True,
    ) -> tuple[int, str]:
        """Run a short command, return ``(exit_code, combined_output)``.

        Use only for quick commands (hashing a file, checking a path).
        Streams every byte into memory — not for ``train`` step output.
        """
        client = self._require_client()

        def _run() -> tuple[int, str]:
            transport = client.get_transport()
            if transport is None:
                raise RuntimeError("SSH transport is gone")
            chan = transport.open_session()
            chan.set_combine_stderr(combine_stderr)
            chan.exec_command(cmd)
            out = bytearray()
            while True:
                data = chan.recv(8192)
                if not data:
                    break
                out.extend(data)
            return chan.recv_exit_status(), bytes(out).decode("utf-8", errors="replace")

        return await asyncio.to_thread(_run)

    # ── file transfer ─────────────────────────────────────────────────────

    async def _open_sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            client = self._require_client()
            self._sftp = await asyncio.to_thread(client.open_sftp)
        return self._sftp

    def _sftp_guard(self) -> asyncio.Lock:
        """Lazy-init the lock on first use (so PodSession can be created
        outside an event-loop context, then used from inside one)."""
        if self._sftp_lock is None:
            self._sftp_lock = asyncio.Lock()
        return self._sftp_lock

    async def _reset_sftp(self) -> None:
        """Close and forget the cached SFTPClient — called when an
        operation raises so the next caller opens a fresh channel
        instead of inheriting a broken one."""
        if self._sftp is not None:
            try:
                await asyncio.to_thread(self._sftp.close)
            except Exception:
                pass
            self._sftp = None

    async def sftp_put(
        self,
        local: str | Path,
        remote: str,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Upload ``local`` to ``remote``. Creates parent dirs if missing.

        Verifies size with our own retry-aware stat instead of paramiko's
        ``confirm=True``: on RunPod's MooseFS volume, an immediate post-close
        ``stat`` sometimes returns 0 bytes before the close has flushed,
        making paramiko raise a spurious "size mismatch" OSError. Polling
        ``stat`` for up to ~3 seconds covers the race.
        """
        local_path = Path(local)
        expected = local_path.stat().st_size

        def _progress_thread_safe(transferred: int, total: int) -> None:
            if on_progress is not None:
                on_progress(transferred, total)

        async with self._sftp_guard():
            sftp = await self._open_sftp()
            # Make remote parent dirs (mkdir -p) — paramiko's SFTP doesn't.
            await self._mkdir_p(remote.rsplit("/", 1)[0])
            try:
                await asyncio.to_thread(
                    sftp.put, str(local_path), remote,
                    _progress_thread_safe if on_progress else None,
                    confirm=False,
                )
            except Exception:
                await self._reset_sftp()
                raise

            # Verify ourselves with retry. expected==0 needs no verification.
            if expected == 0:
                return
            delay = 0.1
            for _ in range(6):  # ~0.1+0.2+0.4+0.8+1.6+3.2s ≈ 6.3s total worst-case
                try:
                    actual = (await asyncio.to_thread(sftp.stat, remote)).st_size
                except OSError:
                    actual = -1
                if actual == expected:
                    return
                await asyncio.sleep(delay)
                delay *= 2
            raise OSError(
                f"sftp_put size mismatch after retries: {actual} != {expected} "
                f"(remote={remote!r}, local={local_path})"
            )

    async def sftp_put_bytes(
        self,
        data: bytes,
        remote: str,
        *,
        mode: int = 0o600,
    ) -> None:
        """Write ``data`` to ``remote`` directly, no local tempfile.

        Used for secrets we shouldn't write to local disk in cleartext
        (wandb / R2 keys). ``mode`` is applied to the remote file with
        ``chmod`` after the write.
        """
        async with self._sftp_guard():
            sftp = await self._open_sftp()
            await self._mkdir_p(remote.rsplit("/", 1)[0])

            def _write() -> None:
                with sftp.open(remote, "wb") as fh:
                    fh.write(data)
                sftp.chmod(remote, mode)

            try:
                await asyncio.to_thread(_write)
            except Exception:
                await self._reset_sftp()
                raise

    async def sftp_get(
        self,
        remote: str,
        local: str | Path,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Download ``remote`` to ``local``. Creates local parent dirs.

        Preserves the remote file's mtime on the local copy so subsequent
        size+mtime comparisons can correctly detect whether the remote has
        moved on since the last download.
        """
        local_path = Path(local)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._sftp_guard():
            sftp = await self._open_sftp()
            try:
                await asyncio.to_thread(
                    sftp.get, remote, str(local_path),
                    on_progress if on_progress else None,
                )
            except Exception:
                await self._reset_sftp()
                raise
            try:
                st = await asyncio.to_thread(sftp.stat, remote)
                if st.st_mtime is not None:
                    import os as _os
                    _os.utime(local_path, (float(st.st_atime or st.st_mtime),
                                           float(st.st_mtime)))
            except Exception:
                pass  # mtime preservation is best-effort

    async def _mkdir_p(self, remote_dir: str) -> None:
        """Equivalent of ``mkdir -p`` on the remote, via shell."""
        if not remote_dir or remote_dir == "/":
            return
        rc, _ = await self.exec_capture(f"mkdir -p {shlex.quote(remote_dir)}")
        if rc != 0:
            raise RuntimeError(f"mkdir -p {remote_dir!r} failed (rc={rc})")

    # ── helpers ───────────────────────────────────────────────────────────

    async def remote_sha256(self, path: str) -> str | None:
        """SHA-256 of a remote file, or ``None`` if it doesn't exist."""
        rc, out = await self.exec_capture(
            f"sha256sum {shlex.quote(path)} 2>/dev/null"
        )
        if rc != 0 or not out.strip():
            return None
        return out.strip().split()[0]

    async def remote_exists(self, path: str) -> bool:
        rc, _ = await self.exec_capture(f"test -e {shlex.quote(path)}")
        return rc == 0

    async def remote_stat(self, path: str) -> tuple[int, float] | None:
        """Return (size_bytes, mtime_unix) for a remote file, or None if absent."""
        async with self._sftp_guard():
            sftp = await self._open_sftp()
            try:
                st = await asyncio.to_thread(sftp.stat, path)
            except (IOError, FileNotFoundError):
                return None
            return int(st.st_size), float(st.st_mtime or 0.0)
