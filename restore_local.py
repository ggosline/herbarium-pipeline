"""
Pull an R2-archived herbarium project onto the local machine.

Cross-platform replacement for `pod_bootstrap.sh restore`: works on Windows,
macOS, Linux. Requires `rclone` on PATH with the same remote configured as
the pod (default name: `r2`).

Layout pulled into <target>:
  checkpoints/        — *.ckpt + *.json (nameslist, metadata)
  specsin.csv
  gbif.zip            (if present in the archive)
  predictions/        (if present)
  images/             — extracted from <project>/<images_dir>.tar

Usage:
  python restore_local.py --project ebenaceae --target /mnt/e/Pipeline/data
  python restore_local.py --project ebenaceae --target D:\\herbarium\\ebenaceae
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tarfile
from pathlib import Path


def _have_rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        sys.exit(
            "rclone not found on PATH. Install from https://rclone.org/install/ "
            "and configure the same remote you use on the pod (e.g. `rclone "
            "config` → new R2 remote)."
        )
    return exe


def _r2_env() -> dict[str, str]:
    """Return rclone env vars for the r2 remote read from the OS keyring.

    If no credentials are stored returns an empty dict (rclone will fall back
    to whatever is in rclone.conf, or fail with its own error message).
    """
    try:
        from cloud.secrets import get_r2_credentials
        creds = get_r2_credentials()
    except Exception:
        return {}
    if creds is None:
        return {}
    return {
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Other",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": creds.access_key_id,
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": creds.secret_access_key,
        "RCLONE_CONFIG_R2_ENDPOINT": creds.endpoint,
        "RCLONE_CONFIG_R2_ACL": "private",
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    }


def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> int:
    """Run a command, streaming its output line-by-line to stdout (which the
    webui captures). Returns the process exit code."""
    print(f"$ {' '.join(cmd)}", flush=True)
    env = {**os.environ, **(extra_env or {})}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    proc.wait()
    return proc.returncode


def _rclone_lsf(rclone: str, remote_path: str,
                env: dict[str, str] | None = None) -> list[str]:
    """List files at remote_path (no recursion). Empty list if missing."""
    try:
        out = subprocess.run(
            [rclone, "lsf", remote_path],
            capture_output=True, text=True, check=False,
            env={**os.environ, **(env or {})},
        )
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _copy(rclone: str, src: str, dst: Path, *extra: str,
          env: dict[str, str] | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    return _run([rclone, "copy", src, str(dst), "--progress",
                 "--transfers", "4", "--s3-chunk-size", "64M", *extra],
                extra_env=env)


def restore(project: str, target: Path, remote: str = "r2:herbarium-backup",
            images_dirname: str = "images") -> int:
    rclone = _have_rclone()
    env = _r2_env()
    if env:
        print("  (using R2 credentials from OS keyring)", flush=True)
    base = f"{remote}/{project}"
    print(f"→ Restoring '{project}' from {base} to {target}", flush=True)
    target.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoints (+ nameslist.json next to them)
    rc = _copy(rclone, f"{base}/checkpoints/", target / "checkpoints", env=env)
    if rc != 0:
        print(f"⚠ checkpoint copy returned {rc}", flush=True)

    # 2. Per-project state (specsin, dwca) — copied selectively from project root
    _copy(rclone, f"{base}/", target,
          "--include", "specsin.csv", "--include", "gbif.zip", env=env)

    # 3. Predictions (optional)
    if _rclone_lsf(rclone, f"{base}/predictions/", env=env):
        _copy(rclone, f"{base}/predictions/", target / "predictions", env=env)
    else:
        print("  (no predictions/ in archive — skipping)", flush=True)

    # 4. Images tarball — try the modern <images_dirname>.tar first, then the
    #    legacy `images_1024.tar` name used by older archives.
    candidates = [f"{images_dirname}.tar", "images_1024.tar"]
    found = None
    for name in candidates:
        if _rclone_lsf(rclone, f"{base}/{name}", env=env):
            found = name
            break
    if found is None:
        print(f"⚠ No image tarball found at {base}/ (looked for: "
              f"{', '.join(candidates)}).", flush=True)
    else:
        tar_path = target / found
        print(f"→ Pulling {found}…", flush=True)
        rc = _run([rclone, "copy", f"{base}/{found}", str(target),
                   "--progress", "--transfers", "4", "--s3-chunk-size", "64M"],
                  extra_env=env)
        if rc != 0:
            return rc
        # Extract with stdlib tarfile so Windows doesn't need tar.exe.
        # Reset the images dir if it already exists so we don't leave stale
        # files from a previous run mixed in with the restored set.
        img_dst = target / images_dirname
        if img_dst.exists():
            print(f"  removing existing {img_dst} before extract…", flush=True)
            shutil.rmtree(img_dst)
        print(f"→ Extracting {tar_path.name} → {target}…", flush=True)
        with tarfile.open(tar_path) as tf:
            # The archive contains a top-level dir matching the original
            # IMG_BASENAME (whatever the pod's IMAGES_DIR was named). If that
            # name is not images_dirname, rename after extraction.
            members = tf.getmembers()
            top_dirs = {m.name.split("/", 1)[0] for m in members if m.name}
            tf.extractall(target)  # noqa: S202 (archive is from our own backup)
        tar_path.unlink()
        # Rename top-level dir to the requested images_dirname if needed.
        if len(top_dirs) == 1:
            extracted = target / next(iter(top_dirs))
            if extracted.exists() and extracted != img_dst:
                extracted.rename(img_dst)
        n_files = sum(1 for _ in img_dst.rglob("*") if _.is_file())
        print(f"  extracted {n_files} files into {img_dst}", flush=True)

    print(f"✓ Restore complete at {target}", flush=True)
    print(f"  Identify pointers: ckpt={target / 'checkpoints'}, "
          f"specsin={target / 'specsin.csv'}, images={target / images_dirname}",
          flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True,
                   help="Project name (e.g. 'ebenaceae'). Must match the "
                        "PROJECT used at backup time.")
    p.add_argument("--target", required=True, type=Path,
                   help="Local directory to restore into (will be created).")
    p.add_argument("--remote", default="r2:herbarium-backup",
                   help="rclone remote + bucket. Default: r2:herbarium-backup")
    p.add_argument("--images-dirname", default="images",
                   help="Final name of the images directory under <target>. "
                        "Default: images")
    args = p.parse_args()
    return restore(args.project, args.target.resolve(),
                   remote=args.remote, images_dirname=args.images_dirname)


if __name__ == "__main__":
    sys.exit(main())
