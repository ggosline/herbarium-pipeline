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
import json
import os
import re
import shutil
import subprocess
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tarfile
from pathlib import Path

# Same convention as push_model.py: training saves a loss-best and an
# accuracy-best checkpoint per stage, with the metric in the filename.
_LOSS_RE = re.compile(r"valid_loss=(\d+\.\d+)")
_ACC_RE  = re.compile(r"val_Accuracy=(\d+\.\d+)")


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


def _pick_remote_checkpoints(rclone: str, checkpoints_url: str, select_by: str,
                             env: dict[str, str] | None = None) -> list[str]:
    """Return the .ckpt filename(s) to pull from ``checkpoints_url``.

    ``select_by`` is "accuracy" (max val_Accuracy, default), "loss" (min
    valid_loss), or "all" (every .ckpt — the old behaviour). Mirrors
    push_model.py's _pick_checkpoint fallback order: preferred metric, then
    the other metric, then most-recently-modified. A multi-GB .ckpt archive
    otherwise means downloading every stage's checkpoint just to use one.
    """
    out = subprocess.run(
        [rclone, "lsjson", checkpoints_url],
        capture_output=True, text=True, check=False,
        env={**os.environ, **(env or {})},
    )
    if out.returncode != 0 or not out.stdout.strip():
        return []
    try:
        entries = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    ckpts = [e for e in entries if str(e.get("Name", "")).endswith(".ckpt")]
    if not ckpts:
        return []
    if select_by == "all":
        return [e["Name"] for e in ckpts]

    def metric(name: str, rx: re.Pattern) -> float | None:
        m = rx.search(name)
        return float(m.group(1)) if m else None

    by_acc  = (_ACC_RE,  True)
    by_loss = (_LOSS_RE, False)
    order = [by_acc, by_loss] if select_by == "accuracy" else [by_loss, by_acc]
    for rx, want_max in order:
        cand = [(metric(e["Name"], rx), e["Name"]) for e in ckpts
                if metric(e["Name"], rx) is not None]
        if cand:
            cand.sort(key=lambda x: x[0], reverse=want_max)
            return [cand[0][1]]
    # No filename carries either metric — fall back to most recently modified.
    ckpts.sort(key=lambda e: e.get("ModTime", ""), reverse=True)
    return [ckpts[0]["Name"]]


def restore(project: str, target: Path, remote: str = "r2:herbarium-backup",
            images_dirname: str = "images",
            skip_images_if_present: bool = False,
            select_by: str = "accuracy") -> int:
    rclone = _have_rclone()
    env = _r2_env()
    if env:
        print("  (using R2 credentials from OS keyring)", flush=True)
    base = f"{remote}/{project}"
    print(f"→ Restoring '{project}' from {base} to {target}", flush=True)
    target.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint(s) — by default just the one Identify/Publish would use
    # (accuracy-best), not every stage's multi-GB .ckpt. --select-by all
    # restores the old pull-everything behaviour.
    ckpt_dir = target / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    chosen = _pick_remote_checkpoints(rclone, f"{base}/checkpoints/", select_by, env=env)
    if chosen:
        print(f"→ Pulling checkpoint (select_by={select_by}): {', '.join(chosen)}", flush=True)
        for name in chosen:
            rc = _run([rclone, "copyto", f"{base}/checkpoints/{name}",
                       str(ckpt_dir / name), "--progress",
                       "--s3-chunk-size", "64M"], extra_env=env)
            if rc != 0:
                print(f"⚠ checkpoint copy returned {rc}", flush=True)
        # nameslist.json + any other metadata sitting alongside the checkpoints.
        _copy(rclone, f"{base}/checkpoints/", ckpt_dir, "--include", "*.json", env=env)
    else:
        print("  no .ckpt files found in archive checkpoints/ — skipping", flush=True)

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
    #    Skip the (multi-GB) image pull entirely when the caller says images
    #    are already present locally — the Review "Fetch" path passes this so a
    #    re-fetch of predictions doesn't re-download images. Delete the folder
    #    to force a refresh.
    img_present = (target / images_dirname)
    if skip_images_if_present and img_present.is_dir() and any(img_present.iterdir()):
        print(f"  images already present at {img_present} — skipping image pull "
              f"(delete the folder to force a refresh).", flush=True)
        print(f"✓ Restore complete at {target}", flush=True)
        return 0
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
    p.add_argument("--skip-images-if-present", action="store_true",
                   help="Skip pulling/extracting the image tarball when the "
                        "images directory already exists and is non-empty.")
    p.add_argument("--select-by", choices=("accuracy", "loss", "all"), default="accuracy",
                   help="Which archived checkpoint(s) to pull: highest "
                        "val_Accuracy (default, matches Identify/Publish), "
                        "lowest valid_loss, or all of them.")
    args = p.parse_args()
    return restore(args.project, args.target.resolve(),
                   select_by=args.select_by,
                   remote=args.remote, images_dirname=args.images_dirname,
                   skip_images_if_present=args.skip_images_if_present)


if __name__ == "__main__":
    sys.exit(main())
