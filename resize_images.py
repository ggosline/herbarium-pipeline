"""
Resize a directory of JPEG images so the longer side <= max_size, preserving aspect ratio.

Uses NVIDIA DALI (GPU nvJPEG decode + resize) when available, falls back to
multi-threaded PIL otherwise.

Usage:
  # DALI GPU (fast, ~5000-20000 img/s)
  python resize_images.py --input-dir /path/to/project/images --output-dir /path/to/project/images_1024

  # Overwrite originals in-place
  python resize_images.py --input-dir /path/to/project/images --in-place

  # Force PIL fallback (no GPU required)
  python resize_images.py --input-dir /path/to/project/images --output-dir /path/to/project/images_1024 --no-dali

  # Skip images whose longer side is already <= max_size (no upscaling)
  python resize_images.py --input-dir /path/to/images --output-dir /path/to/out --no-upscale
"""

import argparse
import io
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # herbarium scans are legitimately large

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SUPPORTED_EXTS = {".jpg", ".jpeg"}
DEFAULT_MAX_SIZE = 1024
JPEG_QUALITY = 90


# ---------------------------------------------------------------------------
# PIL fallback
# ---------------------------------------------------------------------------

def _resize_one_pil(args: tuple[Path, Path, int, bool]) -> bool:
    src, dst, max_size, no_upscale = args
    try:
        img = Image.open(src).convert("RGB")
        if no_upscale and max(img.size) <= max_size:
            if dst != src:
                shutil.copy2(src, dst)
            return True
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        tmp = dst.with_suffix('.tmp')
        img.save(tmp, format="JPEG", quality=JPEG_QUALITY)
        tmp.replace(dst)
        return True
    except Exception as exc:
        print(f"  FAILED {src.name}: {exc}")
        return False


def resize_with_pil(
    files: list[Path],
    output_paths: list[Path],
    max_size: int,
    no_upscale: bool,
    workers: int = 8,
) -> tuple[int, int]:
    """Multi-threaded PIL resize. Returns (n_done, n_failed)."""
    print(f"Using PIL (CPU) with {workers} threads...")
    job_args = [(src, dst, max_size, no_upscale) for src, dst in zip(files, output_paths)]
    done = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_resize_one_pil, a): a for a in job_args}

        if tqdm is not None:
            bar = tqdm(as_completed(futures), total=len(files),
                       desc="Resizing (PIL)", unit="img", dynamic_ncols=True,
                       postfix={"fail": 0})
        else:
            bar = as_completed(futures)

        for future in bar:
            if future.result():
                done += 1
            else:
                failed += 1
                if tqdm is not None:
                    bar.set_postfix({"fail": failed})

    return done, failed


# ---------------------------------------------------------------------------
# Pre-flight verification
# ---------------------------------------------------------------------------

def verify_and_remove_corrupt(
    files: list[Path],
    output_paths: list[Path],
    workers: int = 8,
) -> tuple[list[Path], list[Path]]:
    """
    Open + verify every file with PIL. Deletes corrupt/unreadable files in-place
    and returns (files, output_paths) filtered to healthy files only.
    Uses threads since verification is IO-bound.
    """
    def _check(p: Path) -> bool:
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False

    print(f"Verifying {len(files):,} images…")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok_flags = list(pool.map(_check, files))

    bad  = [p for p, ok in zip(files, ok_flags) if not ok]
    good_files = [p for p, ok in zip(files, ok_flags) if ok]
    good_out   = [o for o, ok in zip(output_paths, ok_flags) if ok]

    if bad:
        print(f"  Removing {len(bad)} corrupt file(s):")
        for p in bad:
            print(f"    {p.name}")
            try:
                p.unlink()
            except OSError as exc:
                print(f"    (could not delete: {exc})")
    else:
        print("  All files OK.")
    return good_files, good_out


# ---------------------------------------------------------------------------
# DALI GPU path
# ---------------------------------------------------------------------------

def resize_with_dali(
    files: list[Path],
    output_paths: list[Path],
    max_size: int,
    no_upscale: bool,
    batch_size: int = 8,
    device_id: int = 0,
    num_threads: int = 8,
    prefetch_queue_depth: int = 1,
) -> tuple[int, int]:
    """
    DALI compiled pipeline: GPU decode + resize, parallel PIL JPEG encode.

    The @pipeline_def pipeline lets DALI manage its own internal prefetch queue
    (prefetch_queue_depth batches ahead), overlapping IO/decode/resize internally.
    A thread pool handles JPEG encoding on the CPU side, overlapping with DALI's
    next prefetch batch.
    """
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def

    files, output_paths = verify_and_remove_corrupt(files, output_paths, workers=num_threads)
    if not files:
        print("No valid images remain after verification.")
        return 0, 0

    print(f"Using NVIDIA DALI pipeline (GPU {device_id}) "
          f"batch={batch_size} prefetch={prefetch_queue_depth} encode_threads={num_threads}…")

    done = failed = 0

    # Pre-filter: for no_upscale, separate images already within bounds from those
    # that need downscaling. DALI always upscales to max_size, so we must check
    # original dimensions — not the DALI output — to decide what to copy vs resize.
    # Use a thread pool: JPEG header reads are IO-bound so threads parallelize well.
    if no_upscale:
        def _read_longer(src: Path) -> int:
            try:
                with Image.open(src) as im:
                    return max(im.size)
            except Exception:
                return max_size + 1  # can't read — assume needs resize

        print(f"Checking sizes of {len(files):,} images (no-upscale pre-filter)…")
        resize_files, resize_out = [], []
        copy_pairs = []
        with ThreadPoolExecutor(max_workers=num_threads) as pre_pool:
            for src, dst, longer in zip(
                files, output_paths,
                pre_pool.map(_read_longer, files)
            ):
                if longer <= max_size:
                    copy_pairs.append((src, dst))
                else:
                    resize_files.append(src)
                    resize_out.append(dst)

        # Copy small images in parallel
        if copy_pairs:
            with ThreadPoolExecutor(max_workers=num_threads) as copy_pool:
                futs = [copy_pool.submit(shutil.copy2, s, d)
                        for s, d in copy_pairs if s != d]
                for f in futs:
                    f.result()
            done += len(copy_pairs)

        files, output_paths = resize_files, resize_out
        print(f"  {done:,} already within {max_size}px (copied), "
              f"{len(files):,} need resizing.")
        if not files:
            return done, failed

    file_strs = [str(f) for f in files]

    @pipeline_def
    def _resize_pipeline():
        raw, _ = fn.readers.file(
            files=file_strs,
            random_shuffle=False,
            pad_last_batch=True,
            name="Reader",
        )
        # Mixed decode (nvJPEG on GPU) keeps the full-resolution intermediate
        # in GPU VRAM rather than RAM.  With batch_size=8 the peak VRAM for
        # decode is ~1.7 GB (8 × 7000×10000×3 bytes), safe for any 4+ GB GPU.
        # Increase --batch-size if you have more VRAM; decrease if you see OOM.
        images = fn.decoders.image(raw, device="mixed", output_type=types.RGB)
        return fn.resize(
            images, device="gpu",
            resize_longer=max_size,
            interp_type=types.INTERP_TRIANGULAR,
        )

    pipe = _resize_pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    pipe.build()

    n_batches = math.ceil(len(files) / batch_size)
    # Number of real (non-padded) images in the last batch
    last_real = len(files) % batch_size or batch_size

    b_start = 0

    bar = tqdm(total=len(files), desc="Resizing (DALI)", unit="img",
               dynamic_ncols=True, postfix={"fail": 0}) if tqdm is not None else None

    def _encode(arr, dst: Path) -> bool:
        try:
            tmp = dst.with_suffix('.tmp')
            Image.fromarray(arr).save(tmp, format="JPEG", quality=JPEG_QUALITY)
            tmp.replace(dst)
            return True
        except Exception as exc:
            (tqdm.write if tqdm else print)(f"  encode failed {dst.name}: {exc}")
            return False

    def _collect(futures) -> tuple[int, int]:
        n_ok = n_fail = 0
        for f in futures:
            if f.result():
                n_ok += 1
            else:
                n_fail += 1
        return n_ok, n_fail

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        pending = []  # encode futures from the previous batch
        pipe_dead = False  # once DALI hits a fatal error, switch all remaining to PIL

        for b_idx in range(n_batches):
            n_real = last_real if b_idx == n_batches - 1 else batch_size
            batch_files = files[b_start: b_start + n_real]
            batch_out   = output_paths[b_start: b_start + n_real]

            # Collect previous batch's encodes — they ran while DALI prefetched this batch
            n_ok, n_fail = _collect(pending)
            done += n_ok; failed += n_fail
            pending = []

            if pipe_dead:
                # Pipeline is invalid after a fatal error — use PIL for all remaining batches
                for src, dst in zip(batch_files, batch_out):
                    if _resize_one_pil((src, dst, max_size, no_upscale)):
                        done += 1
                    else:
                        failed += 1
            else:
                try:
                    # pipe.run() returns the next prefetched batch immediately;
                    # DALI is already computing batch b_idx+prefetch_queue_depth in the background
                    (resized_tl,) = pipe.run()
                    cpu_tl = resized_tl.as_cpu()  # GPU→CPU transfer (small: 1024px output)

                    # Copy arrays out of the TensorList: cpu_tl.at(i) is a view
                    # into DALI's ring-buffer slot.  Passing views to encode threads
                    # keeps the slot pinned; copying detaches ownership so DALI can
                    # reuse the slot for the next prefetch batch immediately.
                    arrays = [cpu_tl.at(i).copy() for i in range(len(batch_files))]

                    for arr, (src, dst) in zip(arrays, zip(batch_files, batch_out)):
                        pending.append(pool.submit(_encode, arr, dst))

                except Exception as exc:
                    (tqdm.write if tqdm else print)(
                        f"  DALI batch {b_idx} failed: {exc}\n"
                        f"  Pipeline invalidated — switching remaining {len(files) - b_start} "
                        f"images to PIL fallback.")
                    pipe_dead = True
                    for src, dst in zip(batch_files, batch_out):
                        if _resize_one_pil((src, dst, max_size, no_upscale)):
                            done += 1
                        else:
                            failed += 1

            b_start += n_real
            if bar is not None:
                bar.update(n_real)
                bar.set_postfix({"fail": failed})

        # Drain the last batch's encodes
        n_ok, n_fail = _collect(pending)
        done += n_ok; failed += n_fail

    if bar is not None:
        bar.close()
    return done, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize images so the longer side <= max_size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir", type=Path, required=True, metavar="DIR",
                        help="Directory containing source JPEG images")

    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--output-dir", type=Path, metavar="DIR",
                           help="Directory to write resized images (created if needed)")
    out_group.add_argument("--in-place", action="store_true",
                           help="Overwrite source images in-place")

    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE, metavar="PX",
                        help=f"Longer side target in pixels (default: {DEFAULT_MAX_SIZE})")
    parser.add_argument("--no-upscale", action="store_true",
                        help="Skip images whose longer side is already <= max_size")
    parser.add_argument("--no-dali", action="store_true",
                        help="Disable DALI and use multi-threaded PIL instead")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="DALI batch size (default: 8; increase if GPU has spare VRAM)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU device ID for DALI (default: 0)")
    parser.add_argument("--workers", type=int, default=8,
                        help="PIL fallback thread count (default: 8)")

    args = parser.parse_args()

    # Collect input files
    input_dir = args.input_dir.resolve()
    files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        print(f"No JPEG files found in {input_dir}")
        return

    # Build output paths
    if args.in_place:
        output_dir = input_dir
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [output_dir / f.name for f in files]

    print(f"Input:    {input_dir}  ({len(files)} images)")
    print(f"Output:   {output_dir}")
    print(f"Max size: {args.max_size}px (longer side)")
    print(f"No upscale: {args.no_upscale}\n")

    use_dali = not args.no_dali
    if use_dali:
        try:
            import nvidia.dali.experimental.dynamic  # noqa: F401
            import torch
            if not torch.cuda.is_available():
                print("No CUDA GPU detected — falling back to PIL.")
                use_dali = False
        except ImportError:
            print("DALI not installed — falling back to PIL.")
            use_dali = False

    if use_dali:
        done, failed = resize_with_dali(
            files, output_paths,
            max_size=args.max_size,
            no_upscale=args.no_upscale,
            batch_size=args.batch_size,
            device_id=args.device_id,
            num_threads=args.workers,
        )
    else:
        done, failed = resize_with_pil(
            files, output_paths,
            max_size=args.max_size,
            no_upscale=args.no_upscale,
            workers=args.workers,
        )

    print(f"\nDone — {done} resized, {failed} failed.")


if __name__ == "__main__":
    main()
