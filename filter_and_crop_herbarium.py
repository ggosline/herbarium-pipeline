"""
filter_and_crop_herbarium.py — Post-download herbarium image cleanup.

Steps:
  1. Filter: reject field/living-plant photos, keep genuine herbarium scans.
     - CLIP zero-shot (default, GPU) or HSV heuristic (--filter-method hsv)
  2. Crop: remove dark scanning-bed border by scanning inward from each edge.
  3. Optionally update specsin.csv: set hasfile=False for rejected images.

Usage examples:
  # HSV filter + crop, no GPU needed:
  python filter_and_crop_herbarium.py \
      --input-dir /path/to/project/images \
      --output-dir /path/to/project/images_filtered \
      --filter-method hsv

  # CLIP filter + crop, update specsin:
  python filter_and_crop_herbarium.py \
      --input-dir /path/to/project/images \
      --output-dir /path/to/project/images_filtered \
      --specsin /path/to/project/specsin.csv

  # In-place (overwrite originals), crop only:
  python filter_and_crop_herbarium.py \
      --input-dir /path/to/project/images \
      --in-place \
      --no-filter
"""

import argparse
import shutil
import sys
import time
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # herbarium scans are legitimately large

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ---------------------------------------------------------------------------
# Slide pre-filter (aspect ratio)
# ---------------------------------------------------------------------------

# Microscope slides are narrow rectangles (25 mm × 75 mm ≈ 1:3 ratio).
# Herbarium sheets are roughly A3 paper (≈ 2:3 ratio, min side / max side ≥ 0.55).
# Any image whose shorter side is less than SLIDE_RATIO_THRESHOLD of its longer
# side is rejected immediately — before CLIP or HSV — as a slide scan.
SLIDE_RATIO_THRESHOLD = 0.50


def _is_slide(img_bgr: np.ndarray) -> bool:
    """Return True if the image looks like a microscope slide (very narrow aspect ratio)."""
    h, w = img_bgr.shape[:2]
    return min(h, w) / max(h, w) < SLIDE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Module-level worker functions (picklable for multiprocessing.Pool)
#
# Workers receive pre-read raw JPEG bytes from the main process so that
# all disk IO is sequential in the main process — no cross-worker contention.
# Workers do pure CPU work: JPEG decode, classify/crop, JPEG encode.
# ---------------------------------------------------------------------------

def _hsv_worker(args: tuple) -> tuple[str, str]:
    """HSV filter worker: receives raw JPEG bytes, returns (path_str, status)."""
    p_str, raw_bytes, white_ratio, saturation = args
    nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return p_str, "warn"
    if _is_slide(img):
        return p_str, "rejected"
    if _is_label_hsv(img):
        return p_str, "rejected"
    if _is_herbarium_hsv(img, white_ratio, saturation):
        return p_str, "keep"
    return p_str, "live"


def _hsv_filter_crop_worker(args: tuple) -> tuple[str, str, str]:
    """Combined HSV filter + crop worker — one decode per image.

    Returns (src_str, label, crop_status) where
      label:       'keep' | 'live' | 'rejected' | 'warn'
      crop_status: 'cropped' | 'passthrough' | 'failed' | '' (non-keep)
    """
    src_str, dst_str, raw_bytes, white_ratio, saturation, padding, do_crop = args
    nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return src_str, "warn", ""
    if _is_slide(img):
        return src_str, "rejected", ""
    if _is_label_hsv(img):
        return src_str, "rejected", ""
    if not _is_herbarium_hsv(img, white_ratio, saturation):
        return src_str, "live", ""

    in_place = (src_str == dst_str)

    if not do_crop:
        if not in_place:
            Path(dst_str).write_bytes(raw_bytes)
        return src_str, "keep", "passthrough"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]
    top, bottom, left, right = _find_sheet_bounds(gray)
    x1 = max(0, left   - padding)
    y1 = max(0, top    - padding)
    x2 = min(w_img, right  + padding)
    y2 = min(h_img, bottom + padding)

    if x1 == 0 and y1 == 0 and x2 == w_img and y2 == h_img:
        if not in_place:
            Path(dst_str).write_bytes(raw_bytes)
        return src_str, "keep", "passthrough"

    cropped = img[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    Path(dst_str).parent.mkdir(parents=True, exist_ok=True)
    try:
        pil_img.save(dst_str, "JPEG", quality=90)
        return src_str, "keep", "cropped"
    except Exception:
        return src_str, "keep", "failed"


def _crop_worker(args: tuple) -> tuple[str, str]:
    """Crop worker: receives raw JPEG bytes, crops, writes output, returns status."""
    src_str, dst_str, raw_bytes, padding = args
    in_place = (src_str == dst_str)

    nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return src_str, "failed"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_bgr.shape[:2]
    top, bottom, left, right = _find_sheet_bounds(gray)

    x1 = max(0, left   - padding)
    y1 = max(0, top    - padding)
    x2 = min(w_img, right  + padding)
    y2 = min(h_img, bottom + padding)

    if x1 == 0 and y1 == 0 and x2 == w_img and y2 == h_img:
        if not in_place:
            Path(dst_str).write_bytes(raw_bytes)  # copy original bytes, no re-encode
        return src_str, "passthrough"

    cropped = img_bgr[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    Path(dst_str).parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(dst_str, "JPEG", quality=90)
    return src_str, "cropped"


# ---------------------------------------------------------------------------
# HSV heuristic filter
# ---------------------------------------------------------------------------

def _is_label_hsv(img_bgr: np.ndarray) -> bool:
    """Return True if the image looks like a label card (text on white, no plant material).

    Label cards are almost entirely white/grey with no colour content at all —
    white_ratio > 0.92 AND mean saturation near zero (< 8).
    Herbarium sheets pass the same whiteness check but typically have more colour
    variance from the pressed specimen, so mean saturation is usually >= 8.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white_ratio = np.mean(hsv[:, :, 2] > 200)
    mean_sat = np.mean(hsv[:, :, 1])
    return white_ratio > 0.92 and mean_sat < 8.0


def _is_herbarium_hsv(img_bgr: np.ndarray,
                      white_ratio_thresh: float,
                      saturation_thresh: float) -> bool:
    """Return True if the image looks like a herbarium sheet (high white area, low saturation)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    white_ratio = np.mean(v_channel > 200)
    mean_sat = np.mean(s_channel)
    return white_ratio >= white_ratio_thresh and mean_sat <= saturation_thresh


def filter_crop_hsv(paths: list[Path],
                    output_paths: list[Path],
                    white_ratio: float,
                    saturation: float,
                    padding: int,
                    do_crop: bool,
                    workers: int = 8) -> tuple[list[Path], list[Path], list[Path], dict]:
    """Single-pass HSV filter + crop. Returns (keep, live, rejected, crop_stats).

    Each image is decoded exactly once: classified and, if kept, cropped and
    written to its output path — all in the same worker call.
    """
    def _gen():
        for src, dst in zip(paths, output_paths):
            yield (str(src), str(dst), src.read_bytes(),
                   white_ratio, saturation, padding, do_crop)

    keep, live, rejected = [], [], []
    crop_stats = {"cropped": 0, "passthrough": 0, "failed": 0}

    with multiprocessing.Pool(processes=workers) as pool:
        for src_str, label, crop_status in tqdm(
                pool.imap_unordered(_hsv_filter_crop_worker, _gen(), chunksize=1),
                total=len(paths), desc="Filter+Crop (HSV)", unit="img"):
            if label == "warn":
                print(f"  [warn] cannot read {Path(src_str).name}, skipping")
            elif label == "keep":
                keep.append(Path(src_str))
                if crop_status in crop_stats:
                    crop_stats[crop_status] += 1
            elif label == "live":
                live.append(Path(src_str))
            else:
                rejected.append(Path(src_str))

    return keep, live, rejected, crop_stats


# ---------------------------------------------------------------------------
# CLIP zero-shot filter
# ---------------------------------------------------------------------------

CLIP_PROMPTS = [
    "a herbarium specimen sheet: a dried plant pressed and mounted on white or cream archival paper, with identification labels, barcodes, or institutional stamps visible",
    "a photograph of a living or recently collected plant in a natural, garden, or field setting",
    "a label card or data slip with printed or handwritten text only, no plant material visible",
]

# Minimum fraction of bright pixels (V > 160) an image must have to be sent to
# CLIP.  Genuine herbarium sheets are mostly white paper (white_ratio >> 0.10).
# Field photos, dark-background specimen shots, and slides score near zero and
# are classified as "live" before any GPU inference is performed.
_MIN_WHITE_RATIO = 0.10

# Standard OpenAI CLIP normalisation (all openai/clip-* models)
_CLIP_SIZE = 224
_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _preprocess_clip_bytes(args: tuple) -> tuple[str, "np.ndarray | None", str]:
    """
    Decode one JPEG (given as raw bytes) and preprocess for CLIP inference.
    Runs in a worker process — pure CPU, no disk IO.
    Returns (path_str, chw_float32_array | None, status) where status is:
      ''       — normal; array is ready for CLIP inference
      'slide'  — rejected as microscope slide (aspect ratio)
      'live'   — dark background: not a herbarium scan, skip CLIP inference
      'unread' — could not decode image
    """
    p_str, raw_bytes = args
    nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return p_str, None, "unread"
    h, w = img.shape[:2]
    if min(h, w) / max(h, w) < SLIDE_RATIO_THRESHOLD:
        return p_str, None, "slide"

    # Cheap white-background gate: herbarium sheets are mostly white/cream paper.
    # Images with < _MIN_WHITE_RATIO bright pixels (dark backgrounds, field shots)
    # are classified as 'live' without wasting GPU inference time.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if float(np.mean(hsv[:, :, 2] > 160)) < _MIN_WHITE_RATIO:
        return p_str, None, "live"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize shortest side to _CLIP_SIZE, then centre-crop
    scale = _CLIP_SIZE / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top  = (nh - _CLIP_SIZE) // 2
    left = (nw - _CLIP_SIZE) // 2
    img  = img[top: top + _CLIP_SIZE, left: left + _CLIP_SIZE]
    arr  = (img.astype(np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
    return p_str, arr.transpose(2, 0, 1).copy(), ""  # CHW, C-contiguous


def filter_crop_clip(paths: list[Path],
                     output_paths: list[Path],
                     model_name: str,
                     confidence: float,
                     padding: int,
                     do_crop: bool,
                     batch_size: int = 32,
                     workers: int = 8) -> tuple[list[Path], list[Path], list[Path], dict]:
    """Single-pass CLIP filter + crop. Returns (keep, live, rejected, crop_stats).

    Each image is read from disk exactly once. Raw bytes are cached in the main
    process only until the image's CLIP batch is classified; 'keep' images are
    then dispatched to a crop pool (running concurrently with the next CLIP
    batch), after which the raw bytes are freed.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("[error] transformers / torch not available; falling back to HSV filter+crop.")
        return filter_crop_hsv(paths, output_paths, 0.25, 40, padding, do_crop, workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading CLIP model {model_name!r} on {device} …")
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    with torch.no_grad():
        text_inputs   = processor(text=CLIP_PROMPTS, return_tensors="pt",
                                  padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()

    dst_map          = {str(src): dst for src, dst in zip(paths, output_paths)}
    raw_bytes_cache: dict[str, bytes] = {}   # freed per-image right after classify
    keep, live, rejected = [], [], []
    crop_stats  = {"cropped": 0, "passthrough": 0, "failed": 0}
    crop_pool   = multiprocessing.Pool(processes=workers)
    crop_async  = []  # AsyncResult objects

    def _drain_crop(wait: bool = False):
        """Collect completed crop results; if wait=True block until all done."""
        remaining = []
        for res in crop_async:
            if wait or res.ready():
                try:
                    _, status = res.get(timeout=120)
                    if status in crop_stats:
                        crop_stats[status] += 1
                except Exception:
                    crop_stats["failed"] += 1
            else:
                remaining.append(res)
        crop_async[:] = remaining

    def _classify_and_crop(arr_list: list, path_list: list[Path]):
        img_tensor = torch.from_numpy(np.stack(arr_list)).to(device)
        with torch.no_grad():
            img_features = model.get_image_features(pixel_values=img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            probs = (logit_scale * (img_features @ text_features.T)) \
                        .softmax(dim=-1).cpu().numpy()
        for path, prob_row in zip(path_list, probs):
            herb_p  = float(prob_row[0])
            live_p  = float(prob_row[1])
            label_p = float(prob_row[2])
            raw = raw_bytes_cache.pop(str(path), None)  # free from cache
            dst = dst_map[str(path)]

            if label_p > herb_p:
                rejected.append(path)
            elif herb_p >= confidence:
                keep.append(path)
                if raw is not None:
                    if do_crop:
                        crop_async.append(
                            crop_pool.apply_async(_crop_worker,
                                                  ((str(path), str(dst), raw, padding),)))
                    elif str(path) != str(dst):
                        Path(dst).write_bytes(raw)
                        crop_stats["passthrough"] += 1
            elif live_p >= herb_p:
                live.append(path)
            else:
                rejected.append(path)

        _drain_crop(wait=False)  # collect any already-finished crop tasks

    # Generator: sequential IO in main process, raw bytes also cached here
    def _gen():
        for p in paths:
            raw = p.read_bytes()
            raw_bytes_cache[str(p)] = raw
            yield (str(p), raw)

    pending_arrays: list = []
    pending_paths:  list[Path] = []

    with multiprocessing.Pool(processes=workers) as preprocess_pool:
        for p_str, arr, status in tqdm(
                preprocess_pool.imap_unordered(_preprocess_clip_bytes, _gen(), chunksize=1),
                total=len(paths), desc="Filter+Crop (CLIP)", unit="img"):

            if status == "slide":
                rejected.append(Path(p_str))
                raw_bytes_cache.pop(p_str, None)
                continue
            if status == "live":
                live.append(Path(p_str))
                raw_bytes_cache.pop(p_str, None)
                continue
            if arr is None:  # unread
                raw_bytes_cache.pop(p_str, None)
                continue

            pending_arrays.append(arr)
            pending_paths.append(Path(p_str))

            if len(pending_arrays) >= batch_size:
                _classify_and_crop(pending_arrays, pending_paths)
                pending_arrays, pending_paths = [], []

        if pending_arrays:
            _classify_and_crop(pending_arrays, pending_paths)

    crop_pool.close()
    crop_pool.join()
    _drain_crop(wait=True)

    return keep, live, rejected, crop_stats


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def _find_sheet_bounds(gray: np.ndarray,
                       bright_thresh: int = 160,
                       edge_frac_min: float = 0.10) -> tuple[int, int, int, int]:
    """
    Find the sheet boundary by scanning inward from each image edge until the
    first row/column whose mean brightness fraction exceeds edge_frac_min.

    This approach is robust to:
    - Grey scanning-bed borders (WAG/Naturalis scans): the textured grey border
      has brightness ~80–150, well below bright_thresh=160, so border rows have
      fraction ~0.00 while even heavily specimen-covered sheet rows retain enough
      cream/white background (>160) to exceed edge_frac_min=0.10.
    - No-border images (Kew white-background scans): the edge rows are already
      bright so the function returns (0, h, 0, w) — no crop.
    - Large dark specimens: only edge rows are examined, interior darkness is
      irrelevant.

    Returns (top, bottom, left, right) where top/left are the first bright
    row/col from each edge, and bottom/right are one past the last bright
    row/col from each edge (i.e. suitable for array slicing).
    """
    bright = (gray > bright_thresh).astype(np.float32)
    row_frac = bright.mean(axis=1)
    col_frac = bright.mean(axis=0)
    h, w = gray.shape

    def first_bright(profile: np.ndarray, reverse: bool) -> int:
        indices = range(len(profile) - 1, -1, -1) if reverse else range(len(profile))
        for i in indices:
            if profile[i] > edge_frac_min:
                return i
        return 0 if reverse else len(profile) - 1

    top    = first_bright(row_frac, reverse=False)
    bottom = first_bright(row_frac, reverse=True) + 1
    left   = first_bright(col_frac, reverse=False)
    right  = first_bright(col_frac, reverse=True) + 1
    return top, bottom, left, right


def _crop_white_border(src: Path, dst: Path, padding: int) -> str:
    """
    Remove the dark scanning-bed border from a herbarium sheet scan.
    Scans from each image edge inward to locate where the dark border
    ends and the herbarium sheet begins, then crops to that rectangle.
    Returns "cropped", "passthrough", or "failed".
    """
    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        return "failed"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_bgr.shape[:2]

    top, bottom, left, right = _find_sheet_bounds(gray)

    # Add padding, clamp to image bounds
    x1 = max(0, left   - padding)
    y1 = max(0, top    - padding)
    x2 = min(w_img, right  + padding)
    y2 = min(h_img, bottom + padding)

    # If no border detected (already fills the frame), copy as-is
    if x1 == 0 and y1 == 0 and x2 == w_img and y2 == h_img:
        if src != dst:
            shutil.copy2(src, dst)
        return "passthrough"

    cropped = img_bgr[y1:y2, x1:x2]

    # Save via PIL for reliable JPEG quality control
    dst.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_img.save(str(dst), "JPEG", quality=90)
    return "cropped"


def crop_images(paths: list[Path],
                output_dir: Path | None,
                in_place: bool,
                padding: int,
                workers: int) -> dict:
    """Crop white borders from all accepted images. Returns a stats dict."""
    # Generator reads files sequentially in the main process.
    # Workers decode, crop, and write output — pure CPU, no read contention.
    def _gen():
        for p in paths:
            dst = p if in_place else output_dir / p.name
            yield (str(p), str(dst), p.read_bytes(), padding)

    n_cropped = n_passthrough = n_failed = 0
    with multiprocessing.Pool(processes=workers) as pool:
        for src_str, status in tqdm(
                pool.imap_unordered(_crop_worker, _gen(), chunksize=1),
                total=len(paths), desc="Cropping", unit="img"):
            if status == "cropped":
                n_cropped += 1
            elif status == "passthrough":
                n_passthrough += 1
            elif status == "failed":
                print(f"  [warn] crop failed: {Path(src_str).name}")
                n_failed += 1

    return {"cropped": n_cropped, "passthrough": n_passthrough, "failed": n_failed}


# ---------------------------------------------------------------------------
# specsin.csv update
# ---------------------------------------------------------------------------

def update_specsin(specsin_path: Path,
                   live: list[Path],
                   rejected: list[Path]):
    """Set hasfile=False for live and rejected images in specsin.csv."""
    try:
        import pandas as pd
    except ImportError:
        print("[warn] pandas not available; skipping specsin update.")
        return

    df = pd.read_csv(specsin_path)
    all_non_herb = {p.name for p in live} | {p.name for p in rejected}
    mask = df["fname"].isin(all_non_herb)
    n_updated = mask.sum()
    df.loc[mask, "hasfile"] = False
    df.to_csv(specsin_path, index=False)
    print(f"  specsin: set hasfile=False for {n_updated} row(s) "
          f"({len(live)} live, {len(rejected)} rejected) → {specsin_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter non-herbarium images and crop white borders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input-dir",  required=True,  type=Path, help="Source JPEG directory")

    out_grp = p.add_mutually_exclusive_group(required=True)
    out_grp.add_argument("--output-dir", type=Path,
                         help="Output directory for accepted images")
    out_grp.add_argument("--in-place", action="store_true",
                         help="Overwrite originals in input-dir")

    p.add_argument("--specsin", type=Path, default=None,
                   help="specsin.csv to update hasfile=False for rejected images")

    p.add_argument("--no-filter", action="store_true",
                   help="Skip herbarium/field classification")
    p.add_argument("--no-crop",   action="store_true",
                   help="Skip white border cropping")

    p.add_argument("--filter-method", choices=["clip", "hsv"], default="clip",
                   help="Classification method")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                   help="HuggingFace CLIP model name")
    p.add_argument("--confidence", type=float, default=0.6,
                   help="Min herbarium probability to keep (CLIP)")

    p.add_argument("--hsv-white-ratio", type=float, default=0.25,
                   help="Min white-pixel fraction for herbarium (HSV)")
    p.add_argument("--hsv-saturation",  type=float, default=40.0,
                   help="Max mean saturation for herbarium (HSV)")

    p.add_argument("--crop-padding", type=int, default=10,
                   help="Pixels of padding around detected content")
    p.add_argument("--batch-size",   type=int, default=32,
                   help="CLIP inference batch size")
    p.add_argument("--workers",      type=int, default=8,
                   help="Thread count for crop step")
    p.add_argument("--force", action="store_true",
                   help="Re-process images even if already processed on a previous run")

    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        sys.exit(f"[error] input-dir does not exist: {input_dir}")

    in_place: bool = args.in_place
    output_dir: Path | None = None if in_place else args.output_dir

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Non-herbarium images go into subdirs inside output (or input for --in-place)
    base_dir     = input_dir if in_place else output_dir
    rejected_dir = base_dir / "rejected"   # slides and other junk
    live_dir     = base_dir / "live"       # field/living-plant photos for other use

    # Manifest file tracks processed filenames for in-place re-run skipping
    manifest_path = input_dir / ".filter_crop_done"

    # ------------------------------------------------------------------ 1. Collect
    all_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"} and p.parent == input_dir
    )
    n_total = len(all_paths)
    print(f"Found {n_total:,} JPEG file(s) in {input_dir}")

    if not all_paths:
        print("Nothing to do.")
        return

    # ------------------------------------------------------------------ 1b. Skip already processed
    n_skipped = 0
    if not args.force:
        if not in_place:
            # Skip any image whose output already exists in output_dir, live_dir, or rejected_dir
            already: set[str] = set()
            for d in (output_dir, live_dir, rejected_dir):
                if d and d.is_dir():
                    already |= {f.name for f in d.iterdir()
                                 if f.suffix.lower() in {".jpg", ".jpeg"}}
            all_paths = [p for p in all_paths if p.name not in already]
        else:
            # In-place: use manifest
            done_names: set[str] = set()
            if manifest_path.exists():
                done_names = set(manifest_path.read_text().splitlines())
            all_paths = [p for p in all_paths if p.name not in done_names]

        n_skipped = n_total - len(all_paths)
        if n_skipped:
            print(f"  {n_skipped:,} already processed (skipped) — use --force to reprocess")

    n_new = len(all_paths)
    print(f"  {n_new:,} image(s) to process")

    if not all_paths:
        elapsed = time.time() - t_start
        print(f"\nNothing new to do. ({elapsed:.1f}s)")
        return

    # Output path for each image (where 'keep' images are written)
    output_paths = [
        p if in_place else output_dir / p.name
        for p in all_paths
    ]

    # ------------------------------------------------------------------ 2. Filter (+ crop)
    crop_stats = {"cropped": 0, "passthrough": 0, "failed": 0}
    if args.no_filter:
        keep = all_paths
        live:     list[Path] = []
        rejected: list[Path] = []
        print("\nFilter step skipped (--no-filter).")
    else:
        do_crop = not args.no_crop
        label = "Filter+Crop" if do_crop else "Filter"
        print(f"\n{label} {n_new:,} image(s) with method: {args.filter_method}")
        if args.filter_method == "clip":
            keep, live, rejected, crop_stats = filter_crop_clip(
                all_paths,
                output_paths,
                model_name=args.clip_model,
                confidence=args.confidence,
                padding=args.crop_padding,
                do_crop=do_crop,
                batch_size=args.batch_size,
                workers=args.workers,
            )
        else:
            keep, live, rejected, crop_stats = filter_crop_hsv(
                all_paths,
                output_paths,
                white_ratio=args.hsv_white_ratio,
                saturation=args.hsv_saturation,
                padding=args.crop_padding,
                do_crop=do_crop,
                workers=args.workers,
            )

        def _move_or_copy(paths: list[Path], dest_dir: Path):
            if not paths:
                return
            dest_dir.mkdir(parents=True, exist_ok=True)
            for p in tqdm(paths, desc=f"Moving → {dest_dir.name}/", unit="img"):
                dest = dest_dir / p.name
                if in_place:
                    shutil.move(str(p), dest)
                else:
                    shutil.copy2(p, dest)

        _move_or_copy(live,     live_dir)
        _move_or_copy(rejected, rejected_dir)

    # ------------------------------------------------------------------ 3. Crop (only when --no-filter)
    if args.no_filter:
        if args.no_crop:
            print("\nCrop step skipped (--no-crop).")
            if not in_place and output_dir:
                print(f"  Copying {len(keep):,} accepted image(s) to {output_dir} …")
                for p in tqdm(keep, desc="Copying", unit="img"):
                    shutil.copy2(p, output_dir / p.name)
        else:
            print(f"\nCropping {len(keep):,} image(s) with padding={args.crop_padding}px …")
            if not in_place and output_dir:
                crop_stats = crop_images(keep, output_dir=output_dir, in_place=False,
                                         padding=args.crop_padding, workers=args.workers)
            else:
                crop_stats = crop_images(keep, output_dir=None, in_place=True,
                                         padding=args.crop_padding, workers=args.workers)

    # ------------------------------------------------------------------ 4. specsin
    if args.specsin and (live or rejected):
        print(f"\nUpdating specsin: {args.specsin}")
        update_specsin(args.specsin, live, rejected)

    # ------------------------------------------------------------------ 5. Manifest
    if in_place and not args.force:
        existing_done: set[str] = set()
        if manifest_path.exists():
            existing_done = set(manifest_path.read_text().splitlines())
        existing_done |= {p.name for p in keep} | {p.name for p in live} | {p.name for p in rejected}
        manifest_path.write_text("\n".join(sorted(existing_done)))

    # ------------------------------------------------------------------ 6. Summary
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    def _pct(n, total):
        return f"{100 * n / total:.1f}%" if total else "—"

    sep = "=" * 52
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(f"  Total input images      : {n_total:>6,}")
    if n_skipped:
        print(f"  Already processed       : {n_skipped:>6,}  (skipped)")
    print(f"  Newly processed         : {n_new:>6,}")
    if not args.no_filter:
        print(f"  --- Filter results ---")
        print(f"  Herbarium sheets kept   : {len(keep):>6,}  ({_pct(len(keep), n_new)})")
        print(f"  Live plant photos       : {len(live):>6,}  ({_pct(len(live), n_new)})  → live/")
        print(f"  Rejected (slides/other) : {len(rejected):>6,}  ({_pct(len(rejected), n_new)})  → rejected/")
    if not args.no_crop and crop_stats["cropped"] + crop_stats["passthrough"] > 0:
        n_crop_total = crop_stats["cropped"] + crop_stats["passthrough"] + crop_stats["failed"]
        print(f"  --- Crop results ---")
        print(f"  Border removed          : {crop_stats['cropped']:>6,}  ({_pct(crop_stats['cropped'], n_crop_total)})")
        print(f"  No border (passed thru) : {crop_stats['passthrough']:>6,}  ({_pct(crop_stats['passthrough'], n_crop_total)})")
        if crop_stats["failed"]:
            print(f"  Failed                  : {crop_stats['failed']:>6,}")
    print(f"  Time                    : {elapsed_str}")
    if output_dir:
        print(f"  Output                  : {output_dir}")
    print(sep)


if __name__ == "__main__":
    main()
