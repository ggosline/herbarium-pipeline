"""
image_checks.py — cheap "is this actually a decodable image?" pre-flight.

Shared by filter_and_crop_herbarium.py (which acts on the answer: quarantines
the file and marks specsin) and train_herbarium.py (which keeps it only as a
last-ditch guard for sources that never went through the filter step).

Why it exists
-------------
DALI's GPU decoder has no per-sample error tolerance. One undecodable file
aborts the whole pipeline with `nvImageCodec failure` and takes the training run
down mid-epoch — after the GPU hours have already been spent. Observed on
Simaroubaceae: a 59 KB `<!DOCTYPE html>` error page that GBIF served for a moved
image, saved under a `.jpg` name.

The realistic causes are both download failures rather than corrupt photography:

  * an HTML (or JSON) error page saved under an image extension, and
  * a truncated download, which stops before the JPEG end-of-image marker.

`hasfile` only records that a file exists, so neither is caught upstream.

Deliberately a header check, not a decode: decoding every sheet costs more than
the epoch it protects. It trades away detection of corruption in the middle of
an otherwise well-formed file — rare, and not what download failures produce —
for a cost low enough to run by default.
"""

import os
from concurrent.futures import ThreadPoolExecutor

_IMAGE_SIGNATURES = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\x0a", "png"),
    (b"GIF87a", "gif"), (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"), (b"MM\x00*", "tiff"),
)

# Trailing bytes searched for the JPEG FFD9 end-of-image marker. A window rather
# than the final two bytes because plenty of valid JPEGs carry appended metadata
# or padding after the marker; requiring FFD9 to be last would reject good files.
_JPEG_EOI_WINDOW = 2048


def image_defect(path) -> str | None:
    """Return a short reason string if `path` is not a decodable image, else None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
            if not head:
                return "empty file"
            kind = next((k for sig, k in _IMAGE_SIGNATURES if head.startswith(sig)), None)
            if kind is None:
                if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                    return None
                if head.lstrip()[:1] in (b"<", b"{"):
                    return "not an image (HTML/JSON — a failed download)"
                return "not an image (unrecognised header)"
            if kind == "jpeg":
                size = os.fstat(fh.fileno()).st_size
                fh.seek(max(0, size - _JPEG_EOI_WINDOW))
                if b"\xff\xd9" not in fh.read():
                    return "truncated JPEG (no end-of-image marker)"
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        return f"unreadable ({exc.__class__.__name__})"
    return None


def scan_images(paths, check_headers: bool = True) -> dict:
    """Map path → defect reason for every unusable file. I/O bound, so threads.

    check_headers=False degrades to a plain existence test, which is what the
    callers' opt-out flags select.
    """
    paths = list(paths)
    if not paths:
        return {}
    if not check_headers:
        return {p: "missing" for p in paths if not os.path.exists(p)}
    workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(workers) as ex:
        reasons = ex.map(image_defect, [str(p) for p in paths])
    return {p: r for p, r in zip(paths, reasons) if r is not None}
