"""
Herbarium Classification Pipeline — Web UI frontend.

Uses NiceGUI to provide a modern browser-based interface.

Install:
  pip install nicegui

Run:
  python herbarium_pipeline_webui.py
  # Opens automatically at http://localhost:8765
"""

import asyncio
import json
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

from nicegui import app, ui

from cloud import secrets as cloud_secrets
from cloud import state as cloud_state
from cloud.orchestrator import (
    CloudOrchestrator,
    DEFAULT_DATACENTER,
    DEFAULT_IMAGE,
    DEFAULT_VOLUME_GB,
    GPU_BY_PURPOSE,
    PodHandle,
)
from cloud.runpod_client import RunPodAPIError
from webui.widgets import (
    FilePicker,
    SourcesPanel,
    _accordion,
    _inline,
    _path_input,
    _pill,
    _section,
    _set_pill,
    _setup_card,
    _text_row,
    _v,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
SCRIPTS = {
    "download":        _HERE / "download_gbif_images.py",
    "filter_and_crop": _HERE / "filter_and_crop_herbarium.py",
    "resize":          _HERE / "resize_images.py",
    "train":           _HERE / "train_herbarium.py",
    "identify":        _HERE / "identify_herbarium.py",
}

TIMM_MODELS = [
    "vit_large_patch16_dinov3.lvd1689m",
    "convnext_base_384_in22ft1k",
    "convnext_large_384_in22ft1k",
    "efficientnet_b4",
    "resnet50",
]

CONFIG_PATH = Path.home() / ".config" / "herbarium_pipeline.json"

# ---------------------------------------------------------------------------
# Quick-ID model cache (shared across browser sessions in the same process)
# ---------------------------------------------------------------------------

_quick_id_cache: dict = {}   # keys: ckpt, model, nameslist, geo_dim, device
_qi_url_drop:   dict = {}   # set by /api/qi_fetch_url, consumed by per-client timer
_review_shared: dict = {}   # shared between review tab and /review-carousel page

# Populated by the page builder (@ui.page("/")) so the Get Started tab — which
# is built inside a separate function — can drive path setup and tab
# navigation that otherwise live only in the page-function closure. Filled in
# after the tabs and _apply_paths exist; the Get Started buttons read these at
# click time, by which point they're populated.
_page_hooks: dict = {
    "apply_paths": None,   # callable(base=None, name=None, img_folder=None)
    "goto_tab":    None,   # callable(tab_ref)
    "tab_refs":    {},     # str key -> ui.tab
}

# The seven-stage workflow, in order. (key, label, one-line description).
# key doubles as the _page_hooks["tab_refs"] lookup for the "Open" buttons.
_STEP_FLOW = [
    ("download", "① Download", "Fetch specimen images + metadata from GBIF"),
    ("clean",    "② Clean",    "Drop non-herbarium images and crop scanner borders"),
    ("train",    "③ Train",    "Fine-tune the model on a GPU pod"),
    ("identify", "④ Identify", "Run inference; flag mismatches and unknowns"),
    ("review",   "⑤ Review",   "Browse, correct, and analyse the predictions"),
    ("archive",  "⑥ Archive",  "Back up the whole project to Cloudflare R2"),
    ("publish",  "⑦ Publish",  "Push the trained model to Hugging Face"),
]

# ---------------------------------------------------------------------------
# AI-powered review filter (Claude Haiku)
# ---------------------------------------------------------------------------

def _apply_filter_spec(spec: dict, df) -> "pd.Series":
    """Safely apply a structured filter spec (returned by Claude) to a DataFrame.

    Returns a boolean Series.  Never uses eval — only whitelisted operations.
    """
    import pandas as _pd

    t = spec.get("type", "")
    col = spec.get("column", "")
    val = spec.get("value", "")

    def _col(name: str):
        return df[name].astype(str).str.strip() if name in df.columns else None

    def _has_label(name: str):
        """True where df[name] is a real label. Must use notna(): astype(str)
        leaves float NaN (not the string "nan") for missing cells, so a
        `.ne("nan")` guard silently lets label-less indets through."""
        if name not in df.columns:
            return _pd.Series(False, index=df.index)
        raw = df[name]
        s = raw.astype(str).str.strip().str.lower()
        return raw.notna() & ~s.isin(("", "nan", "none", "<na>", "na", "null"))

    if t == "all":
        return _pd.Series(True, index=df.index)

    def _sp_col():
        """Return the predicted-species column (Series), preferring pred_species."""
        s = _col("pred_species")
        return s if s is not None else _col("top1_name")

    if t == "genus_match":
        sp = _sp_col()
        if sp is not None:
            return sp.str.split().str[0].str.lower() == str(val).lower().strip()

    if t == "species_match":
        sp = _sp_col()
        if sp is not None:
            return sp.str.lower() == str(val).lower().strip()

    if t == "true_genus_match":
        ts = _col("true_species")
        if ts is not None:
            return ts.str.split().str[0].str.lower() == str(val).lower().strip()

    if t == "true_species_match":
        ts = _col("true_species")
        if ts is not None:
            return ts.str.lower() == str(val).lower().strip()

    if t == "column_contains":
        s = _col(col)
        if s is not None:
            return s.str.lower().str.contains(str(val).lower(), na=False, regex=False)

    if t == "column_match":
        s = _col(col)
        if s is not None:
            return s.str.lower() == str(val).lower().strip()

    if t == "column_compare":
        op = spec.get("op", "")
        if col in df.columns:
            try:
                num = df[col].astype(float)
                v   = float(val)
            except (TypeError, ValueError):
                return _pd.Series(True, index=df.index)
            ops = {"<": num.lt, "<=": num.le, ">": num.gt,
                   ">=": num.ge, "==": num.eq, "!=": num.ne}
            fn = ops.get(op)
            if fn:
                return fn(v)

    if t == "value_count":
        # Filter rows where a column's value appears </>/>= N times in the dataset.
        # e.g. {"type":"value_count","column":"pred_species","op":"<","value":5}
        target = col or "pred_species"
        op_str = spec.get("op", "<")
        if target in df.columns:
            counts = df[target].map(df[target].value_counts())
            try:
                v = float(val)
            except (TypeError, ValueError):
                return _pd.Series(True, index=df.index)
            ops = {"<": counts.lt, "<=": counts.le, ">": counts.gt,
                   ">=": counts.ge, "==": counts.eq, "!=": counts.ne}
            fn = ops.get(op_str)
            if fn:
                return fn(v)

    if t == "top5_none_correct":
        true = _col("true_species")
        if true is not None:
            mask = _has_label("true_species")
            for k in range(1, 6):
                c = f"top{k}_name"
                if c in df.columns:
                    mask = mask & (df[c].astype(str).str.strip() != true)
            return mask

    if t == "top1_wrong":
        sp = _sp_col()
        true = _col("true_species")
        if sp is not None and true is not None:
            return _has_label("true_species") & (true != sp)

    if t == "compound":
        logic  = spec.get("logic", "and")
        masks  = [_apply_filter_spec(f, df) for f in spec.get("filters", [])]
        masks  = [m for m in masks if m is not None]
        if not masks:
            return _pd.Series(True, index=df.index)
        import functools, operator
        op = operator.and_ if logic == "and" else operator.or_
        return functools.reduce(op, masks)

    # Unknown type → no filter
    return _pd.Series(True, index=df.index)


async def _ai_build_filter(query: str, df) -> dict | None:
    """Ask Claude Haiku to turn a natural-language filter into a spec dict."""
    try:
        import anthropic
    except ImportError:
        return None

    cols = list(df.columns)
    sample = df.iloc[0].to_dict() if len(df) else {}
    # Truncate long values for the prompt
    sample = {k: (str(v)[:80] if len(str(v)) > 80 else v) for k, v in sample.items()}

    prompt = (
        "You are a data-filter assistant for a herbarium specimen predictions CSV.\n"
        f"Columns: {cols}\n"
        f"Sample row (truncated): {json.dumps(sample, default=str)}\n\n"
        "Species names follow botanical convention: 'Genus epithet' "
        "(e.g. 'Uvaria chamae').  The genus is the first word.\n"
        "IMPORTANT: pred_species/top1_name = model PREDICTION.  "
        "true_species = the actual/known/correct species.  "
        "When the user says 'true genus', 'actual genus', or 'known species', "
        "use true_genus_match or true_species_match (NOT genus_match).\n\n"
        f'User\'s filter request: "{query}"\n\n'
        "Return ONLY a JSON object using one of these types:\n"
        '  {"type":"all"}                                     — no filter\n'
        '  {"type":"genus_match","value":"<Genus>"}            — genus is first word of pred_species\n'
        '  {"type":"species_match","value":"<Genus epithet>"}  — exact predicted species match\n'
        '  {"type":"true_genus_match","value":"<Genus>"}         — match true (actual) genus from true_species column\n'
        '  {"type":"true_species_match","value":"<Genus epithet>"} — match true (actual) species\n'
        '  {"type":"column_contains","column":"<col>","value":"<text>"}\n'
        '  {"type":"column_match","column":"<col>","value":"<text>"}\n'
        '  {"type":"column_compare","column":"<col>","op":"</<=/>/>=","value":<number>}\n'
        '  {"type":"value_count","column":"<col>","op":"</<=/>/>=","value":<N>}  — rows where the column value occurs <N times in the dataset (rare/sparse items)\n'
        '  {"type":"top5_none_correct"}                        — true species not in any top-5\n'
        '  {"type":"top1_wrong"}                               — top-1 ≠ true species\n'
        '  {"type":"compound","logic":"and"|"or","filters":[...]}  — combine filters\n'
        "Return ONLY the JSON, no markdown fences, no explanation."
    )

    client = anthropic.Anthropic()
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        ),
    )
    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _url_fetch_headers(url: str) -> dict:
    """Return browser-like headers for fetching an image URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}/"
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }


@app.get("/api/qi_fetch_url")
async def _qi_fetch_url_handler(url: str):
    """Fetch a remote image URL for Quick-ID drag-and-drop from web pages."""
    import base64 as _b64
    import tempfile as _tmp
    import urllib.request as _urlreq
    from fastapi.responses import JSONResponse
    try:
        req = _urlreq.Request(url, headers=_url_fetch_headers(url))
        with _urlreq.urlopen(req, timeout=15) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        ext = ct.split("/")[-1].replace("jpeg", "jpg") or "jpg"
        tmp = _tmp.mktemp(suffix=f".{ext}")
        with open(tmp, "wb") as f:
            f.write(data)
        b64 = _b64.b64encode(data).decode()
        _qi_url_drop["latest"] = {"tmp": tmp, "data_url": f"data:{ct};base64,{b64}"}
        return JSONResponse({"ok": True})
    except Exception as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)


def _qi_infer(ckpt_path: str, image_path: str,
              lat_str: str, lon_str: str,
              model_name_hint: str = "") -> list[tuple[str, float]]:
    """Load (and cache) a checkpoint, run inference on one image.

    Returns [(species_name, probability), …] top-5.
    Runs in a thread executor — no UI calls allowed here.
    """
    import torch
    import torch.nn as nn
    import timm as _timm
    import sys as _sys
    _sys.path.insert(0, str(_HERE))
    from identify_herbarium import (load_model, encode_coords,
                                    _GeoModel, InferenceDataset)
    from torch.utils.data import DataLoader

    cache = _quick_id_cache
    if cache.get("ckpt") != ckpt_path:
        (state_dict, model_name, num_classes, nameslist, geo_dim, _label_level,
         temperature, _excluded, _class_counts, _genus_head,
         _split) = load_model(Path(ckpt_path), [], 640)
        if not model_name:
            model_name = model_name_hint.strip()
        if not model_name:
            raise ValueError(
                "Cannot determine model architecture from checkpoint. "
                "Set the model name in the Quick ID panel (e.g. vit_large_patch16_dinov3.lvd1689m).")
        if geo_dim:
            backbone = _timm.create_model(model_name, pretrained=False, num_classes=0)
            feat_dim = backbone.num_features
            geo_mlp = nn.Sequential(
                nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim))
            head = nn.Linear(feat_dim + geo_dim, num_classes)
            backbone.load_state_dict(
                {k: v for k, v in state_dict.items()
                 if not k.startswith(("geo_mlp.", "head."))}, strict=False)
            geo_mlp.load_state_dict(
                {k[len("geo_mlp."):]: v for k, v in state_dict.items()
                 if k.startswith("geo_mlp.")})
            head.load_state_dict(
                {k[len("head."):]: v for k, v in state_dict.items()
                 if k.startswith("head.")})
            model = _GeoModel(backbone, geo_mlp, head, geo_dim)
        else:
            model = _timm.create_model(model_name, pretrained=False,
                                       num_classes=num_classes)
            model.load_state_dict(state_dict, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        cache.update(ckpt=ckpt_path, model=model, nameslist=nameslist,
                     geo_dim=geo_dim, device=device, temperature=temperature)
        print(f"[Quick ID] Model loaded: {model_name}, {num_classes} classes, device={device}")

    model       = cache["model"]
    nameslist   = cache["nameslist"]
    geo_dim     = cache["geo_dim"]
    device      = cache["device"]
    temperature = cache.get("temperature", 1.0) or 1.0

    geo_coords = None
    try:
        import numpy as _np
        lat = float(lat_str)
        lon = float(lon_str)
        geo_coords = encode_coords([lat], [lon])
    except (TypeError, ValueError):
        pass

    ds = InferenceDataset([Path(image_path)], 640, geo_coords)
    loader = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=False)

    with torch.inference_mode():
        for imgs, _, geo in loader:
            imgs = imgs.to(device)
            if geo_coords is not None and geo_dim:
                logits = model(imgs, geo.to(device))
            else:
                logits = model(imgs)
            if temperature != 1.0:
                logits = logits / temperature
            probs  = torch.softmax(logits, dim=1)[0]
            top5   = torch.topk(probs, k=min(5, len(probs)))
            return [(nameslist[i], float(p))
                    for i, p in zip(top5.indices.tolist(), top5.values.tolist())]
    return []


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}

def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

# ---------------------------------------------------------------------------
# Process management  (module-level — single-user local app)
# ---------------------------------------------------------------------------

_proc: Optional[asyncio.subprocess.Process] = None
_pipeline: list = []   # list of (name, cmd_fn) pairs

# These are set during UI construction
_log:      Optional[ui.log]    = None
_status:   Optional[ui.label]  = None
_stop_btn: Optional[ui.button] = None

# W&B run link — a clickable chip in the output-panel header that appears the
# moment a run URL is spotted in the training output, so it isn't lost in the
# hundreds of streamed lines. _wandb_url holds the current URL for the click
# handler. Set during UI construction.
_wandb_link: Optional[ui.button] = None
_wandb_url:  list[str]           = [""]
_WANDB_RE = re.compile(r"https?://(?:[\w.-]+\.)?wandb\.ai/[^\s]+?/runs/[\w-]+")


def _scan_wandb(text: str) -> None:
    """Spot a W&B run URL in a streamed output line and surface it in the
    output-panel header. Cheap short-circuit so it can run on every line."""
    if _wandb_link is None or not text or "wandb.ai" not in text or "/runs/" not in text:
        return
    m = _WANDB_RE.search(text)
    if not m:
        return
    url = m.group(0).rstrip(".,);]'\"")
    if _wandb_url[0] == url:
        return
    _wandb_url[0] = url
    try:
        _wandb_link.set_visibility(True)
        _wandb_link.tooltip(url)
    except RuntimeError:
        pass  # client navigated away


def _reset_wandb() -> None:
    """Hide the W&B run chip and forget its URL — called when a new training
    run starts so a stale link from the previous run can't be clicked."""
    _wandb_url[0] = ""
    if _wandb_link is not None:
        try:
            _wandb_link.set_visibility(False)
        except RuntimeError:
            pass  # client navigated away

# Review images are served through ONE route registered at import (below), not
# by mounting each folder on demand. NiceGUI 3.x's app.add_static_files() adds
# an ASGI mount, which must happen before the server starts; calling it lazily
# while showing the first image (after startup) raised at the ASGI layer, dropped
# the socket, and reloaded the page to the start tab. A pre-registered endpoint
# that streams a file by path sidesteps that and handles large scans by streaming.
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
# Directories the review flow has explicitly surfaced — the serve route refuses
# anything outside them, so this isn't an open read-any-file endpoint.
_review_roots: set[str] = set()


@app.get("/review_file")
def _serve_review_file(p: str):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    try:
        rp = Path(p).resolve()
    except OSError:
        raise HTTPException(status_code=404)
    if rp.suffix.lower() not in _IMG_EXTS or not rp.is_file():
        raise HTTPException(status_code=404)
    if not any(rp.is_relative_to(root) for root in _review_roots):
        raise HTTPException(status_code=403)
    return FileResponse(str(rp))


def _review_img_url(abs_path: str) -> str:
    """Return a served URL for abs_path via the /review_file endpoint."""
    from urllib.parse import quote
    p = Path(abs_path)
    if not p.is_file():
        return ""
    _review_roots.add(str(p.parent.resolve()))
    return f"/review_file?p={quote(str(p.resolve()))}"


def _merge_aum(df, review_dir: Path) -> None:
    """Attach an ``aum`` column to the predictions frame, in place.

    AUM (Area Under the Margin) ranks *training* specimens by how likely their
    recorded label is wrong — the lower, the more suspect. It is embedded in the
    checkpoint, not the predictions, so it reaches the Review tab one of two
    ways: baked into predictions.csv by a recent identify run (nothing to do
    here), or as a sibling ``aum.csv`` (fname, aum, …) produced cheaply by
    ``aum_candidates.py`` straight from the checkpoint. Merging on the file
    basename keeps it robust to differing path prefixes between the two files.
    """
    import pandas as _pd  # lazy, matching the rest of this module (slim install)
    if "aum" in df.columns:
        return
    sidecar = review_dir / "aum.csv"
    if not sidecar.is_file():
        return
    try:
        aum = _pd.read_csv(sidecar, usecols=lambda c: c in ("fname", "aum"))
        if "fname" not in aum.columns or "aum" not in aum.columns:
            return
        aum["_base"] = aum["fname"].map(lambda f: Path(str(f)).name)
        aum = aum.drop_duplicates("_base").set_index("_base")["aum"]
        df["aum"] = df["fname"].map(lambda f: Path(str(f)).name).map(aum)
    except Exception as exc:  # a malformed sidecar must not break loading
        print(f"[review] could not merge aum.csv: {exc}")


async def _launch(cmd: list[str], on_done=None, extra_env: dict | None = None) -> None:
    global _proc
    if not cmd:
        return
    if _proc and _proc.returncode is None:
        ui.notify("A step is already running — click Stop to cancel it first "
                  "(a browser refresh can leave one running in the background).",
                  type="warning", timeout=6000)
        return

    _log.push(f"\n$ {shlex.join(cmd)}\n")
    _stop_btn.enable()
    _status.set_text("Running…")

    import os as _os
    proc_env = {**_os.environ, **(extra_env or {})}

    _proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=proc_env,
    )

    # Read in chunks so \r-terminated lines (tqdm progress bars) also arrive.
    # Normalise \r\n and lone \r to \n before pushing to the log widget.
    def _push(text: str) -> None:
        try:
            _log.push(text)
            _scan_wandb(text)
        except RuntimeError:
            pass  # client navigated away

    _buf = ""
    while True:
        chunk = await _proc.stdout.read(512)
        if not chunk:
            break
        _buf += chunk.decode(errors="replace")
        _buf = _buf.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in _buf:
            line, _buf = _buf.split("\n", 1)
            if line:
                _push(line + "\n")
    if _buf:
        _push(_buf + "\n")

    await _proc.wait()
    rc = _proc.returncode
    try:
        _log.push(f"\n[{'Finished OK' if rc == 0 else f'Exited with code {rc}'}]\n")
        _status.set_text("Finished OK" if rc == 0 else f"Exited {rc}")
        _stop_btn.disable()
    except RuntimeError:
        pass  # client navigated away

    if on_done:
        await on_done(rc)


async def _run_pipeline() -> None:
    global _pipeline
    if not _pipeline:
        _log.push("\n✓ Pipeline complete.\n")
        _status.set_text("Pipeline complete")
        return
    name, cmd_fn = _pipeline.pop(0)
    try:
        cmd = cmd_fn()
    except ValueError as exc:
        ui.notify(str(exc), type="negative")
        _pipeline.clear()
        return
    _log.push(f"\n{'='*60}\nStep: {name}\n{'='*60}\n")

    async def _step_done(rc: int) -> None:
        if rc != 0:
            _log.push("\n✗ Step failed — pipeline aborted.\n")
            _pipeline.clear()
        else:
            await _run_pipeline()

    await _launch(cmd, on_done=_step_done)


def _stop_process() -> None:
    global _pipeline
    if _proc and _proc.returncode is None:
        _proc.terminate()
    _pipeline.clear()
    if _status:
        _status.set_text("Stopped")


def _quit() -> None:
    _stop_process()
    app.shutdown()


@app.on_shutdown
def _on_shutdown() -> None:
    if _proc and _proc.returncode is None:
        _proc.kill()


# ---------------------------------------------------------------------------
# Tab builders — each returns a cmd-builder callable (or None for Run All).
# Reusable widgets and helpers (_section, _path_input, FilePicker,
# SourcesPanel, etc.) live in webui/widgets.py.
# ---------------------------------------------------------------------------

def _build_download() -> callable:
    gs = app.storage.general
    _section("Taxon")
    rank = (ui.radio({"family": "Family", "genus": "Genus", "order": "Order"},
                     value="family").props("inline dense")
            .bind_value(gs, "dl_rank"))
    taxon     = _text_row("Taxon name:", "Ebenaceae", "w-48").bind_value(gs, "dl_taxon")
    families  = (_text_row("Families (multi):", "", "w-full")
                 .bind_value(gs, "dl_families"))
    ui.label("Space-separated list for a combined GBIF bulk download (e.g. split clades). "
             "Overrides Taxon name. Cloud mode only — requires GBIF credentials in Get Started tab."
             ).classes("text-caption text-grey-7 ml-48")
    continent = _text_row("Continent:", "AFRICA", "w-36").bind_value(gs, "dl_continent")

    _section("Country filter  (mutually exclusive)")
    inc = _text_row("Include countries:", "", "w-60").bind_value(gs, "dl_inc")
    exc = _text_row("Exclude countries:", "MG", "w-60").bind_value(gs, "dl_exc")
    ui.label("Space-separated ISO-2 codes, e.g. ZA NG TZ").classes("text-caption text-grey-7 ml-48")

    _section("Source  (DwC-A ZIP or live API)")
    dwca = (_path_input("Local DwC-A ZIP:", mode="file",
                        hint="Select a downloaded GBIF DwC-A ZIP to skip the API")
            .bind_value(gs, "dl_dwca"))

    # Output paths are local-only; cloud writes to /workspace/data on the pod.
    with _local_only(ui.column().classes("w-full")):
        _section("Output")
        out_dir = _path_input("Output images dir:", mode="dir").bind_value(gs, "dl_out_dir")
        specsin = _path_input("specsin CSV path:", mode="save").bind_value(gs, "dl_specsin")

    with ui.row().classes("w-full items-center gap-4 flex-wrap mt-1"):
        with ui.row().classes("items-center gap-1"):
            ui.label("Workers:").classes("text-sm")
            workers = ui.input(value="8").classes("w-16").props("dense outlined").bind_value(gs, "dl_workers")
        with ui.row().classes("items-center gap-1"):
            ui.label("Limit (0=all):").classes("text-sm")
            limit = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "dl_limit")
        with ui.row().classes("items-center gap-1"):
            ui.label("IIIF size:").classes("text-sm")
            iiif = (ui.select(["", "1024", "2048", "4096", "max"], value="")
                    .props("dense outlined").classes("w-24")
                    .bind_value(gs, "dl_iiif"))
            ui.label("px (blank = GBIF default)").classes("text-c aption text-grey-7")
        with ui.row().classes("items-center gap-1"):
            ui.label("Resize on download (0=off):").classes("text-sm")
            max_size = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "dl_max_size")
            ui.label("px — shrinks longer side to ≤ N px using PIL (saves disk space)").classes(
                "text-caption text-grey-7")
        with ui.row().classes("items-center gap-1"):
            ui.label("Max per species (0=all):").classes("text-sm")
            max_per_sp = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "dl_max_per_sp")
            ui.label("random subsample per species").classes("text-caption text-grey-7")
        with ui.row().classes("items-center gap-1"):
            ui.label("Max per genus (0=all):").classes("text-sm")
            max_per_ge = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "dl_max_per_ge")
            ui.label("random subsample per genus").classes("text-caption text-grey-7")
        with ui.row().classes("items-center gap-1"):
            ui.label("Max per family (0=all):").classes("text-sm")
            max_per_fa = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "dl_max_per_fa")
            ui.label("stratified — ≥1 per genus, remainder random").classes("text-caption text-grey-7")

        with ui.row().classes("items-center gap-3 mt-1"):
            specsin_only = ui.checkbox("Specsin only (no download)", value=False)\
                .bind_value(gs, "dl_specsin_only")
        with ui.row().classes("items-center gap-1"):
            ui.label("From specsin (optional):").classes("text-sm")
            from_specsin = ui.input(value="").classes("w-72").props("dense outlined")\
                .bind_value(gs, "dl_from_specsin")
            ui.label("use specsin CSV as source instead of DwC-A/API").classes(
                "text-caption text-grey-7")

    ui.button("Run Download", icon="download",
              on_click=lambda: _run_step_mode_aware(
                  "download", _dl_cmd, cloud_env_fn=_cloud_env_download)
              ).props("color=primary unelevated").classes("mt-4")

    # Cloud seeding: push a locally-prepared DwC-A ZIP or specsin.csv to the
    # pod, as an alternative to fetching from GBIF on the pod itself. Cloud
    # mode only. (These used to be a "mystery" Upload menu in the header.)
    with _cloud_only(ui.row().classes("w-full items-center gap-2 mt-2")):
        ui.label("Send to pod:").classes("text-sm text-grey-7 shrink-0")
        ui.button("Upload DwC-A", icon="archive",
                  on_click=lambda: _wrap_cloud_aux(_do_upload_dwca))\
            .props("outlined dense color=primary")\
            .tooltip("Upload the local DwC-A ZIP selected above to the pod.")
        ui.button("Upload specsin", icon="upload_file",
                  on_click=lambda: _wrap_cloud_aux(_do_upload_specsin))\
            .props("outlined dense color=primary")\
            .tooltip("Upload a local specsin CSV to the pod. Set the remote "
                     "path in ☁ Cloud → From specsin.")

    def _dl_cmd() -> list[str]:
        d  = _v(dwca)
        t  = _v(taxon)
        ff = _v(families).split()
        if not d and not t and not ff:
            raise ValueError("Enter a taxon name, a families list, or select a DwC-A ZIP.")
        cmd = [sys.executable, str(SCRIPTS["download"])]
        if d:  cmd += ["--dwca", d]
        elif ff: cmd += ["--families"] + ff
        elif t: cmd += [f"--{rank.value}", t]
        c = _v(continent)
        if c: cmd += ["--continent", c]
        i_  = _v(inc).split()
        ex_ = _v(exc).split()
        if i_ and ex_:
            raise ValueError("Use Include or Exclude countries — not both.")
        if i_: cmd += ["--countries"] + i_
        if ex_: cmd += ["--exclude-countries"] + ex_
        od = _v(out_dir)
        sp = _v(specsin)
        if od: cmd += ["--output-dir", od]
        if sp: cmd += ["--specsin", sp]
        w = _v(workers)
        if w: cmd += ["--workers", w]
        lm = _v(limit)
        if lm and lm != "0": cmd += ["--limit", lm]
        iz = _v(iiif)
        if iz: cmd += ["--iiif-size", iz]
        ms = _v(max_size)
        if ms and ms != "0": cmd += ["--max-size", ms]
        mps = _v(max_per_sp)
        if mps and mps != "0": cmd += ["--max-per-species", mps]
        mpg = _v(max_per_ge)
        if mpg and mpg != "0": cmd += ["--max-per-genus", mpg]
        mpf = _v(max_per_fa)
        if mpf and mpf != "0": cmd += ["--max-per-family", mpf]
        if specsin_only.value:
            cmd += ["--specsin-only"]
        fs = _v(from_specsin)
        if fs:
            cmd += ["--from-specsin", fs]
        return cmd

    return _dl_cmd, out_dir, specsin


def _build_filter_crop() -> callable:
    gs = app.storage.general
    # Cloud notice — on the pod this maps to the bootstrap "prep" step, which
    # does filter + crop + resize together with hardcoded defaults.
    with _cloud_only(ui.column().classes("w-full")):
        ui.label(
            "Cloud mode: this tab's Run button executes the bootstrap 'prep' "
            "step on the pod (filter + crop + resize 1024 px). "
            "The Method dropdown affects the cloud run; path / numeric fields "
            "below configure local runs only."
        ).classes("text-body2").style(
            "background:#f0f7f6;border-left:3px solid #00897b;padding:8px 12px;"
            "border-radius:0 4px 4px 0;color:#37474f;max-width:1000px;margin-bottom:6px")
    # All paths are local-only — on the pod prep reads from images_raw and
    # writes images_filtered/images_1024 by convention.
    with _local_only(ui.column().classes("w-full")):
        _section("Paths")
        inp_dir = _path_input("Input images dir:", mode="dir").bind_value(gs, "fc_inp_dir")
        out_dir = _path_input("Output images dir:", mode="dir").bind_value(gs, "fc_out_dir")
        ui.label("Set same as input to overwrite in-place").classes(
            "text-caption text-grey-7 ml-48")
        fc_spec = _path_input("specsin CSV (optional):", mode="file").bind_value(gs, "fc_spec")

    _section("Steps")
    with ui.row().classes("gap-6"):
        do_filter = ui.checkbox("Filter non-herbarium images", value=True).bind_value(gs, "fc_do_filter")
        do_crop   = ui.checkbox("Crop white borders", value=True).bind_value(gs, "fc_do_crop")

    _section("Filter options")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        with ui.row().classes("items-center gap-1"):
            ui.label("Method:").classes("text-sm")
            method = (ui.select(["clip", "hsv"], value="clip").props("dense outlined").classes("w-24")
                      .bind_value(gs, "fc_method"))
        with ui.row().classes("items-center gap-1"):
            ui.label("Confidence:").classes("text-sm")
            conf = ui.input(value="0.6").classes("w-20").props("dense outlined").bind_value(gs, "fc_conf")
        with ui.row().classes("items-center gap-1"):
            ui.label("HSV white ratio:").classes("text-sm")
            hsv_w = ui.input(value="0.25").classes("w-20").props("dense outlined").bind_value(gs, "fc_hsv_w")
        with ui.row().classes("items-center gap-1"):
            ui.label("HSV saturation:").classes("text-sm")
            hsv_s = ui.input(value="40").classes("w-20").props("dense outlined").bind_value(gs, "fc_hsv_s")

    _section("Crop / performance")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        with ui.row().classes("items-center gap-1"):
            ui.label("Crop padding (px):").classes("text-sm")
            padding = ui.input(value="10").classes("w-16").props("dense outlined").bind_value(gs, "fc_padding")
        with ui.row().classes("items-center gap-1"):
            ui.label("Batch size:").classes("text-sm")
            batch = ui.input(value="32").classes("w-16").props("dense outlined").bind_value(gs, "fc_batch")
        with ui.row().classes("items-center gap-1"):
            ui.label("Workers:").classes("text-sm")
            fc_workers = ui.input(value="8").classes("w-16").props("dense outlined").bind_value(gs, "fc_workers")
    force = ui.checkbox("Force reprocess (ignore already-processed images)", value=False).bind_value(gs, "fc_force")

    ui.button("Run Filter & Crop", icon="filter_alt",
              on_click=lambda: _run_step_mode_aware("prep", _fc_cmd,
                                                    cloud_env_fn=_cloud_env_prep)
              ).props("color=primary unelevated").classes("mt-4")\
              .tooltip("Cloud mode: runs the bootstrap 'prep' step (filter + crop "
                       "+ resize with hardcoded defaults). Configure here only "
                       "matters for local runs.")

    def _fc_cmd() -> list[str]:
        i = _v(inp_dir)
        if not i: raise ValueError("Enter an input directory.")
        o = _v(out_dir)
        cmd = [sys.executable, str(SCRIPTS["filter_and_crop"]), "--input-dir", i]
        if o and Path(o).resolve() == Path(i).resolve():
            cmd += ["--in-place"]
        elif o:
            cmd += ["--output-dir", o]
        else:
            cmd += ["--in-place"]
        if not do_filter.value:
            cmd += ["--no-filter"]
        else:
            cmd += ["--filter-method", method.value,
                    "--confidence", conf.value,
                    "--hsv-white-ratio", hsv_w.value,
                    "--hsv-saturation", hsv_s.value,
                    "--batch-size", batch.value]
        if not do_crop.value:
            cmd += ["--no-crop"]
        else:
            cmd += ["--crop-padding", padding.value]
        cmd += ["--workers", fc_workers.value]
        if force.value: cmd += ["--force"]
        sp = _v(fc_spec)
        if sp: cmd += ["--specsin", sp]
        return cmd

    return _fc_cmd, inp_dir, out_dir, fc_spec


def _build_resize() -> callable:
    gs = app.storage.general
    with _cloud_only(ui.column().classes("w-full")):
        ui.label(
            "Cloud mode: this tab's Run button executes the bootstrap 'prep' "
            "step on the pod (filter + crop + resize 1024 px). Same step as "
            "Filter & Crop. The fields below configure local runs only."
        ).classes("text-body2").style(
            "background:#f0f7f6;border-left:3px solid #00897b;padding:8px 12px;"
            "border-radius:0 4px 4px 0;color:#37474f;max-width:1000px;margin-bottom:6px")
    with _local_only(ui.column().classes("w-full")):
        _section("Paths")
        rs_inp = _path_input("Input images dir:", mode="dir").bind_value(gs, "rs_inp")
        rs_out = _path_input("Output images dir:", mode="dir").bind_value(gs, "rs_out")
        ui.label("Leave blank to resize in-place").classes("text-caption text-grey-7 ml-48")

    _section("Options")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        with ui.row().classes("items-center gap-1"):
            ui.label("Max size (px):").classes("text-sm")
            maxsz = ui.input(value="1024").classes("w-20").props("dense outlined").bind_value(gs, "rs_maxsz")
        noupscale = ui.checkbox("No upscale", value=True).bind_value(gs, "rs_noupscale")
        nodali    = ui.checkbox("Force PIL (no DALI)", value=False).bind_value(gs, "rs_nodali")
        with ui.row().classes("items-center gap-1"):
            ui.label("Batch size:").classes("text-sm")
            rs_batch = ui.input(value="8").classes("w-16").props("dense outlined").bind_value(gs, "rs_batch")
        with ui.row().classes("items-center gap-1"):
            ui.label("Workers:").classes("text-sm")
            rs_workers = ui.input(value="8").classes("w-16").props("dense outlined").bind_value(gs, "rs_workers")

    ui.button("Run Resize", icon="photo_size_select_large",
              on_click=lambda: _run_step_mode_aware("prep", _rs_cmd,
                                                    cloud_env_fn=_cloud_env_prep)
              ).props("color=primary unelevated").classes("mt-4")\
              .tooltip("Cloud mode: runs the bootstrap 'prep' step (filter + "
                       "crop + resize 1024 px). Same step as Filter & Crop.")

    def _rs_cmd() -> list[str]:
        i = _v(rs_inp)
        if not i: raise ValueError("Enter an input directory.")
        cmd = [sys.executable, str(SCRIPTS["resize"]), "--input-dir", i]
        o = _v(rs_out)
        cmd += ["--output-dir", o] if o else ["--in-place"]
        cmd += ["--max-size", maxsz.value]
        if noupscale.value: cmd += ["--no-upscale"]
        if nodali.value:    cmd += ["--no-dali"]
        cmd += ["--batch-size", rs_batch.value, "--workers", rs_workers.value]
        return cmd

    return _rs_cmd, rs_inp


def _build_train() -> tuple:
    gs = app.storage.general
    with _cloud_only(ui.column().classes("w-full")):
        ui.label(
            "Cloud mode: training runs on the pod using "
            "/workspace/data/specsin.csv : /workspace/data/images_1024 "
            "(set up by the Download/Prep steps). Every other knob below "
            "still applies — model, image size, batch sizes, schedule, etc. "
            "If the active pod is 'light', Run will auto-upgrade to a train pod."
        ).classes("text-body2").style(
            "background:#f0f7f6;border-left:3px solid #00897b;padding:8px 12px;"
            "border-radius:0 4px 4px 0;color:#37474f;max-width:1000px;margin-bottom:6px")
    with _local_only(ui.column().classes("w-full")):
        _section("Data sources  (specsin CSV : images directory)")
        tr_sources = SourcesPanel("train_sources")

        _section("Output")
        tr_out = _path_input("Output / run dir:", mode="dir").bind_value(gs, "tr_out")

    with _accordion("Model & batch size", opened=True):
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("timm model:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
            tr_model = (ui.input(value=TIMM_MODELS[0],
                                 placeholder="timm model name")
                        .props("dense outlined clearable")
                        .classes("flex-1")
                        .bind_value(gs, "tr_model"))
            with ui.menu() as _model_menu:
                for _m in TIMM_MODELS:
                    ui.menu_item(_m, on_click=lambda _, m=_m: tr_model.set_value(m))
            ui.button(icon="arrow_drop_down", on_click=_model_menu.open).props("flat dense")

        with ui.row().classes("w-full items-center gap-4 flex-wrap mt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Image size (px):").classes("text-sm")
                tr_imgsz = ui.input(value="640").classes("w-20").props("dense outlined").bind_value(gs, "tr_imgsz")
            with ui.row().classes("items-center gap-1"):
                ui.label("Batch size:").classes("text-sm")
                tr_batch = ui.input(value="4").classes("w-16").props("dense outlined").bind_value(gs, "tr_batch")
                ui.tooltip("Stage 1 batch size (backbone frozen). Can be larger than stage 2.")
            with ui.row().classes("items-center gap-1"):
                ui.label("Stage 2 batch (0=same):").classes("text-sm")
                tr_s2_batch = ui.input(value="0").classes("w-16").props("dense outlined").bind_value(gs, "tr_s2_batch")
                ui.tooltip("Override batch size for stage 2 (full fine-tune). Use a smaller value if stage 2 runs out of VRAM. 0 = use the same batch size as stage 1.")
            with ui.row().classes("items-center gap-1"):
                ui.label("Grad accum:").classes("text-sm")
                tr_accum = ui.input(value="2").classes("w-16").props("dense outlined").bind_value(gs, "tr_accum")
            with ui.row().classes("items-center gap-1"):
                ui.label("GPUs:").classes("text-sm")
                tr_gpus = ui.input(value="2").classes("w-16").props("dense outlined").bind_value(gs, "tr_gpus")
            with ui.row().classes("items-center gap-1"):
                ui.label("Max images per class (0=all):").classes("text-sm")
                tr_max_per_sp = (ui.input(value="0").classes("w-20").props("dense outlined")
                                 .tooltip("Caps images per CLASS at the rank you are training. "
                                          "On a genus model this caps each genus (a big genus is "
                                          "sampled round-robin across its species, so it keeps its "
                                          "morphological breadth). This is the best way to tame a "
                                          "long tail — it balances the data instead of distorting "
                                          "the loss, and cuts training time in proportion. "
                                          "For Rubiaceae genera, 300 takes the imbalance from 552x "
                                          "to 15x and trains 3x faster.")
                                 .bind_value(gs, "tr_max_per_sp"))
            with ui.row().classes("items-center gap-1"):
                ui.label("Min images per class:").classes("text-sm")
                tr_sparse = (ui.input(value="20").classes("w-16").props("dense outlined")
                             .tooltip("Taxa with fewer images than this are dropped from training "
                                      "(listed in excluded_species.csv). A class with only a handful "
                                      "of images cannot be learned, but still competes for every "
                                      "prediction. 20-30 is a sane floor; 5 lets near-empty classes in.")
                             .bind_value(gs, "tr_sparse"))
            with ui.row().classes("items-center gap-1"):
                ui.label("Rare-class boost (beta):").classes("text-sm")
                tr_cw_beta = (ui.input(value="0.0").classes("w-16").props("dense outlined")
                              .tooltip("How hard to up-weight rare taxa in the loss. "
                                       "0 = off (recommended). Leave it at 0 and get your "
                                       "rare-taxa boost on the Identify tab instead ('Common/rare "
                                       "bias' — negative values), which is the same dial but free "
                                       "and tunable after training, and doesn't distort what the "
                                       "backbone learns. "
                                       "1.0 = full inverse-frequency: on a long-tailed flora this "
                                       "backfires badly — near-empty classes soak up predictions "
                                       "from your commonest genera.")
                              .bind_value(gs, "tr_cw_beta"))
            nccl_p2p_disable = (ui.checkbox(
                "NCCL_P2P_DISABLE (only for multi-GPU without NVLink)", value=False)
                .tooltip("Sets NCCL_P2P_DISABLE=1 — do NOT enable if NVLink is present")
                .bind_value(gs, "tr_nccl_p2p"))

    with _accordion("Schedule (epochs, learning rates, cool-down)", opened=True):
        with ui.row().classes("w-full items-center gap-4 flex-wrap"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Stage 1 epochs:").classes("text-sm")
                s1ep = ui.input(value="4").classes("w-20").props("dense outlined").bind_value(gs, "tr_s1ep")
            with ui.row().classes("items-center gap-1"):
                ui.label("Stage 1 LR:").classes("text-sm")
                s1lr = ui.input(value="0.005").classes("w-24").props("dense outlined").bind_value(gs, "tr_s1lr")
            with ui.row().classes("items-center gap-1"):
                ui.label("Stage 2 epochs:").classes("text-sm")
                s2ep = ui.input(value="15").classes("w-20").props("dense outlined").bind_value(gs, "tr_s2ep")
            with ui.row().classes("items-center gap-1"):
                ui.label("Stage 2 LR:").classes("text-sm")
                s2lr = ui.input(value="0.0001").classes("w-24").props("dense outlined").bind_value(gs, "tr_s2lr")

        with ui.row().classes("w-full items-center gap-4 flex-wrap"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Cool-down epochs (0=off):").classes("text-sm")
                cd_ep = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "tr_cd_ep")
            with ui.row().classes("items-center gap-1"):
                ui.label("Cool-down LR:").classes("text-sm")
                cd_lr = ui.input(value="0.0001").classes("w-24").props("dense outlined").bind_value(gs, "tr_cd_lr")
            with ui.row().classes("items-center gap-1"):
                ui.label("Cool-down batch:").classes("text-sm")
                cd_batch = ui.input(value="5").classes("w-16").props("dense outlined").bind_value(gs, "tr_cd_batch")
            with ui.row().classes("items-center gap-1"):
                ui.label("Cool-down accum:").classes("text-sm")
                cd_accum = ui.input(value="2").classes("w-16").props("dense outlined").bind_value(gs, "tr_cd_accum")
        with ui.row().classes("w-full items-center gap-4 flex-wrap mt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Early-stop patience (0=auto):").classes("text-sm")
                tr_es_pat = (ui.input(value="0").classes("w-16").props("dense outlined")
                             .tooltip("Stop when validation accuracy hasn't improved for this many "
                                      "epochs. 0 = auto (2 in stage 2). Early stopping watches "
                                      "ACCURACY, the same metric the best checkpoint is picked by. "
                                      "Replayed on your past runs, the default stops 2-3 epochs "
                                      "early and still catches the exact peak — the tail buys "
                                      "nothing. Raise it to be more cautious. Note this is a "
                                      "graceful stop, so temperature calibration still runs "
                                      "(killing the job skips it).")
                             .bind_value(gs, "tr_es_patience"))
        ui.label("Cool-down runs after stage 2 with reduced batch/LR — helps settle into flatter minima. "
                 "Epochs 0–3 are the frozen-backbone warm-up; the big accuracy jump comes at the first "
                 "stage-2 epoch, so don't judge a run before then."
                 ).classes("text-caption text-grey-7")

    # This determines WHAT THE MODEL IS, so it is open by default and states the
    # outcome in plain words. Ticking Hierarchical silently overrides the rank
    # radio (train_herbarium: rank_col = "species" if hierarchical else label_level),
    # which is easy to get wrong — the radio used to sit here unlabelled next to a
    # caption saying hierarchical merely "adds" heads.
    with _accordion("What the model classifies (rank & hierarchy)", opened=True):
        with ui.row().classes("w-full items-center gap-4 flex-wrap"):
            ui.label("Classify at:").classes("text-sm font-medium")
            label_level = (ui.radio(
                {"species": "Species", "genus": "Genus", "family": "Family"},
                value="species").props("inline dense")
                .bind_value(gs, "tr_label_level"))
            hier = ui.checkbox("Hierarchical multi-head", value=False).bind_value(gs, "tr_hier")
            with ui.row().classes("items-center gap-1"):
                ui.label("Species w:").classes("text-sm")
                w_sp = ui.input(value="1.0").classes("w-16").props("dense outlined").bind_value(gs, "tr_w_sp")
            with ui.row().classes("items-center gap-1"):
                ui.label("Genus w:").classes("text-sm")
                w_ge = ui.input(value="0.5").classes("w-16").props("dense outlined").bind_value(gs, "tr_w_ge")
            with ui.row().classes("items-center gap-1"):
                ui.label("Family w:").classes("text-sm")
                w_fa = ui.input(value="0.0").classes("w-16").props("dense outlined").bind_value(gs, "tr_w_fa")

        # Live, unambiguous statement of what will actually be trained.
        rank_note = ui.label().classes("text-caption")

        def _sync_rank_note() -> None:
            hierarchical = bool(hier.value)
            # The rank radio is meaningless under hierarchical — disable it rather
            # than let it look like it still applies.
            (label_level.disable if hierarchical else label_level.enable)()
            for w in (w_sp, w_ge, w_fa):
                (w.enable if hierarchical else w.disable)()
            if hierarchical:
                rank_note.set_text(
                    "→ The model's classes are SPECIES. 'Classify at' is IGNORED. "
                    "Genus and family become auxiliary heads that sharpen species accuracy "
                    "(and identify reads genus from the genus head, which is usually the more "
                    "accurate one). Careful: 'Min images per class' and 'Max images per class' "
                    "then apply PER SPECIES — a genus whose species are each individually thin "
                    "gets dropped entirely.")
                rank_note.classes(replace="text-caption text-orange-9")
            else:
                rank = str(label_level.value or "species")
                rank_note.set_text(
                    f"→ One flat classifier over {rank.upper()}. The loss weights on the right "
                    f"are unused. 'Min/Max images per class' apply per {rank}.")
                rank_note.classes(replace="text-caption text-grey-7")

        hier.on_value_change(lambda _: _sync_rank_note())
        label_level.on_value_change(lambda _: _sync_rank_note())
        _sync_rank_note()

    with _accordion("Geo features (lat/lon)", opened=False):
        with ui.row().classes("w-full items-center gap-4 flex-wrap"):
            use_location = (ui.checkbox("Use lat/lon during training (--use-location)", value=False)
                            .bind_value(gs, "tr_use_location"))
            with ui.row().classes("items-center gap-1"):
                ui.label("Geo MLP dim:").classes("text-sm")
                geo_dim = ui.input(value="64").classes("w-20").props("dense outlined").bind_value(gs, "tr_geo_dim")
            geo_dim.bind_enabled_from(use_location, "value")
        ui.label("Adds a small MLP that consumes encoded coordinates alongside "
                 "the image features — helps when distinct species share the "
                 "same morphology but live in different ranges."
                 ).classes("text-caption text-grey-7")

    with _accordion("Logging & resume", opened=False):
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("WandB project:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
            wandb_proj = ui.input(value="").classes("w-48").props("dense outlined").bind_value(gs, "tr_wandb_proj")
            ui.label("(rarely changes — set once per project; experiments distinguished by Run name below)"
                     ).classes("text-caption text-grey-7")
        resume = _path_input("Resume checkpoint:", mode="file").bind_value(gs, "tr_resume")
        reset_opt = (ui.checkbox(
            "Reset optimizer  (load weights only — use when starting a fresh stage 2 from a stage-1 checkpoint)",
            value=False).bind_value(gs, "tr_reset_optimizer")
            .tooltip("Discards the saved optimizer/LR-schedule state so stage 2 starts "
                     "with a clean optimizer at the LR you specify above. "
                     "Leave unticked to continue an interrupted stage-2 run."))
        with ui.row().classes("w-full items-center gap-4"):
            (ui.checkbox(
                "Stage images to local disk (escape MooseFS — needed on RunPod for full GPU util)",
                value=True).bind_value(gs, "tr_stage_images")
                .tooltip("Rsync the image dir to /dev/shm (or /root/staged_images) on the pod "
                         "before training. /workspace is a network FS; per-file reads starve "
                         "the GPU. Local staging typically lifts A100 utilisation from ~20% "
                         "to 90%+. Idempotent — only copies new/changed files."))
            (ui.checkbox(
                "Disable gradient checkpointing (A100/H100 only — uses more VRAM, ~30% faster)",
                value=False).bind_value(gs, "tr_no_grad_ckpt")
                .tooltip("Default on (safe on 24 GB cards). Turn off only on a 40+ GB GPU "
                         "where the batch fits without checkpointing — gives back the "
                         "30% compute that checkpointing trades for memory."))
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("DALI prefetch queue:").classes("w-36 text-right shrink-0 font-medium")\
                .style("color:#455a64")
            (ui.number(value=2, min=1, max=8, step=1, format="%d")
                .classes("w-24").props("dense outlined")
                .bind_value(gs, "tr_prefetch_queue")
                .tooltip("DALI prefetch queue depth. 2 is fine for local NVMe; bump to 4 "
                         "when training directly off the network volume."))

    def _tr_env() -> dict:
        env = {}
        if nccl_p2p_disable.value:
            env["NCCL_P2P_DISABLE"] = "1"
        return env

    # Run name lives outside the accordion because it's the one logging
    # field that should change per experiment. Placed just above Run
    # Training so the user sees / edits it on every launch.
    with ui.row().classes("w-full items-center gap-2 mt-3"):
        ui.label("WandB run name:").classes("w-36 text-right shrink-0 font-medium")\
            .style("color:#455a64")
        wandb_name = (ui.input(value="herbarium_run",
                               placeholder="e.g. stage2_lr1e-4_geo")
                      .classes("flex-1").props("dense outlined")
                      .bind_value(gs, "tr_wandb_name"))
        ui.label("descriptive names make the WandB sidebar much easier to scan")\
            .classes("text-caption text-grey-7")

    ui.button("Run Training", icon="model_training",
              on_click=lambda: _run_step_mode_aware(
                  "train", _tr_cmd, cloud_env_fn=_cloud_env_train,
                  extra_env=_tr_env())
              ).props("color=primary unelevated").classes("mt-4")

    def _tr_cmd() -> list[str]:
        srcs = tr_sources.get_sources()
        if not srcs: raise ValueError("Add at least one data source.")
        out = _v(tr_out)
        if not out: raise ValueError("Enter an output directory.")
        n_gpus = int(_v(tr_gpus) or "1")

        if n_gpus > 1:
            cmd = [sys.executable, "-u", "-m", "torch.distributed.run",
                   "--standalone", f"--nproc_per_node={n_gpus}",
                   str(SCRIPTS["train"])]
        else:
            cmd = [sys.executable, "-u", str(SCRIPTS["train"])]

        cmd += ["--sources"] + srcs + [
               "--output-dir",    out,
               "--model",         tr_model.value,
               "--image-sz",      tr_imgsz.value,
               "--batch-size",       tr_batch.value,
               "--stage2-batch-size", tr_s2_batch.value,
               "--accum",            tr_accum.value,
               "--num-gpus",      tr_gpus.value,
               "--stage1-epochs", s1ep.value,
               "--stage1-lr",     s1lr.value,
               "--stage2-epochs", s2ep.value,
               "--stage2-lr",     s2lr.value,
               "--cooldown-epochs",     cd_ep.value,
               "--cooldown-lr",         cd_lr.value,
               "--cooldown-batch-size", cd_batch.value,
               "--cooldown-accum",      cd_accum.value,
        ]
        if hier.value:
            cmd += ["--hierarchical",
                    "--species-weight", w_sp.value,
                    "--genus-weight",   w_ge.value,
                    "--family-weight",  w_fa.value]
        else:
            cmd += ["--label-level", label_level.value]
        proj = _v(wandb_proj)
        if proj:
            cmd += ["--wandb-project", proj, "--wandb-run-name", wandb_name.value]
        else:
            cmd += ["--no-wandb"]
        ck = _v(resume)
        if ck:
            cmd += ["--resume", ck]
            if reset_opt.value:
                cmd += ["--reset-optimizer"]
        if use_location.value:
            cmd += ["--use-location", "--geo-dim", geo_dim.value]
        mps = _v(tr_max_per_sp)
        if mps and mps != "0": cmd += ["--max-per-class", mps]
        esp = _v(tr_es_pat)
        if esp and esp != "0": cmd += ["--early-stop-patience", esp]
        spq = _v(tr_sparse)
        if spq: cmd += ["--sparse-threshold", spq]
        cwb = _v(tr_cw_beta)
        if cwb: cmd += ["--class-weight-beta", cwb]

        return cmd

    return _tr_cmd, tr_out, wandb_name, tr_sources, tr_model


def _build_quick_identify() -> None:
    """Single-image drag-and-drop identification panel."""
    import tempfile, base64
    gs = app.storage.general
    _qi_image_path: list[str | None] = [None]

    with ui.row().classes("w-full gap-4 items-start flex-wrap"):

        # ── Left: upload + coordinates ──────────────────────────────────────
        # Drag handlers go on the column (plain <div>) so they fire regardless
        # of which child element the cursor is over when the user drops.
        with ui.column().classes("gap-2").style("flex:1;min-width:260px") as left_col:
            _section("Image")
            preview = (ui.image("")
                       .classes("w-full rounded")
                       .style("max-height:320px;object-fit:contain;"
                              "background:#e8edf0;min-height:140px;"
                              "border:1px dashed #90a4ae"))

        # Capture-phase handlers fire before Quasar's QUploader (which calls
        # stopPropagation on every drag event).
        #
        # Chrome image-element drags:  dataTransfer.types = ['text/html', 'text/plain']
        # Link / address-bar drags:    dataTransfer.types = ['text/uri-list', ...]
        # Firefox image drags:         dataTransfer.types = ['text/uri-list', ...]
        # Desktop file drags:          dataTransfer.types = ['Files']
        #
        # We only intercept web-content drags (no 'Files' present).
        left_col.on('dragover.capture', js_handler="""
(e) => {
    const types = [...e.dataTransfer.types];
    if (!types.includes('Files') &&
        (types.includes('text/uri-list') || types.includes('text/html'))) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.style.outline = '3px dashed #00897b';
        e.currentTarget.style.borderRadius = '6px';
    }
}""")
        left_col.on('dragleave', js_handler="""
(e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) {
        e.currentTarget.style.outline = '';
    }
}""")
        left_col.on('drop.capture', js_handler="""
async (e) => {
    const types = [...e.dataTransfer.types];
    if (types.includes('Files')) return;  // let file drops reach the uploader

    // Prevent browser default (navigate to image) for ALL web-content drops
    e.preventDefault();
    e.stopPropagation();

    // Extract image URL via every available mechanism (all sync, before any await)
    let url = (e.dataTransfer.getData('text/uri-list') || '').trim().split('\\n')[0].trim();
    if (!url) {
        // Chrome image-element drag: URL is in text/html as <img src="...">
        const html = e.dataTransfer.getData('text/html') || '';
        const m = html.match(/src=[\\x22\\x27]([^\\x22\\x27]+)[\\x22\\x27]/i);
        if (m) url = m[1];
    }
    if (!url) {
        // Chrome also puts the image URL in text/plain
        const plain = (e.dataTransfer.getData('text/plain') || '').trim();
        if (/^https?:\\/\\//.test(plain)) url = plain;
    }

    if (!url || !/^https?:\\/\\//.test(url)) {
        console.warn('[Quick ID] drop: could not find a URL. types:', types,
            'html snippet:', (e.dataTransfer.getData('text/html')||'').slice(0,120));
        return;
    }

    const el = e.currentTarget;   // save ref — currentTarget is null after await
    el.style.outline = '';
    el.style.opacity = '0.5';
    try {
        const resp = await fetch('/api/qi_fetch_url?url=' + encodeURIComponent(url));
        const json = await resp.json();
        if (json.error) console.error('[Quick ID] URL fetch error:', json.error);
    } finally {
        el.style.opacity = '1';
    }
}""")
        with left_col:

            def _load_image_bytes(data: bytes, suffix: str) -> None:
                """Write bytes to a temp file and update the preview via base64."""
                tmp = tempfile.mktemp(suffix=suffix)
                with open(tmp, "wb") as f:
                    f.write(data)
                _qi_image_path[0] = tmp
                ext = suffix.lstrip('.').lower() or 'jpeg'
                preview.set_source(
                    f"data:image/{ext};base64,{base64.b64encode(data).decode()}")
                results_html.set_content("")

            async def _handle_upload(e):
                data = await e.file.read()
                _load_image_bytes(data, Path(e.file.name).suffix or ".jpg")

            def _check_url_drop():
                """Pick up images fetched by the /api/qi_fetch_url endpoint."""
                result = _qi_url_drop.pop("latest", None)
                if result:
                    _qi_image_path[0] = result["tmp"]
                    preview.set_source(result["data_url"])
                    results_html.set_content("")

            ui.timer(0.3, _check_url_drop)

            # File upload (from disk / file manager)
            (ui.upload(label="Drop image file or click to upload",
                       on_upload=_handle_upload, max_files=1)
             .props("accept=image/* flat bordered auto-upload")
             .classes("w-full"))

            # URL paste input — also picks up drag-from-webpage via the
            # preview element's drop handler (see above).
            async def _load_url(url: str):
                url = url.strip()
                if not url.startswith("http"):
                    return
                import urllib.request as _ur
                try:
                    req = _ur.Request(url, headers=_url_fetch_headers(url))
                    with _ur.urlopen(req, timeout=15) as resp:
                        data = resp.read()
                        ct = resp.headers.get("Content-Type",
                                              "image/jpeg").split(";")[0].strip()
                    ext = "." + (ct.split("/")[-1].replace("jpeg", "jpg") or "jpg")
                    _load_image_bytes(data, ext)
                    url_inp.set_value("")
                except Exception as ex:
                    ui.notify(f"Could not load URL: {ex}", type="negative")

            with ui.row().classes("w-full items-center gap-1 mt-1"):
                url_inp = (ui.input(placeholder="Paste image URL and press Enter")
                           .classes("flex-1").props("dense outlined clearable"))
                url_inp.on("keydown.enter", lambda: _load_url(url_inp.value))

            _section("Coordinates (optional)")
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.label("Lat:").classes("font-medium shrink-0").style("color:#455a64")
                qi_lat = (ui.input(placeholder="e.g. 4.93")
                          .classes("w-28").props("dense outlined")
                          .bind_value(gs, "qi_lat"))
                ui.label("Lon:").classes("font-medium shrink-0").style("color:#455a64")
                qi_lon = (ui.input(placeholder="e.g. 9.24")
                          .classes("w-28").props("dense outlined")
                          .bind_value(gs, "qi_lon"))

            _section("Model architecture")
            qi_model = (ui.input(placeholder="e.g. vit_large_patch16_dinov3.lvd1689m")
                        .classes("w-full").props("dense outlined")
                        .bind_value(gs, "tr_model"))

            qi_btn = (ui.button("Identify", icon="search",
                                on_click=lambda: _run_qi())
                      .props("unelevated color=teal").classes("w-full mt-2"))

        # ── Right: results ───────────────────────────────────────────────────
        with ui.column().classes("gap-2").style("flex:1.4;min-width:300px"):
            _section("Top-5 Predictions")
            results_html = ui.html(
                "<div style='color:#90a4ae;padding:12px 4px'>"
                "Upload an image and click Identify.</div>",
                sanitize=False,
            ).classes("w-full")

    async def _run_qi():
        ckpt = gs.get("active_ckpt", "").strip()
        if not ckpt:
            ui.notify("Set an active checkpoint in the model bar above.", type="warning")
            return
        if not _qi_image_path[0]:
            ui.notify("Upload an image first.", type="warning")
            return

        qi_btn.disable()
        results_html.set_content(
            "<div style='color:#888;padding:8px'>Running inference…</div>")
        try:
            model_hint = gs.get("tr_model", "").strip()
            preds = await asyncio.get_event_loop().run_in_executor(
                None, _qi_infer, ckpt, _qi_image_path[0],
                _v(qi_lat), _v(qi_lon), model_hint)

            # Bar widths are normalised to the top prediction so that even
            # low-confidence results still show a readable bar.
            max_p = preds[0][1] if preds else 1.0
            accents = ["#00796b", "#0097a7", "#5c6bc0", "#7b1fa2", "#546e7a"]
            html = "<div style='display:flex;flex-direction:column;gap:7px;padding:4px 0'>"
            for i, (name, prob) in enumerate(preds):
                pct_raw = prob * 100
                pct_bar = (prob / max_p) * 100
                accent  = accents[i]
                bg      = "#f0faf9" if i == 0 else "#fafafa"
                w       = "700" if i == 0 else "500"
                fsz     = "1.05em" if i == 0 else "0.93em"
                html += (
                    f"<div style='display:flex;align-items:center;gap:10px;"
                    f"background:{bg};border-radius:6px;padding:9px 12px;"
                    f"border-left:4px solid {accent}'>"
                    # rank
                    f"<span style='color:{accent};font-weight:700;font-size:11px;"
                    f"min-width:18px;text-align:right;opacity:.85'>#{i+1}</span>"
                    # name + bar
                    f"<div style='flex:1;min-width:0'>"
                    f"<div style='font-size:{fsz};font-weight:{w};color:#1a2a30;"
                    f"font-style:italic;white-space:nowrap;overflow:hidden;"
                    f"text-overflow:ellipsis'>{name}</div>"
                    f"<div style='display:flex;align-items:center;gap:6px;margin-top:5px'>"
                    f"<div style='flex:1;background:#d8e4e2;border-radius:4px;height:9px'>"
                    f"<div style='background:linear-gradient(90deg,{accent}cc,{accent}66);"
                    f"width:{pct_bar:.1f}%;height:100%;border-radius:4px;"
                    f"transition:width .5s ease'></div></div>"
                    f"<span style='font-size:0.85em;font-weight:700;color:{accent};"
                    f"min-width:44px;text-align:right'>{pct_raw:.1f}%</span>"
                    f"</div></div></div>"
                )
            html += "</div>"
            results_html.set_content(html)
        except Exception as ex:
            results_html.set_content(
                f"<div style='color:#c62828;padding:8px;font-weight:500'>"
                f"Error: {ex}</div>")
            ui.notify(str(ex), type="negative")
        finally:
            qi_btn.enable()


def _build_identify(tr_model=None) -> tuple:
    gs = app.storage.general
    with _cloud_only(ui.column().classes("w-full")):
        ui.label(
            "Cloud mode: identify runs on the pod with bootstrap defaults. "
            "It auto-picks the most recent .ckpt under /workspace/data/checkpoints/. "
            "The advanced thresholds, image size, batch size, and geo re-rank "
            "below configure local runs only."
        ).classes("text-body2").style(
            "background:#f0f7f6;border-left:3px solid #00897b;padding:8px 12px;"
            "border-radius:0 4px 4px 0;color:#37474f;max-width:1000px;margin-bottom:6px")
    _section("Model")
    with ui.row().classes("w-full items-center gap-2"):
        ui.label("Checkpoint (.ckpt):").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
        id_ckpt = (ui.input(value="", placeholder="file or checkpoints/ dir")
                   .classes("flex-1").props("dense outlined clearable")
                   .bind_value(gs, "active_ckpt"))

        async def _browse_ckpt():
            cur = _v(id_ckpt) or str(Path.home())
            result = await FilePicker(cur, mode="file")
            if result:
                id_ckpt.value = result

        async def _pick_latest():
            """Find the most recently modified .ckpt in the sibling checkpoints/ dir."""
            cur = _v(id_ckpt)
            search_dirs = []
            if cur:
                p = Path(cur)
                if p.is_dir():
                    search_dirs.append(p)
                else:
                    search_dirs += [p.parent, p.parent.parent / "checkpoints"]
            if not search_dirs:
                ui.notify("Enter a checkpoint path or directory first.", type="warning")
                return
            ckpts = []
            for d in search_dirs:
                ckpts += list(d.glob("*.ckpt"))
            if not ckpts:
                ui.notify("No .ckpt files found.", type="warning")
                return
            latest = max(ckpts, key=lambda p: p.stat().st_mtime)
            id_ckpt.value = str(latest)
            ui.notify(f"Selected: {latest.name}", type="positive")

        ui.button(icon="folder_open", on_click=_browse_ckpt).props("flat dense")
        ui.button("Latest", icon="update", on_click=_pick_latest).props("flat dense")

    with _local_only(ui.column().classes("w-full")):
        id_nl = (_path_input("nameslist.json:", mode="file",
                             hint="optional — embedded in checkpoint from recent runs")
                 .bind_value(gs, "id_nl"))
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("timm model (override):").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
            id_model = (ui.input(value="", placeholder="uses training model if blank")
                        .classes("flex-1").props("dense outlined")
                        .bind_value(gs, "id_model"))

        _section("Data sources  (specsin CSV : images directory)")
        id_sources = SourcesPanel("identify_sources")

        _section("Output")
        id_out = _path_input("Review output dir:", mode="dir").bind_value(gs, "id_out")

    # Visible in both modes — cloud identify now ships these as env vars.
    adv_accordion = _accordion("Advanced — thresholds, image size, geo re-rank", opened=False)
    with adv_accordion:
        ui.label("Defaults work for most runs. Adjust if you need stricter "
                 "mismatch flagging, smaller batches for VRAM, or want geo-aware "
                 "re-ranking on top of the model probability."
                 ).classes("text-caption text-grey-7")
        with ui.row().classes("w-full items-center gap-4 flex-wrap mt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Mismatch threshold:").classes("text-sm")
                id_thresh = ui.input(value="0.7").classes("w-20").props("dense outlined").bind_value(gs, "id_thresh")
            with ui.row().classes("items-center gap-1"):
                ui.label("Low-conf flag (0=off):").classes("text-sm")
                id_lowconf = ui.input(value="0.3").classes("w-20").props("dense outlined").bind_value(gs, "id_lowconf")
            with ui.row().classes("items-center gap-1"):
                ui.label("Image size (px):").classes("text-sm")
                id_imgsz = ui.input(value="640").classes("w-20").props("dense outlined").bind_value(gs, "id_imgsz")
            with ui.row().classes("items-center gap-1"):
                ui.label("Batch size:").classes("text-sm")
                id_batch = ui.input(value="32").classes("w-20").props("dense outlined").bind_value(gs, "id_batch")
        with ui.row().classes("w-full items-center gap-4 flex-wrap mt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Geo rerank weight (0=off):").classes("text-sm")
                id_geo_weight = (ui.input(value="0.0").classes("w-20").props("dense outlined")
                                 .bind_value(gs, "id_geo_weight"))
                ui.tooltip("Blend model probability with geographic range from training occurrences. "
                           "0 = off, 0.3 is a good starting point. Only applied when lat/lon is present.").props("max-width=320px")
            with ui.row().classes("items-center gap-1"):
                ui.label("Geo sigma (km):").classes("text-sm")
                id_geo_sigma = (ui.input(value="500").classes("w-20").props("dense outlined")
                                .bind_value(gs, "id_geo_sigma"))
                ui.tooltip("Kernel bandwidth for geographic scoring. Larger = broader range influence. "
                           "500 km suits most plant families; use 200–300 for highly localised taxa.").props("max-width=320px")
            with ui.row().classes("items-center gap-1"):
                ui.label("Common/rare bias (tau):").classes("text-sm")
                id_logit_adjust = (ui.input(value="0.0").classes("w-20").props("dense outlined")
                                   .bind_value(gs, "id_logit_adjust"))
                ui.tooltip("Two-way dial, applied without retraining. "
                           "POSITIVE favours commoner taxa: set it to the 'Rare-class boost (beta)' "
                           "the model was trained with to cancel it out — use 1.0 for any model "
                           "trained before that setting existed, which fixes near-empty taxa "
                           "hoovering up predictions from your commonest genus. "
                           "NEGATIVE favours rarer taxa: this is the cheap way to get a rare-class "
                           "boost — train with beta 0, then try -0.25 or -0.5 here and compare, "
                           "instead of paying for a retrain per guess. "
                           "0 = leave the model as trained.").props("max-width=360px")

    ui.button("Run Identify", icon="manage_search",
              on_click=lambda: _run_step_mode_aware(
                  "identify", _id_cmd, cloud_env_fn=_cloud_env_identify)
              ).props("color=primary unelevated").classes("mt-4")\
              .tooltip("Cloud mode: runs identify on the pod, auto-picking "
                       "the latest .ckpt under /workspace/data/checkpoints. "
                       "MODEL / IMAGE_SZ / BATCH_SIZE / THRESHOLD / GEO_WEIGHT "
                       "/ GEO_SIGMA are shipped from the fields above.")

    def _id_cmd() -> list[str]:
        ck = _v(id_ckpt)
        if not ck: raise ValueError("Specify a checkpoint file or directory.")
        srcs = id_sources.get_sources()
        if not srcs: raise ValueError("Add at least one data source.")
        out = _v(id_out)
        if not out: raise ValueError("Enter an output directory.")
        cmd = [sys.executable, "-u", str(SCRIPTS["identify"]),
               "--checkpoint", ck,
               "--sources"] + srcs + [
               "--output-dir",           out,
               "--threshold",            id_thresh.value,
               "--low-conf-threshold",   id_lowconf.value,
               "--image-sz",             id_imgsz.value,
               "--batch-size",           id_batch.value,
               "--geo-weight",           id_geo_weight.value,
               "--geo-sigma",            id_geo_sigma.value,
        ]
        la = _v(id_logit_adjust)
        if la: cmd += ["--logit-adjust", la]
        nl = _v(id_nl)
        if nl: cmd += ["--nameslist", nl]
        m = _v(id_model) or (_v(tr_model) if tr_model else "")
        if m: cmd += ["--model", m]
        return cmd

    return _id_cmd, id_ckpt, id_nl, id_out, id_sources


def _build_distribution(tr_sources: "SourcesPanel | None" = None) -> tuple:
    """Tab: load a specsin CSV and display species/genus/family distribution charts."""
    import csv as _csv
    import copy
    import random as _random

    _section("specsin CSV")
    dist_csv = _path_input("specsin CSV:", mode="file")
    dist_img  = _path_input("Images dir:", mode="dir")

    _section("Options")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        with ui.row().classes("items-center gap-1"):
            ui.label("Filter:").classes("text-sm")
            filt = ui.radio(
                {"all": "All rows", "hasfile": "Has file only"},
                value="hasfile").props("inline dense")
        with ui.row().classes("items-center gap-1"):
            ui.label("Cap per species (0 = off):").classes("text-sm")
            cap_inp = ui.input(value="0").classes("w-20").props("dense outlined")
            ui.label("random sample").classes("text-caption text-grey-7")
        with ui.row().classes("items-center gap-1"):
            ui.label("Top N in chart (0 = all):").classes("text-sm")
            top_n = ui.select([10, 20, 30, 50, 100, 0], value=20
                              ).props("dense outlined").classes("w-24")

    load_btn = ui.button("Load & Plot", icon="bar_chart"
                         ).props("color=primary unelevated").classes("mt-3")
    status_lbl = ui.label("").classes("text-caption text-grey-7 mt-1")

    _COLORS = {"species": "#14B8A6", "genus": "#6366F1", "family": "#F59E0B"}
    _CHART_BASE = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid":    {"left": "3%", "right": "5%", "containLabel": True},
        "xAxis":   {"type": "value"},
        "yAxis":   {"type": "category", "data": [],
                    "axisLabel": {"fontSize": 11}},
        "series":  [{"type": "bar", "data": [], "color": "#14B8A6",
                     "label": {"show": True, "position": "right",
                               "fontSize": 10}}],
    }

    charts: dict[str, ui.echart] = {}
    for level in ("species", "genus", "family"):
        ui.separator().classes("my-3")
        ui.label(f"{level.capitalize()} distribution").classes(
            "text-subtitle2 font-bold")
        opts = copy.deepcopy(_CHART_BASE)
        opts["series"][0]["color"] = _COLORS[level]
        charts[level] = ui.echart(opts).style("height:300px; width:100%")

    # ---- Export section ----
    _section("Export subsampled data")
    export_path = _path_input("Save CSV to:", mode="save")
    export_lbl  = ui.label("").classes("text-caption text-teal-700 font-mono mt-1 ml-48")

    with ui.row().classes("w-full items-center gap-3 mt-2"):
        export_btn = ui.button("Export CSV", icon="save_alt"
                               ).props("color=secondary unelevated")
        if tr_sources is not None:
            use_btn = ui.button("→ Use in Training", icon="model_training"
                                ).props("color=teal unelevated")

    # Mutable state shared between load and export
    _state: dict = {"rows": [], "fieldnames": []}

    def _load():
        path = _v(dist_csv)
        if not path or not Path(path).is_file():
            ui.notify("Select a valid specsin CSV file.", type="warning")
            return

        all_rows: list[dict] = []
        fieldnames: list[str] = []
        total = 0

        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = _csv.DictReader(fh)
                fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    total += 1
                    if filt.value == "hasfile":
                        hf = (row.get("hasfile") or "").strip().lower()
                        if hf not in ("true", "1", "yes"):
                            continue
                    all_rows.append(row)
        except Exception as exc:
            ui.notify(f"Error reading CSV: {exc}", type="negative")
            return

        kept = len(all_rows)
        cap = int(cap_inp.value or 0)
        if cap > 0:
            by_species: dict[str, list] = {}
            for row in all_rows:
                sp = (row.get("species") or "").strip() or "(unknown)"
                by_species.setdefault(sp, []).append(row)
            sampled: list[dict] = []
            for sp, rows in by_species.items():
                sampled.extend(_random.sample(rows, min(len(rows), cap)))
            all_rows = sampled

        _state["rows"] = all_rows
        _state["fieldnames"] = fieldnames

        # Auto-suggest export path (stem + _cap{N} or _subsampled)
        src_path = Path(path)
        suffix = f"_cap{cap}" if cap > 0 else "_subsampled"
        suggested = str(src_path.parent / (src_path.stem + suffix + src_path.suffix))
        if not _v(export_path):
            export_path.value = suggested

        counts: dict[str, dict[str, int]] = {
            "species": {}, "genus": {}, "family": {}}
        for row in all_rows:
            for level in ("species", "genus", "family"):
                val = (row.get(level) or "").strip() or "(unknown)"
                counts[level][val] = counts[level].get(val, 0) + 1

        filter_label = "all" if filt.value == "all" else "with images"
        cap_label = f"  ·  capped at {cap}/species → {len(all_rows):,} rows" if cap > 0 else ""
        status_lbl.set_text(
            f"Loaded {kept:,} rows {filter_label} / {total:,} total{cap_label}  ·  "
            f"{len(counts['species']):,} species  ·  "
            f"{len(counts['genus']):,} genera  ·  "
            f"{len(counts['family']):,} families")

        n = int(top_n.value) or None
        for level in ("species", "genus", "family"):
            sorted_items = sorted(counts[level].items(), key=lambda x: x[1])
            if n:
                sorted_items = sorted_items[-n:]
            labels = [it[0] for it in sorted_items]
            values = [it[1] for it in sorted_items]
            row_h  = max(18, min(28, 500 // max(len(labels), 1)))
            height = max(200, len(labels) * row_h + 60)
            charts[level].style(f"height:{height}px; width:100%")
            charts[level].options["yAxis"]["data"] = labels
            charts[level].options["series"][0]["data"] = values
            charts[level].update()

    def _export():
        if not _state["rows"]:
            ui.notify("Load & Plot first.", type="warning")
            return
        dest = _v(export_path)
        if not dest:
            ui.notify("Enter a save path.", type="warning")
            return
        try:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            fieldnames = _state["fieldnames"] or list(_state["rows"][0].keys())
            with open(dest, "w", newline="", encoding="utf-8") as fh:
                writer = _csv.DictWriter(fh, fieldnames=fieldnames,
                                         extrasaction="ignore")
                writer.writeheader()
                writer.writerows(_state["rows"])
            export_lbl.set_text(f"Saved {len(_state['rows']):,} rows → {dest}")
            ui.notify(f"Exported → {Path(dest).name}", type="positive")
        except Exception as exc:
            ui.notify(f"Export failed: {exc}", type="negative")

    def _use_in_training():
        dest = _v(export_path)
        imgs = _v(dist_img)
        if not dest or not Path(dest).is_file():
            ui.notify("Export the CSV first.", type="warning")
            return
        if not imgs:
            ui.notify("Set the Images dir first.", type="warning")
            return
        tr_sources.set_source(f"{dest}:{imgs}")
        ui.notify(f"Training source updated → {Path(dest).name}", type="positive")

    load_btn.on_click(_load)
    export_btn.on_click(_export)
    if tr_sources is not None:
        use_btn.on_click(_use_in_training)

    return dist_csv, dist_img


def _build_review() -> tuple:
    """Tab: interactive carousel for reviewing prediction results."""
    import pandas as _pd

    def _top5_items(row) -> list[tuple[str, float]]:
        """Return [(name, prob), ...] from top1…top5 columns, or pred_species fallback."""
        if "top1_name" in row.index:
            items = []
            for k in range(1, 6):
                raw  = row.get(f"top{k}_name", "")
                name = "" if (raw != raw or raw is None) else str(raw)
                if not name or name == "nan":
                    break
                items.append((name, float(row.get(f"top{k}_prob", 0) or 0)))
            return items or [(str(row.get("pred_species", "")),
                              float(row.get("confidence", 0) or 0))]
        return [(str(row.get("pred_species", "")),
                 float(row.get("confidence", 0) or 0))]

    def _bars_html(row, level: str = "species") -> str:
        """HTML confidence-bar table. level: 'species' | 'genus' | 'family'"""

        def _render(items: list[tuple[str, float]], col_label: str, italic: bool) -> str:
            tag = "i" if italic else "span"
            rows_html = [
                f"<tr>"
                f"<td style='padding:2px 6px;color:#888'>#{k}</td>"
                f"<td style='padding:2px 8px;max-width:220px;overflow:hidden;"
                f"text-overflow:ellipsis;white-space:nowrap'><{tag}>{name}</{tag}></td>"
                f"<td style='padding:2px 6px;width:180px'>"
                f"<div style='background:linear-gradient(to right,#009688 {prob*100:.1f}%,"
                f"#e0e0e0 {prob*100:.1f}%);padding:2px 6px;border-radius:3px;"
                f"font-size:11px'>{prob:.1%}</div></td>"
                f"</tr>"
                for k, (name, prob) in enumerate(items[:5], 1)
            ]
            return (
                f"<table style='border-collapse:collapse;font-size:12px;width:100%'>"
                f"<tr><th style='padding:2px 6px;color:#555;text-align:left'>#</th>"
                f"<th style='padding:2px 8px;color:#555;text-align:left'>{col_label}</th>"
                f"<th style='padding:2px 6px;color:#555;text-align:left'>Confidence</th>"
                f"</tr>" + "".join(rows_html) + "</table>"
            )

        if level == "genus":
            # Prefer the trained genus head's own ranked list (gtop*). It is a
            # different — and far better — answer than marginalising the species
            # head: ~97% vs ~89% top-1. Only fall back to summing the species
            # top-5 by first word for CSVs written before identify emitted these,
            # and say so, because that fallback silently truncates at 5 species.
            if "gtop1_name" in row.index and str(row.get("gtop1_name", "")) not in ("", "nan"):
                items = []
                for k in range(1, 6):
                    raw = row.get(f"gtop{k}_name", "")
                    name = "" if (raw != raw or raw is None) else str(raw)
                    if not name or name == "nan":
                        break
                    items.append((name, float(row.get(f"gtop{k}_prob", 0) or 0)))
                return _render(items, "Genus (genus head)", italic=True)

            agg: dict[str, float] = {}
            for name, prob in _top5_items(row):
                g = name.split()[0] if name and name != "nan" else name
                agg[g] = agg.get(g, 0) + prob
            return (_render(sorted(agg.items(), key=lambda x: x[1], reverse=True),
                            "Genus (from species head)", italic=True)
                    + "<div style='font-size:11px;color:#888;padding:2px 6px'>"
                      "Marginalised from the top-5 species — this CSV predates the "
                      "genus-head columns. Re-run Identify for the genus head's own "
                      "predictions.</div>")

        if level == "family":
            # Family-level CSV: top{k}_name already holds family names.
            # Species/genus-level CSV: aggregate top-5 by mapping each
            # top{k}_name → its family (preferring the per-row top{k}_family
            # column written by identify, falling back to a heuristic).
            agg: dict[str, float] = {}
            for k in range(1, 6):
                name = str(row.get(f"top{k}_name", "") or "")
                if not name or name == "nan":
                    break
                prob = float(row.get(f"top{k}_prob", 0) or 0)
                fam = str(row.get(f"top{k}_family", "") or "")
                if not fam or fam == "nan":
                    # No mapping → assume top{k}_name is itself the family
                    # (i.e. a family-level model whose identify run predates
                    # the top{k}_family column being written).
                    fam = name
                agg[fam] = agg.get(fam, 0) + prob
            if agg:
                items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
                return _render(items, "Family", italic=False)
            # Last-resort fallback for very old CSVs.
            if "pred_family" in row.index and str(row.get("pred_family", "nan")) not in ("", "nan"):
                prob = float(row.get("confidence", row.get("top1_prob", 0)) or 0)
                return _render([(str(row["pred_family"]), prob)], "Family", italic=False)
            return ("<div style='font-size:12px;color:#888;padding:4px'>"
                    "Family predictions not in this CSV. "
                    "Rerun Identify with a hierarchical checkpoint to add family columns.</div>")

        # species (default)
        return _render(_top5_items(row), "Species", italic=True)

    # ── state ────────────────────────────────────────────────────────────────

    _st: dict = {"df": None, "view": None, "idx": 0}

    # ── UI ───────────────────────────────────────────────────────────────────

    # Data-source settings collapse once a CSV is loaded so the image area
    # gets most of the vertical real estate. Header summary stays visible.
    _initial_csv = app.storage.general.get("review_csv", "")
    settings_exp = (ui.expansion("⚙ Data source", icon="settings",
                                 value=not bool(_initial_csv))
                    .classes("w-full"))
    with settings_exp:
        rev_csv  = _path_input("predictions.csv:", mode="file",
                               value=_initial_csv)
        rev_csv.bind_value(app.storage.general, "review_csv")
        rev_imgs = _path_input("Images dir (if CSV has relative paths):", mode="dir",
                               value=app.storage.general.get("review_imgs", ""))
        rev_imgs.bind_value(app.storage.general, "review_imgs")
        with ui.row().classes("w-full items-center gap-2 mt-1 ml-48"):
            load_btn = (ui.button("Load", icon="upload_file")
                        .props("color=primary unelevated"))
        # Fetch the results to review straight from wherever they live — the
        # running pod, or the R2 archive (works with the pod shut down). Pulls
        # predictions + checkpoints; images only when not already downloaded.
        with ui.row().classes("w-full items-center gap-2 mt-2 ml-48"):
            ui.label("Get results:").classes("text-sm text-grey-7 shrink-0")
            fetch_src = (ui.select({"pod": "From pod", "r2": "From R2 archive"},
                                   value=app.storage.general.get("review_fetch_src", "pod"))
                         .props("dense outlined").classes("w-44")
                         .bind_value(app.storage.general, "review_fetch_src"))
            if not app.storage.general.get("cloud_ckpt_filter"):
                app.storage.general["cloud_ckpt_filter"] = "latest"
            (ui.select({"latest": "ckpts: latest",
                        "best+latest": "ckpts: best + latest",
                        "all": "ckpts: all"},
                       value=app.storage.general.get("cloud_ckpt_filter") or "latest")
             .props("dense outlined").classes("w-44")
             .bind_value(app.storage.general, "cloud_ckpt_filter")
             .tooltip("Which checkpoints the pod fetch pulls. 'best + latest' is "
                      "usually what you want for local CPU Identify."))
            ui.button("Fetch & load", icon="cloud_download",
                      on_click=lambda: _wrap_cloud_aux(_run_fetch))\
                .props("unelevated color=teal")\
                .tooltip("Pull predictions + checkpoints (and images if missing), "
                         "wire the paths, and load. Pod source needs a running "
                         "pod; R2 works with the pod shut down.")
        ui.label(
            "Images are pulled only if the local images folder is empty — delete "
            "it to force a refresh. R2 also restores checkpoints for local Identify."
        ).classes("text-caption text-grey-6 ml-48")
    summary_lbl = ui.label("").classes("text-caption text-grey-7 mt-1")
    # Notice: taxa the model can't predict (dropped as too sparse at train time).
    # identify writes excluded_species.json next to predictions.csv. The one-line
    # caption is the summary; the expansion holds the *whole* list — on a long
    # tail that's ~1,300 taxa, so "+N more" would hide nearly all of it, and the
    # desktop review is exactly where a curator wants to see which taxa the model
    # can't predict (any specimen of these is forced to the nearest trained class).
    excluded_lbl = (ui.label("").classes("text-caption text-orange-9 mt-1")
                    .style("white-space:normal;line-height:1.3"))
    excluded_lbl.set_visibility(False)
    excluded_exp = (ui.expansion("Show all excluded taxa", icon="list")
                    .props("dense").classes("w-full"))
    excluded_exp.set_visibility(False)
    with excluded_exp:
        excluded_body = ui.html("")

    def _show_excluded(review_dir: Path) -> None:
        f = review_dir / "excluded_species.json"
        try:
            data = json.loads(f.read_text()) if f.is_file() else {}
            taxa = data.get("taxa", {}) or {}
            rank = data.get("rank", "species")
        except Exception:
            taxa, rank = {}, "species"
        if not taxa:
            excluded_lbl.set_visibility(False)
            excluded_exp.set_visibility(False)
            return
        names = sorted(taxa, key=lambda n: taxa[n])   # rarest first
        excluded_lbl.set_text(
            f"⚠ {len(taxa)} {rank} not in this model — too few images to train, "
            f"so their specimens are forced to the nearest trained class.")
        rows = "".join(
            f"<div style='break-inside:avoid'><i>{n}</i> "
            f"<span style='color:#9e9e9e'>({taxa[n]})</span></div>" for n in names)
        excluded_body.set_content(
            "<div style='max-height:340px;overflow:auto;columns:3;column-gap:28px;"
            "font-size:12px;line-height:1.6;padding:4px 2px'>" + rows + "</div>"
            "<div style='font-size:11px;color:#9e9e9e;margin-top:6px'>"
            "Rarest first; (n) = images available (below the sparse threshold). "
            "Full export in excluded_species.csv beside predictions.csv.</div>")
        excluded_lbl.set_visibility(True)
        excluded_exp.set_visibility(True)

    _section("Filter & Sort")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        filter_sel = ui.select(
            {"all": "All", "indets": "Indets only",
             "flagged": "Flagged only", "misid": "Misidentified (pred ≠ true)",
             "mislabels": "Possible mislabels (low AUM)",
             "high_conf": "High confidence (≥ 90%)",
             "sparse": "Sparse only (true species below threshold)"},
            value="all", label="Show"
        ).classes("w-52").props("dense outlined")
        sort_sel = ui.select(
            {"conf_desc": "Confidence ↓", "conf_asc": "Confidence ↑",
             "aum_asc": "AUM ↑ (most suspect first)",
             "species": "Predicted name A–Z"},
            value="conf_desc", label="Sort by"
        ).classes("w-52").props("dense outlined")
        apply_btn = (ui.button("Apply", icon="filter_list")
                     .props("flat dense color=teal")
                     .classes("self-end"))
    with ui.row().classes("w-full items-center gap-2 mt-1"):
        ui.label("Review level:").classes("text-sm text-grey-7")
        level_sel = (ui.toggle({"species": "Species", "genus": "Genus", "family": "Family"},
                               value="species")
                     .props("dense"))

    carousel_btn = (ui.button("Open Carousel", icon="open_in_new")
                    .props("flat dense color=teal")
                    .classes("self-end ml-auto")
                    .tooltip("Open full-screen review in a new tab"))

    # Free-form AI filter — collapsed by default; rarely changed mid-session.
    with ui.expansion("✨ AI filter", icon="auto_awesome").classes("w-full"):
        with ui.row().classes("w-full items-center gap-2"):
            ai_filter_inp = (ui.input(
                placeholder="e.g.  genus Uvaria · confidence < 30% · none of top 5 correct")
                .classes("flex-1").props("dense outlined clearable"))
            ai_filter_btn = (ui.button("AI Filter", icon="auto_awesome")
                             .props("unelevated dense color=deep-purple-4")
                             .tooltip("Use Claude to interpret your query"))
        ai_filter_lbl = ui.label("").classes("text-caption text-grey-6 mt-0")

    ui.separator().classes("my-2")

    # Carousel layout
    with ui.row().classes("w-full gap-4 items-start"):

        # Left: image + nav. Image grows with the viewport (carousel-like)
        # rather than being pinned at 400px so the user can actually read
        # the specimen labels without leaving the tab.
        with ui.column().classes("items-center gap-2").style(
                "min-width:560px; max-width:720px; flex:1 1 auto"):
            img_el = ui.image("").style(
                "width:100%; height:calc(100vh - 280px); min-height:420px;"
                "object-fit:contain;"
                "background:#f0f0f0; border-radius:6px; border:1px solid #ddd")
            counter_lbl = ui.label("").classes("text-caption text-grey-6")
            with ui.row().classes("gap-2 items-center"):
                prev_btn = (ui.button(icon="chevron_left",
                                      on_click=lambda: _go(-1))
                            .props("round outlined color=teal dense"))
                next_btn = (ui.button(icon="chevron_right",
                                      on_click=lambda: _go(1))
                            .props("round outlined color=teal dense"))
                open_btn = (ui.button(icon="open_in_new",
                                      on_click=lambda: _open_file())
                            .props("round flat dense")
                            .tooltip("Open local 640px image"))
                gbif_btn = (ui.button(icon="public",
                                      on_click=lambda: _open_gbif())
                            .props("round flat dense")
                            .tooltip("Open occurrence on gbif.org"))
                origin_btn = (ui.button(icon="zoom_out_map",
                                        on_click=lambda: _open_original())
                              .props("round flat dense")
                              .tooltip("Open original full-resolution image"))

        # Right: info + bars + actions
        with ui.column().classes("flex-1 gap-1").style("min-width:280px"):
            info_html = ui.html("").style("font-size:13px; color:#444")
            bars_html = ui.html("").style("width:100%; margin-top:6px")

            ui.separator().classes("my-2")

            det_sel = (ui.select([], label="Determine as:")
                       .classes("w-full")
                       .props("dense outlined"))
            with ui.row().classes("gap-2 flex-wrap mt-1"):
                confirm_btn = (ui.button("Confirm determination", icon="check",
                                         on_click=lambda: _confirm())
                               .props("color=positive unelevated dense"))
                invalid_btn = (ui.button("Mark invalid", icon="close",
                                         on_click=lambda: _mark_invalid())
                               .props("color=negative unelevated dense"))
            action_lbl = ui.label("").classes("text-caption text-teal-700 font-mono mt-1")

    # ── logic ────────────────────────────────────────────────────────────────

    def _resolve_path(row) -> str:
        """Return absolute image path.

        Priority: imgs_dir/fname (if imgs_dir is configured — handles cloud
        predictions whose abs_path points at the pod) → abs_path → filename.
        The first existing file wins; otherwise the first non-empty candidate
        is returned so missing-image errors surface instead of being silenced.
        """
        fname = str(row.get("fname", ""))
        imgs  = _v(rev_imgs)
        candidates: list[str] = []
        if fname and imgs:
            candidates.append(str(Path(imgs) / fname))
        for col in ("abs_path", "filename"):
            v = row.get(col, "")
            if v and v == v and str(v) not in ("", "nan"):
                candidates.append(str(v))
        for c in candidates:
            if Path(c).is_file():
                return c
        return candidates[0] if candidates else ""

    def _get_top5(row, level: str = "species") -> list[str]:
        """Candidate determinations for the dropdown, at the rank under review."""
        if level == "genus":
            # The genus head's own ranked list when identify wrote it; otherwise
            # the distinct genera of the top-5 species, in order.
            names = []
            for k in range(1, 6):
                v = row.get(f"gtop{k}_name", "")
                s = "" if (v != v or v is None) else str(v)
                if not s or s == "nan":
                    break
                names.append(s)
            if names:
                return names
            seen: list[str] = []
            for sp_name in _get_top5(row, "species"):
                g = sp_name.split()[0] if sp_name else ""
                if g and g not in seen:
                    seen.append(g)
            return seen

        if "top1_name" in row.index:
            names = []
            for k in range(1, 6):
                v = row.get(f"top{k}_name", "")
                s = "" if (v != v or v is None) else str(v)  # NaN → ""
                if not s or s == "nan":
                    break
                names.append(s)
            return names or [str(row.get("pred_species", ""))]
        return [str(row.get("pred_species", ""))]

    def _show(idx: int):
        view = _st["view"]
        if view is None or len(view) == 0:
            return
        idx = max(0, min(idx, len(view) - 1))
        _st["idx"] = idx
        row = view.iloc[idx]

        path = _resolve_path(row)
        img_el.set_source(_review_img_url(path) if path else "")

        counter_lbl.set_text(f"{idx + 1} / {len(view)}")

        level      = level_sel.value
        conf_val   = float(row.get("confidence", row.get("top1_prob", 0)) or 0)
        fname      = str(row.get("fname", row.get("filename", path)))
        source     = str(row.get("source", ""))
        cat        = str(row.get("catalogNumber", ""))
        true_sp    = str(row.get("true_species", ""))
        is_flagged = str(row.get("flagged", "")).lower() in ("true", "1")
        flag_badge = (" <span style='color:#d32f2f;font-weight:bold'>[FLAGGED]</span>"
                      if is_flagged else "")

        # Level-specific predicted / true label
        if level == "genus":
            # pred_genus is written from the trained genus head when the
            # checkpoint has one; the first-word split is only a fallback.
            pred_g = str(row.get("pred_genus", "") or "").strip()
            if not pred_g or pred_g == "nan":
                pred_sp = str(row.get("pred_species", row.get("top1_name", "")))
                pred_g  = (pred_sp.split()[0]
                           if pred_sp and pred_sp not in ("", "nan") else "?")
            true_g = str(row.get("true_genus", "") or "").strip()
            if not true_g or true_g == "nan":
                true_g = true_sp.split()[0] if true_sp and true_sp not in ("", "nan") else ""
            match   = (f" <span style='color:{'#388e3c' if pred_g==true_g else '#d32f2f'}'>"
                       f"{'✓' if pred_g==true_g else '✗'}</span>") if true_g else ""
            # In genus mode the headline confidence must be the genus head's,
            # not the species head's — they are different numbers and showing
            # the species one next to a genus label is simply wrong.
            g_conf = row.get("genus_conf", None)
            try:
                if g_conf is not None and g_conf == g_conf:
                    conf_val = float(g_conf)
            except (TypeError, ValueError):
                pass
            # Species and genus heads can disagree; that disagreement is itself
            # a "look at this sheet" signal, so surface it rather than hide it.
            agree = row.get("genus_agrees", None)
            disagree_note = ""
            if str(agree).lower() in ("false", "0"):
                sp_g = str(row.get("pred_species", "")).split()[0] if row.get("pred_species") else ""
                disagree_note = ("<span style='color:#f57c00'>⚠ genus head and species "
                                 f"head disagree (species head says <i>{sp_g}</i>)</span><br>")
            level_line = (f"<b>Predicted genus:</b> <i>{pred_g}</i>{match}<br>"
                          + (f"<b>True genus:</b> <i>{true_g}</i><br>" if true_g else "")
                          + disagree_note)
        elif level == "family":
            pred_f = str(row.get("pred_family", ""))
            true_f = str(row.get("true_family", ""))
            match  = ""
            if pred_f and pred_f != "nan" and true_f and true_f != "nan":
                ok = pred_f.strip() == true_f.strip()
                match = (f" <span style='color:{'#388e3c' if ok else '#d32f2f'}'>"
                         f"{'✓' if ok else '✗'}</span>")
            level_line = ((f"<b>Predicted family:</b> {pred_f}{match}<br>"
                           if pred_f and pred_f != "nan" else "")
                          + (f"<b>True family:</b> {true_f}<br>"
                             if true_f and true_f != "nan" else ""))
        else:
            level_line = f"<b>True species:</b> <i>{true_sp}</i><br>" if true_sp else ""

        lat_raw = row.get("decimalLatitude",  "")
        lon_raw = row.get("decimalLongitude", "")
        try:
            lat_f, lon_f = float(lat_raw), float(lon_raw)
            geo_str = f"{lat_f:.4f}, {lon_f:.4f}"
        except (TypeError, ValueError):
            geo_str = ""

        # AUM: present only for training specimens. Low/negative = the label is
        # a mislabel candidate; colour it as a warning so the eye lands on it.
        aum_line = ""
        aum_raw = row.get("aum", None)
        try:
            if aum_raw is not None and float(aum_raw) == float(aum_raw):  # not NaN
                av = float(aum_raw)
                colour = "#d32f2f" if av < 0 else ("#f57c00" if av < 2 else "#888")
                hint = " — possible mislabel" if av < 2 else ""
                aum_line = (f"<b>AUM:</b> <span style='color:{colour}'>{av:+.2f}"
                            f"{hint}</span><br>")
        except (TypeError, ValueError):
            pass

        info_html.set_content(
            f"<div style='font-size:13px;line-height:1.6'>"
            f"<b>Confidence:</b> {conf_val:.1%}{flag_badge}<br>"
            + aum_line
            + level_line
            + (f"<small style='color:#888'>{source}"
               + (f" | {cat}" if cat and cat != "nan" else "")
               + (f" | 📍 {geo_str}" if geo_str else "")
               + "</small><br>"
               if source or (cat and cat != "nan") or geo_str else "")
            + f"<span style='color:#bbb;font-family:monospace;font-size:13px'>{Path(fname).name}</span>"
            "</div>"
        )
        bars_html.set_content(_bars_html(row, level=level))

        top5 = _get_top5(row, level)
        det_sel.set_options(top5, value=top5[0] if top5 else "")
        action_lbl.set_text("")

    def _apply_filter():
        df = _st["df"]
        if df is None:
            return
        filt  = filter_sel.value
        level = level_sel.value
        sp_col   = "pred_species" if "pred_species" in df.columns else "top1_name"
        conf_col = "confidence"   if "confidence"   in df.columns else "top1_prob"
        # Sort/threshold on the confidence of the rank being reviewed. The genus
        # head has its own — a specimen can be a confident genus and an
        # uncertain species, which is exactly the case worth finding.
        if level == "genus" and "genus_conf" in df.columns:
            conf_col = "genus_conf"

        if filt == "indets":
            mask = df["indet"].astype(str).str.lower().isin(("true", "1"))
        elif filt == "flagged":
            mask = df["flagged"].astype(str).str.lower().isin(("true", "1"))
        elif filt == "misid":
            # "Has a real ground-truth label." MUST use notna(), not string
            # matching: `astype(str).str.strip()` leaves float NaN (not the
            # string "nan") for missing cells, so the old `.ne("nan")` guard
            # let every indet (no true species) leak in as "misidentified".
            def _has_label(series):
                s = series.astype(str).str.strip().str.lower()
                return series.notna() & ~s.isin(("", "nan", "none", "<na>", "na", "null"))
            if level == "family" and "pred_family" in df.columns and "true_family" in df.columns:
                pred = df["pred_family"].astype(str).str.strip()
                true = df["true_family"].astype(str).str.strip()
                mask = _has_label(df["true_family"]) & true.ne(pred)
            elif level == "genus" and ("pred_genus" in df.columns
                                       or "true_species" in df.columns):
                # Compare the genus head's answer to the recorded genus. Using
                # the species head's first word instead (the old behaviour)
                # flags disagreements the genus head gets right, so the misid
                # list was mostly noise from the weaker of the two heads.
                if "pred_genus" in df.columns:
                    pred_g = df["pred_genus"].astype(str).str.strip()
                else:
                    pred_g = df[sp_col].astype(str).str.split().str[0]
                if "true_genus" in df.columns:
                    true_col, true_g = df["true_genus"], df["true_genus"].astype(str).str.strip()
                else:
                    true_col = df["true_species"]
                    true_g = df["true_species"].astype(str).str.split().str[0]
                mask = _has_label(true_col) & true_g.ne(pred_g)
            else:
                if "true_species" in df.columns:
                    true_str = df["true_species"].astype(str).str.strip()
                    mask = (_has_label(df["true_species"]) &
                            true_str.ne(df[sp_col].astype(str).str.strip()))
                else:
                    mask = _pd.Series(False, index=df.index)
        elif filt == "mislabels":
            # Candidate mis-determinations: training specimens whose recorded
            # label the model could only fit by memorising it (low/negative
            # AUM). Only training specimens carry an AUM value, so notna() is
            # the whole selector; the ascending sort below puts the most
            # suspect first.
            if "aum" in df.columns:
                mask = df["aum"].notna()
            else:
                mask = _pd.Series(False, index=df.index)
        elif filt == "sparse":
            if "sparse" in df.columns:
                mask = df["sparse"].astype(str).str.lower().isin(("true", "1"))
            else:
                # Pre-sparse-column predictions.csv — fall through to "no rows".
                mask = _pd.Series(False, index=df.index)
        else:
            mask = _pd.Series(True, index=df.index)

        # Combine with AI filter if active
        ai_mask = _st.get("ai_mask")
        if ai_mask is not None:
            mask = mask & ai_mask

        view = df[mask].copy()
        sort = sort_sel.value
        # The mislabels view is only meaningful ordered by AUM, so default it
        # there even if the sort dropdown is still on its confidence default.
        if filt == "mislabels" and sort in ("conf_desc", "conf_asc"):
            sort = "aum_asc"
        if sort == "aum_asc" and "aum" in view.columns:
            view = view.sort_values("aum", ascending=True, na_position="last")
        elif sort == "conf_desc":
            view = view.sort_values(conf_col, ascending=False)
        elif sort == "conf_asc":
            view = view.sort_values(conf_col, ascending=True)
        elif sort == "species":
            if level == "genus":
                if "pred_genus" in view.columns:
                    view = view.sort_values("pred_genus")
                else:
                    view = view.sort_values(sp_col, key=lambda s: s.str.split().str[0])
            elif level == "family" and "pred_family" in view.columns:
                view = view.sort_values("pred_family")
            else:
                view = view.sort_values(sp_col)

        _st["view"] = view.reset_index(drop=True)
        _st["idx"]  = 0
        # Publish to shared state for the carousel page
        _review_shared["view"]     = _st["view"]
        _review_shared["imgs_dir"] = _v(rev_imgs)
        _review_shared["level"]    = level
        n = len(_st["view"])
        summary_lbl.set_text(f"Showing {n:,} specimens")
        if n:
            _show(0)
        else:
            img_el.set_source("")
            counter_lbl.set_text("0 / 0")
            info_html.set_content("")
            bars_html.set_content("")

    async def _run_ai_filter():
        """Send the free-form query to Claude Haiku and apply the result."""
        query = ai_filter_inp.value.strip() if ai_filter_inp.value else ""
        df = _st["df"]
        if df is None:
            ui.notify("Load a predictions CSV first.", type="warning")
            return
        if not query:
            # Clear AI filter
            _st["ai_mask"] = None
            ai_filter_lbl.set_text("")
            _apply_filter()
            return

        ai_filter_btn.props("loading")
        ai_filter_lbl.set_text("Asking Claude…")
        try:
            spec = await _ai_build_filter(query, df)
            if spec is None:
                ui.notify("Install the 'anthropic' package to use AI Filter.",
                          type="warning")
                return
            ai_filter_lbl.set_text(f"Filter: {json.dumps(spec, default=str)}")
            _st["ai_mask"] = _apply_filter_spec(spec, df)
            _apply_filter()
            ui.notify(f"AI filter applied — {_st['view'].shape[0]:,} results",
                      type="positive")
        except Exception as exc:
            ai_filter_lbl.set_text(f"Error: {exc}")
            ui.notify(str(exc), type="negative")
        finally:
            ai_filter_btn.props(remove="loading")

    def _load():
        path = _v(rev_csv)
        if not path or not Path(path).is_file():
            ui.notify("Select a valid predictions CSV.", type="warning")
            return
        try:
            df = _pd.read_csv(path, low_memory=False)
            if "indet"   not in df.columns: df["indet"]   = False
            if "flagged" not in df.columns: df["flagged"] = False
            _merge_aum(df, Path(path).parent)
            _st["df"] = df
            n_total = len(df)
            n_indet = df["indet"].astype(str).str.lower().isin(("true", "1")).sum()
            n_flag  = df["flagged"].astype(str).str.lower().isin(("true", "1")).sum()
            summary_lbl.set_text(
                f"Loaded {n_total:,} total  ·  {n_indet:,} indets  ·  {n_flag:,} flagged")
            _show_excluded(Path(path).parent)
            _apply_filter()
            ui.notify(f"Loaded {n_total:,} predictions", type="positive")
        except Exception as exc:
            ui.notify(f"Error loading CSV: {exc}", type="negative")

    async def _run_fetch() -> None:
        """Fetch the results to review from the pod or the R2 archive, then
        load them. Runs inside a _wrap_cloud_aux slot (error handling +
        serialisation). Images are pulled only when not already present."""
        gs = app.storage.general
        source = (fetch_src.value or "pod")

        if source == "pod":
            orch = _cloud.get("orch"); pod = _cloud.get("pod")
            if not (orch and pod):
                ui.notify("No active pod. Provision or attach one in ☁ Cloud, "
                          "or switch the source to R2.", type="warning")
                return
            # predictions.csv + checkpoints + nameslist (sets review_csv,
            # id_ckpt, active_ckpt, id_nl in gs).
            await _do_download_results()
            imgs = _v(rev_imgs) or gs.get("review_imgs", "")
            if imgs and Path(imgs).is_dir() and any(Path(imgs).iterdir()):
                _cloud_info(f"Images already present at {imgs} — skipping pull "
                            "(delete the folder to force a refresh).")
            else:
                await _do_download_images()          # sets review_imgs in gs
            rev_csv.value  = gs.get("review_csv",  rev_csv.value)
            rev_imgs.value = gs.get("review_imgs", rev_imgs.value)
            _load()
            return

        # ── R2 archive (rclone via restore_local.py; no pod needed) ──────────
        proj   = (gs.get("main_proj") or "").strip()
        base   = (gs.get("main_base_dir") or "").strip()
        remote = (gs.get("rl_remote") or "r2:herbarium-backup").strip()
        if not proj:
            ui.notify("Set a Project name first (Get Started).", type="warning")
            return
        target = str(Path(base) / proj) if base else (gs.get("rl_target") or "").strip()
        if not target:
            ui.notify("No target directory — set Projects root + name in "
                      "Get Started.", type="warning")
            return
        img_folder = (gs.get("main_img_folder") or "images").strip()
        imgs_dir = Path(target) / img_folder
        script = str(Path(__file__).with_name("restore_local.py"))
        # --skip-images-if-present: restore_local skips the multi-GB image pull
        # when the folder already has files, matching the pod-side rule.
        cmd = [sys.executable, "-u", script,
               "--project", proj, "--target", target, "--remote", remote,
               "--images-dirname", img_folder, "--skip-images-if-present"]
        await _launch(cmd)

        # Wire Review + local Identify at the restored layout.
        preds = Path(target) / "predictions" / "predictions.csv"
        if not preds.is_file():
            alt = Path(target) / "predictions.csv"
            if alt.is_file():
                preds = alt
        rev_csv.value  = str(preds)
        rev_imgs.value = str(imgs_dir)
        gs["review_csv"]  = rev_csv.value
        gs["review_imgs"] = rev_imgs.value
        ck_dir = Path(target) / "checkpoints"
        ckpts = (sorted(ck_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
                 if ck_dir.is_dir() else [])
        if ckpts:
            gs["id_ckpt"] = str(ckpts[-1])
            gs["active_ckpt"] = str(ckpts[-1])
        _load()

    def _go(delta: int):
        if _st["view"] is not None:
            _show(_st["idx"] + delta)

    def _open_file():
        view = _st["view"]
        if view is None or len(view) == 0:
            return
        row  = view.iloc[_st["idx"]]
        path = _resolve_path(row)
        if not path or not Path(path).is_file():
            ui.notify(f"Local image not found: {path or '(no path)'}", type="warning")
            return
        # Open via the same static route the carousel uses — works cross-platform,
        # no xdg-open / wslview dependency.
        ui.navigate.to(_review_img_url(path), new_tab=True)

    def _row_field(name: str) -> str:
        view = _st["view"]
        if view is None or len(view) == 0:
            return ""
        v = view.iloc[_st["idx"]].get(name, "")
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none") else s

    def _open_gbif():
        gid = _row_field("gbifID")
        if not gid:
            ui.notify("No gbifID on this row.", type="warning"); return
        ui.navigate.to(f"https://www.gbif.org/occurrence/{gid}", new_tab=True)

    def _open_original():
        url = _row_field("image_url")
        if not url:
            ui.notify("No image_url on this row — re-download with the new schema to populate it.",
                      type="warning")
            return
        ui.navigate.to(url, new_tab=True)

    def _write_back(op: str):
        """Shared write-back logic for confirm and mark-invalid."""
        view = _st["view"]
        if view is None or len(view) == 0:
            return False
        row          = view.iloc[_st["idx"]]
        specsin_file = str(row.get("specsin_file", ""))
        fname        = str(row.get("fname", ""))
        if not specsin_file or not Path(specsin_file).is_file():
            action_lbl.set_text(
                "No specsin_file in CSV — re-run Identify with updated pipeline")
            ui.notify("specsin_file not available", type="warning")
            return False
        try:
            sp   = _pd.read_csv(specsin_file, index_col=0)
            mask = sp["fname"] == fname
            if not mask.any():
                action_lbl.set_text(f"fname not found in {Path(specsin_file).name}")
                return False
            if op == "invalid":
                sp.loc[mask, "invalid"] = True
            elif level_sel.value == "genus":
                # A genus determination is a real curatorial act, not a
                # half-finished species one: the sheet is placed in a genus but
                # NOT to species. So it is recorded as "Psychotria sp." and stays
                # indet — which also keeps it out of training, where a bare genus
                # must never become a species class (train_herbarium drops these).
                new_name = str(det_sel.value or "").split()[0]
                if not new_name:
                    return False
                current = str(sp.loc[mask, "species"].iloc[0])
                sp.loc[mask, "old_determination"] = current
                sp.loc[mask, "species"] = f"{new_name} sp."
                if "genus" in sp.columns:
                    sp.loc[mask, "genus"] = new_name
                if "verbatimName" in sp.columns:
                    sp.loc[mask, "verbatimName"] = f"{new_name}_sp."
                sp.loc[mask, "indet"] = True
            else:  # species determination
                new_name = det_sel.value
                current  = str(sp.loc[mask, "species"].iloc[0])
                sp.loc[mask, "old_determination"] = current
                sp.loc[mask, "species"]           = new_name
                if "genus" in sp.columns:
                    sp.loc[mask, "genus"] = new_name.split()[0]
                if "verbatimName" in sp.columns:
                    sp.loc[mask, "verbatimName"] = new_name.replace(" ", "_")
                sp.loc[mask, "indet"] = False
            sp.to_csv(specsin_file)
            return True
        except Exception as exc:
            action_lbl.set_text(f"Error: {exc}")
            ui.notify(str(exc), type="negative")
            return False

    def _confirm():
        new_name = det_sel.value
        if not new_name:
            return
        # Show what actually lands in specsin, not what was clicked — a genus
        # determination is written as "Psychotria sp.", and the reviewer should
        # see that rather than discover it later in the CSV.
        written = (f"{str(new_name).split()[0]} sp."
                   if level_sel.value == "genus" else new_name)
        if _write_back("determine"):
            action_lbl.set_text(f"Determined → {written}")
            ui.notify(f"Determined: {written}", type="positive")

    def _mark_invalid():
        if _write_back("invalid"):
            view = _st["view"]
            fname = str(view.iloc[_st["idx"]].get("fname", ""))
            action_lbl.set_text(f"Marked invalid: {Path(fname).name}")
            ui.notify(f"Marked invalid: {Path(fname).name}", type="warning")

    load_btn.on_click(_load)
    apply_btn.on_click(_apply_filter)
    ai_filter_btn.on_click(_run_ai_filter)
    ai_filter_inp.on("keydown.enter", _run_ai_filter)
    carousel_btn.on_click(lambda: ui.navigate.to("/review-carousel", new_tab=True))

    return rev_csv, rev_imgs


def _build_confusion() -> "ui.input":
    """Tab: confusion matrix for identified specimens in a predictions CSV."""
    import copy
    import pandas as _pd

    # Render caps. Both charts scale with the number of classes, and a
    # species-level predictions CSV for a whole flora carries ~15k of them —
    # far past what a browser tab can lay out or a websocket frame can carry.
    _MAX_MATRIX_CLASSES = 150   # heatmap is O(n²) cells
    _MAX_ACC_BARS       = 80    # one 16px row + one axis label each

    _section("Predictions CSV")
    conf_csv = _path_input("predictions.csv:", mode="file",
                           value=app.storage.general.get("review_csv", ""))
    conf_csv.bind_value(app.storage.general, "review_csv")

    _section("Options")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        with ui.row().classes("items-center gap-2"):
            ui.label("Level:").classes("text-sm")
            level_sel = ui.toggle(
                {"species": "Species", "genus": "Genus", "family": "Family"},
                value="species").props("dense")
        with ui.row().classes("items-center gap-1"):
            ui.label("Top N most-confused (0 = all):").classes("text-sm")
            top_n_inp = ui.input(value="20").classes("w-20").props("dense outlined")
        with ui.row().classes("items-center gap-1"):
            ui.label("Min samples:").classes("text-sm")
            min_s_inp = ui.input(value="5").classes("w-20").props("dense outlined")
        with ui.row().classes("items-center gap-1"):
            ui.label("Min confusions:").classes("text-sm")
            min_c_inp = ui.input(value="2").classes("w-20").props("dense outlined")
        norm_chk = ui.checkbox("Normalise rows (recall)", value=True)

    load_btn    = (ui.button("Load & Plot", icon="bar_chart")
                   .props("color=primary unelevated").classes("mt-3"))
    status_lbl  = ui.label("").classes("text-caption text-grey-7 mt-1")
    metrics_html = ui.html("").classes("w-full mt-2")

    # ECharts heatmap — populated on load
    _BASE = {
        "tooltip": {
            "position": "top",
            "confine": True,
        },
        "grid":  {"top": "5%", "bottom": "20%", "left": "25%", "right": "5%"},
        "xAxis": {"type": "category", "data": [], "splitArea": {"show": True},
                  "name": "Predicted", "nameLocation": "middle", "nameGap": 55,
                  "axisLabel": {"rotate": 45, "fontSize": 10, "interval": 0}},
        "yAxis": {"type": "category", "data": [], "splitArea": {"show": True},
                  "inverse": True,
                  "name": "True", "nameLocation": "middle", "nameGap": 140,
                  "axisLabel": {"fontSize": 10, "interval": 0}},
        "visualMap": {
            "min": 0, "max": 1,
            "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": "2%",
            "inRange": {"color": ["#f5f5f5", "#009688"]},
        },
        "series": [{
            "type": "heatmap",
            "data": [],
            "label": {"show": False, "fontSize": 9},
            "emphasis": {"itemStyle": {"shadowBlur": 6, "shadowColor": "rgba(0,0,0,.3)"}},
        }],
    }
    chart = ui.echart(copy.deepcopy(_BASE)).style("height:600px; width:100%")

    # Accuracy bar chart — generic title; the per-level label sits on the X-axis.
    _section("Accuracy by class")
    _ACC_BASE = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                    "confine": True,
                    ":formatter": (
                        "(function(){return function(p){"
                        "return p[0].name+'<br>Accuracy: <b>'+p[0].value+'%</b>';"
                        "}})()"
                    )},
        "grid":  {"top": "2%", "bottom": "8%", "left": "35%", "right": "8%"},
        "xAxis": {"type": "value", "min": 0, "max": 100,
                  "name": "Accuracy (%)", "nameLocation": "middle", "nameGap": 25,
                  "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "category", "data": [],
                  "axisLabel": {"fontSize": 9, "interval": 0}},
        "series": [{"type": "bar", "data": [],
                    "itemStyle": {":color": (
                        "(function(){return function(p){"
                        "var v=p.value;"
                        "var r=Math.round(220-v*1.5),g=Math.round(100+v*1.3),b=80;"
                        "return 'rgb('+r+','+g+','+b+')';"
                        "}})()"
                    )}}],
    }
    acc_chart = ui.echart(copy.deepcopy(_ACC_BASE)).style("height:400px; width:100%")

    # Most-confused list
    _section("Most Confused Pairs")
    confused_html = ui.html("").classes("w-full")

    def _load():
        path = _v(conf_csv)
        if not path or not Path(path).is_file():
            ui.notify("Select a valid predictions CSV.", type="warning")
            return
        try:
            df = _pd.read_csv(path)
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")
            return

        # Keep only identified specimens with known true label
        indet_mask = df.get("indet", _pd.Series(False, index=df.index)
                            ).astype(str).str.lower().isin(("true", "1"))
        df = df[~indet_mask].copy()

        level = level_sel.value
        # Resolve the (true_col, pred_col) pair lazily per level so a
        # family-only predictions CSV (no pred_species / true_species) plots
        # correctly when level=family. Each branch checks only the columns
        # it actually needs.
        pred_sp_col = "pred_species" if "pred_species" in df.columns else (
                      "top1_name" if "top1_name" in df.columns else None)

        def _missing(*cols: str) -> str:
            return ", ".join(c for c in cols if c not in df.columns)

        if level == "family":
            if "true_family" not in df.columns:
                ui.notify("Family-level analysis needs the 'true_family' "
                          "column. Re-run Identify after upgrading specsin "
                          "with the family field.", type="warning")
                return
            df["_true"] = df["true_family"].astype(str).str.strip()
            # Prefer pred_family when populated; fall back to pred_species /
            # top1_name for legacy CSVs where a family-level identify run
            # mistakenly wrote family names into the species column.
            fam_col = None
            for cand in ("pred_family", "pred_species", "top1_name"):
                if cand in df.columns:
                    s = df[cand].astype(str).str.strip().replace("nan", "")
                    if (s != "").any():
                        # Heuristic: do these values overlap the known true
                        # families? If yes, this column holds family-rank
                        # predictions and is safe to use.
                        true_fams = set(df["true_family"].astype(str).str.strip())
                        if (s.isin(true_fams)).any():
                            fam_col = cand
                            break
            if fam_col is None:
                ui.notify("No usable predicted-family column found "
                          "(checked pred_family, pred_species, top1_name). "
                          "Re-run Identify.", type="warning")
                return
            df["_pred"] = df[fam_col].astype(str).str.strip()
            level_label = "Family"
            if fam_col != "pred_family":
                ui.notify(f"Using legacy '{fam_col}' as family predictions "
                          f"(pred_family was empty/missing).", type="info")
        elif level == "genus":
            # Prefer explicit genus columns when present (hierarchical model);
            # otherwise derive genus = first word of species.
            has_genus_cols = ("true_genus" in df.columns and
                              "pred_genus" in df.columns)
            if has_genus_cols:
                df["_true"] = df["true_genus"].astype(str).str.strip()
                df["_pred"] = df["pred_genus"].astype(str).str.strip()
            else:
                if "true_species" not in df.columns or pred_sp_col is None:
                    ui.notify("Genus-level analysis needs either "
                              "(true_genus + pred_genus) or "
                              "(true_species + pred_species).",
                              type="warning")
                    return
                df["_true"] = df["true_species"].astype(str).str.split().str[0]
                df["_pred"] = df[pred_sp_col].astype(str).str.split().str[0]
            level_label = "Genus"
        else:
            if "true_species" not in df.columns or pred_sp_col is None:
                ui.notify("Species-level analysis needs 'true_species' and "
                          "'pred_species' (or 'top1_name'). This CSV looks "
                          "like a family-only predictions file — switch the "
                          "Level toggle to Family.", type="warning")
                return
            df["_true"] = df["true_species"].astype(str).str.strip()
            df["_pred"] = df[pred_sp_col].astype(str).str.strip()
            level_label = "Species"

        true_col = "_true"
        pred_col = "_pred"
        # Keep the full frame around as df_full so the Top-5 block can still
        # see true_species / top{k}_* columns. The narrow `df` is just for
        # the heatmap and per-class accuracy.
        df_full = df.copy()
        df = df[[true_col, pred_col]].dropna()
        df = df[df[true_col].str.strip().replace("nan", "") != ""]
        df = df[df[pred_col].str.strip().replace("nan", "") != ""]
        # Apply the same row filtering to df_full so the two stay aligned.
        df_full = df_full.loc[df.index]

        # Min-samples filter
        min_s = max(1, int(min_s_inp.value or 1))
        vc = df[true_col].value_counts()
        df = df[df[true_col].isin(vc[vc >= min_s].index)]
        df_full = df_full.loc[df.index]

        if df.empty:
            ui.notify("No identified specimens after filtering.", type="info")
            return

        # Full crosstab; kept so the top-N view can pull any (true, pred) cell.
        full_ct = _pd.crosstab(df[true_col], df[pred_col])

        # Pick which classes to display, then render a SQUARE matrix that uses
        # ONE ordered class list for BOTH axes — so each class sits at the same
        # position on each axis and true==pred always lands on the main diagonal.
        # (The axes used to be built from different sets in different orders, so
        # the diagonal was scattered and unreadable.)
        n = int(top_n_inp.value or 0)
        if n > 0:
            off = full_ct.copy().astype(float)
            for sp in full_ct.index:
                if sp in full_ct.columns:
                    off.loc[sp, sp] = 0.0
            error_rate = (off.sum(axis=1) / full_ct.sum(axis=1)).sort_values(ascending=False)
            top_true = list(error_rate.head(n).index)
            # Also include the classes those confused specimens were predicted
            # AS, so the off-diagonal destinations stay visible.
            dest = full_ct.loc[top_true].sum(axis=0).sort_values(ascending=False)
            classes = top_true + [c for c in dest.index if c not in top_true][:n]
        else:
            classes = sorted(set(full_ct.index) | set(full_ct.columns))

        # Hard cap the matrix. A species-level run over a big flora has ~15k
        # classes; "all" would be 15k² = 2.4e8 cells, and even serialising that
        # to JSON kills the browser tab (which then reconnects and re-renders,
        # giving an endless reload loop rather than an error). Keep the classes
        # with the most support so the view stays representative.
        if len(classes) > _MAX_MATRIX_CLASSES:
            support = full_ct.sum(axis=1)
            keep = set(support.reindex(classes).fillna(0)
                       .sort_values(ascending=False)
                       .head(_MAX_MATRIX_CLASSES).index)
            dropped = len(classes) - len(keep)
            classes = [c for c in classes if c in keep]
            ui.notify(f"{dropped:,} classes hidden — showing the "
                      f"{len(classes)} best-represented of "
                      f"{dropped + len(classes):,}. Use 'Top N most-confused' "
                      f"to choose a focused subset instead.", type="warning")

        ct = full_ct.reindex(index=classes, columns=classes, fill_value=0)

        # Normalise rows to recall (fraction of true class)
        if norm_chk.value:
            row_sums = ct.sum(axis=1).replace(0, 1)
            ct_plot  = ct.div(row_sums, axis=0).round(3)
            vmax     = 1.0
        else:
            ct_plot = ct.astype(float)
            vmax    = float(ct_plot.values.max()) or 1.0

        true_labels = list(ct_plot.index)
        pred_labels = list(ct_plot.columns)

        # Simple array data — most compatible ECharts heatmap format. Emit only
        # non-zero cells: a confusion matrix is ~99% zeros, and ECharts renders
        # missing cells as empty anyway, so the dense form just inflates the
        # websocket payload quadratically for no visual difference.
        vals = ct_plot.values
        data = [
            [j, i, round(float(vals[i, j]), 4)]
            for i in range(len(true_labels))
            for j in range(len(pred_labels))
            if vals[i, j]
        ]

        # Resize chart height to fit
        cell_px = max(14, min(28, 560 // max(len(true_labels), 1)))
        height  = max(350, len(true_labels) * cell_px + 160)
        chart.style(f"height:{height}px; width:100%")

        # Tooltip: NiceGUI only evaluates string values as JS when the key starts with ":".
        # Use ":formatter" so convertDynamicProperties() evals the IIFE to a function.
        pred_js = str(pred_labels).replace("'", '"')
        true_js = str(true_labels).replace("'", '"')
        val_fmt = "+(v*100).toFixed(1)+'%'" if norm_chk.value else "+String(v)"
        val_lbl = "'Recall: '" if norm_chk.value else "'Count: '"
        chart.options["tooltip"].pop("formatter", None)   # remove any plain string left over
        chart.options["tooltip"][":formatter"] = (
            f"(function(){{"
            f"var pred={pred_js};var tr={true_js};"
            f"return function(p){{"
            f"if(p.value[2]===undefined)return '';"
            f"var v=p.value[2];"
            f"return 'True: <b>'+tr[p.value[1]]+'</b><br>Pred: <b>'+pred[p.value[0]]+'</b><br>'+{val_lbl}{val_fmt};"
            f"}}}})();"
        )

        chart.options["xAxis"]["data"]          = pred_labels
        chart.options["xAxis"]["name"]          = f"Predicted {level_label}"
        chart.options["yAxis"]["data"]          = true_labels
        chart.options["yAxis"]["name"]          = f"True {level_label}"
        chart.options["visualMap"]["max"]       = vmax
        chart.options["series"][0]["data"]      = data
        chart.options["series"][0]["label"]["show"] = (
            max(len(true_labels), len(pred_labels)) <= 25)
        chart.update()

        from sklearn.metrics import precision_score, recall_score, f1_score

        total   = len(df)
        correct = int((df[true_col] == df[pred_col]).sum())
        acc     = correct / total

        y_true = df[true_col].tolist()
        y_pred = df[pred_col].tolist()
        prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec    = recall_score(   y_true, y_pred, average="macro", zero_division=0)
        f1     = f1_score(       y_true, y_pred, average="macro", zero_division=0)

        # Top-5 accuracy: is the true label among the top-5 predictions?
        # Source columns differ per level:
        #   species → top{k}_name (the species columns; the model's own top-5)
        #   genus   → top{k}_genus if present, else first word of top{k}_name
        #   family  → top{k}_family if present (hierarchical model) — otherwise
        #             top-5 isn't meaningful at the family rank, so we skip it.
        top5_acc = None
        if level == "family":
            top5_cols = [f"top{k}_family" for k in range(1, 6)
                         if f"top{k}_family" in df_full.columns]
            # Legacy fallback: family-level run may have written family names
            # into the top{k}_name columns instead of top{k}_family.
            if not top5_cols:
                cand = [f"top{k}_name" for k in range(1, 6)
                        if f"top{k}_name" in df_full.columns]
                if cand and "true_family" in df_full.columns:
                    true_fams = set(df_full["true_family"].astype(str).str.strip())
                    if df_full[cand[0]].astype(str).str.strip().isin(true_fams).any():
                        top5_cols = cand
            if top5_cols and "true_family" in df_full.columns:
                true_series = df_full["true_family"].astype(str).str.strip()
                hit = _pd.Series(False, index=df_full.index)
                for col in top5_cols:
                    hit |= (true_series == df_full[col].astype(str).str.strip())
                top5_acc = hit.mean()
        elif level == "genus":
            top5_cols = [f"top{k}_genus" for k in range(1, 6)
                         if f"top{k}_genus" in df_full.columns]
            if top5_cols and "true_genus" in df_full.columns:
                true_series = df_full["true_genus"].astype(str).str.strip()
                hit = _pd.Series(False, index=df_full.index)
                for col in top5_cols:
                    hit |= (true_series == df_full[col].astype(str).str.strip())
                top5_acc = hit.mean()
            else:
                # Derive from species top-5 when the run wasn't hierarchical.
                top5_cols = [f"top{k}_name" for k in range(1, 6)
                             if f"top{k}_name" in df_full.columns]
                if top5_cols and "true_species" in df_full.columns:
                    true_series = df_full["true_species"].astype(str).str.split().str[0]
                    hit = _pd.Series(False, index=df_full.index)
                    for col in top5_cols:
                        hit |= (true_series == df_full[col].astype(str).str.split().str[0])
                    top5_acc = hit.mean()
        else:  # species
            top5_cols = [f"top{k}_name" for k in range(1, 6)
                         if f"top{k}_name" in df_full.columns]
            if top5_cols and "true_species" in df_full.columns:
                true_series = df_full["true_species"].astype(str).str.strip()
                hit = _pd.Series(False, index=df_full.index)
                for col in top5_cols:
                    hit |= (true_series == df_full[col].astype(str).str.strip())
                top5_acc = hit.mean()

        status_lbl.set_text(
            f"{total:,} identified  ·  "
            f"{correct:,} correct ({acc:.1%})  ·  "
            f"matrix: {len(true_labels)} true × {len(pred_labels)} predicted"
        )

        def _metric(label, val, color):
            return (f"<div style='text-align:center;padding:8px 18px;"
                    f"background:{color}18;border-radius:8px;border:1px solid {color}44'>"
                    f"<div style='font-size:22px;font-weight:bold;color:{color}'>{val:.1%}</div>"
                    f"<div style='font-size:11px;color:#666;margin-top:2px'>{label}</div>"
                    f"</div>")

        metrics_html.set_content(
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin:8px 0'>"
            + _metric("Accuracy",          acc,  "#1976d2")
            + ((_metric("Top-5 Accuracy",  top5_acc, "#0097a7")) if top5_acc is not None else "")
            + _metric("Precision (macro)", prec, "#388e3c")
            + _metric("Recall (macro)",    rec,  "#f57c00")
            + _metric("F1 (macro)",        f1,   "#7b1fa2")
            + f"<div style='align-self:center;font-size:12px;color:#888;margin-left:4px'>"
              f"{level_label} level · {len(_pd.Series(y_true).unique())} classes</div>"
            + "</div>"
        )

        # ── Accuracy bar chart ──────────────────────────────────────────────
        # Vectorised per-class accuracy. Avoids a `groupby(col).apply(lambda g:
        # g[col]==…)` pattern that breaks under pandas ≥2.2 (group columns
        # excluded from the lambda's frame by default → silent KeyError →
        # empty chart).
        matches = (df[true_col] == df[pred_col]).astype(float)
        acc_df = (matches.groupby(df[true_col]).mean()
                          .mul(100)
                          .sort_values(ascending=True))
        # Unlike the heatmap this chart is NOT subject to the Top-N control, so
        # it plots every class — thousands of them on a species-level run over a
        # big flora, giving a ~50,000px element with thousands of interval:0
        # axis labels. That is what crashed the tab. Show the worst performers
        # (sorted ascending, so the head) and say how many were left out.
        n_classes = len(acc_df)
        if n_classes > _MAX_ACC_BARS:
            acc_df = acc_df.head(_MAX_ACC_BARS)
        acc_labels = acc_df.index.tolist()
        acc_values = [round(v, 1) for v in acc_df.values]
        acc_chart_title = (
            f"worst {len(acc_labels)} of {n_classes:,} classes"
            if n_classes > _MAX_ACC_BARS else f"all {n_classes:,} classes")
        acc_height = max(300, len(acc_labels) * 16 + 80)
        acc_chart.style(f"height:{acc_height}px; width:100%")
        acc_chart.options["yAxis"]["data"] = acc_labels
        acc_chart.options["series"][0]["data"] = acc_values
        acc_chart.options["xAxis"]["name"] = (
            f"{level_label} accuracy (%) — {acc_chart_title}")
        acc_chart.update()

        # ── Most confused list ──────────────────────────────────────────────
        min_c = max(1, int(min_c_inp.value or 1))
        confused_df = df[df[true_col] != df[pred_col]]
        pairs = (confused_df.groupby([true_col, pred_col])
                 .size()
                 .reset_index(name="count")
                 .query("count >= @min_c")
                 .sort_values("count", ascending=False))
        if pairs.empty:
            confused_html.set_content(
                "<p style='color:#888;font-size:13px'>No confusions above threshold.</p>")
        else:
            rows_html = "".join(
                f"<tr><td style='padding:2px 10px'>{r[true_col]}</td>"
                f"<td style='padding:2px 6px;color:#888'>→</td>"
                f"<td style='padding:2px 10px'><i>{r[pred_col]}</i></td>"
                f"<td style='padding:2px 8px;text-align:right;color:#555'>{r['count']}</td></tr>"
                for _, r in pairs.iterrows()
            )
            confused_html.set_content(
                f"<table style='font-size:12px;font-family:monospace;border-collapse:collapse'>"
                f"<thead><tr>"
                f"<th style='padding:2px 10px;text-align:left'>True</th>"
                f"<th></th>"
                f"<th style='padding:2px 10px;text-align:left'>Predicted</th>"
                f"<th style='padding:2px 8px;text-align:right'>Count</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>"
            )

    load_btn.on_click(_load)
    return conf_csv


# ---------------------------------------------------------------------------
# Setup tab — one-time configuration: credentials, SSH key, environment.
# Everything here persists across runs; once the four sections show ✓ the
# user shouldn't need to revisit this tab.
# ---------------------------------------------------------------------------

def _cred_present(getter) -> bool:
    """True if a keyring getter returns a credential, swallowing keyring errors."""
    try:
        return bool(getter())
    except Exception:
        return False


def _project_artifacts(base: str, name: str, img_folder: str) -> dict:
    """Which pipeline artifacts exist for <base>/<name>/ — drives the progress
    rail. Steps that leave no local trace (archive/publish) return None."""
    proj = (Path(base) / name) if (base and name) else None

    def _exists(*parts) -> bool:
        return bool(proj) and proj.joinpath(*parts).exists()

    def _nonempty_dir(*parts) -> bool:
        p = proj.joinpath(*parts) if proj else None
        try:
            return bool(p) and p.is_dir() and any(p.iterdir())
        except OSError:
            return False

    ckpt_dir = (proj / "runs" / "checkpoints") if proj else None
    has_ckpt = bool(ckpt_dir and ckpt_dir.is_dir() and any(ckpt_dir.glob("*.ckpt")))
    has_preds = _exists("review", "predictions.csv")

    return {
        "download": _exists("specsin.csv"),
        "clean":    _nonempty_dir(img_folder or "images_cropped"),
        "train":    has_ckpt,
        "identify": has_preds,
        "review":   has_preds,   # available once there's something to review
        "archive":  None,        # lives on R2 — not locally detectable
        "publish":  None,        # lives on the Hub — not locally detectable
    }


def _build_get_started_landing(gs) -> None:
    """Orientation + project setup + at-a-glance status and progress.

    The credential/environment cards proper are built afterwards by
    _build_setup(); this landing section sits on top of them so a first-time
    user sees the workflow, a place to make a project, and what's done."""

    with ui.row().classes("w-full items-baseline gap-2 mb-1"):
        ui.icon("rocket_launch").style("color:#00897b;font-size:26px")
        ui.label("Get Started").classes("text-h6").style("color:#00695c")
    ui.label(
        "The pipeline runs left to right — download specimen images, clean them, "
        "train a model, identify new specimens, review the results, then archive "
        "and publish. Create or pick a project below to begin; everything else is "
        "credentials you set once."
    ).classes("text-body2").style("color:#455a64;max-width:900px")

    # ── Flow chips ────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center gap-1 flex-wrap my-2"):
        for i, (_key, title, _desc) in enumerate(_STEP_FLOW):
            _pill(title, "neutral")
            if i < len(_STEP_FLOW) - 1:
                ui.icon("chevron_right").classes("text-grey-5")

    # ── Project card ──────────────────────────────────────────────────────
    with ui.card().classes("w-full").style("border-left:3px solid #00897b"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("create_new_folder").style("color:#00897b;font-size:20px")
            ui.label("Project").classes("text-subtitle1 font-bold").style("color:#00695c")
        ui.label(
            "Enter a family (and optional region). We create the folder and wire "
            "every step's paths to it — no hand-editing. These fields stay in sync "
            "with the Projects root / name in the header."
        ).classes("text-caption text-grey-7")

        with ui.row().classes("w-full items-center gap-2 mt-1"):
            root_inp = (ui.input(label="Projects root",
                                 value=gs.get("main_base_dir") or str(Path.home()))
                        .classes("flex-1").props("dense outlined")
                        .bind_value(gs, "main_base_dir"))

            async def _browse_root() -> None:
                r = await FilePicker(root_inp.value or str(Path.home()), mode="dir")
                if r:
                    root_inp.value = r

            ui.button(icon="folder_open", on_click=_browse_root
                      ).props("flat dense round").tooltip("Browse")

        with ui.row().classes("w-full items-center gap-2 mt-1"):
            fam_inp = (ui.input(label="Family", placeholder="e.g. Ebenaceae")
                       .classes("flex-1").props("dense outlined"))
            reg_inp = (ui.input(label="Region (optional)", placeholder="e.g. Africa")
                       .classes("flex-1").props("dense outlined"))
            name_inp = (ui.input(label="Project name")
                        .classes("flex-1").props("dense outlined")
                        .bind_value(gs, "main_proj"))

        # Suggest a project name from family (+ region) unless the user has
        # already typed one. Fires when they leave the family/region field.
        def _suggest_name() -> None:
            fam = _v(fam_inp)
            if not fam or _v(name_inp):
                return
            parts = [p for p in (_v(reg_inp), fam) if p]
            name_inp.value = "-".join(parts).lower().replace(" ", "-")

        fam_inp.on("blur", lambda e: _suggest_name())
        reg_inp.on("blur", lambda e: _suggest_name())

        def _create_project() -> None:
            base = _v(root_inp)
            name = _v(name_inp) or _v(fam_inp)
            if not base:
                ui.notify("Enter a Projects root.", type="warning"); return
            if not name:
                ui.notify("Enter a Family or Project name.", type="warning"); return
            img_folder = (gs.get("main_img_folder") or "images_cropped").strip()
            proj = Path(base) / name
            try:
                (proj / img_folder).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                ui.notify(f"Couldn't create {proj}: {e}", type="negative"); return
            name_inp.value = name
            apply = _page_hooks.get("apply_paths")
            if apply:
                apply(base=base, name=name, img_folder=img_folder)
            else:
                ui.notify(f"Project folder ready: {proj}", type="positive")
            _refresh()

        with ui.row().classes("w-full gap-2 mt-2 items-center"):
            ui.button("Create / open project", icon="check_circle",
                      on_click=_create_project).props("unelevated color=primary")
            ui.label("or").classes("text-caption text-grey-6")
            ui.button("Pick from RunPod volumes", icon="storage",
                      on_click=lambda: _pick_volume()
                      ).props("outlined color=primary")\
                .tooltip("List the network volumes on your RunPod account and "
                         "set the Project name from the one you pick — the "
                         "volume is the durable identity, not a locally-typed "
                         "name, so this can't drift between machines the way "
                         "a hand-typed project name can.")

        async def _pick_volume() -> None:
            api_key = cloud_secrets.get_runpod_api_key()
            if not api_key:
                ui.notify("Save your RunPod API key first (below).", type="warning")
                return
            from cloud.runpod_client import RunPodClient
            try:
                async with RunPodClient(api_key) as rp:
                    volumes = await rp.list_volumes()
                    pods = await rp.list_pods()
            except Exception as e:
                ui.notify(f"Couldn't fetch volumes: {e}", type="negative")
                return
            if not volumes:
                ui.notify("No network volumes on this RunPod account.", type="warning")
                return
            pods_by_volume: dict[str, list] = {}
            for p in pods:
                vid = p.network_volume_id or (p.raw or {}).get("networkVolumeId")
                if vid and p.desired_status == "RUNNING":
                    pods_by_volume.setdefault(vid, []).append(p)

            dialog = ui.dialog()
            with dialog, ui.card().classes("w-full").style("max-width:640px"):
                ui.label("Pick a network volume").classes("text-subtitle1 font-bold")
                ui.label(
                    "Sets Project name to the volume's own name, so this "
                    "machine's local state can't silently drift from which "
                    "volume you're actually using."
                ).classes("text-caption text-grey-7 mb-2")
                with ui.column().classes("w-full gap-1")\
                        .style("max-height:400px;overflow-y:auto"):
                    for v in volumes:
                        live = pods_by_volume.get(v.id)
                        with ui.row().classes("w-full items-center gap-2 pa-2")\
                                .style("border:1px solid #e0e0e0;border-radius:6px"):
                            with ui.column().classes("gap-0 flex-1"):
                                ui.label(v.name).classes("font-medium")
                                ui.label(f"{v.id} · {v.size_gb} GB · "
                                         f"{v.data_center_id or '?'}")\
                                    .classes("text-caption text-grey-6")
                                if live:
                                    ui.label(f"● live pod: {live[0].id}")\
                                        .classes("text-caption")\
                                        .style("color:#2e7d32")
                            ui.button("Select", on_click=(
                                lambda v=v, live=live: _select_volume(v, live, dialog)
                            )).props("dense unelevated color=primary")
                with ui.row().classes("w-full justify-end mt-2"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
            dialog.open()

        def _project_from_volume_name(name: str) -> str:
            """Invert the orchestrator's ``herb-<project>`` volume naming.

            Taking the volume name verbatim double-prefixes on the next
            provision: picking 'herb-Salacia' set project='herb-Salacia',
            whose volume is named 'herb-herb-Salacia' — which didn't exist,
            so a second, empty volume was created and the real one stranded.
            """
            return name[len("herb-"):] if name.startswith("herb-") else name

        def _select_volume(vol, live_pods, dialog) -> None:
            dialog.close()
            proj = _project_from_volume_name(vol.name)
            # Bind the chosen volume by id, not just by name. The id is the
            # durable identity the picker exists to capture; deriving it back
            # from a typed project name is what drifts.
            try:
                st = cloud_state.load(proj)
                if st.volume_id != vol.id:
                    st.volume_id = vol.id
                    st.data_center_id = vol.data_center_id or st.data_center_id
                    cloud_state.save(st)
            except Exception as e:
                ui.notify(f"Couldn't record volume {vol.id} in local state: {e}",
                          type="warning")
            old = (gs.get("main_proj") or "").strip()
            gs["main_proj"] = proj
            if old and old != proj:
                ui.notify(f"Project switched: {old!r} → {proj!r} "
                          f"(from volume {vol.name!r}, {vol.id}).", type="info")
            base = (gs.get("main_base_dir") or "").strip() or str(Path.home())
            gs["main_base_dir"] = base
            name_inp.value = proj
            img_folder = (gs.get("main_img_folder") or "images_cropped").strip()
            apply = _page_hooks.get("apply_paths")
            if apply:
                apply(base=base, name=proj, img_folder=img_folder)
            if live_pods:
                gs["cloud_attach_pod_id"] = live_pods[0].id
                ui.notify(f"Volume {vol.name!r} selected — attaching to live "
                          f"pod {live_pods[0].id}...", type="positive")
                _wrap_cloud(_do_attach, force=True)
            else:
                ui.notify(f"Volume {vol.name!r} selected. No live pod for it — "
                          f"Provision when ready.", type="positive")
            _refresh()

    # ── Status strip + progress rail (rebuilt by _refresh) ────────────────
    status_card = ui.card().classes("w-full").style("border-left:3px solid #00897b")
    with status_card:
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("checklist").style("color:#00897b;font-size:20px")
            ui.label("Status").classes("text-subtitle1 font-bold").style("color:#00695c")
            ui.button(icon="refresh", on_click=lambda: _refresh()
                      ).props("flat dense round").classes("ml-auto")\
                .tooltip("Re-check credentials and project progress")
        status_row = ui.row().classes("w-full items-center gap-2 flex-wrap mt-1")
        ui.separator().classes("my-2")
        rail_col = ui.column().classes("w-full gap-1")

    def _refresh() -> None:
        # -- credential / mode / env pills --
        status_row.clear()
        cloud = (gs.get("main_mode", "cloud") == "cloud")
        with status_row:
            _pill("☁ Cloud" if cloud else "💻 Local", "ok" if cloud else "neutral")
            for label, ok, optional in [
                ("RunPod",       _cred_present(cloud_secrets.get_runpod_api_key), False),
                ("WandB",        _cred_present(cloud_secrets.get_wandb_api_key),  True),
                ("Hugging Face", _cred_present(cloud_secrets.get_hf_token),       False),
                ("R2",           _cred_present(cloud_secrets.get_r2_credentials), False),
                ("GBIF",         _cred_present(cloud_secrets.get_gbif_credentials), True),
            ]:
                kind = "ok" if ok else ("neutral" if optional else "warn")
                mark = "✓" if ok else ("·" if optional else "—")
                _pill(f"{label} {mark}", kind)

        # -- progress rail --
        base = (gs.get("main_base_dir") or "").strip()
        name = (gs.get("main_proj") or "").strip()
        img_folder = (gs.get("main_img_folder") or "images_cropped").strip()
        done = _project_artifacts(base, name, img_folder)
        rail_col.clear()
        with rail_col:
            if not (base and name):
                ui.label("Create or pick a project to track progress.")\
                    .classes("text-caption text-grey-6")
            else:
                ui.label(f"Project: {Path(base) / name}")\
                    .classes("text-caption text-grey-7 mb-1")
            refs = _page_hooks.get("tab_refs") or {}
            goto = _page_hooks.get("goto_tab")
            for key, title, desc in _STEP_FLOW:
                state = done.get(key)
                if state is True:
                    icon, color = "check_circle", "#2e7d32"
                elif state is False:
                    icon, color = "radio_button_unchecked", "#b0bec5"
                else:
                    icon, color = "remove_circle_outline", "#cfd8dc"
                with ui.row().classes("w-full items-center gap-2"):
                    ui.icon(icon).style(f"color:{color};font-size:20px")
                    ui.label(title).classes("font-medium").style("width:120px")
                    ui.label(desc).classes("text-caption text-grey-7 flex-1")
                    if goto and key in refs:
                        ui.button("Open",
                                  on_click=lambda k=key: goto(refs[k]))\
                            .props("flat dense color=primary").classes("shrink-0")

    _refresh()
    # _page_hooks (tab navigation) is populated only after the whole page is
    # built, which happens after this function returns. Re-render once shortly
    # after so the rail's per-step "Open" buttons appear on first load.
    ui.timer(0.1, _refresh, once=True)


def _collect_creds() -> dict:
    """Gather saved credentials from the OS keyring into a plain dict for
    export. Only includes slots that are actually set."""
    c: dict = {}
    if (k := cloud_secrets.get_runpod_api_key()): c["runpod"] = k
    if (k := cloud_secrets.get_wandb_api_key()):  c["wandb"] = k
    if (k := cloud_secrets.get_hf_token()):       c["huggingface"] = k
    if (r2 := cloud_secrets.get_r2_credentials()):
        c["r2"] = {"account_id": r2.account_id,
                   "access_key_id": r2.access_key_id,
                   "secret_access_key": r2.secret_access_key,
                   "bucket": r2.bucket}
    if (g := cloud_secrets.get_gbif_credentials()):
        c["gbif"] = {"username": g.username, "password": g.password}
    return c


def _restore_creds(c: dict) -> int:
    """Write imported credentials back into the OS keyring. Returns the count
    restored. Best-effort — a malformed blob warns rather than aborting."""
    n = 0
    try:
        if c.get("runpod"):      cloud_secrets.set_runpod_api_key(c["runpod"]); n += 1
        if c.get("wandb"):       cloud_secrets.set_wandb_api_key(c["wandb"]); n += 1
        if c.get("huggingface"): cloud_secrets.set_hf_token(c["huggingface"]); n += 1
        if c.get("r2"):
            cloud_secrets.set_r2_credentials(
                cloud_secrets.R2Credentials(**c["r2"])); n += 1
        if c.get("gbif"):
            cloud_secrets.set_gbif_credentials(
                cloud_secrets.GBIFCredentials(**c["gbif"])); n += 1
    except Exception as e:
        ui.notify(f"Some credentials couldn't be imported: {e}", type="warning")
    return n


def _build_portability(gs) -> None:
    """Export / import UI settings (and, opt-in, credentials) so a project can
    be moved to another machine without hand-re-entering everything."""
    card, _ = _setup_card("swap_horiz", "Portability",
                          "move settings to another machine")
    with card:
        ui.label(
            "Export your project paths and options to a JSON file, then import "
            "it on another machine. Credentials are included only if you tick "
            "the box below."
        ).classes("text-body2").style("color:#455a64")
        inc_creds = ui.checkbox(
            "Include credentials — written in PLAINTEXT; keep the file private",
            value=False)

        async def _export() -> None:
            dest = await FilePicker(
                str(Path.home() / "herbarium_settings.json"), mode="save")
            if not dest:
                return
            payload: dict = {"settings": dict(app.storage.general)}
            if inc_creds.value:
                payload["credentials"] = _collect_creds()
            try:
                Path(dest).write_text(json.dumps(payload, indent=2))
            except OSError as e:
                ui.notify(f"Export failed: {e}", type="negative")
                return
            extra = " (with credentials)" if inc_creds.value else ""
            ui.notify(f"Settings exported to {dest}{extra}", type="positive")

        async def _import() -> None:
            src = await FilePicker(str(Path.home()), mode="file")
            if not src:
                return
            try:
                data = json.loads(Path(src).read_text())
            except (OSError, json.JSONDecodeError) as e:
                ui.notify(f"Couldn't read {src}: {e}", type="negative")
                return
            settings = data.get("settings", {})
            if not isinstance(settings, dict):
                ui.notify("File has no 'settings' object — not a herbarium "
                          "export.", type="negative")
                return
            for k, v in settings.items():
                app.storage.general[k] = v
            n = _restore_creds(data.get("credentials") or {})
            msg = f"Imported {len(settings)} setting(s)"
            if n:
                msg += f" + {n} credential(s)"
            ui.notify(msg + ". Reload the page to refresh every field.",
                      type="positive")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Export settings…", icon="upload_file", on_click=_export)\
                .props("outlined dense color=primary")
            ui.button("Import settings…", icon="download", on_click=_import)\
                .props("outlined dense color=primary")

        ui.label(
            f"Config lives in: {CONFIG_PATH} · .nicegui/storage-general.json "
            "(app launch dir) · OS keyring (credentials)."
        ).classes("text-caption text-grey-6 mt-1").style("word-break:break-all")


def _build_setup() -> None:
    """Get Started landing + one-time setup: credentials and environment.

    All credentials persist in the OS keyring (RunPod / WandB / R2) and the
    SSH key path persists in app.storage.general. Once everything is green
    the user only needs to come back here to rotate credentials.
    """
    import platform as _platform
    gs = app.storage.general

    # Landing: orientation, project setup, status + progress. Built first so
    # it's the top of the tab; the credential cards follow beneath it.
    _build_get_started_landing(gs)

    _section("Credentials & environment")
    ui.label(
        "Set these once. RunPod, Hugging Face, and R2 are required for the full "
        "cloud workflow; WandB and GBIF are optional. "
        "Full step-by-step guide: cloud_setup.md."
    ).classes("text-body2").style("color:#455a64;max-width:820px")

    # ── Execution mode (advanced) ────────────────────────────────────────
    # The pipeline runs on a RunPod GPU pod by default. Local mode runs the
    # scripts as subprocesses on this machine and needs a ~20 GB CUDA GPU for
    # training — rarely used, so it's tucked away here rather than in the
    # header where it confused new users.
    with ui.expansion("Execution mode (advanced)", icon="tune")\
            .classes("w-full").style("max-width:820px"):
        ui.label(
            "Cloud (default) orchestrates a RunPod GPU pod from this UI — "
            "no local GPU needed. Local runs each step as a subprocess on "
            "this machine and needs an NVIDIA GPU with ~20 GB VRAM to train."
        ).classes("text-caption text-grey-7 mb-1")
        (ui.toggle({"cloud": "☁ Cloud", "local": "💻 Local"})
            .props("dense color=teal toggle-color=teal-9")
            .bind_value(app.storage.general, "main_mode"))

    # ── Local environment ───────────────────────────────────────────────
    env_card, env_pill = _setup_card("computer", "Local environment")
    with env_card:
        env_html = ui.html("").classes("w-full mt-1").style("font-size:14px")

        def _refresh_env() -> None:
            rows: list[tuple[str, str, bool]] = []
            rows.append(("Python", _platform.python_version(), True))
            try:
                import nicegui as _ng
                ng_ver = getattr(_ng, "__version__", "?")
                rows.append(("NiceGUI", ng_ver, ng_ver != "?"))
            except Exception:
                rows.append(("NiceGUI", "not importable", False))

            cuda_ok = False
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    n = _torch.cuda.device_count()
                    name = _torch.cuda.get_device_name(0) if n else "?"
                    rows.append(("PyTorch GPU", f"{n}× {name}", True))
                    cuda_ok = True
                else:
                    rows.append(("PyTorch", "CPU (fine for local Quick ID; "
                                 "train on the ☁ Cloud pod)", True))
            except ImportError:
                rows.append(("Local AI", "not installed — slim mode "
                             "(click \"Enable offline AI features\" to add it)",
                             False))

            base = (gs.get("main_base_dir") or str(Path.home())).strip()
            import shutil as _shutil
            try:
                free = _shutil.disk_usage(base).free // (1 << 30)
                rows.append(("Projects root",
                             f"{base} ({free} GB free)", free > 5))
            except Exception:
                rows.append(("Projects root", base, False))

            # External tools preflight — so a missing binary is flagged here
            # instead of failing mid-action.
            for tool, why in [
                ("uv",     "runs the app + installs offline AI features"),
                ("rclone", "R2 archive restore / Review 'From R2 archive'"),
            ]:
                path = _shutil.which(tool)
                rows.append((tool, path or f"not found — needed for {why}",
                             bool(path)))

            html_rows = []
            for label, value, ok in rows:
                tick = "✓" if ok else "—"
                color = "#2e7d32" if ok else "#9e9e9e"
                html_rows.append(
                    f"<tr>"
                    f"<td style='padding:4px 14px 4px 0;color:#666'>{label}</td>"
                    f"<td style='padding:4px 8px 4px 0'>"
                    f"<code style='background:#f5f5f5;padding:2px 8px;"
                    f"border-radius:3px;color:#263238'>{value}</code></td>"
                    f"<td style='color:{color};font-weight:700;font-size:15px'>{tick}</td>"
                    f"</tr>"
                )
            env_html.set_content(
                "<table style='border-collapse:collapse'>" + "".join(html_rows) + "</table>"
            )
            _set_pill(env_pill, "ready" if cuda_ok else "CPU only", "ok" if cuda_ok else "warn")

        _refresh_env()

        async def _enable_local_ml() -> None:
            """Install the optional ML stack so Quick ID / local Identify run
            on this machine. Streams `uv sync --extra local-ml` to the log."""
            async def _done(rc: int) -> None:
                _refresh_env()
                if rc == 0:
                    ui.notify("Offline AI features enabled — Quick ID and local "
                              "Identify are ready.", type="positive")
                else:
                    ui.notify("Install failed — see the Output panel.",
                              type="negative")
            ui.notify("Installing the local AI stack (torch, timm, transformers"
                      "…). This can take a few minutes — watch the Output panel.",
                      type="info")
            try:
                await _launch(["uv", "sync", "--extra", "local-ml"], on_done=_done)
            except FileNotFoundError:
                ui.notify("`uv` isn't on PATH. Install uv "
                          "(https://docs.astral.sh/uv/), or launch the app via "
                          "start.bat / start.sh.", type="negative")

        with ui.row().classes("items-center gap-2 mt-2"):
            ui.button("Re-check", icon="refresh", on_click=_refresh_env
                      ).props("flat dense")
            ui.button("Enable offline AI features", icon="download_for_offline",
                      on_click=_enable_local_ml)\
                .props("outlined dense color=primary")\
                .tooltip("Install torch + the ML stack so Quick ID and local "
                         "Identify run on this machine (CPU works, like the HF "
                         "Space). Adds a few hundred MB. Not needed for cloud "
                         "training — that runs on the pod.")
        ui.label(
            "The base install is slim (no torch). Click \"Enable offline AI "
            "features\" to add the ML stack when you want to identify specimens "
            "locally without a pod. One-time; runs uv sync --extra local-ml."
        ).classes("text-caption text-grey-6").style("max-width:820px")

    # ── RunPod ──────────────────────────────────────────────────────────
    rp_card, rp_pill = _setup_card("cloud", "RunPod (the GPU host)")
    with rp_card:
        ui.label(
            "Sign up at runpod.io and add billing. Create an API key at "
            "Settings → API Keys (starts with rpa_…). Also generate an SSH "
            "key-pair on this machine and register the public key at "
            "Settings → SSH Public Keys — RunPod auto-injects it into every "
            "pod, which is how the pipeline runs commands without prompts."
        ).classes("text-body2").style("color:#455a64")
        ui.label(
            "The field below must point at the PRIVATE half of that pair "
            "(no .pub) — e.g. ~/.ssh/id_ed25519, not id_ed25519.pub. RunPod's "
            "SSH Public Keys page is additive: adding a key from a new "
            "machine doesn't remove any machine's key already registered "
            "there, so multiple computers can each keep their own key and "
            "all stay able to connect. If you use this pipeline from more "
            "than one machine, each needs its own key registered."
        ).classes("text-caption text-grey-7 mt-1").style("max-width:820px")

        api_inp = (ui.input(label="API key",
                            placeholder="rpa_… (paste once, saved to OS keyring)")
                   .classes("w-full mt-3").props("dense outlined type=password"))

        default_key = str(Path.home() / ".ssh" / "id_ed25519_herbarium")
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            ssh_inp = (ui.input(label="SSH private key",
                                value=gs.get("cloud_ssh_key") or default_key,
                                placeholder=default_key)
                       .classes("flex-1").props("dense outlined clearable")
                       .bind_value(gs, "cloud_ssh_key"))

            async def _browse_ssh() -> None:
                cur = (ssh_inp.value or default_key)
                result = await FilePicker(cur, mode="file")
                if result:
                    ssh_inp.value = result
                    _refresh_rp_pill()

            ui.button(icon="folder_open", on_click=_browse_ssh
                      ).props("flat dense round").tooltip("Browse")

        ui.label("Use a passwordless automation key — pipeline steps shouldn't "
                 "stop for a passphrase prompt. Changing this path takes effect "
                 "immediately for the next pod action, no restart needed."
                 ).classes("text-caption text-grey-7 mt-1")

        def _refresh_rp_pill() -> None:
            has_key = bool(cloud_secrets.get_runpod_api_key())
            ssh_path = (gs.get("cloud_ssh_key") or "").strip()
            has_ssh = bool(ssh_path) and Path(ssh_path).expanduser().is_file()
            if has_key and has_ssh:
                _set_pill(rp_pill, "✓ ready", "ok")
            elif has_key:
                _set_pill(rp_pill, "API key saved · SSH key path missing", "warn")
            else:
                _set_pill(rp_pill, "not configured", "err")
        _refresh_rp_pill()
        ssh_inp.on("blur", lambda: _refresh_rp_pill())

        def _save_rp() -> None:
            v = (api_inp.value or "").strip()
            if not v:
                ui.notify("Paste your RunPod API key first.", type="warning"); return
            try:
                cloud_secrets.set_runpod_api_key(v)
            except Exception as e:
                ui.notify(f"Keyring save failed: {e}", type="negative"); return
            api_inp.value = ""
            _refresh_rp_pill()
            _cloud["orch"] = None
            ui.notify("RunPod API key saved to OS keyring.", type="positive")

        def _forget_rp() -> None:
            cloud_secrets.delete_runpod_api_key()
            _refresh_rp_pill()
            _cloud["orch"] = None
            ui.notify("RunPod API key removed.", type="info")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Save", on_click=_save_rp).props("unelevated dense color=primary")
            ui.button("Forget", on_click=_forget_rp).props("flat dense")

    # ── WandB ───────────────────────────────────────────────────────────
    wb_card, wb_pill = _setup_card("insights", "WandB", "optional · live training graphs")
    with wb_card:
        ui.label(
            "Free for academic use. Adds live loss/accuracy curves in your "
            "browser during training; without it, training falls back to CSV logs. "
            "Find your key at wandb.ai/authorize."
        ).classes("text-body2").style("color:#455a64")

        wb_inp = (ui.input(label="API key", placeholder="from wandb.ai/authorize")
                  .classes("w-full mt-3").props("dense outlined type=password"))

        def _refresh_wb_pill() -> None:
            if cloud_secrets.get_wandb_api_key():
                _set_pill(wb_pill, "✓ saved", "ok")
            else:
                _set_pill(wb_pill, "not set", "warn")
        _refresh_wb_pill()

        def _save_wb() -> None:
            v = (wb_inp.value or "").strip()
            if not v:
                ui.notify("Paste a key first.", type="warning"); return
            try:
                cloud_secrets.set_wandb_api_key(v)
            except Exception as e:
                ui.notify(f"Keyring save failed: {e}", type="negative"); return
            wb_inp.value = ""
            _refresh_wb_pill()
            ui.notify("WandB key saved (pushed to pod on next provision).", type="positive")

        def _forget_wb() -> None:
            cloud_secrets.delete_wandb_api_key()
            _refresh_wb_pill()
            ui.notify("WandB key removed.", type="info")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Save", on_click=_save_wb).props("unelevated dense color=primary")
            ui.button("Forget", on_click=_forget_wb).props("flat dense")

    # ── Hugging Face ────────────────────────────────────────────────────
    hf_card, hf_pill = _setup_card("smart_toy", "Hugging Face",
                                   "optional · publish trained models to the Hub")
    with hf_card:
        ui.label(
            "A write token lets the pod publish a trained family model to the "
            "Hugging Face Hub (⑦ Publish tab), where the "
            "herbarium-id Space picks it up automatically. Create one at "
            "huggingface.co/settings/tokens with the 'Write' role."
        ).classes("text-body2").style("color:#455a64")

        hf_inp = (ui.input(label="Write token", placeholder="hf_…")
                  .classes("w-full mt-3").props("dense outlined type=password"))

        def _refresh_hf_pill() -> None:
            if cloud_secrets.get_hf_token():
                _set_pill(hf_pill, "✓ saved", "ok")
            else:
                _set_pill(hf_pill, "not set", "warn")
        _refresh_hf_pill()

        def _save_hf() -> None:
            v = (hf_inp.value or "").strip()
            if not v:
                ui.notify("Paste a token first.", type="warning"); return
            try:
                cloud_secrets.set_hf_token(v)
            except Exception as e:
                ui.notify(f"Keyring save failed: {e}", type="negative"); return
            hf_inp.value = ""
            _refresh_hf_pill()
            ui.notify("Hugging Face token saved (pushed to pod at publish time).",
                      type="positive")

        def _forget_hf() -> None:
            cloud_secrets.delete_hf_token()
            _refresh_hf_pill()
            ui.notify("Hugging Face token removed.", type="info")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Save", on_click=_save_hf).props("unelevated dense color=primary")
            ui.button("Forget", on_click=_forget_hf).props("flat dense")

    # ── Cloudflare R2 ───────────────────────────────────────────────────
    r2_card, r2_pill = _setup_card("cloud_done", "Cloudflare R2",
                                   "optional · 50× faster pod setup + project archives")
    with r2_card:
        ui.label(
            "10 GB free tier. Two uses: (1) per-project Archive/Restore so you "
            "can delete RunPod volumes and pull projects back later; "
            "(2) a shared wheel + model-weight cache that makes a fresh pod "
            "~50× faster to set up. Create the API token at Cloudflare → R2 → "
            "Manage R2 API Tokens with permission Object Read & Write."
        ).classes("text-body2").style("color:#455a64")

        r2_acct = (ui.input(label="Account ID",
                            placeholder="32-char hex from your R2 dashboard URL")
                   .classes("w-full mt-3").props("dense outlined"))
        r2_akid = (ui.input(label="Access Key ID")
                   .classes("w-full").props("dense outlined type=password"))
        r2_sec  = (ui.input(label="Secret Access Key",
                            placeholder="shown once at token-creation time")
                   .classes("w-full").props("dense outlined type=password"))
        r2_buck = (ui.input(label="Default backup bucket", value="herbarium-backup")
                   .classes("w-full").props("dense outlined"))

        def _refresh_r2_pill() -> None:
            creds = cloud_secrets.get_r2_credentials()
            if creds:
                _set_pill(r2_pill, f"✓ {creds.bucket}", "ok")
            else:
                _set_pill(r2_pill, "not set", "warn")
        _refresh_r2_pill()

        def _save_r2() -> None:
            acct = (r2_acct.value or "").strip()
            ak   = (r2_akid.value or "").strip()
            sk   = (r2_sec.value or "").strip()
            bk   = (r2_buck.value or "").strip() or "herbarium-backup"
            if not (acct and ak and sk):
                ui.notify("Fill Account ID, Access Key, and Secret.", type="warning"); return
            try:
                cloud_secrets.set_r2_credentials(cloud_secrets.R2Credentials(
                    account_id=acct, access_key_id=ak,
                    secret_access_key=sk, bucket=bk,
                ))
            except Exception as e:
                ui.notify(f"Keyring save failed: {e}", type="negative"); return
            r2_akid.value = ""
            r2_sec.value = ""
            _refresh_r2_pill()
            ui.notify("R2 credentials saved (pushed to pod on next provision).",
                      type="positive")

        def _forget_r2() -> None:
            cloud_secrets.delete_r2_credentials()
            _refresh_r2_pill()
            ui.notify("R2 credentials removed.", type="info")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Save R2 creds", on_click=_save_r2
                      ).props("unelevated dense color=primary")
            ui.button("Forget", on_click=_forget_r2).props("flat dense")

    # ── GBIF ────────────────────────────────────────────────────────────────
    gbif_card, gbif_pill = _setup_card("grass", "GBIF",
                                       "optional · needed for multi-family bulk downloads")
    with gbif_card:
        ui.label(
            "Required when using --families (e.g. a split clade like old Olacaceae). "
            "The pipeline submits a single bulk download job to GBIF on the pod, "
            "which is far faster than paginating the search API for each family. "
            "Register at gbif.org — the same account you use on the website."
        ).classes("text-body2").style("color:#455a64")

        gbif_user_inp = (ui.input(label="GBIF username")
                         .classes("w-full mt-3").props("dense outlined"))
        gbif_pass_inp = (ui.input(label="GBIF password")
                         .classes("w-full").props("dense outlined type=password"))

        def _refresh_gbif_pill() -> None:
            creds = cloud_secrets.get_gbif_credentials()
            if creds:
                _set_pill(gbif_pill, f"✓ {creds.username}", "ok")
            else:
                _set_pill(gbif_pill, "not set", "warn")
        _refresh_gbif_pill()

        def _save_gbif() -> None:
            u = (gbif_user_inp.value or "").strip()
            p = (gbif_pass_inp.value or "").strip()
            if not (u and p):
                ui.notify("Enter both username and password.", type="warning"); return
            try:
                cloud_secrets.set_gbif_credentials(cloud_secrets.GBIFCredentials(
                    username=u, password=p,
                ))
            except Exception as e:
                ui.notify(f"Keyring save failed: {e}", type="negative"); return
            gbif_pass_inp.value = ""
            _refresh_gbif_pill()
            ui.notify("GBIF credentials saved (forwarded to pod during download step).",
                      type="positive")

        def _forget_gbif() -> None:
            cloud_secrets.delete_gbif_credentials()
            _refresh_gbif_pill()
            ui.notify("GBIF credentials removed.", type="info")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Save", on_click=_save_gbif).props("unelevated dense color=primary")
            ui.button("Forget", on_click=_forget_gbif).props("flat dense")

    # ── Portability: export / import settings ────────────────────────────
    _build_portability(gs)


# ---------------------------------------------------------------------------
# Cloud orchestration — module-scope state + helpers.
#
# Cloud is the default mode. The header carries a mode toggle (Local / Cloud)
# and, in Cloud mode, a pod status strip with provision / upload / download /
# terminate buttons. Step-tab Run buttons dispatch to either subprocess
# (Local) or orch.run_step (Cloud). The leftover advanced/destructive
# controls live in the ☁ Cloud Tools tab at the end of the tab list.
# ---------------------------------------------------------------------------

# Per-process cloud state. NiceGUI runs in a single asyncio loop so a plain
# dict is safe.
#   orch    — CloudOrchestrator instance for the current project
#   pod     — current PodHandle, or None
#   task    — single in-flight asyncio.Task (we serialise cloud work)
#   purpose — "light" | "train" — what the active pod was provisioned for.
#             Tracked here because PodHandle doesn't carry it; needed for
#             auto light→train upgrade on the Train tab.
#   setup_done_for_pod — pod_id of the pod we last successfully ran `setup`
#                        on. Cleared on terminate so a fresh pod re-runs
#                        setup before the next step (the venv lives on
#                        container disk and is gone after terminate).
_cloud: dict = {"orch": None, "pod": None, "task": None, "purpose": None,
                "setup_done_for_pod": None}

# Header pod-strip widgets. Populated by _build_pod_strip() during page
# render. Kept at module scope so helpers (refresh_status, progress
# callbacks) can update them from anywhere — including background tasks.
_cloud_widgets: dict = {
    "pod_lbl": None, "cost_lbl": None, "step_lbl": None,
    "progress_bar": None, "progress_lbl": None,
    "setup_warn": None,
}


def _is_cloud_mode() -> bool:
    return app.storage.general.get("main_mode", "cloud") == "cloud"


def _mode_only(elem, mode: str):
    """Show ``elem`` only when ``app.storage.general["main_mode"] == mode``."""
    elem.bind_visibility_from(app.storage.general, "main_mode",
                              lambda v: (v or "cloud") == mode)
    return elem


def _local_only(elem): return _mode_only(elem, "local")
def _cloud_only(elem): return _mode_only(elem, "cloud")


def _cloud_log(line: str) -> None:
    """Adapter: orchestrator emits plain strings, log widget wants newlines."""
    try:
        _log.push(line if line.endswith("\n") else line + "\n")
        _scan_wandb(line)
    except RuntimeError:
        pass  # client navigated away


def _cloud_running() -> bool:
    t = _cloud["task"]
    return t is not None and not t.done()


def _cloud_warn(msg: str) -> None:
    _cloud_log(f"⚠ {msg}")


def _cloud_err(msg: str) -> None:
    _cloud_log(f"✗ {msg}")


def _cloud_info(msg: str) -> None:
    _cloud_log(f"• {msg}")


def _ensure_orch() -> Optional[CloudOrchestrator]:
    """Return the active orchestrator, creating one if credentials allow.

    Rebuilds when the Project name or the SSH key path changed since the
    cached orchestrator was made. Without this, editing either field at the
    top of the page would silently keep the previous orchestrator — so
    provision() would attach to (and save state onto) the old project's
    volume/pod under the new name, or SSH would keep dialing out with a
    stale/missing key path (falling back to multi-key discovery, which can
    choke on an unrelated key in ~/.ssh). Both are fixed at construction, so
    a change to either must discard the cache.
    """
    gs = app.storage.general
    proj = (gs.get("main_proj") or "").strip()
    ssh_key = (gs.get("cloud_ssh_key") or "").strip() or None
    cached: Optional[CloudOrchestrator] = _cloud["orch"]
    if cached is not None and cached.project == proj and cached.key_filename == ssh_key:
        return cached
    api_key = cloud_secrets.get_runpod_api_key()
    if not api_key:
        _cloud_warn("Open the Get Started tab and save your RunPod API key first.")
        return None
    if not proj:
        _cloud_warn("Set the Project name at the top of the page first.")
        return None
    if cached is not None:
        # Switching projects: the cached pod handle belongs to the old
        # project. Drop it so steps don't run against it under the new name;
        # the old project's state file still tracks that pod for later reuse.
        _clear_active_pod()
        if cached.project != proj:
            _cloud_info(f"Project changed to {proj!r} — orchestrator reloaded.")
        else:
            _cloud_info("SSH key path changed — orchestrator reloaded.")
    _cloud["orch"] = CloudOrchestrator(api_key, proj, key_filename=ssh_key)
    return _cloud["orch"]


def _clear_active_pod() -> None:
    """Reset the per-pod state after a terminate. Called from manual
    Terminate, the auto light→train upgrade, and the Run All sequencer's
    pre-train upgrade — keeping all three in sync."""
    _cloud["pod"] = None
    _cloud["purpose"] = None
    _cloud["setup_done_for_pod"] = None


def _cloud_results_dir(orch: CloudOrchestrator) -> Path:
    """Local destination for downloaded cloud artefacts."""
    gs = app.storage.general
    base = gs.get("main_base_dir") or str(Path.home())
    proj = gs.get("main_proj") or orch.project
    return Path(base) / proj / "cloud_results"


def _set_text_if_changed(widget, value: str) -> None:
    """Skip the set_text (and the websocket frame it generates) when the
    label already shows ``value``. The status timer fires every 30 s; a
    long-idle pod would otherwise emit three pointless updates per tick."""
    if widget is not None and getattr(widget, "text", None) != value:
        widget.set_text(value)


def _refresh_cloud_status() -> None:
    pod_lbl = _cloud_widgets["pod_lbl"]
    if pod_lbl is None:
        return  # header not built yet
    cost_lbl = _cloud_widgets["cost_lbl"]
    step_lbl = _cloud_widgets["step_lbl"]
    orch: CloudOrchestrator | None = _cloud["orch"]
    pod: PodHandle | None = _cloud["pod"]
    if pod and orch:
        purpose = _cloud.get("purpose") or "?"
        _set_text_if_changed(pod_lbl,
            f"pod {pod.pod_id} [{purpose}] {pod.ssh_host}:{pod.ssh_port}  "
            f"${pod.cost_per_hr:.2f}/hr")
        _set_text_if_changed(cost_lbl, f"${orch.current_cost_usd():.4f}")
        _set_text_if_changed(step_lbl, f"step: {orch.state.current_step or '(idle)'}")
    else:
        _set_text_if_changed(pod_lbl,  "No active pod")
        _set_text_if_changed(cost_lbl, "$0.0000")
        _set_text_if_changed(step_lbl, "")


def _show_progress() -> None:
    bar = _cloud_widgets["progress_bar"]
    lbl = _cloud_widgets["progress_lbl"]
    if bar is None:
        return
    bar.visible = True
    lbl.visible = True


def _hide_progress() -> None:
    bar = _cloud_widgets["progress_bar"]
    lbl = _cloud_widgets["progress_lbl"]
    if bar is None:
        return
    bar.value = 0.0
    bar.visible = False
    lbl.set_text("")
    lbl.visible = False


def _make_progress_cb(prefix: str):
    """Thread-safe SFTP transfer progress callback. paramiko invokes from its
    IO thread; we marshal updates onto the asyncio loop and throttle to ~10 Hz.
    The bar only appears once bytes flow — short-circuited transfers stay hidden.
    """
    loop = asyncio.get_running_loop()
    last_t = [0.0]
    shown = [False]
    bar = _cloud_widgets["progress_bar"]
    lbl = _cloud_widgets["progress_lbl"]

    def cb(transferred: int, total: int) -> None:
        if bar is None:
            return
        if not shown[0]:
            shown[0] = True
            loop.call_soon_threadsafe(_show_progress)
        now = time.monotonic()
        if transferred < total and (now - last_t[0]) < 0.1:
            return
        last_t[0] = now
        mb_t = transferred / (1 << 20)
        text = (f"{prefix}: {mb_t:.1f} / {total / (1 << 20):.1f} MB"
                if total else f"{prefix}: {mb_t:.1f} MB")
        value = (transferred / total) if total > 0 else 0.0

        def _apply():
            bar.value = value
            lbl.set_text(text)
        loop.call_soon_threadsafe(_apply)
    return cb


def _wrap_cloud(coro_factory, *, force: bool = False):
    """Run an orchestrator coroutine as the single in-flight cloud task.
    ui.notify is unsafe inside a background task (no slot context); the body
    reports through _cloud_log instead.

    ``force=True`` displaces whatever is in the in-flight slot instead of
    refusing. Use it for deliberate "start over" actions (Provision, Attach):
    the common reason the slot is occupied is a *follower* task wedged on a
    dead pod's SSH channel — one that ``task.cancel()`` can't unstick because
    it's blocked in a background thread read. Refusing then traps the user
    ("a step is already running" when nothing is). Displacing is safe: pod-side
    steps run detached and survive; we only stop *watching* an old one, and the
    next Run re-attaches. The orphaned task errors out on its own dead socket.
    """
    if _cloud_running():
        if not force:
            ui.notify("A cloud step is already running.", type="warning")
            _cloud_log("⚠ A cloud step is already in flight — click 'Cancel step' to abort it, "
                       "then retry. (If the server was not restarted, a browser refresh leaves "
                       "the old task running in the background.)\n")
            return
        _cloud["task"].cancel()
        _cloud_log("• Displacing the previous in-flight task to start this one "
                   "(a step already on the pod keeps running and can be re-attached).\n")
    async def _run():
        try:
            await coro_factory()
        except asyncio.CancelledError:
            _cloud_log("[cancelled]")
            raise
        except Exception as e:
            _cloud_err(f"Cloud step failed: {e!r}")
        finally:
            _hide_progress()
            _refresh_cloud_status()
    _cloud["task"] = asyncio.create_task(_run())


def _wrap_cloud_aux(coro_factory):
    """Run an auxiliary cloud task (download / restore / status fetch) on a
    separate slot so it can run alongside a long-running step like identify
    or train. Each aux slot still serialises against itself — clicking
    "Pull images" twice while one is in flight gets the warning.

    Logs interleave with the main task; that's deliberate and tolerable
    because each operation's output is self-describing.
    """
    slot = "aux_task"
    t = _cloud.get(slot)
    if t is not None and not t.done():
        ui.notify("A transfer is already running — click Cancel step to abort "
                  "it first.", type="warning", timeout=6000)
        return
    async def _run():
        try:
            await coro_factory()
        except asyncio.CancelledError:
            _cloud_log("[aux cancelled]")
            raise
        except Exception as e:
            _cloud_err(f"Transfer failed: {e!r}")
        finally:
            _hide_progress()
    _cloud[slot] = asyncio.create_task(_run())


async def _kill_remote_step() -> None:
    """Stop the detached step on the pod, if one is running.

    Runs as a bare background task, so it must never raise and must never
    touch ui.notify (no slot context) — everything reports via _cloud_log.
    """
    try:
        orch = _ensure_orch()
        pod = _cloud.get("pod")
        if orch is None:
            _cloud_warn("No orchestrator — can't reach the pod to cancel.")
            return
        if pod is None:
            _cloud_warn("No active pod handle. Provision or Attach first, "
                        "then Cancel.")
            return
        step = await orch.running_step(pod, on_log=_cloud_log)
        if not step:
            _cloud_log("No step is running on the pod.")
            return
        await orch.cancel_step(pod, step, on_log=_cloud_log)
    except Exception as e:
        _cloud_warn(f"Could not stop the remote step: {e}")


def _cancel_cloud() -> None:
    """Cancel in-flight cloud work: the local tasks *and* the step on the pod.

    Steps run detached (setsid + nohup) so they survive a dropped connection.
    That also means cancelling the local asyncio task only stops us *watching*
    the step — it keeps running, and the next Run re-attaches to it. Kill it
    on the pod too, or Cancel is a lie.
    """
    cancelled = []
    for slot in ("task", "aux_task"):
        t = _cloud.get(slot)
        if t is not None and not t.done():
            t.cancel()
            cancelled.append(slot)

    # Fire-and-forget: the local task above may be the one holding the SSH
    # session, so this needs its own task rather than awaiting inline.
    _cloud["cancel_task"] = asyncio.create_task(_kill_remote_step())

    if cancelled:
        ui.notify(f"Cancelling ({', '.join(cancelled)}) and stopping the pod step…",
                  type="info")
    else:
        ui.notify("Stopping any step running on the pod…", type="info")


# ── per-step env builders (read from gs[...] — same keys the local tabs bind) ──

# Sentinel for "blank value" that covers None, "", whitespace, "0", "0.0".
# A few hyperparams (e.g. STAGE1_EPOCHS) might legitimately be "0" to mean
# "skip stage 1" — those are passed through; the bash side defaults are
# only triggered by genuinely-absent values.
_ZERO_LIKE = {"", "0", "0.0"}


def _env_from_gs(mapping: dict[str, object], *, drop_zero: bool = True) -> dict[str, str]:
    """Build a string-env dict from gs-backed values, dropping blanks.

    ``mapping`` keys are the env var names; values are either a single
    gs key (str), a tuple of fallback gs keys, or a literal string. With
    ``drop_zero=True`` "0"/"0.0" also count as blank — appropriate for
    fields like geo_weight where 0 means "off". With ``drop_zero=False``
    "0" is preserved (e.g. STAGE1_EPOCHS=0 means "skip stage 1").
    """
    gs = app.storage.general
    blanks = _ZERO_LIKE if drop_zero else {""}

    def _resolve(v: object) -> str:
        if isinstance(v, tuple):
            for key in v:
                got = gs.get(key)
                s = str(got).strip() if got is not None else ""
                if s and s not in blanks:
                    return s
            return ""
        if isinstance(v, str) and v.startswith("@"):  # literal escape: "@foo" → "foo"
            return v[1:]
        # bare gs key
        got = gs.get(v) if isinstance(v, str) else v
        return str(got).strip() if got is not None else ""

    out: dict[str, str] = {}
    for env_key, src in mapping.items():
        s = _resolve(src)
        if s and s not in blanks:
            out[env_key] = s
    return out


def _cloud_env_download() -> dict[str, str]:
    # RANK + TAXON drive a single-taxon download (family / genus / order) from
    # the rank radio + name box. TAXON_FAMILIES (the "Families (multi)" field)
    # takes precedence on the pod when set — matching the UI note that the
    # families list overrides the taxon name.
    return _env_from_gs({
        "RANK":              "dl_rank",
        "TAXON":             "dl_taxon",
        "TAXON_FAMILIES":    "dl_families",
        "CONTINENT":         "dl_continent",
        "EXCLUDE_COUNTRIES": "dl_exc",
        "COUNTRIES":         "dl_inc",
        "MAX_PER_SP":        ("cloud_max_per_sp",  "dl_max_per_sp"),
        "MAX_PER_GENUS":     ("cloud_max_per_ge",  "dl_max_per_ge"),
        "MAX_PER_FAMILY":    ("cloud_max_per_fa",  "dl_max_per_fa"),
        "LIMIT":             ("cloud_limit",       "dl_limit"),
        "IIIF":              ("cloud_iiif",        "dl_iiif"),
        "MAX_SIZE":          ("cloud_max_size",    "dl_max_size"),
        "FROM_SPECSIN":      "cloud_from_specsin",
        "SPECSIN_ONLY":      ("cloud_specsin_only", "dl_specsin_only"),
        "WORKERS":           ("cloud_workers",      "dl_workers"),
        "SKIP_FAILED":       ("cloud_skip_failed",  "dl_skip_failed"),
    })


def _cloud_env_prep() -> dict[str, str]:
    # FILTER_METHOD reads from fc_method — the same key the Filter & Crop
    # tab's Method dropdown binds to, so changing it there now actually
    # affects the cloud prep run. cloud_filter_method is kept for any old
    # state files that already have it set (fallback only).
    return _env_from_gs({
        "FILTER_METHOD": ("fc_method", "cloud_filter_method"),
        "NO_FILTER":     "cloud_no_filter",
        "NO_CROP":       "cloud_no_crop",
    })


def _cloud_env_train() -> dict[str, str]:
    # drop_zero=False so STAGE1_EPOCHS=0 (skip stage 1) etc. survive.
    env = _env_from_gs({
        "MODEL":               "tr_model",
        "IMAGE_SZ":             "tr_imgsz",
        "BATCH_SIZE":           "tr_batch",
        "ACCUM":                "tr_accum",
        "STAGE2_BATCH_SIZE":    "tr_s2_batch",
        "STAGE1_EPOCHS":        "tr_s1ep",
        "STAGE1_LR":            "tr_s1lr",
        "STAGE2_EPOCHS":        "tr_s2ep",
        "STAGE2_LR":            "tr_s2lr",
        "COOLDOWN_EPOCHS":      "tr_cd_ep",
        "COOLDOWN_LR":          "tr_cd_lr",
        "COOLDOWN_BATCH_SIZE":  "tr_cd_batch",
        "COOLDOWN_ACCUM":       "tr_cd_accum",
        "SPARSE_THRESHOLD":     "tr_sparse",
        "CLASS_WEIGHT_BETA":    "tr_cw_beta",
        "EARLY_STOP_PATIENCE":  "tr_es_patience",
        "NUM_GPUS":             "tr_gpus",
        "MAX_PER_CLASS":        "tr_max_per_sp",
        "LABEL_LEVEL":          "tr_label_level",
        "GEO_DIM":              "tr_geo_dim",
        "SPECIES_WEIGHT":       "tr_w_sp",
        "GENUS_WEIGHT":         "tr_w_ge",
        "FAMILY_WEIGHT":        "tr_w_fa",
        "WANDB_RUN_NAME":       "tr_wandb_name",
        "RESUME":               "tr_resume",
        "PREFETCH_QUEUE":       "tr_prefetch_queue",
    }, drop_zero=False)
    gs = app.storage.general
    if gs.get("tr_hier"):            env["HIERARCHICAL"]    = "1"
    if gs.get("tr_use_location"):    env["USE_LOCATION"]    = "1"
    if gs.get("tr_reset_optimizer"): env["RESET_OPTIMIZER"] = "1"
    if gs.get("tr_stage_images", True): env["STAGE_IMAGES"] = "1"
    if gs.get("tr_no_grad_ckpt"):    env["NO_GRAD_CKPT"]    = "1"
    return env


def _cloud_env_identify() -> dict[str, str]:
    return _env_from_gs({
        "MODEL":              ("id_model", "tr_model"),
        "IMAGE_SZ":            "id_imgsz",
        "BATCH_SIZE":          "id_batch",
        "THRESHOLD":           "id_thresh",
        "LOW_CONF_THRESHOLD":  "id_lowconf",
        "GEO_WEIGHT":          "id_geo_weight",
        "LOGIT_ADJUST":        "id_logit_adjust",
        "GEO_SIGMA":           "id_geo_sigma",
    })


# ── orchestrator action wrappers ──

async def _do_provision(purpose: str | None = None) -> None:
    """Provision (or reuse) a pod for the requested purpose, and sync code."""
    # First visible line, before any slow network call — otherwise a successful
    # Provision looks like a no-op until orch.provision() gets deep enough to log.
    _cloud_log("Provisioning a pod (reuse if one is live, else create — "
               "this can take 1–2 min)…")
    orch = _ensure_orch()
    if orch is None:
        return
    gs = app.storage.general
    pur = purpose or (gs.get("cloud_purpose") or "light")
    gpu = (gs.get("cloud_gpu_override") or "").strip() or None
    # A global GPU override (Cloud Tools) otherwise replaces the whole purpose
    # fallback list — including on a light→train upgrade, which would then
    # "upgrade" the pod straight back onto a light-tier card and defeat the
    # point. When provisioning a train pod, drop an override that's a light
    # GPU; a deliberate train-GPU override still stands.
    if gpu and pur == "train":
        from cloud.orchestrator import GPU_BY_PURPOSE as _GBP
        if gpu in _GBP.get("light", []):
            _cloud_warn(
                f"Ignoring GPU override '{gpu}' for the train pod — it's a "
                f"light-tier card, so honouring it would cancel the upgrade. "
                f"Using the train GPU list. Clear the override in the ☁ Cloud tab, "
                f"or set it to a train GPU to force a specific one.")
            gpu = None
    dc = (gs.get("cloud_datacenter") or "").strip() or DEFAULT_DATACENTER
    try:
        vol_gb = int(gs.get("cloud_volume_gb") or DEFAULT_VOLUME_GB)
    except ValueError:
        _cloud_warn("Volume size must be an integer.")
        return
    pod = await orch.provision(
        purpose=pur, gpu_type=gpu,
        data_center_id=dc, volume_gb=vol_gb,
        on_log=_cloud_log,
    )
    _cloud["pod"] = pod
    _cloud["purpose"] = pur
    await orch.sync_code(pod, on_log=_cloud_log)
    _refresh_cloud_status()
    # Provision only creates + preps the pod; it does NOT run setup/training.
    # Say so explicitly — otherwise the log ends silently after the last sync
    # line and reads like a hang.
    _cloud_log(
        f"✓ Pod ready ({pur}, ${pod.cost_per_hr:.2f}/hr). Provision only sets up "
        f"the pod — click Run Training, a pipeline step, or Run Full Pipeline "
        f"to continue.")


async def _do_attach() -> None:
    """Connect to an existing pod by its RunPod ID (from the console)."""
    orch = _ensure_orch()
    if orch is None:
        return
    gs = app.storage.general
    pod_id = (gs.get("cloud_attach_pod_id") or "").strip()
    if not pod_id:
        _cloud_warn("Enter a pod ID first.")
        return
    try:
        pod = await orch.attach(pod_id, on_log=_cloud_log)
    except Exception as e:
        _cloud_warn(f"Attach failed: {e}")
        return
    _cloud["pod"] = pod
    _cloud["purpose"] = "train"  # assume manually-created pod is train-capable

    # A step spawned before this UI restarted survives the disconnect (setsid
    # + nohup). Find it before touching anything: sync_code would SFTP over
    # pod_bootstrap.sh while a live bash is still reading it by file offset,
    # which can corrupt the running step.
    step = await orch.running_step(pod, on_log=_cloud_log)
    if step:
        _cloud_log(f"Skipping code sync — '{step}' is running and bash is "
                   f"still reading pod_bootstrap.sh.")
    else:
        await orch.sync_code(pod, on_log=_cloud_log)

    # Start the idle watchdog so the pod self-terminates after inactivity.
    # Surface failures — silent best-effort previously hid bugs (eg the
    # source-time dispatcher fault) and left attached pods billing forever.
    try:
        session = await orch._ensure_session(pod, on_log=_cloud_log)
        rc, out = await session.exec_capture(
            f"RUNPOD_POD_ID={pod_id} bash /workspace/Pipeline/pod_bootstrap.sh start_watchdog"
        )
        if rc != 0:
            _cloud_warn(
                "IDLE WATCHDOG IS NOT RUNNING — this pod will NOT stop itself and "
                "will bill until you terminate it by hand. "
                f"(rc={rc}) {out.strip() or '(no output)'}"
            )
        elif out.strip():
            _cloud_log(out.rstrip())
        # Reset the idle clock so the watchdog doesn't fire on history.
        await session.exec_capture("touch /workspace/.last_activity")
    except Exception as e:
        _cloud_warn(f"Watchdog start raised: {e}")
    _refresh_cloud_status()

    # Re-attach to the live step's log. Blocks until it finishes, exactly as
    # if the user had pressed its Run button — which is what they want after
    # reconnecting to a multi-hour download.
    if step:
        _cloud_log(f"↻ Re-attaching to '{step}' — streaming its log from the start.")
        rc = await orch.follow_running_step(pod, step, on_log=_cloud_log)
        if rc == 0:
            ui.notify(f"'{step}' finished", type="positive")
        else:
            _cloud_warn(f"'{step}' exited with code {rc}")
        _refresh_cloud_status()


async def _copy_pod_image() -> None:
    """Copy the prebaked image string for pasting into the RunPod console."""
    await ui.clipboard.write(DEFAULT_IMAGE)
    ui.notify("Pod image copied to clipboard", type="positive")


async def _do_save_template() -> None:
    """Create (or reuse) a RunPod template pinned to the prebaked image, so a
    hand-allocated pod can select it from the console's template dropdown."""
    orch = _ensure_orch()
    if orch is None:
        return
    try:
        t = await orch.ensure_pod_template(on_log=_cloud_log)
    except Exception as e:
        _cloud_warn(f"Save template failed: {e}")
        return
    ui.notify(
        f"RunPod template '{t.get('name')}' ready (id {t.get('id')}). "
        f"Pick it when you deploy a pod by hand.",
        type="positive",
    )


async def _do_upload_dwca() -> None:
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        _cloud_warn("Provision a pod first.")
        return
    local = (app.storage.general.get("dl_dwca") or "").strip()
    if not local:
        _cloud_warn("No DwC-A path. Set it in Tab 1 (Download).")
        return
    cb = _make_progress_cb(f"upload {Path(local).name}")
    await orch.upload_dwca(pod, local, on_log=_cloud_log, on_progress=cb)


async def _do_upload_specsin() -> None:
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        _cloud_warn("Provision or attach a pod first.")
        return
    gs = app.storage.general
    local = (gs.get("dl_specsin") or "").strip()
    if not local:
        _cloud_warn("No specsin path. Set it in Tab 1 (Download) specsin field.")
        return
    remote = (gs.get("cloud_from_specsin") or "").strip()
    if not remote:
        remote = "/workspace/data/specsin.csv"
    cb = _make_progress_cb(f"upload {Path(local).name}")
    await orch.upload_file(pod, local, remote, on_log=_cloud_log, on_progress=cb)


async def _do_step(step: str, *, env: dict[str, str] | None = None) -> int:
    if step == "train":
        _reset_wandb()   # clear any run link from a previous training run
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        _cloud_warn("Provision a pod first.")
        return 1
    # Auto-run setup once per pod. The venv lives on the network volume
    # (/workspace/venv) so a same-volume reprovision often makes setup a
    # no-op — but a brand-new pod still needs the auto-run because in-memory
    # `setup_done_for_pod` resets on every fresh provision.
    if step != "setup" and _cloud.get("setup_done_for_pod") != pod.pod_id:
        _cloud_log(f"One-time setup for pod {pod.pod_id} (env install)…")
        rc_setup = await orch.run_step(pod, "setup", env={}, on_log=_cloud_log)
        if rc_setup != 0:
            _cloud_err(f"Setup failed (rc={rc_setup}); aborting {step}.")
            return rc_setup
        _cloud["setup_done_for_pod"] = pod.pod_id
    if env:
        _cloud_log(f"{step} env: " + " ".join(f"{k}={v}" for k, v in env.items()))
    rc = await orch.run_step(pod, step, env=env or {}, on_log=_cloud_log)
    if rc != 0:
        _cloud_err(f"Step {step} exited {rc}")
    elif step == "setup":
        _cloud["setup_done_for_pod"] = pod.pod_id
    return rc


async def _do_download_results() -> None:
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        _cloud_warn("Provision a pod first.")
        return
    gs = app.storage.general
    local_dir = _cloud_results_dir(orch)
    cb = _make_progress_cb("download")
    ckpt_filter = (gs.get("cloud_ckpt_filter") or "latest").strip() or "latest"
    written = await orch.download_results(
        pod, local_dir, on_log=_cloud_log, on_progress=cb,
        ckpt_filter=ckpt_filter,
    )
    names = {p.name: str(p) for p in written}

    # Pick the lowest-valid_loss checkpoint over last.ckpt for inference.
    import re
    loss_re = re.compile(r"-(\d+\.\d+)\.ckpt$")
    scored: list[tuple[float, str]] = []
    for name, path in names.items():
        m = loss_re.search(name)
        if m:
            try:
                scored.append((float(m.group(1)), path))
            except ValueError:
                pass
    if scored:
        scored.sort(key=lambda x: x[0])
        best_loss, best_path = scored[0]
        gs["id_ckpt"] = best_path
        gs["active_ckpt"] = best_path
        _cloud_info(f"Identify ckpt → {Path(best_path).name} (best valid_loss={best_loss:.4f})")
    elif "last.ckpt" in names:
        gs["id_ckpt"] = names["last.ckpt"]
        gs["active_ckpt"] = names["last.ckpt"]
        _cloud_info("Identify ckpt → last.ckpt (no per-stage best files were pulled)")

    if "nameslist.json" in names:   gs["id_nl"]      = names["nameslist.json"]
    if "predictions.csv" in names:
        gs["review_csv"] = names["predictions.csv"]
    _cloud_info(f"Downloaded {len(written)} file(s) to {local_dir}")


async def _do_download_images() -> None:
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        _cloud_warn("Provision a pod first.")
        return
    gs = app.storage.general
    local_dir = _cloud_results_dir(orch)
    cb = _make_progress_cb("images.tar")
    out = await orch.download_images(pod, local_dir, on_log=_cloud_log, on_progress=cb)
    gs["review_imgs"] = str(out)
    _cloud_info(f"Pulled images → {out}")


async def _do_publish() -> None:
    """Publish the pod's best checkpoint to the Hugging Face Hub via the
    `publish` step. push_model.py auto-picks the accuracy-best checkpoint and
    reads the embedded nameslist; we only supply the metadata."""
    gs = app.storage.general
    fam  = (gs.get("pub_family") or "").strip()
    user = (gs.get("pub_hfuser") or "").strip()
    repo = (gs.get("pub_repo") or "").strip()
    # Fall back to the project name only when it is itself a plain taxon.
    # Blindly falling back is how a project called "Angiosperm-families_Africa"
    # got published as ggosline/herbarium-africa-angiosperm-families_africa-
    # family — a duplicate of the real repo, listed twice in the Space. The
    # repo id derived from this is public, so guess nothing.
    if not fam:
        proj = (gs.get("main_proj") or "").strip()
        if re.fullmatch(r"[A-Za-z]+", proj):
            fam = proj
        elif not repo:
            _cloud_warn(
                f"Project name {proj!r} is not a taxon, so it can't be used to "
                f"name the Hub repo. Fill in the Family field (e.g. "
                f"'Rubiaceae') or set an explicit Repo below, then publish "
                f"again.")
            return
    if not repo and not (user and fam):
        _cloud_warn("Set a Hugging Face user + family, or an explicit repo, "
                    "before publishing.")
        return
    if not cloud_secrets.get_hf_token():
        _cloud_warn("Add a Hugging Face token first (Get Started → Hugging Face).")
        return
    env = {
        "FAMILY":    fam,
        "HF_USER":   user,
        "REGION":    (gs.get("pub_region") or "").strip(),
        "HF_REPO":   repo,
        "SELECT_BY": (gs.get("pub_select_by") or "").strip(),
    }
    await _do_step("publish", env=env)


async def _do_terminate(*, keep_volume: bool, status_label, close_btn) -> None:
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        status_label.text = "No active pod."
        close_btn.enable()
        return
    try:
        await orch.terminate(pod, keep_volume=keep_volume, on_log=_cloud_log)
        _clear_active_pod()
        status_label.text = "✓ Pod terminated. Safe to close."
    except Exception as e:
        status_label.text = f"✗ Terminate failed: {e!r}  — check RunPod console."
        _cloud_err(f"Terminate failed: {e!r}")
    finally:
        close_btn.enable()
        _refresh_cloud_status()


def _confirm_terminate(keep_volume: bool) -> None:
    label = "Terminate (keep volume)" if keep_volume else "Terminate + DELETE volume"
    with ui.dialog() as dlg, ui.card():
        ui.label(label + "?").classes("text-h6")
        if not keep_volume:
            ui.label("Volume deletion is irreversible — all images and "
                     "checkpoints not previously downloaded will be lost.")\
                .classes("text-caption text-red-9")
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            def _ok() -> None:
                dlg.close()
                with ui.dialog().props(
                    "persistent no-esc-dismiss no-backdrop-dismiss"
                ) as prog_dlg, ui.card():
                    ui.label("Terminating pod…").classes("text-h6")
                    ui.label("DO NOT close this window until termination is confirmed.")\
                        .classes("text-caption text-red-9")
                    ui.spinner(size="lg").classes("self-center my-2")
                    status = ui.label("Calling RunPod API…").classes("text-caption")
                    close_btn = ui.button("Close", on_click=prog_dlg.close).props("flat")
                    close_btn.disable()
                prog_dlg.open()
                _wrap_cloud(lambda: _do_terminate(
                    keep_volume=keep_volume,
                    status_label=status,
                    close_btn=close_btn,
                ))
            ui.button("Terminate", on_click=_ok).props("color=negative unelevated")
    dlg.open()


async def _do_wipe(target: str) -> None:
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if not (orch and pod):
        _cloud_warn("Provision a pod first.")
        return
    targets = {
        "images":          "/workspace/data/images",
        "images_raw":      "/workspace/data/images_raw",
        "images_filtered": "/workspace/data/images_filtered",
        "images_1024":     "/workspace/data/images_1024",
        "predictions":     "/workspace/data/predictions",
    }
    path = targets[target]
    session = await orch._ensure_session(pod, on_log=_cloud_log)  # type: ignore[attr-defined]
    _cloud_log(f"$ rm -rf {path} && mkdir -p {path}")
    rc, out = await session.exec_capture(
        f"rm -rf {path} && mkdir -p {path} && echo cleared"
    )
    if rc == 0:
        _cloud_info(f"Wiped {path}")
    else:
        _cloud_err(f"Wipe failed (rc={rc}): {out.strip()}")


def _confirm_wipe(target: str, label: str) -> None:
    with ui.dialog() as dlg, ui.card():
        ui.label(f"Delete {label} on the pod?").classes("text-h6")
        ui.label(f"Path: /workspace/data/{target}").classes("text-caption")
        ui.label("This is irreversible. Files not previously downloaded "
                 "to this machine will be lost.").classes("text-caption text-red-9")
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            def _ok() -> None:
                dlg.close()
                _wrap_cloud(lambda: _do_wipe(target))
            ui.button("Wipe", on_click=_ok).props("color=negative unelevated")
    dlg.open()


def _do_restore_local() -> None:
    """Run restore_local.py against the user's chosen target dir.

    Uses the same _launch machinery as the local pipeline tabs, so output
    streams to the existing log panel. Defaults the project to main_proj
    when the local-restore field is empty.
    """
    gs = app.storage.general
    project = (gs.get("rl_project") or gs.get("main_proj") or "").strip()
    remote  = (gs.get("rl_remote")  or "r2:herbarium-backup").strip()
    target  = (gs.get("rl_target")  or "").strip()
    if not project:
        ui.notify("Set a Project name (or fill 'main_proj' on the Project tab).",
                  type="warning")
        return
    if not target:
        ui.notify("Pick a Target directory first.", type="warning")
        return
    script = str(Path(__file__).with_name("restore_local.py"))
    cmd = [sys.executable, "-u", script,
           "--project", project, "--target", target, "--remote", remote]
    asyncio.create_task(_launch(cmd))


def _confirm_restore() -> None:
    with ui.dialog() as dlg, ui.card():
        ui.label("Restore project from R2?").classes("text-h6")
        ui.label("This pulls checkpoints, specsin, predictions, and the "
                 "tarred image set back onto the volume. Existing files "
                 "with the same names will be overwritten.")\
            .classes("text-caption")
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            def _ok() -> None:
                dlg.close()
                gs = app.storage.general
                proj = (gs.get("main_proj") or "").strip()
                env = {"PROJECT": proj} if proj else None
                _wrap_cloud(lambda: _do_step("restore", env=env))
            ui.button("Restore", on_click=_ok).props("color=primary unelevated")
    dlg.open()


# ── auto light→train upgrade for the Train tab ──

async def _do_upgrade_to_train_and_run() -> None:
    """Terminate the current light pod (keeping the volume), provision a
    train pod attached to the same volume, sync code, and run train.
    """
    orch = _cloud["orch"]; pod = _cloud["pod"]
    if orch is None:
        _cloud_warn("Provision a pod first.")
        return
    if pod is not None and (_cloud.get("purpose") or "") == "light":
        _cloud_log("Upgrading light → train: terminating light pod (volume kept)…")
        try:
            await orch.terminate(pod, keep_volume=True, on_log=_cloud_log)
        except Exception as e:
            _cloud_err(f"Terminate-before-upgrade failed: {e!r}")
            return
        _clear_active_pod()
        _refresh_cloud_status()
    if _cloud["pod"] is None:
        await _do_provision(purpose="train")
    await _do_step("train", env=_cloud_env_train())


def _confirm_train_upgrade(then_run: callable) -> None:
    """One-time confirm for the auto light→train upgrade. The 'don't ask
    again' choice persists in gs[cloud_auto_upgrade].
    """
    gs = app.storage.general
    if gs.get("cloud_auto_upgrade"):
        then_run()
        return
    with ui.dialog() as dlg, ui.card().classes("p-4"):
        ui.label("Switch to a train pod?").classes("text-h6")
        ui.label(
            "Training needs a beefier GPU. We'll terminate the current "
            "light pod (volume + downloaded images preserved) and provision a "
            "train pod — first free of RTX A6000 / 6000 Ada / L40S / A100 / "
            "H100 / 4090."
        ).classes("text-body2 mt-1").style("max-width:520px")
        dont_ask = ui.checkbox("Don't ask again — auto-upgrade for future trainings",
                               value=False)
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            def _ok() -> None:
                if dont_ask.value:
                    gs["cloud_auto_upgrade"] = True
                dlg.close()
                then_run()
            ui.button("Switch + Train", on_click=_ok)\
                .props("color=primary unelevated")
    dlg.open()


# ── mode-aware Run dispatcher ──

def _run_step_mode_aware(
    step: str,
    local_cmd_fn,
    *,
    cloud_env_fn=None,
    extra_env: dict | None = None,
) -> None:
    """Dispatch the step's Run button: subprocess in Local mode,
    orchestrator in Cloud mode.

    step             — cloud step name ("download" | "prep" | "train" | "identify")
    local_cmd_fn     — callable returning argv list for the local subprocess.
                       Pass None to disable local execution for steps that
                       have no local equivalent.
    cloud_env_fn     — optional callable returning env dict for the pod.
    extra_env        — local subprocess env additions (e.g. NCCL flags).
    """
    if step == "train":
        _reset_wandb()   # clear any run link from a previous training run
    if _is_cloud_mode():
        if _cloud_running():
            ui.notify("A cloud step is already running.", type="warning")
            return
        # Train tab: detect light-pod and prompt for auto-upgrade.
        if step == "train":
            pod = _cloud["pod"]
            purpose = _cloud.get("purpose") or ""
            if pod is not None and purpose == "light":
                _confirm_train_upgrade(
                    lambda: _wrap_cloud(_do_upgrade_to_train_and_run))
                return
            if pod is None:
                # Auto-provision a train pod when none is active.
                async def _provision_and_train() -> None:
                    await _do_provision(purpose="train")
                    await _do_step("train", env=_cloud_env_train())
                _wrap_cloud(_provision_and_train)
                return
        # Generic case.
        env = cloud_env_fn() if cloud_env_fn else {}
        _wrap_cloud(lambda: _do_step(step, env=env))
        return

    # Local mode.
    if local_cmd_fn is None:
        ui.notify("This step has no local-mode equivalent.", type="warning")
        return
    try:
        cmd = local_cmd_fn()
    except ValueError as exc:
        ui.notify(str(exc), type="negative")
        return
    asyncio.create_task(_launch(cmd, extra_env=extra_env))


# ---------------------------------------------------------------------------
# Header pod-strip — visible only in Cloud mode.
# ---------------------------------------------------------------------------

def _build_pod_strip() -> None:
    """Pod status strip rendered inside the page header in Cloud mode.
    Holds: pod info / cost / current step, purpose dropdown, and the
    Provision / Upload-DwC-A / Download-Results / Terminate actions.
    Visibility is bound on the parent row, not here.
    """
    gs = app.storage.general

    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
        # ── Live pod state (always visible) ──────────────────────────────
        ui.icon("cloud").style("color:#80cbc4;font-size:22px")
        pod_lbl = ui.label("No active pod").classes("text-body2 font-mono")\
            .style("color:#e0f2f1")
        cost_lbl = ui.label("$0.0000").classes("font-mono")\
            .style("color:#80cbc4;font-weight:600")
        step_lbl = ui.label("").classes("text-caption font-mono")\
            .style("color:#80cbc4")

        ui.space()

        # Effective-GPU readout: what Provision will actually request (purpose
        # default, or the ☁ Cloud override if one is set). Kept visible so the
        # value isn't a surprise at launch.
        from cloud.orchestrator import GPU_BY_PURPOSE as _GPU_BY_PURPOSE
        def _gpu_label_text() -> str:
            override = (gs.get("cloud_gpu_override") or "").strip()
            if override:
                return f"GPU: {override} (override)"
            purp = gs.get("cloud_purpose") or "light"
            gpus = _GPU_BY_PURPOSE.get(purp) or []
            if not gpus:
                return "GPU: ? (auto)"
            first = gpus[0].replace("NVIDIA ", "")
            extra = f" +{len(gpus) - 1} fallback" if len(gpus) > 1 else ""
            return f"GPU: {first}{extra} (auto)"
        gpu_lbl = ui.label(_gpu_label_text())\
            .classes("text-caption font-mono")\
            .style("color:#b2dfdb;border-left:2px solid #4db6ac;padding-left:6px")
        gpu_lbl.tooltip("Provision offers RunPod this GPU plus fallbacks in "
                        "order and places on the first free one. Change purpose "
                        "or clear the override under Pod options.")
        def _refresh_gpu_lbl() -> None:
            gpu_lbl.set_text(_gpu_label_text())
        ui.timer(1.0, _refresh_gpu_lbl)

        def _clear_gpu_override() -> None:
            gs["cloud_gpu_override"] = ""
            _refresh_gpu_lbl()
            ui.notify("GPU override cleared — purpose default will be used.",
                      type="info")

        # ── Primary action: Provision ────────────────────────────────────
        ui.button("Provision", icon="cloud_upload",
                  on_click=lambda: _wrap_cloud(_do_provision, force=True))\
            .props("dense color=teal-3 unelevated")\
            .tooltip("Auto-provision a pod: offers RunPod the GPU fallback list "
                     "(first free one wins), creates/reuses the volume, syncs "
                     "code, pushes wandb / R2 keys. No free card? Use an "
                     "existing pod under Pod options.")

        # ── Pod options — purpose, GPU override, manual attach, image ────
        # Everything pre-provision and rarely-touched lives in this menu so the
        # header shows only the live state and the primary actions.
        with ui.button("Pod options", icon="tune").props("dense flat color=teal-2"):
            with ui.menu():
                with ui.column().classes("p-3 gap-2").style("min-width:340px"):
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label("Purpose:").classes("text-caption text-grey-8 shrink-0")
                        ui.select({"light": "light (L4)", "train": "train (A6000+)"},
                                  value=gs.get("cloud_purpose") or "light")\
                            .props("dense outlined options-dense")\
                            .classes("flex-1").bind_value(gs, "cloud_purpose")
                    ui.button("Clear GPU override", icon="close",
                              on_click=_clear_gpu_override)\
                        .props("flat dense color=primary")\
                        .tooltip("Use the purpose default instead of the "
                                 "☁ Cloud GPU override.")
                    ui.separator()
                    ui.label("Use an existing pod")\
                        .classes("text-caption text-grey-8 font-medium")
                    with ui.row().classes("items-center gap-1 w-full"):
                        pod_id_inp = ui.input(value="", placeholder="paste Pod ID")\
                            .props("dense outlined").classes("flex-1")\
                            .bind_value(gs, "cloud_attach_pod_id")\
                            .tooltip("Made a pod yourself in the RunPod console? "
                                     "Paste its ID. Attach connects, syncs code "
                                     "and starts the idle watchdog — then every "
                                     "step works as with an auto-provisioned pod.")
                        ui.button("Attach", icon="link",
                                  on_click=lambda: _wrap_cloud(_do_attach, force=True))\
                            .props("dense color=teal unelevated")\
                            .tooltip("Connect to the pod whose ID is in the box.")
                    ui.separator()
                    ui.label("Prebaked pod image")\
                        .classes("text-caption text-grey-8 font-medium")
                    with ui.row().classes("items-center gap-1 w-full"):
                        ui.label(DEFAULT_IMAGE)\
                            .style("font-family:monospace;font-size:11px;"
                                   "overflow-wrap:anywhere")\
                            .classes("flex-1")\
                            .tooltip("Container image to select when you create a "
                                     "pod by hand — the prebaked env, so setup is "
                                     "near-instant. (Auto-provision uses it too.)")
                        ui.button(icon="content_copy", on_click=_copy_pod_image)\
                            .props("dense flat round size=sm")\
                            .tooltip("Copy the image string.")
                    ui.button("Save RunPod template", icon="bookmark_add",
                              on_click=lambda: _wrap_cloud_aux(_do_save_template))\
                        .props("flat dense color=primary")\
                        .tooltip("Create a RunPod template pinned to this image so "
                                 "manual allocation can pick it from the dropdown.")

        # Data movement is no longer in the header: uploads live in the
        # ① Download tab ("Send to pod"), and pulling results back lives in
        # the ⑤ Review tab ("Get results") — where each is actually used.

        # ── Run control (always visible) ─────────────────────────────────
        ui.button("Cancel step", icon="stop",
                  on_click=_cancel_cloud).props("dense flat color=warning")\
            .tooltip("Interrupt the running step; the pod stays alive.")
        ui.button("Terminate", icon="power_settings_new",
                  on_click=lambda: _confirm_terminate(keep_volume=True))\
            .props("dense flat color=red-3")\
            .tooltip("Stop billing. Network volume preserved.")

    # Setup-not-configured banner — pulses every 2s so saving creds in
    # the Setup tab clears it without a reload.
    setup_warn = ui.label("").classes("text-caption mt-1")
    setup_warn.visible = False

    progress_bar = ui.linear_progress(value=0.0, show_value=False).classes("w-full mt-1")
    progress_lbl = ui.label("").classes("text-caption font-mono")\
        .style("color:#b2dfdb")
    progress_bar.visible = False
    progress_lbl.visible = False

    # Register widgets so module-scope helpers can update them.
    _cloud_widgets["pod_lbl"]      = pod_lbl
    _cloud_widgets["cost_lbl"]     = cost_lbl
    _cloud_widgets["step_lbl"]     = step_lbl
    _cloud_widgets["progress_bar"] = progress_bar
    _cloud_widgets["progress_lbl"] = progress_lbl
    _cloud_widgets["setup_warn"]   = setup_warn

    def _check_setup() -> None:
        has_key = bool(cloud_secrets.get_runpod_api_key())
        ssh_path = (gs.get("cloud_ssh_key") or "").strip()
        has_ssh = bool(ssh_path) and Path(ssh_path).expanduser().is_file()
        if has_key and has_ssh:
            setup_warn.visible = False
            return
        missing = []
        if not has_key: missing.append("RunPod API key")
        if not has_ssh: missing.append("SSH private key")
        setup_warn.set_text(
            f"⚠ Missing: {', '.join(missing)} — open the Get Started tab. "
            "Cloud actions will fail until configured.")
        setup_warn.style(
            "background:#ffebee;border-left:3px solid #c62828;padding:6px 10px;"
            "border-radius:0 4px 4px 0;color:#c62828;max-width:1100px"
        )
        setup_warn.visible = True

    _check_setup()
    # 30s rather than 2s: setup state only changes when the user explicitly
    # saves creds in the Setup tab. The 2s poll was hitting the OS keyring
    # (libsecret/D-Bus on Linux, blocking the asyncio loop ~1–10 ms each)
    # forever per browser session for a value that rarely changes.
    ui.timer(30.0, _check_setup)
    ui.timer(30.0, _refresh_cloud_status)


# ---------------------------------------------------------------------------
# ☁ Cloud Tools tab — advanced / rare / destructive actions.
# Visible only in Cloud mode (positioned at end of tab list).
# ---------------------------------------------------------------------------

def _build_archive() -> None:
    """⑥ Archive — pull results back and archive the whole project to R2.

    Promoted out of Cloud Tools onto the workflow spine: this is the tail of
    the normal pipeline (download → … → identify → review → archive → publish).
    """
    gs = app.storage.general

    with ui.row().classes("w-full items-baseline gap-2 mb-1"):
        ui.icon("cloud_done").style("color:#00897b;font-size:24px")
        ui.label("Archive & results").classes("text-h6").style("color:#00695c")
    ui.label(
        "Pull the finished image set and predictions back to this machine, or "
        "archive the whole project to Cloudflare R2 so you can delete the pod's "
        "network volume and restore it later."
    ).classes("text-body2").style(
        "background:#f0f7f6;border-left:3px solid #00897b;padding:8px 12px;"
        "border-radius:0 4px 4px 0;color:#37474f;max-width:1100px;margin-bottom:8px")

    # ── Pull images / R2 archive ─────────────────────────────────────────
    with ui.card().classes("w-full mt-2").style("border-left:3px solid #00897b"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("download").style("color:#00897b;font-size:20px")
            ui.label("Results & Archive").classes("text-subtitle1 font-bold")\
                .style("color:#00695c")
        ui.label(
            "Pull the resized image set to enable Review locally, or archive "
            "the whole project to Cloudflare R2 so you can delete the pod's "
            "network volume and bring it back later."
        ).classes("text-body2 mt-1").style("color:#455a64;max-width:820px")

        with ui.row().classes("w-full gap-2 mt-2 flex-wrap"):
            ui.button("Pull images (tar)", icon="image",
                      on_click=lambda: _wrap_cloud_aux(_do_download_images))\
                .props("flat dense color=primary")\
                .tooltip("Tar the resized image set on the pod and pull it back "
                         "so the Review tab can show specimens.")
            ui.button("Archive project to R2", icon="cloud_done",
                      on_click=lambda: _wrap_cloud(
                          lambda: _do_step("backup",
                              env={"PROJECT": (gs.get("main_proj") or "").strip()})))\
                .props("outlined dense color=primary")\
                .tooltip("Push ckpt, specsin, DwC-A, predictions, and "
                         "the images tar to r2:<bucket>/<project>/.")
            ui.button("Restore project from R2", icon="cloud_download",
                      on_click=_confirm_restore)\
                .props("outlined dense color=primary")\
                .tooltip("Pull a previously archived project back onto a "
                         "fresh volume — skip download/prep entirely.")

        # ── Restore to LOCAL machine (no pod required) ────────────────────
        with ui.row().classes("w-full items-center gap-2 mt-4"):
            ui.icon("download_for_offline").style("color:#00897b;font-size:20px")
            ui.label("Restore to this machine").classes("text-subtitle1 font-bold")\
                .style("color:#00695c")
        ui.label(
            "Pull an archived project directly onto this computer for local "
            "review / Identify, no pod needed. Requires rclone on PATH "
            "(https://rclone.org/install/) configured with the same R2 remote "
            "(default name 'r2'). Cross-platform — works on Windows, macOS, Linux."
        ).classes("text-body2 mt-1").style("color:#455a64;max-width:820px")
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label("Project:").classes("w-24 text-right shrink-0")
            ui.input(value="").classes("w-48").props("dense outlined")\
              .bind_value(gs, "rl_project")\
              .tooltip("Project name as used at backup (PROJECT env var). "
                       "Defaults to the current 'main_proj' if blank.")
            ui.label("Remote:").classes("w-20 text-right shrink-0")
            ui.input(value="r2:herbarium-backup").classes("w-56")\
              .props("dense outlined").bind_value(gs, "rl_remote")\
              .tooltip("rclone remote + bucket, e.g. r2:herbarium-backup")
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            rl_target = _path_input("Target directory:", mode="dir")\
                .bind_value(gs, "rl_target")
        ui.button("Restore to local", icon="download_for_offline",
                  on_click=lambda: _do_restore_local())\
            .props("unelevated dense color=primary").classes("mt-2")


def _build_publish() -> None:
    """⑦ Publish — push the best checkpoint to the Hugging Face Hub."""
    gs = app.storage.general

    with ui.row().classes("w-full items-baseline gap-2 mb-1"):
        ui.icon("smart_toy").style("color:#ff8f00;font-size:24px")
        ui.label("Publish model").classes("text-h6").style("color:#e65100")
    ui.label(
        "Final step: push this project's trained model to Hugging Face so the "
        "herbarium-id Space can serve it. Needs a write token (Get Started → "
        "Hugging Face)."
    ).classes("text-body2").style(
        "background:#fff8e1;border-left:3px solid #ff8f00;padding:8px 12px;"
        "border-radius:0 4px 4px 0;color:#5d4037;max-width:1100px;margin-bottom:8px")

    # ── Publish to Hugging Face ──────────────────────────────────────────
    with ui.card().classes("w-full mt-2").style("border-left:3px solid #ff8f00"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("smart_toy").style("color:#ff8f00;font-size:20px")
            ui.label("Publish to Hugging Face").classes("text-subtitle1 font-bold")\
                .style("color:#e65100")
        ui.label(
            "Push this project's best checkpoint (highest validation accuracy) "
            "to the Hugging Face Hub. It's tagged so the herbarium-id Space "
            "discovers it automatically — no redeploy. Needs a write token "
            "(Get Started → Hugging Face)."
        ).classes("text-body2 mt-1").style("color:#455a64;max-width:820px")

        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label("HF user:").classes("w-24 text-right shrink-0")
            ui.input(value="", placeholder="e.g. ggosline").classes("w-48")\
              .props("dense outlined").bind_value(gs, "pub_hfuser")\
              .tooltip("Your Hugging Face username. Used to build the repo name "
                       "when no explicit repo is given.")
            ui.label("Family:").classes("w-20 text-right shrink-0")
            ui.input(value="", placeholder="defaults to project").classes("w-48")\
              .props("dense outlined").bind_value(gs, "pub_family")\
              .tooltip("Family this model covers. Defaults to the current "
                       "project name if left blank.")
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label("Region:").classes("w-24 text-right shrink-0")
            ui.input(value="", placeholder="optional, e.g. Africa").classes("w-48")\
              .props("dense outlined").bind_value(gs, "pub_region")\
              .tooltip("Optional geographic scope, folded into the repo name "
                       "and model card.")
            ui.label("Select:").classes("w-20 text-right shrink-0")
            ui.select(["", "accuracy", "loss"], value="")\
              .props("dense outlined").classes("w-32")\
              .bind_value(gs, "pub_select_by")\
              .tooltip("Which checkpoint to publish. Blank = accuracy-best "
                       "(default); 'loss' = lowest valid_loss.")
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label("Repo:").classes("w-24 text-right shrink-0")
            ui.input(value="", placeholder="optional override, user/name")\
              .classes("w-80").props("dense outlined").bind_value(gs, "pub_repo")\
              .tooltip("Explicit HF repo id (overrides the user/family/region "
                       "derivation), e.g. ggosline/herbarium-africa-ebenaceae-species.")
        ui.button("Publish model to HF", icon="cloud_upload",
                  on_click=lambda: _wrap_cloud(_do_publish))\
            .props("unelevated dense color=primary").classes("mt-2")\
            .tooltip("Runs the publish step on the pod: picks the best "
                     "checkpoint and uploads it to the Hub.")


def _build_cloud_tools() -> None:
    gs = app.storage.general

    with ui.row().classes("w-full items-baseline gap-2 mb-1"):
        ui.icon("settings_suggest").style("color:#00897b;font-size:24px")
        ui.label("Cloud Tools").classes("text-h6").style("color:#00695c")
    ui.label(
        "Advanced and rare-use cloud actions. Day-to-day Provision / Upload "
        "/ Download / Terminate live in the header pod strip."
    ).classes("text-body2").style(
        "background:#f0f7f6;border-left:3px solid #00897b;padding:8px 12px;"
        "border-radius:0 4px 4px 0;color:#37474f;max-width:1100px;margin-bottom:8px")

    # ── Pod overrides ─────────────────────────────────────────────────────
    with ui.card().classes("w-full").style("border-left:3px solid #00897b"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("tune").style("color:#00897b;font-size:20px")
            ui.label("Pod overrides").classes("text-subtitle1 font-bold")\
                .style("color:#00695c")

        with ui.row().classes("w-full items-center gap-4 flex-wrap pt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("GPU type:").classes("text-sm")
                _saved_gpu = (gs.get("cloud_gpu_override") or "").strip().strip("'\"")
                gs["cloud_gpu_override"] = _saved_gpu
                gpu_inp = (ui.select(options=[""] + ([_saved_gpu] if _saved_gpu else []),
                                      value=_saved_gpu, label="(blank = purpose default)",
                                      with_input=True, clearable=True)
                            .classes("w-72").props("dense outlined")
                            .bind_value(gs, "cloud_gpu_override"))

                async def _refresh_gpu_list() -> None:
                    from cloud.runpod_client import RunPodClient
                    api_key = (cloud_secrets.get_runpod_api_key()
                               or "openapi-spec-is-public")
                    try:
                        async with RunPodClient(api_key) as rp:
                            ids = sorted(await rp.list_gpu_types())
                    except Exception as e:
                        ui.notify(f"Couldn't fetch GPU list: {e}",
                                  type="negative"); return
                    gpu_inp.set_options([""] + ids,
                                        value=gpu_inp.value or "")
                    ui.notify(f"Loaded {len(ids)} GPU types.",
                              type="positive")

                ui.button(icon="refresh", on_click=_refresh_gpu_list
                          ).props("flat dense round")\
                          .tooltip("Reload list from RunPod's OpenAPI schema.")
            with ui.row().classes("items-center gap-1"):
                ui.label("Datacenter:").classes("text-sm")
                ui.input(value=DEFAULT_DATACENTER).classes("w-32")\
                  .props("dense outlined").bind_value(gs, "cloud_datacenter")
            with ui.row().classes("items-center gap-1"):
                ui.label("Volume size (GB):").classes("text-sm")
                ui.input(value=str(DEFAULT_VOLUME_GB)).classes("w-20")\
                  .props("dense outlined").bind_value(gs, "cloud_volume_gb")
        ui.label(
            "RunPod's web console shows marketing names ('A100 SXM 80GB'); "
            "the API only accepts model numbers ('NVIDIA A100-SXM4-80GB'). "
            "Use ↻ to load the current list."
        ).classes("text-caption mt-1").style("color:#546e7a;max-width:780px")

    # ── Download caps ─────────────────────────────────────────────────────
    with ui.card().classes("w-full mt-2").style("border-left:3px solid #00897b"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("speed").style("color:#00897b;font-size:20px")
            ui.label("Download caps").classes("text-subtitle1 font-bold")\
                .style("color:#00695c")
        with ui.row().classes("w-full items-center gap-4 flex-wrap pt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Max per species:").classes("text-sm")
                ui.input(value="").classes("w-20")\
                  .props("dense outlined placeholder=all")\
                  .bind_value(gs, "cloud_max_per_sp")
            with ui.row().classes("items-center gap-1"):
                ui.label("Max per genus:").classes("text-sm")
                ui.input(value="").classes("w-20")\
                  .props("dense outlined placeholder=all")\
                  .bind_value(gs, "cloud_max_per_ge")
            with ui.row().classes("items-center gap-1"):
                ui.label("Max per family:").classes("text-sm")
                ui.input(value="").classes("w-20")\
                  .props("dense outlined placeholder=all")\
                  .bind_value(gs, "cloud_max_per_fa")
            with ui.row().classes("items-center gap-1"):
                ui.label("Total limit:").classes("text-sm")
                ui.input(value="").classes("w-20")\
                  .props("dense outlined placeholder=all")\
                  .bind_value(gs, "cloud_limit")
            with ui.row().classes("items-center gap-1"):
                ui.label("IIIF size (px):").classes("text-sm")
                ui.input(value="1200").classes("w-24")\
                  .props("dense outlined").bind_value(gs, "cloud_iiif")
            with ui.row().classes("items-center gap-1"):
                ui.label("Resize after download (px):").classes("text-sm")
                ui.input(value="1200").classes("w-24")\
                  .props("dense outlined").bind_value(gs, "cloud_max_size")
            with ui.row().classes("items-center gap-1"):
                ui.label("Workers:").classes("text-sm")
                ui.input(value="16").classes("w-20")\
                  .props("dense outlined").bind_value(gs, "cloud_workers")
        with ui.row().classes("items-center gap-1 mt-1"):
            ui.label("From specsin (pod path):").classes("text-sm")
            ui.input(value="").classes("w-72")\
              .props("dense outlined placeholder='e.g. /workspace/data/my_specsin.csv'")\
              .bind_value(gs, "cloud_from_specsin")
        with ui.row().classes("items-center gap-3 mt-1"):
            ui.checkbox("Specsin only (no download)", value=False)\
              .bind_value(gs, "cloud_specsin_only")
            ui.checkbox("Skip previously-failed", value=False)\
              .bind_value(gs, "cloud_skip_failed")\
              .tooltip("On a re-run, drop rows whose hasfile column is False "
                       "(i.e. previously failed). Each failed URL otherwise "
                       "costs up to ~90s to re-attempt. Leave off to retry "
                       "every failed row (transient failures get a chance).")
        ui.label(
            "Many institutions ignore IIIF size and serve full scans; the "
            "Resize-after-download value shrinks each fetched image with PIL "
            "regardless. Max-per-family ensures ≥1 record per genus when the "
            "cap is high enough. Leave \"From specsin\" blank to use the DwC-A "
            "ZIP or GBIF API instead."
        ).classes("text-caption mt-1").style("color:#546e7a;max-width:780px")

    # ── Prep settings ────────────────────────────────────────────────────
    with ui.card().classes("w-full mt-2").style("border-left:3px solid #00897b"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("tune").style("color:#00897b;font-size:20px")
            ui.label("Prep (filter + crop + resize)").classes("text-subtitle1 font-bold")\
                .style("color:#00695c")
        with ui.row().classes("w-full items-center gap-4 flex-wrap pt-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Filter method:").classes("text-sm")
                ui.select(["clip", "hsv"], value="clip")\
                  .props("dense outlined").classes("w-24")\
                  .bind_value(gs, "fc_method")
            with ui.row().classes("items-center gap-1"):
                ui.label("No filter:").classes("text-sm")
                ui.select(["", "1"], value="")\
                  .props("dense outlined").classes("w-20")\
                  .bind_value(gs, "cloud_no_filter")
            with ui.row().classes("items-center gap-1"):
                ui.label("No crop:").classes("text-sm")
                ui.select(["", "1"], value="")\
                  .props("dense outlined").classes("w-20")\
                  .bind_value(gs, "cloud_no_crop")
        ui.label(
            "Filter method: clip = CLIP zero-shot (GPU, more accurate), "
            "hsv = HSV heuristic (CPU, faster). Set No filter / No crop to "
            "\"1\" to skip those phases. The resize step always runs."
        ).classes("text-caption mt-1").style("color:#546e7a;max-width:780px")

    # ── Maintenance ────────────────────────────────────────────────────────
    with ui.card().classes("w-full mt-2").style("border-left:3px solid #ffa726"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("build").style("color:#ef6c00;font-size:20px")
            ui.label("Maintenance").classes("text-subtitle1 font-bold")\
                .style("color:#e65100")
        ui.label(
            "One-off operations. Repair-cache forces a full re-download of "
            "every wheel and pushes them to R2 — use once if an earlier setup "
            "left R2 with metadata-only cache entries."
        ).classes("text-body2 mt-1").style("color:#5d4037;max-width:820px")
        with ui.row().classes("w-full gap-2 mt-2 flex-wrap"):
            ui.button("Repair R2 wheel cache", icon="build",
                      on_click=lambda: _wrap_cloud(
                          lambda: _do_step("repair_cache")))\
                .props("flat dense color=warning")
            ui.button("Run Setup step", icon="settings",
                      on_click=lambda: _wrap_cloud(
                          lambda: _do_step("setup")))\
                .props("flat dense color=primary")\
                .tooltip("One-time env install on the pod (uv sync + DALI). "
                         "The first run on a fresh pod auto-runs setup; this "
                         "button is for re-running after a manual rm of /workspace/venv.")

    # ── Danger zone ────────────────────────────────────────────────────────
    with ui.expansion("⚠ Danger zone — wipe pod-side data"
                       ).classes("w-full mt-2").props("dense"):
        ui.label(
            "Files deleted here are gone unless you previously downloaded or "
            "archived them. Useful when re-running a step from scratch."
        ).classes("text-body2 mt-1").style("color:#c62828;max-width:780px")
        with ui.row().classes("w-full gap-2 mt-2 flex-wrap"):
            ui.button("images",
                      on_click=lambda: _confirm_wipe(
                          "images", "all images (unified layout)"))\
                .props("flat dense color=negative")
            ui.button("images_raw",
                      on_click=lambda: _confirm_wipe(
                          "images_raw", "raw downloaded images (legacy layout)"))\
                .props("flat dense color=negative")
            ui.button("images_filtered",
                      on_click=lambda: _confirm_wipe(
                          "images_filtered", "filter+crop output (legacy layout)"))\
                .props("flat dense color=negative")
            ui.button("images_1024",
                      on_click=lambda: _confirm_wipe(
                          "images_1024", "resized training images (legacy layout)"))\
                .props("flat dense color=negative")
            ui.button("predictions",
                      on_click=lambda: _confirm_wipe(
                          "predictions", "identify output"))\
                .props("flat dense color=negative")



def _build_run_all(dl_cmd, fc_cmd, rs_cmd, tr_cmd, id_cmd) -> None:
    gs = app.storage.general

    # Mode-specific intro labels.
    with _local_only(ui.column().classes("w-full")):
        ui.label("Local mode: runs each selected step as a subprocess on this machine."
                 ).classes("text-body1 mt-2")
    with _cloud_only(ui.column().classes("w-full")):
        ui.label(
            "Cloud mode: provisions a light pod, uploads the DwC-A, runs "
            "Setup → Download → Prep, auto-upgrades to a train pod (volume "
            "preserved), runs Train → Identify, and pulls results back."
        ).classes("text-body1 mt-2")

    with ui.card().classes("w-full mt-4"):
        ui.label("Steps to run").classes("text-subtitle2 font-bold mb-2")
        run_dl = ui.checkbox("1  Download",      value=True)
        # Filter & Crop and Resize are independent in local mode but a single
        # "prep" step on the pod. Show them as separate checkboxes for local;
        # in cloud mode, either being ticked runs prep once.
        run_fc = ui.checkbox("2  Filter & Crop",  value=True)
        run_rs = ui.checkbox("3  Resize",         value=True)
        run_tr = ui.checkbox("4  Train",           value=True)
        run_id = ui.checkbox("5  Identify",        value=True)

    # ── local-mode runner ──
    async def _run_all_local():
        global _pipeline
        if _proc and _proc.returncode is None:
            ui.notify("A process is already running.", type="warning")
            return
        steps = []
        if run_dl.value: steps.append(("Download",       dl_cmd))
        if run_fc.value: steps.append(("Filter & Crop",  fc_cmd))
        if run_rs.value: steps.append(("Resize",         rs_cmd))
        if run_tr.value: steps.append(("Train",          tr_cmd))
        if run_id.value: steps.append(("Identify",       id_cmd))
        if not steps:
            ui.notify("No steps selected.", type="info")
            return
        _pipeline.clear()
        _pipeline.extend(steps)
        await _run_pipeline()

    # ── cloud-mode runner ──
    async def _run_all_cloud():
        """Sequence the full cloud pipeline. Per-step error handling so a
        failed earlier step doesn't tear down a freshly-provisioned train pod.
        """
        # 1) ensure a light pod and DwC-A.
        if _cloud["pod"] is None:
            await _do_provision(purpose="light")
        if _cloud["pod"] is None:
            return  # provision failed; _do_provision already logged.
        await _do_upload_dwca()
        rc = await _do_step("setup")
        if rc != 0:
            return

        # 2) light-pod work — download & prep.
        if run_dl.value:
            rc = await _do_step("download", env=_cloud_env_download())
            if rc != 0:
                _cloud_warn("Download failed — stopping. Inspect logs, fix, re-run.")
                return
        if run_fc.value or run_rs.value:
            rc = await _do_step("prep", env=_cloud_env_prep())
            if rc != 0:
                _cloud_warn("Prep failed — stopping.")
                return

        # 3) upgrade to train pod for the heavy lift.
        if run_tr.value:
            orch = _cloud["orch"]; pod = _cloud["pod"]
            if orch is not None and pod is not None and (_cloud.get("purpose") or "") == "light":
                _cloud_log("Run All: upgrading light → train (volume preserved)…")
                try:
                    await orch.terminate(pod, keep_volume=True, on_log=_cloud_log)
                except Exception as e:
                    _cloud_err(f"Pre-train terminate failed: {e!r}")
                    return
                _cloud["pod"] = None
                _cloud["purpose"] = None
                _refresh_cloud_status()
                await _do_provision(purpose="train")
                if _cloud["pod"] is None:
                    return
            rc = await _do_step("train", env=_cloud_env_train())
            if rc != 0:
                _cloud_warn("Train failed — stopping. The train pod is still "
                            "running so you can investigate; remember to terminate.")
                return

        # 4) identify (stays on whichever pod is current).
        if run_id.value:
            rc = await _do_step("identify", env=_cloud_env_identify())
            if rc != 0:
                _cloud_warn("Identify failed — stopping.")
                return

        # 5) pull artefacts back. Done last so prior failures don't leave
        #    half-baked downloads on the local disk.
        await _do_download_results()
        _cloud_info("Run All complete. Pod is still running — terminate from the header strip when done.")

    def _on_run_all_click() -> None:
        if _is_cloud_mode():
            _wrap_cloud(_run_all_cloud)
        else:
            asyncio.create_task(_run_all_local())

    ui.button("Run Full Pipeline", icon="play_circle",
              on_click=_on_run_all_click)\
        .props("color=positive unelevated size=lg").classes("mt-6")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

@app.on_startup
def _migrate_config() -> None:
    """One-time migration: copy old JSON config keys into app.storage.general."""
    gs = app.storage.general
    if "main_base_dir" in gs:
        return  # already migrated
    old = _load_config()
    if not old:
        return
    mapping = {
        "base_dir":    "main_base_dir",
        "review_csv":  "review_csv",
        "review_imgs": "review_imgs",
    }
    for old_key, new_key in mapping.items():
        if old_key in old and new_key not in gs:
            gs[new_key] = old[old_key]


# ---------------------------------------------------------------------------
# Full-screen review carousel  (opens in new tab from Review → Open Carousel)
# ---------------------------------------------------------------------------

@ui.page("/review-carousel")
def carousel_page():
    """Dedicated full-screen page for reviewing specimens quickly."""
    import pandas as _pd

    view = _review_shared.get("view")
    if view is None or len(view) == 0:
        ui.label("No review data loaded. Load a CSV on the main page first.") \
          .classes("text-h6 text-grey-6 q-pa-xl")
        return

    imgs_dir = _review_shared.get("imgs_dir", "")
    level    = _review_shared.get("level", "species")
    _idx     = [0]

    # ── Styles ───────────────────────────────────────────────────────────────
    ui.query("body").style(
        "font-family:'Roboto',sans-serif; font-weight:500; "
        "background:#1a1a2e; color:#eee; margin:0")
    ui.add_head_html(
        '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700'
        '&display=swap" rel="stylesheet">'
        "<style>"
        ".nicegui-content{padding:0!important}"
        "</style>")

    # ── Layout: image left, info right ───────────────────────────────────────
    with ui.row().style(
            "width:100vw; height:100vh; margin:0; gap:0; flex-wrap:nowrap"):

        # -- Image pane --
        with ui.column().classes("items-center justify-center") \
                .style("flex:2; background:#111; min-width:0;"
                       "height:100vh; position:relative; overflow:hidden"):
            img_el = (ui.image("")
                      .props("fit=contain")
                      .style("width:100%; height:calc(100vh - 36px)"))
            counter_lbl = ui.label("").style(
                "position:absolute; bottom:6px; left:50%; transform:translateX(-50%);"
                "color:#aaa; font-size:14px")

        # -- Info pane (right sidebar, scrollable) --
        with ui.column().classes("gap-4").style(
                "flex:1; min-width:340px; max-width:480px; height:100vh;"
                "background:#222; overflow-y:auto; padding:20px"):
            info_html  = ui.html("", sanitize=False).style("color:#ccc; font-size:16px")
            bars_html  = ui.html("", sanitize=False).style("width:100%")
            ui.separator().style("border-color:#444")
            det_sel    = (ui.select([], label="Determine as:")
                          .classes("w-full")
                          .props("outlined dark"))
            with ui.row().classes("gap-3 flex-wrap"):
                confirm_btn = (ui.button("Confirm", icon="check",
                                          on_click=lambda: _confirm())
                                .props("color=positive unelevated"))
                invalid_btn = (ui.button("Invalid", icon="close",
                                          on_click=lambda: _mark_invalid())
                                .props("color=negative unelevated"))
            action_lbl = ui.label("").style("color:#80cbc4; font-size:14px")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve(row) -> str:
        # Mirror Review tab: imgs_dir/fname wins when configured, since
        # cloud predictions have abs_path pointing at the pod (e.g.
        # /workspace/data/images_1024/...) which doesn't exist locally.
        fname = str(row.get("fname", ""))
        candidates: list[str] = []
        if fname and imgs_dir:
            candidates.append(str(Path(imgs_dir) / fname))
        for col in ("abs_path", "filename"):
            v = row.get(col, "")
            if v and v == v and str(v) not in ("", "nan"):
                candidates.append(str(v))
        for c in candidates:
            if Path(c).is_file():
                return c
        return candidates[0] if candidates else ""

    def _top5(row) -> list[tuple[str, float]]:
        # Honour the rank chosen in the Review tab. At genus, read the trained
        # genus head's own list (gtop*) rather than the species head's.
        prefix = "gtop" if level == "genus" else "top"
        if f"{prefix}1_name" in row.index:
            items = []
            for k in range(1, 6):
                n = row.get(f"{prefix}{k}_name", "")
                n = "" if (n != n or n is None) else str(n)
                if not n or n == "nan":
                    break
                items.append((n, float(row.get(f"{prefix}{k}_prob", 0) or 0)))
            if items:
                return items
        if level == "genus":
            return [(str(row.get("pred_genus", "")),
                     float(row.get("genus_conf", 0) or 0))]
        return [(str(row.get("pred_species", "")),
                 float(row.get("confidence", 0) or 0))]

    def _bars(items: list[tuple[str, float]]) -> str:
        if not items:
            return ""
        max_p = items[0][1] or 1.0
        accents = ["#26a69a", "#4db6ac", "#80cbc4", "#b2dfdb", "#e0f2f1"]
        html = "<div style='display:flex;flex-direction:column;gap:8px'>"
        for i, (name, prob) in enumerate(items[:5]):
            pct_raw = prob * 100
            pct_bar = (prob / max_p) * 100
            a = accents[min(i, 4)]
            fsz = "16px" if i == 0 else "15px"
            html += (
                f"<div style='border-left:4px solid {a};padding:6px 10px;"
                f"border-radius:4px;background:#2a2a3e'>"
                f"<div style='font-style:italic;color:#e0e0e0;font-size:{fsz}'>"
                f"<span style='color:{a};font-weight:700;margin-right:8px'>#{i+1}</span>"
                f"{name}</div>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-top:4px'>"
                f"<div style='flex:1;background:#333;border-radius:4px;height:10px'>"
                f"<div style='background:{a};width:{pct_bar:.1f}%;height:100%;"
                f"border-radius:4px'></div></div>"
                f"<span style='font-size:14px;color:{a};font-weight:600;"
                f"min-width:50px;text-align:right'>{pct_raw:.1f}%</span>"
                f"</div></div>"
            )
        html += "</div>"
        return html

    def _show(idx: int):
        idx = max(0, min(idx, len(view) - 1))
        _idx[0] = idx
        row = view.iloc[idx]

        path = _resolve(row)
        img_el.set_source(_review_img_url(path) if path else "")
        counter_lbl.set_text(f"{idx + 1} / {len(view)}")

        if level == "genus":
            conf = float(row.get("genus_conf", row.get("gtop1_prob", 0)) or 0)
            pred = str(row.get("pred_genus", row.get("gtop1_name", "")))
            true = str(row.get("true_genus", ""))
        else:
            conf = float(row.get("confidence", row.get("top1_prob", 0)) or 0)
            pred = str(row.get("pred_species", row.get("top1_name", "")))
            true = str(row.get("true_species", ""))
        fname = str(row.get("fname", row.get("filename", "")))
        match = ""
        if true and true != "nan":
            ok = true.strip() == pred.strip()
            match = (f" <span style='color:{'#66bb6a' if ok else '#ef5350'}'>"
                     f"{'✓' if ok else '✗'}</span>")

        aum_line = ""
        aum_raw = row.get("aum", None)
        try:
            if aum_raw is not None and float(aum_raw) == float(aum_raw):
                av = float(aum_raw)
                colour = "#ef5350" if av < 0 else ("#ffa726" if av < 2 else "#888")
                hint = " — possible mislabel" if av < 2 else ""
                aum_line = (f"<div style='font-size:16px'><b>AUM:</b> "
                            f"<span style='color:{colour}'>{av:+.2f}{hint}</span></div>")
        except (TypeError, ValueError):
            pass

        info_html.set_content(
            f"<div style='line-height:1.8'>"
            f"<div style='font-size:22px;font-weight:700;font-style:italic;"
            f"color:#e0f2f1;margin-bottom:4px'>{pred}</div>"
            f"<div style='font-size:16px'><b>Confidence:</b> {conf:.1%}{match}</div>"
            + (f"<div style='font-size:16px'><b>True:</b> <i>{true}</i></div>"
               if true and true != "nan" else "")
            + aum_line
            + f"<div style='color:#888;font-size:13px;margin-top:4px'>"
            f"{Path(fname).name}</div>"
            f"</div>"
        )
        bars_html.set_content(_bars(_top5(row)))

        names = [n for n, _ in _top5(row)]
        det_sel.set_options(names, value=names[0] if names else "")
        action_lbl.set_text("")

    def _go(delta: int):
        _show(_idx[0] + delta)

    def _confirm():
        names = [n for n, _ in _top5(view.iloc[_idx[0]])]
        chosen = det_sel.value or (names[0] if names else "")
        action_lbl.set_text(f"Confirmed → {chosen}")
        ui.notify(f"Determined: {chosen}", type="positive")
        _go(1)

    def _mark_invalid():
        fname = str(view.iloc[_idx[0]].get("fname", ""))
        action_lbl.set_text(f"Marked invalid: {Path(fname).name}")
        ui.notify("Marked invalid", type="warning")
        _go(1)

    # ── Keyboard navigation ─────────────────────────────────────────────────
    ui.keyboard(on_key=lambda e: (
        _go(1)  if e.key == "ArrowRight" and e.action.keydown else
        _go(-1) if e.key == "ArrowLeft"  and e.action.keydown else
        None
    ))

    _show(0)


@ui.page("/")
def main_page():
    global _log, _status, _stop_btn, _wandb_link

    ui.query("body").style("font-family:'Roboto',sans-serif; font-weight:500; background:#f0f2f4; color:#1a2027")
    ui.add_head_html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700'
        '&family=Roboto+Mono&display=swap" rel="stylesheet">'
        "<style>"
        "body,input,.q-field__native,.q-field__input{font-weight:500!important}"
        ".q-tab-panel{padding:10px 14px!important}"
        ".q-tab__label{font-weight:600!important}"
        ".q-card>.q-card__section{padding:10px 14px!important}"
        ".q-separator{margin:2px 0!important}"
        ".q-tooltip{max-width:320px}"
        "</style>"
    )

    # ---- Header ----
    with ui.header().classes("bg-teal-700 text-white px-6 py-2 q-pa-sm"):
        with ui.column().classes("w-full gap-1"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Herbarium Classification Pipeline").classes("text-h6 font-bold")
                with ui.row().classes("items-center gap-4"):
                    # Cloud is the one mode a new user sees. The Local/Cloud
                    # switch now lives in ⚙ Setup → Advanced (local training is
                    # rarely used); default cloud on first launch.
                    if "main_mode" not in app.storage.general:
                        app.storage.general["main_mode"] = "cloud"
                    # A small badge shows Local mode when it's active, so a
                    # power user who flipped it isn't left guessing.
                    (ui.label("💻 Local mode")
                        .classes("text-caption")
                        .style("color:#b2dfdb;border:1px solid #4db6ac;"
                               "border-radius:4px;padding:1px 6px")
                        .bind_visibility_from(
                            app.storage.general, "main_mode",
                            lambda v: (v or "cloud") == "local"))
                    _status = ui.label("Ready").classes("text-body2")
                    # Persistent "busy" chip — visible whenever a local
                    # subprocess or cloud task is in flight, so a rejected
                    # click ("already running") is self-explanatory and the
                    # user knows to Cancel/Stop first. Driven by a 1s timer.
                    busy_chip = (ui.label("")
                                 .classes("text-caption")
                                 .style("background:#fff3e0;color:#e65100;"
                                        "border-radius:10px;padding:2px 10px;"
                                        "font-weight:600;white-space:nowrap"))
                    busy_chip.set_visibility(False)
                    _stop_btn = (ui.button("Stop", icon="stop", on_click=_stop_process)
                                 .props("flat color=white")
                                 .classes("text-white"))
                    _stop_btn.disable()
                    (ui.button("Quit", icon="power_settings_new", on_click=_quit)
                     .props("flat color=white")
                     .classes("text-white"))
                    log_vis_btn = (ui.button(icon="terminal",
                                             on_click=lambda: _toggle_log_panel())
                                   .props("flat color=white")
                                   .tooltip("Hide / show output panel"))

            # Pod-status strip — visible only in Cloud mode.
            pod_strip_row = ui.column().classes("w-full")
            _cloud_only(pod_strip_row)
            with pod_strip_row:
                _build_pod_strip()

    # ---- Main split layout: left = config+tabs, right = log ----
    with ui.row().classes("w-full gap-0 items-stretch").style(
            "height:calc(100vh - 64px); overflow:hidden"):

        # ---- Left panel: project config + tabs (scrollable) ----
        with ui.scroll_area().style("flex:1; min-width:0; height:100%; background:#f0f2f4"):
            with ui.column().classes("w-full p-3 gap-2"):

                # Project config card
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        with ui.row().classes("items-center gap-1 flex-1"):
                            ui.label("Projects root:").classes("text-sm font-bold shrink-0")
                            base_inp = (ui.input(value=app.storage.general.get("main_base_dir", str(Path.home())))
                                        .classes("flex-1").props("dense outlined")
                                        .bind_value(app.storage.general, "main_base_dir"))

                            async def _browse_base():
                                result = await FilePicker(
                                    base_inp.value or str(Path.home()), mode="dir")
                                if result:
                                    base_inp.value = result

                            ui.button(icon="folder_open", on_click=_browse_base
                                      ).props("flat dense round").tooltip("Browse")

                        with ui.row().classes("items-center gap-1"):
                            ui.label("Project name:").classes("text-sm font-bold shrink-0")
                            proj_inp = (ui.input(value="").props("dense outlined").classes("w-44")
                                        .bind_value(app.storage.general, "main_proj"))
                            # Keep the W&B run name in step with the project. tr_wandb_name
                            # is otherwise only refreshed by "Apply paths", so a project
                            # switch left it stale and W&B logged the run under the previous
                            # family's name. Fires only on an actual change to the project.
                            proj_inp.on_value_change(
                                lambda e: setattr(tr_wandb_name, "value", (e.value or "").strip())
                                if (e.value or "").strip() else None)

                        with ui.row().classes("items-center gap-1"):
                            ui.label("Image folder:").classes("text-sm font-bold shrink-0")
                            img_folder_inp = (
                                ui.select(
                                    options=["images", "images_cropped", "images_filtered"],
                                    value="images_cropped",
                                    label="",
                                )
                                .classes("w-44")
                                .props("dense outlined use-input new-value-mode=add-unique")
                                .bind_value(app.storage.general, "main_img_folder")
                            )

                        ui.button("Apply paths", icon="sync",
                                  on_click=lambda: _apply_paths()
                                  ).props("unelevated color=teal")

                    proj_path_lbl = ui.label("").classes(
                        "text-caption text-teal-700 font-mono mt-1")

                # Active model card
                with ui.card().classes("w-full").style("border-left:3px solid #00897b"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.icon("model_training").style("color:#00897b;font-size:20px")
                        ui.label("Active model:").classes("font-bold shrink-0").style("color:#00695c")
                        active_ckpt_inp = (
                            ui.input(placeholder="checkpoint .ckpt file")
                            .classes("flex-1").props("dense outlined clearable")
                            .bind_value(app.storage.general, "active_ckpt"))

                        async def _browse_active_ckpt():
                            cur = (app.storage.general.get("active_ckpt") or
                                   app.storage.general.get("main_base_dir") or
                                   str(Path.home()))
                            result = await FilePicker(cur, mode="file")
                            if result:
                                active_ckpt_inp.value = result
                                _quick_id_cache.clear()

                        ui.button(icon="folder_open", on_click=_browse_active_ckpt
                                  ).props("flat dense round").tooltip("Browse")

                        def _clear_model_cache():
                            _quick_id_cache.clear()
                            ui.notify("Model cache cleared — will reload on next Quick ID run.",
                                      type="info")

                        ui.button(icon="refresh", on_click=_clear_model_cache
                                  ).props("flat dense round").tooltip("Clear cached model")

                # Tabs — the numbered spine (①–⑦) is the workflow, read
                # left to right. Quick ID and Distribution are ancillary and
                # hidden from the strip, reached via the Tools ▾ menu. ☁ Cloud
                # holds pod plumbing and is hidden in Local mode.
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    with ui.tabs().classes("flex-1") as tabs:
                        t_setup    = ui.tab("Get Started")
                        t_dl       = ui.tab("① Download")
                        t_fc       = ui.tab("② Clean")
                        t_tr       = ui.tab("③ Train")
                        t_id       = ui.tab("④ Identify")
                        t_review   = ui.tab("⑤ Review")
                        t_archive  = ui.tab("⑥ Archive")
                        t_publish  = ui.tab("⑦ Publish")
                        t_all      = ui.tab("Run All")
                        # Ancillary tools — hidden from the strip, opened via
                        # the Tools ▾ menu below.
                        t_qi       = ui.tab("Quick ID")
                        t_dist     = ui.tab("Distribution")
                        t_cloud_tools = ui.tab("☁ Cloud")
                        t_qi.style("display:none")
                        t_dist.style("display:none")
                        _cloud_only(t_cloud_tools)
                    with ui.button("Tools", icon="build").props("flat dense no-caps"):
                        with ui.menu():
                            ui.menu_item("Quick ID",
                                         on_click=lambda: tabs.set_value(t_qi))
                            ui.menu_item("Distribution map",
                                         on_click=lambda: tabs.set_value(t_dist))

                # Default tab: Get Started on first launch (so users configure
                # creds). Sticky tab not implemented — the user lands on
                # whichever they last viewed via the browser's hash routing.
                with ui.tab_panels(tabs, value=t_setup).classes("w-full rounded").style(
                        "border:1px solid #dde1e4;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,.08)"):

                    with ui.tab_panel(t_setup).classes("p-4"):
                        _build_setup()

                    with ui.tab_panel(t_dl).classes("p-4"):
                        dl_cmd, dl_out_dir, dl_specsin = _build_download()

                    # ② Clean = filter + crop, with the (now optional) resize
                    # step folded in as an expansion rather than its own tab.
                    with ui.tab_panel(t_fc).classes("p-4"):
                        fc_cmd, fc_inp, fc_out, fc_spec = _build_filter_crop()
                        with ui.expansion(
                                "Optional: resize images before upload / train",
                                icon="photo_size_select_large").classes("w-full mt-3"):
                            ui.label(
                                "Downloads are already size-capped, so a separate "
                                "resize is usually unnecessary. Use this only to "
                                "shrink an existing image set."
                            ).classes("text-caption text-grey-7 mb-1")
                            rs_cmd, rs_inp = _build_resize()

                    with ui.tab_panel(t_tr).classes("p-4"):
                        tr_cmd, tr_out, tr_wandb_name, tr_sources, tr_model = _build_train()

                    with ui.tab_panel(t_id).classes("p-4"):
                        id_cmd, id_ckpt, id_nl, id_out, id_sources = _build_identify(tr_model)

                    # ⑤ Review = browse/correct predictions + the analysis plots,
                    # merged into one "look at the results" tab.
                    with ui.tab_panel(t_review).classes("p-4"):
                        review_csv, review_imgs = _build_review()
                        ui.separator().classes("my-4")
                        conf_csv = _build_confusion()

                    with ui.tab_panel(t_archive).classes("p-4"):
                        _build_archive()

                    with ui.tab_panel(t_publish).classes("p-4"):
                        _build_publish()

                    with ui.tab_panel(t_all).classes("p-4"):
                        _build_run_all(dl_cmd, fc_cmd, rs_cmd, tr_cmd, id_cmd)

                    with ui.tab_panel(t_qi).classes("p-4"):
                        _build_quick_identify()

                    with ui.tab_panel(t_dist).classes("p-4"):
                        dist_csv_inp, dist_img_inp = _build_distribution(tr_sources)

                    with ui.tab_panel(t_cloud_tools).classes("p-4"):
                        _build_cloud_tools()

        # ---- Right panel: log (full height, dark terminal) ----
        with ui.column().classes("gap-0 shrink-0").style(
                "width:42%; height:100%; overflow:hidden;"
                "background:#1e1e1e; border-left:1px solid #333") as log_col:
            with ui.row().classes("items-center justify-between px-3 py-2 shrink-0").style(
                    "background:#2d2d2d; border-bottom:1px solid #444"):
                ui.label("Output").classes("text-sm font-bold").style("color:#d4d4d4")
                with ui.row().classes("items-center gap-1"):
                    # Clickable W&B run chip — hidden until a run URL is spotted
                    # in the training output, then opens the live dashboard.
                    _wandb_link = (ui.button("W&B run", icon="insights",
                                   on_click=lambda: ui.navigate.to(_wandb_url[0], new_tab=True)
                                                    if _wandb_url[0] else None)
                                   .props("flat dense no-caps")
                                   .style("color:#ffd54f; font-weight:600"))
                    _wandb_link.set_visibility(False)
                    ui.button("Clear", icon="delete_sweep",
                              on_click=lambda: _log.clear()
                              ).props("flat dense").style("color:#aaa")
            _log = ui.log(max_lines=5000).style(
                "flex:1 1 0; min-height:0; width:100%; overflow-y:auto;"
                "font-family:'Roboto Mono',monospace; font-size:13px;"
                "background:#1e1e1e; color:#d4d4d4; padding:10px")

    # ---- Busy indicator: reflect any in-flight local proc or cloud task ----
    def _update_busy() -> None:
        proc_busy = bool(_proc and _proc.returncode is None)
        task = _cloud.get("task")
        aux  = _cloud.get("aux_task")
        cloud_busy = ((task is not None and not task.done()) or
                      (aux is not None and not aux.done()))
        if proc_busy or cloud_busy:
            what = "cloud step" if cloud_busy else "step"
            if proc_busy and cloud_busy:
                what = "step + transfer"
            busy_chip.set_text(f"⏳ {what} running — Cancel/Stop before another action")
            busy_chip.set_visibility(True)
        else:
            busy_chip.set_visibility(False)
        # A fresh page (e.g. after a refresh) starts Stop disabled, but the
        # local process it would stop may still be alive in the background.
        # Keep Stop clickable whenever a process is actually running.
        if proc_busy:
            _stop_btn.enable()
    ui.timer(1.0, _update_busy)

    # ---- Output-panel toggle ----
    _log_panel_vis = [True]

    def _toggle_log_panel():
        _log_panel_vis[0] = not _log_panel_vis[0]
        vis = _log_panel_vis[0]
        log_col.set_visibility(vis)
        log_vis_btn.props(
            f"flat {'color=white' if vis else 'color=teal-2'} "
            f"icon={'terminal' if vis else 'chevron_right'}"
        )

    # ---- Apply-paths logic (closure over all inputs) ----
    def _apply_paths(base=None, name=None, img_folder=None):
        # Explicit args come from the Get Started "Create / open project" flow;
        # when the header button calls this they're None and we read the inputs.
        if base is not None:
            base_inp.value = base
        if name is not None:
            proj_inp.value = name
        if img_folder is not None:
            img_folder_inp.value = img_folder

        base = _v(base_inp)
        name = _v(proj_inp)
        if not base:
            ui.notify("Enter a Projects root first.", type="warning")
            return
        if not name:
            ui.notify("Enter a Project name.", type="warning")
            return

        proj = Path(base) / name
        proj_path_lbl.set_text(str(proj))

        img_folder = (img_folder_inp.value or "images").strip()
        images  = str(proj / img_folder)
        specsin = str(proj / "specsin.csv")
        runs    = str(proj / "runs")
        review  = str(proj / "review")
        ckpt    = str(proj / "runs" / "checkpoints" / "last.ckpt")
        nl      = str(proj / "runs" / "nameslist.json")
        pair    = f"{specsin}:{images}"

        dl_out_dir.value = images
        dl_specsin.value = specsin
        fc_inp.value     = images
        fc_out.value     = images
        fc_spec.value    = specsin
        rs_inp.value     = images
        tr_out.value     = runs
        tr_wandb_name.value = name
        tr_sources.set_source(pair)
        id_ckpt.value    = ckpt
        id_nl.value      = nl
        id_out.value     = review
        id_sources.set_source(pair)
        predictions_csv   = str(proj / "review" / "predictions.csv")
        review_csv.value  = predictions_csv
        review_imgs.value = images
        conf_csv.value    = predictions_csv
        dist_csv_inp.value = specsin
        dist_img_inp.value = images

        ui.notify(f"Paths set for {name}", type="positive")

    # Expose the page-closure navigation + path setup to the Get Started tab,
    # which is built in a separate function and can't see these directly.
    _page_hooks["apply_paths"] = _apply_paths
    _page_hooks["goto_tab"] = lambda t: tabs.set_value(t)
    _page_hooks["tab_refs"] = {
        "setup":    t_setup, "download": t_dl,     "clean":   t_fc,
        "train":    t_tr,    "identify": t_id,      "review":  t_review,
        "archive":  t_archive, "publish": t_publish, "run_all": t_all,
    }


import os as _os_run

ui.run(
    title="Herbarium Pipeline",
    port=int(_os_run.environ.get("HERBARIUM_PORT", "8765")),
    reload=False,
    favicon="🌿",
    dark=False,
    storage_secret="herbarium-pipeline-local",
)
