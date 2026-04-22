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
import shlex
import sys
from pathlib import Path
from typing import Optional

from nicegui import app, ui

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
            mask = true.ne("") & true.ne("nan")
            for k in range(1, 6):
                c = f"top{k}_name"
                if c in df.columns:
                    mask = mask & (df[c].astype(str).str.strip() != true)
            return mask

    if t == "top1_wrong":
        sp = _sp_col()
        true = _col("true_species")
        if sp is not None and true is not None:
            return true.ne("") & true.ne("nan") & (true != sp)

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
        state_dict, model_name, num_classes, nameslist, geo_dim = load_model(
            Path(ckpt_path), [], 640)
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
                     geo_dim=geo_dim, device=device)
        print(f"[Quick ID] Model loaded: {model_name}, {num_classes} classes, device={device}")

    model     = cache["model"]
    nameslist = cache["nameslist"]
    geo_dim   = cache["geo_dim"]
    device    = cache["device"]

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

# Static-file routes registered for the Review image carousel.
# Keyed by directory path so each dir is mounted only once per server session.
_img_routes: dict[str, str] = {}


def _review_img_url(abs_path: str) -> str:
    """Return a served URL for abs_path, mounting its parent dir if needed."""
    p = Path(abs_path)
    if not p.is_file():
        return ""
    parent = str(p.parent)
    if parent not in _img_routes:
        route = f"/review_img/{len(_img_routes)}"
        app.add_static_files(route, parent)
        _img_routes[parent] = route
    return f"{_img_routes[parent]}/{p.name}"


async def _launch(cmd: list[str], on_done=None, extra_env: dict | None = None) -> None:
    global _proc
    if not cmd:
        return
    if _proc and _proc.returncode is None:
        ui.notify("A process is already running.", type="warning")
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
# Local file-picker dialog
# ---------------------------------------------------------------------------

class FilePicker(ui.dialog):
    """Navigable local-filesystem picker for files or directories."""

    def __init__(self, initial: str = "", mode: str = "dir"):
        super().__init__()
        self.mode = mode          # "dir" | "file" | "save"
        start = Path(initial or Path.home()).expanduser().resolve()
        self._cur = start.parent if start.is_file() else start
        # Walk up to the nearest directory that actually exists
        while not self._cur.is_dir() and self._cur != self._cur.parent:
            self._cur = self._cur.parent

        # For "save" mode, derive default filename from initial path
        default_name = Path(initial).name if initial and not Path(initial).is_dir() else ""

        with self, ui.card().style("min-width:560px"):
            with ui.row().classes("items-center w-full gap-1"):
                ui.button(icon="arrow_upward", on_click=self._up).props("flat round dense")
                self._loc = (ui.input("")
                             .classes("flex-1")
                             .props("dense outlined")
                             .on("keydown.enter", lambda e: self._goto(self._loc.value)))
            ui.separator()
            self._area = ui.scroll_area().style("height:380px; width:100%")
            ui.separator()
            if mode == "save":
                self._fname_inp = (ui.input(value=default_name, placeholder="filename.csv")
                                   .classes("w-full").props("dense outlined"))
            with ui.row().classes("w-full justify-end gap-2 pt-2"):
                ui.button("Cancel", on_click=self.close).props("flat")
                if mode == "dir":
                    ui.button("Select this folder",
                              on_click=lambda: self.submit(str(self._cur))
                              ).props("color=primary unelevated")
                elif mode == "save":
                    ui.button("Save here",
                              on_click=lambda: self.submit(
                                  str(self._cur / self._fname_inp.value.strip())
                              ) if self._fname_inp.value.strip() else ui.notify(
                                  "Enter a filename.", type="warning")
                              ).props("color=primary unelevated")

        self._render()

    def _goto(self, path: str) -> None:
        p = Path(path.strip()).expanduser().resolve()
        if p.is_dir():
            self._cur = p
            self._render()
        elif p.is_file() and self.mode in ("file", "save"):
            self.submit(str(p))

    def _render(self) -> None:
        self._loc.value = str(self._cur)
        entries: list[Path] = []
        try:
            entries = sorted(self._cur.iterdir(),
                             key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            pass

        self._area.clear()
        # Build the list inside the scroll area's own slot explicitly.
        # NiceGUI 3.x requires the target container to be entered via `with`
        # so element creation lands in the right slot after clear().
        with self._area:
            lst = ui.list().props("dense separator").classes("w-full")
        with lst:
            for e in entries:
                if e.is_dir():
                    with ui.item(on_click=lambda d=e: self._into(d)).props("clickable v-ripple"):
                        with ui.item_section().props("avatar"):
                            ui.icon("folder").classes("text-amber-6")
                        with ui.item_section():
                            ui.item_label(e.name)
                elif e.is_file() and self.mode in ("file", "save"):
                    if self.mode == "save":
                        action = lambda f=e: setattr(self._fname_inp, "value", f.name)
                    else:
                        action = lambda f=e: self.submit(str(f))
                    with ui.item(on_click=action).props("clickable v-ripple"):
                        with ui.item_section().props("avatar"):
                            ui.icon("description").classes("text-blue-grey-4")
                        with ui.item_section():
                            ui.item_label(e.name)

    def _up(self) -> None:
        self._cur = self._cur.parent
        self._render()

    def _into(self, d: Path) -> None:
        self._cur = d
        self._render()


# ---------------------------------------------------------------------------
# Sources panel (Train / Identify)
# ---------------------------------------------------------------------------

class SourcesPanel:
    """Editable list of  specsin.csv : images_dir  pairs."""

    def __init__(self, config_key: str = "") -> None:
        self._config_key = config_key
        self._sources: list[str] = (
            app.storage.general.get(config_key, []) if config_key else []
        )
        self._container = ui.column().classes("w-full gap-1")
        ui.button("Add Source…", icon="add", on_click=self._add).props("flat dense")
        self._refresh()

    def _persist(self) -> None:
        if self._config_key:
            app.storage.general[self._config_key] = self._sources

    async def _add(self) -> None:
        with ui.dialog() as dlg, ui.card().classes("w-full").style("min-width:480px"):
            ui.label("Add data source").classes("text-subtitle1 font-bold")
            sv = _path_input("specsin CSV:", mode="file")
            iv = _path_input("Images dir:", mode="dir")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                def _ok():
                    s, i = _v(sv), _v(iv)
                    if s and i:
                        self._sources.append(f"{s}:{i}")
                        self._refresh()
                    dlg.close()
                ui.button("Add", on_click=_ok).props("color=primary unelevated")
        await dlg

    def _refresh(self) -> None:
        self._persist()
        self._container.clear()
        with self._container:
            for idx, src in enumerate(self._sources):
                with ui.row().classes("w-full items-center gap-1"):
                    ui.label(src).classes(
                        "text-caption font-mono flex-1 bg-grey-2 px-2 py-1 rounded")
                    ui.button(icon="close",
                              on_click=lambda i=idx: self._remove(i)
                              ).props("flat dense round").tooltip("Remove")

    def _remove(self, idx: int) -> None:
        if 0 <= idx < len(self._sources):
            self._sources.pop(idx)
            self._refresh()

    def get_sources(self) -> list[str]:
        return list(self._sources)

    def set_source(self, pair: str) -> None:
        self._sources = [pair]
        self._refresh()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _v(widget) -> str:
    """Return widget value as a stripped string, handling None from clearable inputs."""
    return (widget.value or "").strip()


def _section(title: str) -> None:
    (ui.label(title)
     .classes("w-full font-bold")
     .style("color:#00695c; border-left:3px solid #00897b; background:#f0f7f6;"
            "padding:3px 10px; margin:8px 0 3px; font-size:11px;"
            "letter-spacing:.7px; text-transform:uppercase;"
            "border-radius:0 3px 3px 0; display:block"))


def _path_input(label: str, value: str = "", mode: str = "dir",
                hint: str = "") -> ui.input:
    """Label + text input + browse button row. Returns the input."""
    with ui.row().classes("w-full items-center gap-2"):
        ui.label(label).classes("w-36 text-right shrink-0 font-medium").style("color:#455a64").style("color:#455a64")
        inp = (ui.input(value=value, placeholder=hint or "")
               .classes("flex-1").props("dense outlined clearable"))

        async def _browse():
            cur = _v(inp) or str(Path.home())
            result = await FilePicker(cur, mode=mode)
            if result:
                inp.value = result

        ui.button(icon="folder_open", on_click=_browse
                  ).props("flat dense round").tooltip("Browse")
    return inp


def _text_row(label: str, value: str = "", width: str = "w-36") -> ui.input:
    with ui.row().classes("w-full items-center gap-2"):
        ui.label(label).classes("w-36 text-right shrink-0 font-medium").style("color:#455a64").style("color:#455a64")
        inp = ui.input(value=value).classes(width).props("dense outlined")
    return inp


def _inline(*items):
    """Render several (label, widget_factory) pairs in one row."""
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        for label, factory in items:
            with ui.row().classes("items-center gap-1"):
                ui.label(label).classes("text-sm")
                factory()


# ---------------------------------------------------------------------------
# Tab builders — each returns a cmd-builder callable (or None for Run All)
# ---------------------------------------------------------------------------

def _build_download() -> callable:
    gs = app.storage.general
    _section("Taxon")
    rank = (ui.radio({"family": "Family", "genus": "Genus", "order": "Order"},
                     value="family").props("inline dense")
            .bind_value(gs, "dl_rank"))
    taxon     = _text_row("Taxon name:", "Ebenaceae", "w-48").bind_value(gs, "dl_taxon")
    continent = _text_row("Continent:", "AFRICA", "w-36").bind_value(gs, "dl_continent")

    _section("Country filter  (mutually exclusive)")
    inc = _text_row("Include countries:", "", "w-60").bind_value(gs, "dl_inc")
    exc = _text_row("Exclude countries:", "MG", "w-60").bind_value(gs, "dl_exc")
    ui.label("Space-separated ISO-2 codes, e.g. ZA NG TZ").classes("text-caption text-grey-7 ml-48")

    _section("Source  (DwC-A ZIP or live API)")
    dwca = (_path_input("Local DwC-A ZIP:", mode="file",
                        hint="Select a downloaded GBIF DwC-A ZIP to skip the API")
            .bind_value(gs, "dl_dwca"))

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

    ui.button("Run Download", icon="download",
              on_click=lambda: _launch(_dl_cmd())
              ).props("color=primary unelevated").classes("mt-4")

    def _dl_cmd() -> list[str]:
        d = _v(dwca)
        t = _v(taxon)
        if not d and not t:
            raise ValueError("Enter a taxon name or select a DwC-A ZIP.")
        cmd = [sys.executable, str(SCRIPTS["download"])]
        if d: cmd += ["--dwca", d]
        if t: cmd += [f"--{rank.value}", t]
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
        return cmd

    return _dl_cmd, out_dir, specsin


def _build_filter_crop() -> callable:
    gs = app.storage.general
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
              on_click=lambda: _launch(_fc_cmd())
              ).props("color=primary unelevated").classes("mt-4")

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
              on_click=lambda: _launch(_rs_cmd())
              ).props("color=primary unelevated").classes("mt-4")

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
    _section("Data sources  (specsin CSV : images directory)")
    tr_sources = SourcesPanel("train_sources")

    _section("Output")
    tr_out = _path_input("Output / run dir:", mode="dir").bind_value(gs, "tr_out")

    _section("Model")
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
            ui.label("Max per species (0=all):").classes("text-sm")
            tr_max_per_sp = ui.input(value="0").classes("w-20").props("dense outlined").bind_value(gs, "tr_max_per_sp")
        nccl_p2p_disable = (ui.checkbox(
            "NCCL_P2P_DISABLE (only for multi-GPU without NVLink)", value=False)
            .tooltip("Sets NCCL_P2P_DISABLE=1 — do NOT enable if NVLink is present")
            .bind_value(gs, "tr_nccl_p2p"))

    _section("Training stages")
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
    ui.label("Cool-down runs after stage 2 with reduced batch/LR — helps settle into flatter minima"
             ).classes("text-caption text-grey-7")

    _section("Classification target")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
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

    _section("Location")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        use_location = (ui.checkbox("Use lat/lon during training (--use-location)", value=False)
                        .bind_value(gs, "tr_use_location"))
        with ui.row().classes("items-center gap-1"):
            ui.label("Geo MLP dim:").classes("text-sm")
            geo_dim = ui.input(value="64").classes("w-20").props("dense outlined").bind_value(gs, "tr_geo_dim")
        geo_dim.bind_enabled_from(use_location, "value")

    _section("Logging")
    with ui.row().classes("w-full items-center gap-2"):
        ui.label("WandB project:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
        wandb_proj = ui.input(value="").classes("w-48").props("dense outlined").bind_value(gs, "tr_wandb_proj")
        ui.label("Run name:").classes("text-sm ml-4")
        wandb_name = ui.input(value="herbarium_run").classes("w-48").props("dense outlined").bind_value(gs, "tr_wandb_name")
    resume = _path_input("Resume checkpoint:", mode="file").bind_value(gs, "tr_resume")
    reset_opt = (ui.checkbox(
        "Reset optimizer  (load weights only — use when starting a fresh stage 2 from a stage-1 checkpoint)",
        value=False).bind_value(gs, "tr_reset_optimizer")
        .tooltip("Discards the saved optimizer/LR-schedule state so stage 2 starts "
                 "with a clean optimizer at the LR you specify above. "
                 "Leave unticked to continue an interrupted stage-2 run."))

    # ── Remote training (optional) ────────────────────────────────────────────
    ui.separator().classes("my-3")
    with ui.expansion("Remote training  (optional)").classes("w-full"):
        with ui.column().classes("w-full gap-2 pt-2"):

            with ui.row().classes("w-full items-center gap-2"):
                ui.label("SSH host:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
                ssh_host = (ui.input(placeholder="user@hostname")
                            .classes("flex-1").props("dense outlined")
                            .bind_value(gs, "tr_ssh_host"))
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Local project root:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
                ssh_lroot = (ui.input(placeholder="/mnt/e/MyProject")
                             .classes("flex-1").props("dense outlined")
                             .bind_value(gs, "tr_ssh_lroot"))
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Remote project root:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
                ssh_rroot = (ui.input(placeholder="/home/user/MyProject")
                             .classes("flex-1").props("dense outlined")
                             .bind_value(gs, "tr_ssh_rroot"))
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Activation command:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
                ssh_activate = (ui.input(value="conda activate p12")
                                .classes("flex-1").props("dense outlined")
                                .bind_value(gs, "tr_ssh_activate"))

            ui.separator().classes("my-1")

            # Transfer method
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Transfer via:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
                xfer_mode = (ui.radio(
                    {"rsync": "rsync (direct SSH)", "rclone": "rclone (cloud bucket)"},
                    value="rsync").props("inline dense")
                    .bind_value(gs, "tr_xfer_mode"))

            # rclone-specific fields — shown only when rclone is selected
            with ui.row().classes("w-full items-center gap-2") as rclone_row:
                ui.label("rclone remote:bucket:").classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
                rclone_remote = (ui.input(placeholder="r2:mybucket  or  s3:mybucket")
                                 .classes("flex-1").props("dense outlined")
                                 .bind_value(gs, "tr_rclone_remote"))

            # Toggle visibility
            xfer_mode.bind_value_to(rclone_row, "visible",
                                    forward=lambda v: v == "rclone")
            rclone_row.visible = False

            with ui.row().classes("gap-2 mt-1 ml-48"):
                ui.button("Sync data →", icon="upload",
                          on_click=lambda: _sync_to()
                          ).props("outlined color=teal dense")
                ui.button("← Sync checkpoint", icon="download",
                          on_click=lambda: _sync_from()
                          ).props("outlined color=teal dense")
            ui.label(
                "rsync: copies directly over SSH using --partial so interrupted transfers resume. "
                "rclone: uploads to a cloud bucket then the remote downloads from it — "
                "faster for large datasets on slow uplinks. Leave SSH host blank to run locally."
            ).classes("text-caption text-grey-6 mt-1")

    def _tr_env() -> dict:
        env = {}
        if nccl_p2p_disable.value:
            env["NCCL_P2P_DISABLE"] = "1"
        return env

    ui.button("Run Training", icon="model_training",
              on_click=lambda: _launch(_tr_cmd(), extra_env=_tr_env())
              ).props("color=primary unelevated").classes("mt-4")

    def _tr_cmd() -> list[str]:
        srcs = tr_sources.get_sources()
        if not srcs: raise ValueError("Add at least one data source.")
        out = _v(tr_out)
        if not out: raise ValueError("Enter an output directory.")
        n_gpus = int(_v(tr_gpus) or "1")

        host   = _v(ssh_host)
        lroot  = _v(ssh_lroot).rstrip("/")
        rroot  = _v(ssh_rroot).rstrip("/")

        def _repath(p: str) -> str:
            """Swap local root for remote root in a path string."""
            return p.replace(lroot, rroot) if (host and lroot and rroot) else p

        if host:
            # Remote: single-GPU python call (torchrun works too but needs nproc agreed)
            cmd = ["python", "-u", _repath(str(SCRIPTS["train"]))]
        elif n_gpus > 1:
            cmd = [sys.executable, "-u", "-m", "torch.distributed.run",
                   "--standalone", f"--nproc_per_node={n_gpus}",
                   str(SCRIPTS["train"])]
        else:
            cmd = [sys.executable, "-u", str(SCRIPTS["train"])]

        cmd += ["--sources"] + [_repath(s) for s in srcs] + [
               "--output-dir",    _repath(out),
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
            cmd += ["--resume", _repath(ck)]
            if reset_opt.value:
                cmd += ["--reset-optimizer"]
        if use_location.value:
            cmd += ["--use-location", "--geo-dim", geo_dim.value]
        mps = _v(tr_max_per_sp)
        if mps and mps != "0": cmd += ["--max-per-species", mps]

        if host:
            activate = _v(ssh_activate)
            remote_str = " ".join(shlex.quote(a) for a in cmd)
            if activate:
                remote_str = f"{activate} && {remote_str}"
            return ["ssh", "-T", host, remote_str]

        return cmd

    async def _sync_to():
        lroot  = _v(ssh_lroot).rstrip("/")
        rroot  = _v(ssh_rroot).rstrip("/")
        if not lroot or not rroot:
            ui.notify("Set local and remote project roots first.", type="warning")
            return
        if xfer_mode.value == "rclone":
            bucket = _v(rclone_remote).rstrip("/")
            if not bucket:
                ui.notify("Enter a rclone remote:bucket.", type="warning")
                return
            await _launch(["rclone", "copy", "--progress",
                           f"{lroot}/", f"{bucket}/"])
        else:
            host = _v(ssh_host)
            if not host:
                ui.notify("Set SSH host first.", type="warning")
                return
            await _launch(["rsync", "-avz", "--partial", "--info=progress2",
                           f"{lroot}/", f"{host}:{rroot}/"])

    async def _sync_from():
        lroot = _v(ssh_lroot).rstrip("/")
        rroot = _v(ssh_rroot).rstrip("/")
        out   = _v(tr_out)
        if not lroot or not rroot or not out:
            ui.notify("Set local root, remote root and output dir first.", type="warning")
            return
        rel        = out[len(lroot):].lstrip("/") if out.startswith(lroot) else Path(out).name
        remote_out = f"{rroot}/{rel}"
        if xfer_mode.value == "rclone":
            bucket = _v(rclone_remote).rstrip("/")
            if not bucket:
                ui.notify("Enter a rclone remote:bucket.", type="warning")
                return
            await _launch(["rclone", "copy", "--progress",
                           f"{bucket}/{rel}/", f"{out}/"])
        else:
            host = _v(ssh_host)
            if not host:
                ui.notify("Set SSH host first.", type="warning")
                return
            await _launch(["rsync", "-avz", "--partial",
                           f"{host}:{remote_out}/", f"{out}/"])

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

    _section("Thresholds")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
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

    ui.button("Run Identify", icon="manage_search",
              on_click=lambda: _launch(_id_cmd())
              ).props("color=primary unelevated").classes("mt-4")

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
            agg: dict[str, float] = {}
            for name, prob in _top5_items(row):
                g = name.split()[0] if name and name != "nan" else name
                agg[g] = agg.get(g, 0) + prob
            return _render(sorted(agg.items(), key=lambda x: x[1], reverse=True),
                           "Genus", italic=True)

        if level == "family":
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

    _section("Predictions CSV")
    rev_csv  = _path_input("predictions.csv:", mode="file",
                           value=app.storage.general.get("review_csv", ""))
    rev_csv.bind_value(app.storage.general, "review_csv")
    rev_imgs = _path_input("Images dir (if CSV has relative paths):", mode="dir",
                           value=app.storage.general.get("review_imgs", ""))
    rev_imgs.bind_value(app.storage.general, "review_imgs")
    with ui.row().classes("w-full items-center gap-2 mt-1 ml-48"):
        load_btn = (ui.button("Load", icon="upload_file")
                    .props("color=primary unelevated"))
    summary_lbl = ui.label("").classes("text-caption text-grey-7 mt-1")

    _section("Filter & Sort")
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        filter_sel = ui.select(
            {"all": "All", "indets": "Indets only",
             "flagged": "Flagged only", "misid": "Misidentified (pred ≠ true)",
             "high_conf": "High confidence (≥ 90%)"},
            value="all", label="Show"
        ).classes("w-52").props("dense outlined")
        sort_sel = ui.select(
            {"conf_desc": "Confidence ↓", "conf_asc": "Confidence ↑",
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

    # Free-form AI filter
    with ui.row().classes("w-full items-center gap-2 mt-2"):
        ai_filter_inp = (ui.input(
            placeholder="e.g.  genus Uvaria · confidence < 30% · none of top 5 correct")
            .classes("flex-1").props("dense outlined clearable"))
        ai_filter_btn = (ui.button("AI Filter", icon="auto_awesome")
                         .props("unelevated dense color=deep-purple-4")
                         .tooltip("Use Claude to interpret your query"))
    ai_filter_lbl = ui.label("").classes("text-caption text-grey-6 mt-0")

    ui.separator().classes("my-3")

    # Carousel layout
    with ui.row().classes("w-full gap-4 items-start"):

        # Left: image + nav
        with ui.column().classes("items-center gap-2").style(
                "min-width:490px; max-width:490px"):
            img_el = ui.image("").style(
                "width:480px; height:400px; object-fit:contain;"
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
                            .tooltip("Open full image"))

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
        """Return absolute image path, trying abs_path → filename → imgs_dir/fname."""
        for col in ("abs_path", "filename"):
            v = row.get(col, "")
            if v and v == v and str(v) not in ("", "nan"):  # non-NaN, non-empty
                return str(v)
        fname = str(row.get("fname", ""))
        imgs  = _v(rev_imgs)
        if fname and imgs:
            return str(Path(imgs) / fname)
        return ""

    def _get_top5(row) -> list[str]:
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
            pred_sp = str(row.get("pred_species", row.get("top1_name", "")))
            pred_g  = pred_sp.split()[0] if pred_sp and pred_sp not in ("", "nan") else "?"
            true_g  = true_sp.split()[0] if true_sp and true_sp not in ("", "nan") else ""
            match   = (f" <span style='color:{'#388e3c' if pred_g==true_g else '#d32f2f'}'>"
                       f"{'✓' if pred_g==true_g else '✗'}</span>") if true_g else ""
            level_line = (f"<b>Predicted genus:</b> <i>{pred_g}</i>{match}<br>"
                          + (f"<b>True genus:</b> <i>{true_g}</i><br>" if true_g else ""))
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

        info_html.set_content(
            f"<div style='font-size:13px;line-height:1.6'>"
            f"<b>Confidence:</b> {conf_val:.1%}{flag_badge}<br>"
            + level_line
            + (f"<small style='color:#888'>{source}"
               + (f" | {cat}" if cat and cat != "nan" else "")
               + (f" | 📍 {geo_str}" if geo_str else "")
               + "</small><br>"
               if source or (cat and cat != "nan") or geo_str else "")
            + f"<small style='color:#aaa;font-family:monospace'>{Path(fname).name}</small>"
            "</div>"
        )
        bars_html.set_content(_bars_html(row, level=level))

        top5 = _get_top5(row)
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

        if filt == "indets":
            mask = df["indet"].astype(str).str.lower().isin(("true", "1"))
        elif filt == "flagged":
            mask = df["flagged"].astype(str).str.lower().isin(("true", "1"))
        elif filt == "misid":
            if level == "family" and "pred_family" in df.columns and "true_family" in df.columns:
                pred = df["pred_family"].astype(str).str.strip()
                true = df["true_family"].astype(str).str.strip()
                mask = true.ne("") & true.ne("nan") & true.ne(pred)
            elif level == "genus" and "true_species" in df.columns:
                pred_g = df[sp_col].astype(str).str.split().str[0]
                true_g = df["true_species"].astype(str).str.split().str[0]
                mask   = true_g.ne("") & true_g.ne("nan") & true_g.ne(pred_g)
            else:
                if "true_species" in df.columns:
                    true_str = df["true_species"].astype(str).str.strip()
                    mask = (true_str.ne("") & true_str.ne("nan") &
                            true_str.ne(df[sp_col].astype(str).str.strip()))
                else:
                    mask = _pd.Series(False, index=df.index)
        elif filt == "high_conf":
            mask = df[conf_col].astype(float) >= 0.90
        else:
            mask = _pd.Series(True, index=df.index)

        # Combine with AI filter if active
        ai_mask = _st.get("ai_mask")
        if ai_mask is not None:
            mask = mask & ai_mask

        view = df[mask].copy()
        sort = sort_sel.value
        if sort == "conf_desc":
            view = view.sort_values(conf_col, ascending=False)
        elif sort == "conf_asc":
            view = view.sort_values(conf_col, ascending=True)
        elif sort == "species":
            if level == "genus":
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
            df = _pd.read_csv(path)
            if "indet"   not in df.columns: df["indet"]   = False
            if "flagged" not in df.columns: df["flagged"] = False
            _st["df"] = df
            n_total = len(df)
            n_indet = df["indet"].astype(str).str.lower().isin(("true", "1")).sum()
            n_flag  = df["flagged"].astype(str).str.lower().isin(("true", "1")).sum()
            summary_lbl.set_text(
                f"Loaded {n_total:,} total  ·  {n_indet:,} indets  ·  {n_flag:,} flagged")
            _apply_filter()
            ui.notify(f"Loaded {n_total:,} predictions", type="positive")
        except Exception as exc:
            ui.notify(f"Error loading CSV: {exc}", type="negative")

    def _go(delta: int):
        if _st["view"] is not None:
            _show(_st["idx"] + delta)

    def _open_file():
        view = _st["view"]
        if view is None or len(view) == 0:
            return
        import subprocess as _sp
        row  = view.iloc[_st["idx"]]
        path = str(row.get("abs_path", row.get("filename", "")))
        if path and Path(path).is_file():
            _sp.Popen(["xdg-open", path], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

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
            else:  # determination
                new_name = det_sel.value
                current  = str(sp.loc[mask, "species"].iloc[0])
                sp.loc[mask, "old_determination"] = current
                sp.loc[mask, "species"]           = new_name
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
        if _write_back("determine"):
            action_lbl.set_text(f"Determined → {new_name}")
            ui.notify(f"Determined: {new_name}", type="positive")

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

    # Accuracy bar chart
    _section("Per-species Accuracy")
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
        pred_sp_col = "pred_species" if "pred_species" in df.columns else "top1_name"
        if "true_species" not in df.columns or pred_sp_col not in df.columns:
            ui.notify("Need 'true_species' and 'pred_species' columns.", type="warning")
            return

        if level == "genus":
            df["_true"] = df["true_species"].astype(str).str.split().str[0]
            df["_pred"] = df[pred_sp_col].astype(str).str.split().str[0]
            level_label = "Genus"
        elif level == "family":
            if "true_family" not in df.columns or "pred_family" not in df.columns:
                ui.notify("Family columns not in this CSV — re-run Identify to add them.",
                          type="warning")
                return
            df["_true"] = df["true_family"].astype(str).str.strip()
            df["_pred"] = df["pred_family"].astype(str).str.strip()
            level_label = "Family"
        else:
            df["_true"] = df["true_species"].astype(str).str.strip()
            df["_pred"] = df[pred_sp_col].astype(str).str.strip()
            level_label = "Species"

        true_col = "_true"
        pred_col = "_pred"
        df = df[[true_col, pred_col]].dropna()
        df = df[df[true_col].str.strip().replace("nan", "") != ""]
        df = df[df[pred_col].str.strip().replace("nan", "") != ""]

        # Min-samples filter
        min_s = max(1, int(min_s_inp.value or 1))
        vc = df[true_col].value_counts()
        df = df[df[true_col].isin(vc[vc >= min_s].index)]

        if df.empty:
            ui.notify("No identified specimens after filtering.", type="info")
            return

        # Full crosstab
        ct = _pd.crosstab(df[true_col], df[pred_col])

        # Most-confused restriction.
        # Y-axis: top N true species by off-diagonal error rate.
        # X-axis: top N predicted species by total count in those rows (different set/order).
        n = int(top_n_inp.value or 0)
        if n > 0:
            off = ct.copy().astype(float)
            for sp in ct.index:
                if sp in ct.columns:
                    off.loc[sp, sp] = 0.0
            error_rate = (off.sum(axis=1) / ct.sum(axis=1)).sort_values(ascending=False)
            top_true = list(error_rate.head(n).index)
            ct = ct.loc[[s for s in top_true if s in ct.index]]
            # Predicted axis: most commonly predicted species for these rows
            col_counts = ct.sum(axis=0).sort_values(ascending=False)
            top_pred = list(col_counts.head(n).index)
            ct = ct[top_pred]

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

        # Simple array data — most compatible ECharts heatmap format.
        data = [
            [j, i, round(float(ct_plot.iloc[i, j]), 4)]
            for i in range(len(true_labels))
            for j in range(len(pred_labels))
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
        top5_cols = [c for c in df.columns if c in {f"top{k}_name" for k in range(1, 6)}]
        top5_acc  = None
        if top5_cols:
            true_series = df["true_species"].astype(str)
            if level == "genus":
                true_series = true_series.str.split().str[0]
                hit = _pd.Series(False, index=df.index)
                for col in top5_cols:
                    hit |= (true_series == df[col].astype(str).str.split().str[0])
            else:  # species (family top-5 not meaningful without per-rank lookup)
                hit = _pd.Series(False, index=df.index)
                for col in top5_cols:
                    hit |= (true_series == df[col].astype(str))
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
        acc_df = (df.groupby(true_col)
                  .apply(lambda g: (g[true_col] == g[pred_col]).mean() * 100)
                  .sort_values(ascending=True)
                  .rename("accuracy"))
        acc_labels = acc_df.index.tolist()
        acc_values = [round(v, 1) for v in acc_df.values]
        acc_height = max(300, len(acc_labels) * 16 + 80)
        acc_chart.style(f"height:{acc_height}px; width:100%")
        acc_chart.options["yAxis"]["data"] = acc_labels
        acc_chart.options["series"][0]["data"] = acc_values
        acc_chart.options["xAxis"]["name"] = f"{level_label} accuracy (%)"
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


def _build_run_all(dl_cmd, fc_cmd, rs_cmd, tr_cmd, id_cmd) -> None:
    ui.label("Runs steps in sequence using the settings configured in each tab."
             ).classes("text-body1 mt-2")

    with ui.card().classes("w-full mt-4"):
        ui.label("Steps to run").classes("text-subtitle2 font-bold mb-2")
        run_dl = ui.checkbox("1  Download",      value=True)
        run_fc = ui.checkbox("2  Filter & Crop",  value=True)
        run_rs = ui.checkbox("3  Resize",         value=True)
        run_tr = ui.checkbox("4  Train",           value=True)
        run_id = ui.checkbox("5  Identify",        value=True)

    async def _run_all():
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

    ui.button("Run Full Pipeline", icon="play_circle",
              on_click=_run_all).props("color=positive unelevated size=lg").classes("mt-6")


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
        for col in ("abs_path", "filename"):
            v = row.get(col, "")
            if v and v == v and str(v) not in ("", "nan"):
                return str(v)
        fname = str(row.get("fname", ""))
        if fname and imgs_dir:
            return str(Path(imgs_dir) / fname)
        return ""

    def _top5(row) -> list[tuple[str, float]]:
        if "top1_name" in row.index:
            items = []
            for k in range(1, 6):
                n = row.get(f"top{k}_name", "")
                n = "" if (n != n or n is None) else str(n)
                if not n or n == "nan":
                    break
                items.append((n, float(row.get(f"top{k}_prob", 0) or 0)))
            return items or [(str(row.get("pred_species", "")),
                              float(row.get("confidence", 0) or 0))]
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

        conf  = float(row.get("confidence", row.get("top1_prob", 0)) or 0)
        pred  = str(row.get("pred_species", row.get("top1_name", "")))
        true  = str(row.get("true_species", ""))
        fname = str(row.get("fname", row.get("filename", "")))
        match = ""
        if true and true != "nan":
            ok = true.strip() == pred.strip()
            match = (f" <span style='color:{'#66bb6a' if ok else '#ef5350'}'>"
                     f"{'✓' if ok else '✗'}</span>")

        info_html.set_content(
            f"<div style='line-height:1.8'>"
            f"<div style='font-size:22px;font-weight:700;font-style:italic;"
            f"color:#e0f2f1;margin-bottom:4px'>{pred}</div>"
            f"<div style='font-size:16px'><b>Confidence:</b> {conf:.1%}{match}</div>"
            + (f"<div style='font-size:16px'><b>True:</b> <i>{true}</i></div>"
               if true and true != "nan" else "")
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
    global _log, _status, _stop_btn

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
    with ui.header().classes("items-center justify-between bg-teal-700 text-white px-6 py-2"):
        ui.label("Herbarium Classification Pipeline").classes("text-h6 font-bold")
        with ui.row().classes("items-center gap-4"):
            _status = ui.label("Ready").classes("text-body2")
            _stop_btn = (ui.button("Stop", icon="stop", on_click=_stop_process)
                         .props("flat color=white")
                         .classes("text-white"))
            _stop_btn.disable()
            (ui.button("Quit", icon="power_settings_new", on_click=_quit)
             .props("flat color=white")
             .classes("text-white"))

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

                # Tabs
                with ui.tabs().classes("w-full") as tabs:
                    t_dl     = ui.tab("1  Download")
                    t_fc     = ui.tab("2  Filter & Crop")
                    t_rs     = ui.tab("3  Resize")
                    t_tr     = ui.tab("4  Train")
                    t_id     = ui.tab("5  Identify")
                    t_review   = ui.tab("Review")
                    t_conf     = ui.tab("Analysis")
                    t_qi       = ui.tab("Quick ID")
                    t_dist     = ui.tab("Distribution")
                    t_all      = ui.tab("Run All")

                with ui.tab_panels(tabs, value=t_dl).classes("w-full rounded").style(
                        "border:1px solid #dde1e4;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,.08)"):

                    with ui.tab_panel(t_dl).classes("p-4"):
                        dl_cmd, dl_out_dir, dl_specsin = _build_download()

                    with ui.tab_panel(t_fc).classes("p-4"):
                        fc_cmd, fc_inp, fc_out, fc_spec = _build_filter_crop()

                    with ui.tab_panel(t_rs).classes("p-4"):
                        rs_cmd, rs_inp = _build_resize()

                    with ui.tab_panel(t_tr).classes("p-4"):
                        tr_cmd, tr_out, tr_wandb_name, tr_sources, tr_model = _build_train()

                    with ui.tab_panel(t_id).classes("p-4"):
                        id_cmd, id_ckpt, id_nl, id_out, id_sources = _build_identify(tr_model)

                    with ui.tab_panel(t_review).classes("p-4"):
                        review_csv, review_imgs = _build_review()

                    with ui.tab_panel(t_conf).classes("p-4"):
                        conf_csv = _build_confusion()

                    with ui.tab_panel(t_qi).classes("p-4"):
                        _build_quick_identify()

                    with ui.tab_panel(t_dist).classes("p-4"):
                        dist_csv_inp, dist_img_inp = _build_distribution(tr_sources)

                    with ui.tab_panel(t_all).classes("p-4"):
                        _build_run_all(dl_cmd, fc_cmd, rs_cmd, tr_cmd, id_cmd)

        # ---- Right panel: log (full height, dark terminal) ----
        with ui.column().classes("gap-0 shrink-0").style(
                "width:42%; height:100%; overflow:hidden;"
                "background:#1e1e1e; border-left:1px solid #333"):
            with ui.row().classes("items-center justify-between px-3 py-2 shrink-0").style(
                    "background:#2d2d2d; border-bottom:1px solid #444"):
                ui.label("Output").classes("text-sm font-bold").style("color:#d4d4d4")
                ui.button("Clear", icon="delete_sweep",
                          on_click=lambda: _log.clear()
                          ).props("flat dense").style("color:#aaa")
            _log = ui.log(max_lines=5000).style(
                "flex:1 1 0; min-height:0; width:100%; overflow-y:auto;"
                "font-family:'Roboto Mono',monospace; font-size:13px;"
                "background:#1e1e1e; color:#d4d4d4; padding:10px")

    # ---- Apply-paths logic (closure over all inputs) ----
    def _apply_paths():
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


ui.run(
    title="Herbarium Pipeline",
    port=8765,
    reload=False,
    favicon="🌿",
    dark=False,
    storage_secret="herbarium-pipeline-local",
)
