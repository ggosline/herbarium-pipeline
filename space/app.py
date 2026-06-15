"""
Gradio Space for herbarium specimen identification.

Hosts a picker over one or more trained classifiers (currently the
Africa Angiosperms family-level model). Each entry in MODELS resolves
to a Hugging Face Hub repo containing:

    model.ckpt        — Lightning checkpoint (state_dict + hyper_parameters)
    nameslist.json    — list of class names at the model's label rank
    config.json       — { "model_name", "image_sz", "label_level" }

Geo-aware checkpoints (use_location=True at training time) are detected
automatically — the Space then exposes optional latitude / longitude
inputs and reconstructs the backbone + geo_mlp + head architecture
that the training script used.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torchvision import transforms

# ZeroGPU support is optional. The `spaces` lib only has a working runtime
# on Spaces with ZeroGPU hardware (HF Pro). On a free CPU Space, importing
# the decorator works but calling it raises "No API found". Fall back to
# a no-op decorator so the same code runs on either tier.
try:
    import spaces  # noqa: F401
    _ZEROGPU = True
except Exception:
    _ZEROGPU = False

def _gpu_decorator(*dargs, **dkwargs):
    if _ZEROGPU:
        try:
            return spaces.GPU(*dargs, **dkwargs)
        except Exception:
            pass
    def _identity(fn):
        return fn
    return _identity


# --- Model registry --------------------------------------------------------
# Models are discovered on the Hub instead of being hardcoded: every repo
# published by push_model.py carries the `herbarium-pipeline` tag and a
# config.json, so a freshly-published family appears here automatically
# (use the Refresh button — no Space redeploy needed).
HF_AUTHOR = os.environ.get("HF_AUTHOR", "ggosline")
PIPELINE_TAG = "herbarium-pipeline"

# Used only if discovery returns nothing (offline / Hub hiccup) so the
# Space never launches empty.
FALLBACK_MODELS: dict[str, dict[str, str]] = {
    "Africa — Angiosperms (Magnoliopsida + Liliopsida), family rank": {
        "repo": "ggosline/herbarium-africa-angiosperms-family",
        "description": (
            "Family-level classifier (val_Accuracy ≈ 0.92) trained on GBIF-"
            "sourced herbarium sheets of African angiosperms — both "
            "Magnoliopsida (dicots) and Liliopsida (monocots)."
        ),
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
TOPK = 5

# Bounded LRU of fully-built models. Family models are ~0.4–1.2 GB each, so
# we keep only the few most-recently-used in memory and evict the rest.
_MODEL_CACHE_MAX = int(os.environ.get("MODEL_CACHE_MAX", "3"))
_loaded: "OrderedDict[str, Any]" = OrderedDict()  # repo -> {model, nameslist, config, geo_dim}


def _describe(cfg: dict[str, Any]) -> str:
    bits = []
    if cfg.get("label_level"):
        bits.append(f"{cfg['label_level']}-rank")
    if cfg.get("num_classes"):
        bits.append(f"{cfg['num_classes']} classes")
    if cfg.get("val_accuracy") is not None:
        bits.append(f"val_acc≈{cfg['val_accuracy']:.3f}")
    elif cfg.get("valid_loss") is not None:
        bits.append(f"valid_loss≈{cfg['valid_loss']:.3f}")
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f"Backbone `{cfg.get('model_name', '?')}`{suffix}."


def discover_models() -> dict[str, dict[str, Any]]:
    """Scan the Hub for this project's published models.

    Returns {display_name: {repo, description, **config}}. Falls back to
    FALLBACK_MODELS when nothing is found so the UI is never empty.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        repos = list(HfApi().list_models(author=HF_AUTHOR, filter=PIPELINE_TAG))
    except Exception as e:  # network / auth issues shouldn't crash the Space
        print(f"[discover] list_models failed: {e}")
        repos = []
    for r in repos:
        try:
            with open(hf_hub_download(repo_id=r.id, filename="config.json")) as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[discover] skipping {r.id}: no usable config.json ({e})")
            continue
        name = cfg.get("display_name") or r.id.split("/")[-1]
        out[name] = {"repo": r.id, "description": _describe(cfg), **cfg}
    if not out:
        print("[discover] no models found on the Hub — using fallback registry")
        return dict(FALLBACK_MODELS)
    print(f"[discover] found {len(out)} model(s): {', '.join(out)}")
    return dict(sorted(out.items()))


MODELS: dict[str, dict[str, Any]] = discover_models()
DEFAULT_MODEL = next(iter(MODELS))


# ---------------------------------------------------------------------------
# Architecture matching train_herbarium.py / identify_herbarium.py
# ---------------------------------------------------------------------------

class _GeoModel(nn.Module):
    """Backbone + geo MLP + head — same shape as the training script."""

    def __init__(self, backbone: nn.Module, geo_mlp: nn.Module,
                 head: nn.Module, geo_dim: int):
        super().__init__()
        self.backbone = backbone
        self.geo_mlp = geo_mlp
        self.head = head
        self.geo_dim = geo_dim

    def forward(self, x: torch.Tensor, geo: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.backbone(x)
        if geo is None:
            geo = torch.zeros(feats.shape[0], 4, device=feats.device)
        geo_feats = self.geo_mlp(geo)
        return self.head(torch.cat([feats, geo_feats], dim=1))


def _encode_coords(lat: float | None, lon: float | None) -> torch.Tensor:
    """4-feature encoding matching train_herbarium._encode_coords:
    (cos(lat)cos(lon), cos(lat)sin(lon), sin(lat), has_location)."""
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return torch.zeros(1, 4, dtype=torch.float32)
    lr, ln = np.radians(lat), np.radians(lon)
    return torch.tensor(
        [[np.cos(lr) * np.cos(ln), np.cos(lr) * np.sin(ln), np.sin(lr), 1.0]],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _strip_lightning_prefix(sd: dict) -> dict:
    """Remove Lightning + torch.compile wrapping. Training saves keys like
    'model._orig_mod.backbone.cls_token' or 'model.head.weight'."""
    out: dict = {}
    for k, v in sd.items():
        # Skip non-tensor metadata like *_labels_t / *_coords_t.
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for p in ("model._orig_mod.", "model.module.", "model."):
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out


def _load_from_hub(repo: str) -> dict[str, Any]:
    if repo in _loaded:
        _loaded.move_to_end(repo)  # mark most-recently-used
        return _loaded[repo]
    ckpt_path = hf_hub_download(repo_id=repo, filename="model.ckpt")
    names_path = hf_hub_download(repo_id=repo, filename="nameslist.json")
    cfg_path = hf_hub_download(repo_id=repo, filename="config.json")

    with open(names_path) as f:
        nameslist: list[str] = json.load(f)
    with open(cfg_path) as f:
        config: dict[str, Any] = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {}) or {}
    use_location = bool(hp.get("use_location", False))
    geo_dim = int(hp.get("geo_dim", 0)) if use_location else 0

    state = _strip_lightning_prefix(ckpt.get("state_dict", ckpt))
    num_classes = len(nameslist)

    if use_location and geo_dim > 0:
        backbone = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
        feat_dim = backbone.num_features
        geo_mlp = nn.Sequential(
            nn.Linear(4, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim),
        )
        head = nn.Linear(feat_dim + geo_dim, num_classes)
        # Backbone keys: backbone.* (timm strips its own num_classes head when 0).
        bb_sd = {k[len("backbone."):]: v for k, v in state.items()
                 if k.startswith("backbone.")}
        gm_sd = {k[len("geo_mlp."):]: v for k, v in state.items()
                 if k.startswith("geo_mlp.")}
        hd_sd = {k[len("head."):]: v for k, v in state.items()
                 if k.startswith("head.")}
        bb_missing, bb_unexp = backbone.load_state_dict(bb_sd, strict=False)
        gm_missing, gm_unexp = geo_mlp.load_state_dict(gm_sd, strict=False)
        hd_missing, hd_unexp = head.load_state_dict(hd_sd, strict=False)
        print(f"[load] geo: backbone missing={len(bb_missing)} unexp={len(bb_unexp)}; "
              f"geo_mlp missing={len(gm_missing)} unexp={len(gm_unexp)}; "
              f"head missing={len(hd_missing)} unexp={len(hd_unexp)}")
        model = _GeoModel(backbone, geo_mlp, head, geo_dim)
    else:
        model = timm.create_model(config["model_name"], pretrained=False,
                                  num_classes=num_classes)
        # Backbone-then-head namespace: training stored as backbone.* + head.*.
        # Try that first; fall back to direct apply.
        bb_sd = {k[len("backbone."):]: v for k, v in state.items()
                 if k.startswith("backbone.")}
        if bb_sd:
            adj: dict = dict(bb_sd)
            for k, v in state.items():
                if k.startswith("head."):
                    adj[k] = v
            state = adj
        missing, unexp = model.load_state_dict(state, strict=False)
        print(f"[load] plain: missing={len(missing)} unexp={len(unexp)}")

    model.eval()
    _loaded[repo] = {
        "model": model,
        "nameslist": nameslist,
        "config": config,
        "use_location": use_location,
        "geo_dim": geo_dim,
    }
    _loaded.move_to_end(repo)
    while len(_loaded) > _MODEL_CACHE_MAX:
        evicted, _ = _loaded.popitem(last=False)
        print(f"[cache] evicted {evicted} (LRU, max={_MODEL_CACHE_MAX})")
    return _loaded[repo]


def _build_transform(image_sz: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_sz),
        transforms.CenterCrop(image_sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@_gpu_decorator(duration=30)
def _infer_on_gpu(repo: str, x: torch.Tensor,
                  geo: torch.Tensor | None) -> torch.Tensor:
    """Runs in a ZeroGPU subprocess. Args go through pickle — pass a
    short string + ~5 MB image tensor + 16-byte geo, not the 1.2 GB
    model. The model is fetched from the module-level cache, which
    ZeroGPU promotes to GPU memory on first call."""
    bundle = _load_from_hub(repo)
    model = bundle["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)
    with torch.no_grad():
        if isinstance(model, _GeoModel):
            g = geo.to(device) if geo is not None else None
            logits = model(x, g)
        else:
            logits = model(x)
    return F.softmax(logits, dim=1).cpu()


def identify(image: Image.Image, model_choice: str,
             lat: float | None, lon: float | None) -> dict[str, float]:
    if image is None:
        return {}
    entry = MODELS.get(model_choice)
    if entry is None:
        return {}
    repo = entry["repo"]
    bundle = _load_from_hub(repo)
    cfg = bundle["config"]
    image_sz = int(cfg.get("image_sz", 640))
    tfm = _build_transform(image_sz)
    x = tfm(image.convert("RGB")).unsqueeze(0)
    geo = _encode_coords(lat, lon) if bundle["use_location"] else None
    probs = _infer_on_gpu(repo, x, geo).squeeze(0)
    topk = torch.topk(probs, k=min(TOPK, probs.numel()))
    nameslist = bundle["nameslist"]
    return {nameslist[i]: float(p) for i, p in zip(topk.indices.tolist(),
                                                    topk.values.tolist())}


def _model_info(model_choice: str) -> str:
    e = MODELS.get(model_choice)
    if not e:
        return "_No model selected._"
    return f"**{model_choice}** — {e['description']}\n\nRepo: `{e['repo']}`"


def _refresh_models():
    """Re-scan the Hub and repopulate the dropdown. Lets a just-published
    family appear without redeploying the Space."""
    global MODELS, DEFAULT_MODEL
    MODELS = discover_models()
    DEFAULT_MODEL = next(iter(MODELS))
    return (gr.update(choices=list(MODELS.keys()), value=DEFAULT_MODEL),
            _model_info(DEFAULT_MODEL))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _image_kwargs() -> dict:
    """Camera-first image input. On a phone the 'upload' source opens the
    OS camera (no mirroring); 'webcam' adds live capture on desktop. Disable
    webcam mirroring so sheet labels aren't flipped — guarded so an older
    Gradio without WebcamOptions still loads."""
    kw = dict(type="pil", label="Specimen photo or scan",
              sources=["upload", "webcam"])
    try:
        kw["webcam_options"] = gr.WebcamOptions(mirror=False)
    except Exception:
        pass
    return kw


with gr.Blocks(title="Herbarium ID") as demo:
    gr.Markdown("# Herbarium specimen identification")
    gr.Markdown(
        "Photograph a herbarium sheet with your phone (or upload an image) "
        "and pick a model — it returns the top-5 predictions with confidence "
        "scores. Tap ⟳ to refresh the model list."
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                # Radio, not Dropdown: a Gradio dropdown's popup panel is
                # unreliable on mobile browsers (it would only ever show the
                # selected option). Radio renders every model as an inline
                # tappable item — no popup, works everywhere.
                model_dd = gr.Radio(
                    choices=list(MODELS.keys()), value=DEFAULT_MODEL,
                    label="Model", interactive=True, scale=5,
                )
                refresh = gr.Button("⟳", scale=1, min_width=48)
            info = gr.Markdown(_model_info(DEFAULT_MODEL))
            img = gr.Image(**_image_kwargs())
            # Location is irrelevant for sheets shot inside the herbarium and
            # only used by geo-aware models, so keep it out of the way.
            with gr.Accordion("Location (optional — geo-aware models only)",
                              open=False):
                with gr.Row():
                    lat_in = gr.Number(label="Latitude (°)", value=None,
                                       minimum=-90, maximum=90)
                    lon_in = gr.Number(label="Longitude (°)", value=None,
                                       minimum=-180, maximum=180)
            run = gr.Button("Identify", variant="primary")
        with gr.Column(scale=1):
            out = gr.Label(num_top_classes=TOPK, label="Top-5 predictions")

    model_dd.change(fn=_model_info, inputs=model_dd, outputs=info)
    refresh.click(fn=_refresh_models, outputs=[model_dd, info])
    run.click(fn=identify, inputs=[img, model_dd, lat_in, lon_in], outputs=out)


# ---------------------------------------------------------------------------
# Pre-load only the default model at import. ZeroGPU's pickle-based arg
# passing makes ferrying 1.2 GB per call infeasible, so the model must live
# in the parent process cache (the GPU subprocess inherits it on fork).
# We warm just the default — the rest load lazily on selection, bounded by
# the LRU above so many published families don't exhaust memory.
# ---------------------------------------------------------------------------
try:
    _load_from_hub(MODELS[DEFAULT_MODEL]["repo"])
    print(f"[startup] preloaded {MODELS[DEFAULT_MODEL]['repo']}")
except Exception as e:
    print(f"[startup] preload of default model failed: {e}")


if __name__ == "__main__":
    demo.launch()
