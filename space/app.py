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

class _HerbModel(nn.Module):
    """Backbone (+ optional geo MLP) feeding one or two heads.

    Mirrors TimmModelHierarchical in train_herbarium.py: both heads read the
    same fused representation, so predicting genus as well as species costs one
    extra 1024×N matmul — the backbone pass is shared. That is why the Space
    returns both instead of making the user pick a rank up front.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module,
                 geo_mlp: nn.Module | None = None,
                 genus_head: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.geo_mlp = geo_mlp
        self.genus_head = genus_head

    def forward(self, x: torch.Tensor, geo: torch.Tensor | None = None):
        z = self.backbone(x)
        if self.geo_mlp is not None:
            if geo is None:
                geo = torch.zeros(z.shape[0], 4, device=z.device)
            z = torch.cat([z, self.geo_mlp(geo)], dim=1)
        genus_logits = self.genus_head(z) if self.genus_head is not None else None
        return self.head(z), genus_logits


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
    """Remove Lightning + torch.compile prefixes, matching identify_herbarium.py.

    TimmModel (non-geo) stores timm as self.model, so checkpoint keys are
    model.model.* — the double prefix must be stripped before the single one.
    TimmModelHierarchical uses self.backbone, giving model.backbone.* keys.
    on_save_checkpoint normalises _orig_mod away but we keep the fallback.
    """
    out: dict = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for p in ("model._orig_mod.model.", "model._orig_mod.",
                  "model.model.",            "model."):
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out


def _optional_download(repo: str, filename: str) -> str | None:
    try:
        return hf_hub_download(repo_id=repo, filename=filename)
    except Exception:
        return None


def _linear_from(state: dict, prefix: str) -> nn.Linear | None:
    """Rebuild an nn.Linear from state_dict keys under `prefix`.

    The shape is read off the saved weight rather than assumed, which is what
    makes this safe: a hierarchical head's input width depends on whether geo
    features were fused, and its attribute name in timm varies by architecture
    (`head` on ViT, `fc` on ResNet, `classifier` on EfficientNet). Reconstructing
    from the weight sidesteps both problems.
    """
    sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if "weight" not in sd:
        return None
    out_dim, in_dim = sd["weight"].shape
    lin = nn.Linear(in_dim, out_dim, bias="bias" in sd)
    lin.load_state_dict(sd)
    return lin


def _build_genus_head(state: dict, genus_names: list[str]) -> nn.Module | None:
    head = _linear_from(state, "head_genus.")
    if head is None:
        return None
    if genus_names and head.out_features != len(genus_names):
        print(f"[load] genus head has {head.out_features} outputs but "
              f"genus_nameslist has {len(genus_names)} names — skipping genus head")
        return None
    return head


def _load_from_hub(repo: str) -> dict[str, Any]:
    if repo in _loaded:
        _loaded.move_to_end(repo)  # mark most-recently-used
        return _loaded[repo]
    ckpt_path = hf_hub_download(repo_id=repo, filename="model.ckpt")
    names_path = hf_hub_download(repo_id=repo, filename="nameslist.json")
    cfg_path = hf_hub_download(repo_id=repo, filename="config.json")
    # Only hierarchical models publish this.
    genus_names_path = _optional_download(repo, "genus_nameslist.json")

    with open(names_path) as f:
        nameslist: list[str] = json.load(f)
    with open(cfg_path) as f:
        config: dict[str, Any] = json.load(f)
    genus_nameslist: list[str] = []
    if genus_names_path:
        with open(genus_names_path) as f:
            genus_nameslist = json.load(f)

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
        # TimmModelHierarchical names the primary head head_species regardless
        # of label_level; fall back to bare head.* for non-hierarchical models.
        hd_sd = {k[len("head_species."):]: v for k, v in state.items()
                 if k.startswith("head_species.")}
        if not hd_sd:
            hd_sd = {k[len("head."):]: v for k, v in state.items()
                     if k.startswith("head.")}
        bb_missing, bb_unexp = backbone.load_state_dict(bb_sd, strict=False)
        gm_missing, gm_unexp = geo_mlp.load_state_dict(gm_sd, strict=False)
        hd_missing, hd_unexp = head.load_state_dict(hd_sd, strict=False)
        print(f"[load] geo: backbone missing={len(bb_missing)} unexp={len(bb_unexp)}; "
              f"geo_mlp missing={len(gm_missing)} unexp={len(gm_unexp)}; "
              f"head missing={len(hd_missing)} unexp={len(hd_unexp)}")
        genus_head = _build_genus_head(state, genus_nameslist)
        model = _HerbModel(backbone, head, geo_mlp=geo_mlp, genus_head=genus_head)
    elif any(k.startswith("backbone.") for k in state):
        # Hierarchical, no geo (TimmModelHierarchical): backbone.* +
        # head_species.* (+ head_genus/head_family). Build the backbone headless
        # and reconstruct each head from its saved weight. The previous code
        # renamed head_species.* to head.* and pushed it through the timm model,
        # which only works when timm happens to call its classifier `head` (ViT).
        # On a ResNet (`fc`) or EfficientNet (`classifier`) those keys were
        # silently unexpected and the species head stayed randomly initialised.
        backbone = timm.create_model(config["model_name"], pretrained=False,
                                     num_classes=0)
        bb_sd = {k[len("backbone."):]: v for k, v in state.items()
                 if k.startswith("backbone.")}
        bb_missing, bb_unexp = backbone.load_state_dict(bb_sd, strict=False)
        head = _linear_from(state, "head_species.")
        if head is None:
            raise RuntimeError("hierarchical checkpoint has no head_species.* weights")
        genus_head = _build_genus_head(state, genus_nameslist)
        print(f"[load] hier: backbone missing={len(bb_missing)} unexp={len(bb_unexp)}; "
              f"head {head.in_features}→{head.out_features}")
        model = _HerbModel(backbone, head, genus_head=genus_head)
    else:
        # Legacy flat model (TimmModel): bare timm keys after stripping
        # model.model. Keep timm's own head, then detach it so _HerbModel can
        # apply it explicitly. reset_classifier(0) handles the per-architecture
        # attribute name for us.
        timm_model = timm.create_model(config["model_name"], pretrained=False,
                                       num_classes=num_classes)
        missing, unexp = timm_model.load_state_dict(state, strict=False)
        print(f"[load] plain: missing={len(missing)} unexp={len(unexp)}")
        head = timm_model.get_classifier()
        timm_model.reset_classifier(0)
        model = _HerbModel(timm_model, head, genus_head=None)

    if model.genus_head is None:
        genus_nameslist = []
    print(f"[load] {repo}: {num_classes} species"
          + (f" + {len(genus_nameslist)} genera (trained genus head)"
             if genus_nameslist else " (no genus head)"))

    model.eval()
    _loaded[repo] = {
        "model": model,
        "nameslist": nameslist,
        "genus_nameslist": genus_nameslist,
        "config": config,
        "use_location": use_location,
        "geo_dim": geo_dim,
        # Taxa dropped as too sparse at train time — embedded in the checkpoint.
        "excluded": ckpt.get("excluded_species") or {},
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
                  geo: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Runs in a ZeroGPU subprocess. Args go through pickle — pass a
    short string + ~5 MB image tensor + 16-byte geo, not the 1.2 GB
    model. The model is fetched from the module-level cache, which
    ZeroGPU promotes to GPU memory on first call.

    Returns (species_probs, genus_probs); genus_probs is None for models
    without a trained genus head.
    """
    bundle = _load_from_hub(repo)
    model = bundle["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)
    with torch.no_grad():
        g = geo.to(device) if geo is not None else None
        logits, genus_logits = model(x, g)
    # Temperature scaling (Guo et al. 2017): divide logits by a calibration
    # temperature before softmax so confidences aren't pinned near 100%.
    # T is per-model, stored in config.json; absent/invalid → 1.0 (unchanged).
    try:
        temperature = float(bundle["config"].get("temperature", 1.0)) or 1.0
    except (TypeError, ValueError):
        temperature = 1.0
    if temperature != 1.0:
        logits = logits / temperature
        if genus_logits is not None:
            genus_logits = genus_logits / temperature
    genus_probs = (F.softmax(genus_logits, dim=1).cpu()
                   if genus_logits is not None else None)
    return F.softmax(logits, dim=1).cpu(), genus_probs


def _excluded_md(excluded: dict) -> str:
    """One-line notice of taxa the model can't predict (dropped as too sparse)."""
    taxa = (excluded or {}).get("taxa", {}) or {}
    if not taxa:
        return ""
    rank = (excluded or {}).get("rank", "species")
    names = sorted(taxa, key=lambda n: taxa[n])   # rarest first
    shown = ", ".join(f"*{n}*" for n in names[:15])
    more  = f" +{len(names) - 15} more" if len(names) > 15 else ""
    return (f"⚠️ **{len(taxa)} {rank} are not in this model** — too few training "
            f"images, so specimens of these get forced to the nearest trained "
            f"class:\n\n{shown}{more}")


def _topk_dict(probs: torch.Tensor, names: list[str]) -> dict[str, float]:
    topk = torch.topk(probs, k=min(TOPK, probs.numel()))
    return {names[i]: float(p)
            for i, p in zip(topk.indices.tolist(), topk.values.tolist())
            if i < len(names)}


# Novelty thresholds on the *calibrated* top-1 confidence. On the African
# Rubiaceae run, held-out (in-distribution) species confidence averaged 0.887,
# vs 0.51 for a novel species and 0.40 for a novel genus — so a species cut
# near 0.6 cleanly separates "known" from "novel", and a genus cut near 0.5
# does the same at genus rank (see docs/novelty_and_mislabel_detection.md §3-4).
# Overridable per model via config.json so a future push can inject values
# fitted on that model's own held-out split.
_OOD_SPECIES_THR_DEFAULT = 0.60
_OOD_GENUS_THR_DEFAULT = 0.50


def _novelty_md(species_conf: float, genus_conf: float | None,
                top_genus: str | None, cfg: dict[str, Any]) -> str:
    """A plain-language novelty verdict from the two free signals the write-up
    validates (§4.3): max-softmax and the trained genus head's confidence.

    The logic mirrors the paper's set-A / set-B split. A new *Psychotria* still
    IS a *Psychotria*: the genus head is right to be confident, and only the
    species head has nowhere to put it. So high-genus/low-species reads as a
    likely *novel species in a known genus*, while low genus confidence reads
    as a genuinely out-of-distribution taxon.
    """
    sp_thr = float(cfg.get("ood_species_conf_thr", _OOD_SPECIES_THR_DEFAULT))
    ge_thr = float(cfg.get("ood_genus_conf_thr", _OOD_GENUS_THR_DEFAULT))

    if genus_conf is not None and top_genus:
        if genus_conf < ge_thr:
            return (
                f"### 🔶 Possibly out-of-distribution\n"
                f"Even the **genus** is uncertain (top genus *{top_genus}*, "
                f"{genus_conf:.0%}). This may be a taxon — or a family — the model "
                f"has never seen. Treat every prediction here with caution."
            )
        if species_conf < sp_thr:
            return (
                f"### 🟡 Possibly a novel species\n"
                f"The **genus is recognised** (*{top_genus}*, {genus_conf:.0%}), but "
                f"species confidence is low ({species_conf:.0%}) — this may be a "
                f"species not represented in the model. **The genus is the more "
                f"trustworthy answer.**"
            )
        return (
            f"### ✅ Recognised taxon\n"
            f"Confident at both ranks (genus *{top_genus}* {genus_conf:.0%}, "
            f"species {species_conf:.0%}). Consistent with a taxon the model was "
            f"trained on."
        )

    # No genus head — fall back to species max-softmax alone.
    if species_conf < sp_thr:
        return (
            f"### 🟡 Low confidence ({species_conf:.0%})\n"
            f"This may be a taxon the model has not seen — the top species is a "
            f"best guess, not a recognition."
        )
    return (f"### ✅ Recognised ({species_conf:.0%})\n"
            f"Consistent with a taxon the model was trained on.")


def identify(image: Image.Image, model_choice: str,
             lat: float | None, lon: float | None):
    """Returns (species preds, genus panel, novelty verdict, excluded notice).

    Both ranks come from one backbone pass, so there is no reason to make the
    user choose: a specimen can be a confident genus and an uncertain species,
    and seeing both at once is the whole point. The genus panel stays hidden
    for models without a genus head.

    The novelty verdict answers the question a softmax classifier otherwise
    hides — "is this even a taxon I know?" — since it must file every input
    under some trained class, however alien.
    """
    blank = gr.update(visible=False)
    if image is None:
        return {}, blank, blank, ""
    entry = MODELS.get(model_choice)
    if entry is None:
        return {}, blank, blank, ""
    repo = entry["repo"]
    bundle = _load_from_hub(repo)
    cfg = bundle["config"]
    image_sz = int(cfg.get("image_sz", 640))
    tfm = _build_transform(image_sz)
    x = tfm(image.convert("RGB")).unsqueeze(0)
    geo = _encode_coords(lat, lon) if bundle["use_location"] else None
    probs, genus_probs = _infer_on_gpu(repo, x, geo)
    preds = _topk_dict(probs.squeeze(0), bundle["nameslist"])
    species_conf = float(probs.squeeze(0).max())

    notice = _excluded_md(bundle.get("excluded") or {})
    genus_names = bundle.get("genus_nameslist") or []
    if genus_probs is None or not genus_names:
        novelty = _novelty_md(species_conf, None, None, cfg)
        return preds, blank, gr.update(value=novelty, visible=True), notice
    genus_preds = _topk_dict(genus_probs.squeeze(0), genus_names)
    top_genus, genus_conf = next(iter(genus_preds.items()))
    novelty = _novelty_md(species_conf, genus_conf, top_genus, cfg)
    return (preds, gr.update(value=genus_preds, visible=True),
            gr.update(value=novelty, visible=True), notice)


def _model_info(model_choice: str) -> str:
    e = MODELS.get(model_choice)
    if not e:
        return "_No model selected._"
    return f"**{model_choice}** — {e['description']}\n\nRepo: `{e['repo']}`"


# Show the filter box only once the list gets long enough to be awkward.
_PICKER_SEARCH_THRESHOLD = 6


def _refresh_models():
    """Re-scan the Hub and repopulate the picker. Lets a just-published
    family appear without redeploying the Space."""
    global MODELS, DEFAULT_MODEL
    MODELS = discover_models()
    DEFAULT_MODEL = next(iter(MODELS))
    return (gr.update(choices=list(MODELS), value=DEFAULT_MODEL),
            _model_info(DEFAULT_MODEL),
            gr.update(value="", visible=len(MODELS) > _PICKER_SEARCH_THRESHOLD))


def _filter_models(query: str, current: str):
    """Filter the model radio by a case-insensitive substring. The current
    selection is always kept selectable so changing the filter never drops
    it."""
    q = (query or "").strip().lower()
    keys = [k for k in MODELS if q in k.lower()] if q else list(MODELS)
    if current in MODELS and current not in keys:
        keys = [current, *keys]
    value = current if current in keys else (keys[0] if keys else None)
    return gr.update(choices=keys, value=value)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Force mobile file inputs to open the rear camera directly instead of a
# file/gallery chooser. Browsers read capture="environment" when the input
# is tapped; the MutationObserver re-applies it to inputs Gradio creates
# lazily (e.g. after clearing an image). Runs on page load via Blocks(js=).
_CAPTURE_JS = """
() => {
  const apply = () => document.querySelectorAll('input[type=file]')
      .forEach(el => el.setAttribute('capture', 'environment'));
  apply();
  new MutationObserver(apply).observe(document.body, {childList: true, subtree: true});
}
"""


def _image_kwargs() -> dict:
    """Camera-first image input. Use only the 'upload' source: on a phone it
    invokes the native camera app (rear lens, no selfie mirroring) plus the
    photo library. The in-browser 'webcam' source defaults to the front
    camera and mirrors the frame — wrong for photographing a sheet."""
    return dict(type="pil", label="Specimen photo (tap to use your camera)",
                sources=["upload"])


with gr.Blocks(title="Herbarium ID", js=_CAPTURE_JS) as demo:
    gr.Markdown("# Herbarium specimen identification")
    gr.Markdown(
        "Photograph a herbarium sheet with your phone (or upload an image) "
        "and pick a model — it returns the top-5 species, and, where the model "
        "has a genus head, the top-5 genera predicted directly. Genus is "
        "usually the more reliable of the two. Tap ⟳ to refresh the model list."
    )
    with gr.Row():
        with gr.Column(scale=1):
            search = gr.Textbox(
                placeholder="Filter models…", show_label=False, container=False,
                visible=len(MODELS) > _PICKER_SEARCH_THRESHOLD,
            )
            with gr.Row():
                # Radio, not Dropdown: a Gradio dropdown's popup panel is
                # unreliable on mobile browsers (it would only ever show the
                # selected option). Radio renders every model as an inline
                # tappable item — no popup, works everywhere. The filter box
                # above keeps it manageable once there are many models.
                model_dd = gr.Radio(
                    choices=list(MODELS), value=DEFAULT_MODEL,
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
            # Novelty verdict first: whether to trust the lists below at all is
            # the thing to read before the ranked names themselves. Hidden until
            # an identification produces one.
            novelty_out = gr.Markdown(visible=False)
            out = gr.Label(num_top_classes=TOPK, label="Species — top 5")
            # Hidden until a model with a trained genus head produces a result.
            # Genus is the model's most reliable answer, so it is shown beside
            # the species list rather than behind a rank selector.
            genus_out = gr.Label(num_top_classes=TOPK,
                                 label="Genus — top 5 (predicted directly)",
                                 visible=False)
            excluded_note = gr.Markdown("")

    model_dd.change(fn=_model_info, inputs=model_dd, outputs=info)
    search.change(fn=_filter_models, inputs=[search, model_dd], outputs=model_dd)
    refresh.click(fn=_refresh_models, outputs=[model_dd, info, search])
    run.click(fn=identify, inputs=[img, model_dd, lat_in, lon_in],
              outputs=[out, genus_out, novelty_out, excluded_note])


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
