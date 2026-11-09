"""JEM ViT-B/16 backbone (self-supervised on Pl@ntNet-300K-v2) as a timm model.

https://huggingface.co/saulane/jem-vitb16-plantnet300kv2 ships its own ViT
class rather than a timm one, but the architecture is timm's DINOv3 Eva with
three changes: no class token (4 registers only), a full qkv bias, and either
no LayerScale (the "nols" release, better per the model card) or LayerScale
1e-5. Same axial RoPE (coords normalised to [-1, 1], base 100, rotate-half),
same ImageNet mean/std, any input size divisible by 16. The converted model
reproduces upstream's patch tokens exactly (max |diff| 0.0 at 224 and 640x448).

Importing this module registers two names with timm:

    vit_base_patch16_jem_plantnet     no LayerScale (recommended)
    vit_base_patch16_jem_ls_plantnet  LayerScale 1e-5

so --model / timm.create_model work as for any other backbone. Pooling is the
mean of patch tokens after the final norm (timm global_pool='avg' skips the
registers), which is what JEM's linear probe used.

timm's Eva attention keeps k_bias as a zero, non-persistent buffer (EVA drops
it because without RoPE a key bias cancels in softmax). Under RoPE it doesn't
cancel: zeroing JEM's k bias moves the pooled feature ~22%. So it is promoted
to a Parameter here, which also puts it in state_dict and hence in our
checkpoints; identify rebuilds with pretrained=False and gets it back from there.
"""

import torch
import torch.nn as nn
from timm.models import create_model, register_model

HF_REPO = "saulane/jem-vitb16-plantnet300kv2"
WEIGHTS = {
    False: "jem_nols_vitb16_plantnet300kv2_ep100.safetensors",
    True:  "jem_vitb16_plantnet300kv2_ep100.safetensors",
}


def _convert(sd: dict, model: nn.Module) -> dict:
    """Map upstream JEM key names onto timm Eva's."""
    out = {}
    for k, v in sd.items():
        if k == "mask_token":           # pretraining-only
            continue
        if k == "registers":
            out["reg_token"] = v
        elif k.startswith("patch_embed."):
            out["patch_embed.proj." + k.split(".", 1)[1]] = v
        elif k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            p = f"blocks.{i}."
            if rest == "qkv.weight":
                out[p + "attn.qkv.weight"] = v
            elif rest == "qkv.bias":
                q, kb, vb = v.chunk(3)
                out[p + "attn.q_bias"], out[p + "attn.k_bias"], out[p + "attn.v_bias"] = q, kb, vb
            elif rest.startswith("proj."):
                out[p + "attn." + rest] = v
            elif rest.startswith(("fc1.", "fc2.")):
                out[p + "mlp." + rest] = v
            elif rest in ("ls1", "ls2"):
                out[p + "gamma_" + rest[-1]] = v
            else:                       # norm1 / norm2
                out[k] = v
        else:                           # final norm
            out[k] = v
    if "blocks.0.ls1" not in sd:
        # timm drops init_values=None in create_model, so the blocks always
        # have gammas; ones make them the identity, i.e. no LayerScale.
        for i, blk in enumerate(model.blocks):
            out[f"blocks.{i}.gamma_1"] = torch.ones_like(blk.gamma_1)
            out[f"blocks.{i}.gamma_2"] = torch.ones_like(blk.gamma_2)
    return out


def _create_jem(layerscale: bool, pretrained: bool, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    model = create_model("vit_base_patch16_dinov3", pretrained=False,
                         class_token=False, qkv_bias=True, **kwargs)
    for blk in model.blocks:
        attn = blk.attn
        kb = attn.k_bias.detach().clone()
        del attn._buffers["k_bias"]
        attn.k_bias = nn.Parameter(kb)
    if pretrained:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = _convert(load_file(hf_hub_download(HF_REPO, WEIGHTS[layerscale])), model)
        # strict: a silent partial load would train from random init.
        # The classifier head is ours, so it is the one thing allowed missing.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(f"JEM weight conversion mismatch: missing={missing} "
                               f"unexpected={unexpected}")
    return model


@register_model
def vit_base_patch16_jem_plantnet(pretrained: bool = False, **kwargs):
    return _create_jem(False, pretrained, **kwargs)


@register_model
def vit_base_patch16_jem_ls_plantnet(pretrained: bool = False, **kwargs):
    return _create_jem(True, pretrained, **kwargs)
