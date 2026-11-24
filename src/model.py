"""BiomedCLIP loading, embedding helpers and LoRA injection."""
import json
import math
from pathlib import Path

import open_clip
import torch
import torch.nn as nn

from . import config as C


def load_clip(device="cuda"):
    """Load BiomedCLIP fully offline from models/BiomedCLIP."""
    name = f"local-dir:{C.CLIP_DIR}"
    text_cfg = json.loads((Path(C.CLIP_DIR) / "open_clip_config.json").read_text("utf-8"))["model_cfg"]["text_cfg"]
    text_cfg.update(hf_model_name=str(C.CLIP_DIR), hf_tokenizer_name=str(C.CLIP_DIR), hf_model_pretrained=False)
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, device=device, precision="fp32", text_cfg=text_cfg)
    model.eval()
    tokenizer = open_clip.get_tokenizer(name)
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_images(model, x, device="cuda", batch=256):
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            f = model.encode_image(x[i:i + batch].to(device, non_blocking=True))
        out.append(f.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device="cuda"):
    tokens = tokenizer(texts).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        f = model.encode_text(tokens)
    return f.float().cpu()


class LoRALinear(nn.Module):
    """Wraps a frozen Linear with a rank-r update B @ A."""

    def __init__(self, base, r=16, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def blocks(model):
    """Transformer blocks of the ViT trunk (BiomedCLIP uses a timm tower)."""
    trunk = getattr(model.visual, "trunk", model.visual)
    return trunk.blocks


def inject_lora(model, r=16, alpha=16):
    """Patch attention projections of the visual transformer in place."""
    n = 0
    for block in blocks(model):
        attn = block.attn
        for name in ("qkv", "proj"):
            layer = getattr(attn, name)
            if isinstance(layer, LoRALinear):
                continue
            setattr(attn, name, LoRALinear(layer, r, alpha))
            n += 1
    return n


def load_lora(model, path, r=16):
    """Restore LoRA + head weights saved by src.train."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state"]
    if not hasattr(model, "head"):
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {path}: {unexpected[:3]}")
    return ck


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)
