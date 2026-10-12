"""Krea 2 GPU parity against the diffusers golden bundle (manual, not collected).

    python tests/_krea2_parity.py te [TE_FILE]
    python tests/_krea2_parity.py dit DIT_FILE
    python tests/_krea2_parity.py e2e DIT_FILE TAG [--sampler euler]

``dit`` runs the reference's single forwards (golden inputs, golden text
features) in fp16: a bf16 file is read one block at a time from its mmap, so the
26 GB file needs neither 26 GB of RAM nor of VRAM; an INT8 file streams.
``e2e`` samples a golden run from its own initial noise and compares the PNG.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffucore.bundle import (ModelBundle, _build, _extract_component, _KREA2_DIT_LEAF,
                              _krea2_text_encoder, _place, load_krea2_dit)
from diffucore.conditioning import Krea2Tokenizer
from diffucore.conditioning.krea2_tokenizer import PREFIX_TOKENS, SELECT_LAYERS
from diffucore.loading import detect_architecture, load_state_dict, read_header
from diffucore.models import Krea2Config, Krea2DiT, QwenImageVAE
from diffucore.models._int8 import adopt_int8_layers, int8_layers
from diffucore.pipelines._krea2 import krea2_text_to_image
from diffucore.runtime import ConditioningCache, DevicePolicy

GOLDEN = Path(os.environ.get(
    "KREA2_GOLDEN",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/krea2-golden-20261012/golden.safetensors"))
MODELS = Path(__file__).resolve().parents[2] / "models"
DEV = torch.device("cuda")


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return float((a - b).norm() / b.norm())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().cpu().flatten(), b.float().cpu().flatten(), dim=0))


def unpack(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    B, _, d = tokens.shape
    return tokens.view(B, h // 2, w // 2, d // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, d // 4, h, w)


def pack(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)


def golden():
    return load_file(str(GOLDEN)), json.loads((GOLDEN.parent / "meta.json").read_text())


def stage_te(te_path: str):
    g, meta = golden()
    te, _ = _krea2_text_encoder(load_state_dict(te_path, device="cpu"))
    te = te.to(DEV, torch.float16).eval()
    tok = Krea2Tokenizer()
    for name in ("anime", "photo", "empty"):
        ids = tok(meta["prompts"][name]).ids.to(DEV)
        with torch.no_grad():
            feats = torch.stack(te(ids, hidden_layers=list(SELECT_LAYERS)), dim=2)[0, PREFIX_TOKENS:]
        want = g[f"te.{name}.features_valid"]
        per_layer = [rel_l2(feats[:, i], want[:, i]) for i in range(want.shape[1])]
        print(f"[te] {name:6s} L={feats.shape[0]:3d}  rel_l2 max {max(per_layer):.4f} "
              f"(layer {SELECT_LAYERS[per_layer.index(max(per_layer))]}), all {rel_l2(feats, want):.4f}, "
              f"cos {cos(feats, want):.6f}, max|x| {feats.abs().max().item():.0f}")


def _blockwise(dit_path: str) -> Krea2DiT:
    """Non-block modules on the GPU in fp16; each block copied up from the mmap
    for its own forward, then pointed back at the mmap (bf16 or INT8 files)."""
    sd = _extract_component(load_state_dict(dit_path, device="cpu"), _KREA2_DIT_LEAF)
    model = _build(Krea2DiT, Krea2Config.from_state_dict(sd))
    layers = int8_layers(sd)
    if layers:
        sd = adopt_int8_layers(model, sd, layers)
    model.load_state_dict(sd, strict=True, assign=True)
    for name, child in model.named_children():
        if name != "blocks":
            child.to(DEV, torch.float16)
    def up(m, args):
        m.to(DEV, torch.float16)

    def back(m, args, out, src):
        m.load_state_dict(src, assign=True)

    for blk in model.blocks:
        src = dict(blk.state_dict())
        blk.register_forward_pre_hook(up)
        blk.register_forward_hook(lambda m, a, o, src=src: back(m, a, o, src))
    model.residual_scale = 0.25
    return model.eval()


def load_dit(dit_path: str, variant: str):
    """Disk-backed blocks by default; ``KREA2_STREAM=1`` uses the real loader's
    pinned-RAM block streaming instead (needs the file's size in free RAM)."""
    if os.environ.get("KREA2_STREAM") == "1":
        policy = DevicePolicy(device=DEV, compute_dtype=torch.float16, offload="stream", stream_prefetch=True)
        return load_krea2_dit(dit_path, policy, variant=variant), policy
    return _blockwise(dit_path), DevicePolicy(device=DEV, compute_dtype=torch.float16)


def stage_dit(dit_path: str):
    g, meta = golden()
    variant = "turbo" if "turbo" in Path(dit_path).name.lower() else "raw"
    model, _ = load_dit(dit_path, variant)
    ctx = {}
    with torch.inference_mode():
        for name in ("anime", "photo", "empty"):
            ctx[name] = model.encode_text(g[f"te.{name}.features_valid"][None].to(DEV))
        for tag, run in meta["runs"].items():
            if not tag.startswith(variant):
                continue
            h, w = run["h"] // 8, run["w"] // 8
            for key in sorted(k for k in g if k.startswith(tag + ".step") and k.endswith(".out")):
                stem = key[: -len(".out")]
                branch = stem.rsplit(".", 1)[1]
                x = unpack(g[stem + ".hidden_states"].float(), h, w).to(DEV)
                t = g[stem + ".timestep"].to(DEV)
                c = ctx[run["prompt"] if branch == "cond" else "empty"]
                torch.cuda.synchronize()
                t0 = time.time()
                out = pack(model(x, t, c).float()).cpu()
                torch.cuda.synchronize()
                want = g[key].float()
                print(f"[dit] {stem:40s} t={t.item():.4f} rel_l2 {rel_l2(out, want):.4f} "
                      f"cos {cos(out, want):.6f}  finite {bool(torch.isfinite(out).all())}  "
                      f"{time.time() - t0:.2f}s", flush=True)


def stage_e2e(dit_path: str, tag: str, sampler: str = "euler"):
    g, meta = golden()
    run = meta["runs"][tag]
    variant = "turbo" if tag.startswith("turbo") else "raw"
    model, policy = load_dit(dit_path, variant)
    vae = _build(QwenImageVAE)
    vae_sd = load_state_dict(str(MODELS / "vae" / "qwen_image_vae.safetensors"), device="cpu")
    vae.load_state_dict(vae_sd, strict=True, assign=True)
    vae = _place(vae, DEV, policy.vae_dtype, vae_sd).eval()
    prompt = meta["prompts"][run["prompt"]]
    cache = ConditioningCache()
    with torch.inference_mode():
        cache.put(("krea2", prompt, ""), {
            "cond_ctx": model.encode_text(g[f"te.{run['prompt']}.features_valid"][None].to(DEV)).cpu(),
            "uncond_ctx": model.encode_text(g["te.empty.features_valid"][None].to(DEV)).cpu()})
    spec = replace(detect_architecture(read_header(dit_path)), variant=variant)
    bundle = ModelBundle(spec=spec, schedule=None, tokenizer=Krea2Tokenizer(), text_encoder=None,
                         backbone=model, vae=vae, policy=policy, cond_cache=cache)
    noise = unpack(g[f"{tag}.noise"].float(), run["h"] // 8, run["w"] // 8)
    torch.cuda.synchronize()
    t0 = time.time()
    img = krea2_text_to_image(bundle, prompt, "", steps=run["steps"], cfg_scale=run["guidance"] + 1.0,
                              width=run["w"], height=run["h"], sampler=sampler, scheduler="krea2",
                              noise=noise)
    dt = time.time() - t0
    out = Path(os.environ.get("KREA2_OUT", "/tmp")) / f"ours_{tag}_{sampler}.png"
    img.save(out)
    a = np.asarray(img, dtype=np.float64)
    b = np.asarray(Image.open(GOLDEN.parent / f"{tag}.png").convert("RGB"), dtype=np.float64)
    rmse = math.sqrt(((a - b) ** 2).mean())
    print(f"[e2e] {tag} {sampler}: {dt:.1f}s  RMSE {rmse:.2f}/255  PSNR {20 * math.log10(255 / rmse):.1f} dB"
          f"  -> {out}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "te":
        stage_te(sys.argv[2] if len(sys.argv) > 2 else str(MODELS / "text-encoders" / "qwen3vl_4b_bf16.safetensors"))
    elif cmd == "dit":
        stage_dit(sys.argv[2])
    elif cmd == "e2e":
        stage_e2e(sys.argv[2], sys.argv[3], sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--sampler" else "euler")
    else:
        raise SystemExit(__doc__)
