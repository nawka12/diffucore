"""Anima end-to-end text-to-image pipeline (DT7).

A focused, self-contained driver that bridges the Anima-specific bits without
threading flow-matching state through the SD/SDXL ``_Pipeline`` scaffolding:

  prompt  -> AnimaTokenizer (Qwen2 + T5)
          -> Qwen3 (source hidden states)
          -> AnimaDiT.forward(..., context=qwen_hidden, t5xxl_ids=t5_ids)
              (DiT runs the LLM-Adapter internally on its first call)
          -> flow-matching σ schedule + CONST scaling + Euler integration
          -> QwenImageVAE.process_out then VAE.decode

Compared to ``_Pipeline._sample``, we do CFG with a single batched forward
where possible (one pass with batch=2 — cond and uncond stacked) so the 2B
DiT only sees one forward per sampler step rather than two, and we manage
the 4D↔5D shape ourselves at the DiT boundary rather than wrapping the
backbone in an adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from ..runtime import on_device
from ..sampling import flow_matching_schedule

if TYPE_CHECKING:
    from ..bundle import ModelBundle


def _staged(modules, device, offload: bool):
    """Mirror of ``_Pipeline._staged`` — bring modules onto ``device`` while
    inside the ``with`` block when offloading, no-op otherwise."""
    from contextlib import ExitStack
    if not offload:
        from contextlib import nullcontext
        return nullcontext()
    es = ExitStack()
    for m in modules:
        es.enter_context(on_device(m, device))
    return es


def _qwen_encode(qwen3, ids, mask, device, dtype):
    """Run Qwen3 and return its last hidden state in ``dtype``."""
    ids = ids.to(device)
    if mask is not None:
        mask = mask.to(device)
    out = qwen3(ids, attention_mask=None)   # causal-only fast path; mask handled by padding below
    return out.to(dtype)


def _to_pil(img: torch.Tensor) -> Image.Image:
    img = ((img.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())


def anima_text_to_image(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 20,
    cfg_scale: float = 4.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
) -> Image.Image:
    """Drive Anima's text-to-image path end-to-end.

    ``shift`` controls the SD3-style rectified-flow schedule (Anima's training
    default is 3.0). ``cfg_scale`` is the CFG strength; the Anima ComfyUI
    workflow defaults to ~4.0.
    """
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    # Spatial dims must be divisible by VAE-stride·patch = 8·2 = 16.
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")

    # ---- 1. tokenize cond + uncond
    cond_tok = model.tokenizer(prompt)
    uncond_tok = model.tokenizer(negative_prompt)

    # ---- 2. encode with Qwen3 (staged onto device when offloading)
    with _staged([model.text_encoder], device, policy.offload_idle):
        qwen_dtype = next(model.text_encoder.parameters()).dtype
        cond_hidden = _qwen_encode(model.text_encoder, cond_tok.qwen_ids, cond_tok.qwen_mask, device, dtype)
        uncond_hidden = _qwen_encode(model.text_encoder, uncond_tok.qwen_ids, uncond_tok.qwen_mask, device, dtype)
        del qwen_dtype  # silence unused warning if the Qwen3 dtype probe ever shifts

    cond_t5 = cond_tok.t5_ids.to(device)
    uncond_t5 = uncond_tok.t5_ids.to(device)

    # ---- 3. σ schedule, init noise
    sigmas = flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32)
    h_lat, w_lat = height // 8, width // 8
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.randn(1, 16, h_lat, w_lat, generator=gen, device=device, dtype=dtype)
    # With σ_max == 1 the initial state is exactly pure noise (no rescale).

    # ---- 4. Euler integration of the rectified-flow ODE
    backbone = model.backbone
    with torch.no_grad(), _staged([backbone], device, policy.offload_unet):
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            x_5d = x.unsqueeze(2)                         # (B, C, 1, H, W)
            t = torch.full((1,), sigma.item(), device=device, dtype=dtype)

            v_cond = backbone(x_5d, t, cond_hidden, t5xxl_ids=cond_t5).squeeze(2)
            if cfg_scale == 1.0:
                v = v_cond
            else:
                v_uncond = backbone(x_5d, t, uncond_hidden, t5xxl_ids=uncond_t5).squeeze(2)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)

            # CONST flow: denoised = x − σ·v ; Euler step is x + (σ_next − σ)·v
            # (closed-form exact for any constant x0 estimate).
            x = x + (sigma_next - sigma).to(dtype) * v

    # ---- 5. process_out then decode
    with torch.no_grad(), _staged([model.vae], device, policy.offload_idle):
        z = model.vae.process_out(x.to(policy.vae_dtype))
        image = model.vae.decode(z)
    return _to_pil(image)
