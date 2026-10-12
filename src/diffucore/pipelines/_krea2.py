"""Krea 2 text-to-image (Raw and Turbo).

prompt -> Krea2Tokenizer -> Qwen3-VL-4B hidden layers 2..35 -> DiT text fusion
(once per generation) -> Krea2DiT each step on a flow-matching σ schedule ->
QwenImageVAE decode.

Raw and Turbo are the same network; the variant only picks defaults. The
``krea2`` scheduler is each checkpoint's own shift: Raw interpolates mu in the
image token count from (256, 0.5) to (6400, 1.15), Turbo was distilled at a
fixed mu = 1.15. Krea's guidance g (``cond + g·(cond − uncond)``) is the usual
CFG scale g + 1, which is what ``cfg_scale`` takes here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch

from ..conditioning.krea2_tokenizer import PREFIX_TOKENS, SELECT_LAYERS
from ..runtime import fp32_accumulation, perf_context, staged, vae_decode_safe
from ..sampling import (
    flow_matching_dynamic_shift,
    flow_matching_schedule,
    flow_table_schedule,
    get_sampler,
    guidance_interval_bounds,
)
from ._anima import _ANIMA_SAMPLERS, _FLOW_AWARE_SAMPLERS, _to_pil
from ._base import PipelineInfo, _step_progress

if TYPE_CHECKING:
    from ..bundle import ModelBundle

_KREA2_SAMPLERS = _ANIMA_SAMPLERS
_KREA2_SCHEDULERS = (
    "krea2", "flow", "flow_dyn", "sgm_uniform", "simple", "normal", "infinity", "infinity_htds",
    "kl_optimal", "linear_quadratic", "smoothstep", "beta", "beta_mix", "pump_dual", "pump_taper",
    "secant_tilt", "relay_tilt",
)
KREA2_DEFAULTS = {"raw": {"steps": 28, "cfg_scale": 5.5}, "turbo": {"steps": 8, "cfg_scale": 1.0}}
_TURBO_MU = 1.15


def krea2_shift(variant: str, width: int, height: int) -> float:
    """The checkpoint's own flow shift (``e^mu``) at this resolution."""
    if variant == "turbo":
        return math.exp(_TURBO_MU)
    return flow_matching_dynamic_shift((height // 16) * (width // 16), base_seq_len=256, max_seq_len=6400)


def krea2_encode(model: "ModelBundle", prompt: str, device) -> torch.Tensor:
    """Prompt -> selected Qwen3-VL hidden states ``(1, L, 12, 2560)``."""
    tok = model.tokenizer(prompt)
    if tok.truncated:
        print("[krea2] prompt truncated to 512 tokens", flush=True)
    with torch.no_grad(), fp32_accumulation():
        layers = model.text_encoder(tok.ids.to(device), hidden_layers=list(SELECT_LAYERS))
    return torch.stack(layers, dim=2)[:, PREFIX_TOKENS:]


def _text_features(model: "ModelBundle", prompt: str, negative_prompt: str, need_uncond: bool, device):
    """``(cache hit, None)`` or ``(None, (cond, uncond) encoder features)``;
    the uncond branch is skipped when no step uses it."""
    cache = model.cond_cache
    hit = cache.get(("krea2", prompt, negative_prompt)) if cache is not None else None
    if hit is not None and (hit["uncond_ctx"] is not None or not need_uncond):
        return hit, None
    with staged([model.text_encoder], device, model.policy.offload_idle):
        cond = krea2_encode(model, prompt, device)
        uncond = krea2_encode(model, negative_prompt, device) if need_uncond else None
    return None, (cond, uncond)


def _contexts(model: "ModelBundle", key, hit, feats, need_uncond: bool, device):
    """Text-stream tokens for both branches (inside the backbone's staging)."""
    if hit is not None:
        unc = hit["uncond_ctx"]
        return hit["cond_ctx"].to(device), (unc.to(device) if unc is not None and need_uncond else None)
    with fp32_accumulation():
        cond_ctx = model.backbone.encode_text(feats[0])
        uncond_ctx = model.backbone.encode_text(feats[1]) if feats[1] is not None else None
    if model.cond_cache is not None:
        model.cond_cache.put(key, {"cond_ctx": cond_ctx.detach().to("cpu"),
                                   "uncond_ctx": uncond_ctx.detach().to("cpu") if uncond_ctx is not None else None})
    return cond_ctx, uncond_ctx


def krea2_sigmas(scheduler: str, steps: int, variant: str, width: int, height: int, shift: float,
                 device, **table_kwargs) -> tuple[torch.Tensor, float]:
    """``(sigmas, shift used)``. ``krea2`` is the checkpoint's own shift, ``flow``
    takes ``shift``, ``flow_dyn`` the FLUX-range dynamic shift."""
    if scheduler == "krea2":
        shift = krea2_shift(variant, width, height)
    elif scheduler == "flow_dyn":
        shift = flow_matching_dynamic_shift((height // 16) * (width // 16))
    if scheduler in ("krea2", "flow", "flow_dyn"):
        return flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32), shift
    return flow_table_schedule(scheduler, shift, steps, device=device, dtype=torch.float32,
                               **table_kwargs), shift


def krea2_text_to_image(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int | None = None,
    cfg_scale: float | None = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    sampler: str = "euler",
    scheduler: str = "krea2",
    curvature: float = 0.25,
    eta_max: float = 1.0,
    gate_reduce: str = "all",
    beta_alpha: float = 0.6,
    beta_beta: float = 0.6,
    lq_threshold: float = 0.025,
    bm_weight: float = 0.5,
    bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
    bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
    noise: torch.Tensor | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
):
    """Drive Krea 2 text-to-image. ``steps``/``cfg_scale`` default to the
    variant's (Raw 28 / 5.5, Turbo 8 / 1.0). ``noise`` overrides the seeded
    initial latent ``(1, 16, H/8, W/8)``."""
    if sampler not in _KREA2_SAMPLERS:
        raise ValueError(f"Krea 2 sampler must be one of {sorted(_KREA2_SAMPLERS)}; got {sampler!r}")
    if scheduler not in _KREA2_SCHEDULERS:
        raise ValueError(f"Krea 2 scheduler must be one of {_KREA2_SCHEDULERS}; got {scheduler!r}")
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    variant = model.spec.variant or "raw"
    defaults = KREA2_DEFAULTS[variant]
    steps = defaults["steps"] if steps is None else steps
    cfg_scale = defaults["cfg_scale"] if cfg_scale is None else cfg_scale
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    with perf_context(policy):
        sigmas, sched_shift = krea2_sigmas(
            scheduler, steps, variant, width, height, shift, device,
            alpha=beta_alpha, beta=beta_beta, threshold_noise=lq_threshold, bm_weight=bm_weight,
            bm_alpha1=bm_alpha1, bm_beta1=bm_beta1, bm_alpha2=bm_alpha2, bm_beta2=bm_beta2)
        cfg_lo, cfg_hi = guidance_interval_bounds(sigmas, cfg_interval_start, cfg_interval_end)
        need_uncond = cfg_scale != 1.0 and cfg_hi > cfg_lo
        hit, feats = _text_features(model, prompt, negative_prompt, need_uncond, device)

        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        if noise is not None:
            x = noise.to(device=device, dtype=torch.float32)
        else:
            x = torch.randn(1, 16, height // 8, width // 8, generator=gen, device=device,
                            dtype=dtype).float()

        backbone = model.backbone
        cond_ctx = uncond_ctx = None

        def velocity(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            t = sigma.reshape(-1)[:1].float()
            v = backbone(x_in, t, cond_ctx).float()
            if uncond_ctx is not None and cfg_lo < float(sigma.max()) <= cfg_hi:
                v_uncond = backbone(x_in, t, uncond_ctx).float()
                v = v_uncond + cfg_scale * (v - v_uncond)
            return v

        def denoise(x_in, sigma_b):
            """CONST x0 estimate ``x − σ·v`` in fp32."""
            return x_in.float() - sigma_b.float().view(-1, 1, 1, 1) * velocity(x_in, sigma_b)

        with staged([backbone], device, policy.offload_unet), torch.inference_mode():
            cond_ctx, uncond_ctx = _contexts(model, ("krea2", prompt, negative_prompt), hit, feats,
                                             need_uncond, device)
            total = len(sigmas) - 1
            with _step_progress(total, progress_callback, preview_callback) as on_step:
                if sampler == "euler":
                    for i in range(total):
                        sigma, sigma_next = sigmas[i], sigmas[i + 1]
                        v = velocity(x, sigma)
                        denoised = x - sigma * v
                        x = x + (sigma_next - sigma) * v
                        on_step(i, sigma, x, denoised)
                else:
                    kwargs = {}
                    if sampler in _FLOW_AWARE_SAMPLERS:
                        kwargs = dict(generator=gen, model_type="flow", shift=sched_shift)
                    elif sampler in ("exp_heun_2_x0", "uni_pc", "uni_pc_bh2"):
                        kwargs = dict(model_type="flow", shift=sched_shift)
                    if sampler in ("secant", "secant_anneal"):
                        kwargs.setdefault("generator", gen)
                        kwargs["curvature"] = curvature
                    if sampler == "infinity_aether":
                        kwargs["generator"] = gen
                    if sampler in ("euler_ancestral_anneal", "secant_anneal", "cogent", "cogent3",
                                   "cogent3_pump", "cogent3_pump_rate"):
                        kwargs["eta_max"] = eta_max
                    if sampler in ("cogent", "cogent3", "cogent3_pump", "cogent3_pump_rate"):
                        kwargs["gate_reduce"] = gate_reduce
                    x = get_sampler(sampler)(denoise, x, sigmas, callback=on_step, **kwargs)

        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z = model.vae.process_out(x.to(policy.vae_dtype))
            image, decode_mode = vae_decode_safe(model.vae, z, policy)
        image = _to_pil(image)
        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image
