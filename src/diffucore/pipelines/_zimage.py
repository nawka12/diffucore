"""Z-Image text-to-image, img2img and inpaint (Turbo and Base).

prompt -> ZImageTokenizer (chat template, unpadded) -> Qwen3-4B hidden state
after layer 35 of 36 (the one before the last, no final norm) -> ZImageDiT
caption path (once per prompt) -> ZImageDiT each step on a shifted flow sigma
schedule -> FLUX.1 VAE decode.

Turbo and Base are the same network; the variant only picks defaults and the
checkpoint's own shift (3.0 and 6.0). Turbo is distilled for 8 steps and no
CFG; Base runs full CFG with a negative prompt. The model timestep is
``1 - sigma`` and its output is minus the flow velocity. The reference's
``guidance_scale`` g combines ``cond + g * (cond - uncond)``, the usual CFG scale
g + 1, which is what ``cfg_scale`` takes here. ``cfg_normalization`` > 0 caps
the guided prediction's norm at that multiple of the conditional one (the
reference's option, off by default); the reference's ``cfg_truncation`` is the
shared guidance interval (``cfg_interval_*``). Cond and uncond have different
text lengths and run unpadded, so CFG is two sequential forwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from PIL import Image

from ..runtime import (fp32_accumulation, no_cudnn_autotune, perf_context, staged, vae_decode_safe,
                       vae_fallback_to_fp32)
from ..sampling import (
    flow_matching_dynamic_shift,
    flow_matching_schedule,
    flow_table_schedule,
    get_sampler,
    guidance_interval_bounds,
)
from ._anima import _ANIMA_SAMPLERS, _to_pil
from ._base import PipelineInfo, _step_progress, preprocess_image
from ._krea2 import _sampler_kwargs

if TYPE_CHECKING:
    from ..bundle import ModelBundle

_ZIMAGE_SAMPLERS = _ANIMA_SAMPLERS
_ZIMAGE_SCHEDULERS = (
    "zimage", "flow", "flow_dyn", "sgm_uniform", "simple", "normal", "infinity", "infinity_htds",
    "kl_optimal", "linear_quadratic", "smoothstep", "beta", "beta_mix", "pump_dual", "pump_taper",
    "secant_tilt", "relay_tilt",
)
ZIMAGE_DEFAULTS = {
    "turbo": {"steps": 8, "cfg_scale": 1.0, "shift": 3.0},
    "base": {"steps": 50, "cfg_scale": 5.0, "shift": 6.0},
}
TE_LAYER = 35


def zimage_sigmas(scheduler: str, steps: int, variant: str, width: int, height: int, shift: float,
                  device, **table_kwargs) -> tuple[torch.Tensor, float]:
    """``(sigmas, shift used)``. ``zimage`` is the variant's own shift (the
    reference scheduler: sigma = shift*t / (1 + (shift-1)*t) over
    ``t = linspace(1, 1/N, N)``, plus the terminal 0), ``flow`` takes ``shift``,
    ``flow_dyn`` the FLUX-range dynamic shift."""
    if scheduler == "zimage":
        shift = ZIMAGE_DEFAULTS[variant]["shift"]
    elif scheduler == "flow_dyn":
        shift = flow_matching_dynamic_shift((height // 16) * (width // 16))
    if scheduler in ("zimage", "flow", "flow_dyn"):
        return flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32), shift
    return flow_table_schedule(scheduler, shift, steps, device=device, dtype=torch.float32,
                               **table_kwargs), shift


def zimage_encode(model: "ModelBundle", prompt: str, device) -> torch.Tensor:
    """Prompt -> Qwen3-4B hidden state after layer 35 ``(1, L, 2560)``."""
    tok = model.tokenizer(prompt)
    if tok.truncated:
        print("[z_image] prompt truncated to 512 tokens", flush=True)
    with torch.no_grad(), fp32_accumulation():
        return model.text_encoder(tok.ids.to(device), hidden_layers=[TE_LAYER])[0]


def _text_features(model: "ModelBundle", prompt: str, negative_prompt: str, need_uncond: bool, device):
    """``(cache hit, None)`` or ``(None, (cond, uncond) encoder features)``;
    the uncond branch is skipped when no step uses it."""
    cache = model.cond_cache
    hit = cache.get(("z_image", prompt, negative_prompt)) if cache is not None else None
    if hit is not None and (hit["uncond_ctx"] is not None or not need_uncond):
        return hit, None
    with staged([model.text_encoder], device, model.policy.offload_idle):
        cond = zimage_encode(model, prompt, device)
        uncond = zimage_encode(model, negative_prompt, device) if need_uncond else None
    return None, (cond, uncond)


def _contexts(model: "ModelBundle", key, hit, feats, need_uncond: bool, device):
    """Caption tokens for both branches (inside the backbone's staging)."""
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


def _sample(model: "ModelBundle", key, hit, feats, x: torch.Tensor, sigmas: torch.Tensor, *,
            sampler: str, sampler_kwargs: dict, cfg_scale: float, cfg_lo: float, cfg_hi: float,
            cfg_normalization: float, need_uncond: bool, progress_callback, preview_callback,
            mask_lat: torch.Tensor | None = None, z0: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate ``x`` down ``sigmas``. With ``mask_lat`` (1 = repaint) the keep
    region of each x0 estimate is pinned to ``z0``."""
    policy = model.policy
    device = policy.device
    backbone = model.backbone
    cond_ctx = uncond_ctx = None

    def velocity(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = 1.0 - sigma.reshape(-1)[:1].float()
        v = -backbone(x_in, t, cond_ctx).float()
        if uncond_ctx is not None and cfg_lo < float(sigma.max()) <= cfg_hi:
            v_uncond = -backbone(x_in, t, uncond_ctx).float()
            guided = v_uncond + cfg_scale * (v - v_uncond)
            if cfg_normalization > 0:
                cap = torch.linalg.vector_norm(v) * cfg_normalization
                norm = torch.linalg.vector_norm(guided)
                if norm > cap:
                    guided = guided * (cap / norm)
            v = guided
        return v

    def denoise(x_in, sigma_b):
        """CONST x0 estimate ``x - sigma * v`` in fp32."""
        x0 = x_in.float() - sigma_b.float().view(-1, 1, 1, 1) * velocity(x_in, sigma_b)
        if mask_lat is not None:
            x0 = x0 * mask_lat + z0 * (1.0 - mask_lat)
        return x0

    with staged([backbone], device, policy.offload_unet), torch.inference_mode():
        cond_ctx, uncond_ctx = _contexts(model, key, hit, feats, need_uncond, device)
        total = len(sigmas) - 1
        with _step_progress(total, progress_callback, preview_callback) as on_step:
            if sampler == "euler" and mask_lat is None:
                for i in range(total):
                    sigma, sigma_next = sigmas[i], sigmas[i + 1]
                    v = velocity(x, sigma)
                    denoised = x - sigma * v
                    x = x + (sigma_next - sigma) * v
                    on_step(i, sigma, x, denoised)
            else:
                x = get_sampler(sampler)(denoise, x, sigmas, callback=on_step, **sampler_kwargs)
    return x


def _decode(model: "ModelBundle", x: torch.Tensor):
    policy = model.policy
    with torch.no_grad(), staged([model.vae], policy.device, policy.offload_idle):
        image, decode_mode = vae_decode_safe(model.vae, x.to(policy.vae_dtype), policy)
    return _to_pil(image), decode_mode


def _variant_defaults(model: "ModelBundle", steps: int | None, cfg_scale: float | None):
    variant = model.spec.variant or "turbo"
    d = ZIMAGE_DEFAULTS[variant]
    return variant, (d["steps"] if steps is None else steps), (d["cfg_scale"] if cfg_scale is None else cfg_scale)


def zimage_text_to_image(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int | None = None,
    cfg_scale: float | None = None,
    cfg_normalization: float = 0.0,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    sampler: str = "euler",
    scheduler: str = "zimage",
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
    """Drive Z-Image text-to-image. ``steps``/``cfg_scale`` default to the
    variant's (Turbo 8 / 1.0, Base 50 / 5.0; CFG 1 runs no uncond forward).
    ``noise`` overrides the seeded initial latent ``(1, 16, H/8, W/8)``."""
    if sampler not in _ZIMAGE_SAMPLERS:
        raise ValueError(f"Z-Image sampler must be one of {sorted(_ZIMAGE_SAMPLERS)}; got {sampler!r}")
    if scheduler not in _ZIMAGE_SCHEDULERS:
        raise ValueError(f"Z-Image scheduler must be one of {_ZIMAGE_SCHEDULERS}; got {scheduler!r}")
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    variant, steps, cfg_scale = _variant_defaults(model, steps, cfg_scale)
    policy = model.policy
    device = policy.device

    with perf_context(policy):
        sigmas, sched_shift = zimage_sigmas(
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
            x = torch.randn(1, model.spec.latent_channels, height // 8, width // 8, generator=gen,
                            device=device, dtype=torch.float32)
        x = _sample(model, ("z_image", prompt, negative_prompt), hit, feats, x, sigmas,
                    sampler=sampler,
                    sampler_kwargs=_sampler_kwargs(sampler, gen, sched_shift, curvature, eta_max, gate_reduce),
                    cfg_scale=cfg_scale, cfg_lo=cfg_lo, cfg_hi=cfg_hi, cfg_normalization=cfg_normalization,
                    need_uncond=need_uncond, progress_callback=progress_callback,
                    preview_callback=preview_callback)
        image, decode_mode = _decode(model, x)
        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image


def _encode_image(model: "ModelBundle", image: Image.Image, width: int, height: int) -> torch.Tensor:
    """Init image -> latent ``(1, 16, h/8, w/8)`` in sampling space: the inverse
    of :func:`_decode`. The VAE's mean (no sampling noise), as the reference's
    img2img uses."""
    policy = model.policy
    device = policy.device
    pixels = preprocess_image(image, width, height).to(device, policy.vae_dtype)
    with torch.no_grad(), no_cudnn_autotune(), staged([model.vae], device, policy.offload_idle):
        z = model.vae.encode(pixels, sample=False)
        if z.dtype == torch.float16 and not torch.isfinite(z).all():
            vae_fallback_to_fp32(model.vae, policy)
            z = model.vae.encode(pixels.float(), sample=False)
        return z.float()


def zimage_img2img(
    model: "ModelBundle",
    prompt: str,
    init_image: Image.Image,
    negative_prompt: str = "",
    *,
    mask_image: "Image.Image | None" = None,
    strength: float = 0.75,
    steps: int | None = None,
    cfg_scale: float | None = None,
    cfg_normalization: float = 0.0,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    sampler: str = "euler",
    scheduler: str = "zimage",
    seed: int | None = None,
    curvature: float = 0.25,
    eta_max: float = 1.0,
    gate_reduce: str = "all",
    beta_alpha: float = 0.6,
    beta_beta: float = 0.6,
    lq_threshold: float = 0.025,
    bm_weight: float = 0.5,
    bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
    bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
):
    """Z-Image image-to-image, or inpaint when ``mask_image`` is given (white =
    repaint). Same conventions as Anima's: ComfyUI's denoise (the schedule is
    built for ``steps / strength`` steps and its last ``steps`` run), the
    strength-noised start ``(1-sigma)*z0 + sigma*eps``, the keep region pinned at
    each x0 estimate and its original pixels composited back after decode.
    Unknown sampler / scheduler names fall back to ``euler`` / ``zimage``."""
    if sampler not in _ZIMAGE_SAMPLERS:
        sampler = "euler"
    if scheduler not in _ZIMAGE_SCHEDULERS:
        scheduler = "zimage"
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    variant, steps, cfg_scale = _variant_defaults(model, steps, cfg_scale)
    policy = model.policy
    device = policy.device

    with perf_context(policy):
        sigmas, sched_shift = zimage_sigmas(
            scheduler, int(steps / strength), variant, width, height, shift, device,
            alpha=beta_alpha, beta=beta_beta, threshold_noise=lq_threshold, bm_weight=bm_weight,
            bm_alpha1=bm_alpha1, bm_beta1=bm_beta1, bm_alpha2=bm_alpha2, bm_beta2=bm_beta2)
        sigmas = sigmas[-(steps + 1):]
        cfg_lo, cfg_hi = guidance_interval_bounds(sigmas, cfg_interval_start, cfg_interval_end)
        need_uncond = cfg_scale != 1.0 and cfg_hi > cfg_lo
        hit, feats = _text_features(model, prompt, negative_prompt, need_uncond, device)

        z0 = _encode_image(model, init_image, width, height)
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(z0.shape, generator=gen, device=device, dtype=torch.float32)
        sigma0 = sigmas[0]
        x = (1.0 - sigma0) * z0 + sigma0 * noise

        mask_lat = None
        if mask_image is not None:
            m = mask_image.convert("L").resize((z0.shape[-1], z0.shape[-2]), Image.BILINEAR)
            mask_lat = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None].to(device)

        x = _sample(model, ("z_image", prompt, negative_prompt), hit, feats, x, sigmas,
                    sampler=sampler,
                    sampler_kwargs=_sampler_kwargs(sampler, gen, sched_shift, curvature, eta_max, gate_reduce),
                    cfg_scale=cfg_scale, cfg_lo=cfg_lo, cfg_hi=cfg_hi, cfg_normalization=cfg_normalization,
                    need_uncond=need_uncond, progress_callback=progress_callback,
                    preview_callback=preview_callback, mask_lat=mask_lat, z0=z0)
        image, decode_mode = _decode(model, x)
        if mask_image is not None:
            keep = np.asarray(mask_image.convert("L").resize((width, height), Image.NEAREST)) < 128
            original = np.asarray(init_image.convert("RGB").resize((width, height), Image.LANCZOS))
            image = Image.fromarray(np.where(keep[..., None], original, np.asarray(image)).astype(np.uint8))
        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image
