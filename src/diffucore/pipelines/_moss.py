"""MOSS-Image text-to-image, img2img and inpaint (Base and SFT: one network, one
set of defaults).

prompt -> MossTokenizer (chat template, unpadded) -> Qwen3.5-9B last hidden
state -> MossDiT text embedder (once per prompt) -> MossDiT each step on the
reference's grid-snapped ``moss`` σ schedule -> FLUX.2 VAE decode through the
FLUX.2 latent bridge (pixel-shuffle + batch norm).

Guidance is the plain ``uncond + s·(cond − uncond)``; the uncond forward skips
block 22 on every step (``uncond_skip_layers``), as the reference does. Cond
and uncond have different text lengths and run unpadded, so CFG is two
sequential forwards. Noise, the CFG combine and the sampler update are fp32.

img2img / inpaint follow Anima's and Krea 2's conventions: the encode is the
exact inverse of :func:`_decode` (VAE mean, 2x2 pixel-unshuffle, ``(z - mean) /
std``), ComfyUI's denoise slice, the keep region pinned at each x0 estimate and
its original pixels composited back after decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from PIL import Image

from ..runtime import (fp32_accumulation, no_cudnn_autotune, perf_context, staged, vae_decode_safe,
                       vae_fallback_to_fp32)
from ..sampling import flow_matching_dynamic_shift, flow_matching_schedule, flow_table_schedule, get_sampler
from ..sampling import guidance_interval_bounds
from ._anima import _ANIMA_SAMPLERS, _to_pil
from ._base import PipelineInfo, _step_progress, preprocess_image
from ._flux import _flux2_patchify_latents, _flux2_unpatchify_latents
from ._krea2 import _sampler_kwargs

if TYPE_CHECKING:
    from ..bundle import ModelBundle

_MOSS_SAMPLERS = _ANIMA_SAMPLERS
_MOSS_SCHEDULERS = (
    "moss", "flow", "flow_dyn", "sgm_uniform", "simple", "normal", "infinity", "infinity_htds",
    "kl_optimal", "linear_quadratic", "smoothstep", "beta", "beta_mix", "pump_dual", "pump_taper",
    "secant_tilt", "relay_tilt",
)
MOSS_DEFAULTS = {"steps": 50, "cfg_scale": 3.0, "shift": 5.0}
MOSS_UNCOND_SKIP = (22,)
_MOSS_TRAIN_STEPS = 1000


def moss_sigmas(steps: int, shift: float = 5.0, total: int = _MOSS_TRAIN_STEPS,
                training_shift: float = 1.0) -> torch.Tensor:
    """The reference ``MOSSImageScheduler`` schedule: the shifted grid snapped to
    the nearest value of the training grid, consecutive duplicates dropped,
    terminal 0 appended. Its last nonzero σ (~0.024 at 50 steps) is far below
    the plain shift-5 flow schedule's (~0.093)."""
    if steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("steps and shifts must be positive")

    def grid(count: int, s: float) -> torch.Tensor:
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def moss_schedule(scheduler: str, steps: int, width: int, height: int, shift: float, device,
                  **table_kwargs) -> tuple[torch.Tensor, float]:
    """``(sigmas, shift used)``. ``moss`` is the checkpoint's own schedule (fixed
    shift 5, ``shift`` is ignored), ``flow`` takes ``shift``, ``flow_dyn`` the
    FLUX-range dynamic shift."""
    if scheduler == "moss":
        return moss_sigmas(steps).to(device), MOSS_DEFAULTS["shift"]
    if scheduler == "flow_dyn":
        shift = flow_matching_dynamic_shift((height // 16) * (width // 16))
    if scheduler in ("flow", "flow_dyn"):
        return flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32), shift
    return flow_table_schedule(scheduler, shift, steps, device=device, dtype=torch.float32, **table_kwargs), shift


def moss_encode(model: "ModelBundle", prompt: str, device) -> torch.Tensor:
    """Prompt -> Qwen3.5-9B last hidden state (after the final norm) ``(1, L, 4096)``."""
    tok = model.tokenizer(prompt)
    if tok.truncated:
        print("[moss] prompt truncated to 4096 tokens", flush=True)
    with torch.no_grad(), fp32_accumulation():
        return model.text_encoder(tok.ids.to(device))


def _text_features(model: "ModelBundle", prompt: str, negative_prompt: str, need_uncond: bool, device):
    """``(cache hit, None)`` or ``(None, (cond, uncond) encoder features)``;
    the uncond branch is skipped when no step uses it."""
    cache = model.cond_cache
    hit = cache.get(("moss", prompt, negative_prompt)) if cache is not None else None
    if hit is not None and (hit["uncond_ctx"] is not None or not need_uncond):
        return hit, None
    with staged([model.text_encoder], device, model.policy.offload_idle):
        cond = moss_encode(model, prompt, device)
        uncond = moss_encode(model, negative_prompt, device) if need_uncond else None
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


def _skip_layers(model: "ModelBundle", requested) -> tuple[int, ...]:
    """The uncond branch's skipped blocks: the reference default (22,) when
    ``requested`` is None (dropped if the net is shallower), else validated."""
    depth = len(model.backbone.layers)
    if requested is None:
        return tuple(i for i in MOSS_UNCOND_SKIP if i < depth)
    layers = tuple(sorted({int(i) for i in requested}))
    if any(not 0 <= i < depth for i in layers):
        raise ValueError(f"uncond_skip_layers must be block indices in [0, {depth}); got {list(requested)}")
    return layers


def _sample(model: "ModelBundle", key, hit, feats, x: torch.Tensor, sigmas: torch.Tensor, *,
            sampler: str, sampler_kwargs: dict, cfg_scale: float, cfg_lo: float, cfg_hi: float,
            need_uncond: bool, skip_layers: tuple[int, ...], progress_callback, preview_callback,
            mask_lat: torch.Tensor | None = None, z0: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate ``x`` down ``sigmas``. With ``mask_lat`` (1 = repaint) the keep
    region of each x0 estimate is pinned to ``z0``."""
    policy = model.policy
    device = policy.device
    backbone = model.backbone
    cond_ctx = uncond_ctx = None

    def velocity(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        mod, head = backbone.time_vectors(sigma.reshape(-1)[:1].float())
        v = backbone(x_in, mod, head, cond_ctx).float()
        if uncond_ctx is not None and cfg_lo < float(sigma.max()) <= cfg_hi:
            v_uncond = backbone(x_in, mod, head, uncond_ctx, skip_layers).float()
            v = v_uncond + cfg_scale * (v - v_uncond)
        return v

    def denoise(x_in, sigma_b):
        """CONST x0 estimate ``x − σ·v`` in fp32."""
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
    """DiT-space latent ``(1, 128, h, w)`` -> image: undo the batch norm and the
    2x2 pixel-shuffle (FLUX.2's bridge), then the VAE."""
    policy = model.policy
    with torch.no_grad(), staged([model.vae], policy.device, policy.offload_idle):
        mean = model.vae.flux2_latent_mean.to(x.device, torch.float32)
        std = model.vae.flux2_latent_std.to(x.device, torch.float32)
        z = _flux2_unpatchify_latents(x.float() * std + mean)
        image, decode_mode = vae_decode_safe(model.vae, z.to(policy.vae_dtype), policy)
    return _to_pil(image), decode_mode


def moss_text_to_image(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int | None = None,
    cfg_scale: float | None = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    uncond_skip_layers: "tuple[int, ...] | list[int] | None" = None,
    shift: float = 5.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    sampler: str = "euler",
    scheduler: str = "moss",
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
    """Drive MOSS-Image text-to-image. ``steps`` / ``cfg_scale`` default to the
    reference's 50 / 3.0 (CFG 1 runs no uncond forward). ``uncond_skip_layers``
    are the blocks the uncond forward skips: None = the reference's (22,),
    empty = none. ``noise`` overrides the seeded initial latent
    ``(1, 128, H/16, W/16)``."""
    if sampler not in _MOSS_SAMPLERS:
        raise ValueError(f"MOSS sampler must be one of {sorted(_MOSS_SAMPLERS)}; got {sampler!r}")
    if scheduler not in _MOSS_SCHEDULERS:
        raise ValueError(f"MOSS scheduler must be one of {_MOSS_SCHEDULERS}; got {scheduler!r}")
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    steps = MOSS_DEFAULTS["steps"] if steps is None else steps
    cfg_scale = MOSS_DEFAULTS["cfg_scale"] if cfg_scale is None else cfg_scale
    skip_layers = _skip_layers(model, uncond_skip_layers)
    policy = model.policy
    device = policy.device

    with perf_context(policy):
        sigmas, sched_shift = moss_schedule(
            scheduler, steps, width, height, shift, device,
            alpha=beta_alpha, beta=beta_beta, threshold_noise=lq_threshold, bm_weight=bm_weight,
            bm_alpha1=bm_alpha1, bm_beta1=bm_beta1, bm_alpha2=bm_alpha2, bm_beta2=bm_beta2)
        cfg_lo, cfg_hi = guidance_interval_bounds(sigmas, cfg_interval_start, cfg_interval_end)
        need_uncond = cfg_scale != 1.0 and cfg_hi > cfg_lo
        hit, feats = _text_features(model, prompt, negative_prompt, need_uncond, device)

        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        if noise is not None:
            x = noise.to(device=device, dtype=torch.float32)
        else:
            x = torch.randn(1, model.spec.latent_channels, height // 16, width // 16, generator=gen,
                            device=device, dtype=torch.float32)
        x = _sample(model, ("moss", prompt, negative_prompt), hit, feats, x, sigmas,
                    sampler=sampler,
                    sampler_kwargs=_sampler_kwargs(sampler, gen, sched_shift, curvature, eta_max, gate_reduce),
                    cfg_scale=cfg_scale, cfg_lo=cfg_lo, cfg_hi=cfg_hi, need_uncond=need_uncond,
                    skip_layers=skip_layers, progress_callback=progress_callback,
                    preview_callback=preview_callback)
        image, decode_mode = _decode(model, x)
        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image


def _encode_image(model: "ModelBundle", image: Image.Image, width: int, height: int) -> torch.Tensor:
    """Init image -> DiT-space latent ``(1, 128, h/16, w/16)``: the inverse of
    :func:`_decode`. The VAE's mean (no sampling noise), as FLUX.2's pipelines use."""
    policy = model.policy
    device = policy.device
    pixels = preprocess_image(image, width, height).to(device, policy.vae_dtype)
    with torch.no_grad(), no_cudnn_autotune(), staged([model.vae], device, policy.offload_idle):
        z = model.vae.encode(pixels, sample=False)
        if z.dtype == torch.float16 and not torch.isfinite(z).all():
            vae_fallback_to_fp32(model.vae, policy)
            z = model.vae.encode(pixels.float(), sample=False)
        mean = model.vae.flux2_latent_mean.to(device, torch.float32)
        std = model.vae.flux2_latent_std.to(device, torch.float32)
        return (_flux2_patchify_latents(z.float()) - mean) / std


def moss_img2img(
    model: "ModelBundle",
    prompt: str,
    init_image: Image.Image,
    negative_prompt: str = "",
    *,
    mask_image: "Image.Image | None" = None,
    strength: float = 0.75,
    steps: int | None = None,
    cfg_scale: float | None = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    uncond_skip_layers: "tuple[int, ...] | list[int] | None" = None,
    shift: float = 5.0,
    width: int = 1024,
    height: int = 1024,
    sampler: str = "euler",
    scheduler: str = "moss",
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
    """MOSS-Image image-to-image, or inpaint when ``mask_image`` is given (white
    = repaint). Same conventions as Anima's: ComfyUI's denoise (the schedule is
    built for ``steps / strength`` steps and its last ``steps`` run), the
    strength-noised start ``(1-σ)·z0 + σ·ε``, the keep region pinned at each
    x0 estimate and its original pixels composited back after decode. The mask
    lives on the 16 px token grid. Unknown sampler / scheduler names fall back
    to ``euler`` / ``moss``."""
    if sampler not in _MOSS_SAMPLERS:
        sampler = "euler"
    if scheduler not in _MOSS_SCHEDULERS:
        scheduler = "moss"
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    steps = MOSS_DEFAULTS["steps"] if steps is None else steps
    cfg_scale = MOSS_DEFAULTS["cfg_scale"] if cfg_scale is None else cfg_scale
    skip_layers = _skip_layers(model, uncond_skip_layers)
    policy = model.policy
    device = policy.device

    with perf_context(policy):
        sigmas, sched_shift = moss_schedule(
            scheduler, int(steps / strength), width, height, shift, device,
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

        x = _sample(model, ("moss", prompt, negative_prompt), hit, feats, x, sigmas,
                    sampler=sampler,
                    sampler_kwargs=_sampler_kwargs(sampler, gen, sched_shift, curvature, eta_max, gate_reduce),
                    cfg_scale=cfg_scale, cfg_lo=cfg_lo, cfg_hi=cfg_hi, need_uncond=need_uncond,
                    skip_layers=skip_layers, progress_callback=progress_callback,
                    preview_callback=preview_callback, mask_lat=mask_lat, z0=z0)
        image, decode_mode = _decode(model, x)
        if mask_image is not None:
            keep = np.asarray(mask_image.convert("L").resize((width, height), Image.NEAREST)) < 128
            original = np.asarray(init_image.convert("RGB").resize((width, height), Image.LANCZOS))
            image = Image.fromarray(np.where(keep[..., None], original, np.asarray(image)).astype(np.uint8))
        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image
