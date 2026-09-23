"""FLUX.1 / FLUX.2 text-to-image and img2img / inpaint pipelines.

FLUX.1 conditions on CLIP-L (pooled) + T5-XXL (context); FLUX.2 on three
concatenated intermediate layers of one decoder LM (Klein/Qwen3 or
Dev/Mistral). The latent is patchified into tokens with axial position ids
(FLUX.1: 3 axes, 2×2 patch, 8× VAE; FLUX.2: 4 axes, patch 1, 128-ch 16× VAE).
Guidance is distilled, so each step is a single forward with no CFG pass.
``flux_img2img`` starts from a strength-noised init and, with a mask, pins the
keep region each step (soft latent-mask inpaint).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from ..runtime import perf_context, staged, vae_decode_safe, vae_fallback_to_fp32
from ._base import PipelineInfo, _step_progress, preprocess_image
from ..sampling import (
    flow_matching_schedule,
    flow_table_schedule,
    get_sampler,
)

_FLUX_SAMPLERS = {
    "euler", "heun", "heunpp2", "euler_ancestral", "er_sde", "dpm_2", "dpm_2_ancestral",
    "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_2m_sde_heun",
    "dpmpp_3m_sde", "ipndm", "ipndm_v", "res_multistep", "res_multistep_ancestral",
    "gradient_estimation", "stork2", "infinity", "lms", "lcm", "secant",
    "sa_solver", "sa_solver_pece",
    "exp_heun_2_x0", "uni_pc", "uni_pc_bh2", "cogent", "cogent3",
}
_FLOW_AWARE_SAMPLERS = {
    "er_sde", "dpm_2_ancestral", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_2m_sde_heun",
    "dpmpp_3m_sde", "euler_ancestral", "dpmpp_2s_ancestral", "res_multistep_ancestral", "lcm",
    "cogent", "cogent3", "sa_solver", "sa_solver_pece",
}
# "ddim_uniform" is omitted: it starts below σ_max, but the init is pure noise.
_FLUX_SCHEDULERS = (
    "flux", "flow", "sgm_uniform", "simple",
    "normal", "infinity", "infinity_htds", "kl_optimal", "linear_quadratic",
)

# FLUX.2 uses ModelSamplingFlux with shift=2.02 (the log-shift `mu`).
_FLUX2_SHIFT = math.exp(2.02)

if TYPE_CHECKING:
    from ..bundle import ModelBundle


def _geom(arch: str) -> dict:
    """Per-family latent geometry: VAE downscale, DiT patch, RoPE axis count, and
    which txt-id axis carries text positions (None = all-zero txt ids)."""
    if arch == "flux2":
        return dict(downscale=16, patch=1, n_axes=4, txt_axis=3)
    return dict(downscale=8, patch=2, n_axes=3, txt_axis=None)


def _flux_shift(seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    """FLUX.1 resolution-dependent shift: ``mu`` linear in image-token count
    between (256 → 0.5) and (4096 → 1.15), shift ``exp(mu)`` (BFL
    ``get_schedule``)."""
    x1, x2 = 256, 4096
    m = (max_shift - base_shift) / (x2 - x1)
    mu = m * seq_len + base_shift - m * x1
    return math.exp(mu)


def _patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, C, h, w] -> [B, (h/p·w/p), C·p²] tokens (patch=1 is a plain flatten)."""
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)


def _unpatchify(x: torch.Tensor, h: int, w: int, patch: int) -> torch.Tensor:
    return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h // patch, w=w // patch, ph=patch, pw=patch)


def _flux2_unpatchify_latents(x: torch.Tensor) -> torch.Tensor:
    """Invert FLUX.2's 2×2 latent pixel-shuffle: [B, 4C, H, W] -> [B, C, 2H, 2W]
    (BFL ``Flux2`` ``_unpatchify_latents``)."""
    b, c, h, w = x.shape
    x = x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // 4, h * 2, w * 2)


def _flux2_patchify_latents(x: torch.Tensor) -> torch.Tensor:
    """[B, C, 2H, 2W] -> [B, 4C, H, W], the inverse of
    :func:`_flux2_unpatchify_latents` (BFL ``_patchify_latents``)."""
    b, c, H, W = x.shape
    x = x.reshape(b, c, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, H // 2, W // 2)


def _img_ids(h: int, w: int, patch: int, n_axes: int, device) -> torch.Tensor:
    """Axial position ids for the image tokens over the (h/p × w/p) patch grid:
    axis 1 = patch row, axis 2 = patch col, the rest 0."""
    gh, gw = h // patch, w // patch
    ids = torch.zeros(gh, gw, n_axes, device=device)
    ids[..., 1] = torch.arange(gh, device=device)[:, None]
    ids[..., 2] = torch.arange(gw, device=device)[None, :]
    return rearrange(ids, "h w c -> 1 (h w) c")


def _txt_ids(length: int, n_axes: int, txt_axis: int | None, device) -> torch.Tensor:
    ids = torch.zeros(1, length, n_axes, device=device)
    if txt_axis is not None:
        ids[:, :, txt_axis] = torch.linspace(0, length - 1, length, device=device)
    return ids


def _to_pil(img: torch.Tensor) -> Image.Image:
    img = ((img.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())


def _encode_text(model: "ModelBundle", prompt: str, device, dtype):
    """``(context, pooled)`` for one prompt; FLUX.2 has no pooled vector."""
    tok = model.tokenizer
    if model.spec.architecture == "flux1":
        t = tok(prompt)
        context = model.text_encoder(t.t5_ids.to(device)).to(dtype)
        _, pooled = model.text_encoder_2(t.clip_ids.to(device), return_pooled=True)
        return context, pooled.to(dtype)
    # flux2: run the LM, capture the configured layers, concatenate them.
    ids = tok(prompt).to(device)
    layers = model.text_encoder(ids, hidden_layers=tok.hidden_layers)
    context = torch.cat([h.to(dtype) for h in layers], dim=-1)
    return context, None


def flux_text_to_image(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 20,
    guidance: float = 3.5,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    sampler: str = "euler",
    scheduler: str = "flux",
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
) -> Image.Image:
    """Drive FLUX text-to-image end-to-end.

    ``guidance`` is the distilled guidance scale (ignored without a guidance
    embedding). ``scheduler`` is ``"flux"`` (family-shifted default),
    ``"flow"`` (constant ``shift``) or a flow table scheduler.
    """
    if sampler not in _FLUX_SAMPLERS:
        raise ValueError(f"FLUX sampler must be one of {sorted(_FLUX_SAMPLERS)}; got {sampler!r}")
    if scheduler not in _FLUX_SCHEDULERS:
        raise ValueError(f"FLUX scheduler must be one of {_FLUX_SCHEDULERS}; got {scheduler!r}")
    del negative_prompt  # guidance-distilled: no CFG pass
    arch = model.spec.architecture
    geom = _geom(arch)
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    if width % geom["downscale"] or height % geom["downscale"]:
        raise ValueError(f"width/height must be divisible by {geom['downscale']}; got {width}x{height}")

    with perf_context(policy):
        # 1+2. tokenize + encode. The cache is keyed on the prompt alone and
        # checked before staged(), so a hit skips the T5-XXL / Mistral staging.
        cache = model.cond_cache
        cached_ctx = cache.get((prompt,)) if cache is not None else None
        if cached_ctx is None:
            text_mods = [model.text_encoder]
            if model.text_encoder_2 is not None:
                text_mods.append(model.text_encoder_2)
            with staged(text_mods, device, policy.offload_idle):
                context, pooled = _encode_text(model, prompt, device, dtype)
            if cache is not None:
                cache.put((prompt,), {"context": context.detach().to("cpu"),
                                      "pooled": None if pooled is None else pooled.detach().to("cpu")})
        else:
            context = cached_ctx["context"].to(device)
            pooled = None if cached_ctx["pooled"] is None else cached_ctx["pooled"].to(device)

        h_lat, w_lat = height // geom["downscale"], width // geom["downscale"]
        patch = geom["patch"]
        seq_len = (h_lat // patch) * (w_lat // patch)
        img_ids = _img_ids(h_lat, w_lat, patch, geom["n_axes"], device)
        txt_ids = _txt_ids(context.shape[1], geom["n_axes"], geom["txt_axis"], device)

        # 3. σ schedule: FLUX.2 fixed shift, FLUX.1-dev shifted by resolution,
        # FLUX.1-schnell (no guidance embed) unshifted.
        if arch == "flux2":
            eff_shift = _FLUX2_SHIFT
        elif model.spec.guidance_distilled:
            eff_shift = _flux_shift(seq_len)
        else:
            eff_shift = 1.0
        if scheduler == "flux":
            sigmas = flow_matching_schedule(steps, shift=eff_shift, device=device, dtype=torch.float32)
        elif scheduler == "flow":
            eff_shift = shift
            sigmas = flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32)
        else:
            sigmas = flow_table_schedule(scheduler, eff_shift, steps, device=device, dtype=torch.float32)

        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        x = _patchify(torch.randn(1, model.spec.latent_channels, h_lat, w_lat,
                                  generator=gen, device=device, dtype=dtype), patch)

        guidance_vec = (
            torch.full((1,), guidance, device=device, dtype=dtype)
            if model.spec.guidance_distilled else None
        )

        def velocity(x_tokens, sigma_scalar):
            t = torch.full((1,), float(sigma_scalar), device=device, dtype=dtype)
            return model.backbone(x_tokens, img_ids, context, txt_ids, t, pooled, guidance_vec)

        # 4. integrate (CONST: denoised = x − σ·v). preview_callback isn't wired:
        # the latent is token-space and would need unpatchifying first.
        backbone = model.backbone
        with torch.no_grad(), staged([backbone], device, policy.offload_unet):
            if sampler == "euler":
                total = len(sigmas) - 1
                with _step_progress(total, progress_callback) as on_step:
                    for i in range(total):
                        sigma, sigma_next = sigmas[i], sigmas[i + 1]
                        v = velocity(x, sigma.item())
                        x = x + (sigma_next - sigma).to(dtype) * v
                        on_step(i, sigma, x, None)
            else:
                def denoise(x_in, sigma_b):
                    # Broadcast σ over the token tensor's (L, C).
                    v = velocity(x_in.to(dtype), sigma_b.flatten()[0])
                    sig = sigma_b.float().view(-1, 1, 1)
                    return x_in.float() - sig * v.float()

                kwargs = {}
                if sampler in _FLOW_AWARE_SAMPLERS:
                    kwargs = dict(generator=gen, model_type="flow", shift=eff_shift)
                elif sampler in ("exp_heun_2_x0", "uni_pc", "uni_pc_bh2"):  # deterministic, flow-aware
                    kwargs = dict(model_type="flow", shift=eff_shift)
                if sampler == "secant":
                    kwargs.setdefault("generator", gen)
                with _step_progress(len(sigmas) - 1, progress_callback) as on_step:
                    x = get_sampler(sampler)(denoise, x.float(), sigmas, callback=on_step, **kwargs)

        x = _unpatchify(x.to(dtype), h_lat, w_lat, patch)

        # 5. decode. FLUX.2 first undoes the latent batch norm and the 2×2
        # pixel-shuffle back to the VAE's 32-ch latent.
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            if arch == "flux2":
                if getattr(model.vae, "flux2_latent_mean", None) is not None:
                    mean = model.vae.flux2_latent_mean.to(x.device, x.dtype)
                    std = model.vae.flux2_latent_std.to(x.device, x.dtype)
                    x = x * std + mean
                x = _flux2_unpatchify_latents(x)
            latent = x.to(policy.vae_dtype)
            image, decode_mode = vae_decode_safe(model.vae, latent, policy)
        image = _to_pil(image)
        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image


def flux_img2img(
    model: "ModelBundle",
    prompt: str,
    init_image: Image.Image,
    negative_prompt: str = "",
    *,
    mask_image: "Image.Image | None" = None,
    strength: float = 0.6,
    steps: int = 20,
    guidance: float = 3.5,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    sampler: str = "euler",
    scheduler: str = "flux",
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
) -> Image.Image:
    """FLUX image-to-image, or inpaint when ``mask_image`` is given (white =
    repaint).

    Starts from ``x_σ = (1-σ)·z0 + σ·ε`` like :func:`._anima.anima_img2img`;
    ``negative_prompt`` is ignored. For inpaint the keep region is pinned to
    ``z0`` at the x0 estimate each step and the original pixels are composited
    back. SD-style sampler/scheduler names fall back to ``euler`` / ``flux``.
    """
    if sampler not in _FLUX_SAMPLERS:
        sampler = "euler"
    if scheduler not in _FLUX_SCHEDULERS:
        scheduler = "flux"
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    del negative_prompt  # guidance-distilled: no CFG pass
    arch = model.spec.architecture
    geom = _geom(arch)
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    if width % geom["downscale"] or height % geom["downscale"]:
        raise ValueError(f"width/height must be divisible by {geom['downscale']}; got {width}x{height}")

    with perf_context(policy):
        # 1. tokenize + encode (shares t2i's cache entries).
        cache = model.cond_cache
        cached_ctx = cache.get((prompt,)) if cache is not None else None
        if cached_ctx is None:
            text_mods = [model.text_encoder]
            if model.text_encoder_2 is not None:
                text_mods.append(model.text_encoder_2)
            with staged(text_mods, device, policy.offload_idle):
                context, pooled = _encode_text(model, prompt, device, dtype)
            if cache is not None:
                cache.put((prompt,), {"context": context.detach().to("cpu"),
                                      "pooled": None if pooled is None else pooled.detach().to("cpu")})
        else:
            context = cached_ctx["context"].to(device)
            pooled = None if cached_ctx["pooled"] is None else cached_ctx["pooled"].to(device)

        h_lat, w_lat = height // geom["downscale"], width // geom["downscale"]
        patch = geom["patch"]
        seq_len = (h_lat // patch) * (w_lat // patch)
        img_ids = _img_ids(h_lat, w_lat, patch, geom["n_axes"], device)
        txt_ids = _txt_ids(context.shape[1], geom["n_axes"], geom["txt_axis"], device)

        # 2. σ schedule, ComfyUI's denoise convention (as in _anima): build at
        # int(steps/strength), keep the last steps + 1. Family shift as in t2i.
        if arch == "flux2":
            eff_shift = _FLUX2_SHIFT
        elif model.spec.guidance_distilled:
            eff_shift = _flux_shift(seq_len)
        else:
            eff_shift = 1.0
        sched_steps = int(steps / strength)
        if scheduler == "flux":
            sigmas = flow_matching_schedule(sched_steps, shift=eff_shift, device=device, dtype=torch.float32)
        elif scheduler == "flow":
            eff_shift = shift
            sigmas = flow_matching_schedule(sched_steps, shift=shift, device=device, dtype=torch.float32)
        else:
            sigmas = flow_table_schedule(scheduler, eff_shift, sched_steps, device=device, dtype=torch.float32)
        sigmas = sigmas[-(steps + 1):]

        # 3. encode init → DiT-space z0 and the strength-noised start. FLUX.2
        # folds the 32-ch latent into 128-ch and normalises (the inverse of the
        # decode bridge).
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        pixels = preprocess_image(init_image, width, height).to(device, policy.vae_dtype)
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            vae_lat = model.vae.encode(pixels)
            if vae_lat.dtype == torch.float16 and not torch.isfinite(vae_lat).all():
                # An fp16 encode overflow poisons the run; retry in fp32.
                vae_fallback_to_fp32(model.vae, policy)
                vae_lat = model.vae.encode(pixels.float())
        if arch == "flux2":
            z0 = _flux2_patchify_latents(vae_lat.float())
            if getattr(model.vae, "flux2_latent_mean", None) is not None:
                mean = model.vae.flux2_latent_mean.to(z0.device, torch.float32)
                std = model.vae.flux2_latent_std.to(z0.device, torch.float32)
                z0 = (z0 - mean) / std
            z0 = z0.to(dtype)
        else:
            z0 = vae_lat.to(dtype)

        sigma0 = sigmas[0].to(dtype)
        noise = torch.randn(1, model.spec.latent_channels, h_lat, w_lat,
                            generator=gen, device=device, dtype=dtype)
        x = _patchify((1.0 - sigma0) * z0 + sigma0 * noise, patch)  # x_σ = (1-σ)·z0 + σ·ε

        mask_lat = None
        if mask_image is not None:
            m = mask_image.convert("L").resize((w_lat, h_lat), Image.BILINEAR)
            mask_lat = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None].to(device)
        z0_f = z0.float()

        guidance_vec = (
            torch.full((1,), guidance, device=device, dtype=dtype)
            if model.spec.guidance_distilled else None
        )

        def velocity(x_tokens, sigma_scalar):
            t = torch.full((1,), float(sigma_scalar), device=device, dtype=dtype)
            return model.backbone(x_tokens, img_ids, context, txt_ids, t, pooled, guidance_vec)

        # 4. integrate through a CONST x0 closure (keep region pinned for inpaint).
        backbone = model.backbone
        with torch.no_grad(), staged([backbone], device, policy.offload_unet):
            def denoise(x_in, sigma_b):
                v = velocity(x_in.to(dtype), sigma_b.flatten()[0])
                sig = sigma_b.float().view(-1, 1, 1)
                x0 = x_in.float() - sig * v.float()
                if mask_lat is not None:
                    x0_chw = _unpatchify(x0, h_lat, w_lat, patch)
                    x0_chw = x0_chw * mask_lat + z0_f * (1.0 - mask_lat)
                    x0 = _patchify(x0_chw, patch)
                return x0

            kwargs = {}
            if sampler in _FLOW_AWARE_SAMPLERS:
                kwargs = dict(generator=gen, model_type="flow", shift=eff_shift)
            elif sampler in ("exp_heun_2_x0", "uni_pc", "uni_pc_bh2"):  # deterministic, flow-aware
                kwargs = dict(model_type="flow", shift=eff_shift)
            if sampler == "secant":
                kwargs.setdefault("generator", gen)
            with _step_progress(len(sigmas) - 1, progress_callback) as on_step:
                x = get_sampler(sampler)(denoise, x.float(), sigmas, callback=on_step, **kwargs)

        x = _unpatchify(x.to(dtype), h_lat, w_lat, patch)

        # 5. decode (same bridge as t2i)
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            if arch == "flux2":
                if getattr(model.vae, "flux2_latent_mean", None) is not None:
                    mean = model.vae.flux2_latent_mean.to(x.device, x.dtype)
                    std = model.vae.flux2_latent_std.to(x.device, x.dtype)
                    x = x * std + mean
                x = _flux2_unpatchify_latents(x)
            latent = x.to(policy.vae_dtype)
            image, decode_mode = vae_decode_safe(model.vae, latent, policy)
        image = _to_pil(image)

        # inpaint: paste the original pixels back into the keep region
        if mask_image is not None:
            keep = np.asarray(mask_image.convert("L").resize((width, height), Image.NEAREST)) < 128
            original = np.asarray(init_image.convert("RGB").resize((width, height), Image.LANCZOS))
            out = np.where(keep[..., None], original, np.asarray(image))
            image = Image.fromarray(out.astype(np.uint8))

        info = PipelineInfo(vae_decode_mode=decode_mode)
        return (image, info) if return_info else image
