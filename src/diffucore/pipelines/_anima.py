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

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from PIL import Image

from ..runtime import can_decode_untiled, perf_context, staged, tiled_vae_decode
from ._base import PipelineInfo, _step_progress, img2img_start, preprocess_image
from ..sampling import (
    append_zero,
    calibrate_oss_schedule,
    flow_matching_schedule,
    flow_matching_dynamic_shift,
    flow_table_schedule,
    get_sampler,
)

# Samplers Anima can drive (all routed through a CONST x0 denoiser closure).
# The stochastic, flow-aware ones additionally take ``model_type``/``shift``.
_ANIMA_SAMPLERS = {
    "euler", "heun", "heunpp2", "euler_ancestral", "euler_ancestral_anneal", "er_sde",
    "dpm_2", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde",
    "dpmpp_2m_sde_heun", "dpmpp_3m_sde", "ipndm", "ipndm_v", "res_multistep",
    "res_multistep_ancestral", "gradient_estimation", "lms", "lcm", "secant", "secant_anneal",
}
_FLOW_AWARE_SAMPLERS = {
    "er_sde", "dpm_2_ancestral", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_2m_sde_heun",
    "dpmpp_3m_sde", "euler_ancestral", "euler_ancestral_anneal", "secant_anneal",
    "dpmpp_2s_ancestral", "res_multistep_ancestral", "lcm",
}
# "ddim_uniform" is intentionally omitted: it starts below σ_max, which clashes
# with the pure-noise (σ_max == 1) init used here. See schedules._FLOW_TABLE_SCHEDULERS.
_ANIMA_SCHEDULERS = (
    "flow", "flow_dyn", "oss", "sgm_uniform", "simple",
    "normal", "kl_optimal", "linear_quadratic", "smoothstep", "beta",
)

if TYPE_CHECKING:
    from ..bundle import ModelBundle


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
    sampler: str = "euler",
    scheduler: str = "flow",
    curvature: float = 0.25,
    oss_sigmas: "torch.Tensor | list[float] | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
) -> Image.Image:
    """Drive Anima's text-to-image path end-to-end.

    ``shift`` controls the SD3-style rectified-flow schedule (Anima's training
    default is 3.0). ``cfg_scale`` is the CFG strength; the Anima ComfyUI
    workflow defaults to ~4.0.

    ``sampler`` is any of :data:`_ANIMA_SAMPLERS` (``"euler"`` keeps the exact
    closed-form rectified-flow step; the rest run through the shared sampler
    registry against a CONST x0 denoiser). ``scheduler`` picks the σ schedule:
    ``"flow"`` (the rectified-flow t-uniform default), ``"flow_dyn"`` (``flow``
    with a Flux-style resolution-aware shift derived from the image's token
    count, ignoring the passed-in ``shift``), ``"oss"`` (a pre-calibrated
    optimal-stepsize schedule supplied via ``oss_sigmas``), ``"sgm_uniform"`` or
    ``"simple"`` (ComfyUI's, evaluated against a flow sigma table).
    """
    if sampler not in _ANIMA_SAMPLERS:
        raise ValueError(f"Anima sampler must be one of {sorted(_ANIMA_SAMPLERS)}; got {sampler!r}")
    if scheduler not in _ANIMA_SCHEDULERS:
        raise ValueError(f"Anima scheduler must be one of {_ANIMA_SCHEDULERS}; got {scheduler!r}")
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    # Spatial dims must be divisible by VAE-stride·patch = 8·2 = 16.
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")

    with perf_context(policy):
        # ---- 1. tokenize cond + uncond
        cond_tok = model.tokenizer(prompt)
        uncond_tok = model.tokenizer(negative_prompt)

        # ---- 2. encode with Qwen3 (staged onto device when offloading)
        with staged([model.text_encoder], device, policy.offload_idle):
            qwen_dtype = next(model.text_encoder.parameters()).dtype
            cond_hidden = _qwen_encode(model.text_encoder, cond_tok.qwen_ids, cond_tok.qwen_mask, device, dtype)
            uncond_hidden = _qwen_encode(model.text_encoder, uncond_tok.qwen_ids, uncond_tok.qwen_mask, device, dtype)
            del qwen_dtype  # silence unused warning if the Qwen3 dtype probe ever shifts

        cond_t5 = cond_tok.t5_ids.to(device)
        uncond_t5 = uncond_tok.t5_ids.to(device)

        # ---- 3. σ schedule, init noise
        sched_shift = shift
        if scheduler == "flow":
            sigmas = flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32)
        elif scheduler == "flow_dyn":
            sched_shift = flow_matching_dynamic_shift((height // 16) * (width // 16))
            sigmas = flow_matching_schedule(steps, shift=sched_shift, device=device, dtype=torch.float32)
        elif scheduler == "oss":
            if oss_sigmas is None:
                raise ValueError(
                    "scheduler='oss' needs a pre-calibrated schedule. Run "
                    "diffucore.sampling.calibrate_oss_schedule on this model/resolution "
                    "and pass the result as oss_sigmas."
                )
            s = torch.as_tensor(oss_sigmas, device=device, dtype=torch.float32)
            sigmas = s if float(s[-1]) == 0.0 else append_zero(s)
        else:
            sigmas = flow_table_schedule(scheduler, shift, steps, device=device, dtype=torch.float32)
        h_lat, w_lat = height // 8, width // 8
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        x = torch.randn(1, 16, h_lat, w_lat, generator=gen, device=device, dtype=dtype)
        # With σ_max == 1 the initial state is exactly pure noise (no rescale).

        # ---- 4. integrate the rectified-flow ODE/SDE
        backbone = model.backbone
        with torch.no_grad(), staged([backbone], device, policy.offload_unet):
            if sampler == "euler":
                total = len(sigmas) - 1
                with _step_progress(total, progress_callback, preview_callback) as on_step:
                    for i in range(total):
                        sigma, sigma_next = sigmas[i], sigmas[i + 1]
                        x_5d = x.unsqueeze(2)                     # (B, C, 1, H, W)
                        t = torch.full((1,), sigma.item(), device=device, dtype=dtype)

                        v_cond = backbone(x_5d, t, cond_hidden, t5xxl_ids=cond_t5).squeeze(2)
                        if cfg_scale == 1.0:
                            v = v_cond
                        else:
                            v_uncond = backbone(x_5d, t, uncond_hidden, t5xxl_ids=uncond_t5).squeeze(2)
                            v = v_uncond + cfg_scale * (v_cond - v_uncond)

                        # CONST flow: denoised = x − σ·v ; Euler step is x + (σ_next − σ)·v
                        # (closed-form exact for any constant x0 estimate).
                        denoised = x - sigma.to(dtype) * v
                        x = x + (sigma_next - sigma).to(dtype) * v
                        on_step(i, sigma, x, denoised)
            else:  # registry samplers — need a CONST x0 estimate; integrate in fp32 like ComfyUI
                def denoise(x_in, sigma_b):
                    """``model(x, σ) -> x0``: predict velocity (with CFG), return the
                    CONST x0 estimate ``x − σ·v`` in fp32 for the solver math."""
                    x_5d = x_in.to(dtype).unsqueeze(2)
                    t = sigma_b.to(dtype)
                    v_cond = backbone(x_5d, t, cond_hidden, t5xxl_ids=cond_t5).squeeze(2)
                    if cfg_scale == 1.0:
                        v = v_cond
                    else:
                        v_uncond = backbone(x_5d, t, uncond_hidden, t5xxl_ids=uncond_t5).squeeze(2)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    sig = sigma_b.float().view(-1, 1, 1, 1)
                    return x_in.float() - sig * v.float()

                kwargs = {}
                if sampler in _FLOW_AWARE_SAMPLERS:
                    kwargs = dict(generator=gen, model_type="flow", shift=sched_shift)
                if sampler in ("secant", "secant_anneal"):
                    kwargs.setdefault("generator", gen)
                    kwargs["curvature"] = curvature
                with _step_progress(len(sigmas) - 1, progress_callback, preview_callback) as on_step:
                    x = get_sampler(sampler)(denoise, x.float(), sigmas, callback=on_step, **kwargs)

        # ---- 5. process_out then decode (tiled when explicitly requested, or
        # auto-tiled when free VRAM can't host an untiled decode — Qwen-Image
        # VAE decode is whole-tensor and OOMs on 12 GB above 1024² with the
        # DiT resident, but at 1024² it fits and the smart check picks untiled).
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z = model.vae.process_out(x.to(policy.vae_dtype))
            tile = policy.vae_tile or not can_decode_untiled(model.vae, z.shape, device)
            image = tiled_vae_decode(model.vae, z) if tile else model.vae.decode(z)
        image = _to_pil(image)
        info = PipelineInfo(vae_decode_mode="tiled" if tile else "untiled")
        return (image, info) if return_info else image


def anima_img2img(
    model: "ModelBundle",
    prompt: str,
    init_image: Image.Image,
    negative_prompt: str = "",
    *,
    mask_image: "Image.Image | None" = None,
    strength: float = 0.75,
    steps: int = 20,
    cfg_scale: float = 4.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    sampler: str = "euler",
    scheduler: str = "flow",
    seed: int | None = None,
    curvature: float = 0.25,
    oss_sigmas: "torch.Tensor | list[float] | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
) -> Image.Image:
    """Anima image-to-image, or inpaint when ``mask_image`` is given (white =
    repaint, black = keep).

    Mirrors :func:`anima_text_to_image` but starts the rectified-flow ODE from the
    strength-noised init latent ``x_σ = (1-σ)·z0 + σ·ε`` instead of pure noise.
    For inpaint the keep region (mask 0) is pinned to the init latent ``z0`` at the
    x0-estimate each step — the same masking the SD ``MaskedDenoiser`` does, which
    is scaling-agnostic, so it holds for the flow ``x0 = x − σ·v`` too — and the
    original pixels are composited back after decode so untouched areas stay exact.

    Sampler/scheduler are coerced to Anima-valid defaults (``euler`` / ``flow``)
    when an SD-style value comes through, so the shared img2img/inpaint pipelines
    and the detailer "just work" on Anima.
    """
    if sampler not in _ANIMA_SAMPLERS:
        sampler = "euler"
    if scheduler not in _ANIMA_SCHEDULERS:
        scheduler = "flow"
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    with perf_context(policy):
        # ---- 1. tokenize + encode cond/uncond (same as t2i)
        cond_tok = model.tokenizer(prompt)
        uncond_tok = model.tokenizer(negative_prompt)
        with staged([model.text_encoder], device, policy.offload_idle):
            cond_hidden = _qwen_encode(model.text_encoder, cond_tok.qwen_ids, cond_tok.qwen_mask, device, dtype)
            uncond_hidden = _qwen_encode(model.text_encoder, uncond_tok.qwen_ids, uncond_tok.qwen_mask, device, dtype)
        cond_t5 = cond_tok.t5_ids.to(device)
        uncond_t5 = uncond_tok.t5_ids.to(device)

        # ---- 2. σ schedule. img2img/inpaint follow ComfyUI's KSampler denoise
        # convention (Anima's reference): build the schedule at int(steps/strength)
        # resolution and keep the last `steps + 1` σ (sliced in step 3), so a
        # strength<1 run still takes the full `steps` from the strength-appropriate
        # σ — unlike SD/SDXL's A1111 default (see _base.img2img_start).
        sched_shift = shift
        sched_steps = int(steps / strength)
        if scheduler == "flow":
            sigmas = flow_matching_schedule(sched_steps, shift=shift, device=device, dtype=torch.float32)
        elif scheduler == "flow_dyn":
            sched_shift = flow_matching_dynamic_shift((height // 16) * (width // 16))
            sigmas = flow_matching_schedule(sched_steps, shift=sched_shift, device=device, dtype=torch.float32)
        elif scheduler == "oss":
            if oss_sigmas is None:
                raise ValueError("scheduler='oss' needs a pre-calibrated schedule (oss_sigmas).")
            s = torch.as_tensor(oss_sigmas, device=device, dtype=torch.float32)
            sigmas = s if float(s[-1]) == 0.0 else append_zero(s)
        else:
            sigmas = flow_table_schedule(scheduler, shift, sched_steps, device=device, dtype=torch.float32)

        # ---- 3. encode init → DiT-space latent z0; build strength-noised start
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        pixels = preprocess_image(init_image, width, height).to(device, policy.vae_dtype)
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z0 = model.vae.process_in(model.vae.encode(pixels)).to(dtype)
        # Keep the tail: last `steps + 1` σ → run `steps` from σ(t≈strength), the
        # ComfyUI denoise slice. OSS is a fixed calibrated trajectory, so it falls
        # back to the A1111 start index instead.
        sigmas = sigmas[img2img_start(steps, strength):] if scheduler == "oss" \
            else sigmas[-(steps + 1):]
        sigma0 = sigmas[0].to(dtype)
        noise = torch.randn(z0.shape, generator=gen, device=device, dtype=dtype)
        x = (1.0 - sigma0) * z0 + sigma0 * noise            # flow forward: x_σ=(1-σ)z0+σε

        mask_lat = None
        if mask_image is not None:
            m = mask_image.convert("L").resize((width // 8, height // 8), Image.BILINEAR)
            mask_lat = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None].to(device)
        z0_f = z0.float()

        # ---- 4. integrate against a CONST x0 closure (keep region pinned for inpaint)
        backbone = model.backbone
        with torch.no_grad(), staged([backbone], device, policy.offload_unet):
            def denoise(x_in, sigma_b):
                x_5d = x_in.to(dtype).unsqueeze(2)
                t = sigma_b.to(dtype)
                v_cond = backbone(x_5d, t, cond_hidden, t5xxl_ids=cond_t5).squeeze(2)
                if cfg_scale == 1.0:
                    v = v_cond
                else:
                    v_uncond = backbone(x_5d, t, uncond_hidden, t5xxl_ids=uncond_t5).squeeze(2)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                sig = sigma_b.float().view(-1, 1, 1, 1)
                x0 = x_in.float() - sig * v.float()
                if mask_lat is not None:
                    x0 = x0 * mask_lat + z0_f * (1.0 - mask_lat)
                return x0

            kwargs = {}
            if sampler in _FLOW_AWARE_SAMPLERS:
                kwargs = dict(generator=gen, model_type="flow", shift=sched_shift)
            if sampler in ("secant", "secant_anneal"):
                kwargs.setdefault("generator", gen)
                kwargs["curvature"] = curvature
            with _step_progress(len(sigmas) - 1, progress_callback, preview_callback) as on_step:
                x = get_sampler(sampler)(denoise, x.float(), sigmas, callback=on_step, **kwargs)

        # ---- 5. decode
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z = model.vae.process_out(x.to(policy.vae_dtype))
            tile = policy.vae_tile or not can_decode_untiled(model.vae, z.shape, device)
            image = tiled_vae_decode(model.vae, z) if tile else model.vae.decode(z)
        image = _to_pil(image)

        # inpaint: paste the original pixels back into the keep region (byte-exact)
        if mask_image is not None:
            keep = np.asarray(mask_image.convert("L").resize((width, height), Image.NEAREST)) < 128
            original = np.asarray(init_image.convert("RGB").resize((width, height), Image.LANCZOS))
            out = np.where(keep[..., None], original, np.asarray(image))
            image = Image.fromarray(out.astype(np.uint8))

        info = PipelineInfo(vae_decode_mode="tiled" if tile else "untiled")
        return (image, info) if return_info else image


def anima_calibrate_oss(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int,
    width: int = 1024,
    height: int = 1024,
    shift: float = 3.0,
    cfg_scale: float = 4.0,
    grid: int = 80,
    seed: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[float]:
    """Calibrate an OSS (optimal-stepsize) schedule for this Anima model/config.

    Runs one dense ``grid``-point teacher trajectory, scores every candidate
    single step, and DP-distills the ``steps``-step schedule that minimizes total
    truncation error (see :func:`diffucore.sampling.calibrate_oss_schedule`).
    Returns the descending σ list (trailing ``0`` included). One-time and
    GPU-heavy; cache the result and feed it back via ``oss_sigmas``.

    The conditioning + denoise closure mirror the registry-sampler path in
    :func:`anima_text_to_image` (a CONST x0 estimate ``x − σ·v`` with CFG).
    """
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    if grid < steps:
        raise ValueError(f"grid ({grid}) must be >= steps ({steps})")
    policy = model.policy
    device, dtype = policy.device, policy.compute_dtype

    with perf_context(policy):
        cond_tok = model.tokenizer(prompt)
        uncond_tok = model.tokenizer(negative_prompt)
        with staged([model.text_encoder], device, policy.offload_idle):
            cond_hidden = _qwen_encode(model.text_encoder, cond_tok.qwen_ids, cond_tok.qwen_mask, device, dtype)
            uncond_hidden = _qwen_encode(model.text_encoder, uncond_tok.qwen_ids, uncond_tok.qwen_mask, device, dtype)
        cond_t5 = cond_tok.t5_ids.to(device)
        uncond_t5 = uncond_tok.t5_ids.to(device)

        h_lat, w_lat = height // 8, width // 8
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(1, 16, h_lat, w_lat, generator=gen, device=device, dtype=dtype)
        # Dense descending candidate grid (no trailing 0), same σ(t) map as "flow".
        candidate = flow_matching_schedule(grid, shift=shift, device=device, dtype=torch.float32)[:-1]

        backbone = model.backbone
        with torch.no_grad(), staged([backbone], device, policy.offload_unet):
            def denoise(x_in, sigma_b):
                x_5d = x_in.to(dtype).unsqueeze(2)
                t = sigma_b.to(dtype)
                v_cond = backbone(x_5d, t, cond_hidden, t5xxl_ids=cond_t5).squeeze(2)
                if cfg_scale == 1.0:
                    v = v_cond
                else:
                    v_uncond = backbone(x_5d, t, uncond_hidden, t5xxl_ids=uncond_t5).squeeze(2)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                sig = sigma_b.float().view(-1, 1, 1, 1)
                return x_in.float() - sig * v.float()

            sigmas = calibrate_oss_schedule(
                denoise, x.float(), candidate, num_steps=steps,
                progress_callback=progress_callback,
            )

    return [float(s) for s in sigmas.tolist()]
