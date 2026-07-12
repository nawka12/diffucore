"""Anima end-to-end text-to-image pipeline (DT7).

A focused, self-contained driver that bridges the Anima-specific bits without
threading flow-matching state through the SD/SDXL ``_Pipeline`` scaffolding:

  prompt  -> AnimaTokenizer (Qwen2 + T5)
          -> Qwen3 (source hidden states)
          -> AnimaDiT.preprocess_text_embeds (LLM-Adapter, once per generation)
          -> AnimaDiT.forward(..., context=prepared_ctx) each step
          -> flow-matching σ schedule + CONST scaling + Euler integration
          -> QwenImageVAE.process_out then VAE.decode

Compared to ``_Pipeline._sample``, CFG runs as two sequential batch-1
forwards per step (cond, then uncond) instead of one stacked batch-2 pass —
deliberately: a batch-2 forward measures no faster on a compute-bound GPU
(RTX 2060: 2149 ms vs 2×1066 ms), while sequential halves peak activation
memory and lets the guidance-interval and per-branch TeaCache skips drop
the uncond forward entirely. We manage the 4D↔5D shape ourselves at the
DiT boundary rather than wrapping the backbone in an adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch
from PIL import Image

from ..models.anima_dit import TeaCache
from ..runtime import perf_context, staged, vae_decode_safe, vae_fallback_to_fp32
from ._base import PipelineInfo, _step_progress, img2img_start, preprocess_image
from ..sampling import (
    append_zero,
    calibrate_oss_schedule,
    flow_matching_schedule,
    flow_matching_dynamic_shift,
    flow_table_schedule,
    get_sampler,
    guidance_interval_bounds,
)

# Samplers Anima can drive (all routed through a CONST x0 denoiser closure).
# The stochastic, flow-aware ones additionally take ``model_type``/``shift``.
_ANIMA_SAMPLERS = {
    "euler", "heun", "heunpp2", "euler_ancestral", "euler_ancestral_anneal", "er_sde",
    "dpm_2", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde",
    "dpmpp_2m_sde_heun", "dpmpp_3m_sde", "ipndm", "ipndm_v", "res_multistep",
    "res_multistep_ancestral", "gradient_estimation", "stork2", "lms", "lcm", "secant", "secant_anneal",
    "dpmpp_2m_anneal", "exp_heun_2_x0", "uni_pc", "uni_pc_bh2", "uni_pc_anneal",
}
_FLOW_AWARE_SAMPLERS = {
    "er_sde", "dpm_2_ancestral", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_2m_sde_heun",
    "dpmpp_3m_sde", "euler_ancestral", "euler_ancestral_anneal", "secant_anneal",
    "dpmpp_2s_ancestral", "res_multistep_ancestral", "lcm", "dpmpp_2m_anneal",
    "uni_pc_anneal",
}
# "ddim_uniform" is intentionally omitted: it starts below σ_max, which clashes
# with the pure-noise (σ_max == 1) init used here. See schedules._FLOW_TABLE_SCHEDULERS.
_ANIMA_SCHEDULERS = (
    "flow", "flow_dyn", "oss", "sgm_uniform", "simple",
    "normal", "kl_optimal", "linear_quadratic", "smoothstep", "beta",
    "beta_mix",
)

if TYPE_CHECKING:
    from ..bundle import ModelBundle


def _qwen_encode(qwen3, ids, mask, device, dtype):
    """Run the Qwen text encoder and return its last hidden state in ``dtype``.

    Wrapped in ``no_grad`` because encoding is pure inference: without it the
    Qwen3.5 hybrid encoder's unrolled O(L) SSM scan retains every per-timestep
    state for a backward that never comes — enough to OOM by itself — and even
    the plain Qwen3 transformer needlessly holds its activation graph."""
    ids = ids.to(device)
    if mask is not None:
        mask = mask.to(device)
    with torch.no_grad():
        out = qwen3(ids, attention_mask=None)   # causal-only fast path; mask handled by padding below
    return out.to(dtype)


def _to_pil(img: torch.Tensor) -> Image.Image:
    img = ((img.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())


def _make_teacache(thresh: float, coeffs: "Sequence[float] | None", cfg_scale: float):
    """Build the per-CFG-branch TeaCache streams (or ``(None, None)`` when off).
    The uncond stream is omitted when CFG is disabled (single forward per step)."""
    if thresh <= 0:
        return None, None
    args = (thresh,) if coeffs is None else (thresh, coeffs)
    cond = TeaCache(*args)
    uncond = TeaCache(*args) if cfg_scale != 1.0 else None
    return cond, uncond


def anima_text_to_image(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 20,
    cfg_scale: float = 4.0,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    sampler: str = "euler",
    scheduler: str = "flow",
    curvature: float = 0.25,
    eta_max: float = 1.0,
    beta_alpha: float = 0.6,
    beta_beta: float = 0.6,
    lq_threshold: float = 0.025,
    bm_weight: float = 0.5,
    bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
    bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
    oss_sigmas: "torch.Tensor | list[float] | None" = None,
    teacache_thresh: float = 0.0,
    teacache_coefficients: "Sequence[float] | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[object], None] | None = None,
    return_info: bool = False,
) -> Image.Image:
    """Drive Anima's text-to-image path end-to-end.

    ``shift`` controls the SD3-style rectified-flow schedule (Anima's training
    default is 3.0). ``cfg_scale`` is the CFG strength; the Anima ComfyUI
    workflow defaults to ~4.0. ``cfg_interval_start``/``cfg_interval_end``
    restrict CFG to that fraction of the run (Kynkäänniemi et al., 2024) — the
    uncond forward is skipped outside the band, saving a full backbone pass per
    skipped step; the ``(0, 1)`` default guides every step.

    ``sampler`` is any of :data:`_ANIMA_SAMPLERS` (``"euler"`` keeps the exact
    closed-form rectified-flow step; the rest run through the shared sampler
    registry against a CONST x0 denoiser). ``scheduler`` picks the σ schedule:
    ``"flow"`` (the rectified-flow t-uniform default), ``"flow_dyn"`` (``flow``
    with a Flux-style resolution-aware shift derived from the image's token
    count, ignoring the passed-in ``shift``), ``"oss"`` (a pre-calibrated
    optimal-stepsize schedule supplied via ``oss_sigmas``), ``"sgm_uniform"`` or
    ``"simple"`` (ComfyUI's, evaluated against a flow sigma table).

    ``teacache_thresh`` > 0 enables TeaCache (arXiv:2411.19108): steps whose
    accumulated rescaled relative-L1 drift stays under the threshold reuse the
    cached transformer-block residual instead of recomputing it. Larger = more
    skipping = faster but lower fidelity; 0 disables. Uncalibrated on Anima
    (identity rescale), so tune the threshold empirically.
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
        # Conditioning cache: the post-adapter context depends only on the prompt
        # pair, so a repeat (seed hunting, X/Y/Z sweeps) skips tokenize + encode +
        # the LLM-Adapter — and, under offload, the Qwen encoder's PCIe round-trip.
        # The check sits before staged() so a hit never brings the encoder onto the
        # GPU. The cached value (post-adapter cond/uncond ctx) is produced inside
        # the backbone staged() block below, where the adapter runs today.
        cache = model.cond_cache
        cache_key = (prompt, negative_prompt)
        cached_ctx = cache.get(cache_key) if cache is not None else None

        if cached_ctx is None:
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
            sigmas = flow_table_schedule(scheduler, shift, steps, alpha=beta_alpha,
                                         beta=beta_beta, threshold_noise=lq_threshold,
                                         bm_weight=bm_weight, bm_alpha1=bm_alpha1,
                                         bm_beta1=bm_beta1, bm_alpha2=bm_alpha2,
                                         bm_beta2=bm_beta2,
                                         device=device, dtype=torch.float32)
        h_lat, w_lat = height // 8, width // 8
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        x = torch.randn(1, 16, h_lat, w_lat, generator=gen, device=device, dtype=dtype)
        # With σ_max == 1 the initial state is exactly pure noise (no rescale).

        # Guidance interval: CFG only while σ in (lo, hi]; the uncond forward is
        # skipped outside the band (see guidance_interval_bounds). (0, 1) = always.
        cfg_lo, cfg_hi = guidance_interval_bounds(sigmas, cfg_interval_start, cfg_interval_end)

        # ---- 4. integrate the rectified-flow ODE/SDE
        backbone = model.backbone
        # TeaCache: one cache stream per CFG branch (cond/uncond are separate
        # forwards whose modulated inputs coincide, so they can't share one).
        # ``teacache_coefficients`` rescales the raw drift (identity if None).
        tc_cond, tc_uncond = _make_teacache(teacache_thresh, teacache_coefficients, cfg_scale)
        # staged() OUTSIDE inference_mode: offloaded weights moved under inference
        # mode become inference tensors that break later in-place LoRA (add_/copy_).
        with staged([backbone], device, policy.offload_unet), torch.inference_mode():
            # The LLM-Adapter output depends only on the prompt: run it once per
            # generation and feed the DiT the prepared context (t5xxl_ids=None),
            # instead of re-running the adapter inside every backbone forward
            # (per step, per CFG branch). On a cache hit both contexts come straight
            # off CPU (adapter skipped too); on a miss we compute and cache them.
            if cached_ctx is None:
                cond_ctx = backbone.preprocess_text_embeds(cond_hidden, cond_t5)
                uncond_ctx = backbone.preprocess_text_embeds(uncond_hidden, uncond_t5)
                if cache is not None:
                    cache.put(cache_key, {"cond_ctx": cond_ctx.detach().to("cpu"),
                                          "uncond_ctx": uncond_ctx.detach().to("cpu")})
            else:
                cond_ctx = cached_ctx["cond_ctx"].to(device)
                uncond_ctx = cached_ctx["uncond_ctx"].to(device)
            if sampler == "euler":
                total = len(sigmas) - 1
                with _step_progress(total, progress_callback, preview_callback) as on_step:
                    for i in range(total):
                        sigma, sigma_next = sigmas[i], sigmas[i + 1]
                        x_5d = x.unsqueeze(2)                     # (B, C, 1, H, W)
                        t = torch.full((1,), sigma.item(), device=device, dtype=dtype)

                        v_cond = backbone(x_5d, t, cond_ctx, teacache=tc_cond).squeeze(2)
                        if cfg_scale == 1.0 or not (cfg_lo < float(sigma) <= cfg_hi):
                            v = v_cond
                        else:
                            # CUDA Graphs (reduce-overhead) returns a view of the
                            # graph's static output buffer, which the uncond replay
                            # below overwrites — clone the cond output first
                            # (PyTorch's documented fix for sequential invocations).
                            if policy.cuda_graphs:
                                v_cond = v_cond.clone()
                            v_uncond = backbone(x_5d, t, uncond_ctx, teacache=tc_uncond).squeeze(2)
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
                    v_cond = backbone(x_5d, t, cond_ctx, teacache=tc_cond).squeeze(2)
                    if cfg_scale == 1.0 or not (cfg_lo < float(sigma_b.max()) <= cfg_hi):
                        v = v_cond
                    else:
                        if policy.cuda_graphs:  # uncond replay overwrites v_cond's buffer (see euler path)
                            v_cond = v_cond.clone()
                        v_uncond = backbone(x_5d, t, uncond_ctx, teacache=tc_uncond).squeeze(2)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    sig = sigma_b.float().view(-1, 1, 1, 1)
                    return x_in.float() - sig * v.float()

                kwargs = {}
                if sampler in _FLOW_AWARE_SAMPLERS:
                    kwargs = dict(generator=gen, model_type="flow", shift=sched_shift)
                elif sampler in ("exp_heun_2_x0", "uni_pc", "uni_pc_bh2"):  # deterministic, flow-aware
                    kwargs = dict(model_type="flow", shift=sched_shift)
                if sampler in ("secant", "secant_anneal"):
                    kwargs.setdefault("generator", gen)
                    kwargs["curvature"] = curvature
                if sampler in ("euler_ancestral_anneal", "secant_anneal", "dpmpp_2m_anneal"):
                    kwargs["eta_max"] = eta_max
                # uni_pc_anneal intentionally omitted: even with its order-ramp the
                # shared 1.0 panel default over-softens it (deterministic stays
                # cleanest), so it ships a low baked-in eta_max (0.2).
                with _step_progress(len(sigmas) - 1, progress_callback, preview_callback) as on_step:
                    x = get_sampler(sampler)(denoise, x.float(), sigmas, callback=on_step, **kwargs)

        if tc_cond is not None:
            print(f"[teacache] cached {tc_cond.skips}/{tc_cond.calls} steps", flush=True)

        # ---- 5. process_out then decode (tiled when explicitly requested, or
        # auto-tiled when free VRAM can't host an untiled decode — Qwen-Image
        # VAE decode is whole-tensor and OOMs on 12 GB above 1024² with the
        # DiT resident, but at 1024² it fits and the smart check picks untiled).
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z = model.vae.process_out(x.to(policy.vae_dtype))
            image, decode_mode = vae_decode_safe(model.vae, z, policy)
        image = _to_pil(image)
        info = PipelineInfo(vae_decode_mode=decode_mode)
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
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    sampler: str = "euler",
    scheduler: str = "flow",
    seed: int | None = None,
    curvature: float = 0.25,
    eta_max: float = 1.0,
    beta_alpha: float = 0.6,
    beta_beta: float = 0.6,
    lq_threshold: float = 0.025,
    bm_weight: float = 0.5,
    bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
    bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
    oss_sigmas: "torch.Tensor | list[float] | None" = None,
    teacache_thresh: float = 0.0,
    teacache_coefficients: "Sequence[float] | None" = None,
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
        # ---- 1. tokenize + encode cond/uncond (same as t2i, and shares its cache —
        # the post-adapter context is resolution-independent). See anima_text_to_image.
        cache = model.cond_cache
        cache_key = (prompt, negative_prompt)
        cached_ctx = cache.get(cache_key) if cache is not None else None
        if cached_ctx is None:
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
            sigmas = flow_table_schedule(scheduler, shift, sched_steps, alpha=beta_alpha,
                                         beta=beta_beta, threshold_noise=lq_threshold,
                                         bm_weight=bm_weight, bm_alpha1=bm_alpha1,
                                         bm_beta1=bm_beta1, bm_alpha2=bm_alpha2,
                                         bm_beta2=bm_beta2,
                                         device=device, dtype=torch.float32)

        # ---- 3. encode init → DiT-space latent z0; build strength-noised start
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        pixels = preprocess_image(init_image, width, height).to(device, policy.vae_dtype)
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z_vae = model.vae.encode(pixels)
            if z_vae.dtype == torch.float16 and not torch.isfinite(z_vae).all():
                # fp16 encode overflow poisons the whole run — retry fp32 (see
                # vae_fallback_to_fp32; the decode below then also runs fp32).
                vae_fallback_to_fp32(model.vae, policy)
                z_vae = model.vae.encode(pixels.float())
            z0 = model.vae.process_in(z_vae).to(dtype)
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

        # Guidance interval over the sliced schedule (the steps that actually
        # run); the uncond forward is skipped outside (lo, hi]. (0, 1) = always.
        cfg_lo, cfg_hi = guidance_interval_bounds(sigmas, cfg_interval_start, cfg_interval_end)

        # ---- 4. integrate against a CONST x0 closure (keep region pinned for inpaint)
        backbone = model.backbone
        # TeaCache: one cache stream per CFG branch (see anima_text_to_image).
        tc_cond, tc_uncond = _make_teacache(teacache_thresh, teacache_coefficients, cfg_scale)
        # staged() OUTSIDE inference_mode: offloaded weights moved under inference
        # mode become inference tensors that break later in-place LoRA (add_/copy_).
        with staged([backbone], device, policy.offload_unet), torch.inference_mode():
            # Adapter runs once per generation, not per forward (see t2i); the
            # post-adapter context is cached across repeats (shared with t2i).
            if cached_ctx is None:
                cond_ctx = backbone.preprocess_text_embeds(cond_hidden, cond_t5)
                uncond_ctx = backbone.preprocess_text_embeds(uncond_hidden, uncond_t5)
                if cache is not None:
                    cache.put(cache_key, {"cond_ctx": cond_ctx.detach().to("cpu"),
                                          "uncond_ctx": uncond_ctx.detach().to("cpu")})
            else:
                cond_ctx = cached_ctx["cond_ctx"].to(device)
                uncond_ctx = cached_ctx["uncond_ctx"].to(device)

            def denoise(x_in, sigma_b):
                x_5d = x_in.to(dtype).unsqueeze(2)
                t = sigma_b.to(dtype)
                v_cond = backbone(x_5d, t, cond_ctx, teacache=tc_cond).squeeze(2)
                if cfg_scale == 1.0 or not (cfg_lo < float(sigma_b.max()) <= cfg_hi):
                    v = v_cond
                else:
                    if policy.cuda_graphs:  # uncond replay overwrites v_cond's buffer (see t2i)
                        v_cond = v_cond.clone()
                    v_uncond = backbone(x_5d, t, uncond_ctx, teacache=tc_uncond).squeeze(2)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                sig = sigma_b.float().view(-1, 1, 1, 1)
                x0 = x_in.float() - sig * v.float()
                if mask_lat is not None:
                    x0 = x0 * mask_lat + z0_f * (1.0 - mask_lat)
                return x0

            kwargs = {}
            if sampler in _FLOW_AWARE_SAMPLERS:
                kwargs = dict(generator=gen, model_type="flow", shift=sched_shift)
            elif sampler in ("exp_heun_2_x0", "uni_pc", "uni_pc_bh2"):  # deterministic, flow-aware
                kwargs = dict(model_type="flow", shift=sched_shift)
            if sampler in ("secant", "secant_anneal"):
                kwargs.setdefault("generator", gen)
                kwargs["curvature"] = curvature
            if sampler in ("euler_ancestral_anneal", "secant_anneal", "dpmpp_2m_anneal"):
                kwargs["eta_max"] = eta_max
            # uni_pc_anneal intentionally omitted: see the t2i path — it ships a
            # low baked-in eta_max (0.2) instead of the shared 1.0 panel default.
            with _step_progress(len(sigmas) - 1, progress_callback, preview_callback) as on_step:
                x = get_sampler(sampler)(denoise, x.float(), sigmas, callback=on_step, **kwargs)

        if tc_cond is not None:
            print(f"[teacache] cached {tc_cond.skips}/{tc_cond.calls} steps", flush=True)

        # ---- 5. decode
        with torch.no_grad(), staged([model.vae], device, policy.offload_idle):
            z = model.vae.process_out(x.to(policy.vae_dtype))
            image, decode_mode = vae_decode_safe(model.vae, z, policy)
        image = _to_pil(image)

        # inpaint: paste the original pixels back into the keep region (byte-exact)
        if mask_image is not None:
            keep = np.asarray(mask_image.convert("L").resize((width, height), Image.NEAREST)) < 128
            original = np.asarray(init_image.convert("RGB").resize((width, height), Image.LANCZOS))
            out = np.where(keep[..., None], original, np.asarray(image))
            image = Image.fromarray(out.astype(np.uint8))

        info = PipelineInfo(vae_decode_mode=decode_mode)
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
        # staged() OUTSIDE inference_mode: offloaded weights moved under inference
        # mode become inference tensors that break later in-place LoRA (add_/copy_).
        with staged([backbone], device, policy.offload_unet), torch.inference_mode():
            # Adapter runs once per calibration, not per forward (see t2i).
            cond_ctx = backbone.preprocess_text_embeds(cond_hidden, cond_t5)
            uncond_ctx = backbone.preprocess_text_embeds(uncond_hidden, uncond_t5)

            def denoise(x_in, sigma_b):
                x_5d = x_in.to(dtype).unsqueeze(2)
                t = sigma_b.to(dtype)
                v_cond = backbone(x_5d, t, cond_ctx).squeeze(2)
                if cfg_scale == 1.0:
                    v = v_cond
                else:
                    if policy.cuda_graphs:  # uncond replay overwrites v_cond's buffer (see t2i)
                        v_cond = v_cond.clone()
                    v_uncond = backbone(x_5d, t, uncond_ctx).squeeze(2)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                sig = sigma_b.float().view(-1, 1, 1, 1)
                return x_in.float() - sig * v.float()

            sigmas = calibrate_oss_schedule(
                denoise, x.float(), candidate, num_steps=steps,
                progress_callback=progress_callback,
            )

    return [float(s) for s in sigmas.tolist()]


def anima_calibrate_teacache(
    model: "ModelBundle",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 50,
    width: int = 1024,
    height: int = 1024,
    shift: float = 3.0,
    cfg_scale: float = 4.0,
    seed: int = 0,
    degree: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[float]:
    """Fit TeaCache rescaling coefficients for this Anima model (arXiv:2411.19108).

    Runs one full euler trajectory and records, per step, the pair
    ``(x = rel-L1 drift of the block-0 timestep-modulated input,
       y = rel-L1 drift of the conditioned velocity output)`` — then least-squares
    fits a degree-``degree`` polynomial ``y ≈ f(x)``. That polynomial is what
    rescales raw input drift into an output-change estimate so the accumulated
    threshold means the same thing across step counts and resolutions.

    The fit is a property of the **architecture**, not the prompt/seed, so one
    calibration transfers across Anima checkpoints and settings. Run at a high
    ``steps`` (dense σ sampling) so the curve covers lower step counts too.
    Returns coefficients highest-degree-first (``numpy.polyfit`` / ``poly1d``
    order), ready to hand to :class:`~diffucore.models.anima_dit.TeaCache`.
    """
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16; got {width}x{height}")
    if steps < degree + 2:
        raise ValueError(f"steps ({steps}) too small to fit degree {degree}; use >= {degree + 2}")
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

        sigmas = flow_matching_schedule(steps, shift=shift, device=device, dtype=torch.float32)
        h_lat, w_lat = height // 8, width // 8
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(1, 16, h_lat, w_lat, generator=gen, device=device, dtype=dtype)

        # Recording stream on the cond branch: never skips, logs raw input rel-L1.
        tc_cond = TeaCache(0.0, record=True)
        out_rel: list[float] = []   # y: rel-L1 drift of the cond velocity output
        prev_v = None

        backbone = model.backbone
        with staged([backbone], device, policy.offload_unet), torch.inference_mode():
            # Adapter runs once per calibration, not per forward (see t2i).
            cond_ctx = backbone.preprocess_text_embeds(cond_hidden, cond_t5)
            uncond_ctx = backbone.preprocess_text_embeds(uncond_hidden, uncond_t5)
            total = len(sigmas) - 1
            with _step_progress(total, progress_callback) as on_step:
                for i in range(total):
                    sigma, sigma_next = sigmas[i], sigmas[i + 1]
                    x_5d = x.unsqueeze(2)
                    t = torch.full((1,), sigma.item(), device=device, dtype=dtype)
                    v_cond = backbone(x_5d, t, cond_ctx, teacache=tc_cond).squeeze(2)
                    if policy.cuda_graphs:
                        # prev_v is read on the NEXT step, after every intervening
                        # replay has overwritten the graph's static output buffer —
                        # clone at production (also covers the uncond replay below).
                        v_cond = v_cond.clone()
                    if prev_v is not None:
                        denom = prev_v.abs().mean().clamp_min(1e-8)
                        out_rel.append(((v_cond - prev_v).abs().mean() / denom).item())
                    prev_v = v_cond
                    if cfg_scale == 1.0:
                        v = v_cond
                    else:
                        v_uncond = backbone(x_5d, t, uncond_ctx).squeeze(2)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    x = x + (sigma_next - sigma).to(dtype) * v
                    on_step(i, sigma, x, None)

    # tc_cond.rel_history and out_rel are both step-2..N, so already aligned.
    xs = np.asarray(tc_cond.rel_history, dtype=np.float64)
    ys = np.asarray(out_rel, dtype=np.float64)
    coeffs = np.polyfit(xs, ys, degree)
    return [float(c) for c in coeffs]
