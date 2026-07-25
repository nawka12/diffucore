"""Shared machinery for the diffusion pipelines.

:class:`TextToImage` and :class:`ImageToImage` differ only in how they produce the
initial latent ``x`` at ``sigmas[0]``; the conditioning, sigma schedule, staged
sampling loop, and staged VAE decode are identical and live here so each pipeline
stays a thin wrapper. Placement (offload / tiling) is read from the bundle's
``DevicePolicy`` and applied per stage — see ``docs/RUNTIME_SPEC.md``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ..conditioning import Conditioner, SDXLConditioner
from ..models.unet import DeepCache, timestep_embedding
from ..runtime import DevicePolicy, staged, vae_decode_safe, vae_fallback_to_fp32
from ..sampling import (
    CFGDenoiser,
    EpsScaling,
    ModelDenoiser,
    VScaling,
    ddim_uniform_schedule,
    exponential_schedule,
    get_sampler,
    guidance_interval_bounds,
    infinity_htds_schedule,
    infinity_schedule,
    karras_schedule,
    kl_optimal_schedule,
    linear_quadratic_schedule,
    normal_schedule,
    polyexponential_schedule,
    sgm_uniform_schedule,
    simple_schedule,
)

if TYPE_CHECKING:  # avoid importing the bundle (and torch-heavy models) eagerly
    from ..bundle import ModelBundle

_SCHEDULERS = {
    "karras": karras_schedule,
    "exponential": exponential_schedule,
    "polyexponential": polyexponential_schedule,
    "kl_optimal": kl_optimal_schedule,
}

# Schedulers that read the model's discrete sigma table / timestep map rather
# than just (sigma_min, sigma_max); called with the schedule object.
_SCHEDULE_FROM_MODEL = {
    "simple": simple_schedule,
    "sgm_uniform": sgm_uniform_schedule,
    "normal": normal_schedule,
    "infinity": infinity_schedule,
    "infinity_htds": infinity_htds_schedule,
    "ddim_uniform": ddim_uniform_schedule,
    "linear_quadratic": linear_quadratic_schedule,
}


@dataclass
class PipelineInfo:
    """Runtime details produced during generation."""

    vae_decode_mode: str = "unknown"


def img2img_start(steps: int, strength: float) -> int:
    """Index into the full ([steps + 1]) sigma schedule where a strength-based run
    starts. Runs ``int(strength * steps)`` denoising steps (the k-diffusion / A1111
    convention): ``strength=1`` starts at index 0 (the full schedule), smaller
    values start later (fewer steps, more of the init image preserved). Shared by
    img2img and inpainting."""
    return steps - int(strength * steps)


def preprocess_image(image: Image.Image, width: int, height: int) -> torch.Tensor:
    """PIL image -> ``FloatTensor[1, 3, height, width]`` in ``[-1, 1]``, resized to
    ``(width, height)`` — the input range the VAE encoder expects."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _match_context_chunks(ctx_c, ctx_u, conditioner):
    """Pad the shorter of the SDXL cond/uncond contexts so both span the same
    number of 77-token chunks. With long-prompt weighting the positive and
    negative prompts can split into different chunk counts (e.g. a long positive
    prompt -> 154, a short negative -> 77), leaving their contexts unequal along
    the sequence axis. CFG batches them along the batch axis (see ``CFGDenoiser``),
    which needs equal sequence lengths; pad the shorter with the empty-prompt
    chunk embedding (A1111 LPW)."""
    if ctx_c.shape[1] == ctx_u.shape[1]:
        return ctx_c, ctx_u
    empty = conditioner("", batch=1)[0]  # [1, 77, dim] empty-prompt window
    target = max(ctx_c.shape[1], ctx_u.shape[1])

    def pad(ctx):
        reps = (target - ctx.shape[1]) // empty.shape[1]
        return ctx if reps == 0 else torch.cat([ctx, empty.repeat(1, reps, 1)], dim=1)

    return pad(ctx_c), pad(ctx_u)


@contextmanager
def _step_progress(total: int, progress_callback: Callable[[int, int], None] | None = None,
                   preview_callback: Callable[[object], None] | None = None):
    """A tqdm bar over the ``total`` sampling steps, advanced through the
    sampler's ``callback`` hook. Yields the callback; closes the bar on exit.

    The sampler calls ``on_step(i, sigma, x, denoised)``; when ``preview_callback``
    is set the ``denoised`` x0 estimate (4th arg) is forwarded to it for live
    latent previews."""
    bar = tqdm(total=total, desc="sampling", leave=True)
    try:
        def on_step(*args):
            bar.update(1)
            if progress_callback is not None:
                progress_callback(bar.n, total)
            if preview_callback is not None and len(args) >= 4 and args[3] is not None:
                preview_callback(args[3])

        yield on_step
    finally:
        bar.close()


class _Pipeline:
    """Common conditioning / sampling / decode plumbing for the pipelines."""

    def __init__(self, model: "ModelBundle"):
        self.model = model

    # --- placement -----------------------------------------------------------
    def _policy(self) -> DevicePolicy:
        return self.model.policy or self._fallback_policy(self.model)

    @staticmethod
    def _fallback_policy(model):
        """Policy for a bundle built without one (e.g. direct construction):
        read current placement off the modules, offload/tiling off."""
        backbone_param = next(model.backbone.parameters())
        vae_param = next(model.vae.parameters())
        return DevicePolicy(
            device=backbone_param.device,
            compute_dtype=backbone_param.dtype,
            vae_dtype=vae_param.dtype,
        )

    # --- conditioning --------------------------------------------------------
    def _encode_prompts(self, prompt, negative_prompt, width, height, policy):
        """Cond/uncond kwarg dicts for the backbone, with the text encoder(s)
        staged onto the GPU for the duration when offloading.

        The resolution-independent half (context, and SDXL's pooled vector) is
        cached on the bundle when a ``cond_cache`` is present, so a repeat prompt
        skips both the encode and — under offload — the encoder's PCIe staging. The
        check sits before ``staged()`` so a hit never brings the encoder onto the
        GPU. SDXL's size-conditioning ``y`` depends on width/height, so it is
        assembled fresh each call and kept out of the cache."""
        cache = self.model.cond_cache
        # clip_skip is fixed at 1 here, so (prompt, negative) fully keys the value;
        # add clip_skip to the key if it ever becomes a runtime argument.
        key = (prompt, negative_prompt)
        enc = cache.get(key) if cache is not None else None
        if enc is None:
            with staged(self._text_modules(), policy.device, policy.offload_idle):
                enc = self._encode_conditioning(prompt, negative_prompt)
            if cache is not None:
                cache.put(key, {k: v.detach().to("cpu") for k, v in enc.items()})
        else:
            enc = {k: v.to(policy.device) for k, v in enc.items()}
        return self._assemble_conditioning(enc, width, height, policy.device)

    def _text_modules(self):
        """The text encoder(s) resident during the conditioning stage."""
        mods = [self.model.text_encoder]
        if self.model.text_encoder_2 is not None:
            mods.append(self.model.text_encoder_2)
        return mods

    def _encode_conditioning(self, prompt, negative_prompt):
        """The resolution-independent (cacheable) half of conditioning: the context
        per branch, plus SDXL's pooled text vector. SDXL's chunk-matching runs here
        — it depends only on the prompt pair, not on width/height."""
        model = self.model
        if model.spec.architecture == "sdxl":
            conditioner = SDXLConditioner(model.tokenizer, model.text_encoder, model.text_encoder_2)
            ctx_c, pooled_c = conditioner(prompt, batch=1)
            ctx_u, pooled_u = conditioner(negative_prompt, batch=1)
            ctx_c, ctx_u = _match_context_chunks(ctx_c, ctx_u, conditioner)
            return {"ctx_c": ctx_c, "pooled_c": pooled_c, "ctx_u": ctx_u, "pooled_u": pooled_u}
        conditioner = Conditioner(model.tokenizer, model.text_encoder, clip_skip=1)
        return {"ctx_c": conditioner(prompt, batch=1), "ctx_u": conditioner(negative_prompt, batch=1)}

    def _assemble_conditioning(self, enc, width, height, device):
        """Build the cond/uncond backbone kwargs from an encode. SDXL adds the
        size-conditioning ``y`` (pooled text + time_ids), which depends on
        width/height and so is (re)built here rather than cached."""
        if self.model.spec.architecture == "sdxl":
            # time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
            time_ids = torch.tensor([height, width, 0, 0, height, width], device=device)
            return ({"context": enc["ctx_c"], "y": self._sdxl_y(enc["pooled_c"], time_ids)},
                    {"context": enc["ctx_u"], "y": self._sdxl_y(enc["pooled_u"], time_ids)})
        return {"context": enc["ctx_c"]}, {"context": enc["ctx_u"]}

    @staticmethod
    def _sdxl_y(pooled, time_ids):
        """Assemble SDXL's added conditioning vector [B, 2816]: pooled text (1280)
        concatenated with the sinusoidal embedding (256 each) of the 6 time_ids."""
        size_emb = timestep_embedding(time_ids.float(), 256).flatten().unsqueeze(0)  # [1, 1536]
        return torch.cat([pooled, size_emb.to(pooled.dtype)], dim=-1)

    # --- sampling ------------------------------------------------------------
    def _denoiser(self, cond, uncond, cfg_scale, cfg_rescale=None,
                  sigmas=None, cfg_interval=(0.0, 1.0)):
        # ZTSNR checkpoints default to CFG rescale 0.7 (Lin et al.); everything else
        # to plain CFG. Pass an explicit ``cfg_rescale`` to override either way.
        if cfg_rescale is None:
            cfg_rescale = 0.7 if self.model.spec.zero_terminal_snr else 0.0
        scaling = VScaling() if self.model.spec.prediction == "v" else EpsScaling()
        denoiser = ModelDenoiser(self.model.backbone, scaling, self.model.schedule)
        # Guidance interval: skip the uncond forward outside the band. Bounds are
        # computed from the run's actual (sliced) schedule; ``sigmas=None`` or the
        # (0, 1) default leaves guidance on at every step.
        lo, hi = (guidance_interval_bounds(sigmas, *cfg_interval)
                  if sigmas is not None else (-math.inf, math.inf))
        return CFGDenoiser(denoiser, cond, uncond, scale=cfg_scale, rescale=cfg_rescale,
                           sigma_lo=lo, sigma_hi=hi)

    def _sigmas(self, scheduler, steps, device, dtype):
        """The full descending sigma schedule ([steps + 1] values, ending at 0)."""
        if scheduler in _SCHEDULE_FROM_MODEL:
            return _SCHEDULE_FROM_MODEL[scheduler](
                self.model.schedule, steps, device=device, dtype=dtype
            )
        try:
            schedule_fn = _SCHEDULERS[scheduler]
        except KeyError:
            available = sorted([*_SCHEDULERS, *_SCHEDULE_FROM_MODEL])
            raise ValueError(f"unknown scheduler {scheduler!r}; available: {available}") from None
        return schedule_fn(
            steps,
            self.model.schedule.sigma_min.item(),
            self.model.schedule.sigma_max.item(),
            device=device,
            dtype=dtype,
        )

    def _sample(self, sampler, cfg, x, sigmas, policy, progress_callback=None,
                preview_callback=None, deepcache_interval=1):
        # Match the input layout to the (NHWC) UNet weights when channels_last is
        # on, so cuDNN runs entirely in channels-last instead of transposing each
        # step. Cheap one-shot reorder; semantically a no-op.
        if policy.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        # DeepCache (SD/SDXL UNet only): reuse deep features across steps. Attach
        # a fresh cache to the backbone for this run; > 1 enables it. Detached in
        # the finally so it never leaks into a later (cache-off) generation.
        cache = DeepCache(deepcache_interval) if deepcache_interval > 1 else None
        with staged([self.model.backbone], policy.device, policy.offload_unet):
            # inference_mode INSIDE staged: weights move (.to) in normal mode so
            # they stay normal tensors for later in-place LoRA; only the forward
            # runs under inference_mode.
            with torch.inference_mode():
                with _step_progress(len(sigmas) - 1, progress_callback, preview_callback) as on_step:
                    if cache is not None:
                        self.model.backbone._deepcache = cache
                    try:
                        return get_sampler(sampler)(cfg, x, sigmas, callback=on_step)
                    finally:
                        if cache is not None:
                            self.model.backbone._deepcache = None

    # --- decode --------------------------------------------------------------
    def _decode(self, x0, policy, width, height) -> tuple[Image.Image, str]:
        del width, height  # tile decision now reads free VRAM, not resolution
        with torch.no_grad():
            latent = x0.to(policy.vae_dtype)
            if policy.channels_last:
                latent = latent.contiguous(memory_format=torch.channels_last)
            with staged([self.model.vae], policy.device, policy.offload_idle):
                # Tile-vs-untiled is decided *after* staging (inside the helper)
                # so free VRAM reflects the actual decode-time state (VAE on
                # device, UNet/encoders gone or resident per the policy).
                image, mode = vae_decode_safe(self.model.vae, latent, policy)
        image = ((image.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return Image.fromarray(image[0].permute(1, 2, 0).cpu().numpy()), mode

    # --- encode (img2img / inpaint) ------------------------------------------
    def _encode_image(self, init_image, width, height, policy, generator):
        """Encode ``init_image`` to a scaled latent on the compute device, with the
        (fp32) VAE staged onto the GPU when offloading."""
        image = preprocess_image(init_image, width, height).to(policy.device, policy.vae_dtype)
        if policy.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            with staged([self.model.vae], policy.device, policy.offload_idle):
                z = self.model.vae.encode(image, generator=generator)
                if z.dtype == torch.float16 and not torch.isfinite(z).all():
                    # fp16 VAEs can overflow on encode too (same failure class
                    # as decode); a poisoned latent would ruin the whole run.
                    vae_fallback_to_fp32(self.model.vae, policy)
                    z = self.model.vae.encode(image.float(), generator=generator)
        return z.to(policy.compute_dtype)
