"""Shared machinery for the SD/SDXL pipelines: conditioning, sigma schedule,
staged sampling and VAE decode. Placement comes from the bundle's
``DevicePolicy`` (see ``docs/RUNTIME_SPEC.md``).
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
    align_your_steps_schedule,
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

if TYPE_CHECKING:  # avoid importing torch-heavy models eagerly
    from ..bundle import ModelBundle

_SCHEDULERS = {
    "karras": karras_schedule,
    "exponential": exponential_schedule,
    "polyexponential": polyexponential_schedule,
    "kl_optimal": kl_optimal_schedule,
}

# Schedulers that need the model's sigma table / timestep map, not just the range.
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
    """Start index into the [steps + 1] schedule for a strength-based run:
    ``int(strength * steps)`` steps, the A1111 convention. Shared by img2img and
    inpaint."""
    return steps - int(strength * steps)


def preprocess_image(image: Image.Image, width: int, height: int) -> torch.Tensor:
    """PIL image -> ``[1, 3, height, width]`` in ``[-1, 1]``, resized."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _match_context_chunks(ctx_c, ctx_u, conditioner):
    """Pad the shorter SDXL cond/uncond context with empty-prompt chunks so both
    span the same number of 77-token chunks (long-prompt weighting can split
    them differently, and CFG batches them). A1111 LPW behaviour."""
    if ctx_c.shape[1] == ctx_u.shape[1]:
        return ctx_c, ctx_u
    empty = conditioner("", batch=1)[0]  # [1, 77, dim]
    target = max(ctx_c.shape[1], ctx_u.shape[1])

    def pad(ctx):
        reps = (target - ctx.shape[1]) // empty.shape[1]
        return ctx if reps == 0 else torch.cat([ctx, empty.repeat(1, reps, 1)], dim=1)

    return pad(ctx_c), pad(ctx_u)


@contextmanager
def _step_progress(total: int, progress_callback: Callable[[int, int], None] | None = None,
                   preview_callback: Callable[[object], None] | None = None):
    """A tqdm bar over ``total`` steps, advanced by the sampler's callback; the
    ``denoised`` estimate is forwarded to ``preview_callback``."""
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
        """Policy for a bundle built without one: current placement, offload and
        tiling off."""
        backbone_param = next(model.backbone.parameters())
        vae_param = next(model.vae.parameters())
        return DevicePolicy(
            device=backbone_param.device,
            compute_dtype=backbone_param.dtype,
            vae_dtype=vae_param.dtype,
        )

    # --- conditioning --------------------------------------------------------
    def _encode_prompts(self, prompt, negative_prompt, width, height, policy):
        """Cond/uncond backbone kwargs, with the text encoders staged when
        offloading. The resolution-independent half is cached (checked before
        staging, so a hit skips it); SDXL's size-conditioning ``y`` is rebuilt."""
        cache = self.model.cond_cache
        # clip_skip is fixed at 1, so (prompt, negative) fully keys the cache.
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
        """The cacheable half of conditioning: context per branch plus SDXL's
        pooled vector (chunk-matched)."""
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
        """Cond/uncond backbone kwargs from an encode; SDXL's size ``y`` is built
        here since it depends on width/height."""
        if self.model.spec.architecture == "sdxl":
            # time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
            time_ids = torch.tensor([height, width, 0, 0, height, width], device=device)
            return ({"context": enc["ctx_c"], "y": self._sdxl_y(enc["pooled_c"], time_ids)},
                    {"context": enc["ctx_u"], "y": self._sdxl_y(enc["pooled_u"], time_ids)})
        return {"context": enc["ctx_c"]}, {"context": enc["ctx_u"]}

    @staticmethod
    def _sdxl_y(pooled, time_ids):
        """SDXL's added conditioning [B, 2816]: pooled text (1280) plus 256-d
        sinusoidal embeddings of the 6 time_ids."""
        size_emb = timestep_embedding(time_ids.float(), 256).flatten().unsqueeze(0)  # [1, 1536]
        return torch.cat([pooled, size_emb.to(pooled.dtype)], dim=-1)

    # --- sampling ------------------------------------------------------------
    def _denoiser(self, cond, uncond, cfg_scale, cfg_rescale=None,
                  sigmas=None, cfg_interval=(0.0, 1.0)):
        # ZTSNR checkpoints default to CFG rescale 0.7 (Lin et al.).
        if cfg_rescale is None:
            cfg_rescale = 0.7 if self.model.spec.zero_terminal_snr else 0.0
        scaling = VScaling() if self.model.spec.prediction == "v" else EpsScaling()
        denoiser = ModelDenoiser(self.model.backbone, scaling, self.model.schedule)
        # Guidance interval over the run's actual (sliced) schedule.
        lo, hi = (guidance_interval_bounds(sigmas, *cfg_interval)
                  if sigmas is not None else (-math.inf, math.inf))
        return CFGDenoiser(denoiser, cond, uncond, scale=cfg_scale, rescale=cfg_rescale,
                           sigma_lo=lo, sigma_hi=hi)

    def _sigmas(self, scheduler, steps, device, dtype):
        """The full descending sigma schedule ([steps + 1] values, ending at 0)."""
        if scheduler == "align_your_steps":
            # AYS tables are VE-range; a zero-terminal-SNR model (σ_max ~ 4500)
            # falls back to karras instead of erroring.
            if getattr(self.model.spec, "zero_terminal_snr", False):
                return karras_schedule(
                    steps,
                    self.model.schedule.sigma_min.item(),
                    self.model.schedule.sigma_max.item(),
                    device=device,
                    dtype=dtype,
                )
            return align_your_steps_schedule(
                steps,
                self.model.schedule.sigma_min.item(),
                self.model.schedule.sigma_max.item(),
                model="sdxl" if self.model.spec.architecture == "sdxl" else "sd15",
                device=device,
                dtype=dtype,
            )
        if scheduler in _SCHEDULE_FROM_MODEL:
            return _SCHEDULE_FROM_MODEL[scheduler](
                self.model.schedule, steps, device=device, dtype=dtype
            )
        try:
            schedule_fn = _SCHEDULERS[scheduler]
        except KeyError:
            available = sorted([*_SCHEDULERS, *_SCHEDULE_FROM_MODEL, "align_your_steps"])
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
        # Match the input layout to NHWC weights so cuDNN never transposes.
        if policy.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        # DeepCache (SD/SDXL UNet): a fresh cache per run, detached in finally.
        cache = DeepCache(deepcache_interval) if deepcache_interval > 1 else None
        with staged([self.model.backbone], policy.device, policy.offload_unet):
            # inference_mode inside staged: weights move in normal mode so later
            # in-place LoRA still works.
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
        del width, height  # tiling reads free VRAM, not resolution
        with torch.no_grad():
            latent = x0.to(policy.vae_dtype)
            if policy.channels_last:
                latent = latent.contiguous(memory_format=torch.channels_last)
            with staged([self.model.vae], policy.device, policy.offload_idle):
                # Tiling is decided after staging, so free VRAM is accurate.
                image, mode = vae_decode_safe(self.model.vae, latent, policy)
        image = ((image.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return Image.fromarray(image[0].permute(1, 2, 0).cpu().numpy()), mode

    # --- encode (img2img / inpaint) ------------------------------------------
    def _encode_image(self, init_image, width, height, policy, generator):
        """Encode ``init_image`` to a scaled latent on the compute device, with the
        VAE staged when offloading."""
        image = preprocess_image(init_image, width, height).to(policy.device, policy.vae_dtype)
        if policy.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            with staged([self.model.vae], policy.device, policy.offload_idle):
                z = self.model.vae.encode(image, generator=generator)
                if z.dtype == torch.float16 and not torch.isfinite(z).all():
                    # fp16 VAEs can overflow on encode too.
                    vae_fallback_to_fp32(self.model.vae, policy)
                    z = self.model.vae.encode(image.float(), generator=generator)
        return z.to(policy.compute_dtype)
