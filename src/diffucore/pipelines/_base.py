"""Shared machinery for the diffusion pipelines.

:class:`TextToImage` and :class:`ImageToImage` differ only in how they produce the
initial latent ``x`` at ``sigmas[0]``; the conditioning, sigma schedule, staged
sampling loop, and staged VAE decode are identical and live here so each pipeline
stays a thin wrapper. Placement (offload / tiling) is read from the bundle's
``DevicePolicy`` and applied per stage — see ``docs/RUNTIME_SPEC.md``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ..conditioning import Conditioner, SDXLConditioner
from ..models.unet import timestep_embedding
from ..runtime import DevicePolicy, can_decode_untiled, staged, tiled_vae_decode
from ..sampling import (
    CFGDenoiser,
    EpsScaling,
    ModelDenoiser,
    VScaling,
    exponential_schedule,
    get_sampler,
    karras_schedule,
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
}

# Schedulers that read the model's discrete sigma table / timestep map rather
# than just (sigma_min, sigma_max); called with the schedule object.
_SCHEDULE_FROM_MODEL = {
    "simple": simple_schedule,
    "sgm_uniform": sgm_uniform_schedule,
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


@contextmanager
def _step_progress(total: int, progress_callback: Callable[[int, int], None] | None = None,
                   preview_callback: Callable[[object], None] | None = None):
    """A tqdm bar over the ``total`` sampling steps, advanced through the
    sampler's ``callback`` hook. Yields the callback; closes the bar on exit.

    The sampler calls ``on_step(i, sigma, x, denoised)``; when ``preview_callback``
    is set the ``denoised`` x0 estimate (4th arg) is forwarded to it for live
    latent previews."""
    bar = tqdm(total=total, desc="sampling", leave=False)
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
        staged onto the GPU for the duration when offloading."""
        with staged(self._text_modules(), policy.device, policy.offload_idle):
            return self._conditioning(prompt, negative_prompt, width, height, policy.device)

    def _text_modules(self):
        """The text encoder(s) resident during the conditioning stage."""
        mods = [self.model.text_encoder]
        if self.model.text_encoder_2 is not None:
            mods.append(self.model.text_encoder_2)
        return mods

    def _conditioning(self, prompt, negative_prompt, width, height, device):
        """Build the cond/uncond kwarg dicts forwarded to the backbone. SDXL adds
        a pooled-text + size-conditioning ``y`` vector alongside the context."""
        model = self.model
        if model.spec.architecture == "sdxl":
            conditioner = SDXLConditioner(model.tokenizer, model.text_encoder, model.text_encoder_2)
            ctx_c, pooled_c = conditioner(prompt, batch=1)
            ctx_u, pooled_u = conditioner(negative_prompt, batch=1)
            # time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
            time_ids = torch.tensor([height, width, 0, 0, height, width], device=device)
            return ({"context": ctx_c, "y": self._sdxl_y(pooled_c, time_ids)},
                    {"context": ctx_u, "y": self._sdxl_y(pooled_u, time_ids)})

        conditioner = Conditioner(model.tokenizer, model.text_encoder, clip_skip=1)
        return {"context": conditioner(prompt, batch=1)}, {"context": conditioner(negative_prompt, batch=1)}

    @staticmethod
    def _sdxl_y(pooled, time_ids):
        """Assemble SDXL's added conditioning vector [B, 2816]: pooled text (1280)
        concatenated with the sinusoidal embedding (256 each) of the 6 time_ids."""
        size_emb = timestep_embedding(time_ids.float(), 256).flatten().unsqueeze(0)  # [1, 1536]
        return torch.cat([pooled, size_emb.to(pooled.dtype)], dim=-1)

    # --- sampling ------------------------------------------------------------
    def _denoiser(self, cond, uncond, cfg_scale, cfg_rescale=None):
        # ZTSNR checkpoints default to CFG rescale 0.7 (Lin et al.); everything else
        # to plain CFG. Pass an explicit ``cfg_rescale`` to override either way.
        if cfg_rescale is None:
            cfg_rescale = 0.7 if self.model.spec.zero_terminal_snr else 0.0
        scaling = VScaling() if self.model.spec.prediction == "v" else EpsScaling()
        denoiser = ModelDenoiser(self.model.backbone, scaling, self.model.schedule)
        return CFGDenoiser(denoiser, cond, uncond, scale=cfg_scale, rescale=cfg_rescale)

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

    def _sample(self, sampler, cfg, x, sigmas, policy, progress_callback=None, preview_callback=None):
        # Match the input layout to the (NHWC) UNet weights when channels_last is
        # on, so cuDNN runs entirely in channels-last instead of transposing each
        # step. Cheap one-shot reorder; semantically a no-op.
        if policy.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            with staged([self.model.backbone], policy.device, policy.offload_unet):
                with _step_progress(len(sigmas) - 1, progress_callback, preview_callback) as on_step:
                    return get_sampler(sampler)(cfg, x, sigmas, callback=on_step)

    # --- decode --------------------------------------------------------------
    def _decode(self, x0, policy, width, height) -> tuple[Image.Image, str]:
        del width, height  # tile decision now reads free VRAM, not resolution
        with torch.no_grad():
            latent = x0.to(policy.vae_dtype)
            if policy.channels_last:
                latent = latent.contiguous(memory_format=torch.channels_last)
            with staged([self.model.vae], policy.device, policy.offload_idle):
                # Decide tile-vs-untiled *after* staging so free VRAM reflects
                # the actual decode-time state (VAE on device, UNet/encoders
                # gone or resident per the policy).
                tile = policy.vae_tile or not can_decode_untiled(
                    self.model.vae, latent.shape, policy.device
                )
                image = tiled_vae_decode(self.model.vae, latent) if tile else self.model.vae.decode(latent)
        image = ((image.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return Image.fromarray(image[0].permute(1, 2, 0).cpu().numpy()), ("tiled" if tile else "untiled")

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
        return z.to(policy.compute_dtype)
