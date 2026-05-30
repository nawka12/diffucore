"""Text-to-image pipeline.

Orchestrates the verified sampling core (already implemented) with the model
components (M4–M6). The denoising loop, schedules, CFG, and parameterizations it
relies on are done and tested; only the model forwards remain. See
``docs/IMPLEMENTATION_SPEC.md`` §Pipeline for the exact wiring.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING

import torch
from PIL import Image

from ..conditioning import Conditioner, SDXLConditioner
from ..models.unet import timestep_embedding
from ..runtime import DevicePolicy, on_device, tiled_vae_decode
from ..sampling import (
    CFGDenoiser,
    EpsScaling,
    ModelDenoiser,
    exponential_schedule,
    get_sampler,
    karras_schedule,
    polyexponential_schedule,
)

if TYPE_CHECKING:  # avoid importing the bundle (and torch-heavy models) eagerly
    from ..bundle import ModelBundle

_SCHEDULERS = {
    "karras": karras_schedule,
    "exponential": exponential_schedule,
    "polyexponential": polyexponential_schedule,
}


@contextmanager
def _staged(modules, device, offload):
    """Bring ``modules`` onto ``device`` for the duration when ``offload`` is on,
    parking them back on CPU afterward. A no-op when offload is off (modules are
    already resident — never touch them)."""
    if not offload:
        yield
        return
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(on_device(module, device))
        yield


class TextToImage:
    def __init__(self, model: "ModelBundle"):
        self.model = model

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        steps: int = 20,
        cfg_scale: float = 7.0,
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
    ) -> Image.Image:
        """Return a ``PIL.Image`` for ``prompt``. ``width``/``height`` default to
        the model's native resolution (512 for SD1.5, 1024 for SDXL)."""
        model = self.model
        policy = model.policy or self._fallback_policy(model)
        device, compute_dtype = policy.device, policy.compute_dtype
        if width is None:
            width = model.spec.image_size
        if height is None:
            height = model.spec.image_size

        scaling = EpsScaling()
        denoiser = ModelDenoiser(model.backbone, scaling, model.schedule)

        with _staged(self._text_modules(model), device, policy.offload):
            cond, uncond = self._conditioning(prompt, negative_prompt, width, height, device)
        cfg = CFGDenoiser(denoiser, cond, uncond, scale=cfg_scale)

        try:
            schedule_fn = _SCHEDULERS[scheduler]
        except KeyError:
            raise ValueError(f"unknown scheduler {scheduler!r}; available: {sorted(_SCHEDULERS)}") from None
        sigmas = schedule_fn(
            steps,
            model.schedule.sigma_min.item(),
            model.schedule.sigma_max.item(),
            device=device,
            dtype=compute_dtype,
        )

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        x = torch.randn(
            1, 4, height // 8, width // 8,
            generator=generator, device=device, dtype=compute_dtype,
        ) * sigmas[0]

        with torch.no_grad():
            with _staged([model.backbone], device, policy.offload):
                x0 = get_sampler(sampler)(cfg, x, sigmas)
            latent = x0.to(policy.vae_dtype)
            tile = policy.vae_tile or max(width, height) >= policy.vae_tile_threshold
            with _staged([model.vae], device, policy.offload):
                image = tiled_vae_decode(model.vae, latent) if tile else model.vae.decode(latent)

        image = ((image.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        array = image[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)

    @staticmethod
    def _text_modules(model):
        """The text encoder(s) resident during the conditioning stage."""
        mods = [model.text_encoder]
        if model.text_encoder_2 is not None:
            mods.append(model.text_encoder_2)
        return mods

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
