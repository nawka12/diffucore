"""Text-to-image pipeline.

Orchestrates the verified sampling core (already implemented) with the model
components (M4–M6). The denoising loop, schedules, CFG, and parameterizations it
relies on are done and tested; only the model forwards remain. See
``docs/IMPLEMENTATION_SPEC.md`` §Pipeline for the exact wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image

from ..conditioning import Conditioner
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
        width: int = 512,
        height: int = 512,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
    ) -> Image.Image:
        """Return a ``PIL.Image`` for ``prompt``."""
        model = self.model
        param = next(model.backbone.parameters())
        device, compute_dtype = param.device, param.dtype

        scaling = EpsScaling()
        denoiser = ModelDenoiser(model.backbone, scaling, model.schedule)

        conditioner = Conditioner(model.tokenizer, model.text_encoder, clip_skip=1)
        cond = {"context": conditioner(prompt, batch=1)}
        uncond = {"context": conditioner(negative_prompt, batch=1)}
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

        vae_dtype = next(model.vae.parameters()).dtype
        with torch.no_grad():
            x0 = get_sampler(sampler)(cfg, x, sigmas)
            image = model.vae.decode(x0.to(vae_dtype))

        image = ((image.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        array = image[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)
