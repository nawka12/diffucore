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

from ..conditioning import Conditioner, SDXLConditioner
from ..models.unet import timestep_embedding
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
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
    ) -> Image.Image:
        """Return a ``PIL.Image`` for ``prompt``. ``width``/``height`` default to
        the model's native resolution (512 for SD1.5, 1024 for SDXL)."""
        model = self.model
        param = next(model.backbone.parameters())
        device, compute_dtype = param.device, param.dtype
        if width is None:
            width = model.spec.image_size
        if height is None:
            height = model.spec.image_size

        scaling = EpsScaling()
        denoiser = ModelDenoiser(model.backbone, scaling, model.schedule)

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

        vae_dtype = next(model.vae.parameters()).dtype
        with torch.no_grad():
            x0 = get_sampler(sampler)(cfg, x, sigmas)
            image = model.vae.decode(x0.to(vae_dtype))

        image = ((image.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        array = image[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)

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
