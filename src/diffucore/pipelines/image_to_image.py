"""Image-to-image pipeline (latent-init / strength).

Encode an init image to a latent, add noise to it at the level the sampler
expects partway down the schedule, then denoise the rest of the way. ``strength``
sets how far down to start: 1.0 runs the full schedule (init mostly overwritten),
small values keep most of the init image. Conditioning / sampling / decode reuse
the shared :class:`._base._Pipeline` machinery, so offload + tiling apply here too
(the VAE is staged for both the encode and the decode).
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image

from ..runtime import perf_context
# ``img2img_start`` / ``preprocess_image`` live in ``._base`` (shared with inpaint);
# re-exported here so they stay importable from this module.
from ._base import PipelineInfo, _Pipeline, img2img_start, preprocess_image  # noqa: F401


class ImageToImage(_Pipeline):
    def __call__(
        self,
        prompt: str,
        init_image: Image.Image,
        negative_prompt: str = "",
        *,
        strength: float = 0.75,
        steps: int = 20,
        cfg_scale: float = 7.0,
        cfg_rescale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[object], None] | None = None,
        return_info: bool = False,
    ) -> Image.Image:
        """Return a ``PIL.Image`` derived from ``init_image``. ``width``/``height``
        default to the model's native resolution; ``init_image`` is resized to fit.
        ``strength`` in ``(0, 1]`` controls how much of the init image survives."""
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0, 1], got {strength}")
        if self.model.spec.architecture == "anima":
            from ._anima import anima_img2img
            return anima_img2img(
                self.model, prompt, init_image, negative_prompt,
                strength=strength, steps=steps, cfg_scale=cfg_scale,
                width=width or self.model.spec.image_size,
                height=height or self.model.spec.image_size,
                sampler=sampler, scheduler=scheduler, seed=seed,
                progress_callback=progress_callback, preview_callback=preview_callback,
                return_info=return_info,
            )
        if self.model.spec.architecture in ("flux1", "flux2"):
            # FLUX is guidance-distilled: cfg_scale is the distilled guidance.
            from ._flux import flux_img2img
            return flux_img2img(
                self.model, prompt, init_image, negative_prompt,
                strength=strength, steps=steps, guidance=cfg_scale,
                width=width or self.model.spec.image_size,
                height=height or self.model.spec.image_size,
                sampler=sampler, scheduler=scheduler, seed=seed,
                progress_callback=progress_callback, preview_callback=preview_callback,
                return_info=return_info,
            )
        model = self.model
        policy = self._policy()
        device, compute_dtype = policy.device, policy.compute_dtype
        if width is None:
            width = model.spec.image_size
        if height is None:
            height = model.spec.image_size

        with perf_context(policy):
            cond, uncond = self._encode_prompts(prompt, negative_prompt, width, height, policy)
            cfg = self._denoiser(cond, uncond, cfg_scale, cfg_rescale)

            sigmas = self._sigmas(scheduler, steps, device, compute_dtype)
            sigmas = sigmas[img2img_start(steps, strength):]

            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)
            z0 = self._encode_image(init_image, width, height, policy, generator)
            noise = torch.randn(z0.shape, generator=generator, device=device, dtype=compute_dtype)
            x = z0 + noise * sigmas[0]

            x0 = self._sample(sampler, cfg, x, sigmas, policy, progress_callback, preview_callback)
            image, vae_decode_mode = self._decode(x0, policy, width, height)
            info = PipelineInfo(vae_decode_mode=vae_decode_mode)
            return (image, info) if return_info else image
