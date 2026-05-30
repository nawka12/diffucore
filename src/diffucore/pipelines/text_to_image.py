"""Text-to-image pipeline.

A thin wrapper over the shared pipeline machinery in :mod:`._base`: build the
conditioning, sample a fresh-noise latent down the full sigma schedule, decode.
Conditioning / sampling / decode (and their offload + tiling placement) live in
:class:`._base._Pipeline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image

from ._base import _Pipeline

if TYPE_CHECKING:  # avoid importing the bundle (and torch-heavy models) eagerly
    from ..bundle import ModelBundle


class TextToImage(_Pipeline):
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        steps: int = 20,
        cfg_scale: float = 7.0,
        cfg_rescale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
    ) -> Image.Image:
        """Return a ``PIL.Image`` for ``prompt``. ``width``/``height`` default to
        the model's native resolution (512 for SD1.5, 1024 for SDXL). ``cfg_rescale``
        defaults to 0.7 for ZTSNR checkpoints, 0 otherwise (see ``CFGDenoiser``)."""
        model = self.model
        policy = self._policy()
        device, compute_dtype = policy.device, policy.compute_dtype
        if width is None:
            width = model.spec.image_size
        if height is None:
            height = model.spec.image_size

        cond, uncond = self._encode_prompts(prompt, negative_prompt, width, height, policy)
        cfg = self._denoiser(cond, uncond, cfg_scale, cfg_rescale)
        sigmas = self._sigmas(scheduler, steps, device, compute_dtype)

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        x = torch.randn(
            1, 4, height // 8, width // 8,
            generator=generator, device=device, dtype=compute_dtype,
        ) * sigmas[0]

        x0 = self._sample(sampler, cfg, x, sigmas, policy)
        return self._decode(x0, policy, width, height)
