"""Image-to-image pipeline (latent-init / strength).

Encode an init image to a latent, add noise to it at the level the sampler
expects partway down the schedule, then denoise the rest of the way. ``strength``
sets how far down to start: 1.0 runs the full schedule (init mostly overwritten),
small values keep most of the init image. Conditioning / sampling / decode reuse
the shared :class:`._base._Pipeline` machinery, so offload + tiling apply here too
(the VAE is staged for both the encode and the decode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from ._base import _Pipeline, _staged

if TYPE_CHECKING:  # avoid importing the bundle (and torch-heavy models) eagerly
    from ..bundle import ModelBundle


def img2img_start(steps: int, strength: float) -> int:
    """Index into the full ([steps + 1]) sigma schedule where img2img starts.

    Runs ``int(strength * steps)`` denoising steps (the k-diffusion / A1111
    convention): ``strength=1`` starts at index 0 (the full schedule), smaller
    values start later (fewer steps, more of the init image preserved)."""
    return steps - int(strength * steps)


def preprocess_image(image: Image.Image, width: int, height: int) -> torch.Tensor:
    """PIL image -> ``FloatTensor[1, 3, height, width]`` in ``[-1, 1]``, resized to
    ``(width, height)`` — the input range the VAE encoder expects."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


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
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
    ) -> Image.Image:
        """Return a ``PIL.Image`` derived from ``init_image``. ``width``/``height``
        default to the model's native resolution; ``init_image`` is resized to fit.
        ``strength`` in ``(0, 1]`` controls how much of the init image survives."""
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0, 1], got {strength}")
        model = self.model
        policy = self._policy()
        device, compute_dtype = policy.device, policy.compute_dtype
        if width is None:
            width = model.spec.image_size
        if height is None:
            height = model.spec.image_size

        cond, uncond = self._encode_prompts(prompt, negative_prompt, width, height, policy)
        cfg = self._denoiser(cond, uncond, cfg_scale)

        sigmas = self._sigmas(scheduler, steps, device, compute_dtype)
        sigmas = sigmas[img2img_start(steps, strength):]

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        z0 = self._encode_image(init_image, width, height, policy, generator)
        noise = torch.randn(z0.shape, generator=generator, device=device, dtype=compute_dtype)
        x = z0 + noise * sigmas[0]

        x0 = self._sample(sampler, cfg, x, sigmas, policy)
        return self._decode(x0, policy, width, height)

    def _encode_image(self, init_image, width, height, policy, generator):
        """Encode ``init_image`` to a scaled latent on the compute device, with the
        (fp32) VAE staged onto the GPU when offloading."""
        image = preprocess_image(init_image, width, height).to(policy.device, policy.vae_dtype)
        with torch.no_grad():
            with _staged([self.model.vae], policy.device, policy.offload_idle):
                z = self.model.vae.encode(image, generator=generator)
        return z.to(policy.compute_dtype)
