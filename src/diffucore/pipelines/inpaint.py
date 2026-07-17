"""Inpainting pipeline (masked image-to-image).

Same shape as img2img — encode the init image, noise it, denoise — but a
:class:`~diffucore.sampling.MaskedDenoiser` pins the keep region to the original
latent so only the masked region is repainted (no sampler changes; see that
class). After decoding, the original pixels are composited back into the keep
region so untouched areas are byte-exact (the VAE round-trip never touches them).

Mask convention: **white (255) = repaint, black (0) = keep** — the common A1111 /
diffusers convention.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from PIL import Image

from ..runtime import perf_context
from ..sampling import MaskedDenoiser
from ._base import PipelineInfo, _Pipeline, img2img_start, preprocess_image  # noqa: F401  (re-export)


def preprocess_mask(mask: Image.Image, width: int, height: int) -> torch.Tensor:
    """PIL mask -> ``FloatTensor[1, 1, height//8, width//8]`` in ``[0, 1]`` (white=1
    repaint, black=0 keep), downsampled to latent resolution for the masked
    denoiser. Bilinear so the latent-space boundary blends over the 8 px grid."""
    mask = mask.convert("L").resize((width // 8, height // 8), Image.BILINEAR)
    arr = np.asarray(mask, dtype=np.float32) / 255.0  # [h, w]
    return torch.from_numpy(arr)[None, None]  # [1, 1, h, w]


class Inpaint(_Pipeline):
    def __call__(
        self,
        prompt: str,
        init_image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str = "",
        *,
        strength: float = 1.0,
        steps: int = 20,
        cfg_scale: float = 7.0,
        cfg_rescale: float | None = None,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
        curvature: float = 0.25,
        eta_max: float = 1.0,
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        lq_threshold: float = 0.025,
        bm_weight: float = 0.5,
        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
        teacache_thresh: float = 0.0,
        teacache_coefficients: "list[float] | None" = None,
        teacache_forecast: str = "hermite",
        deepcache_interval: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[object], None] | None = None,
        return_info: bool = False,
    ) -> Image.Image:
        """Repaint the white region of ``mask_image`` in ``init_image`` to match
        ``prompt``; the black region is preserved exactly. ``width``/``height``
        default to the model's native resolution (inputs are resized to fit).
        ``strength`` in ``(0, 1]`` sets how far the masked region is renoised
        (1.0 = fully regenerated)."""
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0, 1], got {strength}")
        if self.model.spec.architecture == "anima":
            from ._anima import anima_img2img
            return anima_img2img(
                self.model, prompt, init_image, negative_prompt,
                mask_image=mask_image,
                strength=strength, steps=steps, cfg_scale=cfg_scale,
                width=width or self.model.spec.image_size,
                height=height or self.model.spec.image_size,
                sampler=sampler, scheduler=scheduler, seed=seed,
                cfg_interval_start=cfg_interval_start, cfg_interval_end=cfg_interval_end,
                curvature=curvature, eta_max=eta_max, beta_alpha=beta_alpha,
                beta_beta=beta_beta, lq_threshold=lq_threshold,
                bm_weight=bm_weight, bm_alpha1=bm_alpha1, bm_beta1=bm_beta1,
                bm_alpha2=bm_alpha2, bm_beta2=bm_beta2,
                teacache_thresh=teacache_thresh, teacache_coefficients=teacache_coefficients,
                teacache_forecast=teacache_forecast,
                progress_callback=progress_callback, preview_callback=preview_callback,
                return_info=return_info,
            )
        if self.model.spec.architecture in ("flux1", "flux2"):
            # FLUX is guidance-distilled: cfg_scale is the distilled guidance.
            from ._flux import flux_img2img
            return flux_img2img(
                self.model, prompt, init_image, negative_prompt,
                mask_image=mask_image,
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

            sigmas = self._sigmas(scheduler, steps, device, compute_dtype)
            sigmas = sigmas[img2img_start(steps, strength):]
            # Denoiser built after the strength slice so the guidance-interval
            # fractions span the steps that actually run.
            cfg = self._denoiser(cond, uncond, cfg_scale, cfg_rescale, sigmas=sigmas,
                                 cfg_interval=(cfg_interval_start, cfg_interval_end))

            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)
            z0 = self._encode_image(init_image, width, height, policy, generator)
            noise = torch.randn(z0.shape, generator=generator, device=device, dtype=compute_dtype)
            x = z0 + noise * sigmas[0]

            mask = preprocess_mask(mask_image, width, height).to(device, compute_dtype)
            masked = MaskedDenoiser(cfg, z0, mask)

            x0 = self._sample(sampler, masked, x, sigmas, policy, progress_callback,
                              preview_callback, deepcache_interval)
            image, vae_decode_mode = self._decode(x0, policy, width, height)
            image = self._composite(image, init_image, mask_image, width, height)
            info = PipelineInfo(vae_decode_mode=vae_decode_mode)
            return (image, info) if return_info else image

    @staticmethod
    def _composite(generated, init_image, mask_image, width, height) -> Image.Image:
        """Paste the original pixels back into the keep region (hard mask edge) so
        untouched areas are byte-exact, independent of the VAE round-trip."""
        keep = np.asarray(mask_image.convert("L").resize((width, height), Image.NEAREST)) < 128
        original = np.asarray(init_image.convert("RGB").resize((width, height), Image.LANCZOS))
        out = np.where(keep[..., None], original, np.asarray(generated))
        return Image.fromarray(out.astype(np.uint8))
