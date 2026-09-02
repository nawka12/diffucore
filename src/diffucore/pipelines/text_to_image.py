"""Text-to-image pipeline.

A thin wrapper over the shared pipeline machinery in :mod:`._base`: build the
conditioning, sample a fresh-noise latent down the full sigma schedule, decode.
Conditioning / sampling / decode (and their offload + tiling placement) live in
:class:`._base._Pipeline`.

For Anima (a flow-matching DiT with its own VAE, TE, and tokenizer pair) the
call dispatches into :func:`._anima.anima_text_to_image` — that path doesn't
share enough machinery with SD/SDXL to ride on top of ``_Pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from PIL import Image

from ..runtime import perf_context
from ._anima import anima_text_to_image, _ANIMA_SCHEDULERS
from ._flux import flux_text_to_image, _FLUX_SCHEDULERS
from ._base import PipelineInfo, _Pipeline

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
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
        shift: float = 3.0,
        curvature: float = 0.25,
        eta_max: float = 1.0,
        gate_reduce: str = "all",
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        lq_threshold: float = 0.025,
        bm_weight: float = 0.5,
        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
        oss_sigmas: "torch.Tensor | list[float] | None" = None,
        teacache_thresh: float = 0.0,
        teacache_coefficients: "list[float] | None" = None,
        teacache_forecast: str = "hermite",
        teacache_rule: str = "drift",
        teacache_uncond_scale: float = 1.0,
        deepcache_interval: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[object], None] | None = None,
        return_info: bool = False,
    ) -> Image.Image:
        if self.model.spec.architecture == "anima":
            # The default scheduler ("karras") and other SD-only schedules don't
            # apply to a flow model; fall back to the rectified-flow schedule.
            anima_scheduler = scheduler if scheduler in _ANIMA_SCHEDULERS else "flow"
            return anima_text_to_image(
                self.model, prompt, negative_prompt,
                steps=steps, cfg_scale=cfg_scale, shift=shift,
                width=width if width is not None else self.model.spec.image_size,
                height=height if height is not None else self.model.spec.image_size,
                seed=seed, sampler=sampler, scheduler=anima_scheduler,
                cfg_interval_start=cfg_interval_start, cfg_interval_end=cfg_interval_end,
                curvature=curvature, eta_max=eta_max, gate_reduce=gate_reduce,
                beta_alpha=beta_alpha,
                beta_beta=beta_beta, lq_threshold=lq_threshold,
                bm_weight=bm_weight, bm_alpha1=bm_alpha1, bm_beta1=bm_beta1,
                bm_alpha2=bm_alpha2, bm_beta2=bm_beta2,
                oss_sigmas=oss_sigmas, teacache_thresh=teacache_thresh,
                teacache_coefficients=teacache_coefficients,
                teacache_forecast=teacache_forecast,
                teacache_rule=teacache_rule,
                teacache_uncond_scale=teacache_uncond_scale,
                progress_callback=progress_callback,
                preview_callback=preview_callback,
                return_info=return_info,
            )
        if self.model.spec.architecture in ("flux1", "flux2"):
            # FLUX is flow-matching + guidance-distilled: cfg_scale is the distilled
            # guidance, and SD schedulers don't apply (fall back to "flux").
            flux_scheduler = scheduler if scheduler in _FLUX_SCHEDULERS else "flux"
            return flux_text_to_image(
                self.model, prompt, negative_prompt,
                steps=steps, guidance=cfg_scale, shift=shift,
                width=width if width is not None else self.model.spec.image_size,
                height=height if height is not None else self.model.spec.image_size,
                seed=seed, sampler=sampler, scheduler=flux_scheduler,
                progress_callback=progress_callback,
                preview_callback=preview_callback,
                return_info=return_info,
            )
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

        with perf_context(policy):
            cond, uncond = self._encode_prompts(prompt, negative_prompt, width, height, policy)
            sigmas = self._sigmas(scheduler, steps, device, compute_dtype)
            cfg = self._denoiser(cond, uncond, cfg_scale, cfg_rescale, sigmas=sigmas,
                                 cfg_interval=(cfg_interval_start, cfg_interval_end))

            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)
            x = torch.randn(
                1, 4, height // 8, width // 8,
                generator=generator, device=device, dtype=compute_dtype,
            ) * sigmas[0]

            x0 = self._sample(sampler, cfg, x, sigmas, policy, progress_callback,
                              preview_callback, deepcache_interval)
            image, vae_decode_mode = self._decode(x0, policy, width, height)
            info = PipelineInfo(vae_decode_mode=vae_decode_mode)
            return (image, info) if return_info else image
