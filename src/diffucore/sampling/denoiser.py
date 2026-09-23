"""Denoisers: adapt a raw diffusion backbone into the ``(x, sigma) -> x0``
callable that samplers consume.

``ModelDenoiser`` applies the sigma<->t mapping and the prediction scalings so a
backbone that predicts epsilon (or v) becomes a clean-sample estimator.
``CFGDenoiser`` layers classifier-free guidance on top by evaluating the
conditioned and unconditioned predictions and extrapolating between them.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from .parameterization import DiscreteSchedule, Scaling


def guidance_interval_bounds(sigmas, start: float, end: float) -> tuple[float, float]:
    """Map a step-fraction guidance interval (Kynkäänniemi et al., 2024) onto
    sigma bounds: CFG is active for steps ``start*n <= i < end*n``, returned as
    ``(lo, hi)`` with CFG on while ``lo < sigma <= hi`` (so mid-step evaluations
    land in the right band). ``(0, 1)`` gives ``(-inf, inf)``.
    """
    if not 0.0 <= start < end <= 1.0:
        raise ValueError(f"need 0 <= start < end <= 1; got {start}, {end}")
    n = len(sigmas) - 1
    hi = math.inf if start <= 0.0 else float(sigmas[min(n, round(start * n))])
    lo = -math.inf if end >= 1.0 else float(sigmas[min(n, round(end * n))])
    return lo, hi


class ModelDenoiser:
    """Wrap a backbone into an x0 estimator in sigma space.

    ``backbone(model_input, t, **cond) -> prediction`` predicts epsilon or v.
    ``__call__(x, sigma, **cond)`` returns the estimated clean latent x0.
    """

    def __init__(self, backbone: Callable[..., torch.Tensor], scaling: Scaling, schedule: DiscreteSchedule):
        self.backbone = backbone
        self.scaling = scaling
        self.schedule = schedule

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **cond: Any) -> torch.Tensor:
        t = self.schedule.sigma_to_t(sigma)
        model_input = self.scaling.model_input(x, sigma)
        prediction = self.backbone(model_input, t, **cond)
        return self.scaling.denoise(x, sigma, prediction)


class CFGDenoiser:
    """Classifier-free guidance: ``x0 = x0_uncond + scale * (x0_cond -
    x0_uncond)``, with cond/uncond batched into one backbone forward.

    ``rescale`` in ``(0, 1]`` enables CFG rescale (Lin et al., 2024) against
    over-exposure, mostly on zero-terminal-SNR models. ``sigma_lo``/``sigma_hi``
    restrict guidance to ``(lo, hi]`` (:func:`guidance_interval_bounds`); outside
    it only the conditioned forward runs.
    """

    def __init__(self, denoiser: ModelDenoiser, cond: dict, uncond: dict, scale: float, rescale: float = 0.0,
                 sigma_lo: float = -math.inf, sigma_hi: float = math.inf):
        self.denoiser = denoiser
        self.cond = cond
        self.uncond = uncond
        self.scale = scale
        self.rescale = rescale
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0 or not (self.sigma_lo < float(sigma.max()) <= self.sigma_hi):
            return self.denoiser(x, sigma, **self.cond)
        # Batch cond+uncond into one forward when values are tensors (the common
        # production case). Fall back to separate forwards for non-tensor values
        # (e.g. test backbones that take scalar kwargs).
        if all(isinstance(v, torch.Tensor) for v in (*self.cond.values(), *self.uncond.values())):
            x0_2 = self.denoiser(
                torch.cat([x, x]),
                torch.cat([sigma, sigma]),
                **{k: torch.cat([self.cond[k], self.uncond[k]]) for k in self.cond},
            )
            x0_cond, x0_uncond = x0_2.chunk(2)
        else:
            x0_cond = self.denoiser(x, sigma, **self.cond)
            x0_uncond = self.denoiser(x, sigma, **self.uncond)
        x0_cfg = x0_uncond + self.scale * (x0_cond - x0_uncond)
        if self.rescale == 0.0:
            return x0_cfg
        dims = tuple(range(1, x0_cfg.ndim))  # per-sample std over C, H, W
        factor = x0_cond.std(dim=dims, keepdim=True) / x0_cfg.std(dim=dims, keepdim=True)
        return self.rescale * (x0_cfg * factor) + (1.0 - self.rescale) * x0_cfg


class MaskedDenoiser:
    """Pin the keep region (``mask == 0``) of the x0 estimate to the original
    latent ``z0`` for inpainting. With a constant target the ODE solution is
    ``x = z0 + noise * sigma``, so the keep region lands on ``z0`` without any
    sampler changes. ``mask`` broadcasts to ``x``; soft values blend linearly.
    """

    def __init__(self, denoiser, z0: torch.Tensor, mask: torch.Tensor):
        self.denoiser = denoiser
        self.z0 = z0
        self.mask = mask

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        denoised = self.denoiser(x, sigma)
        return denoised * self.mask + self.z0 * (1 - self.mask)
