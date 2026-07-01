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
    """Map a step-fraction guidance interval onto absolute sigma bounds.

    Applying CFG only in a middle band of the schedule (Kynkäänniemi et al.,
    2024) keeps its quality benefit while skipping the uncond forward outside
    the band. ``start``/``end`` are fractions of the sampling run: CFG is active
    for steps ``start*n <= i < end*n`` of the ``n``-step descending ``sigmas``
    schedule. Returns ``(lo, hi)`` such that CFG applies while ``lo < sigma <=
    hi`` — a sigma-space test, so mid-step evaluations by higher-order samplers
    land in the right band too. ``(0, 1)`` returns ``(-inf, inf)`` (always on).
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
    """Classifier-free guidance around a conditioning-aware denoiser.

        x0 = x0_uncond + scale * (x0_cond - x0_uncond)

    ``scale == 1`` is the conditioned estimate; ``scale == 0`` is unconditioned.
    ``cond`` / ``uncond`` are kwarg dicts forwarded to the underlying denoiser
    (e.g. ``{"context": embeddings}``).

    ``rescale`` in ``(0, 1]`` enables CFG rescale (Lin et al., 2024): the guided
    estimate is renormalized toward the conditioned estimate's per-sample std and
    blended back by ``rescale``, which counteracts the over-exposure high guidance
    causes (especially on zero-terminal-SNR models). ``0`` is plain CFG.

    Cond and uncond are batched into a single backbone forward (tensors stacked
    along the batch axis) so the model sees half as many invocations per step.

    ``sigma_lo``/``sigma_hi`` restrict guidance to sigmas in ``(lo, hi]`` (see
    :func:`guidance_interval_bounds`); outside the band only the conditioned
    forward runs — half the backbone work for those steps. Defaults are
    unbounded (guidance at every step, the previous behavior).
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
    """Pin the keep region of the x0 estimate to the original latent (inpainting).

    Wraps any ``(x, sigma) -> x0`` denoiser. Where ``mask == 0`` (the keep region)
    it overrides the model's estimate with the original latent ``z0``; where
    ``mask == 1`` (the region to repaint) it passes the estimate through. With a
    constant target ``z0``, the sampler's ODE ``dx/dsigma = (x - z0) / sigma`` has
    the exact solution ``x = z0 + noise * sigma`` — which Euler/Heun integrate
    exactly — so the keep region tracks the noised original and lands on ``z0`` at
    ``sigma -> 0``. No sampler changes are needed; the masking lives here.

    ``mask`` is broadcastable to ``x`` (e.g. ``[1, 1, h, w]``) and matches ``x``'s
    dtype/device; soft values in ``[0, 1]`` blend linearly at the boundary.
    """

    def __init__(self, denoiser, z0: torch.Tensor, mask: torch.Tensor):
        self.denoiser = denoiser
        self.z0 = z0
        self.mask = mask

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        denoised = self.denoiser(x, sigma)
        return denoised * self.mask + self.z0 * (1 - self.mask)
