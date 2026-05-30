"""Denoisers: adapt a raw diffusion backbone into the ``(x, sigma) -> x0``
callable that samplers consume.

``ModelDenoiser`` applies the sigma<->t mapping and the prediction scalings so a
backbone that predicts epsilon (or v) becomes a clean-sample estimator.
``CFGDenoiser`` layers classifier-free guidance on top by evaluating the
conditioned and unconditioned predictions and extrapolating between them.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .parameterization import DiscreteSchedule, Scaling


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

    Note: this evaluates the backbone twice per step. Batching the two passes is
    a throughput optimization left for the model-integration milestone.
    """

    def __init__(self, denoiser: ModelDenoiser, cond: dict, uncond: dict, scale: float):
        self.denoiser = denoiser
        self.cond = cond
        self.uncond = uncond
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x0_cond = self.denoiser(x, sigma, **self.cond)
        if self.scale == 1.0:
            return x0_cond
        x0_uncond = self.denoiser(x, sigma, **self.uncond)
        return x0_uncond + self.scale * (x0_cond - x0_uncond)


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
