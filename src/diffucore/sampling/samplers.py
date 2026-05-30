"""Sigma-space samplers (the denoising loop).

A sampler walks a latent ``x`` down a descending sigma schedule, calling a
denoiser ``model(x, sigma) -> x0_estimate`` at each step and integrating the
probability-flow ODE

    dx/dsigma = (x - x0(x, sigma)) / sigma

toward sigma = 0. Samplers know nothing about models, text, or VAEs; they only
need the denoiser callable and the schedule. ``model`` is called with ``sigma``
broadcast to the batch dimension.

References:
    Karras et al. (2022) for the ODE form and Euler/Heun (Algorithm 2);
    Euler-ancestral follows the DDPM-style ancestral step (Ho et al., 2020) as
    popularized by Katherine Crowson's k-diffusion.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .parameterization import append_dims

__all__ = [
    "to_d",
    "get_ancestral_step",
    "sample_euler",
    "sample_heun",
    "sample_euler_ancestral",
    "get_sampler",
    "SAMPLERS",
]

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Callback = Optional[Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], None]]


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """ODE derivative dx/dsigma = (x - x0) / sigma (Karras et al., 2022)."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float = 1.0):
    """Split a step into a deterministic part (``sigma_down``) and the std of
    fresh noise to re-inject (``sigma_up``). ``eta=0`` recovers a deterministic
    step; ``eta=1`` is fully ancestral."""
    if eta == 0 or bool(sigma_to == 0):
        return sigma_to, torch.zeros_like(sigma_to)
    var = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2).clamp(min=0)
    sigma_up = torch.minimum(sigma_to, eta * var.sqrt())
    sigma_down = (sigma_to ** 2 - sigma_up ** 2).clamp(min=0).sqrt()
    return sigma_down, sigma_up


def sample_euler(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None) -> torch.Tensor:
    """First-order (Euler) ODE sampler."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        x = x + d * (sigma_next - sigma)
    return x


def sample_heun(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None) -> torch.Tensor:
    """Second-order (Heun / trapezoidal) ODE sampler: two evaluations per step,
    averaging the derivative for higher accuracy."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        dt = sigma_next - sigma
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = x + d * dt
        else:
            x_pred = x + d * dt
            denoised_2 = model(x_pred, sigma_next * s_in)
            d_2 = to_d(x_pred, sigma_next * s_in, denoised_2)
            x = x + 0.5 * (d + d_2) * dt
    return x


def sample_euler_ancestral(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
) -> torch.Tensor:
    """Euler sampler with ancestral (stochastic) noise re-injection."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        x = x + d * (sigma_down - sigma)
        if bool(sigma_up > 0):
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = x + noise * sigma_up
    return x


SAMPLERS: dict[str, Denoiser] = {
    "euler": sample_euler,
    "heun": sample_heun,
    "euler_ancestral": sample_euler_ancestral,
}


def get_sampler(name: str):
    """Look up a sampler function by name (see :data:`SAMPLERS`)."""
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
