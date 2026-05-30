"""Sampling-time sigma schedules.

A *schedule* chooses the decreasing sequence of noise levels (sigmas) the
sampler walks down, from a high ``sigma_max`` to ``sigma_min`` and finally to 0
(the clean sample). All schedules here return a 1-D tensor of length
``steps + 1``, descending, with a trailing ``0.0`` appended.

References:
    Karras, Aittala, Aila, Laine. "Elucidating the Design Space of Diffusion-
    Based Generative Models." NeurIPS 2022 (the ``karras`` schedule, eq. 5).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "append_zero",
    "karras_schedule",
    "exponential_schedule",
    "polyexponential_schedule",
]


def append_zero(sigmas: torch.Tensor) -> torch.Tensor:
    """Append a trailing ``0.0`` (the fully-denoised endpoint) to a sigma run."""
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def karras_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Karras et al. (2022) schedule.

    Interpolates linearly in ``sigma ** (1/rho)`` between ``sigma_max`` and
    ``sigma_min``. ``rho=7`` is the paper's default and front-loads steps at low
    noise. Returns ``steps + 1`` sigmas (the last is 0).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


def exponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Geometric schedule: evenly spaced in ``log(sigma)`` from max to min."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), steps, device=device, dtype=dtype
    ).exp()
    return append_zero(sigmas)


def polyexponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Polynomial-in-log schedule. ``rho=1`` reduces to :func:`exponential_schedule`;
    larger ``rho`` concentrates steps toward low noise."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ramp = torch.linspace(1, 0, steps, device=device, dtype=dtype) ** rho
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    sigmas = (ramp * (log_max - log_min) + log_min).exp()
    return append_zero(sigmas)
