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
    "flow_matching_schedule",
    "flow_karras_schedule",
    "simple_schedule",
    "sgm_uniform_schedule",
    "FlowSamplingView",
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


def flow_matching_schedule(
    steps: int,
    shift: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """SD3-style shifted rectified-flow schedule.

    Anima (Cosmos-Predict2) and Flux sample σ values from
    ``σ(t) = shift·t / (1 + (shift − 1)·t)`` for ``t = (N − i)/N``,
    ``i = 0..N−1`` — descending uniform-in-t with the SD3 shift applied.
    ``shift = 1`` collapses to the plain linear schedule. Higher ``shift``
    concentrates more steps near ``σ = 1`` (where the model spent more
    training compute). A trailing ``0`` is appended for the clean endpoint.

    Anima ships with ``shift = 3.0``; the canonical Flux default is 1.15.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if shift < 1.0:
        raise ValueError("shift must be >= 1")
    # descending uniform t in (0, 1]
    t = torch.arange(steps, 0, -1, device=device, dtype=dtype) / steps
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    return append_zero(sigmas)


def flow_karras_schedule(
    steps: int,
    shift: float = 3.0,
    rho: float = 7.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Karras-ρ-warped shifted rectified-flow schedule.

    Generalizes :func:`flow_matching_schedule`: instead of spacing the flow
    time ``t`` uniformly, it spaces ``t`` with the Karras et al. (2022) ρ-warp
    (uniform in ``t ** (1/rho)``) and then applies Anima's SD3 shift map
    ``σ(t) = shift·t / (1 + (shift − 1)·t)``.

    The ρ-warp concentrates steps toward low ``t`` — i.e. low σ, near the data
    manifold, where the rectified-flow trajectory bends most and single-step
    (Euler) truncation error is largest. ``shift`` still biases toward σ = 1
    (matching where training compute went); ``rho`` is an orthogonal knob for
    sampling-error concentration. ``rho == 1`` recovers
    :func:`flow_matching_schedule` exactly; ``rho > 1`` front-loads detail.

    A strict log-SNR ρ-warp is deliberately *not* used: the pure-noise start
    (σ = 1) is log-SNR −∞, and capping σ_max < 1 to make it finite would break
    the "init is exactly pure noise, no rescale" assumption of the Anima
    sampler. Warping in ``t`` is the bounded equivalent that keeps σ_max == 1.
    A trailing ``0`` is appended for the clean endpoint.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if shift < 1.0:
        raise ValueError("shift must be >= 1")
    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    t_min = 1.0 / steps
    # Karras ρ-warp of t over [t_min, 1]: ramp 0→1 maps t from 1 down to t_min.
    ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
    min_inv_rho = t_min ** (1.0 / rho)
    max_inv_rho = 1.0 ** (1.0 / rho)
    t = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    return append_zero(sigmas)


class FlowSamplingView:
    """Minimal :class:`~diffucore.sampling.parameterization.DiscreteSchedule`-like
    view of a rectified-flow model, so the table/timestep-based schedulers
    (:func:`simple_schedule`, :func:`sgm_uniform_schedule`) work on Anima.

    Mirrors ComfyUI's ``ModelSamplingDiscreteFlow``: a ``multiplier``-entry
    ascending sigma table from ``sigma(t) = shift·t/(1+(shift-1)·t)``, with
    ``sigma_to_t``/``t_to_sigma`` as the (invertible) timestep map ``t·multiplier``.
    """

    def __init__(self, shift: float, *, multiplier: int = 1000,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.shift = shift
        self.multiplier = multiplier
        t = torch.arange(1, multiplier + 1, device=device, dtype=dtype) / multiplier
        self.sigmas = shift * t / (1.0 + (shift - 1.0) * t)  # ascending

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def sigma_to_t(self, sigma) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, dtype=self.sigmas.dtype, device=self.sigmas.device)
        t = sigma / (self.shift - (self.shift - 1.0) * sigma)
        return t * self.multiplier

    def t_to_sigma(self, ts) -> torch.Tensor:
        ts = torch.as_tensor(ts, dtype=self.sigmas.dtype, device=self.sigmas.device)
        t = ts / self.multiplier
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)


def simple_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ComfyUI ``simple``: pick ``steps`` sigmas from the model's (ascending)
    training sigma table at evenly spaced strides from the high-noise end."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    table = schedule.sigmas
    n = len(table)
    ss = n / steps
    idx = torch.tensor([n - 1 - int(x * ss) for x in range(steps)], device=table.device)
    sigmas = table[idx].to(device=device, dtype=dtype)
    return append_zero(sigmas)


def sgm_uniform_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ComfyUI ``sgm_uniform``: ``steps`` sigmas uniform in timestep between
    ``sigma_max`` and ``sigma_min`` (``normal_scheduler`` with ``sgm=True`` —
    ``steps + 1`` timesteps with the last dropped)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = float(schedule.sigma_to_t(schedule.sigma_max))
    end = float(schedule.sigma_to_t(schedule.sigma_min))
    ts = torch.linspace(start, end, steps + 1, device=device, dtype=torch.float32)[:-1]
    sigmas = schedule.t_to_sigma(ts).to(device=device, dtype=dtype)
    return append_zero(sigmas)
