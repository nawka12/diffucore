"""Model parameterization: the bridge between a diffusion model's discrete
training schedule and the continuous sigma space samplers work in.

A variance-preserving model is trained with a noise schedule given by ``betas``
(Ho et al., 2020), from which ``alphas_cumprod`` follow. The Karras framework
expresses the same noise level as a standard deviation ``sigma``:

    sigma(t) = sqrt((1 - alphas_cumprod[t]) / alphas_cumprod[t])

:class:`DiscreteSchedule` builds that table and converts between ``sigma`` and
the (continuous) timestep the backbone expects.

A *parameterization* (:class:`Scaling`) says how to turn a raw model prediction
at noise level ``sigma`` into an estimate of the clean sample ``x0``:

    x0 = c_skip * x + c_out * model(c_in * x, t(sigma))

:class:`EpsScaling` covers epsilon-prediction models (e.g. SD1.5);
:class:`VScaling` covers v-prediction models. The scaling coefficients are the
preconditioning of Karras et al. (2022); the v parameterization follows Salimans
& Ho (2022).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "append_dims",
    "make_betas",
    "DiscreteSchedule",
    "Scaling",
    "EpsScaling",
    "VScaling",
]


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Right-pad ``x`` with singleton dims until it has ``target_ndim`` dims,
    so a per-sample ``sigma`` broadcasts against an ``[N, C, H, W]`` latent."""
    pad = target_ndim - x.ndim
    if pad < 0:
        raise ValueError(f"x already has more dims ({x.ndim}) than target ({target_ndim})")
    return x[(...,) + (None,) * pad]


def make_betas(
    schedule: str,
    num_timesteps: int,
    *,
    linear_start: float = 0.00085,
    linear_end: float = 0.012,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Training beta schedule.

    ``scaled_linear`` (linear in sqrt-space) is the Stable Diffusion / LDM
    default and the reason for the unusual ``linear_start``/``linear_end``.
    ``cosine`` follows Nichol & Dhariwal (2021).
    """
    if schedule == "scaled_linear":
        return torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=dtype) ** 2
    if schedule == "linear":
        return torch.linspace(linear_start, linear_end, num_timesteps, dtype=dtype)
    if schedule == "cosine":
        s = 0.008
        t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=dtype) / num_timesteps
        acp = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0].clone()
        betas = 1 - acp[1:] / acp[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown beta schedule: {schedule!r}")


class DiscreteSchedule:
    """Per-timestep sigma table derived from training betas, plus sigma<->t.

    ``sigmas`` is ascending (index 0 = least noise). ``sigma_to_t`` /
    ``t_to_sigma`` interpolate linearly in ``log(sigma)`` over the table, which
    lets a continuous sampler address a model trained on discrete timesteps.
    """

    def __init__(self, betas: torch.Tensor):
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt().float()
        self.sigmas = sigmas
        self.log_sigmas = sigmas.log()

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def sigma_to_t(self, sigma) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, dtype=self.log_sigmas.dtype, device=self.log_sigmas.device)
        log_sigma = sigma.reshape(-1).log()
        dists = log_sigma - self.log_sigmas[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = ((low - log_sigma) / (low - high)).clamp(0, 1)
        t = (1 - w) * low_idx.to(w.dtype) + w * high_idx.to(w.dtype)
        return t.reshape(sigma.shape)

    def t_to_sigma(self, t) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32, device=self.log_sigmas.device)
        low_idx = t.floor().long()
        high_idx = t.ceil().long().clamp(max=self.log_sigmas.shape[0] - 1)
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class Scaling:
    """Base prediction parameterization.

    Subclasses implement :meth:`scalings`, returning ``(c_skip, c_out, c_in)``
    shaped like ``sigma``.
    """

    def scalings(self, sigma: torch.Tensor):
        raise NotImplementedError

    def model_input(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Scale the latent before it is fed to the backbone (the ``c_in * x``)."""
        _, _, c_in = self.scalings(sigma)
        return append_dims(c_in, x.ndim) * x

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """Combine the raw model output into a clean-sample (x0) estimate."""
        c_skip, c_out, _ = self.scalings(sigma)
        return append_dims(c_skip, x.ndim) * x + append_dims(c_out, x.ndim) * model_output


class EpsScaling(Scaling):
    """Epsilon-prediction (the model predicts the added noise). Used by SD1.5."""

    def scalings(self, sigma: torch.Tensor):
        sigma = torch.as_tensor(sigma)
        c_in = 1.0 / (sigma ** 2 + 1.0).sqrt()
        c_out = -sigma
        c_skip = torch.ones_like(sigma)
        return c_skip, c_out, c_in


class VScaling(Scaling):
    """v-prediction (Salimans & Ho, 2022)."""

    def scalings(self, sigma: torch.Tensor):
        sigma = torch.as_tensor(sigma)
        denom = sigma ** 2 + 1.0
        c_in = 1.0 / denom.sqrt()
        c_skip = 1.0 / denom
        c_out = -sigma / denom.sqrt()
        return c_skip, c_out, c_in
