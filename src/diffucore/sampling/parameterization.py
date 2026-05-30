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

:class:`FlowMatchingConstScaling` covers rectified-flow models that use the
CONST convention — the model sees the raw noisy latent (no input scaling) and
predicts a velocity ``v = ε - x0`` so that ``x_t = (1 − σ)·x0 + σ·ε``. Anima
(Cosmos-Predict2), Flux, and SD3 all share this form. The sigma value here
plays the role of the rectified-flow timestep ``t ∈ (0, 1]``; the existing
Euler/Heun samplers integrate the ODE exactly (one step is closed-form for
linear interpolation).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "append_dims",
    "make_betas",
    "rescale_zero_terminal_snr",
    "DiscreteSchedule",
    "Scaling",
    "EpsScaling",
    "VScaling",
    "FlowMatchingConstScaling",
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


def rescale_zero_terminal_snr(sigmas: torch.Tensor) -> torch.Tensor:
    """Rescale an ascending sigma table to (near-)zero terminal SNR.

    Lin et al. (2024): the default SD schedule leaves a little signal at the last
    timestep, so the model never sees pure noise and can't render very dark or
    bright images. This shifts/scales ``alphas_cumprod`` so its terminal value is
    ~0 (``sigma_max`` → large) while the first entry (``sigma_min``) is preserved.
    The terminal is clamped to a tiny epsilon rather than exactly 0 so
    ``sigma_max`` is large-but-finite instead of ``inf``. Requires v-prediction —
    eps-prediction is ill-conditioned as ``sigma → ∞``.
    """
    alphas_cumprod = 1.0 / (sigmas ** 2 + 1.0)
    sqrt_acp = alphas_cumprod.sqrt()
    first, last = sqrt_acp[0].clone(), sqrt_acp[-1].clone()
    sqrt_acp = (sqrt_acp - last) * (first / (first - last))  # 0 at the terminal, first preserved
    alphas_cumprod = sqrt_acp ** 2
    alphas_cumprod[-1] = 4.8973451890853435e-08              # avoid sigma_max = inf
    return ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()


class DiscreteSchedule:
    """Per-timestep sigma table derived from training betas, plus sigma<->t.

    ``sigmas`` is ascending (index 0 = least noise). ``sigma_to_t`` /
    ``t_to_sigma`` interpolate linearly in ``log(sigma)`` over the table, which
    lets a continuous sampler address a model trained on discrete timesteps.
    ``zero_terminal_snr`` rescales the table for ZTSNR checkpoints.
    """

    def __init__(self, betas: torch.Tensor, zero_terminal_snr: bool = False):
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt().float()
        if zero_terminal_snr:
            sigmas = rescale_zero_terminal_snr(sigmas)
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


# sigma**2 is computed in fp32 even when the sampler runs fp16: ZTSNR pushes
# sigma_max to ~4500, and 4500**2 overflows fp16's 65504 ceiling to inf (which
# would zero out c_in/c_skip and collapse the latent). Coefficients are cast back
# to the input dtype so the fp16 path is otherwise unchanged.
class EpsScaling(Scaling):
    """Epsilon-prediction (the model predicts the added noise). Used by SD1.5."""

    def scalings(self, sigma: torch.Tensor):
        sigma = torch.as_tensor(sigma)
        s = sigma.float()
        c_in = (1.0 / (s ** 2 + 1.0).sqrt()).to(sigma.dtype)
        c_out = -sigma
        c_skip = torch.ones_like(sigma)
        return c_skip, c_out, c_in


class VScaling(Scaling):
    """v-prediction (Salimans & Ho, 2022)."""

    def scalings(self, sigma: torch.Tensor):
        sigma = torch.as_tensor(sigma)
        s = sigma.float()
        denom = s ** 2 + 1.0
        c_in = (1.0 / denom.sqrt()).to(sigma.dtype)
        c_skip = (1.0 / denom).to(sigma.dtype)
        c_out = (-s / denom.sqrt()).to(sigma.dtype)
        return c_skip, c_out, c_in


class FlowMatchingConstScaling(Scaling):
    """CONST-style rectified flow (Anima / Flux / SD3 convention).

    Forward process:    ``x_t = (1 − σ)·x0 + σ·ε``   with ``σ ∈ (0, 1]``.
    Model prediction:   the velocity ``v = ε − x0``.
    Inversion:          ``x0 = x_t − σ·v``.

    In Karras' preconditioning form that becomes ``c_skip = 1``, ``c_out = −σ``,
    ``c_in = 1`` (the model sees the raw noisy latent, not a rescaled one).
    Note that ``σ`` here is the *rectified-flow time* rather than a noise std;
    the existing σ-space samplers nonetheless integrate the ODE exactly because
    ``d = (x − x0)/σ = v`` for this parameterization.
    """

    def scalings(self, sigma: torch.Tensor):
        sigma = torch.as_tensor(sigma)
        c_in = torch.ones_like(sigma)
        c_skip = torch.ones_like(sigma)
        c_out = -sigma
        return c_skip, c_out, c_in
