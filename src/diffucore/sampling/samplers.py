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

from functools import partial
from typing import Callable, Optional

import numpy as np
import torch

from .parameterization import append_dims

__all__ = [
    "to_d",
    "get_ancestral_step",
    "sample_euler",
    "sample_heun",
    "sample_heunpp2",
    "sample_euler_ancestral",
    "sample_er_sde",
    "sample_dpm_2",
    "sample_dpm_2_ancestral",
    "sample_dpmpp_2s_ancestral",
    "sample_dpmpp_2m",
    "sample_dpmpp_sde",
    "sample_dpmpp_2m_sde",
    "sample_dpmpp_3m_sde",
    "sample_ipndm",
    "sample_ipndm_v",
    "sample_res_multistep",
    "sample_res_multistep_ancestral",
    "sample_gradient_estimation",
    "sample_lms",
    "sample_lcm",
    "sample_ddpm",
    "sample_secant",
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


def _rf_ancestral_step(sigma: torch.Tensor, sigma_next: torch.Tensor, eta: float):
    """Rectified-flow ancestral split (ComfyUI's ``*_RF`` ancestral samplers).

    For CONST / rectified-flow models (``alpha_t = 1 - sigma_t``) the ancestral
    step can't use the VE :func:`get_ancestral_step` variance bookkeeping. Instead
    it shrinks the deterministic target to ``sigma_down = sigma_next·(1 + (sigma_next/sigma
    - 1)·eta)`` and renoises with ``renoise_coeff`` so the marginal at ``sigma_next``
    is preserved. Returns ``(sigma_down, alpha_next, alpha_down, renoise_coeff)``;
    ``eta=0`` ⇒ deterministic (``sigma_down = sigma_next``, ``renoise = 0``)."""
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2).clamp(min=0).sqrt()
    return sigma_down, alpha_next, alpha_down, renoise_coeff


def _offset_first_sigma_for_snr(sigmas: torch.Tensor, model_type: str, shift: float,
                                percent_offset: float = 1e-4) -> torch.Tensor:
    """Nudge a ``flow`` schedule's first sigma off 1.0 (where ``alpha = 1 - sigma``
    is 0 and the half-logSNR is infinite) to ``time_snr_shift(shift, 1 - eps)``,
    matching ComfyUI's ``offset_first_sigma_for_snr``. No-op for ``ve``."""
    if model_type == "flow" and bool(sigmas[0] >= 1):
        sigmas = sigmas.clone()
        t = 1.0 - percent_offset
        sigmas[0] = shift * t / (1.0 + (shift - 1.0) * t)
    return sigmas


def _half_log_snr(sigma: torch.Tensor, model_type: str) -> torch.Tensor:
    """Half-logSNR ``log(alpha_t / sigma_t)``. ``flow``: ``log((1 - sigma)/sigma)``;
    ``ve``: ``log(1/sigma)`` (alpha == 1)."""
    if model_type == "flow":
        return sigma.logit().neg()
    return sigma.log().neg()


def _sigma_from_half_log_snr(lam: torch.Tensor, model_type: str) -> torch.Tensor:
    """Inverse of :func:`_half_log_snr`."""
    if model_type == "flow":
        return lam.neg().sigmoid()
    return lam.neg().exp()


def _noise_like(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


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
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """Euler sampler with ancestral (stochastic) noise re-injection.

    ``model_type="flow"`` uses the rectified-flow ancestral step (ComfyUI's
    ``sample_euler_ancestral_RF``), required for CONST models like Anima/FLUX;
    ``"ve"`` uses the standard sigma-space ancestral split. ``shift`` is accepted
    for pipeline kwarg uniformity (the Euler-ancestral step needs no first-sigma
    offset)."""
    del shift
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if model_type == "flow":
            if bool(sigma_next == 0):
                x = denoised
                continue
            sigma_down, alpha_next, alpha_down, renoise_coeff = _rf_ancestral_step(sigma, sigma_next, eta)
            ratio = sigma_down / sigma
            x = ratio * x + (1.0 - ratio) * denoised
            if eta > 0 and s_noise > 0:
                x = (alpha_next / alpha_down) * x + _noise_like(x, generator) * s_noise * renoise_coeff
        else:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta)
            d = to_d(x, sigma * s_in, denoised)
            x = x + d * (sigma_down - sigma)
            if bool(sigma_up > 0) and s_noise > 0:
                x = x + _noise_like(x, generator) * s_noise * sigma_up
    return x


def _er_sde_snr_terms(sigmas: torch.Tensor, model_type: str, shift: float):
    """Map a sigma schedule to the ER-SDE half-logSNR variables ``(sigmas,
    er_lambda, alpha)`` where ``er_lambda_t = sigma_t / alpha_t``.

    ``"flow"`` (rectified-flow / CONST, e.g. Anima): ``alpha = 1 - sigma`` so
    ``er_lambda = sigma / (1 - sigma)``. The first sigma is offset off 1.0
    (where ``alpha`` would be 0 and ``er_lambda`` infinite) the same way ComfyUI
    does — ``time_snr_shift(shift, 1 - 1e-4)`` — so step 0 is well-defined.

    ``"ve"`` (variance-exploding, SD/SDXL Karras): ``alpha = 1``, ``er_lambda =
    sigma``.
    """
    if model_type == "flow":
        sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
        alpha = 1.0 - sigmas
        er_lambda = sigmas / alpha
    elif model_type == "ve":
        alpha = torch.ones_like(sigmas)
        er_lambda = sigmas.clone()
    else:
        raise ValueError(f"unknown er_sde model_type {model_type!r}; use 've' or 'flow'")
    return sigmas, er_lambda, alpha


def sample_er_sde(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
    s_noise: float = 1.0,
    max_stage: int = 3,
) -> torch.Tensor:
    """Extended Reverse-time SDE solver (ER-SDE-Solver-3), Cui et al. (2023),
    arXiv:2309.06169. A multi-stage stochastic solver: a first-order (Euler)
    update plus second- and third-order corrections built from finite
    differences of the denoiser, with fresh noise re-injected each step.

    Follows the VP/flow-aware formulation in ComfyUI's ``sample_er_sde``: the
    update runs in half-logSNR space (``er_lambda``), so it serves both VE
    (``model_type="ve"``) and rectified-flow (``model_type="flow"``, which also
    uses ``shift`` to offset the first sigma) checkpoints. The denoiser follows
    the usual convention ``model(x, sigma) -> x0_estimate``.
    """
    s_in = x.new_ones([x.shape[0]])

    def noise_scaler(lam: torch.Tensor) -> torch.Tensor:
        return lam * ((lam ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    sigmas, er_lambda, alpha = _er_sde_snr_terms(sigmas, model_type, shift)

    old_denoised = None
    old_denoised_d = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in)
        if callback is not None:
            callback(i, sigmas[i], x, denoised)
        stage_used = min(max_stage, i + 1)
        if bool(sigmas[i + 1] == 0):
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambda[i], er_lambda[i + 1]
            r_alpha = alpha[i + 1] / alpha[i]
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 (Euler) in half-logSNR space.
            x = r_alpha * r * x + alpha[i + 1] * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2 correction.
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambda[i - 1])
                x = x + alpha[i + 1] * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3 correction.
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambda[i - 2]) / 2)
                    x = x + alpha[i + 1] * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
                std = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                x = x + alpha[i + 1] * noise * s_noise * std
        old_denoised = denoised
    return x


def sample_dpm_2(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None) -> torch.Tensor:
    """DPM-Solver-2 (Karras et al. 2022, Algorithm 2): a midpoint method that
    takes one extra denoiser evaluation at the geometric-mean sigma per step.
    Deterministic; model-agnostic (the VE form, as ComfyUI runs it on flow too)."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = x + d * (sigma_next - sigma)
        else:
            sigma_mid = sigma.log().lerp(sigma_next.log(), 0.5).exp()
            x_2 = x + d * (sigma_mid - sigma)
            denoised_2 = model(x_2, sigma_mid * s_in)
            d_2 = to_d(x_2, sigma_mid * s_in, denoised_2)
            x = x + d_2 * (sigma_next - sigma)
    return x


def sample_dpm_2_ancestral(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """Ancestral DPM-Solver-2. ``flow`` uses the rectified-flow ancestral step
    (ComfyUI's ``sample_dpm_2_ancestral_RF``); ``ve`` uses the standard
    sigma-space ancestral split."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if model_type == "flow":
            downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
            sigma_down = sigma_next * downstep_ratio
            alpha_ip1 = 1 - sigma_next
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2).clamp(min=0).sqrt()
        else:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta)
        if callback is not None:
            callback(i, sigma, x, denoised)
        d = to_d(x, sigma * s_in, denoised)
        if bool(sigma_down == 0):
            x = x + d * (sigma_down - sigma)
        else:
            sigma_mid = sigma.log().lerp(sigma_down.log(), 0.5).exp()
            x_2 = x + d * (sigma_mid - sigma)
            denoised_2 = model(x_2, sigma_mid * s_in)
            d_2 = to_d(x_2, sigma_mid * s_in, denoised_2)
            x = x + d_2 * (sigma_down - sigma)
            if s_noise > 0:
                if model_type == "flow":
                    x = (alpha_ip1 / alpha_down) * x + _noise_like(x, generator) * s_noise * renoise_coeff
                else:
                    x = x + _noise_like(x, generator) * s_noise * sigma_up
    return x


def sample_dpmpp_2m(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None) -> torch.Tensor:
    """DPM-Solver++(2M): second-order multistep in logSNR space, reusing the
    previous step's denoiser output (one evaluation per step). Deterministic;
    model-agnostic (ComfyUI's VE data-prediction form, applied to flow as-is)."""
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        if old_denoised is None or bool(sigma_next == 0):
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


def sample_dpmpp_sde(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """DPM-Solver++ (stochastic, single-step 2nd order). Flow-aware via the
    half-logSNR mapping; noise is seeded Gaussian (not a Brownian tree)."""
    if len(sigmas) <= 1:
        return x
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigma_fn = lambda lam: _sigma_from_half_log_snr(lam, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            lambda_s_1 = lambda_s + r * h
            fac = 1 / (2 * r)
            sigma_s_1 = sigma_fn(lambda_s_1)
            alpha_s = sigma * lambda_s.exp()
            alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
            alpha_t = sigma_next * lambda_t.exp()

            # Step 1 (to the intermediate point)
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp(), eta)
            h_ = sd.log().neg() - lambda_s
            x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
            if eta > 0 and s_noise > 0:
                x_2 = x_2 + alpha_s_1 * _noise_like(x, generator) * s_noise * su
            denoised_2 = model(x_2, sigma_s_1 * s_in)

            # Step 2
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp(), eta)
            h_ = sd.log().neg() - lambda_s
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * _noise_like(x, generator) * s_noise * su
    return x


def sample_dpmpp_2m_sde(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    solver_type: str = "midpoint",
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """DPM-Solver++(2M) SDE: second-order multistep with stochastic noise
    re-injection. Flow-aware; ``solver_type`` is ``"midpoint"`` or ``"heun"``."""
    if len(sigmas) <= 1:
        return x
    if solver_type not in ("heun", "midpoint"):
        raise ValueError("solver_type must be 'heun' or 'midpoint'")
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    old_denoised = None
    h, h_last = None, None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            if old_denoised is not None:
                rr = h_last / h
                if solver_type == "heun":
                    x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / rr) * (denoised - old_denoised)
                else:
                    x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / rr) * (denoised - old_denoised)
            if eta > 0 and s_noise > 0:
                x = x + _noise_like(x, generator) * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
    return x


def sample_dpmpp_3m_sde(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """DPM-Solver++(3M) SDE: third-order multistep with stochastic noise
    re-injection (falls back to 2M, then 1st order, on the first steps).
    Flow-aware; noise is seeded Gaussian."""
    if len(sigmas) <= 1:
        return x
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                rr = h_1 / h
                d = (denoised - denoised_1) / rr
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + (alpha_t * phi_2) * d
            if eta > 0 and s_noise > 0:
                x = x + _noise_like(x, generator) * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


def sample_secant(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    curvature: float = 0.25,
    s_noise: float = 0.0,
    eps_sigma: float = 1e-8,
) -> torch.Tensor:
    """SECANT: σ-space x0-secant multistep sampler.

    A 2nd-order AB2-style multistep sampler that lives in σ space (no λ change
    of variables). At each step it draws a secant line through the previous and
    current ``x0`` estimates, projects it forward to ``σ_{i+1}``, recovers the
    noise component ``ε`` from the current latent, and reconstructs
    ``x_{i+1} = (1-σ_{i+1})·x0_pred + σ_{i+1}·ε`` — preserving the rectified-flow
    identity exactly.

    The correction is blended with plain Euler by
    ``beta = curvature·(1 − r)·(1 − σ)`` where ``r = |Δσ|/σ``, so dense-step,
    lower-noise regions (where the secant extrapolation is reliable) trust the
    correction, while sparse-step and high-noise regions fall back to Euler.
    ``curvature=0`` recovers Euler exactly. Works on any descending σ schedule.

    ``s_noise > 0`` enables the SDE variant: noise of magnitude
    ``s_noise·σ_next·sqrt(|Δσ|/σ)`` is injected per step.
    """
    s_in = x.new_ones([x.shape[0]])
    old_x0 = None
    old_sigma = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        elif old_x0 is None or bool((sigma - old_sigma).abs() < eps_sigma):
            # Warmup or σ-collision: plain Euler.
            x = x + d * (sigma_next - sigma)
        else:
            # SECANT correction: linearly extrapolate x0(σ) along the secant
            # through (σ_{i-1}, x0_{i-1}) and (σ_i, x0_i), hold ε fixed,
            # reconstruct x at σ_next.
            x0_slope = (denoised - old_x0) / (sigma - old_sigma)
            x0_pred = denoised + x0_slope * (sigma_next - sigma)
            # ε_i = x + (1 − σ_i)·v_i (equals (x − (1−σ)·x0)/σ but no /σ divide).
            eps_i = x + (1.0 - sigma) * d
            x_corrected = (1.0 - sigma_next) * x0_pred + sigma_next * eps_i

            x_euler = x + d * (sigma_next - sigma)

            # The x0 estimate is unreliable at high noise, so extrapolating it
            # there over-corrects. Gate the correction off as σ→1 (pure noise)
            # and ramp it in as σ→0, where x0 is reliable and detail refinement
            # matters. σ∈[0,1] is the rectified-flow parameterization this
            # reconstruction assumes.
            r = ((sigma_next - sigma).abs() / sigma).clamp(0.0, 1.0)
            trust = (1.0 - sigma).clamp(0.0, 1.0)
            beta = float(curvature) * (1.0 - r) * trust
            x = (1.0 - beta) * x_euler + beta * x_corrected

            if s_noise > 0:
                noise = _noise_like(x, generator)
                x = x + s_noise * sigma_next * r.sqrt() * noise
        old_x0 = denoised
        old_sigma = sigma
    return x


def sample_heunpp2(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None) -> torch.Tensor:
    """Heun++ — a higher-order Heun that, away from the schedule endpoint, takes a
    third evaluation and blends the three derivatives with sigma-proportional
    weights. Deterministic; model-agnostic (works in sigma space for VE and flow).
    After the original MIT-licensed sd-webui-samplers-scheduler implementation."""
    s_in = x.new_ones([x.shape[0]])
    s_end = sigmas[-1]
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        dt = sigmas[i + 1] - sigma
        if bool(sigmas[i + 1] == s_end):
            # Last step: plain Euler onto the clean sample.
            x = x + d * dt
        elif bool(sigmas[i + 2] == s_end):
            # Penultimate step: 2nd-order Heun with sigma-weighted derivatives.
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * s_in, model(x_2, sigmas[i + 1] * s_in))
            w = 2 * sigmas[0]
            w2 = sigmas[i + 1] / w
            x = x + (d * (1 - w2) + d_2 * w2) * dt
        else:
            # 3rd-order: extrapolate two extra points and blend all three slopes.
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * s_in, model(x_2, sigmas[i + 1] * s_in))
            x_3 = x_2 + d_2 * (sigmas[i + 2] - sigmas[i + 1])
            d_3 = to_d(x_3, sigmas[i + 2] * s_in, model(x_3, sigmas[i + 2] * s_in))
            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            x = x + ((1 - w2 - w3) * d + w2 * d_2 + w3 * d_3) * dt
    return x


def sample_dpmpp_2s_ancestral(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """DPM-Solver++(2S) ancestral: single-step 2nd-order ancestral sampler (two
    evaluations per step). ``model_type="flow"`` uses ComfyUI's
    ``sample_dpmpp_2s_ancestral_RF`` (the data-prediction update in flow
    half-logSNR space, with the σ=1 singularity guarded); ``"ve"`` uses the
    standard logSNR form. ``shift`` is accepted for kwarg uniformity (unused)."""
    del shift
    s_in = x.new_ones([x.shape[0]])
    if model_type == "flow":
        sigma_fn = lambda lam: (lam.exp() + 1.0) ** -1          # σ from half-logSNR
        lambda_fn = lambda sig: ((1.0 - sig) / sig).log()       # half-logSNR from σ
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma * s_in)
            if callback is not None:
                callback(i, sigma, x, denoised)
            if bool(sigma_next == 0):
                x = denoised
                continue
            sigma_down, alpha_next, alpha_down, renoise_coeff = _rf_ancestral_step(sigma, sigma_next, eta)
            if bool(sigma >= 1):
                sigma_s = torch.full_like(sigma, 0.9999)        # guard log((1-σ)/σ) at σ=1
            else:
                t_i, t_down = lambda_fn(sigma), lambda_fn(sigma_down)
                sigma_s = sigma_fn(t_i + 0.5 * (t_down - t_i))
            ratio_s = sigma_s / sigma
            u = ratio_s * x + (1.0 - ratio_s) * denoised
            denoised_2 = model(u, sigma_s * s_in)
            ratio_down = sigma_down / sigma
            x = ratio_down * x + (1.0 - ratio_down) * denoised_2
            if eta > 0 and s_noise > 0:
                x = (alpha_next / alpha_down) * x + _noise_like(x, generator) * s_noise * renoise_coeff
    else:
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sig: sig.log().neg()
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma * s_in)
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta)
            if callback is not None:
                callback(i, sigma, x, denoised)
            if bool(sigma_down == 0):
                d = to_d(x, sigma * s_in, denoised)
                x = x + d * (sigma_down - sigma)
            else:
                t, t_next = t_fn(sigma), t_fn(sigma_down)
                h = t_next - t
                s = t + 0.5 * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-0.5 * h).expm1() * denoised
                denoised_2 = model(x_2, sigma_fn(s) * s_in)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            if bool(sigma_next > 0) and s_noise > 0:
                x = x + _noise_like(x, generator) * s_noise * sigma_up
    return x


def sample_ipndm(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None,
                 max_order: int = 4) -> torch.Tensor:
    """iPNDM — improved pseudo-numerical (Adams–Bashforth) multistep solver in σ
    space, up to 4th order with the fixed AB coefficients. Deterministic and
    model-agnostic. After the Apache-2.0 zju-pi/diff-sampler implementation."""
    s_in = x.new_ones([x.shape[0]])
    buffer: list[torch.Tensor] = []
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        d = to_d(x, sigma * s_in, denoised)
        order = min(max_order, i + 1)
        dt = sigma_next - sigma
        if bool(sigma_next == 0):
            x = denoised
        elif order == 1:
            x = x + dt * d
        elif order == 2:
            x = x + dt * (3 * d - buffer[-1]) / 2
        elif order == 3:
            x = x + dt * (23 * d - 16 * buffer[-1] + 5 * buffer[-2]) / 12
        else:
            x = x + dt * (55 * d - 59 * buffer[-1] + 37 * buffer[-2] - 9 * buffer[-3]) / 24
        buffer.append(d)
        if len(buffer) > max_order - 1:
            buffer.pop(0)
    return x


def sample_ipndm_v(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None,
                   max_order: int = 4) -> torch.Tensor:
    """iPNDM_v — the variable-step variant of :func:`sample_ipndm`: the AB
    coefficients are recomputed from the actual σ spacing each step (suited to
    non-uniform schedules). Deterministic, model-agnostic. zju-pi/diff-sampler
    (Apache-2.0)."""
    s_in = x.new_ones([x.shape[0]])
    t = sigmas
    buffer: list[torch.Tensor] = []
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        d = to_d(x, sigma * s_in, denoised)
        order = min(max_order, i + 1)
        dt = sigma_next - sigma
        if bool(sigma_next == 0):
            x = denoised
        elif order == 1:
            x = x + dt * d
        elif order == 2:
            h_n = sigma_next - sigma
            h_n_1 = sigma - t[i - 1]
            c1 = (2 + (h_n / h_n_1)) / 2
            c2 = -(h_n / h_n_1) / 2
            x = x + dt * (c1 * d + c2 * buffer[-1])
        elif order == 3:
            h_n = sigma_next - sigma
            h_n_1 = sigma - t[i - 1]
            h_n_2 = t[i - 1] - t[i - 2]
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            c1 = (2 + (h_n / h_n_1)) / 2 + temp
            c2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            c3 = temp * h_n_1 / h_n_2
            x = x + dt * (c1 * d + c2 * buffer[-1] + c3 * buffer[-2])
        else:
            h_n = sigma_next - sigma
            h_n_1 = sigma - t[i - 1]
            h_n_2 = t[i - 1] - t[i - 2]
            h_n_3 = t[i - 2] - t[i - 3]
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            c1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            c2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 \
                - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            c3 = temp1 * h_n_1 / h_n_2 \
                + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            c4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x = x + dt * (c1 * d + c2 * buffer[-1] + c3 * buffer[-2] + c4 * buffer[-3])
        buffer.append(d)
        if len(buffer) > max_order - 1:
            buffer.pop(0)
    return x


def _res_multistep(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta: float,
    s_noise: float,
    generator: Optional[torch.Generator],
    callback: Callback,
    model_type: str,
) -> torch.Tensor:
    """Shared body for :func:`sample_res_multistep` (``eta=0``) and
    :func:`sample_res_multistep_ancestral` (``eta>0``).

    Second-order multistep exponential (RES) solver in data-prediction form,
    Zhang et al. (arXiv:2308.02157), evaluated in the half-logSNR-free
    ``t = -log σ`` space (so it serves VE and flow alike). The ancestral split is
    rectified-flow-aware when ``model_type="flow"`` and VE otherwise."""
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sig: sig.log().neg()
    phi1_fn = lambda t: t.expm1() / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t
    old_denoised = None
    old_sigma_down = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        alpha_next = alpha_down = renoise_coeff = sigma_up = None
        if eta > 0 and bool(sigma_next > 0):
            if model_type == "flow":
                sigma_down, alpha_next, alpha_down, renoise_coeff = _rf_ancestral_step(sigma, sigma_next, eta)
            else:
                sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta)
        else:
            sigma_down = sigma_next
        if bool(sigma_down == 0) or old_denoised is None:
            d = to_d(x, sigma * s_in, denoised)
            x = x + d * (sigma_down - sigma)
        else:
            t, t_old = t_fn(sigma), t_fn(old_sigma_down)
            t_next, t_prev = t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        if eta > 0 and bool(sigma_next > 0) and s_noise > 0:
            noise = _noise_like(x, generator)
            if model_type == "flow":
                x = (alpha_next / alpha_down) * x + noise * s_noise * renoise_coeff
            else:
                x = x + noise * s_noise * sigma_up
        old_denoised = denoised
        old_sigma_down = sigma_down
    return x


def sample_res_multistep(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *,
                         callback: Callback = None) -> torch.Tensor:
    """RES (refined exponential solver), deterministic 2nd-order multistep. See
    :func:`_res_multistep`. Model-agnostic."""
    return _res_multistep(model, x, sigmas, eta=0.0, s_noise=1.0, generator=None,
                          callback=callback, model_type="ve")


def sample_res_multistep_ancestral(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """Ancestral RES multistep solver (stochastic). ``model_type="flow"`` uses
    the rectified-flow ancestral step. See :func:`_res_multistep`. ``shift`` is
    accepted for kwarg uniformity (unused)."""
    del shift
    return _res_multistep(model, x, sigmas, eta=eta, s_noise=s_noise, generator=generator,
                          callback=callback, model_type=model_type)


def sample_gradient_estimation(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *,
                               callback: Callback = None, ge_gamma: float = 2.0) -> torch.Tensor:
    """Gradient-estimation sampler (Liu et al., openreview o2ND9v0CeK): an Euler
    step plus a first-order correction ``(γ-1)·(d_i - d_{i-1})`` from the change in
    the ODE derivative between steps. Deterministic; model-agnostic. ``ge_gamma=1``
    reduces to Euler."""
    s_in = x.new_ones([x.shape[0]])
    old_d = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        dt = sigma_next - sigma
        if bool(sigma_next == 0):
            x = denoised
        else:
            x = x + d * dt
            if old_d is not None:
                x = x + (ge_gamma - 1.0) * (d - old_d) * dt
        old_d = d
    return x


def _linear_multistep_coeff(order: int, t: np.ndarray, i: int, j: int) -> float:
    """Adams–Bashforth coefficient for :func:`sample_lms`: the integral over
    ``[t_i, t_{i+1}]`` of the ``j``-th Lagrange basis polynomial through the last
    ``order`` nodes. Integrated exactly in closed form (numpy polynomials), which
    matches ComfyUI's ``scipy.integrate.quad`` without the scipy dependency."""
    poly = np.polynomial.Polynomial([1.0])
    for k in range(order):
        if k == j:
            continue
        poly = poly * np.polynomial.Polynomial([-t[i - k], 1.0]) / (t[i - j] - t[i - k])
    integ = poly.integ()
    return float(integ(t[i + 1]) - integ(t[i]))


def sample_lms(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None,
               order: int = 4) -> torch.Tensor:
    """LMS — linear multistep (Adams–Bashforth) in σ space, with coefficients
    computed from the actual σ nodes each step. Deterministic; model-agnostic.
    The classic k-diffusion sampler (MIT)."""
    s_in = x.new_ones([x.shape[0]])
    sigmas_np = sigmas.detach().cpu().numpy()
    ds: list[torch.Tensor] = []
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [_linear_multistep_coeff(cur_order, sigmas_np, i, j) for j in range(cur_order)]
            x = x + sum(c * d_ for c, d_ in zip(coeffs, reversed(ds)))
    return x


def sample_lcm(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """LCM (Latent Consistency Model) sampling: each step jumps straight to the x0
    estimate, then — unless it's the last step — re-noises to ``sigma_next``.
    ``model_type="flow"`` re-noises as ``(1-σ)·x0 + σ·ε`` (the rectified-flow
    forward), ``"ve"`` as ``x0 + σ·ε``. ``shift`` accepted for uniformity (unused)."""
    del shift
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        x = denoised
        if bool(sigma_next > 0):
            noise = _noise_like(x, generator)
            if model_type == "flow":
                x = (1.0 - sigma_next) * denoised + sigma_next * noise
            else:
                x = denoised + sigma_next * noise
    return x


def sample_ddpm(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *,
                generator: Optional[torch.Generator] = None, callback: Callback = None) -> torch.Tensor:
    """DDPM ancestral sampling (Ho et al., 2020), expressed in Karras σ space via
    the VP mapping ``alpha_cumprod = 1/(σ²+1)``. Stochastic. For VE/VP (eps/v)
    checkpoints — SD/SDXL — not rectified-flow models."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        noise_pred = (x - denoised) / sigma                       # ε estimate
        x_vp = x / (1.0 + sigma ** 2).sqrt()                      # VE latent → VP latent
        alpha_cumprod = 1.0 / (sigma ** 2 + 1.0)
        alpha_cumprod_prev = 1.0 / (sigma_next ** 2 + 1.0)
        alpha = alpha_cumprod / alpha_cumprod_prev
        x_vp = (1.0 / alpha).sqrt() * (x_vp - (1.0 - alpha) * noise_pred / (1.0 - alpha_cumprod).sqrt())
        if bool(sigma_next > 0):
            std = ((1.0 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt()
            x_vp = x_vp + std * _noise_like(x, generator)
            x = x_vp * (1.0 + sigma_next ** 2).sqrt()             # VP latent → VE latent
        else:
            x = x_vp
    return x


SAMPLERS: dict[str, Denoiser] = {
    "euler": sample_euler,
    "heun": sample_heun,
    "heunpp2": sample_heunpp2,
    "euler_ancestral": sample_euler_ancestral,
    "er_sde": sample_er_sde,
    "dpm_2": sample_dpm_2,
    "dpm_2_ancestral": sample_dpm_2_ancestral,
    "dpmpp_2s_ancestral": sample_dpmpp_2s_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_sde": sample_dpmpp_sde,
    "dpmpp_2m_sde": sample_dpmpp_2m_sde,
    "dpmpp_2m_sde_heun": partial(sample_dpmpp_2m_sde, solver_type="heun"),
    "dpmpp_3m_sde": sample_dpmpp_3m_sde,
    "ipndm": sample_ipndm,
    "ipndm_v": sample_ipndm_v,
    "res_multistep": sample_res_multistep,
    "res_multistep_ancestral": sample_res_multistep_ancestral,
    "gradient_estimation": sample_gradient_estimation,
    "lms": sample_lms,
    "lcm": sample_lcm,
    "ddpm": sample_ddpm,
    "secant": sample_secant,
}


def get_sampler(name: str):
    """Look up a sampler function by name (see :data:`SAMPLERS`)."""
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
