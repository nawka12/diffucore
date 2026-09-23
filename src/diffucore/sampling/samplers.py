"""Sigma-space samplers (the denoising loop).

A sampler walks a latent ``x`` down a descending sigma schedule, calling a
denoiser ``model(x, sigma) -> x0_estimate`` at each step and integrating the
probability-flow ODE ``dx/dsigma = (x - x0(x, sigma)) / sigma`` toward
sigma = 0. ``model`` is called with ``sigma`` broadcast to the batch dimension.

References: Karras et al. (2022) for the ODE form and Euler/Heun; the ancestral
step follows DDPM (Ho et al., 2020) as popularized by k-diffusion.
"""

from __future__ import annotations

import math
from functools import lru_cache, partial
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .parameterization import append_dims

__all__ = [
    "to_d",
    "get_ancestral_step",
    "sample_euler",
    "sample_heun",
    "sample_heunpp2",
    "sample_euler_ancestral",
    "sample_euler_ancestral_anneal",
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
    "sample_stork2",
    "sample_infinity",
    "sample_infinity_realism",
    "sample_infinity_omega",
    "sample_infinity_aether",
    "sample_lms",
    "sample_lcm",
    "sample_ddpm",
    "sample_sa_solver",
    "sample_sa_solver_pece",
    "sample_secant",
    "sample_secant_anneal",
    "sample_cogent",
    "sample_cogent3",
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
    fresh noise to re-inject (``sigma_up``). ``eta=0`` is deterministic."""
    if eta == 0 or bool(sigma_to == 0):
        return sigma_to, torch.zeros_like(sigma_to)
    var = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2).clamp(min=0)
    sigma_up = torch.minimum(sigma_to, eta * var.sqrt())
    sigma_down = (sigma_to ** 2 - sigma_up ** 2).clamp(min=0).sqrt()
    return sigma_down, sigma_up


def _rf_ancestral_step(sigma: torch.Tensor, sigma_next: torch.Tensor, eta: float):
    """Rectified-flow ancestral split (ComfyUI's ``*_RF`` samplers): shrink the
    target to ``sigma_down`` and renoise so the marginal at ``sigma_next`` is
    preserved. Returns ``(sigma_down, alpha_next, alpha_down, renoise_coeff)``;
    ``eta=0`` is deterministic."""
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2).clamp(min=0).sqrt()
    return sigma_down, alpha_next, alpha_down, renoise_coeff


def _offset_first_sigma_for_snr(sigmas: torch.Tensor, model_type: str, shift: float,
                                percent_offset: float = 1e-4) -> torch.Tensor:
    """Move a ``flow`` schedule's first sigma off 1.0, where the half-logSNR is
    infinite (ComfyUI's ``offset_first_sigma_for_snr``). No-op for ``ve``."""
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
    """Second-order (Heun) ODE sampler, two evaluations per step."""
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
    """Euler with ancestral noise re-injection. ``model_type="flow"`` uses the
    rectified-flow step (ComfyUI's ``sample_euler_ancestral_RF``). ``shift`` is
    accepted for kwarg uniformity."""
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


def sample_euler_ancestral_anneal(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 1.0,
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "flow",
    shift: float = 1.0,
) -> torch.Tensor:
    """Euler-ancestral (flow only) with ``eta_i = eta_max·σ_i``: a stochastic
    burn-in at high σ that averages out an imperfect velocity field, tapering to
    a near-deterministic step as σ→0 so low-σ detail survives. ``shift`` is
    unused."""
    if model_type != "flow":
        raise ValueError("euler_ancestral_anneal is rectified-flow only (model_type='flow')")
    del shift
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
            continue
        eta = eta_max * float(sigma.clamp(max=1.0))
        sigma_down, alpha_next, alpha_down, renoise_coeff = _rf_ancestral_step(sigma, sigma_next, eta)
        ratio = sigma_down / sigma
        x = ratio * x + (1.0 - ratio) * denoised
        if eta > 0 and s_noise > 0:
            x = (alpha_next / alpha_down) * x + _noise_like(x, generator) * s_noise * renoise_coeff
    return x


def _er_sde_snr_terms(sigmas: torch.Tensor, model_type: str, shift: float):
    """ER-SDE half-logSNR variables ``(sigmas, er_lambda, alpha)`` with
    ``er_lambda = sigma / alpha``. ``flow`` has ``alpha = 1 - sigma`` and offsets
    the first sigma off 1.0 like ComfyUI; ``ve`` has ``alpha = 1``.
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
    """ER-SDE-Solver-3 (Cui et al., 2023, arXiv:2309.06169): Euler plus second-
    and third-order finite-difference corrections, with fresh noise each step.
    Runs in half-logSNR space after ComfyUI's ``sample_er_sde``, so it serves
    ``ve`` and ``flow`` models.
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
    """DPM-Solver-2 (Karras et al. 2022): midpoint method with one extra
    evaluation at the geometric-mean sigma. The VE form, run on flow as-is (as
    ComfyUI does)."""
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
    """Ancestral DPM-Solver-2; ``flow`` uses the rectified-flow ancestral step
    (ComfyUI's ``sample_dpm_2_ancestral_RF``)."""
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
    """DPM-Solver++(2M): second-order multistep in logSNR space, one evaluation
    per step. ComfyUI's VE data-prediction form, applied to flow as-is."""
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
    """DPM-Solver++ SDE (single-step 2nd order). Flow-aware via half-logSNR;
    noise is seeded Gaussian, not a Brownian tree."""
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
    """DPM-Solver++(2M) SDE. Flow-aware; ``solver_type`` is ``"midpoint"`` or
    ``"heun"``."""
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
    """DPM-Solver++(3M) SDE (2M, then 1st order, on the first steps). Flow-aware;
    noise is seeded Gaussian."""
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


def sample_exp_heun_2_x0(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
    solver_type: str = "phi_2",
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """Exponential Heun in x0 / half-logSNR form: a DPM-Solver++(1) predictor, a
    second x0 evaluation at sigma_next, and a phi-weighted corrector. Two
    evaluations per step, no history. Flow-aware.

    The deterministic, full-step case of SEEDS (Gonzalez et al., NeurIPS 2023,
    arXiv:2305.14267) on the DPM-Solver++ integrator (arXiv:2211.01095).
    ``solver_type`` ``"phi_2"`` (default) is the phi_2-weighted Heun corrector,
    ``"phi_1"`` the trapezoidal average."""
    if len(sigmas) <= 1:
        return x
    if solver_type not in ("phi_1", "phi_2"):
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
            continue
        lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
        h = lambda_t - lambda_s
        alpha_t = sigma_next * lambda_t.exp()
        phi_1 = (-h).expm1()                       # e^{-h} - 1 (== h·phi_1(-h))
        x_pred = (sigma_next / sigma) * x - alpha_t * phi_1 * denoised
        denoised_2 = model(x_pred, sigma_next * s_in)
        if solver_type == "phi_1":
            denoised_d = 0.5 * (denoised + denoised_2)
            x = (sigma_next / sigma) * x - alpha_t * phi_1 * denoised_d
        else:
            phi_2 = (phi_1 + h) / (-h)             # (e^{-h} - 1 + h)/(-h) (== h·phi_2(-h))
            b2 = phi_2
            b1 = phi_1 - b2
            x = (sigma_next / sigma) * x - alpha_t * (b1 * denoised + b2 * denoised_2)
    return x


def _uni_pc_bh_update(model, x, model_prev, sigma_prev, lambda_prev,
                      sigma_t, lambda_t, s_in, order, variant):
    """One UniPC predictor + corrector step in x0 form; returns ``(x_t, model_t)``.

    ``model_prev`` holds the last ``order`` x0 estimates, newest last. ``model_t``
    (the corrector's evaluation at ``sigma_t``) is reused as the next step's
    newest history. ``variant`` is the ``B(h)`` type (``"bh1"``/``"bh2"``)."""
    device = x.device
    m0 = model_prev[-1]
    sigma_prev_0 = sigma_prev[-1]
    h = lambda_t - lambda_prev[-1]
    alpha_t = sigma_t * lambda_t.exp()

    rks, D1s = [], []
    for i in range(1, order):
        rk = (lambda_prev[-(i + 1)] - lambda_prev[-1]) / h
        rks.append(rk)
        D1s.append((model_prev[-(i + 1)] - m0) / rk)
    rks.append(torch.ones((), device=device, dtype=h.dtype))
    rks = torch.stack(rks)

    hh = -h                                  # x0 form
    h_phi_1 = hh.expm1()                     # e^{hh} - 1 == hh·phi_1(hh)
    h_phi_k = h_phi_1 / hh - 1
    B_h = hh if variant == "bh1" else hh.expm1()

    R, b = [], []
    factorial_i = 1
    for i in range(1, order + 1):
        R.append(rks ** (i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= (i + 1)
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i
    R = torch.stack(R)
    b = torch.stack(b)

    D1s = torch.stack(D1s, dim=1) if D1s else None   # (B, K, *spatial)

    def combine(rhos, D):
        rr = rhos.to(D.dtype).view(1, -1, *([1] * (D.ndim - 2)))
        return (rr * D).sum(dim=1)

    # Predictor.
    x_t_ = (sigma_t / sigma_prev_0) * x - alpha_t * h_phi_1 * m0
    if D1s is not None:
        if order == 2:                       # closed form for 2nd order
            rhos_p = torch.tensor([0.5], device=device, dtype=b.dtype)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        pred_res = combine(rhos_p, D1s)
    else:
        pred_res = 0.0
    x_t = x_t_ - alpha_t * B_h * pred_res

    model_t = model(x_t, sigma_t * s_in)
    if order == 1:
        rhos_c = torch.tensor([0.5], device=device, dtype=b.dtype)
    else:
        rhos_c = torch.linalg.solve(R, b)
    corr_res = combine(rhos_c[:-1], D1s) if D1s is not None else 0.0
    D1_t = model_t - m0
    x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    return x_t, model_t


def sample_uni_pc(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
    order: int = 3,
    variant: str = "bh1",
    lower_order_final: bool = True,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """UniPC multistep predictor-corrector (Zhao et al., NeurIPS 2023,
    arXiv:2302.04867), x0 form. The corrector's evaluation becomes the next
    step's history, so it stays ~one evaluation per step. ``variant`` is
    ``"bh1"`` or ``"bh2"``; ``lower_order_final`` ramps the order down over the
    last steps. Flow-aware."""
    if len(sigmas) <= 1:
        return x
    if variant not in ("bh1", "bh2"):
        raise ValueError("variant must be 'bh1' or 'bh2'")
    if order < 1:
        raise ValueError("order must be >= 1")
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    n = len(sigmas) - 1

    model_prev, sigma_prev, lambda_prev = [], [], []

    def push(sig, m):
        sigma_prev.append(sig)
        lambda_prev.append(lambda_fn(sig))
        model_prev.append(m)
        if len(model_prev) > order:
            sigma_prev.pop(0)
            lambda_prev.pop(0)
            model_prev.pop(0)

    m0 = model(x, sigmas[0] * s_in)
    if callback is not None:
        callback(0, sigmas[0], x, m0)
    push(sigmas[0], m0)

    for step in range(1, n + 1):
        sigma_t = sigmas[step]
        if bool(sigma_t == 0):           # land on the x0 estimate
            x = model_prev[-1]
            break
        cur_order = min(order, len(model_prev))
        if lower_order_final:
            cur_order = min(cur_order, n - step)
        x, model_t = _uni_pc_bh_update(
            model, x, model_prev, sigma_prev, lambda_prev,
            sigma_t, lambda_fn(sigma_t), s_in, cur_order, variant,
        )
        if callback is not None:
            callback(step, sigma_t, x, model_t)
        push(sigma_t, model_t)
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

    Extrapolates x0 along the secant through the previous and current
    estimates, holds ε fixed and reconstructs ``x = (1-σ)·x0 + σ·ε`` at the next
    σ (the rectified-flow identity). The correction is blended with Euler by
    ``beta = curvature·(1 − |Δσ|/σ)·(1 − σ)``, so it only acts on dense,
    low-noise steps; ``curvature=0`` is Euler. ``s_noise > 0`` injects
    ``s_noise·σ_next·sqrt(|Δσ|/σ)`` of noise per step.
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
            x0_slope = (denoised - old_x0) / (sigma - old_sigma)
            x0_pred = denoised + x0_slope * (sigma_next - sigma)
            # ε_i = x + (1 − σ_i)·v_i, i.e. (x − (1−σ)·x0)/σ without the divide.
            eps_i = x + (1.0 - sigma) * d
            x_corrected = (1.0 - sigma_next) * x0_pred + sigma_next * eps_i

            x_euler = x + d * (sigma_next - sigma)

            # x0 is unreliable at high noise: gate the correction off as σ→1.
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


def sample_secant_anneal(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 1.0,
    s_noise: float = 1.0,
    curvature: float = 0.25,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "flow",
    shift: float = 1.0,
) -> torch.Tensor:
    """SECANT-ANNEAL (flow only): σ-annealed ancestral burn-in
    (:func:`sample_euler_ancestral_anneal`) plus the x0-secant correction of
    :func:`sample_secant` toward ``σ_down``.

    ``eta`` is large exactly where the secant weight is ≈0 and vice versa, so
    high σ behaves like ``euler_ancestral_anneal`` and low σ like ``secant``.
    ``curvature=0`` recovers ``euler_ancestral_anneal``; ``eta_max=0`` recovers
    ``secant``. ``shift`` is unused."""
    if model_type != "flow":
        raise ValueError("secant_anneal is rectified-flow only (model_type='flow')")
    del shift
    s_in = x.new_ones([x.shape[0]])
    old_x0 = None
    old_sigma = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            eta = eta_max * float(sigma.clamp(max=1.0))
            sigma_down, alpha_next, alpha_down, renoise_coeff = _rf_ancestral_step(sigma, sigma_next, eta)
            d = to_d(x, sigma * s_in, denoised)
            # Hold x0 fixed (Euler-equivalent) unless a usable secant exists.
            if old_x0 is None or bool((sigma - old_sigma).abs() < 1e-8):
                x0_eff = denoised
            else:
                x0_slope = (denoised - old_x0) / (sigma - old_sigma)
                x0_pred = denoised + x0_slope * (sigma_down - sigma)
                r = ((sigma_down - sigma).abs() / sigma).clamp(0.0, 1.0)
                reliable = (1.0 - sigma).clamp(0.0, 1.0)
                beta = float(curvature) * (1.0 - r) * reliable
                x0_eff = (1.0 - beta) * denoised + beta * x0_pred
            # Reconstruct at σ_down holding ε fixed, then re-noise to σ_next.
            eps_i = x + (1.0 - sigma) * d
            x = (1.0 - sigma_down) * x0_eff + sigma_down * eps_i
            if eta > 0 and s_noise > 0:
                x = (alpha_next / alpha_down) * x + _noise_like(x, generator) * s_noise * renoise_coeff
        old_x0 = denoised
        old_sigma = sigma
    return x


def _validate_gate_reduce(reduce: str) -> None:
    """Validate the reduction mode shared by cogent's two gates."""
    if reduce not in ("all", "per_channel"):
        raise ValueError(f"reduce must be 'all' or 'per_channel', got {reduce!r}")


def _coherence_gate(diff: torch.Tensor, old_diff: Optional[torch.Tensor],
                    h: torch.Tensor, *, reduce: str = "all",
                    stats_out: Optional[dict] = None) -> torch.Tensor:
    """Scale factor for a multistep solver's divided-difference term::

        psi = max( (1 + 2·rho)/3 ,  1 - e^(-h) )        clamped to [0, 1]

    ``rho`` is the cosine of two consecutive x0 differences
    ``D_i = x0_i - x0_{i-1}``. With ``x0_i = f_i + n_i`` (iid noise energy
    ``v``, signal change ``S = ‖Δf‖²``) it reads ``rho = (S - v)/(S + 2v)``, and
    the Wiener shrink ``S/(S + 2v)`` becomes ``(1 + 2·rho)/3``: 1 on a clean
    trajectory, 0 on pure noise. Curvature also lowers ``rho`` exactly when the
    correction is needed, so ``1 - e^(-h)``, the integrator's own phi-weight,
    is the floor. ``old_diff is None`` gives the floor alone.

    ``reduce="all"`` reduces over every non-batch dim; ``"per_channel"`` (the
    default-off cogent4 gate) reduces over ``(H, W)`` of a 4-D latent only and
    falls back to ``"all"`` elsewhere.

    ``stats_out``, if given, receives detached logging tensors: ``rho``, ``d2``,
    ``s_est``, ``v_est``, ``psi_linear``, ``floor_active`` and ``bootstrap``
    (shape ``[B]``, or ``[B, C]`` per channel).
    """
    # Validate before the bootstrap return, or validity would depend on the step.
    _validate_gate_reduce(reduce)
    floor = (-h).expm1().neg()
    if old_diff is None:
        if stats_out is not None:
            stats_out["bootstrap"] = True
            stats_out["floor_active"] = True
        return floor
    if reduce == "per_channel":
        if diff.ndim == 4:
            dims = (2, 3)
        else:
            reduce = "all"                                  # no spatial axes
            dims = tuple(range(1, diff.ndim))
    else:  # reduce == "all"
        dims = tuple(range(1, diff.ndim))
    num = (diff * old_diff).sum(dim=dims)
    d2 = diff.pow(2).sum(dim=dims)
    o2 = old_diff.pow(2).sum(dim=dims)
    den = (d2 * o2).sqrt()
    rho = num / den.clamp_min(torch.finfo(diff.dtype).tiny)
    psi = ((1.0 + 2.0 * rho) / 3.0).clamp(0.0, 1.0)
    if stats_out is not None:
        # Detached so a stats list doesn't retain the denoiser graph.
        stats_out["rho"] = rho.detach()
        stats_out["d2"] = d2.detach()
        stats_out["s_est"] = ((2.0 * num + d2) / 3.0).detach()
        stats_out["v_est"] = ((d2 - num) / 3.0).detach()
        stats_out["psi_linear"] = psi.detach()
        stats_out["floor_active"] = (floor > psi).detach()
        stats_out["bootstrap"] = False
    return torch.maximum(psi.view(*psi.shape, *([1] * (diff.ndim - psi.ndim))), floor)


def sample_cogent(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 1.0,
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
    gate_reduce: str = "all",
) -> torch.Tensor:
    """COGENT: coherence-gated exponential multistep with σ-annealed ancestral
    noise (``eta_i = eta_max·σ_frac``). One evaluation per step; all families.

    The DPM-Solver++(2M) SDE core (Lu et al. 2022, arXiv:2211.01095) with its
    2nd-order correction scaled by :func:`_coherence_gate`, which damps the term
    on a noisy or imperfect model and keeps it on a clean one. Prefers 24+ steps
    and the ``flow`` / ``simple`` / ``sgm_uniform`` schedulers.

    ``eta_max=0`` is deterministic, and with ``psi ≡ 1`` it is
    :func:`sample_dpmpp_2m_sde` (``eta=0``, flow). ``gate_reduce`` is ``"all"`` or
    the default-off ``"per_channel"``. ``σ_frac`` is σ on flow and ``σ/(1+σ)``
    on VE (both ``sigmoid(-lambda)``); ``shift`` offsets the first flow σ.
    """
    _validate_gate_reduce(gate_reduce)
    if len(sigmas) <= 1:
        return x
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    old_denoised, old_diff = None, None
    h, h_last = None, None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        diff = None if old_denoised is None else denoised - old_denoised
        if bool(sigma_next == 0):
            x = denoised
        else:
            # On flow σ is the noise fraction; on VE it's σ/(1+σ).
            sigma_frac = (float(sigma.clamp(max=1.0)) if model_type == "flow"
                          else float(sigma / (1.0 + sigma)))
            eta = eta_max * sigma_frac
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            if diff is not None:
                psi = _coherence_gate(diff, old_diff, h, reduce=gate_reduce)
                rr = h_last / h
                x = x + psi * (0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / rr) * diff)
            if eta > 0 and s_noise > 0:
                x = x + _noise_like(x, generator) * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        old_denoised, old_diff = denoised, diff
        h_last = h
    return x


def _cogent3_curvature_gate(second_diff: torch.Tensor,
                            old_second_diff: Optional[torch.Tensor],
                            *, reduce: str = "all",
                            ) -> torch.Tensor:
    """Scale factor for a multistep solver's 3rd-order term::

        psi = (2 + 3·rho) / 5                             clamped to [0, 1]

    ``rho`` is the cosine of two consecutive second differences
    ``E_i = x0_i - 2·x0_{i-1} + x0_{i-2}``. Under the :func:`_coherence_gate`
    noise model, ``<E_i, E_{i-1}> = S2 − 4v`` and ``‖E‖² = S2 + 6v``, so the
    Wiener shrink ``S2/(S2 + 6v)`` becomes ``(2 + 3·rho)/5``.

    No step-size floor: the 3rd-order term is never load-bearing, so damping it
    to zero just reverts to the gated 2nd-order step. ``old_second_diff is
    None`` returns 1.0. ``reduce`` works as in :func:`_coherence_gate`.
    """
    _validate_gate_reduce(reduce)
    if old_second_diff is None:
        return torch.ones(second_diff.shape[0], *([1] * (second_diff.ndim - 1)),
                          dtype=second_diff.dtype, device=second_diff.device)
    if reduce == "per_channel":
        if second_diff.ndim == 4:
            dims = (2, 3)
        else:
            reduce = "all"
            dims = tuple(range(1, second_diff.ndim))
    else:  # reduce == "all"
        dims = tuple(range(1, second_diff.ndim))
    num = (second_diff * old_second_diff).sum(dim=dims)
    den = (second_diff.pow(2).sum(dim=dims) * old_second_diff.pow(2).sum(dim=dims)).sqrt()
    rho = num / den.clamp_min(torch.finfo(second_diff.dtype).tiny)
    psi = ((2.0 + 3.0 * rho) / 5.0).clamp(0.0, 1.0)
    return psi.view(*psi.shape, *([1] * (second_diff.ndim - psi.ndim)))


# The pumped band's λ-step of ``pump_dual`` at 50 steps (pump_share 0.85 of
# σ 0.99 → 0.45). The fixed per-step pump (0.08) was judged there, so
# ``cogent3_pump_rate`` scales to it: a step of this size gets exactly 0.08.
PUMP_H_REF = (math.log(0.55 / 0.45) - math.log(0.01 / 0.99)) / (0.85 * 49)


def _ou_step_var(h: float, eta: float) -> float:
    """Variance ``(1 − e^(−2·eta·h)) / (2·eta)`` an OU process with unit
    diffusion and mean reversion ``eta`` accumulates over a λ-step ``h`` (→ ``h``
    as ``eta → 0``). Scaling each pump injection by it leaves the same excess
    noise at every step count."""
    z = 2.0 * eta * h
    return h if z < 1e-8 else -math.expm1(-z) / (2.0 * eta)


def sample_cogent3(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 1.0,
    s_noise: float = 1.0,
    pump_strength: float = 0.0,
    pump_end: float = 0.45,
    pump_span: float = 0.25,
    pump_h_ref: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
    gate_reduce: str = "all",
    gate_stats: Optional[list] = None,
) -> torch.Tensor:
    """COGENT3: COGENT's measured gate carried to third order. One evaluation
    per step; all families.

    The DPM-Solver++(3M) core with the 2nd-order term scaled by
    :func:`_coherence_gate` and the 3rd-order term by
    :func:`_cogent3_curvature_gate` (no floor). With both gates at 1 and
    ``eta_max=0`` it is :func:`sample_dpmpp_3m_sde` (``eta=0``) bit-for-bit.
    Prefers 24+ steps. ``eta_max``, ``gate_reduce``, ``model_type`` and
    ``shift`` work as in :func:`sample_cogent`. ``gate_stats``, if given,
    collects one :func:`_coherence_gate` stats dict per correctable step.

    ``pump_strength > 0`` (the ``cogent3_pump`` sampler) adds a high-σ coherence
    pump, the load-bearing mechanism of :func:`sample_infinity_aether` with a
    hard low-σ shutoff::

        nu  = pump_strength · sigma_next · clamp((sigma_frac − pump_end)/pump_span, 0, 1)
        x  += nu · (1 − C) · noise

    where ``C`` is the structure-tensor coherence of the denoised prediction.
    The gate uses the family-invariant ``sigma_frac``, the amplitude absolute
    ``sigma_next``, and it never touches the final latent. ``pump_h_ref`` (the
    ``cogent3_pump_rate`` sampler) scales each injection by
    ``sqrt(V(h)/V(pump_h_ref))`` (:func:`_ou_step_var`) so the dose is the same
    at any step count; ``None`` is the fixed per-step pump. The pump needs a 4-D
    latent; ``pump_strength=0`` is plain cogent3 and draws no extra noise.
    """
    _validate_gate_reduce(gate_reduce)
    if len(sigmas) <= 1:
        return x
    if pump_strength > 0 and x.ndim != 4:
        raise ValueError(
            f"cogent3's coherence pump needs a 4-D [B, C, H, W] latent (the "
            f"structure tensor is a 2-D convolution); got rank {x.ndim}. FLUX "
            f"packs the latent into a [B, L, C·p²] token sequence, so use plain "
            f"cogent3 (pump_strength=0) there."
        )
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    denoised_1, denoised_2, denoised_3 = None, None, None
    h, h_1, h_2 = None, None, None
    _stats = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            # σ-annealed ancestral fraction, as in cogent.
            sigma_frac = (float(sigma.clamp(max=1.0)) if model_type == "flow"
                          else float(sigma / (1.0 + sigma)))
            eta = eta_max * sigma_frac
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            _stats = None
            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                # ψ₂ bootstraps from ψ₁ until there is curvature history.
                _stats = {} if gate_stats is not None else None
                psi_1 = _coherence_gate(denoised - denoised_1,
                                        denoised_1 - denoised_2, h,
                                        reduce=gate_reduce, stats_out=_stats)
                e_cur = denoised - 2.0 * denoised_1 + denoised_2
                psi_2 = (_cogent3_curvature_gate(
                    e_cur, denoised_1 - 2.0 * denoised_2 + denoised_3,
                    reduce=gate_reduce)
                    if denoised_3 is not None else psi_1)
                x = x + psi_1 * (alpha_t * phi_2) * d1 - psi_2 * (alpha_t * phi_3) * d2
            elif h_1 is not None:
                rr = h_1 / h
                d = (denoised - denoised_1) / rr
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                _stats = {} if gate_stats is not None else None
                psi_1 = _coherence_gate(denoised - denoised_1, None, h,
                                        reduce=gate_reduce, stats_out=_stats)
                x = x + psi_1 * (alpha_t * phi_2) * d
            if _stats is not None:
                _stats["step"] = i
                _stats["sigma"] = float(sigma)
                _stats["sigma_next"] = float(sigma_next)
                _stats["h"] = float(h)
                gate_stats.append(_stats)
            if eta > 0 and s_noise > 0:
                x = x + _noise_like(x, generator) * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise
            # Coherence pump. No noise is drawn below pump_end, so
            # pump_strength=0 leaves the generator stream untouched.
            if pump_strength > 0:
                ramp = (1.0 if pump_span <= 0 else
                        min(1.0, max(0.0, (sigma_frac - pump_end) / pump_span)))
                if sigma_frac < pump_end:
                    ramp = 0.0
                if ramp > 0:
                    nu = pump_strength * float(sigma_next) * ramp
                    if pump_h_ref is not None:
                        nu *= math.sqrt(_ou_step_var(float(h), eta) / _ou_step_var(pump_h_ref, eta))
                    # Read coherence off the denoised prediction: at low σ the
                    # velocity is mostly residual noise.
                    c = _structure_tensor_coherence(denoised.float(), multi_scale=True)
                    x = x + (nu * (1.0 - c)).to(x.dtype) * _noise_like(x, generator)
        denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
        h_1, h_2 = h, h_1
    return x


def sample_heunpp2(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None) -> torch.Tensor:
    """Heun++: away from the schedule end, takes a third evaluation and blends
    the three derivatives with sigma-proportional weights. After the MIT
    sd-webui-samplers-scheduler implementation."""
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
    """DPM-Solver++(2S) ancestral: single-step 2nd order, two evaluations per
    step. ``flow`` follows ComfyUI's ``sample_dpmpp_2s_ancestral_RF`` (σ=1
    guarded); ``shift`` is unused."""
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
    """iPNDM: Adams–Bashforth multistep in σ space, up to 4th order with fixed
    coefficients. After zju-pi/diff-sampler (Apache-2.0)."""
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
    """iPNDM_v: :func:`sample_ipndm` with coefficients recomputed from the
    actual σ spacing each step. After zju-pi/diff-sampler (Apache-2.0)."""
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
    """Shared body of :func:`sample_res_multistep` (``eta=0``) and
    :func:`sample_res_multistep_ancestral`: 2nd-order multistep exponential
    (RES) solver, Zhang et al. (arXiv:2308.02157), in ``t = -log σ`` so it
    serves VE and flow. The ancestral split is flow-aware for ``"flow"``."""
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
    """RES, deterministic 2nd-order multistep (see :func:`_res_multistep`)."""
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
    """Ancestral RES multistep (see :func:`_res_multistep`). ``shift`` is unused."""
    del shift
    return _res_multistep(model, x, sigmas, eta=eta, s_noise=s_noise, generator=generator,
                          callback=callback, model_type=model_type)


def sample_gradient_estimation(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *,
                               callback: Callback = None, ge_gamma: float = 2.0) -> torch.Tensor:
    """Gradient-estimation sampler (Liu et al., openreview o2ND9v0CeK): Euler
    plus ``(γ-1)·(d_i - d_{i-1})``. ``ge_gamma=1`` is Euler."""
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


@lru_cache(maxsize=None)
def _rkg2_coeffs(s: int):
    """Closed-form stage coefficients of the ``s``-stage second-order
    Runge–Kutta–Gegenbauer method (Skaras & O'Sullivan, J. Comput. Phys. 2021),
    with ``b_0 = 1``, ``b_1 = 1/3`` and ``c_j = (j²+j-2)/(s²+s-2)``
    (``c_1 = c_2/3``). Returns ``(w1, c, stage)``: ``c[j]`` is stage ``j``'s
    offset as a fraction of the step and ``stage[j-2] = (mu_j, nu_j,
    mu_tilde_j, gamma_tilde_j)`` for ``j = 2..s``."""
    w1 = 6.0 / ((s + 4.0) * (s - 1.0))

    def b(j: int) -> float:
        if j == 0:
            return 1.0
        if j == 1:
            return 1.0 / 3.0
        return 4.0 * (j - 1.0) * (j + 4.0) / (3.0 * j * (j + 1.0) * (j + 2.0) * (j + 3.0))

    den = s * s + s - 2.0
    c = [0.0] * (s + 1)
    c[1] = 4.0 / (3.0 * den)
    stage = []
    for j in range(2, s + 1):
        c[j] = (j * j + j - 2.0) / den
        a_prev = 1.0 - j * (j + 1.0) / 2.0 * b(j - 1)
        mu = (2.0 * j + 1.0) / j * b(j) / b(j - 1)
        nu = -(j + 1.0) / j * b(j) / b(j - 2)
        mut = mu * w1
        gat = -mut * a_prev
        stage.append((mu, nu, mut, gat))
    return w1, c, stage


def sample_stork2(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
    stages: int = 9,
    taylor_order: int = 1,
) -> torch.Tensor:
    """STORK-2 (Tan et al., ICLR 2026, arXiv:2505.24210), clean-room from the
    paper. Deterministic, one evaluation per step, all families.

    Each step runs a ``stages``-stage RKG2 cascade (:func:`_rkg2_coeffs`) on
    the σ-space ODE whose stage velocities are Taylor expansions built from
    divided differences of previous real evaluations. With ``taylor_order=1``
    it collapses to a variable-step AB2 whose derivative correction is damped
    from 1/2 to ``C1(s)`` (≈0.463 at s=9), so ``stages`` is a robustness dial,
    not a cost dial (smaller damps more). ``taylor_order=2`` adds a ``v̈`` term.
    Divided differences are exact on nonuniform grids; the first step is Euler.
    """
    if stages < 2:
        raise ValueError("stages must be >= 2")
    if taylor_order not in (1, 2):
        raise ValueError("taylor_order must be 1 or 2")
    s_in = x.new_ones([x.shape[0]])
    w1, c, stage_coeffs = _rkg2_coeffs(stages)
    hist_sigma: list[float] = []       # σ of the previous real evaluations
    hist_v: list[torch.Tensor] = []    # their velocities, newest last
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        dt = sigma_next - sigma
        s0 = float(sigma)
        # dv/dσ estimates from the history; σ-collisions degrade the order.
        vp = vpp = None
        if hist_sigma and abs(s0 - hist_sigma[-1]) > 1e-8:
            f01 = (d - hist_v[-1]) / (s0 - hist_sigma[-1])
            vp = f01
            if taylor_order >= 2 and len(hist_sigma) >= 2:
                s1, s2 = hist_sigma[-1], hist_sigma[-2]
                if abs(s1 - s2) > 1e-8 and abs(s0 - s2) > 1e-8:
                    f12 = (hist_v[-1] - hist_v[-2]) / (s1 - s2)
                    f012 = (f01 - f12) / (s0 - s2)
                    vp = f01 + (s0 - s1) * f012
                    vpp = 2.0 * f012
        if bool(sigma_next == 0):
            x = denoised
        elif vp is None:
            x = x + d * dt             # Euler warmup
        else:
            Y0 = x
            Yjm2, Yjm1 = Y0, Y0 + (w1 * dt) * d          # stage 1
            for j, (mu, nu, mut, gat) in enumerate(stage_coeffs, start=2):
                t_off = c[j - 1] * dt
                v_approx = d + t_off * vp                 # virtual NFE at stage j-1
                if vpp is not None:
                    v_approx = v_approx + (0.5 * t_off * t_off) * vpp
                Yj = (mu * Yjm1 + nu * Yjm2 + (1.0 - mu - nu) * Y0
                      + (mut * dt) * v_approx + (gat * dt) * d)
                Yjm2, Yjm1 = Yjm1, Yj
            x = Yjm1
        hist_sigma.append(s0)
        hist_v.append(d)
        if len(hist_sigma) > 2:
            hist_sigma.pop(0)
            hist_v.pop(0)
    return x


def sample_infinity(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
) -> torch.Tensor:
    """Infinity Diffusion sampler (galpt/infinity-diffusion, MIT; upstream
    ``main`` @4f72d8f, 2026-07-17). Deterministic, one evaluation per step,
    all families.

    Euler plus an invariant-gated IIR correction: a velocity EMA of the first
    derivative difference and, from the third step, an acceleration EMA of the
    second. The correction is clamped to 50% of the derivative's magnitude,
    halved if the derivative reversed direction, and zeroed when both apply.
    Constants are upstream's fixed ``α₁=β₁=0.5, α₂=β₂=0.3``.

    One deviation: each difference is divided by its step size before the EMA
    and the output multiplied by the current one::

        dd  ← (d − d_prev) / dt_prev
        vel ← (1−α₁)·vel + α₁·dd
        acc ← (1−α₂)·acc + α₂·(dd − dd_prev)/dt_prev
        correction = β₁·dt·vel + β₂·dt²·acc

    On a uniform grid this is upstream's recursion bit-for-bit; on nonuniform
    ones it is the step-size-consistent form (~3× more accurate on
    ``flow``/``normal``, and ``karras`` becomes usable). The clamp and cosine
    reduce over the whole tensor, as upstream does."""
    alpha1, beta1 = 0.5, 0.5
    alpha2, beta2 = 0.3, 0.3
    s_in = x.new_ones([x.shape[0]])
    vel = acc = d_prev = dd_prev = None
    dt_prev = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        dt = sigma_next - sigma
        if d_prev is None:
            vel = torch.zeros_like(d)
            acc = torch.zeros_like(d)
            x = x + d * dt                      # Euler bootstrap
            d_prev, dt_prev = d, dt
            continue
        dd = (d - d_prev) / dt_prev             # ≈ d′(σ)
        vel = (1.0 - alpha1) * vel + alpha1 * dd
        if dd_prev is None:
            correction = beta1 * dt * vel       # acceleration needs 3 d's
        else:
            acc = (1.0 - alpha2) * acc + alpha2 * ((dd - dd_prev) / dt_prev)
            correction = beta1 * dt * vel + beta2 * dt * dt * acc
        d_mag = d.abs().mean() + 1e-8
        c_mag = correction.abs().mean()
        clamped = bool(c_mag > 0.5 * d_mag)
        if clamped:
            correction = correction * (0.5 * d_mag / c_mag)
        cos_sim = (d * d_prev).sum() / (d.norm() * d_prev.norm() + 1e-8)
        reversed_dir = bool(cos_sim < 0.0)
        if clamped and reversed_dir:
            correction = torch.zeros_like(correction)
        elif reversed_dir:
            correction = correction * 0.5
        x = x + (d + correction) * dt
        dd_prev, d_prev, dt_prev = dd, d, dt
    return x


def _variance_stabilize(denoised: torch.Tensor, ema_std: Optional[torch.Tensor],
                        momentum: float, progress: float, total_steps: int,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``realism`` branch's variance stabilizer: pull each channel's std
    toward its running EMA by ``deviation/(deviation+0.3) ·
    progress/(progress+0.2) · steps/(steps+8)`` (the last factor is upstream's
    Turbo/LCM guard). Reduces over the batch axis too, as upstream does. Runs
    in float32; the ``[0.1, 10]`` clamp is upstream's guard against a
    near-uniform channel."""
    eps = 1e-4
    d = denoised.float()
    mean = d.mean(dim=(0, 2, 3), keepdim=True)
    centered = d - mean
    cur_std = centered.std(dim=(0, 2, 3)).clamp(min=eps)
    if ema_std is None:
        return denoised, cur_std
    new_ema = momentum * ema_std + (1.0 - momentum) * cur_std
    deviation = (cur_std / (new_ema + eps) - 1.0).abs()
    strength = ((deviation / (deviation + 0.3))
                * (progress / (progress + 0.2))
                * (total_steps / (total_steps + 8.0)))
    target = cur_std + (new_ema - cur_std) * strength
    corr = (target / cur_std).clamp(min=0.1, max=10.0)
    result = centered * corr.reshape(1, -1, 1, 1) + mean
    return result.to(denoised.dtype), new_ema


def sample_infinity_realism(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
) -> torch.Tensor:
    """Infinity Diffusion, ``realism`` branch (galpt/infinity-diffusion, MIT;
    upstream @21084d9, 2026-07-21). Deterministic, one evaluation per step.

    Euler (upstream's ``x ← r·x − (r−1)·x0``) with the per-channel variance
    stabilizer of :func:`_variance_stabilize` applied to x0 before each step.
    The upstream rewrite dropped the old branch's absolute ``γ·σ`` noise
    injection, which over-injected badly on flow, so it is no longer SD-only.
    Needs a 4-D latent (so not FLUX).
    """
    if x.ndim != 4:
        raise ValueError(
            f"infinity_realism needs a 4-D [B, C, H, W] latent (its variance "
            f"stabilizer takes a per-channel statistic over the spatial axes); "
            f"got rank {x.ndim}. FLUX packs the latent into a [B, L, C·p²] "
            f"token sequence, so this sampler is not available for it; use "
            f"infinity there."
        )
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    ema_std = None
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if i == 0:
            # Bootstrap: record the spread, correct nothing.
            _, ema_std = _variance_stabilize(denoised, None, 0.0, 0.0, total_steps)
        else:
            denoised, ema_std = _variance_stabilize(
                denoised, ema_std, 1.0 - 1.0 / total_steps, i / total_steps, total_steps)
        ratio = sigma_next / sigma                  # r·x − (r−1)·x0 ≡ Euler in σ
        x = ratio * x - (ratio - 1.0) * denoised
    return x


def _gaussian_blur2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur for :func:`sample_infinity_omega`'s pyramid
    (one grouped conv2d, as upstream does)."""
    radius = kernel_size // 2
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


def _quantile_variance_preserve(denoised: torch.Tensor, ema_q95: Optional[torch.Tensor],
                                total_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """NQVP: pull each channel's 95th-percentile spatial deviation toward its
    running EMA, within ``[0.88, 1.12]``. Float32, since ``torch.quantile``
    rejects fp16/bf16."""
    eps = 6.1035e-5
    d = denoised.float()
    mean = d.mean(dim=(2, 3), keepdim=True)
    centered = d - mean
    q95 = torch.quantile(centered.abs().flatten(2), 0.95, dim=2,
                         keepdim=True).unsqueeze(-1).clamp(min=eps)
    if ema_q95 is None:
        return denoised, q95
    momentum = 1.0 - 1.0 / max(1.0, float(total_steps))
    new_ema = momentum * ema_q95 + (1.0 - momentum) * q95
    ratio = (new_ema / (q95 + eps)).clamp(min=0.88, max=1.12)
    return (centered * ratio + mean).to(denoised.dtype), new_ema


def _adaptive_velocity_normalize(v: torch.Tensor, ema_std: Optional[torch.Tensor],
                                 total_steps: int, clamp_min: float,
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """AVN: damp each channel's spatial std of the velocity field toward its
    EMA, within ``[clamp_min, 1.0]``. It only ever shrinks a spread and leaves
    the mean alone (unlike the ACS it replaced, which cast colour on flow).
    Float32."""
    eps = 6.1035e-5
    d = v.float()
    mean = d.mean(dim=(2, 3), keepdim=True)
    centered = d - mean
    cur_std = centered.std(dim=(2, 3), keepdim=True).clamp(min=eps)
    if ema_std is None:
        return v, cur_std
    momentum = 1.0 - 1.0 / max(1.0, float(total_steps))
    new_ema = momentum * ema_std + (1.0 - momentum) * cur_std
    corr = (new_ema / (cur_std + eps)).clamp(min=clamp_min, max=1.0)
    return (centered * corr + mean).to(v.dtype), new_ema


# omega (@8d81e76) skips NQVP when ``is_flow = sigma_max < 5``, so NQVP runs
# only on VE models (SD/SDXL start at 14.6, flow at 1.0).
_NQVP_SIGMA_MIN_OMEGA = 5.0


def _sample_infinity_pyramid(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    name: str,
    nqvp_sigma_min: float,
    avn: bool,
    dog: bool,
    callback: Callback = None,
) -> torch.Tensor:
    """The ``omega`` loop. ``avn``/``dog`` and the gate constant are arguments so
    tests can pin each piece."""
    if x.ndim != 4:
        raise ValueError(
            f"{name} needs a 4-D [B, C, H, W] latent (its band decomposition is "
            f"2-D convolution); got rank {x.ndim}. FLUX packs the latent into a "
            f"[B, L, C·p²] token sequence, so this sampler is not available for "
            f"it; use infinity there."
        )
    eps = 6.1035e-5
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    # Upstream's is_flow / split-resume test.
    is_flow = float(sigmas[0]) < nqvp_sigma_min
    ema_q95 = ema_v_std = None
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        dt = sigma_next - sigma
        if total_steps <= 6:                    # distilled/Turbo: plain Euler
            x = x + to_d(x, sigma * s_in, denoised) * dt
            continue
        if not is_flow:
            denoised, ema_q95 = _quantile_variance_preserve(denoised, ema_q95, total_steps)
        v = to_d(x, sigma * s_in, denoised).float()
        if avn:
            # Upstream damps flow harder than VE (0.70 vs 0.85).
            v, ema_v_std = _adaptive_velocity_normalize(
                v, ema_v_std, total_steps, 0.70 if is_flow else 0.85)

        macro = _gaussian_blur2d(v, 5, 2.0)
        mid = _gaussian_blur2d(v, 3, 1.0)
        meso = mid - macro
        nano = v - mid

        # Local std map of the nano band.
        var = _gaussian_blur2d(nano * nano, 3, 1.0) - _gaussian_blur2d(nano, 3, 1.0) ** 2
        s_nano = var.clamp(min=eps).sqrt()
        # Upstream's literal knee; on flow (σ <= 1) eta stays below 0.167.
        eta = 0.25 * min(1.0, max(0.1, float(sigma) / 1.5))
        gain = 1.0 + eta * torch.tanh(s_nano / (s_nano.mean(dim=(2, 3), keepdim=True) + eps))

        if dog:
            band = _gaussian_blur2d(nano, 3, 0.5) - _gaussian_blur2d(nano, 5, 1.0)
            nano = nano + (0.15 * eta) * band

        x = x + (macro + meso + gain * nano).to(x.dtype) * dt
    return x


def sample_infinity_omega(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
) -> torch.Tensor:
    """Infinity Diffusion, ``omega`` branch (galpt/infinity-diffusion, MIT;
    upstream @8d81e76, 2026-07-25).

    Plain Euler plus a per-step spatial filter on the velocity and two
    stabilizers:

    * **LPVD** splits ``d`` into macro / meso / nano bands with a Gaussian
      pyramid.
    * **AHFRI** amplifies the nano band by ``1 + η·tanh(s/s̄)`` (``s`` its local
      std), ``η = 0.25·clamp(σ/1.5, 0.1, 1)``.
    * **DoG** adds ``0.15·η`` of a band-pass of the nano band (near no-op).
    * **NQVP** (:func:`_quantile_variance_preserve`), SD/SDXL only.
    * **AVN** (:func:`_adaptive_velocity_normalize`), every family.

    Below 7 steps it is exactly ``euler``. Needs a 4-D latent (so not FLUX).
    The only port deviation is float32 stabilizers; fed a float32 latent it
    matches upstream bit-for-bit. Upstream's 5-D ``(B, C, T, H, W)`` fold is
    omitted because the pipelines sample 4-D latents with ``T=1``.
    """
    return _sample_infinity_pyramid(model, x, sigmas, name="infinity_omega",
                                    nqvp_sigma_min=_NQVP_SIGMA_MIN_OMEGA,
                                    avn=True, dog=True, callback=callback)


# --- aether -----------------------------------------------------------------
# Upstream's noise-injection constants (galpt/infinity-diffusion `aether`
# @c3ba017). The injected std is
#     n(σ) = min(0.25·σ, 0.08) · ramp(σ) · (1 − C),   ramp = clamp((σ−0.02)/0.08)
# capped at max(0.30·σ_next, 0.03). All are absolute noise scales, which do not
# transfer to rectified flow; see :func:`sample_infinity_aether`.
_AETHER_NOISE_SIGMA_COEF = 0.25
_AETHER_NOISE_ABS_CAP = 0.08
_AETHER_NOISE_TERMINAL_FLOOR = 0.03


def _central_gradients(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences with edge-replicated padding."""
    p = F.pad(v, (1, 1, 1, 1), mode="replicate")
    return (p[..., 1:-1, 2:] - p[..., 1:-1, :-2],
            p[..., 2:, 1:-1] - p[..., :-2, 1:-1])


def _structure_tensor_coherence(v: torch.Tensor, eps: float = 1e-5,
                                multi_scale: bool = False) -> torch.Tensor:
    """Local structure-tensor coherence ``C ∈ [0, 1]``: near 1 along a strong
    edge, near 0 where the gradient is isotropic or noise.
    ``((J_xx − J_yy)² + 4·J_xy²) / tr(J)²`` of the smoothed gradient outer
    product. ``multi_scale`` takes the per-pixel max over three blur scales."""
    v_x, v_y = _central_gradients(v)

    def at_scale(ks: int, sg: float) -> torch.Tensor:
        j_xx = _gaussian_blur2d(v_x * v_x, ks, sg)
        j_yy = _gaussian_blur2d(v_y * v_y, ks, sg)
        j_xy = _gaussian_blur2d(v_x * v_y, ks, sg)
        tr = j_xx + j_yy + eps
        return (((j_xx - j_yy) ** 2 + 4 * j_xy ** 2) / (tr * tr + eps)).clamp(0.0, 1.0)

    if not multi_scale:
        return at_scale(3, 1.0)
    c = at_scale(3, 0.5)
    for ks, sg in ((5, 1.0), (7, 2.0)):
        c = torch.maximum(c, at_scale(ks, sg))
    return c


# Laws' 5×5 texture-energy masks as separable 1-D kernels, and the nine outer
# products upstream selects, in its argmax order.
_LAWS_1D = {
    "L5": (1.0, 4.0, 6.0, 4.0, 1.0),        # level
    "E5": (-1.0, -2.0, 0.0, 2.0, 1.0),      # edge
    "S5": (-1.0, 0.0, 2.0, 0.0, -1.0),      # spot
    "R5": (1.0, -4.0, 6.0, -4.0, 1.0),      # ripple
    "W5": (-1.0, 2.0, -2.0, 2.0, -1.0),     # wave
}
_LAWS_PAIRS = (("L5", "L5"), ("E5", "E5"), ("S5", "S5"), ("R5", "R5"),
               ("W5", "W5"), ("L5", "E5"), ("E5", "L5"), ("L5", "S5"),
               ("S5", "L5"))
# The material class each response votes for: 0 flat, 1 skin/texture,
# 2 line art, 3 fabric/ripple.
_LAWS_MATERIAL = (0, 2, 1, 3, 3, 2, 2, 1, 1)


def _classify_material(v: torch.Tensor) -> torch.Tensor:
    """Per-pixel material class from the strongest Laws texture-energy response
    (nine 5×5 masks, each normalized by 36). Returns an integer tensor shaped
    like the input."""
    b, c, h, w = v.shape
    flat = v.reshape(b * c, 1, h, w)
    resp = []
    for k1, k2 in _LAWS_PAIRS:
        a = torch.tensor(_LAWS_1D[k1], dtype=v.dtype, device=v.device)
        bb = torch.tensor(_LAWS_1D[k2], dtype=v.dtype, device=v.device)
        kernel = (a[:, None] * bb[None, :] / 36.0)[None, None]
        resp.append(F.conv2d(flat, kernel, padding=2).abs())
    argmax = torch.stack(resp, dim=-1).max(dim=-1).indices.reshape(b, c, h, w)
    lut = torch.tensor(_LAWS_MATERIAL, dtype=torch.uint8, device=v.device)
    return lut[argmax]


def _phase_edge_saliency(v: torch.Tensor, eps: float = 6.1035e-5) -> torch.Tensor:
    """Contrast-invariant edge saliency, a cheap stand-in for phase congruency
    (Kovesi 1995): local energy ``√(|∇v|² + ∇²v²)`` as ``E / (E + Ē)`` against
    its smoothed self, so a faint crease scores like a bold outline. ~0.5 in
    featureless regions."""
    v_x, v_y = _central_gradients(v)
    px, py = F.pad(v_x, (1, 1)), F.pad(v_y, (0, 0, 1, 1))
    laplacian = (px[..., 2:] - px[..., :-2]) + (py[..., 2:, :] - py[..., :-2, :])
    grad_mag = torch.sqrt(v_x ** 2 + v_y ** 2 + eps)
    energy = torch.sqrt(grad_mag ** 2 + laplacian ** 2 + eps)
    smoothed = _gaussian_blur2d(energy, 7, 2.0)
    return (energy / (energy + smoothed + eps)).clamp(0.0, 1.0)


def _coherence_lisc(v: torch.Tensor, light_angle_deg: float, strength: float,
                    eps: float = 6.1035e-5) -> torch.Tensor:
    """LISC: add ``strength·C·(∇v · l̂)``, directional shading along a virtual
    light, masked by structure-tensor coherence so it only lands on coherent
    structure. Upstream's ``eps = 6.1035e-5`` is passed through explicitly."""
    v_x, v_y = _central_gradients(v)
    rad = math.radians(light_angle_deg)
    coherence = _structure_tensor_coherence(v, eps=eps)
    return v + strength * coherence * (v_x * math.cos(rad) + v_y * math.sin(rad))


def _velocity_norm_normalize(v_enhanced: torch.Tensor,
                             v_reference: torch.Tensor) -> torch.Tensor:
    """VNN: rescale each sample so the enhanced velocity keeps the L2 norm of
    the one it was built from; the band enhancements move energy between
    frequencies but can't add any."""
    eps = 6.1035e-5
    shape = (-1,) + (1,) * (v_enhanced.ndim - 1)
    ref = torch.norm(v_reference.flatten(1), p=2, dim=1).reshape(shape)
    enh = torch.norm(v_enhanced.flatten(1), p=2, dim=1).reshape(shape) + eps
    return v_enhanced * (ref / enh)


def sample_infinity_aether(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    light_angle_deg: float = 135.0,
    lisc_strength: float = 0.06,
) -> torch.Tensor:
    """Infinity Diffusion, ``aether`` branch (galpt/infinity-diffusion, MIT;
    upstream @c3ba017, 2026-07-31).

    :func:`sample_infinity_omega`'s stack with the isotropic DoG replaced by a
    material-aware one: Laws texture masks label each pixel flat / skin / line
    art / fabric, and the band-pass gain per class comes from structure-tensor
    coherence and :func:`_phase_edge_saliency`. It adds LISC shading on the
    macro band while ``σ ≥ 0.80``, VNN energy normalization, a TZTD ramp of
    every enhancement to zero between σ 0.80 and 0.15, and grain scaled by
    ``1 − C``.

    The one behavioural deviation: noise is drawn from ``generator``, so seeds
    reproduce. All σ thresholds are absolute and tuned for SD's range, so flow
    results are unvalidated. Needs a 4-D latent.
    """
    if x.ndim != 4:
        raise ValueError(
            f"infinity_aether needs a 4-D [B, C, H, W] latent (its band "
            f"decomposition, structure tensor and Laws masks are all 2-D "
            f"convolutions); got rank {x.ndim}. FLUX packs the latent into a "
            f"[B, L, C·p²] token sequence, so this sampler is not available for "
            f"it; use infinity there."
        )
    eps = 6.1035e-5
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    is_flow = float(sigmas[0]) < _NQVP_SIGMA_MIN_OMEGA
    ema_q95 = ema_v_std = None
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        s_cur, s_next = float(sigma), float(sigma_next)
        dt = sigma_next - sigma
        if total_steps <= 6:                    # distilled/Turbo: plain Euler
            x = x + to_d(x, sigma * s_in, denoised) * dt
            continue
        if not is_flow:
            denoised, ema_q95 = _quantile_variance_preserve(denoised, ema_q95, total_steps)

        # Analysis maps read off the denoised prediction, which survives the
        # low-σ steps where the velocity is mostly residual noise.
        dc = denoised.float()
        noise_coherence = _structure_tensor_coherence(dc, eps=eps, multi_scale=True)
        material = _classify_material(dc)
        phase_sal = _phase_edge_saliency(dc)

        v = to_d(x, sigma * s_in, denoised).float()
        v, ema_v_std = _adaptive_velocity_normalize(
            v, ema_v_std, total_steps, 0.70 if is_flow else 0.85)

        # TZTD: 1 above σ=0.80, linearly to 0 at σ=0.15.
        gamma = min(1.0, max(0.0, (s_cur - 0.15) / 0.65))
        if gamma <= 1e-4:
            v_step = v                          # terminal steps: pure Euler
        else:
            macro = _gaussian_blur2d(v, 5, 2.0)
            mid = _gaussian_blur2d(v, 3, 1.0)
            meso = mid - macro
            nano = v - mid

            var = _gaussian_blur2d(nano * nano, 3, 1.0) - _gaussian_blur2d(nano, 3, 1.0) ** 2
            s_nano = var.clamp(min=eps).sqrt()
            eta = 0.25 * min(1.0, max(0.1, s_cur / 1.5))
            ahfri = 1.0 + eta * torch.tanh(s_nano / (s_nano.mean(dim=(2, 3), keepdim=True) + eps))

            if s_cur >= 0.80 and lisc_strength > 0:
                macro = _coherence_lisc(macro, light_angle_deg,
                                        strength=lisc_strength * gamma, eps=eps)

            band = _gaussian_blur2d(nano, 3, 0.5) - _gaussian_blur2d(nano, 5, 1.0)
            coherence = _structure_tensor_coherence(nano, eps=eps)
            # s_nano/(s_nano+eps) is ~1 wherever the band has amplitude, so the
            # skin/fabric blends are nearly a flat lift. Upstream's form, literal.
            iso = s_nano / (s_nano + eps)
            gain = torch.where(
                material == 0, coherence * 0.5,
                torch.where(
                    material == 1,
                    coherence + (1.0 - coherence) * (iso * 0.50 + phase_sal * 0.30),
                    torch.where(
                        material == 2, torch.maximum(coherence, phase_sal),
                        coherence + (1.0 - coherence) * iso * 0.65)))
            nano = nano + (0.15 * eta * gamma) * gain * band

            v_step = _velocity_norm_normalize(macro + meso + ahfri * nano, v)

        x = x + v_step.to(x.dtype) * dt

        # Coherence-gated grain, skipped below σ=0.02.
        if s_cur > 0.02:
            n_s = min(_AETHER_NOISE_SIGMA_COEF * s_cur, _AETHER_NOISE_ABS_CAP)
            n_s *= min(1.0, max(0.0, (s_cur - 0.02) / 0.08))
            n_s = min(n_s, max(0.30 * s_next, _AETHER_NOISE_TERMINAL_FLOOR))
            x = x + (n_s * (1.0 - noise_coherence)).to(x.dtype) * _noise_like(x, generator)
    return x


def _linear_multistep_coeff(order: int, t: np.ndarray, i: int, j: int) -> float:
    """Adams–Bashforth coefficient for :func:`sample_lms`: the exact integral
    over ``[t_i, t_{i+1}]`` of the ``j``-th Lagrange basis polynomial (matches
    ComfyUI's ``scipy.integrate.quad`` without scipy)."""
    poly = np.polynomial.Polynomial([1.0])
    for k in range(order):
        if k == j:
            continue
        poly = poly * np.polynomial.Polynomial([-t[i - k], 1.0]) / (t[i - j] - t[i - k])
    integ = poly.integ()
    return float(integ(t[i + 1]) - integ(t[i]))


def sample_lms(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *, callback: Callback = None,
               order: int = 4) -> torch.Tensor:
    """LMS: linear multistep in σ space with coefficients from the actual σ
    nodes each step (k-diffusion, MIT)."""
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
    """LCM: jump to the x0 estimate each step, then re-noise to ``sigma_next``
    (``(1-σ)·x0 + σ·ε`` on flow, ``x0 + σ·ε`` on VE). ``shift`` is unused."""
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


def _sa_solver_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int,
                                  tau_t: float) -> torch.Tensor:
    """SA-Solver exponential-integrator coefficients: ``(1 + τ²)·∫_s^t
    exp((1+τ²)·x)·x^p dx`` with ``exp((1+τ²)·t)`` factored out, for
    ``p = 0..solver_order−1``, via the reference's integration-by-parts table."""
    tau_mul = 1 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    # x^p·exp((1+τ²)·x)/(1+τ²) at x=s, factored by exp((1+τ²)·t).
    product_terms_factored = (t ** p - s ** p * (-tau_mul * h).exp())
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()
    return recursive_coeff_mat @ product_terms_factored


def _sa_solver_simple_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor,
                               lambda_s: torch.Tensor, lambda_t: torch.Tensor,
                               tau_t: float, is_corrector_step: bool = False) -> torch.Tensor:
    """The SA-Solver paper's closed-form order-2 b-coefficients (Appendix D).
    Returns ``[b_2, b_1]``."""
    tau_mul = 1 + tau_t ** 2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        b_2 = alpha_t * (0.5 * tau_mul * h ** 2) / (curr_lambdas[-2] - lambda_s)
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    return torch.stack([b_2, b_1])


def _sa_solver_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor,
                        lambda_s: torch.Tensor, lambda_t: torch.Tensor,
                        tau_t: float, simple_order_2: bool = False,
                        is_corrector_step: bool = False) -> torch.Tensor:
    """SA-Solver's ``b_i`` coefficients (paper eqs. 15 and 18), x0 form: the
    Lagrange-basis integrals of the exponential integrator over the step, via a
    Vandermonde solve. The order is ``len(curr_lambdas)``."""
    num_timesteps = curr_lambdas.shape[0]
    if simple_order_2 and num_timesteps == 2:
        return _sa_solver_simple_b_coeffs(sigma_next, curr_lambdas, lambda_s,
                                          lambda_t, tau_t, is_corrector_step)
    exp_integral_coeffs = _sa_solver_exponential_coeffs(
        lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_T = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_T, exp_integral_coeffs)
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


def _sa_solver_tau_interval(sigmas: torch.Tensor, eta: float):
    """Default SA-Solver stochasticity window: ``eta`` on the middle 20%–80% of
    the schedule, zero elsewhere. ComfyUI derives the band from the model's
    ``percent_to_sigma``; here it's read off the schedule. ``eta <= 0`` is the
    deterministic ODE."""
    if eta <= 0:
        return lambda sigma: 0.0
    n = len(sigmas) - 1
    start_sigma = float(sigmas[min(n, round(0.2 * n))])
    end_sigma = float(sigmas[min(n, round(0.8 * n))])
    return lambda sigma: float(eta) if start_sigma >= float(sigma) >= end_sigma else 0.0


def sample_sa_solver(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    tau_func=None,
    s_noise: float = 1.0,
    predictor_order: int = 3,
    corrector_order: int = 4,
    use_pece: bool = False,
    simple_order_2: bool = False,
    eta: float = 1.0,
    model_type: str = "ve",
    shift: float = 1.0,
) -> torch.Tensor:
    """SA-Solver: stochastic Adams predictor-corrector (Xue et al., NeurIPS
    2023, arXiv:2309.05019), in half-logSNR space, x0 form. Flow-aware.

    Each step corrects the current latent from the x0 history, then predicts
    the next with exponential-integrator coefficients; noise of strength
    ``τ(sigma)`` is re-injected on a middle band of the schedule
    (:func:`_sa_solver_tau_interval`). Orders ramp up from 1 and back down near
    σ → 0. ``use_pece=True`` re-evaluates the corrected state (one extra
    evaluation per corrected step; registered as ``sa_solver_pece``).
    ``simple_order_2`` uses the paper's closed-form order-2 coefficients.

    After the official codebase as carried in ComfyUI's ``sa_solver.py``,
    except that the tau window comes from the schedule and ``s_noise`` is not
    scaled by a model ``noise_scale``."""
    if len(sigmas) <= 1:
        return x
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: _half_log_snr(sigma, model_type)
    sigmas = _offset_first_sigma_for_snr(sigmas, model_type, shift)
    lambdas = lambda_fn(sigmas)

    if tau_func is None:
        tau_func = _sa_solver_tau_interval(sigmas, eta)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x
    h = 0.0
    tau_t = 0.0
    noise = 0.0
    pred_list = []
    lower_order_to_end = bool(sigmas[-1] == 0)

    for i in range(len(sigmas) - 1):
        denoised = model(x_pred, sigmas[i] * s_in)
        if callback is not None:
            callback(i, sigmas[i], x_pred, denoised)
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (bool(sigmas[i + 1] == 0) and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))
        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector: re-derive the state at sigma[i] from the x0 history.
        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs = _sa_solver_b_coeffs(
                sigmas[i], curr_lambdas, lambdas[i - 1], lambdas[i],
                tau_t, simple_order_2, True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res
            if tau_t > 0 and s_noise > 0:
                x = x + noise
            if use_pece:
                denoised = model(x, sigmas[i] * s_in)
                pred_list[-1] = denoised

        # Predictor: exponential-integrator step to sigma[i+1].
        if bool(sigmas[i + 1] == 0):
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = _sa_solver_b_coeffs(
                sigmas[i + 1], curr_lambdas, lambdas[i], lambdas[i + 1],
                tau_t, simple_order_2, False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res
            if tau_t > 0 and s_noise > 0:
                noise = _noise_like(x_pred, generator) * sigmas[i + 1] \
                    * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
    return x_pred


def sample_sa_solver_pece(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """SA-Solver in PECE mode (``use_pece=True``)."""
    kwargs["use_pece"] = True
    return sample_sa_solver(model, x, sigmas, **kwargs)


def sample_ddpm(model: Denoiser, x: torch.Tensor, sigmas: torch.Tensor, *,
                generator: Optional[torch.Generator] = None, callback: Callback = None) -> torch.Tensor:
    """DDPM ancestral sampling (Ho et al., 2020) in Karras σ space via the VP
    mapping ``alpha_cumprod = 1/(σ²+1)``. For SD/SDXL, not rectified flow."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        noise_pred = (x - denoised) / sigma                       # ε estimate
        x_vp = x / (1.0 + sigma ** 2).sqrt()                      # VE → VP latent
        alpha_cumprod = 1.0 / (sigma ** 2 + 1.0)
        alpha_cumprod_prev = 1.0 / (sigma_next ** 2 + 1.0)
        alpha = alpha_cumprod / alpha_cumprod_prev
        x_vp = (1.0 / alpha).sqrt() * (x_vp - (1.0 - alpha) * noise_pred / (1.0 - alpha_cumprod).sqrt())
        if bool(sigma_next > 0):
            std = ((1.0 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt()
            x_vp = x_vp + std * _noise_like(x, generator)
            x = x_vp * (1.0 + sigma_next ** 2).sqrt()             # VP → VE latent
        else:
            x = x_vp
    return x


SAMPLERS: dict[str, Denoiser] = {
    "euler": sample_euler,
    "heun": sample_heun,
    "heunpp2": sample_heunpp2,
    "euler_ancestral": sample_euler_ancestral,
    "euler_ancestral_anneal": sample_euler_ancestral_anneal,
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
    "stork2": sample_stork2,
    "infinity": sample_infinity,
    "infinity_realism": sample_infinity_realism,
    "infinity_omega": sample_infinity_omega,
    "infinity_aether": sample_infinity_aether,
    "lms": sample_lms,
    "lcm": sample_lcm,
    "ddpm": sample_ddpm,
    "sa_solver": sample_sa_solver,
    "sa_solver_pece": sample_sa_solver_pece,
    "secant": sample_secant,
    "secant_anneal": sample_secant_anneal,
    "exp_heun_2_x0": sample_exp_heun_2_x0,
    "uni_pc": partial(sample_uni_pc, variant="bh1"),
    "uni_pc_bh2": partial(sample_uni_pc, variant="bh2"),
    "cogent": sample_cogent,
    "cogent3": sample_cogent3,
    "cogent3_pump": partial(sample_cogent3, pump_strength=0.08),
    "cogent3_pump_rate": partial(sample_cogent3, pump_strength=0.08, pump_h_ref=PUMP_H_REF),
}


def get_sampler(name: str):
    """Look up a sampler function by name (see :data:`SAMPLERS`)."""
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
