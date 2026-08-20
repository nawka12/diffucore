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
    "sample_infinity_nano",
    "sample_infinity_omega",
    "sample_infinity_aether",
    "sample_lms",
    "sample_lcm",
    "sample_ddpm",
    "sample_sa_solver",
    "sample_sa_solver_pece",
    "sample_secant",
    "sample_secant_anneal",
    "sample_dpmpp_2m_anneal",
    "sample_uni_pc_anneal",
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
    """Euler-ancestral with a σ-annealed ``eta`` (rectified-flow only).

    Identical to :func:`sample_euler_ancestral`'s ``flow`` branch, but the
    ancestral noise fraction is ``eta_i = eta_max·σ_i`` rather than a constant
    ``eta``: near-full ancestral re-noise at high σ (a stochastic burn-in that
    lets an imperfect / merged velocity field average out its inconsistencies)
    tapering to a near-deterministic step as σ→0 (so low-σ detail isn't washed
    out, the failure mode of constant ``eta=1``). Pairs with a high-σ-dense
    schedule (e.g. ``linear_quadratic``) on rectified-flow merges. ``shift`` is
    accepted for kwarg uniformity (unused; σ_max == 1 needs no first-σ offset)."""
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
    """Exponential Heun, second order, in data-prediction (x0) and half-logSNR
    time. Deterministic single-step true Heun: predict x0 at the current sigma,
    take a first-order exponential (DPM-Solver++(1)) step to the next sigma,
    re-evaluate x0 there, then combine the two estimates with the exponential
    integrator's phi_1/phi_2 weights. Two model evaluations per step, no history
    reuse (unlike the multistep :func:`sample_dpmpp_2m`). Flow-aware via the
    half-logSNR mapping (``model_type``/``shift``).

    This is the deterministic (eta=0), full-step (r=1) special case of the SEEDS
    exponential SDE solver (Gonzalez et al., "SEEDS: Exponential SDE Solvers for
    Fast High-Quality Sampling from Diffusion Models", NeurIPS 2023,
    arXiv:2305.14267), built on the DPM-Solver++ exponential integrator (Lu et
    al., arXiv:2211.01095). ``solver_type`` picks the corrector: ``"phi_2"``
    (default) uses the phi_2-weighted Heun combination; ``"phi_1"`` the simpler
    trapezoidal (phi_1) average of the two x0 estimates."""
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
        phi_1 = (-h).expm1()                       # e^{-h} - 1  (== h·phi_1(-h))
        # Predictor: first-order exponential (DPM++(1)) step to sigma_next.
        x_pred = (sigma_next / sigma) * x - alpha_t * phi_1 * denoised
        denoised_2 = model(x_pred, sigma_next * s_in)
        # Corrector with the second x0 estimate at sigma_next.
        if solver_type == "phi_1":
            denoised_d = 0.5 * (denoised + denoised_2)
            x = (sigma_next / sigma) * x - alpha_t * phi_1 * denoised_d
        else:
            phi_2 = (phi_1 + h) / (-h)             # (e^{-h} - 1 + h)/(-h)  (== h·phi_2(-h))
            b2 = phi_2
            b1 = phi_1 - b2
            x = (sigma_next / sigma) * x - alpha_t * (b1 * denoised + b2 * denoised_2)
    return x


def _uni_pc_bh_update(model, x, model_prev, sigma_prev, lambda_prev,
                      sigma_t, lambda_t, s_in, order, variant,
                      *, eta: float = 0.0, s_noise: float = 1.0,
                      generator: Optional[torch.Generator] = None):
    """One UniPC predictor + corrector step in data-prediction (x0) form.

    Builds the predictor from the last ``order`` x0 estimates (``model_prev``,
    newest last) and their half-logSNRs, then re-evaluates the model at the
    predicted point to apply the corrector. Returns ``(x_t, model_t)``; ``model_t``
    is the corrector's x0 evaluation at ``sigma_t``, reused as the next step's
    newest history (so UniPC stays ~1 evaluation per step). ``variant`` selects
    the ``B(h)`` solver type (``"bh1"``/``"bh2"``).

    ``eta > 0`` turns the step stochastic the same way the exponential multistep
    SDE solvers (:func:`sample_dpmpp_2m_sde`) do: the exponential weights use the
    η-folded step ``hh = -h·(1+η)`` (the ``phi``/``B(h)`` terms only — the ``rks``
    step-size *ratios* are pure geometry and stay on ``h``), the first-order carry
    is contracted by ``e^{-h·η}``, and Gaussian noise of std
    ``σ_t·sqrt(-expm1(-2·h·η))·s_noise`` is re-injected after the corrector.
    ``eta=0`` leaves every term untouched, so the deterministic UniPC step is
    recovered bit-for-bit (``-h·(1+0)==-h``, ``e^{0}==1``, ``std==0``)."""
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

    hh = -h * (1.0 + eta)                    # data-prediction (x0) form, η-folded
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

    # Predictor: first-order exponential step plus the higher-order residual.
    # ``e^{-h·η}`` contracts the carry (η=0 ⇒ factor 1, exact UniPC).
    x_t_ = (sigma_t / sigma_prev_0) * (-h * eta).exp() * x - alpha_t * h_phi_1 * m0
    if D1s is not None:
        if order == 2:                       # closed form for the 2nd-order case
            rhos_p = torch.tensor([0.5], device=device, dtype=b.dtype)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        pred_res = combine(rhos_p, D1s)
    else:
        pred_res = 0.0
    x_t = x_t_ - alpha_t * B_h * pred_res

    # Corrector: re-evaluate x0 at the predicted point and fold it in.
    model_t = model(x_t, sigma_t * s_in)
    if order == 1:
        rhos_c = torch.tensor([0.5], device=device, dtype=b.dtype)
    else:
        rhos_c = torch.linalg.solve(R, b)
    corr_res = combine(rhos_c[:-1], D1s) if D1s is not None else 0.0
    D1_t = model_t - m0
    x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)

    # σ-annealed ancestral re-noise (rectified-flow SDE), as in dpmpp_2m_sde.
    if eta > 0 and s_noise > 0:
        std = (-2.0 * h * eta).expm1().neg().sqrt()
        x_t = x_t + _noise_like(x, generator) * sigma_t * std * s_noise
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
    """UniPC: Unified Predictor-Corrector multistep solver (Zhao et al., "UniPC:
    A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion
    Models", NeurIPS 2023, arXiv:2302.04867). Data-prediction (x0) form.

    Each step predicts the next latent from the last ``order`` x0 estimates, then
    re-evaluates the model once at the prediction to apply the corrector; that
    evaluation becomes the next step's newest history, so UniPC stays ~one model
    evaluation per step despite being predictor-corrector. ``variant`` is the
    ``B(h)`` solver type: ``"bh1"`` or ``"bh2"`` (bh2 often edges ahead at very
    low step counts). ``lower_order_final`` ramps the order back down over the
    last steps for stability. Flow-aware via the half-logSNR map
    (``model_type``/``shift``)."""
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
        if bool(sigma_t == 0):           # final clean step: land on the x0 estimate
            x = model_prev[-1]
            break
        cur_order = min(order, len(model_prev))
        if lower_order_final:
            cur_order = min(cur_order, n - step)   # ramp order down near σ→0
        x, model_t = _uni_pc_bh_update(
            model, x, model_prev, sigma_prev, lambda_prev,
            sigma_t, lambda_fn(sigma_t), s_in, cur_order, variant,
        )
        if callback is not None:
            callback(step, sigma_t, x, model_t)
        push(sigma_t, model_t)
    return x


def sample_uni_pc_anneal(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 0.2,
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    order: int = 3,
    variant: str = "bh2",
    lower_order_final: bool = True,
    model_type: str = "flow",
    shift: float = 1.0,
) -> torch.Tensor:
    """UniPC predictor-corrector with a σ-annealed ancestral noise level
    (rectified-flow only).

    The highest-order deterministic core this engine has — UniPC's unified
    predictor-corrector multistep (:func:`sample_uni_pc`, Zhao et al., NeurIPS
    2023, arXiv:2302.04867) — fitted with the σ-annealed ancestral burn-in of the
    ``*_anneal`` family (:func:`sample_dpmpp_2m_anneal`,
    :func:`sample_euler_ancestral_anneal`). The ancestral fraction is
    ``eta_i = eta_max·σ_i``: a near-full stochastic re-noise at high σ (which
    averages out a merged / imperfect velocity field's inconsistencies — the
    reason the stochastic ``er_sde`` is the robust default on these models)
    tapering to a deterministic step as σ→0 (so the low-σ detail UniPC resolves
    so cheaply isn't washed out — the failure mode of constant ``eta``).

    It is the strict upgrade of :func:`sample_dpmpp_2m_anneal`: same σ-annealed
    burn-in, same flow half-logSNR exponential-integrator family, but the
    deterministic core is UniPC's arbitrary-order predictor-corrector (which adds
    a corrector re-evaluation reused as the next step's history, so it stays ~one
    model evaluation per step) instead of the fixed 2nd-order DPM++(2M) multistep
    — more accuracy per NFE, which is where the step savings come from. Pairs with
    a high-σ-dense flow schedule (``beta`` / ``beta_mix`` / ``smoothstep``), same
    as its siblings.

    The stochasticity is folded in exactly as :func:`sample_dpmpp_2m_sde` folds
    ``eta`` into an exponential multistep (see :func:`_uni_pc_bh_update`): the
    ``phi``/``B(h)`` weights use ``hh = -h·(1+η)``, the carry is contracted by
    ``e^{-h·η}``, and noise of std ``σ_t·sqrt(-expm1(-2·h·η))·s_noise`` is
    re-injected. ``eta_max=0`` recovers the deterministic UniPC step bit-for-bit
    (same ``variant``/``order``). ``variant`` is the ``B(h)`` solver type
    (``"bh1"``/``"bh2"``; ``bh2`` — the default here — often edges ahead at very
    low step counts); ``lower_order_final`` ramps the order down over the last
    steps. ``shift`` offsets the first σ off 1.0 for the half-logSNR map.

    To keep that high-order core from amplifying the injected noise, the predictor
    order is **ramped up with decreasing σ**: while η > 0 the order is held near 1
    (a noise-robust first-order exponential step, no divided-difference residual)
    at high σ, rising toward ``order`` as σ→0 where the noise has annealed away and
    the x0 history is clean. The ramp is gated on η > 0, so the ``eta_max=0``
    degradation to deterministic UniPC is untouched. It widens the usable noise
    range markedly: a GPU sweep on Anima collapsed the image to a washed-out ghost
    at ``eta_max=1.0`` *without* the ramp, but kept a coherent (if softer) image
    *with* it. Quality is still highest near ``eta_max=0`` (deterministic UniPC is
    the cleanest), so ``eta_max`` defaults to a low 0.2 and behaves as a small
    stochastic-diversity dial, with the order-ramp as the safety net against
    over-cranking rather than a route to a quality gain."""
    if model_type != "flow":
        raise ValueError("uni_pc_anneal is rectified-flow only (model_type='flow')")
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
        if bool(sigma_t == 0):           # final clean step: land on the x0 estimate
            x = model_prev[-1]
            break
        # σ-annealed ancestral fraction (eta = eta_max·σ_cur, as in the *_anneal
        # family): full burn-in at σ≈1, deterministic as σ→0. σ_cur is where x
        # currently lives — the newest history sigma.
        sigma_cur = float(sigma_prev[-1].clamp(0.0, 1.0))
        eta = eta_max * sigma_cur
        cur_order = min(order, len(model_prev))
        if lower_order_final:
            cur_order = min(cur_order, n - step)   # ramp order down near σ→0
        # Stochastic order-ramp-UP: UniPC's higher-order terms are divided
        # differences of the x0 history, which AMPLIFY the ancestral noise injected
        # at high σ (η = η_max·σ is largest there, and last step's noise lands in
        # this step's history). While noise is being injected (η > 0) hold the
        # order low at high σ — a near-1st-order, noise-robust step — and let it
        # rise toward full order as σ→0, where η, and the noise it leaves behind,
        # has annealed away and high-order extrapolation is safe. η = 0 (eta_max=0,
        # or σ→0) leaves the cap at full order, so the deterministic-UniPC
        # degradation is preserved bit-for-bit. η/η_max = σ, so the ramp is
        # η_max-independent (it tracks position on the trajectory, not noise scale).
        if eta > 0:
            cur_order = min(cur_order, 1 + int((1.0 - sigma_cur) * order))
        x, model_t = _uni_pc_bh_update(
            model, x, model_prev, sigma_prev, lambda_prev,
            sigma_t, lambda_fn(sigma_t), s_in, cur_order, variant,
            eta=eta, s_noise=s_noise, generator=generator,
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
    """SECANT-ANNEAL: σ-annealed ancestral burn-in handing off to a 2nd-order
    x0-secant refinement (rectified-flow only).

    A union of this repo's two flow samplers, split by σ because their useful
    ranges are complementary. Each step is built from:

      1. σ-annealed ancestral re-noise (:func:`sample_euler_ancestral_anneal`):
         the ancestral fraction is ``eta_i = eta_max·σ_i``, so high-σ steps get a
         near-full stochastic burn-in that lets an imperfect / merged velocity
         field average out its inconsistencies, tapering to deterministic as σ→0.
      2. A 2nd-order x0-secant correction (:func:`sample_secant`): the
         deterministic core extrapolates the x0 estimate along the secant through
         the previous and current x0 (toward ``σ_down``), blended in by
         ``beta = curvature·(1−r)·(1−σ)`` so it only acts at low σ, where x0 is
         reliable and detail refinement matters.

    The two are anti-correlated by construction — ``eta`` is large exactly where
    ``beta≈0`` and vice-versa — so high σ behaves like ``euler_ancestral_anneal``
    and low σ like deterministic ``secant``, with a smooth handoff between. The
    hybrid is an exact generalization of both: ``curvature=0`` recovers
    ``euler_ancestral_anneal``; ``eta_max=0`` recovers deterministic ``secant``.
    ``shift`` is accepted for kwarg uniformity (unused; σ_max == 1 needs no
    first-σ offset)."""
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
            # x0 for the deterministic core: hold it fixed (⇒ Euler-equivalent
            # reconstruction) unless a usable secant exists, then extrapolate it
            # toward σ_down, gated to trust only low σ (the SECANT correction).
            if old_x0 is None or bool((sigma - old_sigma).abs() < 1e-8):
                x0_eff = denoised
            else:
                x0_slope = (denoised - old_x0) / (sigma - old_sigma)
                x0_pred = denoised + x0_slope * (sigma_down - sigma)
                r = ((sigma_down - sigma).abs() / sigma).clamp(0.0, 1.0)
                reliable = (1.0 - sigma).clamp(0.0, 1.0)
                beta = float(curvature) * (1.0 - r) * reliable
                x0_eff = (1.0 - beta) * denoised + beta * x0_pred
            # Deterministic rectified-flow reconstruct to σ_down holding ε fixed
            # (ε = x + (1−σ)·d), then the annealed ancestral re-noise to σ_next.
            eps_i = x + (1.0 - sigma) * d
            x = (1.0 - sigma_down) * x0_eff + sigma_down * eps_i
            if eta > 0 and s_noise > 0:
                x = (alpha_next / alpha_down) * x + _noise_like(x, generator) * s_noise * renoise_coeff
        old_x0 = denoised
        old_sigma = sigma
    return x


def sample_dpmpp_2m_anneal(
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
    """DPM-Solver++(2M) with a σ-annealed ancestral noise level (rectified-flow only).

    The "good and fast" counterpart to :func:`sample_secant_anneal`: it differs
    from that sampler in exactly one way — the deterministic core is the DPM++(2M)
    flow exponential integrator (:func:`sample_dpmpp_2m_sde`'s midpoint form: a
    2nd-order multistep in half-logSNR space, one evaluation per step) instead of
    the σ-secant. The σ-secant *gates itself off* at low step counts (its
    correction weight ``∝ (1 − |Δσ|/σ)`` collapses as Δσ grows), so it degrades to
    1st-order Euler exactly when steps are scarce; the 2M core stays genuinely
    2nd-order there, which is where the step savings come from.

    The annealed ancestral burn-in is identical to
    :func:`sample_euler_ancestral_anneal`: ``eta_i = eta_max·σ_i`` — near-full
    stochastic re-noise at high σ (averaging out an imperfect / merged velocity
    field) tapering to a deterministic step as σ→0. Pairs with a high-σ-dense flow
    schedule (``beta`` / ``flow`` / ``smoothstep``), same as its siblings.

    ``eta_max=0`` makes every step deterministic, recovering the DPM++(2M) flow
    multistep exactly — bit-identical to ``dpmpp_2m_sde`` with ``eta=0`` (the same
    flow half-logSNR map, midpoint form). It is *not* bit-identical to the
    standalone ``dpmpp_2m``, which applies the VE logSNR map to flow as-is; this
    uses the rectified-flow half-logSNR, the correct one for a CONST model.
    ``shift`` offsets the first σ off 1.0 for that map."""
    if model_type != "flow":
        raise ValueError("dpmpp_2m_anneal is rectified-flow only (model_type='flow')")
    if len(sigmas) <= 1:
        return x
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
            # σ-annealed ancestral fraction (eta = eta_max·σ, as in
            # euler_ancestral_anneal): full burn-in at σ≈1, deterministic as σ→0.
            eta = eta_max * float(sigma.clamp(max=1.0))
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            if old_denoised is not None:
                rr = h_last / h
                x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / rr) * (denoised - old_denoised)
            if eta > 0 and s_noise > 0:
                x = x + _noise_like(x, generator) * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
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

    The first term is an MSE-optimal shrinkage measured from the *coherence* of
    two consecutive x0 differences; the second is a step-size floor. ``old_diff
    is None`` (no second difference yet) gives the floor alone.

    A multistep sampler's 2nd-order term is built from ``D_i = x0_i - x0_{i-1}``.
    Model the denoiser output as signal plus per-step noise, ``x0_i = f_i + n_i``
    (``n_i`` iid, energy ``E‖n‖² = v``) — the noise being everything the step's
    x0 estimate got wrong, dominated on a stochastic sampler by the ancestral
    noise injected into ``x`` last step. Then, writing ``S = ‖Δf‖²`` and assuming
    ``Δf`` varies slowly across a step (the same assumption the 2nd-order term
    itself makes)::

        <D_i, D_{i-1}> = S - v        (the shared -n_{i-1} term is anticorrelated)
        ‖D_i‖² = ‖D_{i-1}‖² = S + 2v

    so the cosine ``rho`` between them measures the derivative estimate's SNR:
    ``rho = (S - v)/(S + 2v)``, i.e. ``v/S = (1 - rho)/(1 + 2 rho)``. Substituting
    that into the Wiener shrinkage factor that minimises ``E‖psi·D_i - Δf‖²`` —
    ``psi = S/(S + 2v)`` — collapses to a straight line::

        psi = (1 + 2·rho) / 3

    ``rho = 1`` (clean, straight trajectory) ⇒ ``psi = 1``, the undamped textbook
    coefficient; ``rho = -1/2`` (pure noise, the floor of the model above) ⇒
    ``psi = 0``, no correction at all. Clamped to ``[0, 1]``.

    Reduced per batch sample (over every dim but the first), so the estimate
    averages over the whole latent — tens of thousands of elements, which makes
    the cosine a precise statistic rather than a noisy one.

    Curvature in the trajectory also lowers ``rho`` (the derivation assumes
    ``Δf_i ≈ Δf_{i-1}``), and there it is measuring the wrong thing: curvature is
    when the 2nd-order term is *most* needed, not least — and that is exactly the
    coarse-step regime. Hence the floor: ``1 - e^(-h)`` is the ``phi``-weight the
    exponential integrator already multiplies this term by, so the rule is "never
    damp the correction below the weight the step itself gives it". It is not a
    tuned constant, and it vanishes as ``h -> 0`` (fine steps, where the coherence
    reading is trustworthy and can damp all the way to zero).

    **Spatial reduction** (``reduce``). ``"all"`` (the default) reduces over every
    dim but the batch — the gate described above, bit-for-bit the family's
    shipped behaviour. ``"per_channel"`` reduces over the spatial axes ``(H, W)``
    of a 4-D latent only, giving a length-``C`` vector of shrinks (broadcast
    ``[B, C, 1, 1]``) so each channel of the correction term is scaled by its own
    coherence reading. The per-channel gate is a *different* estimator, not a
    refinement of the global one: with equal per-channel cosines the global cosine
    equals them only when the channel-norm vectors of the two differences are
    proportional (the global cosine is a sub-convex combination of the channel
    cosines and 0). Per-channel is the cogent4 spatial gate, default-off; on
    non-4-D latents there are no spatial axes, so ``"per_channel"`` falls back to
    ``"all"`` (bit-for-bit ``cogent3``), matching the documented non-4-D path.

    **Measurement hook** (``stats_out``). When given a dict, the raw statistics
    the gate computes are written into it — the pre-floor ``rho``, the lag-1
    noise-energy and signal-energy estimates the model above implies
    (``v_est = (‖D_i‖² - <D_i, D_{i-1}>)/3``, ``s_est = (2·<D_i, D_{i-1}> + ‖D_i‖²)/3``),
    the unfloored Wiener shrink ``psi_linear``, and whether the step-size floor
    won over it. These are the family's first direct readings of its own
    measurement, and they are the quantities the measurement-falsification
    harness (``scripts/ab_cogent3.py --measure``) logs against known truth.
    ``stats_out`` is write-only: values are detached logging tensors, so
    collecting them neither changes the returned gate nor retains its autograd
    graph.
    Shapes follow ``reduce``: ``[B]`` for ``"all"``, ``[B, C]`` for
    ``"per_channel"``.
    """
    # Validate before the bootstrap return. Otherwise an invalid mode is
    # silently accepted on the first correctable step and only fails once more
    # history happens to exist, making validation depend on the step count.
    _validate_gate_reduce(reduce)
    floor = (-h).expm1().neg()
    if old_diff is None:
        if stats_out is not None:
            stats_out["bootstrap"] = True
            stats_out["floor_active"] = True
        return floor
    if reduce == "per_channel":
        if diff.ndim == 4:
            dims = (2, 3)                                   # spatial (H, W) only
        else:
            reduce = "all"                                  # no spatial axes to reduce over
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
        # This is a logging hook, not a differentiable auxiliary output. Keep
        # the sampled result's graph intact while preventing a stats list from
        # retaining the denoiser graph for every step.
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
    """COGENT: coherence-gated exponential multistep with a σ-annealed ancestral
    noise level. One model evaluation per step; all model families.

    The deterministic core is the exponential-integrator 2nd-order multistep in
    half-logSNR space, data-prediction form (DPM-Solver++(2M) SDE, Lu et al. 2022,
    arXiv:2211.01095 — the same core as :func:`sample_dpmpp_2m_sde` /
    :func:`sample_dpmpp_2m_anneal`), and the ancestral fraction is the
    ``*_anneal`` family's ``eta_i = eta_max·σ_i``: a near-full stochastic burn-in
    at high σ (which lets an imperfect / merged velocity field average out its
    inconsistencies) tapering to a deterministic step as σ→0 (so low-σ detail
    isn't washed out).

    What is new here is the **gate**. Those two ingredients fight each other: the
    2nd-order term is a divided difference of the x0 history, so it amplifies both
    the ancestral noise the burn-in injects and whatever the model itself got
    wrong. Every sampler in this family answers that with a hardcoded rule —
    :func:`sample_secant` gates its correction by ``curvature·(1−|Δσ|/σ)·(1−σ)``,
    :func:`sample_uni_pc_anneal` ramps its order with ``σ``, :func:`sample_stork2`
    damps the derivative by a fixed ``C1(s) < 1/2`` — all of them proxies for "is
    this extrapolation trustworthy right now?" that never look at the data. A
    fixed rule has to be tuned for the worst case it might meet, so it over-damps
    a good model and under-damps a bad one.

    COGENT measures it instead, and the measurement is cheap: two dot products
    per step. The correction is scaled by ::

        psi = max( (1 + 2·rho)/3 ,  1 − e^(−h) )        clamped to [0, 1]

    where ``rho = cos(D_i, D_{i-1})`` is the coherence of the last two x0
    differences. The first term is the MSE-optimal (Wiener) shrinkage for the
    derivative estimate implied by that coherence (derived in
    :func:`_coherence_gate`); the second is a **step-size floor**, and it is not a
    free parameter — ``1 − e^(−h)`` is the very ``phi``-weight this integrator
    multiplies the correction by, so the floor says "never damp the term below the
    weight the step itself gives it".

    Both halves are needed, and they cover each other's blind spot. The coherence
    term reads model quality: measured on a toy flow whose exact denoiser is known,
    ``rho ≈ 0.97`` with a clean model but ``≈ −0.25`` once the model carries
    high-frequency error, so a merged / imperfect field damps itself automatically
    while a good one keeps the full textbook coefficient. But coherence also drops
    on a sharply *curved* trajectory, where the 2nd-order term is needed most, not
    least — and that is exactly the coarse-step regime, so the ``h``-floor holds
    the correction up precisely there. Without the floor the sampler collapses to
    first order at low step counts; with it, it does not.

    Relative to :func:`sample_secant_anneal` this is a better solver on both axes.
    Deterministic accuracy against an exactly-integrated reference trajectory is
    ~2.3x better at matched steps (its core is a full 2nd-order exponential
    integrator, where the σ-secant is Euler plus a nudge capped at ``curvature``
    that self-gates to ~0 as steps get sparse). And on the same toy with a rough
    model error — the regime that motivates the whole annealed-ancestral family —
    it is 12–15% closer to the data law at 8 steps and 12–25% closer at 24–32
    steps, across error strengths and roughness scales. It gives up a few percent
    to ``secant_anneal`` in the 12–16 step band. Prefer 24+ steps, where its
    margin is largest.

    **Scheduler pairing differs from the rest of the family.** Its σ-secant
    siblings want a high-σ-dense schedule (``beta`` / ``smoothstep``); this one
    does not, because it inherits the λ-space exponential core's preference for
    *fine, smooth steps at the low-σ end*. Measured on the same toy, ``flow`` /
    ``simple`` / ``sgm_uniform`` (near-identical for flow models) are the safe
    default and ``linear_quadratic`` is the best at 24–32 steps under strong model
    error; ``beta`` / ``beta_mix`` / ``smoothstep`` land a coarser minimum λ-step
    after the shift map, which both costs accuracy and pins the gate's floor high
    enough that it can no longer damp. ``normal`` / ``infinity`` / ``infinity_htds``
    / ``kl_optimal`` are markedly worse again — but note this is a property of the
    core, not the gate: :func:`sample_dpmpp_2m_anneal` degrades on exactly the same
    schedules, and by more.

    ``eta_max=0`` makes every step deterministic; ``psi ≡ 1`` recovers
    :func:`sample_dpmpp_2m_anneal` exactly. The first correctable step has no
    second difference to measure against and simply runs at the floor.

    ``gate_reduce`` selects how the coherence gate reduces over the latent:
    ``"all"`` (the default) is the shipped global gate, bit-for-bit; the cogent4
    ``"per_channel"`` option reduces over the spatial axes of a 4-D latent only,
    scaling each channel of the correction by its own shrink. It is default-off
    (see :func:`_coherence_gate`).

    Family-agnostic: ``model_type="flow"`` (Anima / FLUX) uses the rectified-flow
    half-logSNR map, where ``eta_i = eta_max·σ_i`` literally; ``"ve"`` (SD / SDXL)
    uses the VE map, where the annealing variable is the equivalent noise fraction
    ``σ/(1+σ) = sigmoid(-lambda)`` — the same quantity σ already is on flow, so
    the anneal means the same thing on both. ``shift`` offsets the first σ off 1.0
    for the flow map.
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
            # σ-annealed ancestral fraction. On flow σ IS the noise fraction; on
            # VE the same quantity is σ/(1+σ) (both are sigmoid(-lambda)).
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
                # Same operand order as sample_dpmpp_2m_anneal, so psi == 1
                # reproduces it bit-for-bit rather than merely closely.
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
    """Scale factor for a multistep solver's second-divided-difference
    (3rd-order) term::

        psi = (2 + 3·rho) / 5                             clamped to [0, 1]

    where ``rho = cos(E_i, E_{i-1})`` is the coherence of two consecutive
    *second* differences of the x0 history — the curvature analogue of
    :func:`_coherence_gate`'s ``rho`` on first differences. The gate is
    evaluated per batch sample, reduced over every dim but the first (a cosine
    averaged over the whole latent, the same precise statistic as cogent's).

    Derivation, mirroring :func:`_coherence_gate`: model ``x0_i = f_i + n_i``
    with iid ``n_i`` of energy ``v``. Second differences are
    ``E_i = Δ²f_i + n_i − 2·n_{i-1} + n_{i-2}``, so two consecutive ones share
    the ``−2·n_{i-1}`` and ``n_{i-2}`` terms with opposite signs::

        <E_i, E_{i-1}> = S2 − 4v        ‖E_i‖² = ‖E_{i-1}‖² = S2 + 6v

    with ``S2 = <Δ²f_i, Δ²f_{i-1}>``. The Wiener shrink that minimises
    ``E‖psi·E_i − Δ²f‖²`` is ``psi = S2/(S2 + 6v)``; writing ``u = v/S2`` and
    ``rho = (1 − 4u)/(1 + 6u)`` gives ``u = (1 − rho)/(4 + 6·rho)`` and::

        psi = 1/(1 + 6u) = (2 + 3·rho) / 5

    ``rho = 1`` (clean, smoothly-curving trajectory) ⇒ ``psi = 1``; the pure-
    noise floor of the model (``rho → −2/3``) ⇒ ``psi = 0``.

    There is deliberately **no step-size floor**, unlike the 2nd-order gate.
    cogent3's 3rd-order term is never load-bearing — worst case it reverts to
    the gated 2nd-order behaviour — so damping it to zero on untrustworthy
    curvature is never a failure mode. And the coherence reading itself does
    not dual-read curvature the way the first-difference ``rho`` does: ``rho``
    measures whether the second differences are consistent, which is precisely
    what decides whether extrapolating them is trustworthy. ``old_second_diff
    is None`` (no curvature history yet) returns 1.0; the caller substitutes
    its own bootstrap for the very first 3rd-order-capable step, where the raw
    Wiener term would be 1.0 regardless (see :func:`sample_cogent3`).

    ``reduce`` mirrors :func:`_coherence_gate`: ``"all"`` (default) reduces
    over every dim but the batch — the shipped behaviour, bit-for-bit —
    ``"per_channel"`` reduces over the spatial axes of a 4-D latent only
    (falling back to ``"all"`` on non-4-D inputs), for the cogent4 spatial
    gate. Per-channel is default-off and never engaged unless the sampler is
    asked for it.
    """
    # As with the first-difference gate, validation must not depend on whether
    # enough history exists to leave the bootstrap path.
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
    generator: Optional[torch.Generator] = None,
    callback: Callback = None,
    model_type: str = "ve",
    shift: float = 1.0,
    gate_reduce: str = "all",
    gate_stats: Optional[list] = None,
) -> torch.Tensor:
    """COGENT3: COGENT's measured gate carried to third order. One model
    evaluation per step; all model families.

    The deterministic core is the third-order DPM-Solver++ (3M) exponential
    integrator in half-logSNR space, data-prediction form (Lu et al. 2022,
    arXiv:2211.01095 — the same core as :func:`sample_dpmpp_3m_sde` with
    ``eta=0``, which this reproduces bit-for-bit when both gates are the
    identity — plus the ``*_anneal`` family's ``eta = eta_max·σ`` ancestral
    burn-in). The gates are :func:`sample_cogent`'s, applied to the 2nd-order
    term, and a new second gate on the 3rd-order term:

    * **2nd-order term** — built from the first divided difference of the x0
      history, scaled by cogent's ``psi_1 = max((1 + 2·rho_1)/3, 1 − e^(−h))``,
      ``rho_1 = cos(D_i, D_{i-1})`` of consecutive first differences, with the
      step-size floor (a coarse step needs the term whatever its SNR).
    * **3rd-order term** — built from the *second* divided difference, scaled
      by a new measured Wiener shrink ``psi_2 = (2 + 3·rho_2)/5``,
      ``rho_2 = cos(E_i, E_{i-1})`` of consecutive second differences
      (:func:`_cogent3_curvature_gate`), with **no floor**.

    The 3rd-order term is the noisiest quantity in this family — it is a
    difference of differences, so it amplifies both the ancestral burn-in's
    injected noise and whatever the model itself got wrong by the square. That
    is why the 2nd-order-only family (:func:`sample_cogent`, ``stork2``)
    exists at all, and it is the reason cogent3's second gate exists too:
    ``rho_2`` reads directly whether the curvature itself is trustworthy, and
    the worst case is not a failure but a graceful degradation.

    Measured on the GMM-flow toy of ``scripts/ab_cogent3.py`` (Anima's
    shift=3.0, exact optimal denoiser, 4000-step Euler reference), relative to
    :func:`sample_cogent` and the ungated 3M core (``dpmpp_3m_sde`` eta=0 —
    what cogent3 would be with no gates at all):

    * **Clean model, deterministic accuracy (RMSE vs the exact ODE).** cogent3
      beats cogent at every step count measured (8..32): ~6% lower error at 8
      steps, ~4-10% at 12-32. It sits between cogent and the ungated 3M core —
      the gate knowingly spends a few percent of the core's clean-model edge
      (8 steps: 0.132 vs 0.127) to buy robustness.
    * **Rough / merged model, stochastic burn-in (eta_max=1.0, energy distance
      to data).** The gate converts the 3M core's fragility into cogent-level
      robustness: at 24-32 steps cogent3 is 22-27% closer to the data law than
      the ungated 3M (which degrades to well behind plain cogent there), and
      at 8 steps cogent3 is clearly better than cogent (e.g. freq=12, tau=0.35:
      0.196 vs 0.238). At 16-32 steps it ties cogent within run noise.
      Deterministically with a rough model it also tracks cogent within noise.

    Prefer 24+ steps, as with cogent; scheduler pairing follows the exponential
    core (``flow`` / ``simple`` / ``sgm_uniform``).

    ``eta_max`` is the same ``*_anneal`` knob (``eta_max=0`` fully
    deterministic); ``psi_1 = psi_2 ≡ 1`` with ``eta_max=0`` reproduces
    :func:`sample_dpmpp_3m_sde` with ``eta=0`` bit-for-bit. Family-agnostic:
    ``model_type="flow"`` (Anima / FLUX) anneals on σ, ``"ve"`` (SD / SDXL) on
    ``σ/(1+σ)`` — the same quantity, as in cogent. ``shift`` offsets the first
    σ off 1.0 for the flow map.

    ``gate_reduce`` selects the coherence gate's reduction over the latent:
    ``"all"`` (the default) is the shipped global gate, bit-for-bit; the cogent4
    ``"per_channel"`` option reduces over the spatial axes of a 4-D latent only,
    scaling each channel of the correction by its own shrink (falling back to
    ``"all"`` on non-4-D latents — see :func:`_coherence_gate`). Default-off.

    ``gate_stats``, when given a list, collects one dict per correctable step
    with the raw measurement statistics the 2nd-order gate computed — the
    pre-floor ``rho``, the lag-1 estimates ``v_est`` and ``s_est``, the
    unfloored ``psi_linear``, and whether the step-size floor won (see
    :func:`_coherence_gate`). Collecting stats never changes the sampled output.
    This is the instrumentation the measurement-falsification harness
    (``scripts/ab_cogent3.py --measure``) logs through, so the estimator under
    test is the sampler's own code path rather than a reimplementation.

    ----

    **The high-σ coherence pump** (``pump_strength > 0``; registered as the
    ``cogent3_pump`` sampler, off by default here).

    This is :func:`sample_infinity_aether`'s one load-bearing mechanism on
    rectified flow, isolated from the band-pass stack it ships with and given a
    hard low-σ shutoff. Aether adds grain scaled by ``1 − C`` (the
    structure-tensor coherence of the denoised prediction) *on top of* a
    completed Euler step, so the next model call sees a latent noisier than the
    σ it is handed and must explain the excess as signal — a structure-
    generation pump aimed exactly at the regions that have not yet committed,
    and held off the contours that have. At high σ the neighbouring modes it
    hops between differ in coarse properties (mass, pose, proportion), which is
    why aether reads character stature well; at low σ they differ in texture,
    which is why the same mechanism turns skin and gradients to mush.

    So the pump is gated to the top of the schedule and hard-stopped:

    ```
    nu   = pump_strength · sigma_next · clamp((sigma_frac − pump_end)/pump_span, 0, 1)
    x   += nu · (1 − C) · noise
    ```

    Two details are the whole difference from aether, which pins both to
    constants tuned against SD's σ range and is a substantially different
    sampler on flow as a result (:func:`sample_infinity_aether` documents the
    general problem; measured at 24 flow steps its ``0.30·σ_next``-vs-``0.03``
    terminal floor injects 0.0289 into the *finished* latent, 34× what the same
    code injects on an SDXL karras schedule):

    * **The gate uses ``sigma_frac``** — the family-invariant ``σ`` (flow) /
      ``σ/(1+σ)`` (VE) coordinate this sampler already computes for ``eta`` —
      so ``pump_end`` means the same place on the trajectory in both families.
    * **The amplitude uses absolute ``sigma_next``**, because the noise is being
      added to a latent whose own noise level is σ. A fixed absolute amplitude
      is negligible at SD's σ_max of 14.6 and enormous at flow's 1.0.

    Because the pump lives in the ``sigma_next != 0`` branch it can never touch
    the final latent: the last step is ``x = denoised``, unpumped.

    The pump perturbs x, so it could in principle decorrelate consecutive x0
    predictions and drive ``rho_1``/``rho_2`` — and the gates down with them —
    which would leave the 3rd-order term as dead weight while it runs. Probed
    on two toys (a Gaussian-prior denoiser and a 4-mode mixture, both 2-D and
    structured) it does not: mean ``psi_1`` moved 0.566 → 0.567 and ``psi_2``
    0.143 → 0.144 at ``pump_strength=0.08``. The pump and the exponential core
    appear to occupy different scales rather than fight, which is why they
    compose — but that is a measurement on toys, not a proof.

    Defaults pump at full strength above ``sigma_frac`` 0.70, ramp to zero at
    0.45, off below — roughly 17 pumped / 5 ramping / 9 clean steps on a
    30-step ``smoothstep`` flow schedule. Raise ``pump_strength`` or lower
    ``pump_end`` for more coarse-structure revision at the cost of detail.
    ``pump_strength=0`` is bit-for-bit plain cogent3, drawing no extra noise.

    **4-D latents only when the pump is on** (the structure tensor is a 2-D
    convolution) — FLUX packs to a token sequence, so use plain ``cogent3``
    there.

    **What it turned out to be good at.** This was built to chase aether's
    character-stature strength. On real images (user A/B, 28–32 steps,
    ``beta_mix``) the striking win is instead **prompt coherency**, with
    stature improved but no longer the headline. That fits the mechanism better
    than the original target did: stature is a property of an object the model
    has already decided to draw, so it lives in *coherent* structure that the
    ``1 − C`` weighting deliberately protects. Prompt adherence is about what
    gets drawn at all in regions still ambiguous — precisely the low-``C``
    regions the pump perturbs — and each perturbation forces the CFG-guided
    model to re-answer "what belongs here, given the conditioning?". An
    accurate solver that commits to a partially prompt-compliant layout will
    refine *that* layout faithfully; it has no mechanism to restructure. This
    is a spatially-selective, coherence-gated relative of why restart /
    stochastic sampling improves text alignment over an ODE.

    Two practical consequences. The pump earns its keep in the *high-σ* band,
    so raising ``pump_end`` (shutting off earlier) is the first thing to try if
    detail feels soft — it should keep the coherency win while returning steps
    to refinement. And do not pair it with a CFG guidance interval that drops
    the uncond pass inside the pumped band: the pump's value *is* CFG
    re-deciding, so switching CFG off there removes the point of it.

    **Offline evidence, stated honestly.** The design rests on a mechanism
    argument plus measurements of what aether actually does on flow. On a
    4-mode toy whose modes differ only in a figure's height/width ratio, with
    mixture weights skewed 0.70 toward the stockiest, the pump moved sampled
    mode coverage toward the true weights (total-variation 0.0312 → 0.0238) but
    at 320 seeds that sits inside ±0.07 (2 s.e.) — **not significant**, and the
    toy never tested prompt adherence at all, which is what the real A/B found.
    What does reproduce offline is the *cutoff*: pumping all the way down
    (aether's behaviour, ``pump_end=0``) matched the gated version's coverage
    exactly while landing consistently further from the nearest mode (0.3253 vs
    0.3240, cogent3 0.3234) — the low-σ half costs sharpness and buys no
    coarse-structure benefit.
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
            # σ-annealed ancestral fraction, as in cogent / dpmpp_2m_anneal.
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
                # ψ₁ reads the raw first differences (same operands as cogent);
                # ψ₂ reads the second differences. First occurrence bootstraps
                # ψ₂ from ψ₁ (no curvature history yet).
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
            # Coherence pump. Ramp is on the family-invariant sigma_frac, amplitude
            # on absolute sigma_next; both zero below pump_end, where no noise is
            # drawn at all so pump_strength=0 leaves the generator stream untouched.
            if pump_strength > 0:
                ramp = (1.0 if pump_span <= 0 else
                        min(1.0, max(0.0, (sigma_frac - pump_end) / pump_span)))
                if sigma_frac < pump_end:
                    ramp = 0.0
                if ramp > 0:
                    nu = pump_strength * float(sigma_next) * ramp
                    # Read off the denoised prediction, not the velocity: at low
                    # sigma the velocity is mostly residual noise and its
                    # coherence map says nothing about committed structure.
                    c = _structure_tensor_coherence(denoised.float(), multi_scale=True)
                    x = x + (nu * (1.0 - c)).to(x.dtype) * _noise_like(x, generator)
        denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
        h_1, h_2 = h, h_1
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


@lru_cache(maxsize=None)
def _rkg2_coeffs(s: int):
    """Closed-form stage coefficients of the ``s``-stage second-order
    Runge–Kutta–Gegenbauer (RKG2) method (Skaras & O'Sullivan, J. Comput. Phys.
    2021), as used by STORK-2 (see :func:`sample_stork2`).

    The stability polynomial is ``R_s(z) = a_s + b_s·C_s^{3/2}(1 + w1·z)`` with
    the shifted Gegenbauer polynomial ``C^{3/2}``; matching ``e^z`` to second
    order fixes ``w1 = 6/((s+4)(s-1))``, ``b_j = 4(j-1)(j+4)/(3j(j+1)(j+2)(j+3))``
    and ``a_j = 1 - (j+1)(j+2)/2·b_j``, and the Gegenbauer three-term recurrence
    turns ``R_s`` into an ``s``-stage Runge–Kutta cascade. ``b_0 = 1`` and
    ``b_1 = 1/3`` (so stage 1 is ``R_1(z) = 1 + w1·z`` exactly) and the stage
    abscissae ``c_j = (j²+j-2)/(s²+s-2)`` (with ``c_1 = c_2/3``) follow the
    method's published conventions. Returns ``(w1, c, stage)`` where ``c[j]`` is
    stage ``j``'s time offset as a fraction of the step and ``stage[j-2] =
    (mu_j, nu_j, mu_tilde_j, gamma_tilde_j)`` for ``j = 2..s``."""
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
    """STORK-2: Stabilized Taylor Orthogonal Runge–Kutta, second order (Tan et
    al., "STORK: Faster Diffusion And Flow Matching Sampling By Resolving Both
    Stiffness And Structure-Dependence", ICLR 2026, arXiv:2505.24210).
    Clean-room implementation from the paper, following the reference
    conventions (RKG2 coefficients with ``b_0 = 1, b_1 = 1/3`` and the
    ``c_j = (j²+j-2)/(s²+s-2)`` abscissae).

    Each step runs an ``stages``-stage Runge–Kutta–Gegenbauer cascade
    (:func:`_rkg2_coeffs`) on the σ-space ODE ``dx/dσ = (x − x0)/σ``, but the
    intermediate stage velocities are "virtual NFEs": Taylor expansions of the
    velocity in σ around the current point, with the derivatives estimated by
    divided differences of the *previous steps'* real evaluations — so the cost
    stays one model evaluation per step, like ``dpmpp_2m``/``ipndm``.
    Deterministic and model-agnostic (the raw σ-space ODE serves VE — SD/SDXL —
    and rectified flow — Anima/FLUX — alike, exactly as ``ipndm`` /
    ``res_multistep`` do).

    What the cascade buys, honestly: with ``taylor_order=1`` the whole
    super-step collapses algebraically to ``x + Δσ·v + C1(s)·Δσ²·v̇`` — a
    variable-step 2-step Adams–Bashforth (``ipndm_v`` order 2) whose
    derivative correction is *damped* from 1/2 to ``C1(s) < 1/2``
    (≈0.4628 at s=9, →1/2 as s→∞). The divided-difference ``v̇`` is the
    noisiest term of any multistep solver, and the paper's FID tables show
    the damping is worth real quality at practical step counts on flow models
    (STORK-2 beats Flow-UniPC/Flow-DPM++ at 7–10 NFE on SANA; STORK-4, whose
    ROCK4 coefficient tables are not redistributable, is better still).
    ``stages`` is therefore a robustness/accuracy dial, not a cost dial:
    *smaller* damps the derivative correction more (steadier on imperfect /
    merged models and stiff low-σ regions), *larger* approaches undamped AB2.
    ``stages=9`` is the paper's optimum for latent flow-matching models.

    ``taylor_order=2`` adds a second divided difference (``v̈``) and a
    ``Δσ³`` term through the cascade — sharper when the trajectory is smooth
    and steps are many, noisier at low step counts (the paper's flow-matching
    experiments prefer order 1). Derivative estimates here use exact
    nonuniform-grid divided differences (the reference's first-derivative
    3-point formula assumes uniform steps; ours reduces to it on uniform
    grids), and warmup degrades gracefully: Euler on the first step, then
    order-limited estimates until enough history exists. The final step lands
    on the x0 estimate, as this registry's other data-prediction samplers do."""
    if stages < 2:
        raise ValueError("stages must be >= 2")
    if taylor_order not in (1, 2):
        raise ValueError("taylor_order must be 1 or 2")
    s_in = x.new_ones([x.shape[0]])
    w1, c, stage_coeffs = _rkg2_coeffs(stages)
    hist_sigma: list[float] = []       # σ of the previous real evaluations
    hist_v: list[torch.Tensor] = []    # matching real velocities, newest last
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        d = to_d(x, sigma * s_in, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        dt = sigma_next - sigma
        s0 = float(sigma)
        # Velocity time-derivative estimates from the real-eval history (the
        # Taylor expansion treats v as a function of σ along the trajectory).
        # σ-collisions (possible at a schedule's σ_min floor) degrade the order.
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
            x = x + d * dt             # Euler warmup (no usable history yet)
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
    ``main`` @4f72d8f, 2026-07-17), with one deliberate correction — see
    *Step-size scaling* below.

    Euler on the σ-space ODE with an invariant-gated IIR correction: the first
    step is plain Euler, after which each step advances by ``d + correction``
    where the correction combines a *velocity* EMA of the first derivative
    difference (gain ``β₁``) and — from the third step — an *acceleration* EMA
    of the second difference (gain ``β₂``). Before stepping, three invariants
    gate the correction: (1) its mean magnitude is clamped to 50% of the
    derivative's, (2) it is halved if the derivative reversed direction
    (cosine vs. the previous step < 0), and (3) it is zeroed — a pure Euler
    step — when both trigger. Constants are upstream's fixed ``α₁=0.5, β₁=0.5,
    α₂=0.3, β₂=0.3`` — upstream briefly shipped signal-adaptive coefficients
    and reverted to fixed the same day, so no knobs are exposed here either.
    Deterministic; one model evaluation per step; model-agnostic (raw σ-space
    serves VE — SD/SDXL — and rectified flow — Anima/FLUX — alike, as
    ``ipndm``/``stork2`` do).

    **Step-size scaling (deviation from upstream).** Upstream filters the raw
    differences ``d − d_prev`` and applies the result with fixed gains, so the
    correction is AB2-consistent only when neighboring steps are equal; on a
    nonuniform grid it silently mis-scales. Here each difference is divided by
    the step it was taken over *before* entering the EMA, so the filter carries
    derivative estimates rather than raw differences::

        dd  ← (d − d_prev) / dt_prev                    ≈ d′(σ)
        vel ← (1−α₁)·vel + α₁·dd
        acc ← (1−α₂)·acc + α₂·(dd − dd_prev)/dt_prev    ≈ d″(σ)
        correction = β₁·dt·vel + β₂·dt²·acc

    On a uniform σ grid the ``dt`` factors cancel exactly and this reduces to
    upstream's recursion bit-for-bit (the EMA is linear, so a constant divisor
    passes straight through); on a nonuniform grid it is the AB2/AB3-consistent
    form of the same filter. Upstream's ``micro`` branch reaches for the same
    idea with a post-hoc ``h/(2·h_prev)`` factor on the filter *output*; that
    form measured strictly worse than this one on every grid tested, because it
    leaves the filter's memory dimensionally inconsistent.

    Measured against a converged reference on a nonlinear ODE (max abs error
    relative to Euler, 16–32 steps): ``flow``/``normal`` 0.28→0.09,
    ``infinity`` 0.35→0.12, and ``karras`` ρ=7 — which upstream's scaling
    could not integrate at all — 2.5→1.6 at 16 steps and 1.18→0.44 at 32,
    i.e. from *behind* Euler to comfortably ahead of it. The trade is real
    though: ``sgm_uniform``/``simple`` regress (0.27→0.66), as do
    ``kl_optimal``, ``linear_quadratic`` and marginally
    ``beta``/``beta_mix``/``smoothstep``. Both forms stay well ahead of Euler
    there, so the practical advice is unchanged — pair with ``flow``/``normal``
    or the matching ``infinity`` schedule — except that ``karras`` is no longer
    a trap.

    Upstream applies the correction through the final (σ→0) step and never
    resets the EMAs; both behaviors are kept. The magnitude clamp and the
    cosine test reduce over the *whole* tensor (upstream semantics), which
    couples batch entries — harmless here since the pipelines sample one
    latent at a time."""
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
        dd = (d - d_prev) / dt_prev             # ≈ d′(σ); dt-normalized
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
    """The ``realism`` branch's variance stabilizer: pull each channel's spread
    toward its running EMA by a fraction that is the product of three smooth
    asymptotes, with no thresholds anywhere.

    * ``deviation/(deviation+0.3)`` — how far this step's std has drifted from
      the EMA, relative. Zero drift, zero correction.
    * ``progress/(progress+0.2)`` — sampling progress, so the correction is
      inert during structure formation and near-full (0.83 at the end) during
      cleanup.
    * ``steps/(steps+8)`` — step count, which is upstream's Turbo/LCM guard: at
      4 steps the EMA has not converged to anything worth correcting toward, so
      the factor holds the whole product to 1/3.

    Unlike NQVP/AVN this reduces over the **batch** axis too (upstream's
    ``dim=(0, 2, 3)``), which couples batch entries — harmless here, since the
    pipelines sample one latent at a time. Float32, as with the other
    stabilizers: an fp16 ``std`` over a full feature map is where these break.
    The final ``[0.1, 10]`` clamp on the correction factor is upstream's guard
    against a near-uniform channel at an early step."""
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
    upstream @21084d9, 2026-07-21).

    Euler in x0 form — upstream writes ``x ← r·x − (r−1)·x0`` with
    ``r = σ_next/σ``, which is algebraically the same first-order step
    :func:`sample_infinity` takes — with one addition, the **variance
    stabilizer**: before each step, every channel's spatial standard deviation
    is pulled toward its running EMA by a smoothly-ramped fraction (see
    :func:`_variance_stabilize`). The stated target is the distribution drift
    that non-uniform step sizes cause, which the sine-perturbed ``infinity``
    schedule produces by design. Deterministic; one model evaluation per step;
    model-agnostic.

    **This branch was rewritten upstream on 2026-07-21 and is now a different
    sampler than the one we shipped through 2026-07-25.** Gone: the x0-space
    velocity/acceleration EMA correction, the three invariant gates (magnitude
    clamp, cosine reversal test, zero-on-both), the self-correcting scheduler,
    and — the reason this matters here — the adaptive noise injection that gave
    the branch its name. That injection was ``γ·σ·ε`` with ``γ`` saturating at
    0.20, an *absolute* noise scale that took no account of how far the step
    actually travelled, and it is why we had this sampler restricted to SD/SDXL:
    on Anima at flow shift=3.0 over 32 steps the first step injected 18.8× what
    it removed (46× under the ``infinity`` scheduler, whose sine warp shrinks
    that first gap further), with 28 of 32 steps over-injecting. With the
    injection deleted the sampler is deterministic and carries no absolute noise
    scale at all, so **the SD/SDXL restriction is lifted** and Anima and FLUX
    offer it again.

    The trade is that "realism" no longer means grain. What is left is the
    gentlest member of the family — plain Euler plus a spread correction that
    is deliberately inert for the first third of the trajectory. If you want the
    old behavior, it is not recoverable from any upstream branch; it was
    deleted, not moved.

    **4-D latents only**, which is a new restriction and a different one from
    the old branch's. The stabilizer takes a per-channel statistic over the
    spatial axes, so it needs ``[B, C, H, W]``. SD/SDXL and Anima qualify; FLUX
    patchifies to ``[B, L, C·p²]`` before sampling, where dim 1 is a token index
    and a "per-channel" spread over it means nothing. The engine keeps realism
    out of the FLUX dropdown; the guard below is for direct library callers.

    **Port deviations:** the stabilizer runs in float32 (see its docstring).
    Upstream's ``sigmas[-1]`` tolerance clamp is omitted — our schedules always
    terminate at exactly 0 — and its ``while``-loop scaffolding is written as a
    ``for``, which is what it now is since the self-correcting scheduler that
    needed the mutable list is gone."""
    if x.ndim != 4:
        raise ValueError(
            f"infinity_realism needs a 4-D [B, C, H, W] latent (its variance "
            f"stabilizer takes a per-channel statistic over the spatial axes); "
            f"got rank {x.ndim}. FLUX packs the latent into a [B, L, C·p²] "
            f"token sequence, so this sampler is not available for it — use "
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
    """Depthwise Gaussian blur for :func:`sample_infinity_omega`'s pyramid.
    Upstream builds the full 2-D kernel as an outer product and runs one
    grouped ``conv2d``; kept that way (the kernels are 3×3/5×5, so separating
    the passes would not pay for the extra launch)."""
    radius = kernel_size // 2
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


def _quantile_variance_preserve(denoised: torch.Tensor, ema_q95: Optional[torch.Tensor],
                                total_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """NQVP — pull each channel's 95th-percentile spatial deviation toward its
    running EMA, within a ``[0.88, 1.12]`` band. Computed in float32:
    ``torch.quantile`` rejects fp16/bf16 outright, and our pipelines sample in
    fp16."""
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
    """AVN — track each channel's spatial std of the *velocity* field and damp
    it back toward its EMA, within ``[clamp_min, 1.0]``. The ceiling of exactly
    1.0 is the point: AVN only ever shrinks a channel's spread, never grows it,
    so it can bleed CFG's velocity blowup off without being able to re-inflate a
    channel the model deliberately quieted. Centering is preserved (the mean is
    added back untouched), so unlike the ACS it replaces it cannot drag a
    channel's DC level toward an early-seeded EMA — which is what produced the
    colour cast we measured on flow. Float32, as with NQVP: an fp16 ``std``
    reduction over a full feature map is exactly where a stabilizer goes wrong."""
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


# Both pyramid branches skip NQVP below a sigma_max threshold, but they reach
# that test from opposite directions and the constant differs, so it is a
# per-branch parameter rather than a shared one.
#
# ``nano`` (@355b792) still calls it "split resume detection": upstream's intent
# is to skip the EMA stabilizers on a ComfyUI KSamplerAdvanced mid-schedule
# restart, where an EMA seeded from step 0 would be meaningless. But the test is
# on the *absolute* sigma, and only variance-exploding models have a sigma_max
# above it: SD/SDXL start at 14.6, so NQVP runs; rectified flow starts at 1.0,
# so it never does. ComfyUI's Anima is ModelSamplingDiscreteFlow(multiplier=1.0,
# shift=3.0) — sigma_max = 3·1/(1+2·1) = 1.0 — so under ComfyUI *every* Anima
# generation runs nano with NQVP switched off. Note it also closes on almost
# every SD *img2img* run: only the top ~13% of a karras schedule sits above
# sigma 8, so NQVP needs roughly strength >= 0.87 (karras) or >= 0.92
# (exponential) to run at all.
#
# ``omega`` (@8d81e76) renamed the same test to what it always actually was —
# ``is_flow = sigma_max < 5.0``, a VE-vs-rectified-flow discriminator — and
# dropped the split-resume story. NQVP is now deliberately SD/SDXL-only rather
# than incidentally so. Between 5 and 8 the two constants disagree (a partial-
# denoise SD img2img), which is the only case where nano's gate and omega's
# differ in outcome; both are ported literally.
_NQVP_SIGMA_MIN_NANO = 8.0
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
    """Shared loop for the Laplacian-pyramid branches. ``nano`` is upstream's
    ``omega`` with AVN and DoG removed and the older NQVP gate constant
    (verified by diffing the two branch files), so they run one implementation
    with three parameters rather than two copies of the numerics.

    AVN is the one piece that is live on rectified flow, so it is also the only
    thing separating ``omega`` from ``nano`` there — NQVP is gated off on flow
    for both, and DoG is near-nil by construction."""
    if x.ndim != 4:
        raise ValueError(
            f"{name} needs a 4-D [B, C, H, W] latent (its band decomposition is "
            f"2-D convolution); got rank {x.ndim}. FLUX packs the latent into a "
            f"[B, L, C·p²] token sequence, so this sampler is not available for "
            f"it — use infinity there."
        )
    eps = 6.1035e-5
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    # Upstream's is_flow / split-resume test; see the two _NQVP_SIGMA_MIN_*.
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
            # Upstream damps flow harder than VE (0.70 vs 0.85) on the reasoning
            # that a flow velocity field is the cleaner of the two to begin with.
            v, ema_v_std = _adaptive_velocity_normalize(
                v, ema_v_std, total_steps, 0.70 if is_flow else 0.85)

        macro = _gaussian_blur2d(v, 5, 2.0)
        mid = _gaussian_blur2d(v, 3, 1.0)
        meso = mid - macro
        nano = v - mid

        # Local std map of the nano band (E[n²] − E[n]², both blurred).
        var = _gaussian_blur2d(nano * nano, 3, 1.0) - _gaussian_blur2d(nano, 3, 1.0) ** 2
        s_nano = var.clamp(min=eps).sqrt()
        # Upstream's literal knee. On flow (sigma <= 1) it never saturates, so
        # eta spans 0.025..0.167 instead of 0.025..0.25 — a weaker nano gain
        # than an SD run gets, and again what upstream's flow results used.
        eta = 0.25 * min(1.0, max(0.1, float(sigma) / 1.5))
        gain = 1.0 + eta * torch.tanh(s_nano / (s_nano.mean(dim=(2, 3), keepdim=True) + eps))

        if dog:
            band = _gaussian_blur2d(nano, 3, 0.5) - _gaussian_blur2d(nano, 5, 1.0)
            nano = nano + (0.15 * eta) * band

        x = x + (macro + meso + gain * nano).to(x.dtype) * dt
    return x


def sample_infinity_nano(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
) -> torch.Tensor:
    """Infinity Diffusion, ``nano`` branch (galpt/infinity-diffusion, MIT;
    upstream @355b792, 2026-07-24, unchanged upstream since) —
    :func:`sample_infinity_omega` without AVN and without the DoG term, which
    is exactly what upstream's ``nano`` is.

    Same Euler step, same LPVD/AHFRI velocity filter, same NQVP spread clamp,
    same ≤6-step bypass and same 4-D requirement; see ``infinity_omega`` for
    all of it. What is gone:

    * **AVN**, which damps each channel's velocity spread toward a running EMA.
    * **DoG**, whose ``0.15·η ≤ 0.038`` on a band-pass of an already-small band
      makes it close to a no-op anyway.

    Nano is now the *older* of the two branches, and the gap has widened. It
    keeps the ``σ_max < 8`` split-resume gate on NQVP that omega has since
    replaced, and it never received AVN. On **rectified flow** NQVP is gated
    off, so nano there is LPVD + AHFRI on Euler with nothing stabilized at all
    — where omega now damps the velocity every step. That makes the two
    genuinely different on Anima for the first time; previously they were
    separated only by the near-no-op DoG term.

    Deterministic; one model evaluation per step; six small depthwise
    convolutions per step (two fewer than omega)."""
    return _sample_infinity_pyramid(model, x, sigmas, name="infinity_nano",
                                    nqvp_sigma_min=_NQVP_SIGMA_MIN_NANO,
                                    avn=False, dog=False, callback=callback)


def sample_infinity_omega(
    model: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    callback: Callback = None,
) -> torch.Tensor:
    """Infinity Diffusion, ``omega`` branch (galpt/infinity-diffusion, MIT;
    upstream @8d81e76, 2026-07-25).

    Be clear about what this is: **the integrator is plain Euler**. Both
    ``infinity``'s invariant-gated correction and the ``micro`` branch's
    second-order term were dropped upstream along the way, so as an ODE solver
    omega is strictly weaker than :func:`sample_infinity`. What it adds is a
    per-step *spatial* filter on the velocity field plus two stabilizers, aimed
    at detail retention rather than accuracy:

    * **LPVD** splits ``d`` into three bands with a Gaussian pyramid —
      ``macro`` (σ=2, 5×5), ``meso`` (σ=1, 3×3, minus macro) and ``nano``
      (the residual).
    * **AHFRI** amplifies the nano band by ``1 + η·tanh(s/s̄)`` where ``s`` is
      a local standard-deviation map of that band, so the boost lands where
      high-frequency structure already exists. ``η = 0.25·clamp(σ/1.5, 0.1,
      1)`` — at most a 25% gain, and on flow at most 16.7% (σ ≤ 1 never
      saturates the knee).
    * **DoG** adds ``0.15·η`` of an isotropic band-pass of the nano band.
      Faithful to upstream and near-free, but be aware ``0.15·η ≤ 0.038``
      applied to a band-pass of an already-small band: it is close to a no-op.
    * **NQVP** holds the denoised prediction's per-channel 95th-percentile
      spread near its running EMA. **SD/SDXL only** — upstream now gates it on
      ``sigmas[0] ≥ 5`` explicitly as a "not a flow model" test, on the
      reasoning that ``σ·ε`` is what makes VE's early-step latent swings large
      enough to need it.
    * **AVN** damps each channel's *velocity* spread toward its running EMA,
      within ``[0.70, 1.0]`` on flow and ``[0.85, 1.0]`` on VE. It runs on
      **every family**, and the 1.0 ceiling means it can only ever shrink a
      channel's spread — upstream's current answer to CFG oversaturation.

    So there are still two omegas, but they are closer than they were. On
    **SD/SDXL** it is the full stack: the ≤25% nano gain pushes toward detail
    while NQVP and AVN push back. On **Anima and any other rectified-flow
    model** NQVP stays off but AVN is live, so the detail filter is no longer
    unclamped the way it was through @4319bc7.

    **This replaced ACS, and the replacement is the interesting part.** Through
    @4319bc7 the second stabilizer was ACS, which pulled each channel's spatial
    *mean* 50% of the way to an early-seeded EMA. That is a DC-level correction,
    and pinning DC per channel on a 16-channel flow latent is exactly how you
    manufacture a colour cast — which is what we measured on Anima when we
    briefly dropped upstream's ``σ_max < 8`` gate. Upstream reached the same
    conclusion from the other end: it first tried excluding flow models from
    ACS, then deleted ACS outright in favour of AVN, which touches only the
    spread and only downward, leaves the mean alone, and acts on the velocity
    rather than the prediction. The gate is gone because it is no longer load-
    bearing — nothing left in the stack can cast.

    Note ``η`` is 10× larger at high σ than at low σ, so the amplification
    lands during structure formation rather than during fine-texture cleanup.
    That matches ``infinity_htds``, which despite its name is also high-σ-dense
    — the pairing is coherent, just not where the README says it aims.

    One caution on upstream's evidence: its F-PTLS benchmark measures FFT power
    density, which AHFRI inflates by construction, so it is not independent
    support for the branch's detail claims.

    Deterministic; one model evaluation per step; six (nano) to eight (omega)
    small depthwise convolutions per step on top, which is noise next to a DiT
    forward. Below 7 steps the whole filter is bypassed and this is exactly
    ``euler`` — upstream's guard for distilled/Turbo models.

    **4-D latents only.** The band decomposition is 2-D convolution over the
    latent's spatial axes, so this needs ``[B, C, H, W]``. SD/SDXL and Anima
    qualify; FLUX does not — it patchifies to a ``[B, L, C·p²]`` token
    sequence before sampling, where a spatial blur is meaningless. The engine
    keeps omega out of the FLUX dropdown; the guard below is for direct
    library callers.

    **Port deviations:** exactly one, and it is forced — the stabilizers and
    the pyramid run in float32 (see their docstrings), because
    ``torch.quantile`` rejects fp16 outright and an fp16 ``std`` reduction over
    a full feature map is precisely where a numerical stabilizer goes wrong.
    Everything else is literal, including the ``sigmas[0] < 5`` NQVP gate, the
    two AVN clamp floors and the ``σ/1.5`` knee. Fed a float32 latent, omega and
    nano reproduce ``galpt/infinity-diffusion``'s ``InfinitySampler`` bit-for-bit
    on both SD-scale and flow-scale schedules.

    Any Anima measurement of this sampler taken before 2026-08-09 describes the
    @4319bc7 ACS build, not this one — the two differ on flow by the whole of
    AVN, which is the first stabilizer omega has ever run there.

    Upstream's 5-D ``(B, C, T, H, W)`` folding path is omitted, but its README
    is right that Anima needs one — under ComfyUI it does. There Anima carries
    ``latent_format = Wan21`` (``latent_dimensions = 3``,
    ``temporal_downscale_ratio = 4``) and its DiT is Cosmos-Predict2's
    ``MiniTrainDIT``, whose forward names its input ``x_B_C_T_H_W`` outright,
    so ComfyUI samplers see 5-D latents for Anima. Our pipelines do not: they
    sample a 4-D ``(B, 16, H, W)`` latent and add the temporal axis only at the
    DiT call boundary (``x.unsqueeze(2)`` … ``.squeeze(2)``), pinning ``T=1``.
    The vendored DiT itself is general over ``T``, so if multi-frame Anima is
    ever wired up here, this sampler and :func:`sample_infinity_nano` need the
    fold restored — fold ``(B, C, T, H, W)`` to ``(B·T, C, H, W)`` before the
    convolutions and back after — and the rank guard below relaxed."""
    return _sample_infinity_pyramid(model, x, sigmas, name="infinity_omega",
                                    nqvp_sigma_min=_NQVP_SIGMA_MIN_OMEGA,
                                    avn=True, dog=True, callback=callback)


# --- aether -----------------------------------------------------------------
# Upstream's noise-injection constants (galpt/infinity-diffusion `aether`
# @c3ba017). The injected std is
#     n(σ) = min(0.25·σ, 0.08) · ramp(σ) · (1 − C),   ramp = clamp((σ−0.02)/0.08)
# capped at max(0.30·σ_next, 0.03). Upstream's notes: the 0.25·σ coefficient
# anchors the low-σ regime inside Song's corrector band, the 0.08 ceiling holds
# mid-schedule injection to ~2.7× the 0.03 an earlier aether shipped clean, and
# the 0.03 floor on the cap exists so the terminal stamp stays at exactly that
# proven value rather than being trimmed to 0.0088 by the σ_next term. All three
# are absolute noise scales, which is the same construction that made the old
# `realism` branch unusable on rectified flow — see the caution in
# :func:`sample_infinity_aether`.
_AETHER_NOISE_SIGMA_COEF = 0.25
_AETHER_NOISE_ABS_CAP = 0.08
_AETHER_NOISE_TERMINAL_FLOOR = 0.03


def _central_gradients(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences with edge-replicated padding. Upstream reaches this
    through ``narrow()`` with explicit positive indices to dodge an Intel XPU
    indexing bug; plain slicing is the same arithmetic and we do not target
    XPU."""
    p = F.pad(v, (1, 1, 1, 1), mode="replicate")
    return (p[..., 1:-1, 2:] - p[..., 1:-1, :-2],
            p[..., 2:, 1:-1] - p[..., :-2, 1:-1])


def _structure_tensor_coherence(v: torch.Tensor, eps: float = 1e-5,
                                multi_scale: bool = False) -> torch.Tensor:
    """Local structure-tensor coherence ``C ∈ [0, 1]`` — near 1 along a strong
    edge normal, near 0 where the gradient is isotropic or noise. Computed from
    the Gaussian-smoothed outer product of the gradients as
    ``((J_xx − J_yy)² + 4·J_xy²) / tr(J)²``, i.e. the squared relative
    eigenvalue split. ``multi_scale`` takes the per-pixel max over three blur
    scales (3/σ=0.5, 5/σ=1, 7/σ=2), which catches both fine strands and broad
    contours; upstream uses it for the noise gate and the single scale for
    the DoG weight."""
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


# Laws' 5×5 texture-energy masks, as separable 1-D kernels. The nine outer
# products upstream selects, in the order its argmax indexes them.
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
# Which of the four material classes each of those nine responses votes for:
# 0 = flat, 1 = skin/texture, 2 = line art, 3 = fabric/ripple.
_LAWS_MATERIAL = (0, 2, 1, 3, 3, 2, 2, 1, 1)


def _classify_material(v: torch.Tensor) -> torch.Tensor:
    """Per-pixel material class from the strongest Laws texture-energy response.

    Convolves with the nine 5×5 masks (each normalized by 36, the ``L5·L5``
    central value, so the responses stay comparable), takes ``|·|``, and maps
    the argmax through :data:`_LAWS_MATERIAL`. Returns an integer tensor shaped
    like the input.

    Upstream spells the 9→4 mapping as a nested ``torch.where`` chain to avoid a
    gather on Intel XPU; a lookup-table index is the same mapping and reads as
    what it is. Upstream's ``normalize=True`` branch is not ported — it exists
    for its image-space analysis scripts, and the sampler's call site is the
    ``False`` default."""
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
    (Kovesi 1995). Local energy ``√(|∇v|² + ∇²v²)`` is compared against its own
    Gaussian-smoothed self as ``E / (E + Ē)``, so the result depends on the
    *ratio* of local to neighbourhood energy rather than its magnitude — a faint
    skin crease scores like a bold outline. Sits at ~0.5 in featureless regions,
    which is why the sampler only ever adds it on top of a band-pass.

    The Laplacian uses zero padding (upstream's ``F.pad`` default), unlike the
    replicate padding in :func:`_central_gradients`; the one-pixel border
    difference is immaterial to a ratio."""
    v_x, v_y = _central_gradients(v)
    px, py = F.pad(v_x, (1, 1)), F.pad(v_y, (0, 0, 1, 1))
    laplacian = (px[..., 2:] - px[..., :-2]) + (py[..., 2:, :] - py[..., :-2, :])
    grad_mag = torch.sqrt(v_x ** 2 + v_y ** 2 + eps)
    energy = torch.sqrt(grad_mag ** 2 + laplacian ** 2 + eps)
    smoothed = _gaussian_blur2d(energy, 7, 2.0)
    return (energy / (energy + smoothed + eps)).clamp(0.0, 1.0)


def _coherence_lisc(v: torch.Tensor, light_angle_deg: float, strength: float,
                    eps: float = 6.1035e-5) -> torch.Tensor:
    """LISC — add ``strength·C·(∇v · l̂)`` to the band, i.e. directional shading
    along a virtual light vector, masked by the structure-tensor coherence so it
    only lands on coherent structure instead of imprinting a lighting gradient
    on noise. Upstream recomputes the coherence inline at the single 3/σ=1
    scale with ``eps = 6.1035e-5`` rather than the ``1e-5`` default, so that eps
    is passed through explicitly here."""
    v_x, v_y = _central_gradients(v)
    rad = math.radians(light_angle_deg)
    coherence = _structure_tensor_coherence(v, eps=eps)
    return v + strength * coherence * (v_x * math.cos(rad) + v_y * math.sin(rad))


def _velocity_norm_normalize(v_enhanced: torch.Tensor,
                             v_reference: torch.Tensor) -> torch.Tensor:
    """VNN — rescale each sample so the enhanced velocity carries the same L2
    norm as the one it was built from. The band enhancements are free to move
    energy between frequencies but not to add any, which is what keeps the ODE
    trajectory from drifting as the per-step gains accumulate."""
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
    upstream @c3ba017, 2026-07-31) — :func:`sample_infinity_omega`'s stack with
    the isotropic DoG term replaced by a *material-aware, anisotropic* one, plus
    directional shading, an energy clamp and stochastic grain.

    Everything omega does is still here and unchanged: Euler integration, the
    3-band LPVD pyramid, AHFRI's ``1 + η·tanh(s/s̄)`` nano gain, NQVP on
    SD/SDXL only, AVN on every family. What aether adds, per step:

    * **Material classification.** Nine Laws 5×5 texture-energy masks are run
      over the denoised prediction and each pixel is labelled by its strongest
      response: flat, skin/texture, line art, or fabric/ripple.
    * **Phase saliency.** A contrast-invariant edge map (see
      :func:`_phase_edge_saliency`) that scores faint edges like bold ones.
    * **Coherence-weighted DoG.** The DoG band-pass is now gated by a per-class
      gain built from the structure-tensor coherence ``C``: flat gets ``0.5·C``,
      line art gets ``max(C, phase)``, and skin/fabric blend an isotropic term
      into ``1−C`` (coefficients 0.50 + 0.30·phase and 0.65 respectively).
      This is the anisotropy — omega applied the same scalar everywhere.
    * **LISC** adds ``0.06·C·(∇v·l̂)`` of directional shading to the *macro*
      band, and only while ``σ ≥ 0.80``.
    * **VNN** rescales the assembled velocity back to the pre-enhancement L2
      norm, so none of the above can inject energy into the trajectory.
    * **TZTD** ramps every enhancement strength linearly to zero between
      ``σ = 0.80`` and ``σ = 0.15``, below which the step is pure Euler.
    * **Coherence-gated noise**, the one non-deterministic piece: grain scaled
      by ``1 − C`` so it lands in flat regions and not on edges.

    **Stochastic.** Upstream draws with ``torch.randn_like``; we draw through
    ``_noise_like(x, generator)`` like every other stochastic sampler here, so a
    given seed reproduces. That is the one behavioral deviation.

    **The σ thresholds are absolute, and that is worth knowing per family.**
    ``0.80``/``0.15`` for TZTD, ``0.80`` for LISC, and the noise schedule's
    ``0.25·σ`` capped at ``0.08`` with a ``0.03`` floor are all fixed numbers
    compared against raw σ. On SD/SDXL (σ_max 14.6) that means TZTD is inert for
    all but the last few steps and LISC runs nearly the whole way. On rectified
    flow (σ_max 1.0) the same constants land completely differently: TZTD is
    already ramping down by the second step, and LISC fires only at the very
    top of the schedule if at all. This is the same class of construction that
    made the old ``realism`` branch unusable on Anima. It is not the same
    severity — the noise here is capped absolutely at 0.08 and gated to
    low-coherence pixels, where realism's ``0.20·σ`` was not — but on flow the
    branch is substantially a different sampler from the one upstream tuned,
    and the grain is proportionally much heavier relative to what each step
    removes. Treat flow results as unvalidated.

    **4-D latents only**, for the same reason as omega and nano: the pyramid,
    the structure tensor and the Laws masks are all 2-D convolutions.

    **Port deviations** beyond the generator: the pyramid and the analysis maps
    run in float32 (as in omega — ``torch.quantile`` rejects fp16 and these are
    variance reductions over full feature maps), upstream's 5-D fold and tqdm
    bar are omitted, and its ``sigmas[-1]`` tolerance clamp is dropped since our
    schedules terminate at exactly 0. Everything else is literal, including all
    of the constants above.

    Deterministic parts aside, this is the most expensive sampler here: on top
    of the DiT forward it runs ~20 depthwise convolutions plus nine 5×5 Laws
    convolutions per step. Still small next to the forward, but not free."""
    if x.ndim != 4:
        raise ValueError(
            f"infinity_aether needs a 4-D [B, C, H, W] latent (its band "
            f"decomposition, structure tensor and Laws masks are all 2-D "
            f"convolutions); got rank {x.ndim}. FLUX packs the latent into a "
            f"[B, L, C·p²] token sequence, so this sampler is not available for "
            f"it — use infinity there."
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

        # Analysis maps, all read off the denoised prediction rather than the
        # velocity — upstream's choice, and the reason they survive the σ→0
        # steps where the velocity is mostly the residual noise.
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
            # ``s_nano/(s_nano+eps)`` is ~1 wherever the nano band has any
            # amplitude at all, so the skin/fabric blends are close to a flat
            # lift of the 1−C (incoherent) share. Upstream's form, kept literal.
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

        # Coherence-gated grain. Skipped below σ=0.02, and never reached on the
        # ≤6-step bypass above.
        if s_cur > 0.02:
            n_s = min(_AETHER_NOISE_SIGMA_COEF * s_cur, _AETHER_NOISE_ABS_CAP)
            n_s *= min(1.0, max(0.0, (s_cur - 0.02) / 0.08))
            n_s = min(n_s, max(0.30 * s_next, _AETHER_NOISE_TERMINAL_FLOOR))
            x = x + (n_s * (1.0 - noise_coherence)).to(x.dtype) * _noise_like(x, generator)
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


def _sa_solver_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int,
                                  tau_t: float) -> torch.Tensor:
    """Exponential integrator coefficients for SA-Solver (see :func:`sample_sa_solver`).

    Computes ``(1 + τ²)·∫_s^t exp((1+τ²)·x)·x^p dx`` with ``exp((1+τ²)·t)``
    factored out, for ``p = 0..solver_order−1``, via the integration-by-parts
    recurrence of the reference implementation (the recursion matrix is the
    lower-triangular coefficient table from the SA-Solver codebase)."""
    tau_mul = 1 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    # x^p·exp((1+τ²)·x)/(1+τ²) at x=s factored by exp((1+τ²)·t); the (1+τ²)
    # cancels the outside factor.
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
    """The SA-Solver paper's closed-form order-2 b-coefficients (Appendix D),
    for the ``simple_order_2`` fast path. Returns ``[b_2, b_1]``."""
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
    """SA-Solver's ``b_i`` coefficients (paper eqs. 15 and 18), data-prediction
    (x0) form. The solver order is the number of half-logSNR points in
    ``curr_lambdas``: the coefficients are the Lagrange-basis integrals of the
    exponential integrator over the step, found by solving a Vandermonde
    system against the analytically-integrated exponential terms."""
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
    """Default SA-Solver stochasticity window: constant ``eta`` on the middle
    20%–80% of the run, zero elsewhere.

    ComfyUI builds the same band through the model's ``percent_to_sigma``; here
    it is read off the actual schedule instead (``start = sigmas[round(0.2·n)]``,
    ``end = sigmas[round(0.8·n)]``), which is the same geometry and needs no
    model access. ``eta <= 0`` makes the whole run deterministic (pure ODE)."""
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
    """SA-Solver: Stochastic Adams predictor-corrector solver (Xue et al.,
    "SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion
    Models", NeurIPS 2023, arXiv:2309.05019).

    A multi-step Adams solver in half-logSNR space, data-prediction (x0) form,
    with a predictor-corrector structure: each step first *corrects* the
    current latent using all available x0 estimates (re-deriving it more
    accurately than the prediction), then *predicts* the next latent with the
    exponential-integrator coefficients. The stochastic (SDE) form re-injects
    Gaussian noise with strength ``τ(sigma)`` on a middle band of the schedule
    (default 20%–80%, via :func:`_sa_solver_tau_interval`); ``eta <= 0``
    recovers the deterministic ODE solver.

    Solver order grows with history: the predictor/corrector use the last
    ``predictor_order`` / ``corrector_order`` x0 estimates, ramping from
    first order on the opening steps and winding back down near σ → 0 (the
    reference's ``lower_order_to_end`` stability rule; our schedules always end
    at exactly 0, so it is always active). ``use_pece=True`` adds the final
    "E" — a re-evaluation of the model at the corrected state — which costs one
    extra NFE per corrected step in exchange for accuracy (registered as
    ``sa_solver_pece``). ``simple_order_2`` uses the paper's closed-form
    second-order coefficients instead of solving the Vandermonde system.

    Flow-aware via the shared half-logSNR map: ``model_type="flow"`` (Anima /
    FLUX) uses ``log((1−σ)/σ)`` with ``shift`` offsetting the first σ off 1.0,
    ``"ve"`` (SD / SDXL) uses ``−log σ`` — the same convention as the DPM++
    SDE family. Noise is seeded Gaussian (``generator``), matching every other
    stochastic sampler in this registry.

    Reference: the official SA-Solver codebase (github.com/scxue/SA-Solver),
    as carried in ComfyUI's ``comfy/k_diffusion/sa_solver.py``. Two deliberate
    deviations: the tau window is read off the schedule (see
    :func:`_sa_solver_tau_interval`) instead of the model's
    ``percent_to_sigma``, and ``s_noise`` is not scaled by a model ``noise_scale``
    (this engine has none; the parameter is the effective scale)."""
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

        # Corrector: re-derive the current state at sigma[i] from the x0 history.
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
    """SA-Solver in PECE mode: predict, evaluate, correct, and re-evaluate the
    corrected state (``use_pece=True`` in :func:`sample_sa_solver`). The extra
    evaluation per corrected step buys accuracy; see the main docstring."""
    kwargs["use_pece"] = True
    return sample_sa_solver(model, x, sigmas, **kwargs)


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
    "infinity_nano": sample_infinity_nano,
    "infinity_omega": sample_infinity_omega,
    "infinity_aether": sample_infinity_aether,
    "lms": sample_lms,
    "lcm": sample_lcm,
    "ddpm": sample_ddpm,
    "sa_solver": sample_sa_solver,
    "sa_solver_pece": sample_sa_solver_pece,
    "secant": sample_secant,
    "secant_anneal": sample_secant_anneal,
    "dpmpp_2m_anneal": sample_dpmpp_2m_anneal,
    "exp_heun_2_x0": sample_exp_heun_2_x0,
    "uni_pc": partial(sample_uni_pc, variant="bh1"),
    "uni_pc_bh2": partial(sample_uni_pc, variant="bh2"),
    "uni_pc_anneal": sample_uni_pc_anneal,
    "cogent": sample_cogent,
    "cogent3": sample_cogent3,
    "cogent3_pump": partial(sample_cogent3, pump_strength=0.08),
}


def get_sampler(name: str):
    """Look up a sampler function by name (see :data:`SAMPLERS`)."""
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
