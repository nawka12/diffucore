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

from functools import lru_cache, partial
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
    "sample_lms",
    "sample_lcm",
    "sample_ddpm",
    "sample_secant",
    "sample_secant_anneal",
    "sample_dpmpp_2m_anneal",
    "sample_uni_pc_anneal",
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
    """Infinity Diffusion sampler (galpt/infinity-diffusion, MIT; verified
    equivalent to upstream @4f72d8f, 2026-07-17).

    Euler on the σ-space ODE with an invariant-gated IIR correction: the first
    step is plain Euler, after which each step advances by ``d + correction``
    where the correction combines a *velocity* EMA of the first derivative
    difference (``vel ← (1−α₁)·vel + α₁·(d − d_prev)``, gain ``β₁``) and — from
    the third step — an *acceleration* EMA of the second difference
    (``acc ← (1−α₂)·acc + α₂·(d − 2·d_prev + d_prev2)``, gain ``β₂``). Before
    stepping, three invariants gate the correction: (1) its mean magnitude is
    clamped to 50% of the derivative's, (2) it is halved if the derivative
    reversed direction (cosine vs. the previous step < 0), and (3) it is
    zeroed — a pure Euler step — when both trigger. The velocity half at
    ``α₁=β₁=0.5`` is the damped-AB2 memory the earlier upstream version was;
    the acceleration term adds curvature tracking on top. Constants are
    upstream's fixed ``α₁=0.5, β₁=0.5, α₂=0.3, β₂=0.3`` — upstream briefly
    shipped signal-adaptive coefficients and reverted to fixed the same day,
    so no knobs are exposed here either. Deterministic; one model evaluation
    per step; model-agnostic (raw σ-space serves VE — SD/SDXL — and rectified
    flow — Anima/FLUX — alike, as ``ipndm``/``stork2`` do).

    Upstream applies the correction through the final (σ→0) step and never
    resets the EMAs; both behaviors are kept. The magnitude clamp and the
    cosine test reduce over the *whole* tensor (upstream semantics), which
    couples batch entries — harmless here since the pipelines sample one
    latent at a time.

    Grid caveat, honestly: the fixed gains never rescale the difference terms
    by the step-size ratio, so the correction is AB2-consistent only when
    neighboring steps are comparable. Pair with ``normal``-like grids or the
    matching ``infinity`` schedule (sine-perturbed timesteps); on strongly
    nonuniform grids (``karras`` ρ=7) it can land behind Euler — the clamp
    bounds the damage but does not fix the scaling."""
    alpha1, beta1 = 0.5, 0.5
    alpha2, beta2 = 0.3, 0.3
    s_in = x.new_ones([x.shape[0]])
    vel = acc = d_prev = d_prev2 = None
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
            d_prev = d
            continue
        delta = d - d_prev
        vel = (1.0 - alpha1) * vel + alpha1 * delta
        if d_prev2 is None:
            correction = beta1 * vel            # acceleration needs 3 d's
        else:
            acc = (1.0 - alpha2) * acc + alpha2 * (delta - (d_prev - d_prev2))
            correction = beta1 * vel + beta2 * acc
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
        d_prev2 = d_prev
        d_prev = d
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
    "lms": sample_lms,
    "lcm": sample_lcm,
    "ddpm": sample_ddpm,
    "secant": sample_secant,
    "secant_anneal": sample_secant_anneal,
    "dpmpp_2m_anneal": sample_dpmpp_2m_anneal,
    "exp_heun_2_x0": sample_exp_heun_2_x0,
    "uni_pc": partial(sample_uni_pc, variant="bh1"),
    "uni_pc_bh2": partial(sample_uni_pc, variant="bh2"),
    "uni_pc_anneal": sample_uni_pc_anneal,
}


def get_sampler(name: str):
    """Look up a sampler function by name (see :data:`SAMPLERS`)."""
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
