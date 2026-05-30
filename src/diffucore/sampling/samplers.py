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
    "sample_er_sde",
    "sample_dpm_2",
    "sample_dpm_2_ancestral",
    "sample_dpmpp_2m",
    "sample_dpmpp_sde",
    "sample_dpmpp_2m_sde",
    "sample_dpmpp_3m_sde",
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


SAMPLERS: dict[str, Denoiser] = {
    "euler": sample_euler,
    "heun": sample_heun,
    "euler_ancestral": sample_euler_ancestral,
    "er_sde": sample_er_sde,
    "dpm_2": sample_dpm_2,
    "dpm_2_ancestral": sample_dpm_2_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_sde": sample_dpmpp_sde,
    "dpmpp_2m_sde": sample_dpmpp_2m_sde,
    "dpmpp_3m_sde": sample_dpmpp_3m_sde,
}


def get_sampler(name: str):
    """Look up a sampler function by name (see :data:`SAMPLERS`)."""
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
