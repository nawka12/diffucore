"""Sampling-time sigma schedules.

Every schedule returns ``steps + 1`` descending sigmas from ``sigma_max`` down
to ``sigma_min``, with a trailing ``0.0`` appended. Reference for ``karras``:
Karras et al., NeurIPS 2022, eq. 5.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "append_zero",
    "karras_schedule",
    "exponential_schedule",
    "polyexponential_schedule",
    "kl_optimal_schedule",
    "align_your_steps_schedule",
    "flow_matching_schedule",
    "flow_matching_dynamic_shift",
    "simple_schedule",
    "sgm_uniform_schedule",
    "normal_schedule",
    "infinity_schedule",
    "infinity_htds_schedule",
    "ddim_uniform_schedule",
    "linear_quadratic_schedule",
    "smoothstep_schedule",
    "beta_schedule",
    "beta_mix_schedule",
    "pump_dual_schedule",
    "pump_taper_schedule",
    "flow_table_schedule",
    "FlowSamplingView",
]


def append_zero(sigmas: torch.Tensor) -> torch.Tensor:
    """Append a trailing ``0.0`` (the fully denoised endpoint)."""
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
    """Karras et al. (2022): linear in ``sigma ** (1/rho)``. ``rho=7`` (the
    paper's default) concentrates steps at low noise.
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


def kl_optimal_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """KL-optimal schedule (arXiv:2407.12173): linear in ``arctan(sigma)``.
    Works for VE and flow ranges."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ramp = torch.arange(steps, device=device, dtype=dtype) / max(steps - 1, 1)
    sigmas = (ramp * math.atan(sigma_min) + (1.0 - ramp) * math.atan(sigma_max)).tan()
    return append_zero(sigmas)


# AYS 10-step schedules (Sabour, Fidler & Kreis, "Align Your Steps", ICML 2024,
# arXiv:2404.14507, Table 3), σ_max down to σ_min, as shipped by NVIDIA's AYS
# page and the A1111 / ComfyUI integrations.
_AYS_TABLES = {
    "sd15": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029],
    "sdxl": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029],
    "svd": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002],
    "deepfloyd": [160.41, 8.081, 3.315, 1.885, 1.207, 0.785, 0.553, 0.293, 0.186, 0.030, 0.006],
}


def _loglinear_interp(t_steps, num_steps: int) -> torch.Tensor:
    """Log-linear interpolation of a descending sequence to ``num_steps`` points,
    keeping both endpoints (NVIDIA's AYS ``loglinear_interp``, in torch)."""
    t = torch.as_tensor(t_steps, dtype=torch.float64)
    ys = t.flip(0).log()                                  # ascending in index
    pos = torch.linspace(0, 1, num_steps, dtype=torch.float64) * (len(t) - 1)
    lo = pos.floor().long()
    hi = pos.ceil().long().clamp(max=len(t) - 1)
    w = (pos - lo).clamp(0.0, 1.0)
    interp = (ys[lo] * (1.0 - w) + ys[hi] * w).exp()
    return interp.flip(0)


def align_your_steps_schedule(
    steps: int,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    model: str = "sdxl",
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Align Your Steps (Sabour, Fidler & Kreis, ICML 2024, arXiv:2404.14507).

    The paper's per-family 10-step tables (:data:`_AYS_TABLES`), log-linearly
    interpolated to any step count as the authors recommend (same output as
    A1111's ``get_align_your_steps_sigmas``). ``model`` is ``"sd15"``,
    ``"sdxl"``, ``"svd"`` or ``"deepfloyd"``. ``sigma_min``/``sigma_max``, if
    given, are only checked against the table's range (zero-terminal-SNR
    checkpoints fail), not used to rescale. VE-only (SD/SDXL)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    try:
        table = _AYS_TABLES[model]
    except KeyError:
        raise ValueError(
            f"unknown AYS model {model!r}; available: {sorted(_AYS_TABLES)}") from None
    if sigma_min is not None and sigma_max is not None:
        t_min, t_max = min(table), max(table)
        if sigma_max > 1.05 * t_max or sigma_min < 0.95 * t_min:
            raise ValueError(
                f"AYS schedule for {model!r} spans σ ∈ [{t_min:g}, {t_max:g}]; "
                f"model reports [{sigma_min:g}, {sigma_max:g}]. AYS is not "
                f"defined for this noise range (zero-terminal-SNR checkpoints "
                f"should use 'karras').")
    sigmas = _loglinear_interp(table, steps)
    return append_zero(sigmas.to(device=device, dtype=dtype))


def flow_matching_schedule(
    steps: int,
    shift: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """SD3-style shifted rectified-flow schedule:
    ``σ(t) = shift·t / (1 + (shift − 1)·t)`` for ``t = (N − i)/N``. Higher
    ``shift`` puts more steps near ``σ = 1``. Anima ships with 3.0; Flux's
    default is 1.15.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if shift < 1.0:
        raise ValueError("shift must be >= 1")
    t = torch.arange(steps, 0, -1, device=device, dtype=dtype) / steps
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    return append_zero(sigmas)


def flow_matching_dynamic_shift(
    seq_len: int,
    *,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Flux-style resolution-aware shift for :func:`flow_matching_schedule`.

    Interpolates the log-shift ``mu`` linearly in the token count
    ``seq_len = (H // 16) * (W // 16)`` between ``base_shift`` and
    ``max_shift`` (Flux's ``calculate_shift``) and returns ``exp(mu)``. At
    1024² this is ≈ 3.16, close to Anima's training shift of 3.0.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = base_shift + m * (seq_len - base_seq_len)
    return math.exp(mu)


class FlowSamplingView:
    """``DiscreteSchedule``-like view of a rectified-flow model, so the
    table/timestep schedulers work on Anima. Mirrors ComfyUI's
    ``ModelSamplingDiscreteFlow``: an ascending ``multiplier``-entry sigma table
    from ``sigma(t) = shift·t/(1+(shift-1)·t)``, with timesteps ``t·multiplier``.
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
    """ComfyUI ``simple``: ``steps`` sigmas from the ascending training table at
    even strides from the high-noise end."""
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
    """ComfyUI ``sgm_uniform``: uniform in timestep, ``steps + 1`` timesteps
    with the last dropped."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = float(schedule.sigma_to_t(schedule.sigma_max))
    end = float(schedule.sigma_to_t(schedule.sigma_min))
    ts = torch.linspace(start, end, steps + 1, device=device, dtype=torch.float32)[:-1]
    sigmas = schedule.t_to_sigma(ts).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def normal_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ComfyUI ``normal``: uniform in timestep, all ``steps`` kept."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = float(schedule.sigma_to_t(schedule.sigma_max))
    end = float(schedule.sigma_to_t(schedule.sigma_min))
    ts = torch.linspace(start, end, steps, device=device, dtype=torch.float32)
    sigmas = schedule.t_to_sigma(ts).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def infinity_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Infinity Diffusion's sine-perturbed schedule (galpt/infinity-diffusion,
    MIT; matches upstream @4f72d8f): ``normal``'s timestep ramp warped by
    ``f(u) = u − s·sin(πu)/π`` with ``s = min(0.6, steps/50)``, shrinking the
    first gap and growing the last. Endpoints are fixed and every sigma comes
    from the model's σ(t), so it is flow-safe."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = float(schedule.sigma_to_t(schedule.sigma_max))
    end = float(schedule.sigma_to_t(schedule.sigma_min))
    u = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
    strength = min(0.6, steps / 50.0)
    f = u - strength * (torch.sin(math.pi * u) / math.pi)
    ts = start + (end - start) * f
    sigmas = schedule.t_to_sigma(ts).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def infinity_htds_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Infinity Diffusion's "Hyperbolic Tail-Density" schedule
    (galpt/infinity-diffusion ``omega``/``nano``, MIT, upstream @4319bc7):
    ``normal``'s ramp bent by ``tanh(δ·(1−u)) / tanh(δ)`` with
    ``δ = clamp((steps − 4)/26, 0, 1.8)``; exactly linear at ``steps ≤ 4``.

    The name is backwards: the curve is convex, so sigma stays high early and
    plunges late (at 50 flow steps it puts 7 sigmas below 0.5σ_max where
    ``normal`` puts 13). Ported as upstream ships it; it suits structure, not
    texture, and pairs coherently with ``infinity_omega``."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = float(schedule.sigma_to_t(schedule.sigma_max))
    end = float(schedule.sigma_to_t(schedule.sigma_min))
    u = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
    delta = max(0.0, min(1.80, (float(steps) - 4.0) / 26.0))
    decay = 1.0 - u if delta <= 1e-5 else torch.tanh(delta * (1.0 - u)) / math.tanh(delta)
    ts = end + (start - end) * decay
    sigmas = schedule.t_to_sigma(ts).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def ddim_uniform_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ComfyUI ``ddim_uniform``: sigmas from the ascending table at a fixed
    stride from the low-noise end, reversed, with a trailing 0."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    table = schedule.sigmas
    n = len(table)
    stride = max(n // steps, 1)
    sigs = [0.0]
    x = 1
    while x < n:
        sigs.append(float(table[x]))
        x += stride
    sigs.reverse()
    return torch.tensor(sigs, device=device, dtype=dtype)


def linear_quadratic_schedule(schedule, steps: int, *, threshold_noise: float = 0.025,
                              linear_steps: int | None = None,
                              device: torch.device | str = "cpu",
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ComfyUI ``linear_quadratic`` (from Mochi): linear up to
    ``threshold_noise`` over the first ``linear_steps`` (default ``steps // 2``),
    quadratic after, scaled to ``sigma_max``."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps == 1:
        normalized = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        diff = linear_steps - threshold_noise * steps
        quad_steps = steps - linear_steps
        quad_coef = diff / (linear_steps * quad_steps ** 2)
        lin_coef = threshold_noise / linear_steps - 2 * diff / (quad_steps ** 2)
        const = quad_coef * (linear_steps ** 2)
        quad = [quad_coef * (i ** 2) + lin_coef * i + const for i in range(linear_steps, steps)]
        normalized = [1.0 - v for v in (linear + quad + [1.0])]
    sigma_max = float(schedule.sigma_max)
    return torch.tensor(normalized, device=device, dtype=dtype) * sigma_max


def smoothstep_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """U-shaped flow schedule: ``t`` warped by smoothstep ``u = t²·(3 − 2t)``
    before the shift map, so steps cluster near σ = 1 and σ = 0 with a sparser
    middle. Made for the σ-annealed ancestral samplers."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    t = torch.arange(steps, 0, -1, device=device, dtype=torch.float32) / steps
    u = t * t * (3.0 - 2.0 * t)
    sigmas = schedule.t_to_sigma(u * schedule.multiplier).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def _beta_inv_cdf(q: torch.Tensor, alpha: float, beta: float, *, grid: int = 4096) -> torch.Tensor:
    """Beta(α, β) quantile function in pure torch: midpoint-rule CDF on a
    uniform grid (exact leading-order masses in the two singular edge cells),
    inverted by linear interpolation. Accurate to ~1e-5; float64."""
    edges = torch.linspace(0.0, 1.0, grid + 1, dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = centers ** (alpha - 1.0) * (1.0 - centers) ** (beta - 1.0)
    h = 1.0 / grid
    pdf[0] = h ** alpha / alpha / h          # exact ∫₀ʰ x^(α−1) dx, as a cell mean
    pdf[-1] = h ** beta / beta / h           # exact ∫₁₋ₕ¹ (1−x)^(β−1) dx
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), pdf.cumsum(0)])
    cdf = cdf / cdf[-1].clone()
    q = q.to(torch.float64).clamp(0.0, 1.0)
    idx = torch.searchsorted(cdf, q).clamp(1, grid)
    c0, c1 = cdf[idx - 1], cdf[idx]
    w = ((q - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0.0, 1.0)
    return edges[idx - 1] + w * (edges[idx] - edges[idx - 1])


def beta_schedule(schedule, steps: int, *, alpha: float = 0.6, beta: float = 0.6,
                  device: torch.device | str = "cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Beta-quantile schedule (ComfyUI's ``beta``; Lee et al., "Beta Sampling
    is All You Need", arXiv:2407.12173): ``t_i = 1 − BetaInvCDF(i/(n−1))``
    through the model's σ(t). α = β = 0.6 is U-shaped in t; α tunes the low-t
    end, β the high-t end. Runs from ``σ(1)`` to the table floor
    ``σ(1/multiplier)``."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be > 0")
    q = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    t = (1.0 - _beta_inv_cdf(q, alpha, beta)).clamp(min=1.0 / schedule.multiplier)
    sigmas = schedule.t_to_sigma(t.to(torch.float32) * schedule.multiplier).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def _beta_mixture_inv_cdf(q: torch.Tensor, weight: float, alpha1: float, beta1: float,
                          alpha2: float, beta2: float, *, grid: int = 4096) -> torch.Tensor:
    """Inverse CDF of ``weight·Beta(α1, β1) + (1−weight)·Beta(α2, β2)``, by the
    same scheme as :func:`_beta_inv_cdf`. Float64."""
    edges = torch.linspace(0.0, 1.0, grid + 1, dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf1 = centers ** (alpha1 - 1.0) * (1.0 - centers) ** (beta1 - 1.0)
    pdf2 = centers ** (alpha2 - 1.0) * (1.0 - centers) ** (beta2 - 1.0)
    h = 1.0 / grid
    pdf1[0] = h ** alpha1 / alpha1 / h
    pdf1[-1] = h ** beta1 / beta1 / h
    pdf2[0] = h ** alpha2 / alpha2 / h
    pdf2[-1] = h ** beta2 / beta2 / h
    pdf = weight * pdf1 + (1.0 - weight) * pdf2
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), pdf.cumsum(0)])
    cdf = cdf / cdf[-1].clone()
    q = q.to(torch.float64).clamp(0.0, 1.0)
    idx = torch.searchsorted(cdf, q).clamp(1, grid)
    c0, c1 = cdf[idx - 1], cdf[idx]
    w = ((q - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0.0, 1.0)
    return edges[idx - 1] + w * (edges[idx] - edges[idx - 1])


def beta_mix_schedule(schedule, steps: int, *, weight: float = 0.5,
                           alpha1: float = 0.8, beta1: float = 2.0,
                           alpha2: float = 3.0, beta2: float = 0.7,
                           device: torch.device | str = "cpu",
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Two-component Beta mixture schedule: timestep density
    ``weight·Beta(α1, β1) + (1−weight)·Beta(α2, β2)`` in ``x = 1 − t``, mapped
    through the model's σ(t), so the two endpoint peaks can differ.

    Defaults follow Lee et al.'s (arXiv:2407.12173, Fig. 2d) detail-leaning
    curve but are tuned for the flow shift map: the paper's
    ``Beta(0.5, 2.0) + Beta(3.0, 0.5)`` turns nearly symmetric there, over-packs
    σ ≈ 1 and collides steps at the table floor beyond ~40 steps. The tuned
    ``Beta(0.8, 2.0) + Beta(3.0, 0.7)`` stays strictly descending through ~117
    steps. Runs from ``σ(1)`` to the table floor.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < weight < 1.0:
        raise ValueError("weight must be in (0, 1); 0 or 1 collapses to a "
                         "single Beta; use the 'beta' scheduler instead")
    if alpha1 <= 0 or beta1 <= 0 or alpha2 <= 0 or beta2 <= 0:
        raise ValueError("alpha1, beta1, alpha2, beta2 must all be > 0")
    q = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    x = _beta_mixture_inv_cdf(q, weight, alpha1, beta1, alpha2, beta2)
    t = (1.0 - x).clamp(min=1.0 / schedule.multiplier)
    sigmas = schedule.t_to_sigma(t.to(torch.float32) * schedule.multiplier).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def pump_dual_schedule(schedule, steps: int, *, pump_end: float = 0.45,
                       pump_share: float = 0.85, top_sigma: float = 0.99,
                       device: torch.device | str = "cpu",
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Two-band schedule for ``cogent3_pump``: a step-dense pumped band above
    ``pump_end`` (the pump's hard cutoff) and a short refinement band ending at
    ``flow``'s terminus ``σ(t = 1/steps)``.

    Uniform-in-``u`` points through a piecewise-linear warp of
    ``λ = −logit(σ)``: the first step jumps ``σ_max`` → ``top_sigma`` (the
    model is σ-invariant near 1), then ``pump_share`` of the steps are uniform
    in λ down to ``pump_end``, and the rest uniform in λ to the terminus.
    Stopping at ``flow``'s terminus rather than the σ table floor (0.003) is
    what matters: on the cogent3 toy benchmark, deeper termini were
    monotonically worse (16× worse than ``flow`` at 8 steps at 0.003).
    ``pump_share`` above ~0.9 collapses the tail. With no room below
    ``pump_end`` (few steps, high shift) it is one uniform-λ band.

    Flow-only (``σ_max`` must be 1.0). ``pump_end`` matches ``sample_cogent3``'s
    default, so change them together. Since ``shift`` is a translation in λ it
    only reaches this schedule through the terminus.
    """
    if steps < 3:
        raise ValueError("steps must be >= 3")
    if not 0.0 < pump_share < 1.0:
        raise ValueError(f"pump_share must be in (0, 1); got {pump_share}")
    if not 0.0 < pump_end < 1.0:
        raise ValueError(f"pump_end must be in (0, 1); got {pump_end}")
    sigma_max = float(schedule.sigma_max)
    sigma_min = float(schedule.sigma_min)
    if not sigma_min < pump_end < sigma_max:
        raise ValueError(f"pump_end {pump_end} must be in ({sigma_min}, {sigma_max})")
    if not pump_end < top_sigma < sigma_max:
        raise ValueError(f"top_sigma {top_sigma} must be in ({pump_end}, {sigma_max})")

    def lam(sig: float) -> float:
        return math.log(1.0 / sig - 1.0)  # flow half-logSNR log((1-σ)/σ)

    sigma_end = float(schedule.t_to_sigma(schedule.multiplier / steps))
    lam_top, lam_fin, lam_knee = lam(top_sigma), lam(sigma_end), lam(pump_end)
    u = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    if lam_fin - lam_knee <= 1e-6:
        lamv = lam_top + (lam_fin - lam_top) * u       # one band, pumped throughout
    else:
        u_b = pump_share
        lamv = torch.where(u < u_b,
                           lam_top + ((lam_knee - lam_top) / u_b) * u,
                           lam_knee + ((lam_fin - lam_knee) / (1.0 - u_b)) * (u - u_b))
    sigmas = (-lamv).sigmoid()
    sigmas[0] = sigma_max
    return append_zero(sigmas.to(device=device, dtype=dtype))


def pump_taper_schedule(schedule, steps: int, *, pump_end: float = 0.45,
                        taper: float = 0.25, tail_share: float = 0.13,
                        top_sigma: float = 0.99,
                        device: torch.device | str = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``pump_dual`` for ~30 steps, aiming at the look of ``cogent3_pump`` +
    ``pump_dual`` at 50.

    Measured on that 50-step run: a 4-step refinement tail stays close to its
    8-step one (RMSE ~11/255; 2 steps visibly darkens line art), and the x0
    prediction changes fastest at the top of the band (σ ≈ 0.98–0.9) and ~10×
    slower by 0.7–0.45. So the tail is ``max(1, round(tail_share·steps))``
    λ-uniform steps, and the band's λ-density is tilted by ``exp(−taper·λ)``:
    at the defaults and 30 steps its steps run from 0.12 λ (the 50-step
    density) to 0.35 λ. ``taper=0`` is a uniform band.

    Layout otherwise as ``pump_dual``; with no room below ``pump_end`` the
    tilted band runs to the terminus. Pair with ``cogent3_pump_rate``. With a
    step-fraction CFG interval, end 0.75 stops CFG at σ ≈ 0.69 at 30 steps and
    0.8 at ≈ 0.54. Flow-only.
    """
    if steps < 3:
        raise ValueError("steps must be >= 3")
    if not 0.0 < tail_share < 1.0:
        raise ValueError(f"tail_share must be in (0, 1); got {tail_share}")
    if taper < 0:
        raise ValueError(f"taper must be >= 0; got {taper}")
    sigma_max = float(schedule.sigma_max)
    sigma_min = float(schedule.sigma_min)
    if not sigma_min < pump_end < sigma_max:
        raise ValueError(f"pump_end {pump_end} must be in ({sigma_min}, {sigma_max})")
    if not pump_end < top_sigma < sigma_max:
        raise ValueError(f"top_sigma {top_sigma} must be in ({pump_end}, {sigma_max})")

    def lam(sig: float) -> float:
        return math.log(1.0 / sig - 1.0)  # flow half-logSNR log((1-σ)/σ)

    def tilted(l0: float, l1: float, n: int) -> torch.Tensor:
        # n points after l0 ending on l1, λ-density ∝ exp(−taper·λ)
        u = torch.arange(1, n + 1, dtype=torch.float64) / n
        if taper == 0:
            return l0 + (l1 - l0) * u
        e0, e1 = math.exp(-taper * l0), math.exp(-taper * l1)
        return -torch.log((1 - u) * e0 + u * e1) / taper

    sigma_end = float(schedule.t_to_sigma(schedule.multiplier / steps))
    lam_top, lam_fin, lam_knee = lam(top_sigma), lam(sigma_end), lam(pump_end)
    if lam_fin - lam_knee <= 1e-6:
        lamv = tilted(lam_top, lam_fin, steps - 1)      # one band, pumped throughout
    else:
        n_tail = min(max(1, round(tail_share * steps)), steps - 2)
        n_band = steps - 1 - n_tail
        tail = lam_knee + (lam_fin - lam_knee) * torch.arange(1, n_tail + 1, dtype=torch.float64) / n_tail
        lamv = torch.cat([tilted(lam_top, lam_knee, n_band), tail])
    sigmas = torch.cat([torch.tensor([sigma_max], dtype=torch.float64), (-lamv).sigmoid()])
    return append_zero(sigmas.to(device=device, dtype=dtype))


# Flow table schedulers, evaluated against a FlowSamplingView. ``flow`` /
# ``flow_dyn`` / ``oss`` are computed in the pipelines. ``ddim_uniform`` is
# SD-only: it starts below σ_max, and the flow pipelines init at σ_max == 1.
_FLOW_TABLE_SCHEDULERS = {
    "sgm_uniform": sgm_uniform_schedule,
    "simple": simple_schedule,
    "normal": normal_schedule,
    "infinity": infinity_schedule,
    "infinity_htds": infinity_htds_schedule,
    "linear_quadratic": linear_quadratic_schedule,
    "smoothstep": smoothstep_schedule,
    "beta": beta_schedule,
    "beta_mix": beta_mix_schedule,
    "pump_dual": pump_dual_schedule,
    "pump_taper": pump_taper_schedule,
}


def flow_table_schedule(scheduler: str, shift: float, steps: int, *,
                        alpha: float = 0.6, beta: float = 0.6,
                        threshold_noise: float = 0.025,
                        bm_weight: float = 0.5,
                        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
                        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
                        pump_end: float = 0.45, pump_share: float = 0.85,
                        device: torch.device | str = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a flow sigma schedule for a table/timestep scheduler against a
    :class:`FlowSamplingView`. Also handles ``kl_optimal``.

    ``alpha``/``beta`` tune ``beta``, ``bm_*`` tune ``beta_mix``,
    ``threshold_noise`` tunes ``linear_quadratic``, ``pump_end``/``pump_share``
    tune ``pump_dual`` (``pump_end`` also ``pump_taper``). Schedulers ignore
    the knobs they don't take."""
    view = FlowSamplingView(shift, device=device, dtype=dtype)
    if scheduler == "kl_optimal":
        return kl_optimal_schedule(steps, float(view.sigma_min), float(view.sigma_max),
                                   device=device, dtype=dtype)
    try:
        fn = _FLOW_TABLE_SCHEDULERS[scheduler]
    except KeyError:
        raise ValueError(f"unknown flow table scheduler {scheduler!r}") from None
    extra = {}
    if scheduler == "beta":
        extra = {"alpha": alpha, "beta": beta}
    elif scheduler == "beta_mix":
        extra = {"weight": bm_weight, "alpha1": bm_alpha1, "beta1": bm_beta1,
                 "alpha2": bm_alpha2, "beta2": bm_beta2}
    elif scheduler == "linear_quadratic":
        extra = {"threshold_noise": threshold_noise}
    elif scheduler == "pump_dual":
        extra = {"pump_end": pump_end, "pump_share": pump_share}
    elif scheduler == "pump_taper":
        extra = {"pump_end": pump_end}
    return fn(view, steps, device=device, dtype=dtype, **extra)
