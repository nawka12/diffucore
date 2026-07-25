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
    "kl_optimal_schedule",
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
    "flow_table_schedule",
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


def kl_optimal_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """KL-optimal schedule (Karras-style, arXiv:2407.12173).

    Interpolates linearly in ``arctan(sigma)`` from ``sigma_max`` to ``sigma_min``,
    i.e. ``sigma_i = tan((1-i/(n-1))·atan(sigma_max) + (i/(n-1))·atan(sigma_min))``,
    then appends the trailing 0. Model-agnostic (works for VE and flow ranges)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ramp = torch.arange(steps, device=device, dtype=dtype) / max(steps - 1, 1)
    sigmas = (ramp * math.atan(sigma_min) + (1.0 - ramp) * math.atan(sigma_max)).tan()
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


def flow_matching_dynamic_shift(
    seq_len: int,
    *,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Flux-style resolution-aware shift for :func:`flow_matching_schedule`.

    Linearly interpolates the log-shift ``mu`` in the DiT token count
    ``seq_len = (H // 16) * (W // 16)`` between ``base_shift`` (the ``mu`` at
    ``base_seq_len``) and ``max_shift`` (the ``mu`` at ``max_seq_len``) — Flux's
    ``calculate_shift`` — then returns the multiplicative shift ``exp(mu)`` that
    :func:`flow_matching_schedule` consumes (its σ(t) map equals Flux's
    ``time_shift`` exactly when ``shift == exp(mu)``).

    Larger images get a larger shift, i.e. more steps near σ = 1 where the model
    resolves global composition; smaller images get less. At 1024² (4096 tokens)
    this returns ≈ 3.16, close to Anima's training shift of 3.0. ``base_shift``
    and ``max_shift`` are ``mu`` (log-shift) endpoints, matching Flux's
    (confusingly named) defaults.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = base_shift + m * (seq_len - base_seq_len)
    return math.exp(mu)


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


def normal_schedule(schedule, steps: int, *, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ComfyUI ``normal``: ``steps`` sigmas uniform in timestep between
    ``sigma_max`` and ``sigma_min`` (``normal_scheduler`` with ``sgm=False`` —
    all ``steps`` timesteps kept, trailing 0 appended)."""
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
    MIT; verified equivalent to upstream @4f72d8f, 2026-07-17): ``normal``'s
    linear timestep ramp warped by ``f(u) = u − s·sin(πu)/π``, which shrinks
    the first step's timestep gap to ``(1−s)×`` linear (gentler start) and
    grows the last step's to ``(1+s)×`` (more room for the final cleanup),
    with ``s = min(0.6, steps/50)`` adapting to the step count — near-linear
    at low steps, fully perturbed from 30 up. ``f`` is strictly increasing
    (``f′ ≥ 1−s > 0``) and fixes both endpoints, so the schedule still spans
    exactly ``sigma_max``→``sigma_min`` through the model's native σ(t) —
    flow-safe (starts at σ_max) and every sigma is from the training
    distribution, unlike sigma-space schedules such as ``karras``."""
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
    """Infinity Diffusion's "Hyperbolic Tail-Density" schedule (HTDS —
    galpt/infinity-diffusion ``omega``/``nano``, MIT, upstream @4319bc7):
    ``normal``'s linear timestep ramp bent by a hyperbolic tangent,
    ``decay(u) = tanh(δ·(1−u)) / tanh(δ)``, with ``δ = clamp((steps − 4)/26,
    0, 1.8)`` adapting to the step count. At ``steps ≤ 4``, ``δ`` is 0 and the
    schedule degenerates to exactly linear — upstream's guard for distilled
    models — saturating from ~50 steps up.

    ``decay`` is strictly decreasing with ``decay(0) = 1`` and ``decay(1) =
    0``, so the run spans exactly ``sigma_max``→``sigma_min`` through the
    model's native σ(t) — flow-safe, and every sigma is one the model was
    trained on, like ``normal`` and ``infinity`` and unlike sigma-space
    schedules such as ``karras``.

    **The name is backwards.** ``tanh(δ(1−u))/tanh(δ)`` is *convex* on
    ``[0, 1]`` — its slope at ``u=0`` is ``−δ·sech²(δ)/tanh(δ)``, shallower
    than linear, steepening to ``−δ/tanh(δ)`` at ``u=1``. So sigma is held
    high through the early trajectory and plunges at the end: HTDS is
    high-σ-dense, the *opposite* of the low-noise tail density upstream's
    README advertises (its "up to 45% of steps at σ ≤ 0.8" is, for the code as
    written, closer to 3%). Measured against ``normal`` at 50 flow steps, HTDS
    puts 7 sigmas below 0.5σ_max where ``normal`` puts 13. Ported faithfully
    anyway — it is what upstream ships and what its comparisons were made
    with — but pick it to spend budget on structure, not on texture.

    This is the schedule upstream pairs with ``infinity_omega``, and on that
    reading the pairing is coherent: the sampler's detail gain is also
    strongest at high σ."""
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
    """ComfyUI ``ddim_uniform``: pick sigmas from the model's (ascending) sigma
    table at a fixed stride from the low-noise end, then reverse to descending
    with a trailing 0."""
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
    """ComfyUI ``linear_quadratic`` (Mochi, arXiv:2412.xxxx): a normalized
    schedule that is linear up to ``threshold_noise`` over the first
    ``linear_steps`` (default ``steps // 2``) and quadratic after, scaled to the
    model's ``sigma_max``. Returns ``steps + 1`` descending sigmas ending at 0."""
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
    """U-shaped (endpoint-dense) flow schedule: smoothstep-eased ``t`` mapped
    through the model's shift.

    ``t`` is warped by ``u = t²·(3 − 2t)`` (the smoothstep polynomial, whose
    derivative vanishes at both ends) before the rectified-flow σ(t) map, so
    steps cluster near σ = 1 *and* near σ = 0 with a sparser middle — unlike
    ``linear_quadratic``, which spends its density budget at σ ≈ 1 only and
    ends on a large final jump. Designed to pair with σ-annealed ancestral
    samplers (``euler_ancestral_anneal``): the dense high-σ region feeds the
    stochastic burn-in, the dense low-σ tail feeds the near-deterministic
    detail refinement. Returns ``steps + 1`` descending sigmas ending at 0."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    t = torch.arange(steps, 0, -1, device=device, dtype=torch.float32) / steps
    u = t * t * (3.0 - 2.0 * t)
    sigmas = schedule.t_to_sigma(u * schedule.multiplier).to(device=device, dtype=dtype)
    return append_zero(sigmas)


def _beta_inv_cdf(q: torch.Tensor, alpha: float, beta: float, *, grid: int = 4096) -> torch.Tensor:
    """Inverse regularized incomplete beta function (the Beta(α, β) quantile
    function), in pure torch.

    The CDF is built by midpoint-rule integration of the unnormalized density
    ``x^(α−1)·(1−x)^(β−1)`` on a uniform grid (midpoints avoid evaluating the
    α, β < 1 endpoint singularities; the two edge cells, where the midpoint
    rule is poorest, use the exact leading-order masses ``h^α/α`` and
    ``h^β/β``), then inverted by linear interpolation. Accurate to ~1e-5 —
    far below the σ resolution any schedule needs — without a scipy
    dependency. ``q`` is float64-promoted."""
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
    """Beta-quantile schedule (ComfyUI's ``beta``; cf. Lee et al.,
    "Beta Sampling is All You Need", arXiv:2407.12173).

    Timesteps are placed at the quantiles of a Beta(α, β) distribution —
    ``t_i = 1 − BetaInvCDF(i/(n−1))`` — then mapped through the model's σ(t)
    (the shift map for rectified flow). With the α = β = 0.6 default the
    density is U-shaped in *t*: dense near t = 1 (σ = 1, global composition)
    and near t = 0 (fine detail), sparser in the middle. Unlike
    :func:`smoothstep_schedule` (a fixed easing), the endpoint emphasis is
    tunable via α (low-t end) and β (high-t end). The first sigma is exactly
    ``σ(1)`` (1.0 for flow — the pure-noise init) and the last nonzero sigma is
    the table floor ``σ(1/multiplier)``; a trailing 0 is appended."""
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
    """Inverse CDF of a two-component Beta mixture
    ``weight·Beta(α1, β1) + (1−weight)·Beta(α2, β2)``.

    Same midpoint-rule integration + linear-interp inversion scheme as
    :func:`_beta_inv_cdf`, applied to the *mixture* density. Both edge cells
    use the exact leading-order mass for each component (the mixture's edge
    mass is the weight-weighted sum of the two). Returns float64."""
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
    """Two-component Beta mixture schedule (``beta_mix``).

    A generalization of :func:`beta_schedule` that drops the symmetry
    constraint: the timestep density is
    ``weight·Beta(α1, β1) + (1−weight)·Beta(α2, β2)`` in the ``x = 1 − t``
    variable (so ``x = 0`` is the high-noise end, ``x = 1`` the clean end),
    then ``t_i = 1 − F_mix⁻¹(i/(n−1))`` is mapped through the model's σ(t).
    With symmetric ``α = β`` and the two components equal this collapses to
    the plain Beta schedule; the asymmetry lets the two endpoint peaks
    differ in shape.

    The defaults follow Lee et al.'s ("Beta Sampling is All You Need",
    arXiv:2407.12173 Fig. 2d) observation that latent-diffusion models want an
    *asymmetric* importance curve — more steps at the high-frequency detail
    (low-σ) end than at the high-noise end — but are tuned for Anima's
    rectified-flow ``σ(t)`` rather than transcribed literally. The paper's raw
    LDM params ``Beta(0.5, 2.0) + Beta(3.0, 0.5)`` misbehave under the flow
    shift map: the two ``x^{-1/2}`` singularities make the t-density nearly
    *symmetric* (squandering the mixture) and, once pushed through σ(t),
    over-pack the pure-noise ``σ ≈ 1`` end — where the denoiser changes least —
    while colliding several steps at the table floor for step counts ≳ 40.
    The tuned defaults keep a low-frequency peak at ``t ≈ 1``
    (``Beta(0.8, 2.0)`` — α₁ raised from 0.5 to relieve that over-pack without
    starving the high-σ end) and a stronger high-frequency peak near ``t ≈ 0``
    (``Beta(3.0, 0.7)`` — genuinely more concentrated than the noise end, yet
    clear of the floor: strictly descending through ~117 steps).
    ``weight = 0.5`` balances the two; symmetric ``Beta(0.6, 0.6)`` (the plain
    ``beta`` scheduler) instead forces both peaks to equal width.

    Pair with a 2nd-order solver (``dpmpp_2m``, ``heunpp2``) for best
    effect — step placement and per-step solver order are orthogonal
    efficiency axes, so the gain compounds. The first sigma is exactly
    ``σ(1)`` (1.0 for flow — the pure-noise init) and the last nonzero
    sigma is the table floor ``σ(1/multiplier)``; a trailing 0 is appended.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < weight < 1.0:
        raise ValueError("weight must be in (0, 1); 0 or 1 collapses to a "
                         "single Beta — use the 'beta' scheduler instead")
    if alpha1 <= 0 or beta1 <= 0 or alpha2 <= 0 or beta2 <= 0:
        raise ValueError("alpha1, beta1, alpha2, beta2 must all be > 0")
    q = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    x = _beta_mixture_inv_cdf(q, weight, alpha1, beta1, alpha2, beta2)
    t = (1.0 - x).clamp(min=1.0 / schedule.multiplier)
    sigmas = schedule.t_to_sigma(t.to(torch.float32) * schedule.multiplier).to(device=device, dtype=dtype)
    return append_zero(sigmas)


# Flow ("table"-style) schedulers addressable through a FlowSamplingView. ``flow``
# / ``flow_dyn`` / ``oss`` are computed directly in the pipelines; everything else
# routes here so the flow pipelines share one dispatch. ``ddim_uniform`` is absent
# on purpose: it starts below σ_max, which the flow pipelines' σ_max==1 pure-noise
# init assumes; it is offered for SD/SDXL only (which scales its init by σ[0]).
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
}


def flow_table_schedule(scheduler: str, shift: float, steps: int, *,
                        alpha: float = 0.6, beta: float = 0.6,
                        threshold_noise: float = 0.025,
                        bm_weight: float = 0.5,
                        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
                        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
                        device: torch.device | str = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a flow sigma schedule for the table/timestep-based schedulers by
    evaluating them against a :class:`FlowSamplingView` of the rectified-flow
    model. Handles ``sgm_uniform``, ``simple``, ``normal``, ``infinity``,
    ``infinity_htds``, ``linear_quadratic``, ``smoothstep``, ``beta``,
    ``beta_mix`` and ``kl_optimal``.

    ``alpha``/``beta`` tune the ``beta`` scheduler's Beta(α, β) endpoint
    density; ``bm_*`` tune the ``beta_mix`` two-Beta mixture;
    ``threshold_noise`` tunes ``linear_quadratic``'s linear/quadratic knee.
    All are ignored by the schedulers that don't take them."""
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
    return fn(view, steps, device=device, dtype=dtype, **extra)
