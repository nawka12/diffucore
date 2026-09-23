import math

import torch

from diffucore.sampling import schedules as S


def test_append_zero():
    x = torch.tensor([3.0, 2.0, 1.0])
    out = S.append_zero(x)
    assert out.shape[0] == 4
    assert out[-1].item() == 0.0


def test_karras_descending_and_endpoints():
    sig = S.karras_schedule(20, sigma_min=0.0292, sigma_max=14.6)
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])              # non-increasing
    assert abs(sig[0].item() - 14.6) < 1e-3            # starts at sigma_max
    assert abs(sig[-2].item() - 0.0292) < 1e-3         # last nonzero == sigma_min


def test_exponential_is_log_linear():
    sig = S.exponential_schedule(10, sigma_min=0.1, sigma_max=10.0)
    inner = sig[:-1]
    log_diffs = inner.log()[1:] - inner.log()[:-1]
    assert torch.allclose(log_diffs, log_diffs.mean().expand_as(log_diffs), atol=1e-5)
    assert abs(inner[0].item() - 10.0) < 1e-4
    assert abs(inner[-1].item() - 0.1) < 1e-4


def test_polyexponential_rho1_matches_exponential():
    a = S.polyexponential_schedule(12, 0.05, 8.0, rho=1.0)
    b = S.exponential_schedule(12, 0.05, 8.0)
    assert torch.allclose(a, b, atol=1e-5)


def test_invalid_steps_raises():
    import pytest

    with pytest.raises(ValueError):
        S.karras_schedule(0, 0.1, 10.0)


def test_flow_matching_schedule_endpoints_and_descent():
    """Descending, trailing 0, σ_max == shift·1/shift == 1.0, σ_min near 1/N."""
    sig = S.flow_matching_schedule(20, shift=3.0)
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])           # non-increasing
    assert abs(sig[0].item() - 1.0) < 1e-6           # σ_max == 1 for any shift
    # σ_min = shift·(1/N) / (1 + (shift−1)/N).  With N=20, shift=3:
    #   = 3/20 / (1 + 2/20) = 0.15 / 1.10 ≈ 0.13636…
    expected_min = 3.0 * (1.0 / 20.0) / (1.0 + 2.0 * (1.0 / 20.0))
    assert abs(sig[-2].item() - expected_min) < 1e-6


def test_flow_matching_shift_one_is_linear():
    """``shift == 1`` collapses to the uniform-in-t schedule (the SD3 trivial
    case)."""
    sig = S.flow_matching_schedule(10, shift=1.0)
    # inner = [1, 9/10, 8/10, ..., 1/10]
    expected = torch.tensor([(10 - i) / 10 for i in range(10)] + [0.0])
    assert torch.allclose(sig, expected, atol=1e-6)


def test_flow_matching_shift_concentrates_near_one():
    """Higher shift puts more steps near σ = 1 vs the linear baseline."""
    linear = S.flow_matching_schedule(20, shift=1.0)[:-1]
    shifted = S.flow_matching_schedule(20, shift=3.0)[:-1]
    # Every shifted σ should be ≥ the linear σ at the same index
    # (shift expands the high-σ tail at the expense of the low-σ region).
    assert torch.all(shifted >= linear - 1e-6)
    # And mid-range should differ noticeably.
    assert (shifted - linear).abs().max() > 0.1


def test_flow_matching_invalid_args_raise():
    import pytest

    with pytest.raises(ValueError):
        S.flow_matching_schedule(0, shift=3.0)
    with pytest.raises(ValueError):
        S.flow_matching_schedule(5, shift=0.5)


def test_kl_optimal_endpoints_and_descent():
    sig = S.kl_optimal_schedule(20, sigma_min=0.0292, sigma_max=14.6)
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])
    assert abs(sig[0].item() - 14.6) < 1e-3            # tan(atan(σ_max)) == σ_max
    assert abs(sig[-2].item() - 0.0292) < 1e-3


# ── Align Your Steps (AYS) ───────────────────────────────────────────
# Reference values are A1111's `get_align_your_steps_sigmas` output (log-linear
# interpolation of the paper's Table 3), matched bit-for-bit.


def test_align_your_steps_endpoints_and_descent():
    sig = S.align_your_steps_schedule(20, sigma_min=0.029, sigma_max=14.615, model="sdxl")
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])
    assert torch.isfinite(sig).all()
    assert abs(sig[0].item() - 14.615) < 1e-4          # table σ_max preserved
    assert abs(sig[-2].item() - 0.029) < 1e-4          # table σ_min preserved


def test_align_your_steps_10_matches_a1111_reference():
    # The 10-step SDXL schedule as produced by A1111 (log-linear fit through
    # the paper's 11 table points, then the trailing 0).
    expected = [14.615, 5.963397, 3.338965, 1.855046, 1.102326, 0.674959,
                0.431142, 0.260621, 0.122519, 0.029, 0.0]
    sig = S.align_your_steps_schedule(10, sigma_min=0.029, sigma_max=14.615, model="sdxl")
    assert torch.allclose(sig, torch.tensor(expected), atol=1e-5)


def test_align_your_steps_sd15_differs_from_sdxl():
    a = S.align_your_steps_schedule(10, sigma_min=0.029, sigma_max=14.615, model="sd15")
    b = S.align_your_steps_schedule(10, sigma_min=0.029, sigma_max=14.615, model="sdxl")
    assert not torch.allclose(a, b, atol=1e-6)
    assert abs(a[0].item() - 14.615) < 1e-4
    assert abs(a[-2].item() - 0.029) < 1e-4


def test_align_your_steps_invalid_args_raise():
    import pytest
    with pytest.raises(ValueError):
        S.align_your_steps_schedule(0, model="sdxl")
    with pytest.raises(ValueError):
        S.align_your_steps_schedule(10, model="not_a_model")
    # A zero-terminal-SNR range (σ_max ~ 4500) is far outside the AYS tables.
    with pytest.raises(ValueError):
        S.align_your_steps_schedule(10, sigma_min=0.03, sigma_max=4500.0, model="sdxl")


def _flow_view(shift=3.0):
    return S.FlowSamplingView(shift)


def test_normal_schedule_descends_to_zero():
    sig = S.normal_schedule(_flow_view(), 20)
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])
    assert torch.isfinite(sig).all()
    assert abs(sig[0].item() - 1.0) < 1e-3             # flow σ_max == 1


def test_infinity_schedule_endpoints_descent_and_sine_shift():
    # Same span as `normal`, first gap (1−s)× and last (1+s)× linear, with
    # s = min(0.6, steps/50).
    view = _flow_view()
    steps = 30
    sig = S.infinity_schedule(view, steps)
    nor = S.normal_schedule(view, steps)
    assert sig.shape[0] == steps + 1
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] > sig[1:])
    assert torch.equal(sig[0], nor[0])                  # f(0)=0: exact σ_max
    assert torch.allclose(sig[-2], nor[-2], atol=1e-6)  # f(1)=1: σ_min
    t = view.sigma_to_t(sig[:-1])
    lin_gap = (t[0] - t[-1]) / (steps - 1)
    assert abs((t[0] - t[1]) / lin_gap - 0.4) < 0.05    # ≈ 1−s gentler start
    assert abs((t[-2] - t[-1]) / lin_gap - 1.6) < 0.05  # ≈ 1+s more cleanup


def test_infinity_schedule_strength_adapts_to_steps():
    # Below the cap the perturbation scales as s = steps/50: the max deviation
    # of the warped ramp from linear is s·sin(πu)/π ≈ s/π at midpoint.
    view = _flow_view()
    for steps, s in ((5, 0.1), (25, 0.5)):
        t = view.sigma_to_t(S.infinity_schedule(view, steps)[:-1])
        f = (t[0] - t) / (t[0] - t[-1])
        u = torch.linspace(0.0, 1.0, steps)
        dev = (f - u).abs().max().item()
        assert abs(dev - s / math.pi) < 0.1 * s / math.pi, steps


def test_infinity_htds_endpoints_and_high_sigma_density():
    # The tanh bend is convex, so HTDS is high-σ-dense despite its name. Pinned
    # this way on purpose: upstream's README claims the opposite.
    view = _flow_view()
    steps = 30
    sig = S.infinity_htds_schedule(view, steps)
    nor = S.normal_schedule(view, steps)
    assert sig.shape[0] == steps + 1
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] > sig[1:])
    assert torch.equal(sig[0], nor[0])                  # decay(0)=1: exact σ_max
    assert torch.allclose(sig[-2], nor[-2], atol=1e-6)  # decay(1)=0: σ_min
    assert torch.all(sig[:-1] >= nor[:-1] - 1e-6)       # convex: never below normal
    mid = float(nor[0]) / 2.0
    assert int((sig[:-1] < mid).sum()) < int((nor[:-1] < mid).sum())


def test_infinity_htds_degenerates_to_linear_at_low_steps():
    # δ = clamp((steps−4)/26, 0, 1.8) is 0 at steps ≤ 4, upstream's guard for
    # distilled models: the bend vanishes and the ramp is exactly `normal`.
    view = _flow_view()
    for steps in (2, 3, 4):
        assert torch.allclose(S.infinity_htds_schedule(view, steps),
                              S.normal_schedule(view, steps), atol=1e-6)
    # ...and it is genuinely bent once past the guard.
    assert not torch.allclose(S.infinity_htds_schedule(view, 20),
                              S.normal_schedule(view, 20), atol=1e-3)


def test_ddim_uniform_descends_to_zero():
    sig = S.ddim_uniform_schedule(_flow_view(), 20)
    assert sig[-1].item() == 0.0
    assert sig.shape[0] >= 2
    assert torch.all(sig[:-1] >= sig[1:])
    assert torch.isfinite(sig).all()


def test_linear_quadratic_endpoints_and_descent():
    sig = S.linear_quadratic_schedule(_flow_view(), 20)
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])
    assert abs(sig[0].item() - 1.0) < 1e-6             # starts at σ_max (==1 for flow)


def test_smoothstep_endpoints_descent_and_u_shape():
    sig = S.smoothstep_schedule(_flow_view(), 28)
    assert sig.shape[0] == 29
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] > sig[1:])
    assert abs(sig[0].item() - 1.0) < 1e-6             # starts at σ_max (==1 for flow)
    # U-shaped: the first and last σ gaps are smaller than the largest mid gap.
    gaps = sig[:-2] - sig[1:-1]                        # exclude the final →0 jump
    assert gaps[0] < gaps.max() / 10
    assert gaps[-1] < gaps.max() / 2
    # ...and the low-σ tail is dense, unlike linear_quadratic's big final jump.
    assert sig[-2].item() < 0.02


def test_beta_endpoints_descent_and_u_shape():
    sig = S.beta_schedule(_flow_view(), 28)
    assert sig.shape[0] == 29
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] > sig[1:])
    assert abs(sig[0].item() - 1.0) < 1e-6             # σ(t=1) == 1: pure-noise init
    # last nonzero sigma is the table floor σ(1/1000), like the table walks
    view = _flow_view()
    assert abs(sig[-2].item() - float(view.sigma_min)) < 1e-4
    # U-shaped in t: the end gaps are small relative to the mid maximum.
    gaps = sig[:-2] - sig[1:-1]                        # exclude the final →0 jump
    assert gaps[0] < gaps.max() / 4
    assert gaps[-1] < gaps.max() / 2


def test_beta_inv_cdf_against_scipy():
    import pytest

    scipy_stats = pytest.importorskip("scipy.stats")
    q = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    for a, b in ((0.6, 0.6), (0.5, 0.7), (2.0, 2.0)):
        ours = S._beta_inv_cdf(q, a, b)
        ref = torch.tensor(scipy_stats.beta.ppf(q.numpy(), a, b))
        assert (ours - ref).abs().max().item() < 1e-4, (a, b)


def test_beta_invalid_args_raise():
    import pytest

    with pytest.raises(ValueError):
        S.beta_schedule(_flow_view(), 0)
    with pytest.raises(ValueError):
        S.beta_schedule(_flow_view(), 10, alpha=0.0)


def test_beta_mix_endpoints_descent_and_u_shape():
    """beta_mix with its tuned defaults stays descending with beta's endpoints
    and is U-shaped but asymmetric toward the detail (low-σ) end."""
    view = _flow_view()
    sig = S.beta_mix_schedule(view, 28)
    assert sig.shape[0] == 29
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] > sig[1:])
    assert abs(sig[0].item() - 1.0) < 1e-6                       # σ(t=1) == 1
    assert abs(sig[-2].item() - float(view.sigma_min)) < 1e-4    # table floor

    # U-shaped: both end gaps are smaller than the maximum mid-schedule gap.
    gaps = sig[:-2] - sig[1:-1]                                  # exclude the →0 jump
    assert gaps[0] < gaps.max() / 4
    assert gaps[-1] < gaps.max() / 2

    # Asymmetry toward the detail end, judged in timestep space (the shift map
    # alone already makes σ-gaps finer at the noise end).
    t = view.sigma_to_t(sig[:-1]) / view.multiplier             # drop the →0 sigma
    tgaps = t[:-1] - t[1:]
    assert tgaps[-1] < tgaps[0]                                  # clean end denser in t


def test_beta_mix_symmetric_params_match_beta():
    """Sanity: when the two mixture components are identical and equal-weight,
    beta_mix collapses to plain beta with the same (α, β). Verifies the
    mixture math is consistent with the single-Beta path."""
    sig_mix = S.beta_mix_schedule(_flow_view(), 20, weight=0.5,
                                       alpha1=0.6, beta1=0.6,
                                       alpha2=0.6, beta2=0.6)
    sig_beta = S.beta_schedule(_flow_view(), 20, alpha=0.6, beta=0.6)
    assert torch.allclose(sig_mix, sig_beta, atol=1e-4)


def test_beta_mix_invalid_args_raise():
    import pytest

    with pytest.raises(ValueError):
        S.beta_mix_schedule(_flow_view(), 0)
    with pytest.raises(ValueError):
        S.beta_mix_schedule(_flow_view(), 10, weight=0.0)        # collapses to single
    with pytest.raises(ValueError):
        S.beta_mix_schedule(_flow_view(), 10, weight=1.0)
    with pytest.raises(ValueError):
        S.beta_mix_schedule(_flow_view(), 10, alpha1=0.0)


def test_beta_mix_default_strictly_descending_at_high_step_counts():
    """The tuned defaults stay strictly descending at common step counts (the
    SD-literal β₂=0.5 collided steps at the table floor beyond ~40)."""
    view = _flow_view()
    for steps in (40, 50, 64, 100):
        sig = S.beta_mix_schedule(view, steps)
        assert torch.all(sig[:-1] > sig[1:]), f"floor collision at {steps} steps"


def test_flow_table_schedule_dispatches_all_names():
    # ddim_uniform is intentionally SD-only (starts below σ_max), so it is not a
    # flow table scheduler (see schedules._FLOW_TABLE_SCHEDULERS).
    for name in ("sgm_uniform", "simple", "normal", "infinity", "infinity_htds",
                 "linear_quadratic", "smoothstep", "beta", "beta_mix",
                 "pump_dual", "pump_taper", "kl_optimal"):
        sig = S.flow_table_schedule(name, shift=3.0, steps=12)
        assert sig[-1].item() == 0.0
        assert torch.all(sig[:-1] >= sig[1:]), name
        assert torch.isfinite(sig).all(), name
        assert abs(sig[0].item() - 1.0) < 1e-3, name   # flow init assumes σ_max == 1


def test_flow_table_schedule_forwards_knobs():
    # The panel knobs must reach their schedulers, the defaults must reproduce
    # the no-knob call, and other schedulers ignore them.
    base_beta = S.flow_table_schedule("beta", shift=3.0, steps=12)
    assert torch.allclose(base_beta, S.flow_table_schedule("beta", shift=3.0, steps=12, alpha=0.6, beta=0.6))
    assert not torch.allclose(base_beta, S.flow_table_schedule("beta", shift=3.0, steps=12, alpha=0.3, beta=0.9))

    base_mix = S.flow_table_schedule("beta_mix", shift=3.0, steps=12)
    assert torch.allclose(base_mix, S.flow_table_schedule(
        "beta_mix", shift=3.0, steps=12,
        bm_weight=0.5, bm_alpha1=0.8, bm_beta1=2.0, bm_alpha2=3.0, bm_beta2=0.7))
    # Changing any single knob perturbs the schedule (mixture is sensitive to all 5).
    perturbed = S.flow_table_schedule("beta_mix", shift=3.0, steps=12, bm_weight=0.3)
    assert not torch.allclose(base_mix, perturbed)
    perturbed = S.flow_table_schedule("beta_mix", shift=3.0, steps=12, bm_alpha2=5.0)
    assert not torch.allclose(base_mix, perturbed)

    base_lq = S.flow_table_schedule("linear_quadratic", shift=3.0, steps=12)
    assert torch.allclose(base_lq, S.flow_table_schedule("linear_quadratic", shift=3.0, steps=12, threshold_noise=0.025))
    assert not torch.allclose(base_lq, S.flow_table_schedule("linear_quadratic", shift=3.0, steps=12, threshold_noise=0.2))

    base_dual = S.flow_table_schedule("pump_dual", shift=3.0, steps=12)
    assert torch.allclose(base_dual, S.flow_table_schedule("pump_dual", shift=3.0, steps=12, pump_end=0.45, pump_share=0.85))
    assert not torch.allclose(base_dual, S.flow_table_schedule("pump_dual", shift=3.0, steps=12, pump_share=0.6))
    assert not torch.allclose(base_dual, S.flow_table_schedule("pump_dual", shift=3.0, steps=12, pump_end=0.3))

    # Knob-agnostic schedulers ignore all the per-scheduler knobs.
    assert torch.allclose(
        S.flow_table_schedule("sgm_uniform", shift=3.0, steps=12),
        S.flow_table_schedule("sgm_uniform", shift=3.0, steps=12,
                              alpha=0.1, beta=0.9, threshold_noise=0.5,
                              bm_weight=0.7, bm_alpha1=0.3, bm_beta1=1.0,
                              bm_alpha2=2.0, bm_beta2=0.4),
    )


def test_flow_table_schedule_rejects_ddim_uniform():
    import pytest

    with pytest.raises(ValueError):
        S.flow_table_schedule("ddim_uniform", shift=3.0, steps=12)


def test_flow_table_schedule_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        S.flow_table_schedule("nope", shift=3.0, steps=10)


def test_flow_matching_dynamic_shift_monotonic_and_anchor():
    """Flux-style mu interpolation: shift grows with the token count and lands
    near Anima's training shift (~3.16) at 1024² (4096 tokens)."""
    s_lo = S.flow_matching_dynamic_shift(1024)     # 512²
    s_mid = S.flow_matching_dynamic_shift(4096)    # 1024²
    s_hi = S.flow_matching_dynamic_shift(16384)    # 2048²
    assert s_lo < s_mid < s_hi
    assert abs(s_mid - 3.16) < 0.05
    # feeds flow_matching_schedule as a plain shift -> valid descending run
    sig = S.flow_matching_schedule(20, shift=s_mid)
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])


def _lam(sigmas):
    """flow half-logSNR −logit(σ) of a schedule's σ run (excluding trailing 0)."""
    sig = sigmas[:-1]
    return -sig.logit()


def _pumped_steps(sigmas, pump_end=0.45):
    """Steps whose starting σ ≥ pump_end, i.e. the pump injections."""
    sig = sigmas[:-1]
    return sum(1 for i in range(len(sig) - 1) if sig[i] >= pump_end)


def test_pump_dual_endpoints_descent_and_terminus():
    view = _flow_view()
    for steps in (8, 16, 32, 64):
        sig = S.pump_dual_schedule(view, steps)
        assert sig.shape[0] == steps + 1
        assert sig[-1].item() == 0.0
        assert torch.all(sig[:-1] > sig[1:])                 # strictly descending
        assert abs(sig[0].item() - 1.0) < 1e-6               # pure-noise init
        # terminates where `flow` does, σ(t = 1/steps), not the table floor
        ref = S.flow_matching_schedule(steps, shift=3.0)
        assert abs(sig[-2].item() - ref[-2].item()) < 2e-6


def test_pump_dual_terminus_is_flows_not_the_table_floor():
    """The schedule must spend no steps below flow's terminus: deeper termini
    were monotonically worse on the cogent3 toy (16× flow's error at 8 steps
    at the 0.003 table floor)."""
    view = _flow_view()
    for steps in (16, 24, 32):
        sig = S.pump_dual_schedule(view, steps)
        floor_end = S.beta_mix_schedule(view, steps)[-2].item()
        flow_end = S.flow_matching_schedule(steps, shift=3.0)[-2].item()
        assert sig[-2].item() > floor_end * 10, (steps, sig[-2].item())
        assert sum(1 for s in sig[:-1] if float(s) < flow_end) == 0, steps


def test_pump_dual_terminus_tracks_shift():
    """The terminus follows `shift` (σ(t = 1/steps) through the shift map),
    unlike the pumped band, which is defined in σ."""
    ends = [float(S.pump_dual_schedule(S.FlowSamplingView(sh), 32)[-2])
            for sh in (1.0, 3.0, 6.0)]
    assert ends == sorted(ends) and ends[0] < ends[-1], ends
    for sh, end in zip((1.0, 3.0, 6.0), ends):
        assert abs(end - float(S.flow_matching_schedule(32, shift=sh)[-2])) < 2e-6


def test_pump_dual_uniform_lambda_share_is_one_band():
    """At pump_share = S_hi/(S_hi + S_lo) both bands have equal λ-steps (one
    uniform grid). That point drifts with the budget (≈ 0.77 at 16 steps,
    0.69 at 32); the 0.85 default makes the pumped band the finer one."""
    view = _flow_view()
    for steps in (16, 24, 32):
        s_hi = math.log(1 / 0.45 - 1) - math.log(1 / 0.99 - 1)
        sigma_end = float(view.t_to_sigma(view.multiplier / steps))
        s_lo = math.log(1 / sigma_end - 1) - math.log(1 / 0.45 - 1)
        sig = S.pump_dual_schedule(view, steps, pump_share=s_hi / (s_hi + s_lo))
        hs = [float(l) for l in (_lam(sig)[1:] - _lam(sig)[:-1]) if math.isfinite(float(l))]
        assert max(hs) - min(hs) < 1e-3, steps
        assert s_hi / (s_hi + s_lo) < 0.85            # the default is pump-dense


def test_pump_dual_share_trades_injections_for_tail():
    """Raising pump_share moves steps from the tail into the pumped band: more
    injections, coarser final step."""
    view = _flow_view()
    for steps in (24, 32, 50):
        counts, lasts = [], []
        for ps in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
            sig = S.pump_dual_schedule(view, steps, pump_share=ps)
            counts.append(_pumped_steps(sig))
            lasts.append(float(_lam(sig)[-1] - _lam(sig)[-2]))
        assert counts == sorted(counts) and counts[0] < counts[-1], (steps, counts)
        assert lasts == sorted(lasts), (steps, lasts)       # tail coarsens with share


def test_pump_dual_injects_at_least_as_often_as_flow():
    """At 0.85 the pumped-step count beats `flow` (26 at 32 steps) and
    `beta_mix` (21); an early version starved the band to 13 and lost
    coherency."""
    view = _flow_view()
    for steps in (24, 28, 30, 32):
        mine = _pumped_steps(S.pump_dual_schedule(view, steps))
        assert mine >= _pumped_steps(S.flow_matching_schedule(steps, shift=3.0)), steps
        assert mine > _pumped_steps(S.beta_mix_schedule(view, steps)), steps


def test_pump_dual_degrades_to_one_band_without_room():
    """When σ(t = 1/steps) is at or above the cutoff (few steps, high shift) the
    run is one pumped uniform-λ grid. Shift 9 at 12 steps puts the terminus on
    0.45 exactly, which a naive split turns into duplicate sigmas."""
    for shift, steps in ((9.0, 12), (9.0, 8), (3.0, 4)):
        sig = S.pump_dual_schedule(S.FlowSamplingView(shift), steps)
        assert torch.all(sig[:-1] > sig[1:]), (shift, steps)
        hs = [float(l) for l in (_lam(sig)[2:] - _lam(sig)[1:-1])]
        assert max(hs) - min(hs) < 1e-3, (shift, steps, hs)   # single uniform-λ band


def test_pump_dual_join_lands_at_pump_end():
    """The band knee sits on the pump cutoff: fine λ-steps above (~0.18 λ),
    coarser below (~0.46 λ)."""
    view = _flow_view()
    sig = S.pump_dual_schedule(view, 32)
    run = sig[:-1]
    l = _lam(sig)
    steps = [float(l[i + 1] - l[i]) for i in range(len(run) - 1)]
    pumped = [s for i, s in enumerate(steps) if math.isfinite(s)
              and run[i + 1] >= 0.45 and run[i] < 1.0]
    fine = [s for i, s in enumerate(steps) if run[i + 1] < 0.45]
    assert max(pumped) < min(fine), (max(pumped), min(fine))
    # the knee point itself is at the requested pump_end
    lo, hi = None, None
    for i in range(len(run) - 1):
        if run[i] > 0.45 >= run[i + 1]:
            lo, hi = run[i + 1], run[i]
    assert lo is not None and hi is not None
    assert abs(float(hi) - 0.45) < 0.15 and abs(float(lo) - 0.45) < 0.15


def test_pump_dual_pump_end_moves_the_knee():
    view = _flow_view()
    for pe in (0.3, 0.6):
        sig = S.pump_dual_schedule(view, 32, pump_end=pe)
        assert torch.all(sig[:-1] > sig[1:])
        # knee follows: the boundary σ straddles pump_end
        run = sig[:-1]
        lo = hi = None
        for i in range(len(run) - 1):
            if run[i] > pe >= run[i + 1]:
                lo, hi = run[i + 1], run[i]
        assert lo is not None and hi is not None
        assert abs(float(hi) - pe) < 0.15


def test_pump_dual_top_sigma_caps_the_wasteful_top():
    """top_sigma caps the λ grid so the first step is a real burn-in jump to
    ~0.99 (a naive uniform-λ grid put 9 of 32 σ at ≥ 0.995; flow puts 1)."""
    view = _flow_view()
    for ps in (0.5, 0.65, 0.8):
        sig = S.pump_dual_schedule(view, 32, pump_share=ps)
        run = sig[:-1]
        assert int((run >= 0.995).sum()) <= 2, ps       # no near-identical calls
        assert 0.98 < float(run[1]) < 0.995            # first step lands in-family
    # A higher top_sigma lands the first post-burn-in point higher.
    low = float(S.pump_dual_schedule(view, 32, top_sigma=0.98)[1])
    high = float(S.pump_dual_schedule(view, 32, top_sigma=0.995)[1])
    assert 0.96 < low < high < 0.995


def test_pump_dual_invalid_args_raise():
    import pytest

    view = _flow_view()
    with pytest.raises(ValueError):
        S.pump_dual_schedule(view, 2)                 # needs ≥ 3 steps
    with pytest.raises(ValueError):
        S.pump_dual_schedule(view, 20, pump_share=0.0)
    with pytest.raises(ValueError):
        S.pump_dual_schedule(view, 20, pump_share=1.0)
    with pytest.raises(ValueError):
        S.pump_dual_schedule(view, 20, pump_end=0.0)
    with pytest.raises(ValueError):
        S.pump_dual_schedule(view, 20, pump_end=1.0)


# ── pump_taper ────────────────────────────────────────────────────────

def _lam_list(sig):
    return [math.log((1.0 - float(v)) / float(v)) for v in sig]


def test_pump_taper_endpoints_descent_and_terminus():
    view = _flow_view()
    for steps in (8, 16, 24, 30, 50, 64):
        sig = S.pump_taper_schedule(view, steps)
        assert sig.shape[0] == steps + 1
        assert sig[-1].item() == 0.0
        assert torch.all(sig[:-1] > sig[1:]), steps          # strictly descending
        assert abs(sig[0].item() - 1.0) < 1e-6               # pure-noise init
        # terminates where `flow` (and pump_dual) do: σ(t = 1/steps)
        ref = S.flow_matching_schedule(steps, shift=3.0)
        assert abs(sig[-2].item() - ref[-2].item()) < 2e-6, steps


def test_pump_taper_30_step_layout():
    # Design point: burn-in, 25 band points ending on 0.45, a 4-step tail; the
    # band opens at pump_dual@50's density (~0.115 λ) and widens.
    view = _flow_view()
    sig = S.pump_taper_schedule(view, 30)
    band, tail = sig[1:26], sig[26:30]
    assert abs(float(band[-1]) - 0.45) < 1e-6
    assert all(float(v) < 0.45 for v in tail)
    lb = _lam_list(torch.cat([torch.tensor([0.99]), band]))
    h = [lb[i + 1] - lb[i] for i in range(len(lb) - 1)]
    assert all(h[i] < h[i + 1] for i in range(len(h) - 1))   # tapering
    assert 0.10 < h[0] < 0.13 and 0.30 < h[-1] < 0.40
    lt = _lam_list(sig[25:30])
    ht = [lt[i + 1] - lt[i] for i in range(4)]
    assert max(ht) - min(ht) < 1e-4


def test_pump_taper_zero_taper_is_uniform_band():
    view = _flow_view()
    sig = S.pump_taper_schedule(view, 30, taper=0.0)
    lb = _lam_list(torch.cat([torch.tensor([0.99]), sig[1:26]]))
    h = [lb[i + 1] - lb[i] for i in range(len(lb) - 1)]
    assert max(h) - min(h) < 1e-4                            # float32 output


def test_pump_taper_tail_count_follows_share():
    view = _flow_view()
    for steps, want in ((12, 2), (20, 3), (30, 4), (50, 6)):
        sig = S.pump_taper_schedule(view, steps)
        assert sum(1 for v in sig[1:-1] if float(v) < 0.45 - 1e-6) == want, steps


def test_pump_taper_degrades_to_one_band_without_room():
    # When flow's terminus sits at or above the cutoff there is no refinement
    # band; the whole run is the tapered band, pumped end to end.
    view = S.FlowSamplingView(9.0)
    sig = S.pump_taper_schedule(view, 8)
    assert torch.all(sig[:-1] > sig[1:])
    ref = S.flow_matching_schedule(8, shift=9.0)
    assert float(ref[-2]) >= 0.45
    assert abs(sig[-2].item() - ref[-2].item()) < 2e-6


def test_pump_taper_invalid_args_raise():
    import pytest

    view = _flow_view()
    with pytest.raises(ValueError):
        S.pump_taper_schedule(view, 2)
    with pytest.raises(ValueError):
        S.pump_taper_schedule(view, 20, tail_share=0.0)
    with pytest.raises(ValueError):
        S.pump_taper_schedule(view, 20, taper=-0.1)
    with pytest.raises(ValueError):
        S.pump_taper_schedule(view, 20, pump_end=1.0)
