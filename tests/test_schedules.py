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
    # Infinity Diffusion's sine-perturbed timestep ramp (verified equivalent
    # to upstream @4f72d8f): same σ_max→σ_min span as `normal`, but the first
    # timestep gap shrinks to (1−s)× linear and the last grows to (1+s)×,
    # with s = min(0.6, steps/50) — saturated at 30 steps.
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
    # HTDS bends `normal`'s linear timestep ramp by tanh(δ(1−u))/tanh(δ) over
    # the same σ_max→σ_min span. That bend is CONVEX, so despite the branch's
    # "tail density" name the schedule holds sigma high and plunges at the end
    # — it is high-σ-dense, strictly less tail-dense than `normal`. Pinned in
    # this direction on purpose: upstream's README claims the opposite, and a
    # future "fix" that flips the curve would be a silent behavior change.
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
    # U-shaped density: steps cluster at BOTH ends — the first and last σ gaps
    # are smaller than the largest mid-schedule gap (the low-σ end less so,
    # since the shift=3 map trades some low-σ density for the high end).
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
    # U-shaped in t: quantiles cluster at both t ends, so the first σ gaps and
    # the last nonzero gaps are small relative to the mid-schedule maximum (the
    # low-σ end less so — the shift=3 map expands σ near t=0 by ~shift×).
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
    """beta_mix generalizes beta to a two-Beta mixture; with the tuned defaults
    (w=0.5, Beta(0.8,2.0)+Beta(3.0,0.7)) it stays descending with the same
    endpoints as beta (σ(1)=1, σ(1/1000)=table floor, trailing 0), and the
    density is U-shaped — denser at both ends, sparser in the middle — like
    beta but *asymmetric toward the detail end*: per Lee et al. Fig. 2(d)'s
    LDM importance curve, the high-freq (low-σ) peak is more concentrated than
    the high-noise peak."""
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

    # Asymmetric toward the detail end — the reason beta_mix exists over the
    # symmetric `beta`. Judged in *timestep* space: the flow shift map alone
    # already makes σ-gaps finer at the noise end (dσ/dt is ~9× smaller there),
    # so the asymmetry the mixture controls is only visible pre-shift.
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
    """The tuned defaults stay strictly descending across the step counts users
    actually pick. The original SD-literal defaults (β₂=0.5) over-concentrated
    the detail end so hard that several steps collided at the table floor for
    step counts ≳ 40 (equal σ = a wasted NFE); the tuned β₂=0.7 stays clear."""
    view = _flow_view()
    for steps in (40, 50, 64, 100):
        sig = S.beta_mix_schedule(view, steps)
        assert torch.all(sig[:-1] > sig[1:]), f"floor collision at {steps} steps"


def test_flow_table_schedule_dispatches_all_names():
    # ddim_uniform is intentionally SD-only (starts below σ_max), so it is not a
    # flow table scheduler — see schedules._FLOW_TABLE_SCHEDULERS.
    for name in ("sgm_uniform", "simple", "normal", "infinity", "infinity_htds",
                 "linear_quadratic", "smoothstep", "beta", "beta_mix",
                 "kl_optimal"):
        sig = S.flow_table_schedule(name, shift=3.0, steps=12)
        assert sig[-1].item() == 0.0
        assert torch.all(sig[:-1] >= sig[1:]), name
        assert torch.isfinite(sig).all(), name
        assert abs(sig[0].item() - 1.0) < 1e-3, name   # flow init assumes σ_max == 1


def test_flow_table_schedule_forwards_knobs():
    # beta α/β, beta_mix w/α₁/β₁/α₂/β₂, and linear_quadratic threshold_noise
    # must reach their schedulers; the defaults reproduce the no-knob call (so
    # generation is unchanged when the settings panel is untouched), and a
    # knob-agnostic scheduler ignores them.
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
