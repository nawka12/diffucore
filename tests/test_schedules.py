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


def test_flow_table_schedule_dispatches_all_names():
    # ddim_uniform is intentionally SD-only (starts below σ_max), so it is not a
    # flow table scheduler — see schedules._FLOW_TABLE_SCHEDULERS.
    for name in ("sgm_uniform", "simple", "normal", "linear_quadratic", "smoothstep", "beta", "kl_optimal"):
        sig = S.flow_table_schedule(name, shift=3.0, steps=12)
        assert sig[-1].item() == 0.0
        assert torch.all(sig[:-1] >= sig[1:]), name
        assert torch.isfinite(sig).all(), name
        assert abs(sig[0].item() - 1.0) < 1e-3, name   # flow init assumes σ_max == 1


def test_flow_table_schedule_forwards_knobs():
    # beta α/β and linear_quadratic threshold_noise must reach their schedulers;
    # the defaults reproduce the no-knob call (so generation is unchanged when the
    # settings panel is untouched), and a knob-agnostic scheduler ignores them.
    base_beta = S.flow_table_schedule("beta", shift=3.0, steps=12)
    assert torch.allclose(base_beta, S.flow_table_schedule("beta", shift=3.0, steps=12, alpha=0.6, beta=0.6))
    assert not torch.allclose(base_beta, S.flow_table_schedule("beta", shift=3.0, steps=12, alpha=0.3, beta=0.9))

    base_lq = S.flow_table_schedule("linear_quadratic", shift=3.0, steps=12)
    assert torch.allclose(base_lq, S.flow_table_schedule("linear_quadratic", shift=3.0, steps=12, threshold_noise=0.025))
    assert not torch.allclose(base_lq, S.flow_table_schedule("linear_quadratic", shift=3.0, steps=12, threshold_noise=0.2))

    assert torch.allclose(
        S.flow_table_schedule("sgm_uniform", shift=3.0, steps=12),
        S.flow_table_schedule("sgm_uniform", shift=3.0, steps=12, alpha=0.1, beta=0.9, threshold_noise=0.5),
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


def test_flow_budget_schedule_linear_spacing():
    """Linear-σ spacing: uniform Δσ between consecutive steps, trailing 0."""
    sig = S.flow_budget_schedule(18, 0.002, 80.0)
    assert sig.shape[0] == 19
    assert sig[-1].item() == 0.0
    assert abs(sig[0].item() - 80.0) < 1e-4
    assert abs(sig[-2].item() - 0.002) < 1e-4
    inner = sig[:-1]
    diffs = inner[:-1] - inner[1:]
    assert torch.allclose(diffs, diffs.mean().expand_as(diffs), atol=1e-4)


def test_flow_budget_schedule_flow_range():
    """Works on the flow σ∈[0,1] range (Anima/FLUX) the same as VE."""
    sig = S.flow_budget_schedule(20, 1.0 / 1000, 1.0)
    assert sig.shape[0] == 21
    assert sig[-1].item() == 0.0
    assert abs(sig[0].item() - 1.0) < 1e-6
    assert torch.all(sig[:-1] >= sig[1:])


def test_flow_budget_schedule_invalid_steps():
    import pytest
    with pytest.raises(ValueError):
        S.flow_budget_schedule(0, 0.1, 1.0)
