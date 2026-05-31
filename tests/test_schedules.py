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


def test_acas_flow_schedule_endpoints_and_descent():
    sig = S.acas_flow_schedule(24, shift=3.0)
    assert sig.shape[0] == 25
    assert sig[-1].item() == 0.0
    assert torch.all(sig[:-1] >= sig[1:])
    assert abs(sig[0].item() - 1.0) < 1e-6
    expected_min = 3.0 * (1.0 / 24.0) / (1.0 + 2.0 * (1.0 / 24.0))
    assert abs(sig[-2].item() - expected_min) < 1e-6


def test_acas_flow_schedule_reallocates_steps_from_uniform_flow():
    steps = 32
    acas = S.acas_flow_schedule(steps, shift=3.0)[:-1]
    flow = S.flow_matching_schedule(steps, shift=3.0)[:-1]

    assert not torch.allclose(acas, flow)
    # ACAS allocates more points to the detail/refinement tail than uniform
    # flow, so its late sigma gaps should be smaller on average.
    acas_tail_gap = (acas[-8:-1] - acas[-7:]).abs().mean()
    flow_tail_gap = (flow[-8:-1] - flow[-7:]).abs().mean()
    assert acas_tail_gap < flow_tail_gap


def test_acas_flow_invalid_args_raise():
    import pytest

    with pytest.raises(ValueError):
        S.acas_flow_schedule(0, shift=3.0)
    with pytest.raises(ValueError):
        S.acas_flow_schedule(5, shift=0.5)
    with pytest.raises(ValueError):
        S.acas_flow_schedule(5, shift=3.0, grid_size=8)
    with pytest.raises(ValueError):
        S.acas_flow_schedule(5, shift=3.0, curvature=1.5)


def test_acas_flow_curvature_zero_matches_uniform_flow():
    acas = S.acas_flow_schedule(20, shift=3.0, curvature=0.0)
    flow = S.flow_matching_schedule(20, shift=3.0)
    assert torch.allclose(acas, flow, atol=1e-6)
