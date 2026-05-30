import torch

from diffucore.sampling import parameterization as P


def test_make_betas_scaled_linear():
    betas = P.make_betas("scaled_linear", 1000)
    assert betas.shape[0] == 1000
    assert torch.all(betas > 0) and torch.all(betas < 1)
    assert torch.all(betas[1:] >= betas[:-1])          # monotonically increasing


def test_make_betas_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        P.make_betas("nope", 10)


def test_discrete_schedule_ascending_sigmas():
    sched = P.DiscreteSchedule(P.make_betas("scaled_linear", 1000))
    assert sched.sigma_min < sched.sigma_max
    assert torch.all(sched.sigmas[1:] >= sched.sigmas[:-1])


def test_zero_terminal_snr_rescale():
    """ZTSNR rescale preserves sigma_min (anchored at index 0), blows sigma_max up
    from the ~14.6 default to a large finite value, and stays monotonic ascending."""
    betas = P.make_betas("scaled_linear", 1000)
    base = P.DiscreteSchedule(betas)
    ztsnr = P.DiscreteSchedule(betas, zero_terminal_snr=True)
    assert torch.allclose(ztsnr.sigma_min, base.sigma_min, atol=1e-4)
    assert base.sigma_max < 20.0
    assert ztsnr.sigma_max > 1000.0 and torch.isfinite(ztsnr.sigma_max)
    assert torch.all(ztsnr.sigmas[1:] > ztsnr.sigmas[:-1])


def test_sigma_t_roundtrip():
    sched = P.DiscreteSchedule(P.make_betas("scaled_linear", 1000))
    lo, hi = sched.sigma_min.item(), sched.sigma_max.item()
    probe = torch.tensor([0.1, 0.5, 1.0, 5.0, 10.0]).clamp(lo, hi)
    back = sched.t_to_sigma(sched.sigma_to_t(probe))
    assert torch.allclose(back, probe, rtol=1e-2, atol=1e-2)


def test_eps_scaling_zero_prediction_is_identity():
    # eps-prediction has c_skip == 1, so a zero noise prediction leaves x unchanged.
    s = P.EpsScaling()
    x = torch.randn(2, 4, 8, 8)
    denoised = s.denoise(x, torch.tensor(2.5), torch.zeros_like(x))
    assert torch.allclose(denoised, x)


def test_eps_denoise_matches_formula():
    s = P.EpsScaling()
    x = torch.randn(1, 4, 4, 4)
    eps = torch.randn_like(x)
    sigma = torch.tensor(1.3)
    expected = x - sigma * eps           # c_skip=1, c_out=-sigma
    assert torch.allclose(s.denoise(x, sigma, eps), expected, atol=1e-5)


def test_v_scaling_small_sigma_is_near_identity():
    # As sigma -> 0: c_skip -> 1, c_out -> 0.
    s = P.VScaling()
    x = torch.randn(1, 4, 8, 8)
    denoised = s.denoise(x, torch.tensor(1e-4), torch.randn_like(x))
    assert torch.allclose(denoised, x, atol=1e-3)


def test_c_in_in_unit_range():
    for s in (P.EpsScaling(), P.VScaling()):
        _, _, c_in = s.scalings(torch.tensor([0.01, 1.0, 10.0]))
        assert torch.all(c_in > 0) and torch.all(c_in <= 1.0)


def test_model_input_scaling():
    s = P.EpsScaling()
    x = torch.randn(2, 4, 8, 8)
    sigma = torch.tensor(3.0)
    expected = x / (sigma ** 2 + 1.0).sqrt()
    assert torch.allclose(s.model_input(x, sigma), expected, atol=1e-5)


def test_append_dims():
    x = torch.randn(4)
    assert P.append_dims(x, 4).shape == (4, 1, 1, 1)
    scalar = torch.tensor(2.0)
    assert P.append_dims(scalar, 3).shape == (1, 1, 1)
