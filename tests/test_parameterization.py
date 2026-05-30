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


# --- flow matching (CONST) -------------------------------------------------

def test_flow_matching_scalings_are_const():
    """c_skip = 1, c_in = 1, c_out = -σ for all σ — the CONST convention."""
    s = P.FlowMatchingConstScaling()
    sigma = torch.tensor([0.05, 0.5, 0.99])
    c_skip, c_out, c_in = s.scalings(sigma)
    assert torch.allclose(c_skip, torch.ones_like(sigma))
    assert torch.allclose(c_in, torch.ones_like(sigma))
    assert torch.allclose(c_out, -sigma)


def test_flow_matching_model_input_is_identity():
    """The model sees the raw noisy latent — no input scaling."""
    s = P.FlowMatchingConstScaling()
    x = torch.randn(1, 4, 8, 8)
    sigma = torch.tensor(0.7)
    assert torch.equal(s.model_input(x, sigma), x)


def test_flow_matching_denoise_recovers_x0_from_velocity():
    """If the model returns the true velocity v = ε − x0, ``Scaling.denoise``
    must produce the exact x0 — algebraically, ``x_t − σ·v = x0``."""
    s = P.FlowMatchingConstScaling()
    torch.manual_seed(0)
    x_0 = torch.randn(1, 4, 4, 4)
    eps = torch.randn_like(x_0)
    for sigma_val in (0.05, 0.25, 0.5, 0.75, 0.99):
        sigma = torch.tensor(sigma_val)
        x_t = (1 - sigma) * x_0 + sigma * eps
        v = eps - x_0
        denoised = s.denoise(x_t, sigma, v)
        assert torch.allclose(denoised, x_0, atol=1e-5)


def test_flow_matching_euler_with_x0_oracle_lands_on_x0():
    """A perfect x0 oracle plus the existing Euler sampler should drive a
    flow-matching trajectory back to the true x0 (Euler integration of the
    rectified-flow ODE is exact for any constant x0 estimate — the last step
    is closed-form). This exercises the parameterization + schedule + sampler
    end-to-end without a real backbone."""
    from diffucore.sampling import (
        FlowMatchingConstScaling, DiscreteSchedule, ModelDenoiser,
        flow_matching_schedule, sample_euler, make_betas,
    )
    torch.manual_seed(0)
    x_0 = torch.randn(1, 4, 4, 4)

    class _X0Oracle:
        """Backbone surrogate that ignores its input and returns the true x0
        as a 'velocity' the FlowMatchingConstScaling would yield exactly x0."""
        def __call__(self, model_in, t, **_):
            # Model is asked for v; CONST scaling will compute x_t − σ·v.
            # We want denoised == x_0 ⇒ v = (x_t − x_0) / σ. The denoiser passes
            # us x_t (== model_in for CONST) and t == σ.
            sigma = t.view(-1, 1, 1, 1)
            return (model_in - x_0) / sigma

    sigmas = flow_matching_schedule(steps=20, shift=3.0)
    # DiscreteSchedule isn't really used here (CONST passes σ through as t),
    # but ModelDenoiser requires one for its sigma_to_t mapping; we just need
    # an identity-ish path so we hand it a schedule and override via a thin
    # wrapper that maps σ to itself.
    sched = DiscreteSchedule(make_betas("scaled_linear", 1000))
    sched.sigma_to_t = lambda s: torch.as_tensor(s, dtype=torch.float32)   # identity
    denoiser = ModelDenoiser(_X0Oracle(), FlowMatchingConstScaling(), sched)

    # Start at σ_max ≈ 1 with x = (1 − σ_max)·x_0 + σ_max·ε; with σ_max == 1
    # this is just ε. Use a fresh ε so the test exercises a non-trivial step.
    eps = torch.randn_like(x_0)
    x_start = (1 - sigmas[0]) * x_0 + sigmas[0] * eps
    final = sample_euler(denoiser, x_start, sigmas)
    assert torch.allclose(final, x_0, atol=1e-5)
