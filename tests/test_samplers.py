import torch

from diffucore.sampling import schedules as S
from diffucore.sampling import samplers as K


def const_denoiser(target):
    """A denoiser that always predicts the same clean sample ``target``.

    For this denoiser the probability-flow ODE dx/dσ = (x - target)/σ has the
    exact linear solution x(σ) - target = (x_init - target) · (σ / σ_init), so we
    can check the samplers against a closed form.
    """

    def model(x, sigma):
        return target.expand_as(x).clone()

    return model


def test_euler_matches_analytic_linear_trajectory():
    target = torch.full((1, 3, 4, 4), 0.3)
    sigmas = S.karras_schedule(15, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    sigma0 = sigmas[0].item()

    seen = []

    def cb(i, sigma, x, denoised):
        seen.append((sigma.item(), x.clone()))

    out = K.sample_euler(const_denoiser(target), x_init.clone(), sigmas, callback=cb)

    for sig, x_at in seen:  # Euler is exact for this linear ODE
        expected = target + (x_init - target) * (sig / sigma0)
        assert torch.allclose(x_at, expected, atol=1e-4)
    assert torch.allclose(out, target, atol=1e-4)  # ends at the clean sample


def test_heun_matches_analytic_linear_trajectory():
    target = torch.full((1, 3, 4, 4), -0.2)
    sigmas = S.karras_schedule(12, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    out = K.sample_heun(const_denoiser(target), x_init, sigmas)
    assert torch.allclose(out, target, atol=1e-4)


def test_euler_ancestral_ends_at_clean_sample():
    # Even with stochastic re-injection, the final step (σ_next == 0) lands on the
    # constant prediction.
    target = torch.zeros(1, 3, 4, 4)
    sigmas = S.karras_schedule(20, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    g = torch.Generator().manual_seed(0)
    out = K.sample_euler_ancestral(const_denoiser(target), x_init, sigmas, eta=1.0, generator=g)
    assert torch.allclose(out, target, atol=1e-4)
    assert torch.isfinite(out).all()


def test_er_sde_ve_ends_at_clean_sample():
    # Multi-stage stochastic solver, but σ_next == 0 on the final step lands on
    # the constant prediction.
    target = torch.zeros(1, 3, 4, 4)
    sigmas = S.karras_schedule(20, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    g = torch.Generator().manual_seed(0)
    out = K.sample_er_sde(const_denoiser(target), x_init, sigmas, generator=g, model_type="ve")
    assert torch.allclose(out, target, atol=1e-4)
    assert torch.isfinite(out).all()


def test_er_sde_flow_is_finite_and_ends_clean():
    # Flow mode offsets the first sigma off 1.0 (alpha would be 0 there); the
    # whole trajectory must stay finite and still land on the prediction.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    assert sigmas[0].item() == 1.0
    x_init = torch.randn(1, 16, 4, 4)
    g = torch.Generator().manual_seed(0)
    out = K.sample_er_sde(
        const_denoiser(target), x_init, sigmas,
        generator=g, model_type="flow", shift=3.0,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_er_sde_seed_reproducible():
    target = torch.zeros(1, 3, 4, 4)
    sigmas = S.karras_schedule(10, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    a = K.sample_er_sde(const_denoiser(target), x_init.clone(), sigmas,
                        generator=torch.Generator().manual_seed(7), model_type="ve")
    b = K.sample_er_sde(const_denoiser(target), x_init.clone(), sigmas,
                        generator=torch.Generator().manual_seed(7), model_type="ve")
    assert torch.equal(a, b)


def test_ancestral_step_eta_zero_is_deterministic():
    sigma_from = torch.tensor(5.0)
    sigma_to = torch.tensor(2.0)
    down, up = K.get_ancestral_step(sigma_from, sigma_to, eta=0.0)
    assert down.item() == sigma_to.item()
    assert up.item() == 0.0


def test_ancestral_step_energy_conserved():
    # sigma_down^2 + sigma_up^2 == sigma_to^2 (the variance bookkeeping).
    sigma_from, sigma_to = torch.tensor(8.0), torch.tensor(3.0)
    down, up = K.get_ancestral_step(sigma_from, sigma_to, eta=1.0)
    assert torch.allclose(down ** 2 + up ** 2, sigma_to ** 2, atol=1e-5)


def test_to_d():
    x = torch.ones(1, 1, 2, 2) * 3.0
    denoised = torch.ones(1, 1, 2, 2)
    d = K.to_d(x, torch.tensor([2.0]), denoised)
    assert torch.allclose(d, torch.ones_like(d))  # (3 - 1) / 2 == 1


def test_get_sampler_dispatch_and_error():
    import pytest

    assert K.get_sampler("euler") is K.sample_euler
    with pytest.raises(ValueError):
        K.get_sampler("does_not_exist")


# ── SECANT ────────────────────────────────────────────────────────────


def test_secant_constant_x0_ends_clean():
    # With a constant-x0 denoiser the rectified-flow ODE is exactly linear, so
    # Euler is exact. SECANT's x0-secant slope is zero in that regime, so the
    # corrected branch must match Euler and the trajectory must land on x0.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 16, 4, 4)
    out = K.sample_secant(const_denoiser(target), x_init.clone(), sigmas, curvature=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_curvature_zero_equals_euler():
    # curvature=0 ⇒ beta=0 everywhere ⇒ output must equal sample_euler.
    target = torch.full((1, 8, 4, 4), -0.05)
    sigmas = S.acas_flow_schedule(12, shift=3.0)
    x_init = torch.randn(1, 8, 4, 4)
    euler_out = K.sample_euler(const_denoiser(target), x_init.clone(), sigmas)
    secant_out = K.sample_secant(const_denoiser(target), x_init.clone(), sigmas, curvature=0.0)
    assert torch.allclose(secant_out, euler_out, atol=1e-6)


def test_secant_high_curvature_stays_finite_on_acas_schedule():
    # Full correction (curvature=1) on the ACAS schedule must not NaN/Inf and
    # must still converge to the constant prediction.
    target = torch.full((1, 16, 8, 8), 0.0)
    sigmas = S.acas_flow_schedule(20, shift=3.0, curvature=0.5)
    x_init = torch.randn(1, 16, 8, 8)
    out = K.sample_secant(const_denoiser(target), x_init, sigmas, curvature=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_sigma_collision_falls_back_to_euler():
    # Two consecutive identical sigmas would blow up the slope; the eps_sigma
    # guard must catch it and fall back to Euler for that step.
    target = torch.full((1, 4, 4, 4), 0.2)
    sigmas = torch.tensor([1.0, 0.6, 0.6, 0.3, 0.0])
    x_init = torch.randn(1, 4, 4, 4)
    out = K.sample_secant(const_denoiser(target), x_init, sigmas, curvature=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_s_noise_changes_output_and_is_seed_reproducible():
    # SDE variant: noise injection must (a) change the trajectory and (b) be
    # reproducible under the same generator seed. Capture the latent at the
    # last non-zero sigma via the callback, since σ_next == 0 forces both runs
    # onto the constant prediction at the very end.
    target = torch.zeros(1, 8, 4, 4)
    sigmas = S.acas_flow_schedule(12, shift=3.0)
    x_init = torch.randn(1, 8, 4, 4)

    def run(seed):
        last = {}
        def cb(i, sigma, x, denoised):
            last["x"] = x.clone()
        K.sample_secant(
            const_denoiser(target), x_init.clone(), sigmas,
            curvature=0.5, s_noise=0.5,
            generator=torch.Generator().manual_seed(seed), callback=cb,
        )
        return last["x"]

    def run_det():
        last = {}
        def cb(i, sigma, x, denoised):
            last["x"] = x.clone()
        K.sample_secant(
            const_denoiser(target), x_init.clone(), sigmas,
            curvature=0.5, s_noise=0.0, callback=cb,
        )
        return last["x"]

    sde_a, sde_b = run(11), run(11)
    det = run_det()
    assert torch.isfinite(sde_a).all()
    assert torch.equal(sde_a, sde_b)
    assert not torch.allclose(sde_a, det, atol=1e-5)


def test_secant_registered_in_sampler_table():
    assert K.get_sampler("secant") is K.sample_secant
