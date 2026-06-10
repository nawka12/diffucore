import pytest
import torch

from diffucore.sampling import schedules as S
from diffucore.sampling import samplers as K


def _ve_sigmas():
    return S.karras_schedule(20, 0.03, 14.6)


def _flow_sigmas():
    return S.flow_matching_schedule(16, shift=3.0)


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
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 8, 4, 4)
    euler_out = K.sample_euler(const_denoiser(target), x_init.clone(), sigmas)
    secant_out = K.sample_secant(const_denoiser(target), x_init.clone(), sigmas, curvature=0.0)
    assert torch.allclose(secant_out, euler_out, atol=1e-6)


def test_secant_high_curvature_stays_finite_on_flow_schedule():
    # Full correction (curvature=1) on the flow schedule must not NaN/Inf
    # and must still converge to the constant prediction.
    target = torch.full((1, 16, 8, 8), 0.0)
    sigmas = S.flow_matching_schedule(20, shift=3.0)
    x_init = torch.randn(1, 16, 8, 8)
    out = K.sample_secant(const_denoiser(target), x_init, sigmas, curvature=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_correction_gated_off_at_high_noise():
    # The x0 estimate is unreliable at high σ, so SECANT must not apply its
    # extrapolation there (beta → 0 as σ → 1); the correction ramps in only as
    # σ → 0. Unlike the constant-x0 tests, this uses a σ-dependent denoiser so the
    # secant slope is nonzero and the gate is actually exercised. Without the
    # (1 − σ) gate the high-noise steps over-correct and drift far from Euler.
    torch.manual_seed(0)
    target = torch.randn(1, 4, 8, 8)

    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)          # x0 settles toward target as σ→0
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)

    sigmas = S.flow_matching_schedule(20, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    secant_states, euler_states = {}, {}
    euler_out = K.sample_euler(
        model, x_init.clone(), sigmas,
        callback=lambda i, s, x, d: euler_states.__setitem__(i, x.clone()),
    )
    secant_out = K.sample_secant(
        model, x_init.clone(), sigmas, curvature=1.0,
        callback=lambda i, s, x, d: secant_states.__setitem__(i, x.clone()),
    )

    assert torch.isfinite(secant_out).all()
    # High-σ steps must track Euler (correction gated off there)...
    high_dev = max(
        (secant_states[i] - euler_states[i]).abs().max().item()
        for i in secant_states if float(sigmas[i]) >= 0.9
    )
    final_dev = (secant_out - euler_out).abs().max().item()
    assert high_dev < 0.1 * final_dev    # gate suppresses the high-noise correction
    assert final_dev > 1e-3              # but the correction is still active overall


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
    sigmas = S.flow_matching_schedule(12, shift=3.0)
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


# ── ComfyUI-parity samplers ──────────────────────────────────────────
# With a constant-x0 denoiser the probability-flow ODE is exactly linear (its
# derivative d = (x - target)/σ is constant along the exact trajectory), so every
# consistent solver must land on ``target`` on any descending schedule ending at
# 0. The clean-snap stochastic ones also land exactly because σ_next == 0 forces
# x = denoised on the final step.


@pytest.mark.parametrize("name", ["heunpp2", "ipndm", "ipndm_v", "res_multistep",
                                  "gradient_estimation", "lms"])
@pytest.mark.parametrize("sigmas_fn", [_ve_sigmas, _flow_sigmas])
def test_new_deterministic_samplers_land_on_target(name, sigmas_fn):
    target = torch.full((1, 4, 4, 4), 0.2)
    sigmas = sigmas_fn()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.get_sampler(name)(const_denoiser(target), x_init.clone(), sigmas)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-3)


@pytest.mark.parametrize("name,extra", [
    ("euler_ancestral", {}),
    ("dpmpp_2s_ancestral", {}),
    ("res_multistep_ancestral", {}),
    ("lcm", {}),
])
@pytest.mark.parametrize("model_type", ["ve", "flow"])
def test_new_ancestral_samplers_land_on_target(name, extra, model_type):
    target = torch.zeros(1, 4, 4, 4)
    sigmas = _flow_sigmas() if model_type == "flow" else _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    g = torch.Generator().manual_seed(0)
    out = K.get_sampler(name)(
        const_denoiser(target), x_init.clone(), sigmas,
        generator=g, model_type=model_type, **extra,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_euler_ancestral_flow_differs_from_ve_midtrajectory():
    # The headline fix: on a flow schedule the RF branch must actually run a
    # different (rectified-flow) ancestral step than the VE split, not silently
    # reuse the VE path. Capture the latent at the last non-zero sigma (both snap
    # to target at σ_next == 0).
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    def model(x, sigma):  # σ-dependent so the two branches diverge
        return 0.3 * torch.tanh(x)

    def run(model_type):
        last = {}
        K.sample_euler_ancestral(
            model, x_init.clone(), sigmas, model_type=model_type,
            generator=torch.Generator().manual_seed(1),
            callback=lambda i, s, x, d: last.__setitem__("x", x.clone()),
        )
        return last["x"]

    assert not torch.allclose(run("flow"), run("ve"), atol=1e-3)


def test_euler_ancestral_flow_seed_reproducible():
    target = torch.zeros(1, 4, 4, 4)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 4, 4, 4)
    a = K.sample_euler_ancestral(const_denoiser(target), x_init.clone(), sigmas,
                                 generator=torch.Generator().manual_seed(3), model_type="flow")
    b = K.sample_euler_ancestral(const_denoiser(target), x_init.clone(), sigmas,
                                 generator=torch.Generator().manual_seed(3), model_type="flow")
    assert torch.equal(a, b)


def test_dpmpp_2m_sde_heun_alias_resolves_and_lands_clean():
    target = torch.zeros(1, 4, 4, 4)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    fn = K.get_sampler("dpmpp_2m_sde_heun")
    out = fn(const_denoiser(target), x_init, sigmas, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_ddpm_finite_and_converges():
    # DDPM doesn't clean-snap (no x = denoised at the end); it's a contraction
    # toward the constant prediction, so check it stays finite and lands close.
    target = torch.full((1, 4, 4, 4), 0.1)
    sigmas = S.karras_schedule(30, 0.03, 14.6)
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.sample_ddpm(const_denoiser(target), x_init, sigmas,
                        generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=0.1)


# ── SECANT-ANNEAL ─────────────────────────────────────────────────────


def _anneal_sigma_dependent_model(target):
    # A σ-dependent x0 denoiser, so the secant slope is nonzero and the curvature
    # gate is actually exercised (a constant-x0 denoiser has zero slope, which
    # collapses any curvature back onto euler_ancestral_anneal).
    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)          # x0 settles toward target as σ→0
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)
    return model


def _last_nonzero_latent(fn, model, x_init, sigmas, **kw):
    # The latent the final callback sees is x at the last non-zero σ; σ_next == 0
    # forces every run onto the constant prediction, so compare just before it.
    last = {}
    fn(model, x_init.clone(), sigmas,
       callback=lambda i, s, x, d: last.__setitem__("x", x.clone()), **kw)
    return last["x"]


def test_secant_anneal_flow_only():
    with pytest.raises(ValueError):
        K.sample_secant_anneal(
            const_denoiser(torch.zeros(1, 4, 4, 4)), torch.randn(1, 4, 4, 4),
            S.flow_matching_schedule(8, shift=3.0), model_type="ve",
        )


def test_secant_anneal_constant_x0_ends_clean():
    # Stochastic by default, but the final step (σ_next == 0) lands on the
    # constant prediction regardless of the burn-in.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 16, 4, 4)
    out = K.sample_secant_anneal(
        const_denoiser(target), x_init.clone(), sigmas,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_anneal_curvature_zero_equals_euler_ancestral_anneal():
    # curvature=0 ⇒ x0_eff = x0 every step ⇒ the deterministic core is exactly
    # euler_ancestral_anneal's, and the same seed reproduces its ancestral draws,
    # so the trajectories coincide. curvature=1 (secant active) must diverge.
    torch.manual_seed(0)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    def run(fn, **kw):
        return _last_nonzero_latent(
            fn, model, x_init, sigmas,
            generator=torch.Generator().manual_seed(7), **kw)

    eaa = run(K.sample_euler_ancestral_anneal)
    sa0 = run(K.sample_secant_anneal, curvature=0.0)
    sa1 = run(K.sample_secant_anneal, curvature=1.0)
    assert torch.isfinite(sa0).all()
    assert torch.allclose(sa0, eaa, atol=1e-4)
    assert not torch.allclose(sa1, eaa, atol=1e-3)


def test_secant_anneal_eta_max_zero_equals_deterministic_secant():
    # eta_max=0 ⇒ σ_down = σ_next and no re-noise ⇒ the step reduces to exactly
    # the deterministic secant correction (same curvature, s_noise=0).
    torch.manual_seed(1)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    sec = _last_nonzero_latent(K.sample_secant, model, x_init, sigmas,
                               curvature=0.5, s_noise=0.0)
    saz = _last_nonzero_latent(K.sample_secant_anneal, model, x_init, sigmas,
                               curvature=0.5, eta_max=0.0)
    assert torch.isfinite(saz).all()
    assert torch.allclose(saz, sec, atol=1e-4)


def test_secant_anneal_seed_reproducible_and_stochastic():
    # Same seed ⇒ identical trajectory; the annealed burn-in (eta_max>0) must
    # actually change it versus the deterministic (eta_max=0) run.
    target = torch.zeros(1, 8, 4, 4)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 8, 4, 4)

    def run(**kw):
        return _last_nonzero_latent(K.sample_secant_anneal, const_denoiser(target),
                                    x_init, sigmas, **kw)

    a = run(generator=torch.Generator().manual_seed(3))
    b = run(generator=torch.Generator().manual_seed(3))
    det = run(eta_max=0.0)
    assert torch.equal(a, b)
    assert not torch.allclose(a, det, atol=1e-5)


def test_secant_anneal_registered_in_sampler_table():
    assert K.get_sampler("secant_anneal") is K.sample_secant_anneal


def test_all_registered_samplers_resolve():
    for name, fn in K.SAMPLERS.items():
        assert K.get_sampler(name) is fn
