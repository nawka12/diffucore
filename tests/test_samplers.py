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
                                  "gradient_estimation", "stork2", "infinity", "lms",
                                  "exp_heun_2_x0", "uni_pc", "uni_pc_bh2"])
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


# ── DPM++(2M)-ANNEAL ──────────────────────────────────────────────────


def test_dpmpp_2m_anneal_flow_only():
    with pytest.raises(ValueError):
        K.sample_dpmpp_2m_anneal(
            const_denoiser(torch.zeros(1, 4, 4, 4)), torch.randn(1, 4, 4, 4),
            S.flow_matching_schedule(8, shift=3.0), model_type="ve",
        )


def test_dpmpp_2m_anneal_constant_x0_ends_clean():
    # Stochastic burn-in at high σ, but the final step (σ_next == 0) snaps to the
    # constant prediction.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 16, 4, 4)
    out = K.sample_dpmpp_2m_anneal(
        const_denoiser(target), x_init.clone(), sigmas,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_dpmpp_2m_anneal_eta_max_zero_equals_dpmpp_2m_sde_flow():
    # eta_max=0 ⇒ eta=0 every step ⇒ no re-noise ⇒ the update is exactly the
    # deterministic DPM++(2M) flow multistep (dpmpp_2m_sde with eta=0, same
    # half-logSNR map and shift). A σ-dependent denoiser exercises the 2nd-order
    # correction term.
    torch.manual_seed(2)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    sde0 = _last_nonzero_latent(K.sample_dpmpp_2m_sde, model, x_init, sigmas,
                                eta=0.0, model_type="flow", shift=3.0)
    anz = _last_nonzero_latent(K.sample_dpmpp_2m_anneal, model, x_init, sigmas,
                               eta_max=0.0, model_type="flow", shift=3.0)
    assert torch.isfinite(anz).all()
    assert torch.allclose(anz, sde0, atol=1e-4)


def test_dpmpp_2m_anneal_seed_reproducible_and_stochastic():
    # Same seed ⇒ identical trajectory; the annealed burn-in (eta_max>0) must
    # actually change it versus the deterministic (eta_max=0) run.
    target = torch.zeros(1, 8, 4, 4)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 8, 4, 4)

    def run(**kw):
        return _last_nonzero_latent(K.sample_dpmpp_2m_anneal, const_denoiser(target),
                                    x_init, sigmas, model_type="flow", shift=3.0, **kw)

    a = run(generator=torch.Generator().manual_seed(3))
    b = run(generator=torch.Generator().manual_seed(3))
    det = run(eta_max=0.0)
    assert torch.equal(a, b)
    assert not torch.allclose(a, det, atol=1e-5)


def test_dpmpp_2m_anneal_registered_in_sampler_table():
    assert K.get_sampler("dpmpp_2m_anneal") is K.sample_dpmpp_2m_anneal


# ── EXP-HEUN-2-x0 ─────────────────────────────────────────────────────


def test_exp_heun_2_x0_constant_x0_matches_dpmpp_2m():
    # With a constant-x0 denoiser the semilinear ODE has no truncation error, so
    # every exponential-integrator solver coincides. In ``ve`` mode exp_heun uses
    # the same half-logSNR map (-log σ) as dpmpp_2m, so they must agree exactly.
    target = torch.full((1, 4, 8, 8), 0.15)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 8, 8) * sigmas[0]
    heun = K.sample_exp_heun_2_x0(const_denoiser(target), x_init.clone(), sigmas)
    ref = K.sample_dpmpp_2m(const_denoiser(target), x_init.clone(), sigmas)
    assert torch.isfinite(heun).all()
    assert torch.allclose(heun, ref, atol=1e-5)


def test_exp_heun_2_x0_corrector_is_active():
    # On a σ-dependent denoiser the second evaluation matters: the phi_2 (Heun)
    # and phi_1 (trapezoidal) correctors must differ from each other, and the
    # full method must differ from the multistep dpmpp_2m. A constant-x0 denoiser
    # would collapse all three together (covered by the test above), so use a
    # genuinely nonlinear one here.
    torch.manual_seed(0)
    target = torch.randn(1, 4, 8, 8)

    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)          # x0 settles toward target as σ→0
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)

    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 8, 8) * sigmas[0]
    phi2 = K.sample_exp_heun_2_x0(model, x_init.clone(), sigmas, solver_type="phi_2")
    phi1 = K.sample_exp_heun_2_x0(model, x_init.clone(), sigmas, solver_type="phi_1")
    twom = K.sample_dpmpp_2m(model, x_init.clone(), sigmas)
    assert torch.isfinite(phi2).all()
    assert not torch.allclose(phi2, phi1, atol=1e-4)   # both correctors wired
    assert not torch.allclose(phi2, twom, atol=1e-4)   # genuinely a different scheme


def test_exp_heun_2_x0_deterministic():
    target = torch.zeros(1, 4, 4, 4)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    a = K.sample_exp_heun_2_x0(const_denoiser(target), x_init.clone(), sigmas)
    b = K.sample_exp_heun_2_x0(const_denoiser(target), x_init.clone(), sigmas)
    assert torch.equal(a, b)


def test_exp_heun_2_x0_flow_finite_and_lands_clean():
    # Flow mode offsets the first sigma off 1.0 (where alpha = 1 - σ is 0 and the
    # half-logSNR is infinite); the whole trajectory must stay finite and land on
    # the constant prediction.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    assert sigmas[0].item() == 1.0
    x_init = torch.randn(1, 16, 4, 4)
    out = K.sample_exp_heun_2_x0(
        const_denoiser(target), x_init, sigmas, model_type="flow", shift=3.0,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_exp_heun_2_x0_bad_solver_type_raises():
    with pytest.raises(ValueError):
        K.sample_exp_heun_2_x0(
            const_denoiser(torch.zeros(1, 4, 4, 4)), torch.randn(1, 4, 4, 4),
            _ve_sigmas(), solver_type="phi_3",
        )


def test_exp_heun_2_x0_registered_in_sampler_table():
    assert K.get_sampler("exp_heun_2_x0") is K.sample_exp_heun_2_x0


# ── UniPC ─────────────────────────────────────────────────────────────


def _unipc_sigma_dependent_model(target):
    # A σ-dependent x0 denoiser so the higher-order predictor/corrector terms are
    # actually exercised (a constant-x0 denoiser zeros every divided difference,
    # collapsing all orders to the first-order exponential step).
    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)          # x0 settles toward target as σ→0
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)
    return model


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("variant", ["bh1", "bh2"])
@pytest.mark.parametrize("model_type,sigmas_fn", [("ve", _ve_sigmas), ("flow", _flow_sigmas)])
def test_uni_pc_constant_x0_lands_clean(order, variant, model_type, sigmas_fn):
    # With a constant-x0 denoiser the ODE is exactly linear, so every order/variant
    # must stay finite and land on the prediction (and the σ→0 snap guarantees the
    # endpoint).
    target = torch.full((1, 4, 4, 4), 0.2)
    sigmas = sigmas_fn()
    shift = 3.0 if model_type == "flow" else 1.0
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.sample_uni_pc(
        const_denoiser(target), x_init.clone(), sigmas,
        order=order, variant=variant, model_type=model_type, shift=shift,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-3)


def test_uni_pc_higher_order_and_variant_are_active():
    # On a σ-dependent denoiser the higher-order terms must actually change the
    # result: order 3 differs from order 1, bh1 differs from bh2, and UniPC is a
    # genuinely different scheme than the dpmpp_2m multistep.
    torch.manual_seed(0)
    target = torch.randn(1, 4, 8, 8)
    model = _unipc_sigma_dependent_model(target)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 8, 8) * sigmas[0]

    o1 = K.sample_uni_pc(model, x_init.clone(), sigmas, order=1, variant="bh1")
    o3_bh1 = K.sample_uni_pc(model, x_init.clone(), sigmas, order=3, variant="bh1")
    o3_bh2 = K.sample_uni_pc(model, x_init.clone(), sigmas, order=3, variant="bh2")
    twom = K.sample_dpmpp_2m(model, x_init.clone(), sigmas)

    assert torch.isfinite(o3_bh1).all()
    assert not torch.allclose(o3_bh1, o1, atol=1e-4)       # order actually matters
    assert not torch.allclose(o3_bh1, o3_bh2, atol=1e-5)   # bh1 vs bh2 both wired
    assert not torch.allclose(o3_bh1, twom, atol=1e-4)     # distinct from dpmpp_2m


def test_uni_pc_deterministic():
    target = torch.zeros(1, 4, 4, 4)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    a = K.sample_uni_pc(const_denoiser(target), x_init.clone(), sigmas)
    b = K.sample_uni_pc(const_denoiser(target), x_init.clone(), sigmas)
    assert torch.equal(a, b)


def test_uni_pc_flow_finite_and_lands_clean():
    # Flow mode offsets the first sigma off 1.0 (infinite half-logSNR there); the
    # whole trajectory must stay finite and land on the constant prediction.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    assert sigmas[0].item() == 1.0
    x_init = torch.randn(1, 16, 4, 4)
    out = K.sample_uni_pc(
        const_denoiser(target), x_init, sigmas, model_type="flow", shift=3.0,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_uni_pc_bad_args_raise():
    sigmas = _ve_sigmas()
    x = torch.randn(1, 4, 4, 4)
    with pytest.raises(ValueError):
        K.sample_uni_pc(const_denoiser(torch.zeros(1, 4, 4, 4)), x, sigmas, variant="bh3")
    with pytest.raises(ValueError):
        K.sample_uni_pc(const_denoiser(torch.zeros(1, 4, 4, 4)), x, sigmas, order=0)


def test_uni_pc_variants_registered():
    import functools
    for name, variant in [("uni_pc", "bh1"), ("uni_pc_bh2", "bh2")]:
        fn = K.get_sampler(name)
        assert isinstance(fn, functools.partial)
        assert fn.func is K.sample_uni_pc
        assert fn.keywords == {"variant": variant}


# ── UniPC-ANNEAL ──────────────────────────────────────────────────────


def test_uni_pc_anneal_flow_only():
    with pytest.raises(ValueError):
        K.sample_uni_pc_anneal(
            const_denoiser(torch.zeros(1, 4, 4, 4)), torch.randn(1, 4, 4, 4),
            S.flow_matching_schedule(8, shift=3.0), model_type="ve",
        )


def test_uni_pc_anneal_bad_args_raise():
    sigmas = S.flow_matching_schedule(8, shift=3.0)
    x = torch.randn(1, 4, 4, 4)
    with pytest.raises(ValueError):
        K.sample_uni_pc_anneal(const_denoiser(torch.zeros(1, 4, 4, 4)), x, sigmas, variant="bh3")
    with pytest.raises(ValueError):
        K.sample_uni_pc_anneal(const_denoiser(torch.zeros(1, 4, 4, 4)), x, sigmas, order=0)


def test_uni_pc_anneal_constant_x0_ends_clean():
    # Stochastic burn-in at high σ, but the final step (σ_next == 0) snaps to the
    # constant prediction.
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 16, 4, 4)
    out = K.sample_uni_pc_anneal(
        const_denoiser(target), x_init.clone(), sigmas,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_uni_pc_anneal_eta_max_zero_equals_uni_pc_bh2():
    # eta_max=0 ⇒ eta=0 every step ⇒ hh=-h, carry factor 1, no re-noise ⇒ the
    # update reduces to the deterministic UniPC step bit-for-bit (same default
    # variant bh2 / order 3 / flow map / shift). A σ-dependent denoiser exercises
    # the higher-order predictor-corrector residual, so the equivalence is not the
    # trivial first-order one.
    torch.manual_seed(2)
    target = torch.randn(1, 4, 8, 8)
    model = _unipc_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    upc = _last_nonzero_latent(K.sample_uni_pc, model, x_init, sigmas,
                               variant="bh2", model_type="flow", shift=3.0)
    anz = _last_nonzero_latent(K.sample_uni_pc_anneal, model, x_init, sigmas,
                               eta_max=0.0, variant="bh2", model_type="flow", shift=3.0)
    assert torch.isfinite(anz).all()
    assert torch.allclose(anz, upc, atol=1e-6)


def test_uni_pc_anneal_seed_reproducible_and_stochastic():
    # Same seed ⇒ identical trajectory; the annealed burn-in (eta_max>0) must
    # actually change it versus the deterministic (eta_max=0) run.
    target = torch.zeros(1, 8, 4, 4)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 8, 4, 4)

    def run(**kw):
        return _last_nonzero_latent(K.sample_uni_pc_anneal, const_denoiser(target),
                                    x_init, sigmas, model_type="flow", shift=3.0, **kw)

    a = run(generator=torch.Generator().manual_seed(3))
    b = run(generator=torch.Generator().manual_seed(3))
    det = run(eta_max=0.0)
    assert torch.equal(a, b)
    assert not torch.allclose(a, det, atol=1e-5)


def test_uni_pc_anneal_high_eta_stays_finite_with_order_ramp():
    # The order-ramp-up holds the order near 1 at high σ (where the injected noise
    # is largest), so even eta_max=1.0 on a σ-dependent denoiser — which exercises
    # the higher-order divided differences — stays finite and bounded rather than
    # blowing up. Compared against the deterministic (eta_max=0) run on the same
    # seed, the burn-in must also actually change the trajectory.
    torch.manual_seed(5)
    target = torch.randn(1, 4, 8, 8)
    model = _unipc_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    hi = _last_nonzero_latent(K.sample_uni_pc_anneal, model, x_init, sigmas,
                              eta_max=1.0, model_type="flow", shift=3.0,
                              generator=torch.Generator().manual_seed(0))
    det = _last_nonzero_latent(K.sample_uni_pc_anneal, model, x_init, sigmas,
                               eta_max=0.0, model_type="flow", shift=3.0)
    assert torch.isfinite(hi).all()
    assert hi.abs().max() < 1e3            # bounded, not amplifying to garbage
    assert not torch.allclose(hi, det, atol=1e-4)


def test_uni_pc_anneal_order_ramp_inactive_when_deterministic():
    # The ramp is gated on η > 0, so eta_max=0 must be bit-for-bit deterministic
    # UniPC (bh2) even though the high-order terms (and thus the ramp's target) are
    # exercised by a σ-dependent denoiser. This guards the degradation invariant
    # against the order-ramp change.
    torch.manual_seed(6)
    target = torch.randn(1, 4, 8, 8)
    model = _unipc_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(20, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)
    upc = _last_nonzero_latent(K.sample_uni_pc, model, x_init, sigmas,
                               variant="bh2", model_type="flow", shift=3.0)
    anz = _last_nonzero_latent(K.sample_uni_pc_anneal, model, x_init, sigmas,
                               eta_max=0.0, variant="bh2", model_type="flow", shift=3.0)
    assert torch.allclose(anz, upc, atol=1e-6)


def test_uni_pc_anneal_registered_in_sampler_table():
    assert K.get_sampler("uni_pc_anneal") is K.sample_uni_pc_anneal


# ── STORK-2 ───────────────────────────────────────────────────────────
# Clean-room STORK-2 (arXiv:2505.24210): an s-stage Runge–Kutta–Gegenbauer
# cascade driven by Taylor-extrapolated "virtual" stage velocities (one real
# model evaluation per step). Two structural pins: the cascade must reproduce
# the RKG2 stability polynomial when driven by *true* stage evaluations
# (coefficient correctness), and with taylor_order=1 the super-step must
# collapse to the damped variable-step AB2 closed form (abscissae consistency).


def _gegenbauer_R(s, z):
    """``R_s(z) = a_s + b_s·C_s^{3/2}(1 + w1·z)`` via the standard three-term
    Gegenbauer recurrence — an independent evaluation of the stability
    polynomial the RKG2 cascade is derived from."""
    w1 = 6.0 / ((s + 4.0) * (s - 1.0))
    b_s = 4.0 * (s - 1.0) * (s + 4.0) / (3.0 * s * (s + 1.0) * (s + 2.0) * (s + 3.0))
    a_s = 1.0 - (s + 1.0) * (s + 2.0) / 2.0 * b_s
    w = 1.0 + w1 * z
    Cm2, Cm1 = 1.0, 3.0 * w
    C = Cm1 if s >= 1 else Cm2
    for j in range(2, s + 1):
        C = (2.0 * w * (j + 0.5) * Cm1 - (j + 1.0) * Cm2) / j
        Cm2, Cm1 = Cm1, C
    return a_s + b_s * C


@pytest.mark.parametrize("s", [2, 5, 9, 24])
def test_stork2_stage_cascade_matches_gegenbauer_polynomial(s):
    # Drive the cascade with TRUE stage evaluations v(Y) = z·Y on the scalar
    # linear ODE: the result must equal R_s(z). Every mu/nu/mu~/gamma~ value and
    # the b_0 = 1, b_1 = 1/3 conventions are pinned by this identity, across the
    # whole stability interval [-2/w1, 0].
    w1, c, stage = K._rkg2_coeffs(s)
    for z in (0.0, -0.7, -3.0, -2.0 / w1):
        Y0 = 1.0
        Yjm2, Yjm1 = Y0, Y0 + w1 * z * Y0
        for j, (mu, nu, mut, gat) in enumerate(stage, start=2):
            Yj = mu * Yjm1 + nu * Yjm2 + (1 - mu - nu) * Y0 + mut * z * Yjm1 + gat * z * Y0
            Yjm2, Yjm1 = Yjm1, Yj
        assert abs(Yjm1 - _gegenbauer_R(s, z)) < 1e-12


@pytest.mark.parametrize("s", [5, 9, 24])
def test_stork2_stability_region_scales_quadratically(s):
    # |R_s(z)| ≤ 1 across z ∈ [-2/w1, 0] with 2/w1 = (s+4)(s-1)/3 ~ O(s²): the
    # stabilized-RK property the cascade is built for.
    w1 = 6.0 / ((s + 4.0) * (s - 1.0))
    for k in range(401):
        z = -2.0 / w1 * k / 400.0
        assert abs(_gegenbauer_R(s, z)) <= 1.0 + 1e-9


def test_stork2_taylor1_collapses_to_damped_ab2():
    # With taylor_order=1 the super-step is affine in (v, v̇), so it must equal
    # x + Δσ·v + C1·Δσ²·v̇ with C1 measured off the scalar cascade — and the
    # damping C1 < 1/2 (1/2 being undamped variable-step AB2, i.e. ipndm_v
    # order 2) is the method's signature at this Taylor order.
    s = 9
    w1, c, stage = K._rkg2_coeffs(s)

    def cascade(v0, vp, dt):
        Y0 = 0.0
        Yjm2, Yjm1 = Y0, Y0 + w1 * dt * v0
        for j, (mu, nu, mut, gat) in enumerate(stage, start=2):
            va = v0 + (c[j - 1] * dt) * vp
            Yj = mu * Yjm1 + nu * Yjm2 + (1 - mu - nu) * Y0 + mut * dt * va + gat * dt * v0
            Yjm2, Yjm1 = Yjm1, Yj
        return Yjm1

    dt = -0.37
    assert abs(cascade(1.0, 0.0, dt) / dt - 1.0) < 1e-12       # consistency: A == 1
    C1 = cascade(0.0, 1.0, dt) / dt ** 2
    assert 0.40 < C1 < 0.5                                     # damped vs AB2's 1/2
    # dt-independence of the collapsed coefficients (they are pure cascade sums)
    assert abs(cascade(0.0, 1.0, -0.11) / (-0.11) ** 2 - C1) < 1e-12

    # Full sampler vs the closed form on a σ-dependent model (v = 0.35·x).
    lam = 0.35
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * (lam * x)
    sig = torch.tensor([1.0, 0.8, 0.55, 0.3, 0.12, 0.0])
    torch.manual_seed(0)
    xs = torch.randn(1, 4, 4, 4)
    out = K.sample_stork2(model, xs.clone(), sig, stages=s)

    x = xs.clone()
    s_in = x.new_ones([1])
    prev_sigma, prev_v = None, None
    for i in range(len(sig) - 1):
        den = model(x, sig[i] * s_in)
        d = K.to_d(x, sig[i] * s_in, den)
        dstep = sig[i + 1] - sig[i]
        if float(sig[i + 1]) == 0.0:
            x = den
        elif prev_v is None:
            x = x + d * dstep
        else:
            vp = (d - prev_v) / (float(sig[i]) - prev_sigma)
            x = x + dstep * d + (C1 * dstep * dstep) * vp
        prev_sigma, prev_v = float(sig[i]), d
    assert torch.allclose(out, x, atol=1e-5)


def test_stork2_damping_beats_undamped_ab2_on_smooth_ode():
    # dx/dσ = λ·x has the exact solution x·e^{λΔσ}. The damped derivative
    # correction must beat undamped variable-step AB2 (ipndm_v order 2) on the
    # same schedule, and plain Euler once steps are moderate.
    lam = 0.5
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * (lam * x)
    for steps, beat_euler in ((16, False), (32, True)):
        sigmas = S.karras_schedule(steps, 0.03, 14.6)
        torch.manual_seed(3)
        x_init = torch.randn(2, 4, 4, 4)
        exact = x_init * torch.exp(torch.tensor(-lam * float(sigmas[0])))
        err_stork = (K.sample_stork2(model, x_init.clone(), sigmas) - exact).abs().max()
        err_ab2 = (K.sample_ipndm_v(model, x_init.clone(), sigmas, max_order=2) - exact).abs().max()
        assert torch.isfinite(err_stork)
        assert err_stork < err_ab2
        if beat_euler:
            err_euler = (K.sample_euler(model, x_init.clone(), sigmas) - exact).abs().max()
            assert err_stork < err_euler


def test_stork2_taylor2_lands_clean_both_schedules():
    target = torch.full((1, 4, 4, 4), 0.2)
    for sigmas in (_ve_sigmas(), _flow_sigmas()):
        x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
        out = K.sample_stork2(const_denoiser(target), x_init, sigmas, taylor_order=2)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, target, atol=1e-3)


def test_stork2_deterministic():
    sigmas = _flow_sigmas()
    x_init = torch.randn(1, 4, 4, 4)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    a = K.sample_stork2(model, x_init.clone(), sigmas)
    b = K.sample_stork2(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


def test_stork2_sigma_collision_falls_back_and_stays_finite():
    # A repeated σ (possible at a schedule's σ_min floor) must not divide by ~0;
    # the derivative estimate degrades and the step falls back gracefully.
    sigmas = torch.tensor([1.0, 0.5, 0.5, 0.25, 0.1, 0.0])
    x_init = torch.randn(1, 4, 4, 4)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    out = K.sample_stork2(model, x_init, sigmas)
    assert torch.isfinite(out).all()


def test_stork2_bad_args_raise():
    sigmas = _flow_sigmas()
    x = torch.randn(1, 4, 4, 4)
    with pytest.raises(ValueError):
        K.sample_stork2(const_denoiser(torch.zeros_like(x)), x, sigmas, stages=1)
    with pytest.raises(ValueError):
        K.sample_stork2(const_denoiser(torch.zeros_like(x)), x, sigmas, taylor_order=3)


def test_stork2_registered_in_sampler_table():
    assert K.get_sampler("stork2") is K.sample_stork2


# ── INFINITY ──────────────────────────────────────────────────────────
# Infinity Diffusion (galpt/infinity-diffusion, MIT; verified bit-identical
# to upstream @4f72d8f 2026-07-17): Euler bootstrap, then a velocity EMA
# (α₁=β₁=0.5) plus — from the third step — an acceleration EMA (α₂=β₂=0.3)
# of the derivative differences, gated by three invariants (magnitude clamp
# at 50% of |d|, halving on direction reversal, pure Euler when both fire).
# Fixed constants (upstream tried adaptive coefficients and reverted);
# correction kept through the final step, EMAs never reset.


def test_infinity_constant_derivative_equals_euler():
    # A constant derivative (denoised = x − σ·c) zeroes every difference, so
    # velocity and acceleration stay 0 and the walk is bit-exactly Euler.
    # Power-of-two sigmas, integer x and a dyadic c keep every subtraction
    # exact, so ``x − (x − σ·c)`` recovers σ·c with no rounding residue.
    torch.manual_seed(0)
    x_init = torch.randint(-8, 8, (1, 4, 4, 4)).float()
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * 0.75
    sigmas = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5, 0.0])
    a = K.sample_infinity(model, x_init.clone(), sigmas)
    b = K.sample_euler(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)


def test_infinity_recursion_and_invariants_pinned():
    # Drive the sampler with a scripted derivative sequence (denoised = x − σ·c_i
    # makes d_i == c_i regardless of x) and replay the published recursion with
    # plain floats. With a spatially-constant d the tensor reductions collapse:
    # d.abs().mean() == |c_i| and the cosine test reduces to sign(c_i·c_{i-1}),
    # so the invariant gating is float-replayable too. The sequence is chosen so
    # every gate fires at least once: full correction, magnitude clamp alone,
    # direction-reversal halving alone, and both together (pure Euler step).
    cs = [1.7, 1.5, 0.05, -0.9, 1.2, -0.02]
    sigmas = torch.tensor([1.0, 0.7, 0.45, 0.25, 0.12, 0.05, 0.0])
    torch.manual_seed(2)
    x_init = torch.randn(1, 4, 4, 4)

    it = iter(cs)
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * next(it)
    out = K.sample_infinity(model, x_init.clone(), sigmas)

    a1 = b1 = 0.5
    a2 = b2 = 0.3
    x = x_init.clone()
    vel = acc = 0.0
    d_prev = d_prev2 = None
    hits = set()
    for i, d in enumerate(cs):
        dt = float(sigmas[i + 1] - sigmas[i])
        if d_prev is None:
            x = x + d * dt
            d_prev = d
            continue
        delta = d - d_prev
        vel = (1.0 - a1) * vel + a1 * delta
        if d_prev2 is None:
            raw = b1 * vel
        else:
            acc = (1.0 - a2) * acc + a2 * (delta - (d_prev - d_prev2))
            raw = b1 * vel + b2 * acc
        d_mag = abs(d) + 1e-8
        clamped = abs(raw) > 0.5 * d_mag
        if clamped:
            raw = raw * (0.5 * d_mag / abs(raw))
        reversed_dir = d * d_prev < 0
        if clamped and reversed_dir:
            corr = 0.0
            hits.add("both->euler")
        elif reversed_dir:
            corr = raw * 0.5
            hits.add("reverse-half")
        else:
            corr = raw
            hits.add("clamp-only" if clamped else "full")
        x = x + (d + corr) * dt
        d_prev2 = d_prev
        d_prev = d
    assert hits == {"full", "clamp-only", "reverse-half", "both->euler"}
    assert torch.allclose(out, x, atol=1e-5)


def test_infinity_correction_beats_euler_on_smooth_ode():
    # dx/dσ = λ·x has the exact solution x·e^{λΔσ}. The fixed gains are
    # AB2-consistent only when neighboring steps are comparable, so the win
    # holds on near-uniform grids — a uniform σ ramp, the linear-timestep
    # ``normal`` flow schedule, and its sine-perturbed ``infinity`` variant.
    # On karras(ρ=7) the mis-scaled correction used to land behind Euler;
    # the invariant clamp now bounds that to roughly par (not asserted).
    lam = 0.5
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * (lam * x)
    torch.manual_seed(3)
    x_init = torch.randn(2, 4, 4, 4)
    for sigmas in (S.append_zero(torch.linspace(14.6, 0.03, 32)),
                   S.normal_schedule(S.FlowSamplingView(3.0), 32),
                   S.infinity_schedule(S.FlowSamplingView(3.0), 32)):
        exact = x_init * torch.exp(torch.tensor(-lam * float(sigmas[0])))
        err_inf = (K.sample_infinity(model, x_init.clone(), sigmas) - exact).abs().max()
        err_euler = (K.sample_euler(model, x_init.clone(), sigmas) - exact).abs().max()
        assert torch.isfinite(err_inf)
        assert err_inf < err_euler


def test_infinity_deterministic():
    sigmas = _flow_sigmas()
    torch.manual_seed(1)
    x_init = torch.randn(1, 4, 4, 4)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    a = K.sample_infinity(model, x_init.clone(), sigmas)
    b = K.sample_infinity(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


def test_infinity_registered_in_sampler_table():
    assert K.get_sampler("infinity") is K.sample_infinity


def test_all_registered_samplers_resolve():
    for name, fn in K.SAMPLERS.items():
        assert K.get_sampler(name) is fn
