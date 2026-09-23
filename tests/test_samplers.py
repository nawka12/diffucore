import math

import pytest
import torch

from diffucore.sampling import schedules as S
from diffucore.sampling import samplers as K


def _ve_sigmas():
    return S.karras_schedule(20, 0.03, 14.6)


def _flow_sigmas():
    return S.flow_matching_schedule(16, shift=3.0)


def const_denoiser(target):
    """A denoiser that always predicts ``target``: the ODE then has the exact
    linear solution x(σ) - target = (x_init - target) · (σ / σ_init).
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
    # The final step (σ_next == 0) lands on the constant prediction.
    target = torch.zeros(1, 3, 4, 4)
    sigmas = S.karras_schedule(20, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    g = torch.Generator().manual_seed(0)
    out = K.sample_euler_ancestral(const_denoiser(target), x_init, sigmas, eta=1.0, generator=g)
    assert torch.allclose(out, target, atol=1e-4)
    assert torch.isfinite(out).all()


def test_er_sde_ve_ends_at_clean_sample():
    # The final step (σ_next == 0) lands on the constant prediction.
    target = torch.zeros(1, 3, 4, 4)
    sigmas = S.karras_schedule(20, 0.03, 14.6)
    x_init = torch.randn(1, 3, 4, 4) * sigmas[0]
    g = torch.Generator().manual_seed(0)
    out = K.sample_er_sde(const_denoiser(target), x_init, sigmas, generator=g, model_type="ve")
    assert torch.allclose(out, target, atol=1e-4)
    assert torch.isfinite(out).all()


def test_er_sde_flow_is_finite_and_ends_clean():
    # The first flow sigma is offset off 1.0; the run must stay finite and land.
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
    # Constant x0 makes the ODE linear and the secant slope zero, so SECANT must
    # match Euler and land on x0.
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
    # Full correction (curvature=1) on the flow schedule stays finite and lands.
    target = torch.full((1, 16, 8, 8), 0.0)
    sigmas = S.flow_matching_schedule(20, shift=3.0)
    x_init = torch.randn(1, 16, 8, 8)
    out = K.sample_secant(const_denoiser(target), x_init, sigmas, curvature=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_correction_gated_off_at_high_noise():
    # The correction must be gated off at high σ and ramp in as σ → 0; the
    # σ-dependent denoiser makes the secant slope nonzero.
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
    # A repeated sigma must fall back to Euler instead of dividing by zero.
    target = torch.full((1, 4, 4, 4), 0.2)
    sigmas = torch.tensor([1.0, 0.6, 0.6, 0.3, 0.0])
    x_init = torch.randn(1, 4, 4, 4)
    out = K.sample_secant(const_denoiser(target), x_init, sigmas, curvature=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_secant_s_noise_changes_output_and_is_seed_reproducible():
    # The SDE variant changes the trajectory and is seed-reproducible (checked at
    # the last nonzero σ, since σ_next == 0 lands both runs on the prediction).
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
# With a constant-x0 denoiser the ODE is exactly linear, so every consistent
# solver lands on ``target``; the stochastic ones clean-snap at σ_next == 0.


@pytest.mark.parametrize("name", ["heunpp2", "ipndm", "ipndm_v", "res_multistep",
                                  "lumen", "gradient_estimation", "stork2", "infinity",
                                  "lms", "exp_heun_2_x0", "uni_pc", "uni_pc_bh2"])
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
    # On a flow schedule the RF branch must run a different ancestral step than
    # the VE split (compared at the last nonzero σ).
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
    # DDPM doesn't clean-snap; it contracts toward the prediction.
    target = torch.full((1, 4, 4, 4), 0.1)
    sigmas = S.karras_schedule(30, 0.03, 14.6)
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.sample_ddpm(const_denoiser(target), x_init, sigmas,
                        generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=0.1)


# ── SA-SOLVER ─────────────────────────────────────────────────────────
# SA-Solver (Xue et al., NeurIPS 2023, arXiv:2309.05019): a constant-x0
# denoiser lands every consistent solver on target, and eta=0 is the ODE form.


@pytest.mark.parametrize("model_type", ["ve", "flow"])
def test_sa_solver_deterministic_lands_on_target(model_type):
    target = torch.full((1, 4, 4, 4), 0.2)
    sigmas = _flow_sigmas() if model_type == "flow" else _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.sample_sa_solver(const_denoiser(target), x_init.clone(), sigmas,
                             eta=0.0, model_type=model_type, shift=3.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


@pytest.mark.parametrize("model_type", ["ve", "flow"])
def test_sa_solver_stochastic_lands_clean_and_is_seed_reproducible(model_type):
    # Noise on the middle band, but the final step clean-snaps.
    target = torch.zeros(1, 4, 4, 4)
    sigmas = _flow_sigmas() if model_type == "flow" else _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]

    def run(seed):
        return K.sample_sa_solver(
            const_denoiser(target), x_init.clone(), sigmas,
            generator=torch.Generator().manual_seed(seed),
            model_type=model_type, shift=3.0,
        )

    a, b = run(7), run(7)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)
    assert torch.allclose(a, target, atol=1e-4)


def test_sa_solver_stochastic_differs_from_deterministic_midtrajectory():
    # eta=0 must differ from the stochastic run before the clean-snap.
    sigmas = _flow_sigmas()
    x_init = torch.randn(1, 4, 4, 4)

    def model(x, sigma):  # σ-dependent so the trajectory is not trivially linear
        return 0.3 * torch.tanh(x)

    def run(eta, seed=1):
        last = {}
        K.sample_sa_solver(
            model, x_init.clone(), sigmas, eta=eta,
            generator=torch.Generator().manual_seed(seed), model_type="flow", shift=3.0,
            callback=lambda i, s, x, d: last.__setitem__("x", x.clone()),
        )
        return last["x"]

    assert not torch.allclose(run(0.0), run(1.0), atol=1e-3)


def test_sa_solver_pece_lands_on_target_and_registered():
    target = torch.zeros(1, 4, 4, 4)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.sample_sa_solver_pece(const_denoiser(target), x_init.clone(), sigmas,
                                  generator=torch.Generator().manual_seed(0), model_type="ve")
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)
    assert K.get_sampler("sa_solver_pece") is K.sample_sa_solver_pece


def test_sa_solver_simple_order_2_path_is_finite():
    # The closed-form order-2 coefficients stay finite and land.
    target = torch.full((1, 4, 4, 4), 0.1)
    sigmas = _flow_sigmas()
    x_init = torch.randn(1, 4, 4, 4)
    out = K.sample_sa_solver(const_denoiser(target), x_init.clone(), sigmas,
                             eta=0.0, simple_order_2=True, model_type="flow", shift=3.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_sa_solver_registered_in_sampler_table():
    assert K.get_sampler("sa_solver") is K.sample_sa_solver


# ── SECANT-ANNEAL ─────────────────────────────────────────────────────


def _anneal_sigma_dependent_model(target):
    # σ-dependent x0, so the secant slope is nonzero and the curvature gate runs.
    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)          # x0 settles toward target as σ→0
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)
    return model


def _last_nonzero_latent(fn, model, x_init, sigmas, **kw):
    # x at the last nonzero σ (σ_next == 0 forces every run onto the prediction).
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
    # The final step lands on the constant prediction regardless of burn-in.
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
    # curvature=0 reduces to euler_ancestral_anneal (same seed, same draws);
    # curvature=1 must diverge.
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
    # eta_max=0 reduces to deterministic secant.
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
    # Same seed reproduces; the burn-in must change the trajectory.
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
    # The final step snaps to the constant prediction.
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
    # eta_max=0 is exactly dpmpp_2m_sde with eta=0 (σ-dependent denoiser so the
    # 2nd-order term runs).
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
    # Same seed reproduces; the burn-in must change the trajectory.
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
    # Constant x0 has no truncation error, and both use -log σ in ``ve`` mode, so
    # exp_heun must equal dpmpp_2m exactly.
    target = torch.full((1, 4, 8, 8), 0.15)
    sigmas = _ve_sigmas()
    x_init = torch.randn(1, 4, 8, 8) * sigmas[0]
    heun = K.sample_exp_heun_2_x0(const_denoiser(target), x_init.clone(), sigmas)
    ref = K.sample_dpmpp_2m(const_denoiser(target), x_init.clone(), sigmas)
    assert torch.isfinite(heun).all()
    assert torch.allclose(heun, ref, atol=1e-5)


def test_exp_heun_2_x0_corrector_is_active():
    # On a nonlinear denoiser the phi_2 and phi_1 correctors differ, and both
    # differ from dpmpp_2m.
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
    # The first flow sigma is offset off 1.0; the run must stay finite and land.
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
    # σ-dependent x0 so the higher-order terms run (constant x0 zeroes them).
    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)          # x0 settles toward target as σ→0
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)
    return model


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("variant", ["bh1", "bh2"])
@pytest.mark.parametrize("model_type,sigmas_fn", [("ve", _ve_sigmas), ("flow", _flow_sigmas)])
def test_uni_pc_constant_x0_lands_clean(order, variant, model_type, sigmas_fn):
    # Constant x0: every order/variant stays finite and lands.
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
    # Order 3 differs from order 1, bh1 from bh2, and UniPC from dpmpp_2m.
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
    # The first flow sigma is offset off 1.0; the run must stay finite and land.
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
    # The final step snaps to the constant prediction.
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
    # eta_max=0 is deterministic UniPC (bh2) bit-for-bit, with the higher-order
    # residual exercised.
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
    # Same seed reproduces; the burn-in must change the trajectory.
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
    # The order ramp keeps eta_max=1.0 on a σ-dependent denoiser bounded, and the
    # burn-in must change the trajectory.
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
    # The ramp is gated on η > 0, so eta_max=0 stays bit-for-bit UniPC (bh2).
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


# ── COGENT ────────────────────────────────────────────────────────────
# Coherence-gated exponential multistep; see sample_cogent / _coherence_gate.


_TINY_H = torch.tensor(1e-6)      # floor ≈ 0, isolating the coherence term


def test_coherence_gate_maps_rho_to_wiener_factor():
    # psi = (1 + 2·rho)/3: rho = 1 → 1, rho = 0 → 1/3, rho ≤ -1/2 → 0.
    a = torch.randn(1, 4, 8, 8)
    b = torch.randn(1, 4, 8, 8)
    b = b - (a * b).sum() / (a * a).sum() * a          # orthogonalise b against a
    assert torch.allclose(K._coherence_gate(a, a, _TINY_H), torch.ones(1, 1, 1, 1), atol=1e-5)
    assert torch.allclose(K._coherence_gate(a, b, _TINY_H),
                          torch.full((1, 1, 1, 1), 1 / 3), atol=1e-4)
    assert torch.allclose(K._coherence_gate(a, -a, _TINY_H), torch.zeros(1, 1, 1, 1), atol=1e-5)


def test_coherence_gate_floor_is_the_phi_weight():
    # No second difference gives the floor alone, which also beats psi = 0.
    # h = ln 2 → 1 - e^-h = 1/2.
    a = torch.randn(1, 4, 8, 8)
    h = torch.log(torch.tensor(2.0))
    assert torch.allclose(K._coherence_gate(a, None, h), torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(K._coherence_gate(a, -a, h), torch.full((1, 1, 1, 1), 0.5), atol=1e-6)
    # a fine step lets the coherence term damp all the way down
    assert float(K._coherence_gate(a, -a, _TINY_H)) < 1e-5


def test_coherence_gate_is_per_sample_and_scale_invariant():
    # Per-sample cosine, so positive rescaling leaves it unchanged.
    a = torch.randn(3, 4, 8, 8)
    b = torch.randn(3, 4, 8, 8)
    psi = K._coherence_gate(a, b, _TINY_H)
    assert psi.shape == (3, 1, 1, 1)
    assert torch.allclose(psi, K._coherence_gate(5.0 * a, 0.1 * b, _TINY_H), atol=1e-5)
    # per-sample: perturbing sample 0 must not move sample 1's gate
    a2 = a.clone()
    a2[0] = torch.randn(4, 8, 8)
    assert torch.allclose(psi[1:], K._coherence_gate(a2, b, _TINY_H)[1:], atol=1e-5)


def test_cogent_gate_of_one_equals_dpmpp_2m_anneal(monkeypatch):
    # psi ≡ 1 is exactly the DPM++(2M) annealed multistep: the gate is the only
    # thing cogent adds.
    torch.manual_seed(5)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    monkeypatch.setattr(K, "_coherence_gate", lambda d, od, h, **kw: torch.ones(
        d.shape[0], *([1] * (d.ndim - 1))))
    got = _last_nonzero_latent(K.sample_cogent, model, x_init, sigmas,
                               eta_max=0.0, model_type="flow", shift=3.0)
    want = _last_nonzero_latent(K.sample_dpmpp_2m_anneal, model, x_init, sigmas,
                                eta_max=0.0, model_type="flow", shift=3.0)
    assert torch.isfinite(got).all()
    assert torch.equal(got, want)          # bit-for-bit, not merely close


def test_cogent_step_size_floor_keeps_the_correction_alive(monkeypatch):
    # With the Wiener term at 0, psi is the step-size floor, which must be
    # nonzero; cogent must not match a first-order step.
    torch.manual_seed(11)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(8, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    real_gate = K._coherence_gate
    monkeypatch.setattr(K, "_coherence_gate",
                        lambda d, od, h, **kw: real_gate(d, -d, h))   # rho = -1 ⇒ floor only
    got = _last_nonzero_latent(K.sample_cogent, model, x_init, sigmas,
                               eta_max=0.0, model_type="flow", shift=3.0)

    # First-order (exponential-Euler) reference: the same loop, no 2nd-order term.
    sig = K._offset_first_sigma_for_snr(sigmas, "flow", 3.0)
    x, s_in, last = x_init.clone(), x_init.new_ones([1]), None
    for i in range(len(sig) - 1):
        denoised = model(x, sig[i] * s_in)
        if bool(sig[i + 1] == 0):
            break
        last = x.clone()
        lam_s, lam_t = K._half_log_snr(sig[i], "flow"), K._half_log_snr(sig[i + 1], "flow")
        h = lam_t - lam_s
        x = sig[i + 1] / sig[i] * x + (sig[i + 1] * lam_t.exp()) * (-h).expm1().neg() * denoised
    assert torch.isfinite(got).all()
    assert not torch.allclose(got, last, atol=1e-4)

    # the floor is exactly the integrator's phi-weight: in (0, 1) for every h > 0
    lam = K._half_log_snr(sig[:-1], "flow")
    hs = lam[1:] - lam[:-1]
    floor = (-hs).expm1().neg()
    assert bool((hs > 0).all())
    assert bool(((floor > 0) & (floor < 1)).all())


def test_cogent_constant_x0_ends_clean():
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    out = K.sample_cogent(
        const_denoiser(target), torch.randn(1, 16, 4, 4), sigmas,
        model_type="flow", shift=3.0, generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_cogent_ve_finite_and_lands_clean():
    # Family-agnostic: the VE map (SD/SDXL) anneals on σ/(1+σ) instead of σ.
    target = torch.full((1, 4, 8, 8), -0.2)
    out = K.sample_cogent(const_denoiser(target), torch.randn(1, 4, 8, 8) * 14.6,
                          _ve_sigmas(), generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_cogent_seed_reproducible_and_stochastic():
    torch.manual_seed(4)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    def run(**kw):
        return _last_nonzero_latent(K.sample_cogent, model, x_init, sigmas,
                                    model_type="flow", shift=3.0, **kw)

    a = run(generator=torch.Generator().manual_seed(3))
    b = run(generator=torch.Generator().manual_seed(3))
    det = run(eta_max=0.0)
    det2 = run(eta_max=0.0, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    assert not torch.allclose(a, det, atol=1e-5)
    assert torch.equal(det, det2)          # eta_max=0 ⇒ no noise drawn at all


def test_cogent_registered_in_sampler_table():
    assert K.get_sampler("cogent") is K.sample_cogent


# ── COGENT3 ───────────────────────────────────────────────────────────
# cogent3: the DPM-Solver++(3M) flow core (dpmpp_3m_sde, eta=0, when both gates
# are identity) with psi_1 on the 2nd-order term and psi_2 = (2 + 3·rho2)/5 on
# the 3rd (see _cogent3_curvature_gate).


def _curv_gate_ones_or_zero(value):
    return lambda d, od, **kw: torch.full((d.shape[0], *([1] * (d.ndim - 1))), value)


def test_cogent3_curvature_gate_maps_rho_to_wiener_factor():
    # psi_2 = (2 + 3·rho2)/5: rho2 = 1 → 1, 0 → 0.4, -2/3 → 0; no history → ones.
    a = torch.randn(1, 4, 8, 8)
    b = torch.randn(1, 4, 8, 8)
    b = b - (a * b).sum() / (a * a).sum() * a           # orthogonalise b against a
    assert torch.allclose(K._cogent3_curvature_gate(a, a), torch.ones(1, 1, 1, 1), atol=1e-5)
    assert torch.allclose(K._cogent3_curvature_gate(a, b),
                          torch.full((1, 1, 1, 1), 0.4), atol=1e-4)
    # rho2 = -2/3 ⇒ psi_2 = (2 + 3·(-2/3))/5 = 0 (pure noise reading)
    na = torch.randn(1, 4, 8, 8)
    assert torch.allclose(K._cogent3_curvature_gate(na, -na), torch.zeros(1, 1, 1, 1), atol=1e-5)
    assert torch.allclose(K._cogent3_curvature_gate(a, None), torch.ones(1, 1, 1, 1), atol=1e-5)


def test_cogent3_curvature_gate_is_per_sample_and_scale_invariant():
    # Per-sample cosine: rescaling leaves it unchanged, and sample 0 can't move
    # sample 1's gate.
    a = torch.randn(3, 4, 8, 8)
    b = torch.randn(3, 4, 8, 8)
    psi = K._cogent3_curvature_gate(a, b)
    assert psi.shape == (3, 1, 1, 1)
    assert torch.allclose(psi, K._cogent3_curvature_gate(5.0 * a, 0.1 * b), atol=1e-5)
    a2 = a.clone()
    a2[0] = torch.randn(4, 8, 8)
    assert torch.allclose(psi[1:], K._cogent3_curvature_gate(a2, b)[1:], atol=1e-5)


def test_cogent3_gate_of_one_equals_dpmpp_3m_sde_deterministic(monkeypatch):
    # Both gates at identity and eta_max=0 give the deterministic DPM-Solver++(3M).
    torch.manual_seed(5)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    monkeypatch.setattr(K, "_coherence_gate",
                        lambda d, od, h, **kw: torch.ones(d.shape[0], *([1] * (d.ndim - 1))))
    monkeypatch.setattr(K, "_cogent3_curvature_gate", _curv_gate_ones_or_zero(1.0))
    got = _last_nonzero_latent(K.sample_cogent3, model, x_init, sigmas,
                               eta_max=0.0, model_type="flow", shift=3.0)
    want = _last_nonzero_latent(K.sample_dpmpp_3m_sde, model, x_init, sigmas,
                                eta=0.0, model_type="flow", shift=3.0)
    assert torch.isfinite(got).all()
    assert torch.equal(got, want)            # bit-for-bit, not merely close


def test_cogent3_third_order_term_actually_fires(monkeypatch):
    # The live 3rd-order term must differ from the curvature gate pinned to zero.
    torch.manual_seed(11)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(24, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    real = K._cogent3_curvature_gate
    monkeypatch.setattr(K, "_cogent3_curvature_gate", _curv_gate_ones_or_zero(0.0))
    fallback = _last_nonzero_latent(K.sample_cogent3, model, x_init, sigmas,
                                    eta_max=0.0, model_type="flow", shift=3.0)
    monkeypatch.setattr(K, "_cogent3_curvature_gate", real)
    live = _last_nonzero_latent(K.sample_cogent3, model, x_init, sigmas,
                                eta_max=0.0, model_type="flow", shift=3.0)
    assert torch.isfinite(live).all()
    assert torch.isfinite(fallback).all()
    assert not torch.allclose(live, fallback, atol=1e-6)   # 3rd-order term is live


def test_cogent3_constant_x0_ends_clean():
    target = torch.full((1, 16, 4, 4), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    out = K.sample_cogent3(
        const_denoiser(target), torch.randn(1, 16, 4, 4), sigmas,
        model_type="flow", shift=3.0, generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_cogent3_ve_finite_and_lands_clean():
    # Family-agnostic: the VE map (SD/SDXL) anneals on σ/(1+σ) instead of σ.
    target = torch.full((1, 4, 8, 8), -0.2)
    out = K.sample_cogent3(const_denoiser(target), torch.randn(1, 4, 8, 8) * 14.6,
                           _ve_sigmas(), generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_cogent3_seed_reproducible_and_stochastic():
    torch.manual_seed(4)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    def run(**kw):
        return _last_nonzero_latent(K.sample_cogent3, model, x_init, sigmas,
                                    model_type="flow", shift=3.0, **kw)

    a = run(generator=torch.Generator().manual_seed(3))
    b = run(generator=torch.Generator().manual_seed(3))
    det = run(eta_max=0.0)
    det2 = run(eta_max=0.0, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    assert not torch.allclose(a, det, atol=1e-5)
    assert torch.equal(det, det2)            # eta_max=0 ⇒ no noise drawn at all


def test_cogent3_registered_in_sampler_table():
    assert K.get_sampler("cogent3") is K.sample_cogent3


# ── cogent3's coherence pump (the `cogent3_pump` sampler) ─────────────
# aether's high-σ structure pump with a hard low-σ shutoff (see sample_cogent3).

def _pump_run(x_init, sigmas, model, **kw):
    return _last_nonzero_latent(K.sample_cogent3, model, x_init, sigmas,
                                model_type="flow", shift=3.0, **kw)


def test_pump_off_is_bit_exact_cogent3():
    # pump_strength=0 must not change the numerics or consume generator noise.
    torch.manual_seed(7)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    base = _pump_run(x_init, sigmas, model, generator=torch.Generator().manual_seed(2))
    off = _pump_run(x_init, sigmas, model, pump_strength=0.0,
                    generator=torch.Generator().manual_seed(2))
    assert torch.equal(base, off)            # bit-for-bit, generator stream included


def test_pump_fires_and_is_seed_reproducible():
    torch.manual_seed(8)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    a = _pump_run(x_init, sigmas, model, pump_strength=0.08,
                  generator=torch.Generator().manual_seed(3))
    b = _pump_run(x_init, sigmas, model, pump_strength=0.08,
                  generator=torch.Generator().manual_seed(3))
    off = _pump_run(x_init, sigmas, model, generator=torch.Generator().manual_seed(3))
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)                             # reproducible
    assert not torch.allclose(a, off, atol=1e-5)         # the pump actually moves x


def test_pump_shuts_off_below_pump_end():
    # pump_end above σ_max: no noise is drawn and the run equals plain cogent3.
    torch.manual_seed(9)
    target = torch.randn(1, 4, 8, 8)
    model = _anneal_sigma_dependent_model(target)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)

    base = _pump_run(x_init, sigmas, model, generator=torch.Generator().manual_seed(4))
    shut = _pump_run(x_init, sigmas, model, pump_strength=0.08, pump_end=1.5,
                     generator=torch.Generator().manual_seed(4))
    assert torch.equal(base, shut)


def test_pump_never_touches_the_final_latent():
    # The pump lives in the sigma_next != 0 branch, so a constant-x0 model still
    # lands exactly on target however hard it is pumped.
    target = torch.full((1, 4, 8, 8), 0.1)
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    out = K.sample_cogent3(
        const_denoiser(target), torch.randn(1, 4, 8, 8), sigmas,
        model_type="flow", shift=3.0, pump_strength=0.5, pump_end=0.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, target, atol=1e-4)


def test_pump_requires_4d_latent_only_when_on():
    # The structure tensor needs a 4-D latent, but only while the pump runs.
    target = torch.full((1, 64, 16), 0.1)
    x_init = torch.randn(1, 64, 16)
    sigmas = S.flow_matching_schedule(8, shift=3.0)
    out = K.sample_cogent3(const_denoiser(target), x_init, sigmas,
                           model_type="flow", shift=3.0,
                           generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out).all()         # plain cogent3 is fine on 3-D
    with pytest.raises(ValueError, match="4-D"):
        K.sample_cogent3(const_denoiser(target), x_init, sigmas,
                         model_type="flow", shift=3.0, pump_strength=0.08,
                         generator=torch.Generator().manual_seed(0))


def test_cogent3_pump_registered_in_sampler_table():
    fn = K.get_sampler("cogent3_pump")
    assert fn.func is K.sample_cogent3
    assert fn.keywords["pump_strength"] > 0


def _pump_increment(sigmas, **kw):
    """The pump's contribution to one run: with eta_max=0 the pump is the only
    noise drawn, so (pumped − unpumped) under one generator seed isolates it."""
    target = torch.full((1, 4, 8, 8), 0.1)
    x_init = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(5))
    run = lambda **k: K.sample_cogent3(
        const_denoiser(target), x_init.clone(), sigmas, model_type="flow", shift=1.0,
        eta_max=0.0, pump_end=0.0, generator=torch.Generator().manual_seed(6), **k)
    return run(pump_strength=0.08, **kw) - run()


def test_pump_rate_law_scales_injection_by_step_size():
    # eta_max=0 → V(h) = h, so each injection scales by sqrt(h / h_ref).
    h = 0.2
    lam0 = math.log(0.1 / 0.9)
    sigmas = torch.tensor([1.0 / (1.0 + math.exp(lam0 + k * h)) for k in range(3)])
    fixed = _pump_increment(sigmas)
    assert fixed.abs().max() > 1e-3                          # the pump fired
    same = _pump_increment(sigmas, pump_h_ref=h)
    quarter = _pump_increment(sigmas, pump_h_ref=h / 4)
    assert torch.allclose(same, fixed, rtol=1e-4, atol=1e-7)          # h == h_ref ⇒ plain amplitude
    assert torch.allclose(quarter, 2 * fixed, rtol=1e-4, atol=1e-7)   # h = 4·h_ref ⇒ 2×


def test_ou_step_var_limits():
    assert K._ou_step_var(0.3, 0.0) == 0.3                   # Brownian when eta → 0
    assert abs(K._ou_step_var(0.3, 1e-10) - 0.3) < 1e-9
    assert abs(K._ou_step_var(50.0, 1.0) - 0.5) < 1e-12      # saturates at 1/(2·eta)
    v = [K._ou_step_var(h, 0.9) for h in (0.1, 0.2, 0.4)]
    assert v[0] < v[1] < v[2] and v[2] < 2 * v[1]            # monotone, sub-linear


def test_cogent3_pump_rate_registered_in_sampler_table():
    fn = K.get_sampler("cogent3_pump_rate")
    assert fn.func is K.sample_cogent3
    assert fn.keywords["pump_strength"] == K.get_sampler("cogent3_pump").keywords["pump_strength"]
    # calibrated to pump_dual@50's band step, so the two agree there
    sig = S.flow_table_schedule("pump_dual", 3.0, 50)
    lam = [math.log((1 - float(v)) / float(v)) for v in sig[1:30]]
    assert abs((lam[1] - lam[0]) - fn.keywords["pump_h_ref"]) < 1e-5


# ── sampler allowlist consistency ─────────────────────────────────────
# Pipelines gate on a family allowlist and separately on a flow-aware set;
# drifting apart fails only at generation time (how `cogent` once shipped broken).


@pytest.mark.parametrize("module", ["_anima", "_flux"])
def test_pipeline_sampler_allowlists_are_consistent(module):
    import importlib
    mod = importlib.import_module(f"diffucore.pipelines.{module}")
    allow = mod._ANIMA_SAMPLERS if module == "_anima" else mod._FLUX_SAMPLERS
    # every allowlisted name must actually resolve to a sampler
    assert not (allow - set(K.SAMPLERS)), f"{module}: not in SAMPLERS registry"
    # anything the pipeline special-cases as flow-aware must be runnable there
    assert not (mod._FLOW_AWARE_SAMPLERS - allow), \
        f"{module}: flow-aware sampler missing from the family allowlist"


# ── STORK-2 ───────────────────────────────────────────────────────────
# STORK-2 (arXiv:2505.24210): the cascade must reproduce the RKG2 stability
# polynomial with true stage evaluations, and with taylor_order=1 collapse to
# the damped variable-step AB2.


def _gegenbauer_R(s, z):
    """``R_s(z) = a_s + b_s·C_s^{3/2}(1 + w1·z)`` by the Gegenbauer recurrence,
    an independent evaluation of the stability polynomial."""
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
    # True stage evaluations v(Y) = z·Y must give R_s(z) across [-2/w1, 0].
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
    # |R_s(z)| ≤ 1 on [-2/w1, 0], with 2/w1 = (s+4)(s-1)/3 ~ O(s²).
    w1 = 6.0 / ((s + 4.0) * (s - 1.0))
    for k in range(401):
        z = -2.0 / w1 * k / 400.0
        assert abs(_gegenbauer_R(s, z)) <= 1.0 + 1e-9


def test_stork2_taylor1_collapses_to_damped_ab2():
    # taylor_order=1 must equal x + Δσ·v + C1·Δσ²·v̇, with the damping C1 < 1/2.
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
    # dx/dσ = λ·x: the damped correction beats undamped AB2 and, at moderate
    # steps, Euler.
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
    # A repeated σ must not divide by ~0; the estimate degrades gracefully.
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
# Infinity Diffusion (galpt/infinity-diffusion, MIT; bit-identical to upstream
# @4f72d8f): Euler plus velocity/acceleration EMAs gated by three invariants.


def test_infinity_constant_derivative_equals_euler():
    # A constant derivative zeroes every difference, so the walk is Euler exactly
    # (dyadic values keep every subtraction exact).
    torch.manual_seed(0)
    x_init = torch.randint(-8, 8, (1, 4, 4, 4)).float()
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * 0.75
    sigmas = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5, 0.0])
    a = K.sample_infinity(model, x_init.clone(), sigmas)
    b = K.sample_euler(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)


def _infinity_float_replay(cs, sigmas, x_init, *, dt_scaled):
    """Replay the infinity recursion in floats (a spatially constant ``d`` makes
    the reductions scalar). ``dt_scaled`` picks our step-size-aware filter or
    upstream's fixed gains."""
    a1 = b1 = 0.5
    a2 = b2 = 0.3
    x = x_init.clone()
    vel = acc = 0.0
    d_prev = dd_prev = dt_prev = None
    hits = set()
    for i, d in enumerate(cs):
        dt = float(sigmas[i + 1] - sigmas[i])
        if d_prev is None:
            x = x + d * dt
            d_prev, dt_prev = d, dt
            continue
        if dt_scaled:
            dd = (d - d_prev) / dt_prev
            vel = (1.0 - a1) * vel + a1 * dd
            if dd_prev is None:
                raw = b1 * dt * vel
            else:
                acc = (1.0 - a2) * acc + a2 * ((dd - dd_prev) / dt_prev)
                raw = b1 * dt * vel + b2 * dt * dt * acc
        else:
            dd = d - d_prev
            vel = (1.0 - a1) * vel + a1 * dd
            if dd_prev is None:
                raw = b1 * vel
            else:
                acc = (1.0 - a2) * acc + a2 * (dd - dd_prev)
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
        dd_prev, d_prev, dt_prev = dd, d, dt
    return x, hits


# denoised = x − σ·c_i makes d_i == c_i; chosen so every gate combination fires.
_PINNED_CS = [1.7, 1.5, 0.05, -0.9, 1.2, -0.02]
_ALL_GATES = {"full", "clamp-only", "reverse-half", "both->euler"}


def test_infinity_recursion_and_invariants_pinned():
    sigmas = torch.tensor([1.0, 0.7, 0.45, 0.25, 0.12, 0.05, 0.0])
    torch.manual_seed(2)
    x_init = torch.randn(1, 4, 4, 4)
    it = iter(_PINNED_CS)
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * next(it)
    out = K.sample_infinity(model, x_init.clone(), sigmas)
    x, hits = _infinity_float_replay(_PINNED_CS, sigmas, x_init, dt_scaled=True)
    assert hits == _ALL_GATES
    assert torch.allclose(out, x, atol=1e-5)


def test_infinity_uniform_grid_reduces_to_upstream_fixed_gains():
    # On a uniform grid the dt factors cancel, so we must reproduce upstream's
    # fixed-gain recursion.
    sigmas = torch.tensor([1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    torch.manual_seed(2)
    x_init = torch.randn(1, 4, 4, 4)
    it = iter(_PINNED_CS)
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * next(it)
    out = K.sample_infinity(model, x_init.clone(), sigmas)
    upstream, hits = _infinity_float_replay(_PINNED_CS, sigmas, x_init, dt_scaled=False)
    assert hits == _ALL_GATES
    assert torch.allclose(out, upstream, atol=1e-6)


def test_infinity_step_scaling_helps_on_nonuniform_grids():
    # On karras(ρ=7), where the unscaled form fell behind Euler, the scaled
    # filter must beat it.
    lam = 0.5
    model = lambda x, sg: x - sg.view(-1, 1, 1, 1) * (lam * x)
    torch.manual_seed(3)
    x_init = torch.randn(2, 4, 4, 4)
    for sigmas in (S.karras_schedule(32, 0.03, 14.6),
                   S.normal_schedule(S.FlowSamplingView(3.0), 32),
                   S.infinity_schedule(S.FlowSamplingView(3.0), 32)):
        exact = x_init * torch.exp(torch.tensor(-lam * float(sigmas[0])))
        err_inf = (K.sample_infinity(model, x_init.clone(), sigmas) - exact).abs().max()
        err_euler = (K.sample_euler(model, x_init.clone(), sigmas) - exact).abs().max()
        assert torch.isfinite(err_inf)
        assert err_inf < err_euler


def test_infinity_correction_beats_euler_on_smooth_ode():
    # dx/dσ = λ·x: the correction beats Euler on near-uniform grids.
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


# ── INFINITY OMEGA ────────────────────────────────────────────────────
# galpt/infinity-diffusion `omega` @8d81e76: Euler plus a 3-band pyramid, NQVP
# (SD/SDXL only) and AVN (all families). Not a consistent integrator on purpose.


def test_infinity_omega_low_step_count_is_exactly_euler():
    # ≤6 steps bypasses the filter: bit-identical to euler.
    torch.manual_seed(0)
    x_init = torch.randn(1, 4, 8, 8)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    sigmas = S.flow_matching_schedule(6, shift=3.0)
    assert torch.equal(K.sample_infinity_omega(model, x_init.clone(), sigmas),
                       K.sample_euler(model, x_init.clone(), sigmas))


def test_infinity_omega_rejects_non_4d_latents():
    # FLUX's token-sequence latents are rejected up front, not inside conv2d.
    model = lambda x, sg: 0.3 * torch.tanh(x)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    with pytest.raises(ValueError, match="4-D"):
        K.sample_infinity_omega(model, torch.randn(1, 256, 64), sigmas)


def test_infinity_omega_deterministic_and_finite():
    sigmas = S.flow_matching_schedule(16, shift=3.0)
    torch.manual_seed(1)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    a = K.sample_infinity_omega(model, x_init.clone(), sigmas)
    b = K.sample_infinity_omega(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


def test_infinity_omega_runs_in_fp16():
    # torch.quantile rejects fp16, so NQVP casts to float32.
    sigmas = S.flow_matching_schedule(16, shift=3.0).half()
    torch.manual_seed(1)
    x_init = torch.randn(1, 4, 16, 16, dtype=torch.float16)
    model = lambda x, sg: (0.3 * torch.tanh(x.float())).half()
    out = K.sample_infinity_omega(model, x_init, sigmas)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_infinity_omega_gaussian_blur_preserves_dc_in_the_interior():
    # Normalized kernels keep a constant field intact away from the zero-padded
    # 2px rim.
    x = torch.full((2, 3, 12, 12), 0.7)
    for k, s in ((3, 1.0), (5, 2.0), (3, 0.5), (5, 1.0)):
        out = K._gaussian_blur2d(x, k, s)
        assert torch.allclose(out[:, :, 3:-3, 3:-3], x[:, :, 3:-3, 3:-3], atol=1e-6)


def test_infinity_omega_nqvp_pulls_spread_into_published_band():
    # NQVP saturates at exactly clamp(ema/q95, 0.88, 1.12).
    torch.manual_seed(7)
    base = torch.randn(1, 4, 16, 16)
    _, ema = K._quantile_variance_preserve(base, None, 20)
    wide, _ = K._quantile_variance_preserve(base * 8.0, ema, 20)
    narrow, _ = K._quantile_variance_preserve(base * 0.1, ema, 20)
    # ratio vs the uncorrected input, measured on the centered field
    def spread(y):
        return (y - y.mean(dim=(2, 3), keepdim=True)).abs().flatten(2).quantile(0.95, dim=2)
    assert torch.allclose(spread(wide) / spread(base * 8.0), torch.tensor(0.88), atol=1e-3)
    assert torch.allclose(spread(narrow) / spread(base * 0.1), torch.tensor(1.12), atol=1e-3)


@pytest.mark.parametrize("floor", [0.70, 0.85])
def test_infinity_omega_avn_only_damps_and_saturates_at_its_floor(floor):
    # AVN never re-inflates a channel (1.0 ceiling) and saturates at its floor.
    torch.manual_seed(8)
    base = torch.randn(1, 4, 16, 16)
    _, ema = K._adaptive_velocity_normalize(base, None, 20, floor)
    wide, _ = K._adaptive_velocity_normalize(base * 8.0, ema, 20, floor)
    narrow, _ = K._adaptive_velocity_normalize(base * 0.1, ema, 20, floor)
    std = lambda y: y.std(dim=(2, 3))
    assert torch.allclose(std(wide) / std(base * 8.0), torch.tensor(floor), atol=1e-3)
    assert torch.allclose(narrow, base * 0.1, atol=1e-6)     # ceiling: no gain


def test_infinity_omega_avn_leaves_the_channel_mean_alone():
    # AVN touches the spread only, so a channel's mean survives exactly.
    torch.manual_seed(9)
    base = torch.randn(1, 4, 16, 16)
    _, ema = K._adaptive_velocity_normalize(base, None, 20, 0.70)
    shifted = base * 8.0 + 5.0                 # spread AND mean far off
    out, _ = K._adaptive_velocity_normalize(shifted, ema, 20, 0.70)
    assert torch.allclose(out.mean(dim=(2, 3)), shifted.mean(dim=(2, 3)), atol=1e-5)


def test_infinity_omega_filter_is_live_above_the_bypass():
    # Above 6 steps the filter must change the trajectory.
    sigmas = S.flow_matching_schedule(20, shift=3.0)
    torch.manual_seed(4)
    x_init = torch.randn(1, 4, 32, 32)
    model = lambda x, sg: 0.3 * torch.tanh(x) + 0.1 * x
    omega = K.sample_infinity_omega(model, x_init.clone(), sigmas)
    euler = K.sample_euler(model, x_init.clone(), sigmas)
    assert torch.isfinite(omega).all()
    assert not torch.allclose(omega, euler, atol=1e-4)


@pytest.mark.parametrize("sigmas_fn,nqvp_runs,avn_floor", [
    (lambda: S.flow_matching_schedule(20, shift=3.0), False, 0.70),
    (lambda: S.karras_schedule(20, 0.0292, 14.6146), True, 0.85),
])
def test_infinity_omega_gate_routes_nqvp_and_the_avn_floor_per_family(
        monkeypatch, sigmas_fn, nqvp_runs, avn_floor):
    # `is_flow = sigma_max < 5` both skips NQVP on flow and picks the AVN floor;
    # count the calls.
    sigmas = sigmas_fn()
    calls = {"nqvp": 0, "avn": []}
    real_nqvp, real_avn = K._quantile_variance_preserve, K._adaptive_velocity_normalize

    def spy_nqvp(*a, **kw):
        calls["nqvp"] += 1
        return real_nqvp(*a, **kw)

    def spy_avn(v, ema, steps, floor):
        calls["avn"].append(floor)
        return real_avn(v, ema, steps, floor)

    monkeypatch.setattr(K, "_quantile_variance_preserve", spy_nqvp)
    monkeypatch.setattr(K, "_adaptive_velocity_normalize", spy_avn)

    torch.manual_seed(5)
    model = lambda x, sg: 0.2 * torch.tanh(x)
    K.sample_infinity_omega(model, torch.randn(1, 4, 16, 16), sigmas)

    assert calls["nqvp"] == (20 if nqvp_runs else 0)
    assert calls["avn"] == [avn_floor] * 20


def test_infinity_omega_avn_is_live_on_rectified_flow():
    # On flow, where NQVP is off, AVN must still change the result.
    sigmas = S.flow_matching_schedule(20, shift=3.0)
    assert float(sigmas[0]) == 1.0
    torch.manual_seed(5)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.2 * torch.tanh(x) + 0.1 * x

    omega = K.sample_infinity_omega(model, x_init.clone(), sigmas)
    no_avn = K._sample_infinity_pyramid(
        model, x_init.clone(), sigmas, name="ref",
        nqvp_sigma_min=K._NQVP_SIGMA_MIN_OMEGA, avn=False, dog=True)
    assert torch.isfinite(omega).all()
    assert not torch.allclose(omega, no_avn, atol=1e-5)


def test_infinity_omega_registered_in_sampler_table():
    assert K.get_sampler("infinity_omega") is K.sample_infinity_omega


# ── INFINITY NANO ─────────────────────────────────────────────────────
# galpt/infinity-diffusion `nano` @355b792: omega without AVN and DoG, on the
# older `sigmas[0] < 8` NQVP gate.


def test_infinity_nano_is_omega_without_avn_and_dog():
    # nano must equal omega with AVN and DoG skipped, and differ from omega.
    sigmas = S.karras_schedule(16, 0.0292, 14.6146)
    torch.manual_seed(3)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.3 * torch.tanh(x) + 0.1 * x
    nano = K.sample_infinity_nano(model, x_init.clone(), sigmas)
    ref = K._sample_infinity_pyramid(model, x_init.clone(), sigmas, name="ref",
                                     nqvp_sigma_min=K._NQVP_SIGMA_MIN_NANO,
                                     avn=False, dog=False)
    assert torch.equal(nano, ref)
    assert not torch.allclose(nano, K.sample_infinity_omega(model, x_init.clone(), sigmas),
                              atol=1e-5)


def test_infinity_nano_and_omega_now_differ_on_flow_too():
    # AVN is live on flow in omega only, so they differ by more than DoG.
    sigmas = S.flow_matching_schedule(20, shift=3.0)
    torch.manual_seed(11)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.2 * torch.tanh(x) + 0.1 * x

    nano = K.sample_infinity_nano(model, x_init.clone(), sigmas)
    omega = K.sample_infinity_omega(model, x_init.clone(), sigmas)
    # ... and by more than the DoG term alone accounts for.
    dog_only = K._sample_infinity_pyramid(model, x_init.clone(), sigmas, name="ref",
                                          nqvp_sigma_min=K._NQVP_SIGMA_MIN_NANO,
                                          avn=False, dog=True)
    assert (omega - nano).abs().max() > 5.0 * (dog_only - nano).abs().max()


def test_infinity_nano_keeps_the_older_nqvp_gate_constant():
    # The NQVP gates (8.0 vs 5.0) only disagree between those sigmas.
    assert K._NQVP_SIGMA_MIN_NANO == 8.0 and K._NQVP_SIGMA_MIN_OMEGA == 5.0
    sigmas = S.karras_schedule(20, 0.0292, 14.6146)
    sigmas = sigmas[sigmas < 7.0]                      # sigma_max now in [5, 8)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)]) if sigmas[-1] != 0 else sigmas
    assert 5.0 <= float(sigmas[0]) < 8.0
    calls = []
    real = K._quantile_variance_preserve
    for fn, tag in ((K.sample_infinity_nano, "nano"), (K.sample_infinity_omega, "omega")):
        n = [0]

        def spy(*a, _n=n, **kw):
            _n[0] += 1
            return real(*a, **kw)

        K._quantile_variance_preserve = spy
        try:
            torch.manual_seed(3)
            fn(lambda x, sg: 0.2 * torch.tanh(x), torch.randn(1, 4, 16, 16), sigmas)
        finally:
            K._quantile_variance_preserve = real
        calls.append((tag, n[0]))
    assert calls[0][1] == 0 and calls[1][1] > 0        # nano gate shut, omega's open


def test_infinity_nano_low_step_count_is_exactly_euler():
    torch.manual_seed(0)
    x_init = torch.randn(1, 4, 8, 8)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    sigmas = S.flow_matching_schedule(6, shift=3.0)
    assert torch.equal(K.sample_infinity_nano(model, x_init.clone(), sigmas),
                       K.sample_euler(model, x_init.clone(), sigmas))


def test_infinity_nano_rejects_non_4d_latents():
    model = lambda x, sg: 0.3 * torch.tanh(x)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    with pytest.raises(ValueError, match="4-D"):
        K.sample_infinity_nano(model, torch.randn(1, 256, 64), sigmas)


def test_infinity_nano_runs_in_fp16_and_is_deterministic():
    sigmas = S.flow_matching_schedule(16, shift=3.0).half()
    torch.manual_seed(1)
    x_init = torch.randn(1, 4, 16, 16, dtype=torch.float16)
    model = lambda x, sg: (0.3 * torch.tanh(x.float())).half()
    a = K.sample_infinity_nano(model, x_init.clone(), sigmas)
    b = K.sample_infinity_nano(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)
    assert a.dtype == torch.float16
    assert torch.isfinite(a).all()


def test_infinity_nano_registered_in_sampler_table():
    assert K.get_sampler("infinity_nano") is K.sample_infinity_nano


# ── INFINITY (realism branch) ─────────────────────────────────────────
# Upstream's @21084d9 rewrite: the first-order x0 step plus a per-channel
# variance stabiliser; deterministic and no longer family-restricted.


def _variance_stabilize_reference(denoised, ema_std, momentum, progress, total_steps):
    """Verbatim transcription of upstream's ``_variance_stabilize``."""
    eps_std = 1e-4
    mean = denoised.mean(dim=(0, 2, 3), keepdim=True)
    centered = denoised - mean
    current_std = centered.std(dim=(0, 2, 3)).clamp(min=eps_std)
    if ema_std is None:
        return denoised, current_std.detach().clone()
    new_ema = momentum * ema_std + (1.0 - momentum) * current_std
    deviation = (current_std / (new_ema + eps_std) - 1.0).abs()
    strength = ((deviation / (deviation + 0.3))
                * (progress / (progress + 0.2))
                * (total_steps / (total_steps + 8)))
    target_std = current_std + (new_ema - current_std) * strength
    corr = (target_std / current_std).clamp(min=0.1, max=10.0)
    return centered * corr.reshape(1, -1, 1, 1) + mean, new_ema.detach()


def _infinity_realism_reference(model, x, sigmas):
    """Verbatim transcription of upstream realism's ``InfinitySampler.sample``
    (@21084d9), used to pin the port."""
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    variance_ema = None
    for i in range(total_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, s_cur * s_in)
        if i == 0:
            _, variance_ema = _variance_stabilize_reference(denoised, None, 0.0, 0.0, total_steps)
        else:
            denoised, variance_ema = _variance_stabilize_reference(
                denoised, variance_ema, 1.0 - 1.0 / total_steps, i / total_steps, total_steps)
        ratio = s_next / s_cur
        x = ratio * x - (ratio - 1) * denoised
    return x


@pytest.mark.parametrize("sigmas_fn", [_ve_sigmas, _flow_sigmas])
def test_infinity_realism_matches_upstream_recursion(sigmas_fn):
    sigmas = sigmas_fn()
    torch.manual_seed(5)
    x_init = torch.randn(1, 4, 8, 8) * sigmas[0]
    model = lambda x, sg: 0.4 * torch.tanh(x) - 0.1 * sg.view(-1, 1, 1, 1)
    out = K.sample_infinity_realism(model, x_init.clone(), sigmas)
    ref = _infinity_realism_reference(model, x_init.clone(), sigmas)
    assert torch.equal(out, ref)


def test_infinity_realism_is_deterministic_now():
    # The rewrite deleted the noise injection.
    sigmas = _flow_sigmas()
    torch.manual_seed(1)
    x_init = torch.randn(1, 4, 4, 4)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    a = K.sample_infinity_realism(model, x_init.clone(), sigmas)
    b = K.sample_infinity_realism(model, x_init.clone(), sigmas)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
    assert not torch.allclose(a, K.sample_infinity(model, x_init.clone(), sigmas))


def test_infinity_realism_first_step_is_uncorrected():
    # The bootstrap corrects nothing, so a 1-step run is the plain x0 step.
    sigmas = torch.tensor([1.0, 0.0])
    torch.manual_seed(2)
    x_init = torch.randn(1, 4, 8, 8)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    # allclose: upstream's r·x − (r−1)·x0 rounds differently from euler's form.
    assert torch.allclose(K.sample_infinity_realism(model, x_init.clone(), sigmas),
                          K.sample_euler(model, x_init.clone(), sigmas), atol=1e-7)


def test_infinity_realism_stabiliser_ramps_with_progress_and_step_count():
    # The pull must be inert early and damped at Turbo/LCM step counts.
    torch.manual_seed(6)
    base = torch.randn(1, 4, 16, 16)
    _, ema = K._variance_stabilize(base, None, 0.0, 0.0, 20)
    drifted = base * 3.0

    def pull(progress, steps):
        out, _ = K._variance_stabilize(drifted, ema, 1.0 - 1.0 / steps, progress, steps)
        return (drifted.std(dim=(0, 2, 3)) - out.std(dim=(0, 2, 3))).abs().max().item()

    assert pull(0.05, 20) < pull(0.5, 20) < pull(0.95, 20)   # progress ramp
    assert pull(0.95, 4) < pull(0.95, 40)                    # step-count guard
    # steps/(steps+8) holds the 4-step case to a third of its asymptote
    assert 4 / (4 + 8) == pytest.approx(1 / 3)


@pytest.mark.parametrize("sigmas_fn", [_ve_sigmas, _flow_sigmas])
def test_infinity_realism_lands_on_target(sigmas_fn):
    # Deterministic now, so the usual tolerance applies.
    target = torch.full((1, 4, 4, 4), 0.2)
    sigmas = sigmas_fn()
    x_init = torch.randn(1, 4, 4, 4) * sigmas[0]
    out = K.sample_infinity_realism(const_denoiser(target), x_init.clone(), sigmas)
    assert torch.allclose(out, target, atol=1e-3)


def test_infinity_realism_rejects_non_4d_latents():
    # The stabiliser needs spatial axes, so FLUX's token sequence is rejected.
    model = lambda x, sg: 0.3 * torch.tanh(x)
    with pytest.raises(ValueError, match="4-D"):
        K.sample_infinity_realism(model, torch.randn(1, 256, 64), _flow_sigmas())


def test_infinity_realism_registered_in_sampler_table():
    assert K.get_sampler("infinity_realism") is K.sample_infinity_realism


def test_all_registered_samplers_resolve():
    for name, fn in K.SAMPLERS.items():
        assert K.get_sampler(name) is fn


# ── INFINITY AETHER ───────────────────────────────────────────────────
# galpt/infinity-diffusion `aether` @c3ba017: omega plus a material-aware DoG,
# LISC, VNN, the TZTD decay and coherence-gated grain.


def test_infinity_aether_low_step_count_is_exactly_euler():
    torch.manual_seed(0)
    x_init = torch.randn(1, 4, 8, 8)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    sigmas = S.flow_matching_schedule(6, shift=3.0)
    assert torch.equal(K.sample_infinity_aether(model, x_init.clone(), sigmas),
                       K.sample_euler(model, x_init.clone(), sigmas))


def test_infinity_aether_rejects_non_4d_latents():
    model = lambda x, sg: 0.3 * torch.tanh(x)
    with pytest.raises(ValueError, match="4-D"):
        K.sample_infinity_aether(model, torch.randn(1, 256, 64),
                                 S.flow_matching_schedule(12, shift=3.0))


def test_infinity_aether_is_stochastic_but_seed_reproducible():
    # The one deviation from upstream: seeded noise.
    sigmas = S.karras_schedule(16, 0.0292, 14.6146)
    torch.manual_seed(1)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    run = lambda seed: K.sample_infinity_aether(
        model, x_init.clone(), sigmas, generator=torch.Generator().manual_seed(seed))
    a, b, c = run(3), run(3), run(4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.isfinite(a).all()


def test_infinity_aether_material_classes_are_the_upstream_mapping():
    # The nine Laws responses map to four classes (a LUT upstream spells as where).
    assert K._LAWS_MATERIAL == (0, 2, 1, 3, 3, 2, 2, 1, 1)
    assert len(K._LAWS_PAIRS) == len(K._LAWS_MATERIAL)
    torch.manual_seed(2)
    out = K._classify_material(torch.randn(1, 4, 16, 16))
    assert out.shape == (1, 4, 16, 16)
    assert int(out.min()) >= 0 and int(out.max()) <= 3


def test_infinity_aether_vnn_preserves_the_velocity_norm():
    # VNN keeps the enhancements from adding energy.
    torch.manual_seed(3)
    ref = torch.randn(2, 4, 16, 16)
    out = K._velocity_norm_normalize(ref * 3.7 + 0.5, ref)
    assert torch.allclose(out.flatten(1).norm(dim=1), ref.flatten(1).norm(dim=1), rtol=1e-5)


def test_infinity_aether_tztd_makes_terminal_steps_pure_euler():
    # Below σ = 0.15 steps are plain AVN-corrected Euler.
    sigmas = torch.linspace(0.14, 0.0, 21)
    torch.manual_seed(4)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    # σ < 0.15 still exceeds the 0.02 noise floor: subtract the same seeded draws.
    out = K.sample_infinity_aether(model, x_init.clone(), sigmas,
                                   generator=torch.Generator().manual_seed(0))
    ref = K._sample_infinity_pyramid(model, x_init.clone(), sigmas, name="ref",
                                     nqvp_sigma_min=K._NQVP_SIGMA_MIN_OMEGA,
                                     avn=True, dog=False)
    # They differ only by the grain, bounded by the 0.03 floor per step.
    assert (out - ref).abs().max() < 0.03 * 21


def test_infinity_aether_grain_is_gated_off_below_the_noise_floor():
    # σ <= 0.02 injects nothing, so the run is deterministic.
    sigmas = torch.linspace(0.02, 0.0, 9)
    torch.manual_seed(5)
    x_init = torch.randn(1, 4, 16, 16)
    model = lambda x, sg: 0.3 * torch.tanh(x)
    run = lambda seed: K.sample_infinity_aether(
        model, x_init.clone(), sigmas, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(run(1), run(2))


def test_infinity_aether_absolute_thresholds_land_differently_per_family():
    # The absolute σ thresholds cover most of flow's range but a sliver of SD's.
    band = lambda lo, hi: (0.80 - 0.15) / (hi - lo)
    assert band(0.0, 1.0) > 0.6                      # rectified flow
    assert band(0.0292, 14.6146) < 0.05              # SD/SDXL
    # The grain, capped at 0.08, is far heavier on flow relative to each step.
    worst = lambda sig: max(
        min(0.25 * float(sig[i]), 0.08) / (float(sig[i]) - float(sig[i + 1]))
        for i in range(len(sig) - 1))
    sd = worst(S.karras_schedule(32, 0.0292, 14.6146))
    flow = worst(S.flow_matching_schedule(32, shift=3.0))
    assert flow > 5.0 * sd


def test_infinity_aether_registered_in_sampler_table():
    assert K.get_sampler("infinity_aether") is K.sample_infinity_aether


# ── COGENT4: measurement instrumentation + the per-channel gate ────────
# cogent4: ``stats_out`` exposes the gate's raw statistics (v_est, s_est, rho,
# psi_linear, floor_active), and ``reduce="per_channel"`` is a separate,
# default-off estimator (``reduce="all"`` is the shipped gate bit-for-bit).


def _constant_increment_tensors(seed, D=4096, s_norm=1.0, v=0.35):
    # s_i = s_{i-1} = s (constant increment), n_i iid of energy v:
    # D_i = s + n_i - n_{i-1}, D_{i-1} = s + n_{i-1} - n_{i-2}.
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(D, generator=g, dtype=torch.float64)
    s = s / s.norm() * s_norm
    n = lambda: torch.randn(D, generator=g, dtype=torch.float64) * (v / D) ** 0.5
    n2, n1, n0 = n(), n(), n()
    return s, (s + n2 - n1), (s + n1 - n0)


def test_coherence_gate_stats_match_the_model():
    # Under the gate's own model stats_out reads the truth.
    s, diff, old_diff = _constant_increment_tensors(0)
    stats = {}
    psi = K._coherence_gate(diff.unsqueeze(0), old_diff.unsqueeze(0), _TINY_H,
                            stats_out=stats)
    S, v = 1.0, 0.35
    assert float(stats["v_est"]) == pytest.approx(v, abs=0.05)
    assert float(stats["s_est"]) == pytest.approx(S, abs=0.05)
    assert float(stats["rho"]) == pytest.approx((S - v) / (S + 2 * v), abs=0.02)
    assert float(stats["psi_linear"]) == pytest.approx(S / (S + 2 * v), abs=0.02)
    assert float(psi) == pytest.approx(S / (S + 2 * v), abs=0.02)
    assert not stats["floor_active"] and not stats["bootstrap"]


def test_coherence_gate_stats_floor_flag_and_bootstrap():
    # A huge h makes the floor win; the bootstrap reports floor_active only.
    s, diff, old_diff = _constant_increment_tensors(1)
    h = torch.tensor(3.0, dtype=torch.float64)          # floor ≈ 0.95
    stats = {}
    psi = K._coherence_gate(diff.unsqueeze(0), old_diff.unsqueeze(0), h,
                            stats_out=stats)
    floor = float((-h).expm1().neg())
    assert stats["floor_active"]
    assert float(psi) == pytest.approx(floor)
    assert float(stats["psi_linear"]) < floor           # floor really won
    stats2 = {}
    psi2 = K._coherence_gate(diff.unsqueeze(0), None, h, stats_out=stats2)
    assert stats2["bootstrap"] and stats2["floor_active"]
    assert float(psi2) == pytest.approx(floor)


def test_coherence_gate_stats_do_not_perturb_the_gate():
    # stats_out is write-only and detached.
    torch.manual_seed(3)
    a = torch.randn(3, 4, 8, 8, dtype=torch.float32, requires_grad=True)
    b = torch.randn(3, 4, 8, 8, dtype=torch.float32, requires_grad=True)
    h = torch.tensor(0.2, dtype=torch.float32)
    plain = K._coherence_gate(a, b, h)
    stats = {}
    logged = K._coherence_gate(a, b, h, stats_out=stats)
    assert torch.equal(plain, logged)
    assert logged.requires_grad
    assert all(not value.requires_grad for value in stats.values()
               if isinstance(value, torch.Tensor))
    pc_plain = K._coherence_gate(a, b, h, reduce="per_channel")
    pc_stats = {}
    pc_logged = K._coherence_gate(a, b, h, reduce="per_channel", stats_out=pc_stats)
    assert torch.equal(pc_plain, pc_logged)
    assert all(not value.requires_grad for value in pc_stats.values()
               if isinstance(value, torch.Tensor))


def test_coherence_gate_per_channel_is_a_different_estimator():
    # Aligned channels with swapped norms: per-channel [1, 1], global ≈ 0.347.
    u = torch.randn(1, 8, 16, dtype=torch.float64)
    u = u / u.norm()
    w = torch.randn(1, 8, 16, dtype=torch.float64)
    w = w - (u * w).sum() * u
    w = w / w.norm()
    d = torch.cat([1.0 * u, 100.0 * w], dim=0).unsqueeze(0)
    od = torch.cat([100.0 * u, 1.0 * w], dim=0).unsqueeze(0)
    pc = K._coherence_gate(d, od, _TINY_H, reduce="per_channel")
    g = K._coherence_gate(d, od, _TINY_H)
    assert torch.allclose(pc, torch.ones(1, 2, 1, 1, dtype=torch.float64), atol=1e-6)
    assert float(g) == pytest.approx((1 + 2 * 0.02) / 3, abs=0.02)  # global ≈ 0.347
    assert not torch.allclose(pc, g, atol=0.2)                     # genuinely different


def test_coherence_gate_per_channel_shape_reduce_all_pinned_and_reshape():
    # per_channel gives [B, C, 1, 1] on 4-D, falls back to "all" otherwise, and
    # "all" on the 4-D reshape equals the flat reduction.
    torch.manual_seed(4)
    a = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    b = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    h = torch.tensor(0.1, dtype=torch.float32)
    pc = K._coherence_gate(a, b, h, reduce="per_channel")
    assert pc.shape == (2, 4, 1, 1)
    fallback = K._coherence_gate(a.reshape(2, -1), b.reshape(2, -1), h,
                                 reduce="per_channel")
    assert fallback.shape == (2, 1)
    assert torch.equal(fallback, K._coherence_gate(a.reshape(2, -1),
                                                   b.reshape(2, -1), h))
    # Compare flattened; equality on differently shaped views is a shape check.
    assert torch.equal(K._coherence_gate(a, b, h).flatten(),
                       K._coherence_gate(a.reshape(2, -1), b.reshape(2, -1), h).flatten())
    with pytest.raises(ValueError):
        K._coherence_gate(a, b, h, reduce="bogus")


def test_cogent_gate_reduce_validation_is_eager_on_bootstrap_and_noop_runs():
    # Invalid modes must fail even with no history or no steps.
    a = torch.randn(1, 4, 8, 8)
    h = torch.tensor(0.1)
    with pytest.raises(ValueError, match="reduce must be"):
        K._coherence_gate(a, None, h, reduce="bogus")
    with pytest.raises(ValueError, match="reduce must be"):
        K._cogent3_curvature_gate(a, None, reduce="bogus")

    sigmas = torch.ones(1)
    model = const_denoiser(torch.zeros_like(a))
    with pytest.raises(ValueError, match="reduce must be"):
        K.sample_cogent(model, a, sigmas, gate_reduce="bogus")
    with pytest.raises(ValueError, match="reduce must be"):
        K.sample_cogent3(model, a, sigmas, gate_reduce="bogus")


def test_cogent3_gate_reduce_all_reshape_pin():
    # gate_reduce="all" on the reshaped latent reproduces the flat run.
    torch.manual_seed(7)
    target = torch.randn(1, 4, 8, 8)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_flat = torch.randn(1, 4, 8, 8)

    def run(x, tgt, **kw):
        return _last_nonzero_latent(
            K.sample_cogent3, _anneal_sigma_dependent_model(tgt), x, sigmas,
            model_type="flow", shift=3.0, **kw)

    a = run(x_flat, target, gate_reduce="all", eta_max=0.0)
    b = run(x_flat, target, eta_max=0.0)        # default == "all"
    c = run(x_flat.reshape(1, 2, 8, 16), target.reshape(1, 2, 8, 16),
            gate_reduce="all", eta_max=0.0)
    assert torch.equal(a, b)                    # explicit "all" == default, bit-for-bit
    assert torch.equal(a.flatten(), c.flatten())  # reshape pin, bit-for-bit
    assert c.shape == (1, 2, 8, 16)


# ── measurement oracles for the lag-1 estimator ─────────────────────────
# A1: the Wiener identity; A2: the heteroscedastic oracle; A3: the signal term a
# naive target would misread as estimator bias.

def _draw_diff(signal_i, signal_im1, noise_energy, seed, D=4096):
    # D_i = s_i + n_i - n_{i-1} given s_i and s_{i-1} (through the shared n_{i-1}).
    g = torch.Generator().manual_seed(seed)
    v = lambda e: torch.randn(D, generator=g, dtype=torch.float64) * (e / D) ** 0.5
    n_i, n_im1, n_im2 = v(noise_energy[2]), v(noise_energy[1]), v(noise_energy[0])
    return (signal_i + n_i - n_im1), (signal_im1 + n_im1 - n_im2)


def test_stage_a1_wiener_identity():
    # Constant-increment signal, homoscedastic noise: v_est → v, psi → S/(S+2v).
    s = torch.randn(4096, dtype=torch.float64)
    s = s / s.norm()
    v = 0.35
    vests, psis = [], []
    for seed in range(200):
        diff, old_diff = _draw_diff(s, s, (v, v, v), seed)
        stats = {}
        K._coherence_gate(diff.unsqueeze(0), old_diff.unsqueeze(0), _TINY_H,
                          stats_out=stats)
        vests.append(float(stats["v_est"]))
        psis.append(float(stats["psi_linear"]))
    assert sum(vests) / len(vests) == pytest.approx(v, rel=0.02)
    assert sum(psis) / len(psis) == pytest.approx(1.0 / (1.0 + 2.0 * v), rel=0.02)


def test_stage_a2_heteroscedastic_oracle():
    # Heteroscedastic noise: rho follows (S - v_{i-1}) /
    # sqrt((S+v_i+v_{i-1})(S+v_{i-1}+v_{i-2})).
    s = torch.randn(4096, dtype=torch.float64)
    s = s / s.norm()
    vim2, vim1, vi = 0.20, 0.50, 0.90
    oracle = (1.0 - vim1) / ((1.0 + vi + vim1) * (1.0 + vim1 + vim2)) ** 0.5
    rhos = []
    for seed in range(200):
        diff, old_diff = _draw_diff(s, s, (vim2, vim1, vi), seed)
        stats = {}
        K._coherence_gate(diff.unsqueeze(0), old_diff.unsqueeze(0), _TINY_H,
                          stats_out=stats)
        rhos.append(float(stats["rho"]))
    assert sum(rhos) / len(rhos) == pytest.approx(oracle, abs=0.01)
    # ... and the homoscedastic target is the wrong oracle at this spread.
    homosced = (1.0 - vi) / (1.0 + 2.0 * vi)
    assert abs(sum(rhos) / len(rhos) - oracle) < abs(sum(rhos) / len(rhos) - homosced)


def test_stage_a3_signal_bias_expectation():
    # Non-stationary signal: E[3·v_est] = ‖s_i‖² - <s_i, s_{i-1}> + v_i +
    # 2·v_{i-1}, so a signal term rides along.
    s_i = torch.randn(4096, dtype=torch.float64)
    s_i = s_i / s_i.norm()
    rot = torch.randn(4096, dtype=torch.float64)
    rot = rot - (s_i * rot).sum() * s_i
    rot = rot / rot.norm()
    s_im1 = 0.6 * s_i + 0.8 * rot                       # <s_i, s_{i-1}> = 0.6
    vim2, vim1, vi = 0.20, 0.35, 0.50
    expect = 1.0 - 0.6 + vi + 2.0 * vim1                # = 1.6
    got = []
    for seed in range(200):
        diff, old_diff = _draw_diff(s_i, s_im1, (vim2, vim1, vi), seed)
        stats = {}
        K._coherence_gate(diff.unsqueeze(0), old_diff.unsqueeze(0), _TINY_H,
                          stats_out=stats)
        got.append(3.0 * float(stats["v_est"]))
    assert sum(got) / len(got) == pytest.approx(expect, abs=0.05)
    assert sum(got) / len(got) > vi + 2.0 * vim1        # naive target is low


def test_cogent3_gate_stats_collection_does_not_perturb_output():
    # One stats dict per correctable step, and the output is unchanged.
    torch.manual_seed(9)
    target = torch.randn(1, 4, 8, 8)
    sigmas = S.flow_matching_schedule(12, shift=3.0)
    x_init = torch.randn(1, 4, 8, 8)
    stats = []
    got = _last_nonzero_latent(K.sample_cogent3, _anneal_sigma_dependent_model(target),
                               x_init, sigmas, gate_stats=stats,
                               eta_max=1.0, model_type="flow", shift=3.0,
                               generator=torch.Generator().manual_seed(1))
    want = _last_nonzero_latent(K.sample_cogent3, _anneal_sigma_dependent_model(target),
                                x_init, sigmas,
                                eta_max=1.0, model_type="flow", shift=3.0,
                                generator=torch.Generator().manual_seed(1))
    assert torch.equal(got, want)                       # collection is write-only
    # 13 sigmas: 10 entries (1 floor-only bootstrap + 9 full reads).
    assert len(stats) == len(sigmas) - 3
    assert stats[0]["bootstrap"] and stats[0]["floor_active"]
    for entry in stats[1:]:
        for key in ("rho", "s_est", "v_est", "psi_linear", "floor_active",
                    "bootstrap", "step", "sigma", "sigma_next", "h"):
            assert key in entry
        assert entry["bootstrap"] is False


# ── LUMEN ─────────────────────────────────────────────────────────────
# galpt/infinity-diffusion's geometric solver: res_multistep's correction plus
# three guards (damping, an Euler tail, a magnitude fallback).


def _lumen_curved_model():
    # σ- and x-dependent x0 so the correction and damping both run.
    def model(x, sigma):
        s = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        return torch.tanh(0.7 * x) / (1.0 + s) + 0.1 * torch.sin(x + s)
    return model


def test_lumen_core_equals_res_multistep():
    # With the guards off LUMEN is the RES step (float64: round-off, not tolerance).
    sigmas = S.karras_schedule(12, 0.0292, 14.6146, dtype=torch.float64)
    x_init = torch.randn(1, 4, 8, 8, dtype=torch.float64) * sigmas[0]
    model = _lumen_curved_model()
    got = K.sample_lumen(model, x_init.clone(), sigmas,
                         tau=float("inf"), tail_euler=0, guard_ratio=float("inf"))
    want = K.sample_res_multistep(model, x_init.clone(), sigmas)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("guard_off", [
    {"tau": 0.0},                       # kappa == 0, the correction is scaled away
    {"tail_euler": 12},                 # every non-terminal step is in the tail
    {"guard_ratio": 0.0},               # no correction ever passes the magnitude test
])
def test_lumen_each_guard_falls_back_to_euler(guard_off):
    sigmas = S.karras_schedule(12, 0.0292, 14.6146, dtype=torch.float64)
    x_init = torch.randn(1, 4, 8, 8, dtype=torch.float64) * sigmas[0]
    model = _lumen_curved_model()
    got = K.sample_lumen(model, x_init.clone(), sigmas, **guard_off)
    want = K.sample_euler(model, x_init.clone(), sigmas)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)


def test_lumen_damping_shrinks_on_a_jumping_x0():
    d = torch.randn(2, 4, 8, 8)
    smooth = K._lumen_damping(d, d + 1e-3 * torch.randn_like(d), K.LUMEN_TAU)
    jumpy = K._lumen_damping(d, d + 4.0 * torch.randn_like(d), K.LUMEN_TAU)
    assert torch.allclose(smooth, torch.ones(2))         # saturates at 1 when D barely moves
    assert (jumpy < 0.5).all()
    # Scale-invariant: both mean|·| terms scale with the latent.
    assert torch.allclose(jumpy, K._lumen_damping(7.0 * d, 7.0 * (d + 4.0 * torch.randn_like(d)),
                                                  K.LUMEN_TAU), atol=0.1)


def test_lumen_guards_are_active_at_the_shipped_defaults():
    # Sanity that the defaults are not a no-op wrapper around res_multistep.
    sigmas = S.karras_schedule(12, 0.0292, 14.6146, dtype=torch.float64)
    x_init = torch.randn(1, 4, 8, 8, dtype=torch.float64) * sigmas[0]
    model = _lumen_curved_model()
    got = K.sample_lumen(model, x_init.clone(), sigmas)
    assert not torch.allclose(got, K.sample_res_multistep(model, x_init.clone(), sigmas))
    assert not torch.allclose(got, K.sample_euler(model, x_init.clone(), sigmas))


def test_lumen_registered_in_sampler_table():
    assert K.get_sampler("lumen") is K.sample_lumen
