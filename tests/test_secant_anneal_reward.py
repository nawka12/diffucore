"""secant_anneal stepper refactor (bit-exactness) and the best-of-k secant_anneal_reward sampler."""
from __future__ import annotations

import pytest
import torch

from diffucore.sampling import samplers as K
from diffucore.sampling import schedules as S

from test_cogent3_reward import _Bb, _bundle, _gen, _model, _problem  # noqa: F401  (test helpers)

NAME = "secant_anneal_reward"

# Frozen copy of sample_secant_anneal from before it became a resumable stepper.
def _secant_anneal_reference(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 1.0,
    s_noise: float = 1.0,
    curvature: float = 0.25,
    generator=None,
    callback=None,
    model_type: str = "flow",
    shift: float = 1.0,
) -> torch.Tensor:
    if model_type != "flow":
        raise ValueError("secant_anneal is rectified-flow only (model_type='flow')")
    del shift
    s_in = x.new_ones([x.shape[0]])
    old_x0 = None
    old_sigma = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            eta = eta_max * float(sigma.clamp(max=1.0))
            sigma_down, alpha_next, alpha_down, renoise_coeff = K._rf_ancestral_step(sigma, sigma_next, eta)
            d = K.to_d(x, sigma * s_in, denoised)
            if old_x0 is None or bool((sigma - old_sigma).abs() < 1e-8):
                x0_eff = denoised
            else:
                x0_slope = (denoised - old_x0) / (sigma - old_sigma)
                x0_pred = denoised + x0_slope * (sigma_down - sigma)
                r = ((sigma_down - sigma).abs() / sigma).clamp(0.0, 1.0)
                reliable = (1.0 - sigma).clamp(0.0, 1.0)
                beta = float(curvature) * (1.0 - r) * reliable
                x0_eff = (1.0 - beta) * denoised + beta * x0_pred
            eps_i = x + (1.0 - sigma) * d
            x = (1.0 - sigma_down) * x0_eff + sigma_down * eps_i
            if eta > 0 and s_noise > 0:
                x = (alpha_next / alpha_down) * x + K._noise_like(x, generator) * s_noise * renoise_coeff
        old_x0 = denoised
        old_sigma = sigma
    return x


_GRIDS = {
    "flow16": lambda: S.flow_matching_schedule(16, shift=3.0),
    "taper24": lambda: S.flow_table_schedule("pump_taper", 3.0, 24),
    "flow5": lambda: S.flow_matching_schedule(5, shift=1.0),
}
_VARIANTS = [{}, dict(eta_max=0.5, s_noise=0.9), dict(eta_max=0.0), dict(curvature=1.0), dict(curvature=0.0)]


@pytest.mark.parametrize("grid", sorted(_GRIDS))
@pytest.mark.parametrize("variant", range(len(_VARIANTS)))
@pytest.mark.parametrize("seed", [0, 11])
def test_stepper_is_bit_exact_with_the_pre_refactor_sampler(grid, variant, seed):
    sigmas = _GRIDS[grid]()
    target, x0 = _problem(seed + 1)
    runs = []
    for fn in (_secant_anneal_reference, K.sample_secant_anneal):
        xs = []
        g = torch.Generator().manual_seed(seed)
        out = fn(_model(target), x0.clone(), sigmas, generator=g, shift=3.0,
                 callback=lambda i, s, x, d: xs.append((i, x.clone(), d.clone())), **_VARIANTS[variant])
        runs.append((out, xs, torch.randn(3, generator=g)))
    (a, xa, ga), (b, xb, gb) = runs
    assert torch.equal(a, b) and torch.equal(ga, gb)
    assert len(xa) == len(xb) and all(i == j and torch.equal(p, q) and torch.equal(u, v)
                                      for (i, p, u), (j, q, v) in zip(xa, xb))


def test_stepper_is_flow_only_and_resumes_to_the_same_run():
    target, x0 = _problem()
    sigmas = S.flow_table_schedule("pump_taper", 3.0, 24)
    with pytest.raises(ValueError, match="flow"):
        K.sample_secant_anneal(_model(target), x0, sigmas, model_type="ve")
    with pytest.raises(ValueError, match="flow"):
        K.sample_secant_anneal_reward(_model(target), x0, sigmas, k=3, score_fn=lambda x, s: 0.0,
                                      model_type="ve")
    n = len(sigmas) - 1
    kw = dict(eta_max=1.0, s_noise=1.0, curvature=0.25, callback=None)
    whole = K._SecantAnnealState(x0.clone())
    K._secant_anneal_steps(_model(target), whole, sigmas, 0, n, generator=torch.Generator().manual_seed(2), **kw)
    for i in (1, 7, n - 1):
        part, g = K._SecantAnnealState(x0.clone()), torch.Generator().manual_seed(2)
        K._secant_anneal_steps(_model(target), part, sigmas, 0, i, generator=g, **kw)
        K._secant_anneal_steps(_model(target), part, sigmas, i, n, generator=g, **kw)
        assert torch.equal(part.x, whole.x)
    assert torch.equal(whole.x, K.sample_secant_anneal(_model(target), x0.clone(), sigmas,
                                                       generator=torch.Generator().manual_seed(2)))


# ── reward sampler ─────────────────────────────────────────────────────

def _reward_setup(**over):
    target, x0 = _problem(7)
    kw = dict(model_type="flow", shift=3.0, eta_max=1.0, curvature=0.5)
    kw.update(over)
    return _model(target), x0, S.flow_table_schedule("pump_taper", 3.0, 24), kw


def _plain(model, x0, sigmas, kw, seed):
    return K.sample_secant_anneal(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(seed), **kw)


def _scorer_for(j):
    calls = []

    def score_fn(x, sigma):
        calls.append(sigma)
        return 5.0 if len(calls) - 1 == j else float(len(calls) % 3) * 0.1
    return score_fn


@pytest.mark.parametrize("j", [0, 1, 3, 5])
def test_winner_equals_that_particle_run_alone(j):
    model, x0, sigmas, kw = _reward_setup()
    seed = 1234
    info = {}
    out = K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=6, score_fn=_scorer_for(j),
                                        generator=torch.Generator().manual_seed(seed), info=info, **kw)
    assert info["pick"] == j and info["k"] == 6 and len(info["scores"]) == 6
    assert torch.equal(out, _plain(model, x0, sigmas, kw, K.reward_particle_seed(seed, j)))


def test_particles_stop_at_first_sigma_below_score_sigma_and_score_there():
    model, x0, sigmas, kw = _reward_setup()
    seen = []
    K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=3, score_sigma=0.90,
                                  score_fn=lambda x, s: seen.append((x.clone(), s)) or 0.0,
                                  generator=torch.Generator().manual_seed(1), **kw)
    i = next(n for n, s in enumerate(sigmas.tolist()) if s <= 0.90)
    assert len(seen) == 3 and all(s == pytest.approx(float(sigmas[i])) for _, s in seen)
    ref = {}
    K.sample_secant_anneal(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(1),
                           callback=lambda n, s, x, d: ref.__setitem__(n, x.clone()), **kw)
    assert torch.equal(seen[0][0], ref[i])


def test_ties_go_to_the_lowest_j_and_nan_never_wins():
    model, x0, sigmas, kw = _reward_setup()
    info = {}
    K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=5, score_fn=lambda x, s: 0.5,
                                  generator=torch.Generator().manual_seed(2), info=info, **kw)
    assert info["pick"] == 0
    vals = iter([float("nan"), 1.0, 1.0, 0.2])
    K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=4, score_fn=lambda x, s: next(vals),
                                  generator=torch.Generator().manual_seed(2), info=info, **kw)
    assert info["pick"] == 1


@pytest.mark.parametrize("override", [dict(k=1, score_fn=lambda x, s: 1.0), dict(k=6, score_fn=None)])
def test_k1_and_no_score_fn_are_plain_secant_anneal(override):
    model, x0, sigmas, kw = _reward_setup()
    info, steps = {}, []
    out = K.sample_secant_anneal_reward(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(3),
                                        info=info, callback=lambda *a: steps.append(a[0]), **override, **kw)
    assert torch.equal(out, _plain(model, x0, sigmas, kw, 3))
    assert info == {} and steps == list(range(len(sigmas) - 1))


def test_deterministic_same_seed_same_pick_and_output():
    model, x0, sigmas, kw = _reward_setup()

    def run():
        info = {}
        out = K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=5,
                                            score_fn=lambda x, s: float(x.mean()),
                                            generator=torch.Generator().manual_seed(77), info=info, **kw)
        return out, info
    (a, ia), (b, ib) = run(), run()
    assert torch.equal(a, b) and ia == ib
    assert len(set(ia["scores"])) > 1                      # particles really differ


def test_info_keys_and_callback_order():
    model, x0, sigmas, kw = _reward_setup()
    idx, info = [], {}
    K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=3, score_fn=lambda x, s: 0.0,
                                  generator=torch.Generator().manual_seed(4), info=info,
                                  callback=lambda i, *a: idx.append(i), **kw)
    n, i = len(sigmas) - 1, K.reward_score_index(sigmas)
    assert set(info) == {"k", "pick", "scores", "score_step", "score_sigma"} and info["score_step"] == i
    assert len(idx) == 3 * i + (n - i)
    assert idx == list(range(i)) * 3 + list(range(i, n))


def test_particle_zero_uses_the_given_generator_object():
    model, x0, sigmas, kw = _reward_setup()
    g = torch.Generator().manual_seed(6)
    K.sample_secant_anneal_reward(model, x0.clone(), sigmas, k=3, score_fn=lambda x, s: 0.0, generator=g, **kw)
    ref = torch.Generator().manual_seed(6)
    K.sample_secant_anneal(model, x0.clone(), sigmas, generator=ref, **kw)
    assert torch.equal(torch.randn(4, generator=g), torch.randn(4, generator=ref))


def test_degenerate_grids_fall_back_to_plain():
    sig = S.flow_matching_schedule(8, shift=3.0)
    model, x0, _, kw = _reward_setup()
    out = K.sample_secant_anneal_reward(model, x0.clone(), sig, k=4, score_sigma=1.5,
                                        score_fn=lambda x, s: 1 / 0,
                                        generator=torch.Generator().manual_seed(8), **kw)
    assert torch.equal(out, _plain(model, x0, sig, kw, 8))
    one = sig[:1]
    assert torch.equal(K.sample_secant_anneal_reward(model, x0, one, k=4, score_fn=lambda x, s: 1 / 0), x0)


def test_registry_and_reward_samplers():
    fn = K.get_sampler(NAME)
    assert fn.func is K.sample_secant_anneal_reward and fn.keywords == {"k": 6}
    assert K.REWARD_SAMPLERS == {"cogent3_reward": 6, NAME: 6, "cogent3_pump_onset_reward": 3,
                               "cogent3_pump_rate_reward": 6}
    assert "sample_secant_anneal_reward" in K.__all__
    from diffucore.sampling import sample_secant_anneal_reward
    assert sample_secant_anneal_reward is K.sample_secant_anneal_reward


# ── pipeline wiring (fake backbone, CPU) ───────────────────────────────

def test_pipeline_forces_the_band_scores_k_particles_and_records_the_pick(monkeypatch):
    from diffucore.pipelines import _anima
    from diffucore.pipelines._anima import _ANIMA_SAMPLERS, _FLOW_AWARE_SAMPLERS, _onset_lowres_sigma
    assert NAME in _ANIMA_SAMPLERS and NAME in _FLOW_AWARE_SAMPLERS
    assert _onset_lowres_sigma(NAME, 0.0, True) == 0.9 and _onset_lowres_sigma(NAME, 0.0, False) == 0.0
    scores = []

    def reward(img):
        scores.append(float(img.mean()))
        return torch.tensor([scores[-1]])

    seen = []
    bb, _, info = _gen(monkeypatch, NAME, reward, progress_callback=lambda n, t: seen.append((n, t)))
    sig = _anima.flow_table_schedule("pump_taper", 3.0, 12, device="cpu", dtype=torch.float32)
    i = K.reward_score_index(sig)
    assert len(scores) == 6 and len(set(scores)) > 1
    assert info.reward["k"] == 6 and info.reward["scores"] == scores and info.reward["score_step"] == i
    assert info.reward["pick"] == max(range(6), key=lambda j: (scores[j], -j))
    sizes = [hw for _, hw, _ in bb.calls]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16) and sizes == sorted(sizes)
    steps = [n for n, _ in seen]
    assert steps == sorted(steps) and seen[-1] == (6 * i + 12 - i,) * 2


def test_pipeline_winner_zero_matches_plain_secant_anneal_with_the_band(monkeypatch):
    _, plain, _ = _gen(monkeypatch, "secant_anneal", lowres_sigma=0.9)
    vals = iter([10.0, 0, 0, 0, 0, 0])
    _, out, info = _gen(monkeypatch, NAME, lambda img: torch.tensor([next(vals)]))
    assert info.reward["pick"] == 0 and torch.equal(out, plain)


def test_pipeline_curvature_and_eta_max_reach_the_reward_sampler(monkeypatch):
    got = {}
    real = K.get_sampler

    def spy(name):
        fn = real(name)
        return lambda *a, **kw: got.update(kw) or fn(*a, **kw)
    monkeypatch.setattr("diffucore.pipelines._anima.get_sampler", spy)
    _gen(monkeypatch, NAME, lambda img: torch.tensor([0.0]), curvature=0.7, eta_max=0.4)
    assert got["curvature"] == 0.7 and got["eta_max"] == 0.4 and "gate_reduce" not in got
    assert got["model_type"] == "flow" and "score_fn" in got and "info" in got


def test_no_reward_fn_is_plain_secant_anneal_and_forecast_is_refused_only_with_one(monkeypatch):
    bb, out, info = _gen(monkeypatch, NAME)
    _, out2, _ = _gen(monkeypatch, "secant_anneal")
    assert info.reward is None and torch.equal(out, out2)
    assert {hw for _, hw, _ in bb.calls} == {(16, 16)}
    with pytest.raises(ValueError, match="forecast"):
        _gen(monkeypatch, NAME, lambda img: torch.tensor([0.0]), forecast_ratio=0.3)
    _gen(monkeypatch, NAME, forecast_ratio=0.3, forecast_sigma=0.5)
