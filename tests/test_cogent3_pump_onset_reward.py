"""cogent3_pump_onset_reward: the best-of-k pick on the onset-expanded grid."""
from __future__ import annotations

import pytest
import torch

from diffucore.sampling import samplers as K
from diffucore.sampling import schedules as S

from test_cogent3_reward import _Bb, _bundle, _gen, _model, _problem  # noqa: F401  (test helpers)

NAME = "cogent3_pump_onset_reward"
_PUMP = dict(pump_strength=0.08, pump_h_ref=K.PUMP_H_REF)


def _setup(**over):
    target, x0 = _problem(7)
    kw = dict(model_type="flow", shift=3.0, eta_max=1.0, gate_reduce="per_channel", **_PUMP)
    kw.update(over)
    return _model(target), x0, S.flow_table_schedule("pump_taper", 3.0, 24), kw


def _plain(model, x0, sigmas, kw, seed, callback=None):
    return K.sample_cogent3_onset(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(seed),
                                  callback=callback, **kw)


def _scorer_for(j):
    calls = []

    def score_fn(x, sigma):
        calls.append(sigma)
        return 5.0 if len(calls) - 1 == j else float(len(calls) % 3) * 0.1
    return score_fn


def test_registry_and_reward_samplers():
    fn = K.get_sampler(NAME)
    assert fn.func is K.sample_cogent3_onset_reward
    assert fn.keywords == {"k": K.REWARD_SAMPLERS[NAME], "pump_strength": 0.08, "pump_h_ref": K.PUMP_H_REF}
    assert K.REWARD_SAMPLERS[NAME] >= 3
    assert set(K.REWARD_SAMPLERS) == {"cogent3_reward", "secant_anneal_reward", NAME, "cogent3_pump_rate_reward"}
    assert "sample_cogent3_onset_reward" in K.__all__
    from diffucore.sampling import sample_cogent3_onset_reward
    assert sample_cogent3_onset_reward is K.sample_cogent3_onset_reward


@pytest.mark.parametrize("override", [dict(score_fn=None), dict(score_fn=lambda x, s: 1.0, k=1)])
def test_without_score_fn_it_is_plain_cogent3_pump_onset(override):
    model, x0, sigmas, kw = _setup()
    override = {"k": 6, **override}
    info, idx = {}, []
    out = K.sample_cogent3_onset_reward(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(3),
                                        info=info, callback=lambda i, *a: idx.append(i), **override, **kw)
    ref = []
    plain = _plain(model, x0, sigmas, kw, 3, callback=lambda i, *a: ref.append(i))
    assert torch.equal(out, plain) and info == {} and idx == ref == list(range(len(sigmas) - 1))
    reg = K.get_sampler("cogent3_pump_onset")
    assert torch.equal(plain, reg(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(3),
                                  **{k: v for k, v in kw.items() if k not in _PUMP}))


@pytest.mark.parametrize("j", [0, 2, 5])
def test_winner_equals_that_particle_run_alone_on_the_fine_grid(j):
    model, x0, sigmas, kw = _setup()
    seed, info = 1234, {}
    out = K.sample_cogent3_onset_reward(model, x0.clone(), sigmas, k=6, score_fn=_scorer_for(j),
                                        generator=torch.Generator().manual_seed(seed), info=info, **kw)
    assert info["pick"] == j and len(info["scores"]) == 6
    assert torch.equal(out, _plain(model, x0, sigmas, kw, K.reward_particle_seed(seed, j)))


def test_score_step_lands_on_an_original_grid_point_of_the_fine_grid():
    model, x0, sigmas, kw = _setup()
    fine, outer = S_expand(sigmas)
    info, seen = {}, []
    K.sample_cogent3_onset_reward(model, x0.clone(), sigmas, k=3, score_fn=lambda x, s: seen.append(s) or 0.0,
                                  generator=torch.Generator().manual_seed(1), info=info, **kw)
    i = K.reward_score_index(sigmas)
    i_fine = K.reward_score_index(fine)
    assert i_fine > i and info["score_step"] == i_fine
    assert fine[i_fine].item() == sigmas[i].item()          # an original point, not a sub-step
    assert outer[i_fine] == i and all(s == pytest.approx(float(sigmas[i])) for s in seen)


def S_expand(sigmas):
    return K._expand_onset_sigmas(sigmas, 2, K.ONSET_SIGMA)


def test_callback_fires_once_per_original_step_per_particle():
    model, x0, sigmas, kw = _setup()
    idx = []
    K.sample_cogent3_onset_reward(model, x0.clone(), sigmas, k=4, score_fn=lambda x, s: 0.0,
                                  generator=torch.Generator().manual_seed(4),
                                  callback=lambda i, *a: idx.append(i), **kw)
    n, i = len(sigmas) - 1, K.reward_score_index(sigmas)
    assert idx == list(range(i)) * 4 + list(range(i, n))


def test_tie_goes_to_the_lowest_j_and_runs_are_deterministic():
    model, x0, sigmas, kw = _setup()

    def run(fn):
        info = {}
        out = K.sample_cogent3_onset_reward(model, x0.clone(), sigmas, k=4, score_fn=fn,
                                            generator=torch.Generator().manual_seed(77), info=info, **kw)
        return out, info
    assert run(lambda x, s: 0.5)[1]["pick"] == 0
    (a, ia), (b, ib) = run(lambda x, s: float(x.mean())), run(lambda x, s: float(x.mean()))
    assert torch.equal(a, b) and ia == ib and len(set(ia["scores"])) > 1


# ── pipeline wiring (fake backbone, CPU) ───────────────────────────────

def test_pipeline_forces_the_band_scores_k_particles_and_counts_progress_exactly(monkeypatch):
    from diffucore.pipelines import _anima
    from diffucore.pipelines._anima import _ANIMA_SAMPLERS, _FLOW_AWARE_SAMPLERS, _onset_lowres_sigma
    assert NAME in _ANIMA_SAMPLERS and NAME in _FLOW_AWARE_SAMPLERS
    assert _onset_lowres_sigma(NAME, 0.0, True) == 0.9 and _onset_lowres_sigma(NAME, 0.0, False) == 0.9
    assert _onset_lowres_sigma(NAME, 0.85, True) == 0.85
    k = K.REWARD_SAMPLERS[NAME]
    scores = []

    def reward(img):
        scores.append(float(img.mean()))
        return torch.tensor([scores[-1]])

    seen = []
    bb, _, info = _gen(monkeypatch, NAME, reward, progress_callback=lambda n, t: seen.append((n, t)))
    sig = _anima.flow_table_schedule("pump_taper", 3.0, 12, device="cpu", dtype=torch.float32)
    i = K.reward_score_index(sig)
    assert len(scores) == k and len(set(scores)) > 1
    assert info.reward["k"] == k and info.reward["scores"] == scores
    assert info.reward["pick"] == max(range(k), key=lambda j: (scores[j], -j))
    sizes = [hw for _, hw, _ in bb.calls]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16) and sizes == sorted(sizes)
    steps = [n for n, _ in seen]
    assert steps == sorted(steps) and seen[-1] == (k * i + 12 - i,) * 2 and len({t for _, t in seen}) == 1


def test_pipeline_winner_zero_matches_plain_cogent3_pump_onset(monkeypatch):
    _, plain, _ = _gen(monkeypatch, "cogent3_pump_onset")
    vals = iter([10.0] + [0.0] * 20)
    _, out, info = _gen(monkeypatch, NAME, lambda img: torch.tensor([next(vals)]))
    assert info.reward["pick"] == 0 and torch.equal(out, plain)


def test_pipeline_without_reward_fn_is_plain_cogent3_pump_onset_with_its_band(monkeypatch):
    bb, out, info = _gen(monkeypatch, NAME)
    bb2, out2, _ = _gen(monkeypatch, "cogent3_pump_onset")
    assert info.reward is None and torch.equal(out, out2)
    assert [c[:2] for c in bb.calls] == [c[:2] for c in bb2.calls]
    assert (8, 8) in {hw for _, hw, _ in bb.calls}
    with pytest.raises(ValueError, match="forecast"):
        _gen(monkeypatch, NAME, lambda img: torch.tensor([0.0]), forecast_ratio=0.3)


def test_pipeline_pump_and_gate_knobs_reach_the_sampler(monkeypatch):
    got = {}
    real = K.get_sampler

    def spy(name):
        fn = real(name)
        return lambda *a, **kw: got.update(kw) or fn(*a, **kw)
    monkeypatch.setattr("diffucore.pipelines._anima.get_sampler", spy)
    _gen(monkeypatch, NAME, lambda img: torch.tensor([0.0]), eta_max=0.4, gate_reduce="per_channel")
    assert got["eta_max"] == 0.4 and got["gate_reduce"] == "per_channel" and "score_fn" in got
