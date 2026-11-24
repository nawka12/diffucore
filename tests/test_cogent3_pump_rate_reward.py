"""cogent3_pump_rate_reward: cogent3_reward's best-of-k pick with the cogent3_pump_rate pump."""
from __future__ import annotations

import pytest
import torch

from diffucore.sampling import samplers as K
from diffucore.sampling import schedules as S

from test_cogent3_reward import _Bb, _bundle, _gen, _model, _problem, _scorer_for  # noqa: F401  (test helpers)

NAME = "cogent3_pump_rate_reward"
_PUMP = dict(pump_strength=0.08, pump_h_ref=K.PUMP_H_REF)


def _setup():
    target, x0 = _problem(7)
    kw = dict(model_type="flow", shift=3.0, eta_max=1.0, gate_reduce="per_channel")
    return _model(target), x0, S.flow_table_schedule("pump_taper", 3.0, 24), kw


def _plain(model, x0, sigmas, kw, seed):
    return K.get_sampler("cogent3_pump_rate")(model, x0.clone(), sigmas,
                                              generator=torch.Generator().manual_seed(seed), **kw)


def test_registry_and_reward_samplers():
    fn = K.get_sampler(NAME)
    assert fn.func is K.sample_cogent3_reward
    assert fn.keywords == {"k": 6, **_PUMP} and K.REWARD_SAMPLERS[NAME] == 6


@pytest.mark.parametrize("override", [dict(score_fn=None), dict(score_fn=lambda x, s: 1.0, k=1)])
def test_without_score_fn_it_is_plain_cogent3_pump_rate(override):
    model, x0, sigmas, kw = _setup()
    info = {}
    out = K.get_sampler(NAME)(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(3),
                              info=info, **override, **kw)
    assert torch.equal(out, _plain(model, x0, sigmas, kw, 3)) and info == {}


@pytest.mark.parametrize("j", [0, 2, 5])
def test_winner_equals_that_particle_run_alone_through_cogent3_pump_rate(j):
    model, x0, sigmas, kw = _setup()
    seed, info = 1234, {}
    out = K.get_sampler(NAME)(model, x0.clone(), sigmas, score_fn=_scorer_for(j, 6),
                              generator=torch.Generator().manual_seed(seed), info=info, **kw)
    assert info["pick"] == j and info["k"] == 6 and len(info["scores"]) == 6
    assert torch.equal(out, _plain(model, x0, sigmas, kw, K.reward_particle_seed(seed, j)))


def test_pump_changes_the_picked_image():
    model, x0, sigmas, kw = _setup()
    run = lambda name: K.get_sampler(name)(model, x0.clone(), sigmas, score_fn=_scorer_for(2, 6),
                                           generator=torch.Generator().manual_seed(9), **kw)
    assert not torch.equal(run(NAME), run("cogent3_reward"))


# ── pipeline wiring (fake backbone, CPU) ───────────────────────────────

def test_pipeline_forces_the_band_and_scores_k_particles(monkeypatch):
    from diffucore.pipelines._anima import _ANIMA_SAMPLERS, _FLOW_AWARE_SAMPLERS, _onset_lowres_sigma
    assert NAME in _ANIMA_SAMPLERS and NAME in _FLOW_AWARE_SAMPLERS
    assert _onset_lowres_sigma(NAME, 0.0, True) == 0.9 and _onset_lowres_sigma(NAME, 0.0, False) == 0.0
    scores = []

    def reward(img):
        scores.append(float(img.mean()))
        return torch.tensor([scores[-1]])

    bb, _, info = _gen(monkeypatch, NAME, reward)
    assert len(scores) == 6 and info.reward["k"] == 6 and info.reward["scores"] == scores
    assert info.reward["pick"] == max(range(6), key=lambda j: (scores[j], -j))
    sizes = [hw for _, hw, _ in bb.calls]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16)


def test_pipeline_winner_zero_matches_plain_cogent3_pump_rate_with_the_band(monkeypatch):
    _, plain, _ = _gen(monkeypatch, "cogent3_pump_rate", lowres_sigma=0.9)
    vals = iter([10.0, 0, 0, 0, 0, 0])
    _, out, info = _gen(monkeypatch, NAME, lambda img: torch.tensor([next(vals)]))
    assert info.reward["pick"] == 0 and torch.equal(out, plain)


def test_pipeline_without_reward_fn_is_plain_cogent3_pump_rate(monkeypatch):
    bb, out, info = _gen(monkeypatch, NAME)
    bb2, out2, _ = _gen(monkeypatch, "cogent3_pump_rate")
    assert info.reward is None and torch.equal(out, out2)
    assert [c[:2] for c in bb.calls] == [c[:2] for c in bb2.calls]


def test_pipeline_gate_and_eta_knobs_reach_the_sampler(monkeypatch):
    got = {}
    real = K.get_sampler

    def spy(name):
        fn = real(name)
        return lambda *a, **kw: got.update(kw) or fn(*a, **kw)
    monkeypatch.setattr("diffucore.pipelines._anima.get_sampler", spy)
    _gen(monkeypatch, NAME, lambda img: torch.tensor([0.0]), eta_max=0.4, gate_reduce="per_channel")
    assert got["eta_max"] == 0.4 and got["gate_reduce"] == "per_channel" and "score_fn" in got
