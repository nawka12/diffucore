"""cogent3 stepper refactor (bit-exactness) and the best-of-k reward sampler."""
from __future__ import annotations

import math
from functools import partial

import pytest
import torch

from diffucore.sampling import samplers as K
from diffucore.sampling import schedules as S

# Frozen copy of sample_cogent3 from before it became a resumable stepper.
def _cogent3_reference(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eta_max: float = 1.0,
    s_noise: float = 1.0,
    pump_strength: float = 0.0,
    pump_end: float = 0.45,
    pump_span: float = 0.25,
    pump_h_ref: float | None = None,
    generator: torch.Generator | None = None,
    callback=None,
    model_type: str = "ve",
    shift: float = 1.0,
    gate_reduce: str = "all",
    gate_stats: list | None = None,
) -> torch.Tensor:
    K._validate_gate_reduce(gate_reduce)
    if len(sigmas) <= 1:
        return x
    if pump_strength > 0 and x.ndim != 4:
        raise ValueError(
            f"cogent3's coherence pump needs a 4-D [B, C, H, W] latent (the "
            f"structure tensor is a 2-D convolution); got rank {x.ndim}. FLUX "
            f"packs the latent into a [B, L, C·p²] token sequence, so use plain "
            f"cogent3 (pump_strength=0) there."
        )
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: K._half_log_snr(sigma, model_type)
    sigmas = K._offset_first_sigma_for_snr(sigmas, model_type, shift)
    denoised_1, denoised_2, denoised_3 = None, None, None
    h, h_1, h_2 = None, None, None
    _stats = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in)
        if callback is not None:
            callback(i, sigma, x, denoised)
        if bool(sigma_next == 0):
            x = denoised
        else:
            # σ-annealed ancestral fraction, as in cogent.
            sigma_frac = (float(sigma.clamp(max=1.0)) if model_type == "flow"
                          else float(sigma / (1.0 + sigma)))
            eta = eta_max * sigma_frac
            lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            _stats = None
            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                # ψ₂ bootstraps from ψ₁ until there is curvature history.
                _stats = {} if gate_stats is not None else None
                psi_1 = K._coherence_gate(denoised - denoised_1,
                                        denoised_1 - denoised_2, h,
                                        reduce=gate_reduce, stats_out=_stats)
                e_cur = denoised - 2.0 * denoised_1 + denoised_2
                psi_2 = (K._cogent3_curvature_gate(
                    e_cur, denoised_1 - 2.0 * denoised_2 + denoised_3,
                    reduce=gate_reduce)
                    if denoised_3 is not None else psi_1)
                x = x + psi_1 * (alpha_t * phi_2) * d1 - psi_2 * (alpha_t * phi_3) * d2
            elif h_1 is not None:
                rr = h_1 / h
                d = (denoised - denoised_1) / rr
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                _stats = {} if gate_stats is not None else None
                psi_1 = K._coherence_gate(denoised - denoised_1, None, h,
                                        reduce=gate_reduce, stats_out=_stats)
                x = x + psi_1 * (alpha_t * phi_2) * d
            if _stats is not None:
                _stats["step"] = i
                _stats["sigma"] = float(sigma)
                _stats["sigma_next"] = float(sigma_next)
                _stats["h"] = float(h)
                gate_stats.append(_stats)
            if eta > 0 and s_noise > 0:
                x = x + K._noise_like(x, generator) * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise
            # Coherence pump. No noise is drawn below pump_end, so
            # pump_strength=0 leaves the generator stream untouched.
            if pump_strength > 0:
                ramp = (1.0 if pump_span <= 0 else
                        min(1.0, max(0.0, (sigma_frac - pump_end) / pump_span)))
                if sigma_frac < pump_end:
                    ramp = 0.0
                if ramp > 0:
                    nu = pump_strength * float(sigma_next) * ramp
                    if pump_h_ref is not None:
                        nu *= math.sqrt(K._ou_step_var(float(h), eta) / K._ou_step_var(pump_h_ref, eta))
                    # Read coherence off the denoised prediction: at low σ the
                    # velocity is mostly residual noise.
                    c = K._structure_tensor_coherence(denoised.float(), multi_scale=True)
                    x = x + (nu * (1.0 - c)).to(x.dtype) * K._noise_like(x, generator)
        denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
        h_1, h_2 = h, h_1
    return x



def _model(target):
    def model(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        g = 1.0 / (1.0 + (4.0 * s) ** 2)
        return g * target + (1.0 - g) * 3.0 * torch.sin(3.0 * x)
    return model


def _problem(seed=5):
    torch.manual_seed(seed)
    return torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)


_GRIDS = {
    "flow16": (lambda: S.flow_matching_schedule(16, shift=3.0), dict(model_type="flow", shift=3.0)),
    "taper24": (lambda: S.flow_table_schedule("pump_taper", 3.0, 24), dict(model_type="flow", shift=3.0)),
    "ve14": (lambda: S.karras_schedule(14, 0.03, 14.6), dict(model_type="ve", shift=1.0)),
}
_VARIANTS = [
    {}, dict(gate_reduce="per_channel"), dict(eta_max=0.5, s_noise=0.9), dict(eta_max=0.0),
    dict(pump_strength=0.08), dict(pump_strength=0.08, pump_h_ref=K.PUMP_H_REF, gate_reduce="per_channel"),
]


@pytest.mark.parametrize("grid", sorted(_GRIDS))
@pytest.mark.parametrize("variant", range(len(_VARIANTS)))
def test_stepper_is_bit_exact_with_the_pre_refactor_sampler(grid, variant):
    make, kw = _GRIDS[grid]
    sigmas = make()
    target, x0 = _problem()
    if kw["model_type"] == "ve":
        x0 = x0 * float(sigmas[0])
    kw = {**kw, **_VARIANTS[variant]}
    runs = []
    for fn in (_cogent3_reference, K.sample_cogent3):
        xs, stats = [], []
        g = torch.Generator().manual_seed(11)
        out = fn(_model(target), x0.clone(), sigmas, generator=g, gate_stats=stats,
                 callback=lambda i, s, x, d: xs.append((i, x.clone(), d.clone())), **kw)
        runs.append((out, xs, stats, torch.randn(3, generator=g)))
    (a, xa, sa, ga), (b, xb, sb, gb) = runs
    assert torch.equal(a, b) and torch.equal(ga, gb)
    assert len(xa) == len(xb) and all(i == j and torch.equal(p, q) and torch.equal(u, v)
                                      for (i, p, u), (j, q, v) in zip(xa, xb))
    assert len(sa) == len(sb)
    for da, db in zip(sa, sb):
        assert da.keys() == db.keys()
        assert all(torch.equal(da[k], db[k]) if torch.is_tensor(da[k]) else da[k] == db[k] for k in da)


# ── reward sampler ─────────────────────────────────────────────────────

def _reward_setup(**over):
    target, x0 = _problem(7)
    kw = dict(model_type="flow", shift=3.0, eta_max=1.0, gate_reduce="per_channel")
    kw.update(over)
    return _model(target), x0, S.flow_table_schedule("pump_taper", 3.0, 24), kw


def _plain(model, x0, sigmas, kw, seed):
    return K.sample_cogent3(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(seed), **kw)


def _scorer_for(j, k):
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
    out = K.sample_cogent3_reward(model, x0.clone(), sigmas, k=6, score_fn=_scorer_for(j, 6),
                                  generator=torch.Generator().manual_seed(seed), info=info, **kw)
    assert info["pick"] == j and info["k"] == 6 and len(info["scores"]) == 6
    alone = _plain(model, x0, sigmas, kw, K.reward_particle_seed(seed, j))
    assert torch.equal(out, alone)


def test_winner_equals_that_particle_run_alone_with_pump():
    model, x0, sigmas, kw = _reward_setup(pump_strength=0.08, pump_h_ref=K.PUMP_H_REF)
    out = K.sample_cogent3_reward(model, x0.clone(), sigmas, k=4, score_fn=_scorer_for(2, 4),
                                  generator=torch.Generator().manual_seed(9), **kw)
    assert torch.equal(out, _plain(model, x0, sigmas, kw, K.reward_particle_seed(9, 2)))


def test_particles_stop_at_first_sigma_below_score_sigma_and_score_there():
    model, x0, sigmas, kw = _reward_setup()
    seen = []
    K.sample_cogent3_reward(model, x0.clone(), sigmas, k=3, score_sigma=0.90,
                            score_fn=lambda x, s: seen.append((x.clone(), s)) or 0.0,
                            generator=torch.Generator().manual_seed(1), **kw)
    i = next(n for n, s in enumerate(sigmas.tolist()) if s <= 0.90)
    assert len(seen) == 3 and all(s == pytest.approx(float(sigmas[i])) for _, s in seen)
    # the scored latent is the one a plain run has entering step i
    ref = {}
    K.sample_cogent3(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(1),
                     callback=lambda n, s, x, d: ref.__setitem__(n, x.clone()), **kw)
    assert torch.equal(seen[0][0], ref[i])


def test_ties_go_to_the_lowest_j_and_nan_never_wins():
    model, x0, sigmas, kw = _reward_setup()
    info = {}
    K.sample_cogent3_reward(model, x0.clone(), sigmas, k=5, score_fn=lambda x, s: 0.5,
                            generator=torch.Generator().manual_seed(2), info=info, **kw)
    assert info["pick"] == 0
    vals = iter([float("nan"), 1.0, 1.0, 0.2])
    K.sample_cogent3_reward(model, x0.clone(), sigmas, k=4, score_fn=lambda x, s: next(vals),
                            generator=torch.Generator().manual_seed(2), info=info, **kw)
    assert info["pick"] == 1


@pytest.mark.parametrize("override", [dict(k=1, score_fn=lambda x, s: 1.0), dict(k=6, score_fn=None)])
def test_k1_and_no_score_fn_are_plain_cogent3(override):
    model, x0, sigmas, kw = _reward_setup()
    info = {}
    steps = []
    out = K.sample_cogent3_reward(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(3),
                                  info=info, callback=lambda *a: steps.append(a[0]), **override, **kw)
    assert torch.equal(out, _plain(model, x0, sigmas, kw, 3))
    assert info == {} and steps == list(range(len(sigmas) - 1))


def test_deterministic_same_seed_same_pick_and_output():
    model, x0, sigmas, kw = _reward_setup()

    def run():
        info = {}
        # score = mean of the latent: a deterministic function of the particle
        out = K.sample_cogent3_reward(model, x0.clone(), sigmas, k=5, score_fn=lambda x, s: float(x.mean()),
                                      generator=torch.Generator().manual_seed(77), info=info, **kw)
        return out, info
    (a, ia), (b, ib) = run(), run()
    assert torch.equal(a, b) and ia == ib
    assert len(set(ia["scores"])) > 1                      # particles really differ


def test_callback_runs_every_particle_step_then_the_winners_rest():
    model, x0, sigmas, kw = _reward_setup()
    idx = []
    K.sample_cogent3_reward(model, x0.clone(), sigmas, k=3, score_fn=lambda x, s: 0.0,
                            generator=torch.Generator().manual_seed(4),
                            callback=lambda i, *a: idx.append(i), **kw)
    n, i = len(sigmas) - 1, K.reward_score_index(sigmas)
    assert len(idx) == 3 * i + (n - i) and idx[-(n - i):] == list(range(i, n))


def test_first_sigma_offset_is_applied_once_and_not_again_on_resume():
    model, x0, sigmas, kw = _reward_setup()
    plain = []
    K.sample_cogent3(model, x0.clone(), sigmas, generator=torch.Generator().manual_seed(5),
                     callback=lambda i, s, x, d: plain.append(float(s)), **kw)
    seen = []
    K.sample_cogent3_reward(model, x0.clone(), sigmas, k=2, score_fn=lambda x, s: 0.0,
                            generator=torch.Generator().manual_seed(5),
                            callback=lambda i, s, x, d: seen.append(float(s)), **kw)
    i = K.reward_score_index(sigmas)
    assert plain[0] < 1.0
    assert seen[:i] == plain[:i] and seen[i] == plain[0]    # particle 1 starts offset too
    assert seen[-(len(plain) - i):] == plain[i:]


def test_particle_zero_uses_the_given_generator_object():
    model, x0, sigmas, kw = _reward_setup()
    g = torch.Generator().manual_seed(6)
    K.sample_cogent3_reward(model, x0.clone(), sigmas, k=3, score_fn=lambda x, s: 0.0, generator=g, **kw)
    ref = torch.Generator().manual_seed(6)
    K.sample_cogent3(model, x0.clone(), sigmas, generator=ref, **kw)
    assert torch.equal(torch.randn(4, generator=g), torch.randn(4, generator=ref))


def test_score_index_and_degenerate_grids_fall_back_to_plain():
    sig = S.flow_matching_schedule(8, shift=3.0)
    assert K.reward_score_index(sig, 0.9) == next(n for n, s in enumerate(sig.tolist()) if s <= 0.9)
    assert K.reward_score_index(sig, 1.5) is None          # grid starts below it
    assert K.reward_score_index(sig, 0.0) is None          # only the final sigma reaches it
    model, x0, _, kw = _reward_setup()
    out = K.sample_cogent3_reward(model, x0.clone(), sig, k=4, score_sigma=1.5, score_fn=lambda x, s: 1 / 0,
                                  generator=torch.Generator().manual_seed(8), **kw)
    assert torch.equal(out, _plain(model, x0, sig, kw, 8))


def test_particle_seed_formula_and_registry():
    assert K.reward_particle_seed(1357911, 0) == 1357911
    assert K.reward_particle_seed(1357911, 3) == (1357911 * 1000003 + 7919 * 3) % 2 ** 63
    for name, kw in (("cogent3_reward", {"k": 6}),):
        fn = K.get_sampler(name)
        assert fn.func is K.sample_cogent3_reward and fn.keywords == kw
        assert K.REWARD_SAMPLERS[name] == 6
    assert set(K.REWARD_SAMPLERS) == {"cogent3_reward", "secant_anneal_reward", "cogent3_pump_onset_reward",
                                     "cogent3_pump_rate_reward"}
    assert {n for n in K.SAMPLERS if n.startswith("cogent3_reward")} == {"cogent3_reward"}


# ── pipeline wiring (fake backbone, CPU) ───────────────────────────────

from types import SimpleNamespace  # noqa: E402

from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

from diffucore.pipelines import _anima  # noqa: E402
from diffucore.pipelines._anima import (_dpmpp_2m_lookahead, _onset_lowres_sigma,  # noqa: E402
                                        anima_img2img, anima_text_to_image)
from diffucore.runtime import ConditioningCache, DevicePolicy  # noqa: E402


class _Bb(nn.Module):
    """Velocity depends on x, t and ctx; records (sigma, size, teacache stream) per call."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(4))
        self.calls = []

    def forward(self, x, t, ctx, teacache=None, skip_layers=(), prefix=None):
        self.calls.append((float(t.flatten()[0]), tuple(x.shape[-2:]), teacache))
        return 0.3 * x + 0.1 * t.view(-1, 1, 1, 1, 1) + 0.05 * ctx.mean()


def _bundle(monkeypatch, bb, out):
    cache = ConditioningCache()
    cache.put(("p", "n"), {"cond_ctx": torch.ones(1, 4, 8), "uncond_ctx": torch.zeros(1, 4, 8)})
    policy = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32, vae_dtype=torch.float32)
    vae = SimpleNamespace(process_out=lambda z: z, process_in=lambda z: z,
                          encode=lambda pixels: torch.zeros(1, 16, 16, 16))
    model = SimpleNamespace(spec=SimpleNamespace(architecture="anima", image_size=128), backbone=bb,
                            tokenizer=None, text_encoder=None, policy=policy, cond_cache=cache, vae=vae)
    monkeypatch.setattr(_anima, "vae_decode_safe", lambda vae, z, policy: (z, "full"))
    monkeypatch.setattr(_anima, "_to_pil", lambda image: (out.append(image.clone()), Image.new("RGB", (8, 8)))[1])
    return model


def _gen(monkeypatch, sampler, reward_fn=None, steps=12, **kw):
    bb, out = _Bb(), []
    _, info = anima_text_to_image(_bundle(monkeypatch, bb, out), "p", "n", steps=steps, cfg_scale=3.0,
                                  width=128, height=128, seed=5, sampler=sampler, scheduler="pump_taper",
                                  reward_fn=reward_fn, return_info=True, **kw)
    return bb, out[0], info


def test_reward_forces_the_half_res_band_only_with_a_reward_fn():
    f = lambda v, r: _onset_lowres_sigma("cogent3_reward", v, r)
    assert f(0.0, True) == 0.9 and f(None, True) == 0.9 and f(0.95, True) == 0.9 and f(0.85, True) == 0.85
    assert f(0.0, False) == 0.0 and f(0.95, False) == 0.95
    assert _onset_lowres_sigma("cogent3", 0.0, True) == 0.0


@pytest.mark.parametrize("sampler, k", [("cogent3_reward", 6)])
def test_pipeline_scores_k_particles_and_records_the_pick(monkeypatch, sampler, k):
    scores = []

    def reward(img):
        scores.append(float(img.mean()))
        return torch.tensor([scores[-1]])

    _, _, info = _gen(monkeypatch, sampler, reward)
    sig = _anima.flow_table_schedule("pump_taper", 3.0, 12, device="cpu", dtype=torch.float32)
    assert len(scores) == k and len(set(scores)) > 1
    assert info.reward["k"] == k and info.reward["scores"] == scores
    assert info.reward["pick"] == max(range(k), key=lambda j: (scores[j], -j))
    assert info.reward["score_step"] == K.reward_score_index(sig)


def test_particle_zero_winner_matches_plain_cogent3_with_the_band(monkeypatch):
    _, plain, _ = _gen(monkeypatch, "cogent3", lowres_sigma=0.9)
    vals = iter([10.0, 0, 0, 0, 0, 0])
    _, out, info = _gen(monkeypatch, "cogent3_reward", lambda img: torch.tensor([next(vals)]))
    assert info.reward["pick"] == 0 and torch.equal(out, plain)


def test_half_res_band_spans_particles_and_lookahead_then_full_res(monkeypatch):
    bb, _, info = _gen(monkeypatch, "cogent3_reward", lambda img: torch.tensor([0.0]))
    i = info.reward["score_step"]
    sizes = [hw for _, hw, _ in bb.calls]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16)
    assert sizes == sorted(sizes)                       # every half-res call precedes the full-res ones
    assert sizes.count((8, 8)) >= 6 * 2 * i + 6 * 6     # particle steps (CFG) + lookahead calls
    assert sizes.count((16, 16)) < 2 * (12 - i) + 1


def test_no_reward_fn_is_plain_cogent3_with_no_forced_band(monkeypatch):
    bb, out, info = _gen(monkeypatch, "cogent3_reward")
    bb2, out2, _ = _gen(monkeypatch, "cogent3")
    assert info.reward is None
    assert torch.equal(out, out2)
    assert {hw for _, hw, _ in bb.calls} == {(16, 16)} == {hw for _, hw, _ in bb2.calls}


def test_progress_only_moves_forward_and_cancel_lands_within_a_particle(monkeypatch):
    seen = []
    _gen(monkeypatch, "cogent3_reward", lambda img: torch.tensor([0.0]),
         progress_callback=lambda n, total: seen.append((n, total)))
    steps = [n for n, _ in seen]
    assert steps == sorted(steps) and len({t for _, t in seen}) == 1
    sig = _anima.flow_table_schedule("pump_taper", 3.0, 12, device="cpu", dtype=torch.float32)
    i = K.reward_score_index(sig)
    assert seen[-1] == (6 * i + 12 - i,) * 2
    assert len(seen) > steps[-1]                       # lookahead ticks re-send the count

    class Stop(BaseException):
        pass

    def stop_on_third(n, total):
        if len(calls) >= 2:
            raise Stop
        calls.append(n)
    calls = []
    bb = _Bb()
    with pytest.raises(Stop):
        anima_text_to_image(_bundle(monkeypatch, bb, []), "p", "n", steps=12, cfg_scale=3.0, width=128,
                            height=128, seed=5, sampler="cogent3_reward", scheduler="pump_taper",
                            reward_fn=lambda img: torch.tensor([0.0]), progress_callback=stop_on_third)
    assert len(bb.calls) == 3 * 2                      # three particle steps (cond + uncond each), not a whole particle


def test_reward_is_refused_with_the_ewma_forecast_but_plain_cogent3_is_not(monkeypatch):
    with pytest.raises(ValueError, match="forecast"):
        _gen(monkeypatch, "cogent3_reward", lambda img: torch.tensor([0.0]), forecast_ratio=0.3)
    _gen(monkeypatch, "cogent3_reward", forecast_ratio=0.3, forecast_sigma=0.5)      # no reward_fn: plain


def test_teacache_is_never_touched_by_particles_or_lookahead(monkeypatch):
    made = []
    real = _anima._make_teacache
    monkeypatch.setattr(_anima, "_make_teacache", lambda *a, **k: made.append(real(*a, **k)) or made[-1])
    bb, _, info = _gen(monkeypatch, "cogent3_reward", lambda img: torch.tensor([0.0]), teacache_thresh=0.4)
    streams = {id(s) for s in made[0] if s is not None}
    half = [tc for _, hw, tc in bb.calls if hw == (8, 8)]
    full = [tc for _, hw, tc in bb.calls if hw == (16, 16)]
    assert half and all(tc is None for tc in half)
    assert full and all(id(tc) in streams for tc in full)
    assert made[0][0].calls == 0                       # the fake backbone never feeds it; nothing else did


def test_img2img_reward_names_are_plain_cogent3_without_a_band(monkeypatch):
    def run(sampler):
        bb, out = _Bb(), []
        anima_img2img(_bundle(monkeypatch, bb, out), "p", Image.new("RGB", (128, 128), (9, 9, 9)), "n",
                      strength=0.8, steps=10, cfg_scale=3.0, width=128, height=128, seed=5,
                      sampler=sampler, scheduler="pump_taper")
        return bb, out[0]
    (b1, o1), (b2, o2) = run("cogent3_reward"), run("cogent3")
    assert torch.equal(o1, o2) and {hw for _, hw, _ in b1.calls} == {(16, 16)}


def test_lookahead_is_deterministic_dpmpp_2m_ending_at_the_final_x0():
    calls = []

    def den(x, s):
        calls.append(float(s[0]))
        return torch.full_like(x, 0.25)
    x = torch.randn(1, 4, 6, 6)
    x0 = _dpmpp_2m_lookahead(den, x, 0.9, 6, 3.0)
    assert len(calls) == 6 and calls[0] == pytest.approx(0.9) and calls == sorted(calls, reverse=True)
    assert torch.equal(x0, torch.full_like(x, 0.25))
