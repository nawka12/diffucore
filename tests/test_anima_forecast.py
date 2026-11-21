"""Anima EWMA forecast (``forecast_ratio``): the step plan, the off path, the model-call
budget on the closed-form euler loop and a registry sampler, and inpaint pinning. Offline, CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from diffucore.pipelines import _anima
from diffucore.pipelines._anima import _Forecast, _anima_forecast, anima_img2img, anima_text_to_image
from diffucore.runtime import ConditioningCache, DevicePolicy
from diffucore.sampling import flow_matching_schedule

BETA = 0.8


def _plan(sigmas, ratio, limit=0.75):
    steps = _Forecast(sigmas, ratio, limit).steps
    return "".join("F" if i in steps else "M" for i in range(len(sigmas) - 1))


def test_plan_spreads_the_budget_over_the_eligible_steps():
    ten = torch.linspace(0.7, 0.0, 11)
    assert _plan(ten, 0.4) == "MMFMFMFMFM"
    # steps 0-4 sit at or above the limit, 5-14 below
    fifteen = torch.tensor([0.95, 0.9, 0.85, 0.8, 0.75] + [0.7 - 0.05 * k for k in range(10)] + [0.0])
    assert _plan(fifteen, 0.4) == "MMMMMMFFMFFMFFM"


def test_plan_never_touches_the_first_or_last_step_and_clamps_the_budget():
    assert _plan(torch.linspace(0.7, 0.0, 11), 0.99).startswith("MF") and _plan(torch.linspace(0.7, 0.0, 11), 0.99).endswith("FM")
    assert _plan(torch.linspace(0.7, 0.0, 3), 0.9) == "MM"           # no step in 1..n-2
    assert _plan(torch.linspace(0.7, 0.0, 11), 0.4, limit=0.1) == "M" * 10   # nothing below the limit
    assert _plan(torch.linspace(1.0, 0.0, 11), 0.4, limit=1.0).count("F") == 4   # strict: sigma 1.0 is not below 1.0


def test_off_returns_no_plan_and_bad_values_are_refused():
    assert _anima_forecast(torch.linspace(0.7, 0.0, 11), 0.0, 0.75) is None
    assert _anima_forecast(torch.linspace(0.7, 0.0, 11), 0.4, 0.75) is not None
    for ratio, limit in ((-0.1, 0.75), (1.0, 0.75), (0.4, 0.0), (0.4, 1.5)):
        with pytest.raises(ValueError, match="forecast_"):
            _anima_forecast(torch.linspace(0.7, 0.0, 11), ratio, limit)


def test_ewma_level_and_registry_matching():
    f = _Forecast(torch.linspace(0.7, 0.0, 11), 0.4, 0.75)          # forecast steps 2, 4, 6, 8
    assert not f.take(float(torch.linspace(0.7, 0.0, 11)[2]))       # nothing observed yet
    a, b = torch.ones(2, 2), torch.full((2, 2), 3.0)
    f.observe(a)
    f.observe(b)
    assert torch.allclose(f.level, BETA * b + (1 - BETA) * a)
    sigma = float(torch.linspace(0.7, 0.0, 11)[4])
    assert not f.take(0.5 * sigma)                                  # an off-schedule σ calls the model
    assert f.take(sigma * (1 + 1e-7)) and not f.take(sigma)         # first call at a planned σ only
    assert f.used == 1


class _Backbone(nn.Module):
    """Cond calls see ctx = 1, uncond ctx = 0; the velocity depends on x, t and ctx."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(4))
        self.calls = []

    def forward(self, x, t, ctx, teacache=None, skip_layers=(), prefix=None):
        self.calls.append((float(t.flatten()[0]), float(ctx.mean())))
        return 0.3 * x + 0.1 * t.view(-1, 1, 1, 1, 1) + 0.05 * ctx.mean()

    def cond_sigmas(self):
        return [t for t, c in self.calls if c == 1.0]


def _model(monkeypatch, backbone, out, size=128):
    cache = ConditioningCache()
    cache.put(("p", "n"), {"cond_ctx": torch.ones(1, 4, 8), "uncond_ctx": torch.zeros(1, 4, 8)})
    policy = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32, vae_dtype=torch.float32)
    h = size // 8
    vae = SimpleNamespace(process_out=lambda z: z, process_in=lambda z: z,
                          encode=lambda pixels: torch.zeros(1, 16, h, h))
    model = SimpleNamespace(spec=SimpleNamespace(architecture="anima", image_size=size), backbone=backbone,
                            tokenizer=None, text_encoder=None, policy=policy, cond_cache=cache, vae=vae)
    monkeypatch.setattr(_anima, "vae_decode_safe", lambda vae, z, policy: (z, "full"))
    monkeypatch.setattr(_anima, "_to_pil", lambda image: (out.append(image.clone()), Image.new("RGB", (size, size)))[1])
    return model


def _run(monkeypatch, sampler, steps=10, **kw):
    bb, out, previews = _Backbone(), [], []
    anima_text_to_image(_model(monkeypatch, bb, out), "p", "n", steps=steps, cfg_scale=3.0, width=128, height=128,
                        seed=1, sampler=sampler, preview_callback=lambda d: previews.append(d.clone()), **kw)
    return bb, out[0], previews


def _ewma(previews, planned, region=...):
    """Replay the average over the model steps' x0 and return the level each step saw."""
    level, levels = None, []
    for i, d in enumerate(previews):
        d = d[region]
        levels.append(level)
        if i not in planned:
            level = d if level is None else BETA * d + (1 - BETA) * level
    return levels


@pytest.mark.parametrize("sampler", ["euler", "euler_ancestral"])
def test_ratio_zero_is_bit_identical_to_not_passing_the_kwargs(monkeypatch, sampler):
    bb0, x0, _ = _run(monkeypatch, sampler)
    bb1, x1, _ = _run(monkeypatch, sampler, forecast_ratio=0.0, forecast_sigma=0.5)
    assert torch.equal(x0, x1) and bb0.calls == bb1.calls and len(bb0.calls) == 20


@pytest.mark.parametrize("sampler", ["euler", "euler_ancestral"])
def test_forecast_steps_skip_the_model_and_feed_the_ewma_as_x0(monkeypatch, sampler, capsys):
    steps, planned = 10, {2, 4, 6, 8}
    bb0, x0, _ = _run(monkeypatch, sampler)
    bb, x, previews = _run(monkeypatch, sampler, forecast_ratio=0.4, forecast_sigma=1.0)
    sigmas = [float(s) for s in flow_matching_schedule(steps, shift=3.0)]
    assert len(bb0.cond_sigmas()) == steps and len(bb.cond_sigmas()) == steps - len(planned)
    assert len([c for _, c in bb.calls if c == 0.0]) == steps - len(planned)   # uncond skipped too
    assert bb.cond_sigmas() == pytest.approx([sigmas[i] for i in range(steps) if i not in planned])
    assert bb.cond_sigmas()[0] == pytest.approx(sigmas[0]) and bb.cond_sigmas()[-1] == pytest.approx(sigmas[-1 - 1])
    assert not torch.equal(x0, x)
    for i, (d, level) in enumerate(zip(previews, _ewma(previews, planned))):
        if i in planned:
            assert torch.allclose(d, level, atol=1e-6)
    assert "[forecast] 4/10 steps forecast" in capsys.readouterr().out


def test_euler_forecast_step_is_the_exact_flow_step_toward_the_level(monkeypatch):
    bb, out = _Backbone(), []
    seen = []
    orig = bb.forward

    def spy(x, t, ctx, **kw):
        seen.append(x.clone())
        return orig(x, t, ctx, **kw)

    bb.forward = spy
    anima_text_to_image(_model(monkeypatch, bb, out), "p", "n", steps=10, cfg_scale=1.0, width=128, height=128,
                        seed=1, forecast_ratio=0.4, forecast_sigma=1.0)
    # cfg 1: one call per model step, so calls 0,1 are steps 0,1; step 2 is skipped and step 3 follows it
    sigmas = [float(s) for s in flow_matching_schedule(10, shift=3.0)]
    def vel(x, s):
        return 0.3 * x + 0.1 * s + 0.05

    x1 = seen[1]
    level = BETA * (x1 - sigmas[1] * vel(x1, sigmas[1])) \
        + (1 - BETA) * (seen[0] - sigmas[0] * vel(seen[0], sigmas[0]))
    x2 = x1 + (sigmas[2] - sigmas[1]) * vel(x1, sigmas[1])
    r = sigmas[3] / sigmas[2]
    assert torch.allclose(seen[2], r * x2 + (1 - r) * level, atol=1e-5)


def test_inpaint_pins_the_forecast_x0_like_any_other(monkeypatch):
    bb, out, previews = _Backbone(), [], []
    mask = Image.new("L", (128, 128), 0)
    mask.paste(255, (0, 0, 64, 128))                                  # left half repaints
    anima_img2img(_model(monkeypatch, bb, out), "p", Image.new("RGB", (128, 128), (9, 9, 9)), "n",
                  mask_image=mask, strength=1.0, steps=10, cfg_scale=3.0, width=128, height=128, seed=1,
                  forecast_ratio=0.4, forecast_sigma=1.0, preview_callback=lambda d: previews.append(d.clone()))
    planned = {2, 4, 6, 8}
    assert len(bb.cond_sigmas()) == 10 - len(planned)
    assert all(torch.equal(d[..., 10:], torch.zeros_like(d[..., 10:])) for d in previews)   # keep side = z0
    region = (..., slice(0, 6))
    for i, (d, level) in enumerate(zip(previews, _ewma(previews, planned, region))):
        if i in planned:
            assert torch.allclose(d[region], level, atol=1e-6)
