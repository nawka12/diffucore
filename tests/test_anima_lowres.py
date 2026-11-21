"""Anima half-res layout (``lowres_sigma``): the backbone wrapper's formula, its
off path, odd-size pad/crop, the TeaCache kwarg, and the pipeline wiring. Offline, CPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from diffucore.pipelines import _anima
from diffucore.pipelines._anima import (_anima_lowres, _lowres_backbone, _lowres_cond_sigma,
                                        _onset_lowres_sigma, anima_text_to_image)
from diffucore.runtime import ConditioningCache, DevicePolicy


class _Toy:
    """A backbone with a spatially varying velocity; records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, t, ctx, *a, **kw):
        self.calls.append((x.shape, float(t.flatten()[0]), a, kw))
        return 0.3 * x + 0.2 * torch.roll(x, 1, dims=-1) - 0.1 * t.view(-1, 1, 1, 1, 1)


def _x(h=8, w=12, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 16, 1, h, w, generator=g).to(dtype)


def _reference(bb, x, t, ctx, k=2):
    """The validated harness math (scripts/ab_lowres_band.py, mode "lr"), verbatim."""
    s = float(t.flatten()[0])
    xf = x.squeeze(2).float()
    amp = (1.0 - s) + s / k
    s2 = (s / k) / amp
    xl = F.avg_pool2d(xf, k) / amp
    vl = bb(xl.to(x.dtype).unsqueeze(2), torch.full_like(t, s2), ctx).squeeze(2).float()
    x0 = F.interpolate(xl - s2 * vl, scale_factor=k, mode="bilinear", align_corners=False)
    return ((xf - x0) / s).to(x.dtype).unsqueeze(2)


def test_off_returns_the_backbone_itself():
    bb = _Toy()
    assert _lowres_backbone(bb, 0.0) is bb
    assert _lowres_backbone(bb, -1.0) is bb


def test_below_the_band_calls_the_backbone_unchanged():
    bb = _Toy()
    sentinel = object()
    out = _lowres_backbone(lambda *a, **kw: (bb(*a, **kw), sentinel)[1], 0.9)(
        _x(), torch.tensor([0.5]), "ctx", teacache="tc", skip_layers=(1,))
    assert out is sentinel
    shape, sigma, _, kw = bb.calls[0]
    assert shape == _x().shape and sigma == 0.5 and kw == {"teacache": "tc", "skip_layers": (1,)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("sigma", [0.95, 0.999, 1.0])
def test_band_matches_the_harness_formula(dtype, sigma):
    x, t = _x(dtype=dtype), torch.tensor([sigma], dtype=dtype)
    got = _lowres_backbone(_Toy(), 0.9)(x, t, None)
    want = _reference(_Toy(), x, t, None)
    assert got.dtype == dtype and got.shape == x.shape
    assert torch.equal(got, want)


def test_band_runs_the_backbone_at_half_size_and_the_matching_sigma():
    bb = _Toy()
    s = 0.95
    _lowres_backbone(bb, 0.9)(_x(), torch.tensor([s]), None)
    shape, sigma, _, _ = bb.calls[0]
    amp = (1 - s) + s / 2
    assert shape == (1, 16, 1, 4, 6) and sigma == pytest.approx((s / 2) / amp)


@pytest.mark.parametrize("h, w", [(10, 14), (8, 14), (10, 12), (6, 6)])
def test_non_divisible_latents_pad_and_crop_to_the_original_shape(h, w):
    bb = _Toy()
    x = _x(h, w)
    out = _lowres_backbone(bb, 0.5)(x, torch.tensor([0.9]), None)
    assert out.shape == x.shape
    ph, pw = -h % 4, -w % 4
    assert bb.calls[0][0][-2:] == ((h + ph) // 2, (w + pw) // 2)
    assert torch.isfinite(out).all()


def test_divisible_latents_run_no_pad_op(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("F.pad must not run on a divisible latent")
    monkeypatch.setattr(_anima.F, "pad", boom)
    _lowres_backbone(_Toy(), 0.5)(_x(8, 12), torch.tensor([0.9]), None)


def test_teacache_is_dropped_only_in_the_band_and_other_kwargs_pass_through():
    bb = _Toy()
    wrapped = _lowres_backbone(bb, 0.9)
    wrapped(_x(), torch.tensor([0.95]), None, teacache="tc", skip_layers=(1,), prefix="p")
    wrapped(_x(), torch.tensor([0.5]), None, teacache="tc")
    wrapped(_x(), torch.tensor([0.95]), None, skip_layers=(2,))
    assert bb.calls[0][3] == {"teacache": None, "skip_layers": (1,), "prefix": "p"}
    assert bb.calls[1][3] == {"teacache": "tc"}
    assert bb.calls[2][3] == {"skip_layers": (2,)}


def _model(monkeypatch, backbone, **policy_flags):
    cond, uncond = torch.ones(1, 4, 8), torch.zeros(1, 4, 8)
    cache = ConditioningCache()
    cache.put(("p", "n"), {"cond_ctx": cond, "uncond_ctx": uncond})
    policy = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32, vae_dtype=torch.float32)
    if policy_flags:
        policy = SimpleNamespace(**{"cuda_graphs": False, "compile": False, "device": torch.device("cpu"),
                                    "compute_dtype": torch.float32, **policy_flags})
    model = SimpleNamespace(spec=SimpleNamespace(architecture="anima", image_size=64), backbone=backbone,
                            tokenizer=None, text_encoder=None, policy=policy, cond_cache=cache,
                            vae=SimpleNamespace(process_out=lambda z: z))
    monkeypatch.setattr(_anima, "vae_decode_safe", lambda vae, z, policy: (z, "full"))
    monkeypatch.setattr(_anima, "_to_pil", lambda image: Image.new("RGB", (64, 64)))
    return model


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(4))
        self.shapes = []

    def forward(self, x, t, ctx, teacache=None, skip_layers=(), prefix=None):
        self.shapes.append((float(t.flatten()[0]), tuple(x.shape[-2:])))
        return torch.zeros_like(x)


@pytest.mark.parametrize("sampler", ["euler", "dpmpp_2m"])
def test_pipeline_runs_half_res_only_while_sigma_is_in_the_band(monkeypatch, sampler):
    bb = _Recorder()
    anima_text_to_image(_model(monkeypatch, bb), "p", "n", steps=6, cfg_scale=3.0, width=128, height=128,
                        seed=1, sampler=sampler, lowres_sigma=0.8)
    # a half-res call records its remapped σ' (< σ), a full-res call the real σ
    sizes = [hw for _, hw in bb.shapes]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16)
    assert sizes == sorted(sizes)                  # half-res steps first, then full res, no interleave
    assert all(s < 0.8 for s, hw in bb.shapes if hw == (16, 16))


def test_pipeline_off_never_downsamples(monkeypatch):
    bb = _Recorder()
    anima_text_to_image(_model(monkeypatch, bb), "p", "n", steps=4, cfg_scale=3.0, width=128, height=128, seed=1)
    assert {hw for _, hw in bb.shapes} == {(16, 16)}


@pytest.mark.parametrize("flags, what", [(dict(cuda_graphs=True), "CUDA Graphs"), (dict(compile=True), "torch.compile")])
def test_refused_on_cuda_graphs_and_compile(monkeypatch, flags, what):
    model = _model(monkeypatch, _Recorder(), **flags)
    with pytest.raises(ValueError, match=what):
        anima_text_to_image(model, "p", "n", steps=2, width=64, height=64, seed=1, lowres_sigma=0.9)
    assert _anima_lowres(model, 0.0) == 0.0          # off is never refused
    with pytest.raises(ValueError, match="lowres_sigma"):
        _anima_lowres(model, 1.5)


def test_onset_sampler_switches_the_half_res_band_on_and_never_below_it():
    f = lambda v: _onset_lowres_sigma("cogent3_pump_onset", v)
    assert f(0.0) == 0.9 and f(None) == 0.9
    assert f(0.95) == 0.9 and f(0.85) == 0.85
    assert _onset_lowres_sigma("cogent3_pump_rate", 0.0) == 0.0
    assert _onset_lowres_sigma("dpmpp_2m", 0.95) == 0.95
    assert f(1.5) == 1.5                                  # left for _anima_lowres to refuse
    g = lambda v: _onset_lowres_sigma("secant_onset", v)
    assert g(0.0) == 0.9 and g(None) == 0.9
    assert g(0.95) == 0.9 and g(0.85) == 0.85
    assert _onset_lowres_sigma("secant_anneal", 0.0) == 0.0


def test_pipeline_onset_sampler_runs_half_res_and_keeps_the_refusals(monkeypatch):
    bb = _Recorder()
    anima_text_to_image(_model(monkeypatch, bb), "p", "n", steps=6, cfg_scale=3.0, width=128, height=128,
                        seed=1, sampler="cogent3_pump_onset")
    sizes = [hw for _, hw in bb.shapes]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16)
    assert len(sizes) > 6                                 # the band was sub-stepped
    with pytest.raises(ValueError, match="CUDA Graphs"):
        anima_text_to_image(_model(monkeypatch, _Recorder(), cuda_graphs=True), "p", "n", steps=2,
                            width=64, height=64, seed=1, sampler="cogent3_pump_onset")


def test_pipeline_secant_onset_runs_half_res(monkeypatch):
    bb = _Recorder()
    anima_text_to_image(_model(monkeypatch, bb), "p", "n", steps=6, cfg_scale=3.0, width=128, height=128,
                        seed=1, sampler="secant_onset")
    sizes = [hw for _, hw in bb.shapes]
    assert sizes[0] == (8, 8) and sizes[-1] == (16, 16)
    assert len(sizes) > 6


_CAL_BAND = (0.9, 0.95, 0.98, 0.99, 0.997)


def test_lowres_cond_sigma_sits_below_the_pooled_sigma():
    assert _lowres_cond_sigma(1.0, 2) == 1.0
    pooled = [(s / 2) / ((1 - s) + s / 2) for s in _CAL_BAND]
    cal = [_lowres_cond_sigma(s, 2) for s in _CAL_BAND]
    assert all(c < p for c, p in zip(cal, pooled))
    assert cal == sorted(cal) and len(set(cal)) == len(cal)
    assert _lowres_cond_sigma(0.9, 2) == pytest.approx(1 / (1 + math.exp(math.log(0.1 / 0.9) + 1.2 * math.log(2))), abs=1e-9)


def _half_res_t(monkeypatch, **kw):
    bb = _Recorder()
    anima_text_to_image(_model(monkeypatch, bb), "p", "n", steps=6, cfg_scale=1.0, width=128, height=128,
                        seed=1, **kw)
    return bb.shapes


def test_pipeline_calibrated_half_res_calls_see_a_lower_sigma(monkeypatch):
    plain = _half_res_t(monkeypatch, lowres_sigma=0.9)
    cal = _half_res_t(monkeypatch, lowres_sigma=0.9, lowres_calibrated=True)
    half = [(a, b) for (a, hw), (b, _) in zip(plain, cal) if hw == (8, 8)]
    assert half and [hw for _, hw in plain] == [hw for _, hw in cal]
    assert all(b <= a for a, b in half)            # both map σ = 1 to 1
    assert any(b < a for a, b in half)
    assert {a for a, _ in half} != {b for _, b in half}
    assert [t for t, hw in plain if hw == (16, 16)] == [t for t, hw in cal if hw == (16, 16)]


def test_calibration_is_a_no_op_while_the_band_is_off(monkeypatch):
    off = _half_res_t(monkeypatch, lowres_sigma=0.0)
    cal = _half_res_t(monkeypatch, lowres_sigma=0.0, lowres_calibrated=True)
    assert cal == off and {hw for _, hw in cal} == {(16, 16)}
