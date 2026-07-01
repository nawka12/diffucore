import math

import pytest
import torch

from diffucore.sampling.parameterization import EpsScaling, DiscreteSchedule, make_betas
from diffucore.sampling.denoiser import (
    CFGDenoiser,
    ModelDenoiser,
    guidance_interval_bounds,
)


def _schedule():
    return DiscreteSchedule(make_betas("scaled_linear", 1000))


def test_model_denoiser_zero_eps_returns_input():
    backbone = lambda model_input, t, **c: torch.zeros_like(model_input)
    den = ModelDenoiser(backbone, EpsScaling(), _schedule())
    x = torch.randn(1, 4, 8, 8)
    out = den(x, torch.tensor([1.5]))
    assert torch.allclose(out, x, atol=1e-5)  # eps = 0  ->  x0 = x


def test_model_denoiser_eps_formula():
    eps = torch.randn(1, 4, 8, 8)
    backbone = lambda model_input, t, **c: eps  # ignores input, fixed prediction
    den = ModelDenoiser(backbone, EpsScaling(), _schedule())
    x = torch.randn(1, 4, 8, 8)
    sigma = torch.tensor([1.3])
    expected = x - sigma.view(-1, 1, 1, 1) * eps  # eps-scaling: x0 = x - σ·eps
    assert torch.allclose(den(x, sigma), expected, atol=1e-5)


def test_cfg_endpoints_and_linearity():
    # Backbone prediction is a constant determined by the conditioning 'bias'.
    def backbone(model_input, t, bias):
        return torch.full_like(model_input, bias)

    den = ModelDenoiser(backbone, EpsScaling(), _schedule())
    x = torch.randn(1, 4, 8, 8)
    sigma = torch.tensor([2.0])

    uncond_x0 = den(x, sigma, bias=0.0)
    cond_x0 = den(x, sigma, bias=1.0)

    cfg0 = CFGDenoiser(den, cond={"bias": 1.0}, uncond={"bias": 0.0}, scale=0.0)
    cfg1 = CFGDenoiser(den, cond={"bias": 1.0}, uncond={"bias": 0.0}, scale=1.0)
    cfg2 = CFGDenoiser(den, cond={"bias": 1.0}, uncond={"bias": 0.0}, scale=2.0)

    assert torch.allclose(cfg0(x, sigma), uncond_x0, atol=1e-5)
    assert torch.allclose(cfg1(x, sigma), cond_x0, atol=1e-5)
    assert torch.allclose(cfg2(x, sigma), uncond_x0 + 2.0 * (cond_x0 - uncond_x0), atol=1e-5)


def test_guidance_interval_bounds_mapping():
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])  # 5 steps
    assert guidance_interval_bounds(sigmas, 0.0, 1.0) == (-math.inf, math.inf)
    lo, hi = guidance_interval_bounds(sigmas, 0.2, 0.8)
    assert hi == 8.0  # step 1 of 5 → sigmas[1]
    assert lo == 2.0  # step 4 of 5 → sigmas[4]
    with pytest.raises(ValueError):
        guidance_interval_bounds(sigmas, 0.8, 0.2)


def test_cfg_guidance_interval_skips_uncond():
    calls = []

    def backbone(model_input, t, bias):
        calls.append(1)
        return torch.full_like(model_input, bias)

    den = ModelDenoiser(backbone, EpsScaling(), _schedule())
    x = torch.randn(1, 4, 8, 8)
    cond, uncond = {"bias": 1.0}, {"bias": 0.0}

    banded = CFGDenoiser(den, cond, uncond, scale=7.0, sigma_lo=1.0, sigma_hi=5.0)
    plain = CFGDenoiser(den, cond, uncond, scale=7.0)
    cond_only = CFGDenoiser(den, cond, uncond, scale=1.0)

    # In-band sigma: guided estimate, identical to unbounded CFG (two forwards).
    calls.clear()
    in_band = banded(x, torch.tensor([2.0]))
    assert len(calls) == 2
    assert torch.allclose(in_band, plain(x, torch.tensor([2.0])))

    # Above hi and at lo (exclusive): cond-only estimate, single forward each.
    for sigma in (torch.tensor([7.0]), torch.tensor([1.0])):
        calls.clear()
        out = banded(x, sigma)
        assert len(calls) == 1
        assert torch.allclose(out, cond_only(x, sigma))


def test_cfg_rescale():
    # Distinct cond/uncond predictions so the guided estimate has a different std
    # than the conditioned one (otherwise rescale is a no-op).
    cond_val = torch.randn(1, 4, 8, 8)
    uncond_val = torch.randn(1, 4, 8, 8) * 0.3
    backbone = lambda model_input, t, val: val  # noqa: E731  (returns the conditioning tensor)
    den = ModelDenoiser(backbone, EpsScaling(), _schedule())
    x = torch.zeros(1, 4, 8, 8)
    sigma = torch.tensor([1.0])
    cond, uncond = {"val": cond_val}, {"val": uncond_val}

    plain = CFGDenoiser(den, cond, uncond, scale=7.0, rescale=0.0)(x, sigma)
    # rescale=0 is plain CFG; rescale=1 renormalizes the guided x0 to the cond std.
    full = CFGDenoiser(den, cond, uncond, scale=7.0, rescale=1.0)(x, sigma)

    den_cond = den(x, sigma, **cond)
    assert not torch.allclose(full, plain)                       # rescale changed the result
    assert torch.allclose(full.std(), den_cond.std(), atol=1e-5)  # std matched to cond's
    # A partial rescale lands between plain and the fully-rescaled estimate.
    half = CFGDenoiser(den, cond, uncond, scale=7.0, rescale=0.5)(x, sigma)
    assert torch.allclose(half, 0.5 * full + 0.5 * plain, atol=1e-5)
