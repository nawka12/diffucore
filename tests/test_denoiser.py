import torch

from diffucore.sampling.parameterization import EpsScaling, DiscreteSchedule, make_betas
from diffucore.sampling.denoiser import ModelDenoiser, CFGDenoiser


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
