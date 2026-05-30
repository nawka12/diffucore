"""Tests for the inpainting pipeline.

The mask preprocessing, the masked-denoiser mechanism, and the pixel composite
all run on CPU with no checkpoint — the sampler-level test exercises the core
claim (keep region pinned to z0 through the whole loop) directly. The end-to-end
test needs a real checkpoint and is skipped when absent, mirroring the other
pipeline tests.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffucore.pipelines.inpaint import Inpaint, preprocess_mask
from diffucore.sampling import MaskedDenoiser, karras_schedule, sample_euler

_MODELS = Path(__file__).resolve().parents[1] / "models"
CKPTS = {
    "sd15": Path(os.environ.get("DIFFUCORE_SD15_CKPT", _MODELS / "v1-5-pruned-emaonly.safetensors")),
    "sdxl": Path(os.environ.get("DIFFUCORE_SDXL_CKPT", _MODELS / "sdxl.safetensors")),
}


# --- mask preprocessing -- CPU-runnable --------------------------------------

def test_preprocess_mask_shape_and_convention():
    """White (255) -> 1 (repaint), black (0) -> 0 (keep), at latent resolution."""
    white = preprocess_mask(Image.new("L", (64, 48), color=255), width=64, height=48)
    assert white.shape == (1, 1, 6, 8)  # height//8, width//8
    assert torch.equal(white, torch.ones_like(white))
    black = preprocess_mask(Image.new("L", (64, 48), color=0), width=64, height=48)
    assert torch.equal(black, torch.zeros_like(black))


# --- masked-denoiser mechanism -- CPU-runnable -------------------------------

class _ConstantDenoiser:
    """A stub denoiser: always reports ``target`` as the x0 estimate, regardless of
    input — stands in for 'the model wants this in the repaint region'."""

    def __init__(self, target):
        self.target = target

    def __call__(self, x, sigma):
        return self.target


def test_masked_denoiser_blends_estimate_and_z0():
    z0 = torch.full((1, 1, 2, 2), 5.0)
    target = torch.zeros(1, 1, 2, 2)
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])  # top row repaint, bottom keep
    out = MaskedDenoiser(_ConstantDenoiser(target), z0, mask)(torch.empty_like(z0), torch.tensor(1.0))
    assert torch.equal(out[..., 0, :], target[..., 0, :])  # repaint -> model estimate
    assert torch.equal(out[..., 1, :], z0[..., 1, :])      # keep -> original z0


def test_masked_denoiser_pins_keep_region_through_sampling():
    """The whole point: after a full Euler descent the keep region equals z0
    (exactly, since Euler integrates the constant-z0 target ODE exactly) while the
    repaint region converges to what the model asked for."""
    torch.manual_seed(0)
    shape = (1, 4, 8, 8)
    z0 = torch.randn(shape)
    target = torch.zeros(shape)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4] = 1.0  # left half repaint, right half keep
    sigmas = karras_schedule(15, 0.03, 14.0)  # CPU fp32, ends at 0
    x = z0 + torch.randn(shape) * sigmas[0]

    out = sample_euler(MaskedDenoiser(_ConstantDenoiser(target), z0, mask), x, sigmas)

    keep = mask.expand_as(out) == 0
    repaint = mask.expand_as(out) == 1
    assert torch.allclose(out[keep], z0[keep], atol=1e-4)
    assert torch.allclose(out[repaint], target[repaint], atol=1e-4)


# --- pixel composite -- CPU-runnable -----------------------------------------

def test_composite_preserves_keep_region_exactly():
    rng = np.random.default_rng(0)
    original = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    generated = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    mask_arr = np.zeros((16, 16), dtype=np.uint8)
    mask_arr[:, :8] = 255  # left half repaint (white), right half keep (black)
    mask = Image.fromarray(mask_arr)

    out = np.asarray(Inpaint._composite(generated, original, mask, 16, 16))
    assert np.array_equal(out[:, 8:], np.asarray(original)[:, 8:])    # keep == original
    assert np.array_equal(out[:, :8], np.asarray(generated)[:, :8])   # repaint == generated


# --- end-to-end -- needs a checkpoint ----------------------------------------

@pytest.fixture(scope="module", params=list(CKPTS))
def pipe(request):
    arch = request.param
    ckpt = CKPTS[arch]
    if not ckpt.exists():
        pytest.skip(f"{arch} checkpoint not found at {ckpt}")
    from diffucore import Inpaint, load_checkpoint

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    return Inpaint(load_checkpoint(str(ckpt), device=device, dtype=dtype))


def _inputs():
    rng = np.random.default_rng(0)
    init = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
    mask_arr = np.zeros((128, 128), dtype=np.uint8)
    mask_arr[32:96, 32:96] = 255  # repaint a centered box
    return init, Image.fromarray(mask_arr)


def _run(pipe, seed):
    init, mask = _inputs()
    return pipe("a red cube on a table", init, mask, negative_prompt="blurry",
                steps=4, cfg_scale=7.0, width=128, height=128, seed=seed)


def test_inpaint_produces_rgb_image(pipe):
    img = _run(pipe, seed=0)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128) and img.mode == "RGB"
    assert np.asarray(img).std() > 1.0


def test_inpaint_keeps_outside_mask_and_changes_inside(pipe):
    init, mask = _inputs()
    out = np.asarray(_run(pipe, seed=0))
    original = np.asarray(init.convert("RGB").resize((128, 128), Image.LANCZOS))
    keep = np.asarray(mask) < 128
    assert np.array_equal(out[keep], original[keep])       # outside: byte-exact
    assert not np.array_equal(out[~keep], original[~keep])  # inside: repainted


def test_inpaint_seed_reproducible(pipe):
    a = np.asarray(_run(pipe, seed=123))
    b = np.asarray(_run(pipe, seed=123))
    c = np.asarray(_run(pipe, seed=456))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
