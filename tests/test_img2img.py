"""Tests for the image-to-image pipeline.

The pure helpers (strength -> start index, image preprocessing) run anywhere. The
end-to-end smoke test needs a real checkpoint (gitignored) and is skipped when it
is absent, mirroring ``test_pipeline_smoke.py``. Override the paths with
``DIFFUCORE_SD15_CKPT`` / ``DIFFUCORE_SDXL_CKPT``.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffucore.pipelines.image_to_image import img2img_start, preprocess_image

_MODELS = Path(__file__).resolve().parents[1] / "models"
CKPTS = {
    "sd15": Path(os.environ.get("DIFFUCORE_SD15_CKPT", _MODELS / "v1-5-pruned-emaonly.safetensors")),
    "sdxl": Path(os.environ.get("DIFFUCORE_SDXL_CKPT", _MODELS / "sdxl.safetensors")),
}


# --- pure helpers -- CPU-runnable --------------------------------------------

def test_img2img_start_runs_strength_fraction_of_steps():
    """strength * steps denoising steps are run; the start index leaves exactly
    that many sigma transitions in the sliced ([steps + 1]) schedule."""
    assert img2img_start(20, 1.0) == 0       # full schedule -> all 20 steps
    assert img2img_start(20, 0.75) == 5      # 15 steps
    assert img2img_start(20, 0.5) == 10      # 10 steps
    # A full schedule of 20 steps has 21 sigmas; the slice keeps strength*steps + 1.
    for strength in (1.0, 0.75, 0.5, 0.1):
        kept = 21 - img2img_start(20, strength)
        assert kept - 1 == int(strength * 20)


def test_preprocess_image_shape_and_range():
    img = Image.new("RGB", (40, 30), color=(255, 0, 128))
    t = preprocess_image(img, width=64, height=48)
    assert t.shape == (1, 3, 48, 64)          # resized to (W, H), CHW, batched
    assert t.dtype == torch.float32
    assert t.min() >= -1.0 and t.max() <= 1.0
    # A solid red channel maps to +1, the zero channel to -1.
    assert torch.allclose(t[0, 0], torch.ones_like(t[0, 0]))
    assert torch.allclose(t[0, 1], -torch.ones_like(t[0, 1]))


# --- end-to-end -- needs a checkpoint ----------------------------------------

@pytest.fixture(scope="module", params=list(CKPTS))
def pipe(request):
    arch = request.param
    ckpt = CKPTS[arch]
    if not ckpt.exists():
        pytest.skip(f"{arch} checkpoint not found at {ckpt}")
    from diffucore import ImageToImage, load_checkpoint

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    model = load_checkpoint(str(ckpt), device=device, dtype=dtype)
    return ImageToImage(model)


def _init_image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))


def _run(pipe, seed, strength=0.6):
    return pipe(
        "a red cube on a table",
        _init_image(),
        negative_prompt="blurry",
        strength=strength,
        steps=4,
        cfg_scale=7.0,
        width=128,
        height=128,
        seed=seed,
    )


def test_img2img_produces_rgb_image(pipe):
    img = _run(pipe, seed=0)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    assert img.mode == "RGB"
    assert np.asarray(img).std() > 1.0  # real content, not a flat image


def test_img2img_seed_reproducible(pipe):
    a = np.asarray(_run(pipe, seed=123))
    b = np.asarray(_run(pipe, seed=123))
    c = np.asarray(_run(pipe, seed=456))
    assert np.array_equal(a, b)        # same seed -> identical image
    assert not np.array_equal(a, c)    # different seed -> different image


def test_img2img_strength_changes_output(pipe):
    """Lower strength keeps more of the init image, so the result differs from a
    higher-strength run with the same seed."""
    low = np.asarray(_run(pipe, seed=7, strength=0.3))
    high = np.asarray(_run(pipe, seed=7, strength=0.9))
    assert not np.array_equal(low, high)
