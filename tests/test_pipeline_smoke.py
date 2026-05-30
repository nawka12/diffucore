"""End-to-end smoke test for the text-to-image pipeline.

Requires a real SD1.5 checkpoint (gitignored, ~4 GB) and is skipped when it is
absent, so the CPU-only suite stays runnable without it. Point at a different
file with the ``DIFFUCORE_SD15_CKPT`` env var. Runs on CUDA (fp16) when
available, else CPU (fp32); keeps the resolution/steps small for speed.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "models" / "v1-5-pruned-emaonly.safetensors"
CKPT = Path(os.environ.get("DIFFUCORE_SD15_CKPT", _DEFAULT_CKPT))

pytestmark = pytest.mark.skipif(
    not CKPT.exists(),
    reason=f"SD1.5 checkpoint not found at {CKPT} (set DIFFUCORE_SD15_CKPT to override)",
)


@pytest.fixture(scope="module")
def pipe():
    from diffucore import TextToImage, load_checkpoint

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    model = load_checkpoint(str(CKPT), device=device, dtype=dtype)
    return TextToImage(model)


def _generate(pipe, seed):
    return pipe(
        "a red cube on a table",
        negative_prompt="blurry",
        steps=4,
        cfg_scale=7.0,
        width=128,
        height=128,
        sampler="euler",
        seed=seed,
    )


def test_pipeline_produces_rgb_image(pipe):
    img = _generate(pipe, seed=0)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    assert img.mode == "RGB"
    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    assert arr.std() > 1.0  # real content, not a flat/dead image


def test_pipeline_seed_reproducible(pipe):
    a = np.asarray(_generate(pipe, seed=123))
    b = np.asarray(_generate(pipe, seed=123))
    c = np.asarray(_generate(pipe, seed=456))
    assert np.array_equal(a, b)        # same seed -> identical image
    assert not np.array_equal(a, c)    # different seed -> different image
