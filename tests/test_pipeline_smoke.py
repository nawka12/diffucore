"""End-to-end smoke test for the text-to-image pipeline (SD1.5 and SDXL).

Each architecture requires a real checkpoint (gitignored, several GB) and is
skipped when it is absent, so the CPU-only suite stays runnable without them.
Override the paths with ``DIFFUCORE_SD15_CKPT`` / ``DIFFUCORE_SDXL_CKPT``. Runs
on CUDA (fp16) when available, else CPU (fp32); keeps resolution/steps small.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_MODELS = Path(__file__).resolve().parents[1] / "models"
CKPTS = {
    "sd15": Path(os.environ.get("DIFFUCORE_SD15_CKPT", _MODELS / "v1-5-pruned-emaonly.safetensors")),
    "sdxl": Path(os.environ.get("DIFFUCORE_SDXL_CKPT", _MODELS / "sdxl.safetensors")),
}


@pytest.fixture(scope="module", params=list(CKPTS))
def case(request):
    arch = request.param
    ckpt = CKPTS[arch]
    if not ckpt.exists():
        pytest.skip(f"{arch} checkpoint not found at {ckpt}")
    from diffucore import TextToImage, load_checkpoint

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    model = load_checkpoint(str(ckpt), device=device, dtype=dtype)
    assert model.spec.architecture == arch
    return arch, TextToImage(model)


def _generate(pipe, seed):
    return pipe(
        "a red cube on a table",
        negative_prompt="blurry",
        steps=3,
        cfg_scale=7.0,
        width=128,
        height=128,
        sampler="euler",
        seed=seed,
    )


def test_pipeline_produces_rgb_image(case):
    _, pipe = case
    img = _generate(pipe, seed=0)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    assert img.mode == "RGB"
    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    assert arr.std() > 1.0  # real content, not a flat/dead image


def test_pipeline_seed_reproducible(case):
    _, pipe = case
    a = np.asarray(_generate(pipe, seed=123))
    b = np.asarray(_generate(pipe, seed=123))
    c = np.asarray(_generate(pipe, seed=456))
    assert np.array_equal(a, b)        # same seed -> identical image
    assert not np.array_equal(a, c)    # different seed -> different image
