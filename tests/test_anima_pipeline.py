"""Anima end-to-end pipeline (DT7) — smoke + reproducibility.

Skipped when any of the three Anima checkpoints aren't on disk. Runs on CUDA
fp16 when available, else CPU fp32 (CPU at 1024² is many minutes per step;
the test uses a tiny resolution so CPU is also tractable).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _path(env_var: str, default: str) -> Path:
    return Path(os.environ.get(env_var, default))


_DIT = _path(
    "DIFFUCORE_ANIMA_CKPT",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
)
_VAE = _path(
    "DIFFUCORE_QWEN_IMAGE_VAE",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/vae/qwen_image_vae.safetensors",
)
_TE = _path(
    "DIFFUCORE_QWEN3_TE",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors",
)


def _require_files():
    for p in (_DIT, _VAE, _TE):
        if not p.exists():
            pytest.skip(f"anima file missing: {p}")


@pytest.fixture(scope="module")
def pipe():
    _require_files()
    pytest.importorskip("transformers")  # tokenizer dep until vendoring lands
    from diffucore import TextToImage, load_anima_checkpoint
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    bundle = load_anima_checkpoint(
        dit_path=str(_DIT), vae_path=str(_VAE), te_path=str(_TE),
        device=device, dtype=dtype,
    )
    assert bundle.spec.architecture == "anima"
    return TextToImage(bundle)


def _gen(pipe, *, seed: int, prompt: str = "a fox in a forest", steps: int = 2):
    return pipe(
        prompt, negative_prompt="blurry",
        steps=steps, cfg_scale=4.0,
        width=128, height=128,
        seed=seed,
    )


def test_anima_produces_rgb_image(pipe):
    img = _gen(pipe, seed=0)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    assert img.mode == "RGB"
    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    # The 2-step output isn't pretty, but it's not flat: NaN/inf/black-screen
    # bugs would give std == 0.
    assert arr.std() > 1.0


def test_anima_seed_reproducible(pipe):
    a = np.asarray(_gen(pipe, seed=123))
    b = np.asarray(_gen(pipe, seed=123))
    c = np.asarray(_gen(pipe, seed=456))
    assert np.array_equal(a, b), "same seed must yield identical pixels"
    assert not np.array_equal(a, c), "different seed must yield different pixels"
