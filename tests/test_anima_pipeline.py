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


# --- Vendored tokenizer (no transformers, no model files needed) ---------------

# Golden IDs frozen from the vendored tokenizer.json files; these were verified
# bit-identical to transformers loading ComfyUI's qwen25_tokenizer/ + t5_tokenizer/.
_GOLDEN_QWEN = [64, 8251, 11699, 389, 264, 5517]
_GOLDEN_T5 = [3, 9, 1712, 3823, 30, 3, 9, 6928, 1]  # trailing 1 = T5 </s>


def test_anima_tokenizer_golden_ids():
    from diffucore.conditioning import AnimaTokenizer
    out = AnimaTokenizer()("a cat sitting on a mat")
    assert out.qwen_ids[0].tolist() == _GOLDEN_QWEN
    assert out.qwen_mask[0].tolist() == [1] * len(_GOLDEN_QWEN)
    assert out.t5_ids[0].tolist() == _GOLDEN_T5


def test_anima_tokenizer_truncates_to_max_length():
    from diffucore.conditioning import AnimaTokenizer
    out = AnimaTokenizer()("word " * 1000, max_length=16)
    assert out.qwen_ids.shape[1] <= 16
    assert out.t5_ids.shape[1] <= 16


@pytest.mark.parametrize(
    "vendored,comfy_dir,cls,env",
    [
        ("qwen3_tokenizer.json", "qwen25_tokenizer", "Qwen2Tokenizer", "DIFFUCORE_QWEN2_TOKENIZER_DIR"),
        ("t5_tokenizer.json", "t5_tokenizer", "T5TokenizerFast", "DIFFUCORE_T5_TOKENIZER_DIR"),
    ],
)
def test_vendored_tokenizer_matches_transformers(vendored, comfy_dir, cls, env):
    """Bit-identity of the vendored file against transformers loading the
    upstream tokenizer. Skips unless transformers + a source dir are available."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    import diffucore.conditioning as cond

    src = os.environ.get(
        env,
        f"/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/comfy/text_encoders/{comfy_dir}",
    )
    if not Path(src).exists():
        pytest.skip(f"source tokenizer dir missing: {src}")

    ref = getattr(transformers, cls).from_pretrained(src)
    vend = Tokenizer.from_file(str(Path(cond.__file__).with_name(vendored)))
    vend.enable_truncation(512)
    for prompt in ["a cat sitting on a mat", "Café déjà vu 🤖 漢字", "", "word " * 600]:
        expected = ref(prompt, padding=False, truncation=True, max_length=512)["input_ids"]
        assert vend.encode(prompt).ids == expected, prompt[:30]


@pytest.fixture(scope="module")
def pipe():
    _require_files()
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


def test_anima_calibrate_oss_produces_valid_schedule(pipe):
    from diffucore import anima_calibrate_oss

    sigmas = anima_calibrate_oss(
        pipe.model, "a fox in a forest", "blurry",
        steps=4, width=128, height=128, shift=3.0, cfg_scale=4.0, grid=10, seed=0,
    )
    assert len(sigmas) == 5                                   # steps + 1
    assert sigmas[-1] == 0.0
    assert all(s == s for s in sigmas)                        # no NaN
    assert all(a >= b for a, b in zip(sigmas, sigmas[1:]))    # descending
    assert abs(sigmas[0] - 1.0) < 1e-4                        # starts at σ_max == 1


def test_anima_oss_schedule_drives_generation(pipe):
    # The calibrated schedule must flow through the pipeline and produce an image.
    from diffucore import anima_calibrate_oss

    sigmas = anima_calibrate_oss(
        pipe.model, "a fox in a forest", "blurry",
        steps=4, width=128, height=128, shift=3.0, cfg_scale=4.0, grid=10, seed=0,
    )
    img = pipe(
        "a fox in a forest", negative_prompt="blurry",
        steps=4, cfg_scale=4.0, width=128, height=128, seed=0,
        scheduler="oss", oss_sigmas=sigmas,
    )
    assert isinstance(img, Image.Image)
    assert np.asarray(img).std() > 1.0
