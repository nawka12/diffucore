"""Qwen-Image VAE (DT2) verification.

Layered tests:

1. Key-set match — the module's parameter names match the on-disk safetensors
   keys exactly. Needs no checkpoint.
2. Strict load + round-trip PSNR — load real weights, encode/decode a
   smooth test image, require PSNR above a threshold. Skipped if the
   ``DIFFUCORE_QWEN_IMAGE_VAE`` checkpoint is absent.
3. Numerical agreement vs ComfyUI's ``WanVAE`` (the upstream reference). The
   GPL-licensed ComfyUI tree is imported here in ``tests/`` only; nothing
   under ``src/diffucore/`` touches it. Skipped if ComfyUI is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from diffucore.models import QwenImageVAE

_VAE_CKPT = Path(os.environ.get(
    "DIFFUCORE_QWEN_IMAGE_VAE",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/vae/qwen_image_vae.safetensors",
))
_COMFY_ROOT = Path(os.environ.get(
    "DIFFUCORE_COMFY_ROOT",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI",
))


def _psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 2.0) -> float:
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    return 10.0 * float(np.log10(peak**2 / max(mse, 1e-12)))


def _gradient_image(h: int = 256, w: int = 256, seed: int = 0) -> torch.Tensor:
    """A smooth coloured gradient with mild structure — easy for a VAE to compress."""
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij",
    )
    r = torch.sin(3 * xx) * torch.cos(2 * yy)
    g = torch.sin(2 * xx + 1.1) * torch.sin(2 * yy)
    b = torch.cos(xx * yy * 3)
    img = torch.stack([r, g, b], dim=0).unsqueeze(0)
    return img.clamp(-1, 1)


# --- 1. key-set match (no weights needed) ------------------------------------

def test_qwen_image_vae_key_set_matches_checkpoint_header():
    """The module's parameter layout exactly matches the on-disk safetensors keys."""
    from safetensors import safe_open

    if not _VAE_CKPT.exists():
        pytest.skip(f"qwen-image VAE checkpoint not at {_VAE_CKPT}")

    with safe_open(str(_VAE_CKPT), framework="pt") as f:
        ckpt_keys = set(f.keys())

    mod_keys = set(QwenImageVAE().state_dict().keys())
    assert mod_keys == ckpt_keys, (
        f"key mismatch — missing: {sorted(ckpt_keys - mod_keys)[:5]} | "
        f"extra: {sorted(mod_keys - ckpt_keys)[:5]}"
    )


# --- 2. strict-load + round-trip PSNR ---------------------------------------

@pytest.fixture(scope="module")
def loaded_vae():
    if not _VAE_CKPT.exists():
        pytest.skip(f"qwen-image VAE checkpoint not at {_VAE_CKPT}")
    from safetensors.torch import load_file
    vae = QwenImageVAE()
    vae.load_state_dict(load_file(str(_VAE_CKPT)), strict=True)
    return vae.float().eval()


def test_strict_load_and_latent_shape(loaded_vae):
    """The encoder yields the expected channel-count and 8× spatial downscale."""
    img = _gradient_image(256, 256)
    with torch.no_grad():
        z = loaded_vae.encode(img)
    assert z.shape == (1, 16, 32, 32)


def test_round_trip_psnr_threshold(loaded_vae):
    """encode → decode on a smooth image — the trained VAE clears 40 dB
    comfortably on this kind of input (~49 dB observed); 40 leaves enough
    headroom that fp32-reduction noise on different hardware won't trip it."""
    img = _gradient_image(256, 256)
    with torch.no_grad():
        z = loaded_vae.encode(img)
        rec = loaded_vae.decode(z)
    psnr = _psnr(img, rec)
    assert psnr > 40.0, f"round-trip PSNR={psnr:.2f} dB below threshold"


def test_process_in_out_roundtrips_exactly(loaded_vae):
    """The latent normalization is a per-channel affine map; process_out is its
    exact inverse (up to fp32 rounding)."""
    torch.manual_seed(1)
    z = torch.randn(1, 16, 32, 32)
    z2 = loaded_vae.process_out(loaded_vae.process_in(z))
    assert torch.allclose(z, z2, atol=1e-6)


# --- 3. numerical agreement vs ComfyUI (GPL, tests-only) ---------------------

def _comfy_available() -> bool:
    return (_COMFY_ROOT / "comfy" / "ldm" / "wan" / "vae.py").exists()


@pytest.fixture(scope="module")
def comfy_vae():
    if not _comfy_available():
        pytest.skip(f"ComfyUI not at {_COMFY_ROOT}")
    if not _VAE_CKPT.exists():
        pytest.skip(f"qwen-image VAE checkpoint not at {_VAE_CKPT}")
    sys.path.insert(0, str(_COMFY_ROOT))
    try:
        from comfy.ldm.wan.vae import WanVAE
        from safetensors.torch import load_file
    except ImportError as e:
        # ComfyUI pulls in optional native deps (comfy_aimdo, comfy_kitchen) at
        # import time; if any are missing we cannot use it as an in-process
        # oracle. Skip rather than fail — the round-trip + key-match tests
        # still verify correctness without the oracle.
        pytest.skip(f"ComfyUI import failed (likely missing optional dep): {e}")

    sd = load_file(str(_VAE_CKPT))
    dim = sd["decoder.head.0.gamma"].shape[0]
    vae = WanVAE(
        dim=dim, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
        attn_scales=[], temperal_downsample=[False, True, True],
        image_channels=3, conv_out_channels=3, dropout=0.0,
    )
    vae.load_state_dict(sd, strict=True)
    return vae.float().eval()


def test_encode_matches_comfy_within_tolerance(loaded_vae, comfy_vae):
    """Anima at T=1: our encode output should be numerically very close to
    ComfyUI's WanVAE.encode. We tolerate a small gap from F.scaled_dot_product
    backend selection vs comfy's vae_attention path."""
    img = _gradient_image(128, 128)
    with torch.no_grad():
        ours = loaded_vae.encode(img)
        theirs = comfy_vae.encode(img.unsqueeze(2)).squeeze(2)  # comfy takes 5D
    assert ours.shape == theirs.shape
    diff = (ours - theirs).abs().max().item()
    assert diff < 1e-3, f"max|Δ| encode = {diff:.3e}"


def test_decode_matches_comfy_within_tolerance(loaded_vae, comfy_vae):
    """Same tolerance argument as encode, applied to the decoder path."""
    torch.manual_seed(2)
    z = torch.randn(1, 16, 16, 16)
    with torch.no_grad():
        ours = loaded_vae.decode(z)
        theirs_chunks = comfy_vae.decode(z.unsqueeze(2))
        # ComfyUI's decode returns a list of T-chunks; for T=1 there is one.
        theirs = torch.cat(theirs_chunks, dim=2).squeeze(2)
    assert ours.shape == theirs.shape
    diff = (ours - theirs).abs().max().item()
    assert diff < 1e-3, f"max|Δ| decode = {diff:.3e}"
