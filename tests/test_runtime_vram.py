"""Verification for the runtime VRAM techniques (see docs/RUNTIME_SPEC.md).

Tiled-VAE correctness runs anywhere (no checkpoint, no GPU) on a random-weight
autoencoder — the blend math is what matters, not the weights. The offload
byte-identity and peak-VRAM checks need a real SDXL checkpoint on CUDA and skip
otherwise.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from diffucore.models import AutoencoderKL, VAEConfig
from diffucore.runtime import DevicePolicy, on_device, tiled_vae_decode

_MODELS = Path(__file__).resolve().parents[1] / "models"
_SDXL_CKPT = Path(os.environ.get("DIFFUCORE_SDXL_CKPT", _MODELS / "sdxl.safetensors"))


def _psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 2.0) -> float:
    """PSNR in dB over the [-1, 1] image range (peak-to-peak = 2)."""
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    return 10.0 * np.log10(peak**2 / mse)


@pytest.fixture
def vae():
    torch.manual_seed(0)
    return AutoencoderKL(VAEConfig()).eval()


# --- Tiled VAE decode (R2) -- CPU-runnable -----------------------------------

def test_tiled_single_tile_is_bit_identical(vae):
    """A latent that fits one tile must equal a plain decode, bit-for-bit (no
    blend math runs)."""
    lat = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        assert torch.equal(tiled_vae_decode(vae, lat, tile=64, overlap=16), vae.decode(lat))


def test_tiled_seam_quality_psnr(vae):
    """Multi-tile decode must match a single full decode within PSNR > 35 dB; the
    overlap feather is what makes the seams imperceptible (a hard cut fails this)."""
    lat = torch.randn(1, 4, 96, 96)  # 768 px -> several 64-tiles
    with torch.no_grad():
        full = vae.decode(lat)
        tiled = tiled_vae_decode(vae, lat, tile=64, overlap=16)
    assert tiled.shape == full.shape
    assert _psnr(full, tiled) > 35.0


def test_tiled_decode_is_deterministic(vae):
    lat = torch.randn(1, 4, 96, 96)
    with torch.no_grad():
        a = tiled_vae_decode(vae, lat, tile=64, overlap=16)
        b = tiled_vae_decode(vae, lat, tile=64, overlap=16)
    assert torch.equal(a, b)


# --- on_device / DevicePolicy (R3) -- CPU-runnable ---------------------------

def test_policy_defaults_are_off():
    p = DevicePolicy(device=torch.device("cpu"))
    assert not p.offload and not p.vae_tile
    assert p.offload_device == torch.device("cpu")


def test_on_device_parks_module_back_on_cpu(vae):
    cpu = torch.device("cpu")
    with on_device(vae, cpu) as m:
        assert next(m.parameters()).device == cpu
    assert next(vae.parameters()).device == cpu  # parked back after the stage


# --- Sequential offload on a real model (R3) -- needs SDXL + CUDA ------------

@pytest.fixture
def sdxl_pair():
    if not torch.cuda.is_available():
        pytest.skip("offload verification needs CUDA")
    if not _SDXL_CKPT.exists():
        pytest.skip(f"SDXL checkpoint not found at {_SDXL_CKPT}")
    from diffucore import TextToImage, load_checkpoint

    resident = load_checkpoint(str(_SDXL_CKPT), device="cuda", dtype=torch.float16)
    offloaded = load_checkpoint(
        str(_SDXL_CKPT),
        policy=DevicePolicy(device=torch.device("cuda"), compute_dtype=torch.float16, offload=True),
    )
    return TextToImage(resident), TextToImage(offloaded)


def _gen(pipe, **kw):
    return np.asarray(pipe("a red cube on a table", negative_prompt="blurry",
                           steps=4, cfg_scale=7.0, seed=0, **kw))


def test_offload_is_byte_identical(sdxl_pair):
    """Offload only moves weights between devices — it must not change a pixel."""
    resident, offloaded = sdxl_pair
    a = _gen(resident, width=1024, height=1024)
    b = _gen(offloaded, width=1024, height=1024)
    assert np.array_equal(a, b)


def test_offload_peak_vram_under_budget(sdxl_pair):
    """offload + tiled VAE at 1024 should keep the peak well under the ~10 GB
    all-resident untiled figure — measured ~6.6 GB on the RTX 2060 (the floor is
    the UNet stage; see docs/RUNTIME_SPEC.md)."""
    _, offloaded = sdxl_pair
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _gen(offloaded, width=1024, height=1024)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    assert peak_gb <= 7.0, f"peak {peak_gb:.2f} GB exceeds budget"
