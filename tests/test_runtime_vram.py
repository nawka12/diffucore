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


def test_offload_mode_flags():
    """Per-module placement decisions per mode: 'encoders' parks the idle modules
    (encoders + VAE) but keeps the UNet resident; full offload parks all three."""
    cpu = torch.device("cpu")
    off = DevicePolicy(device=cpu)
    assert (off.offload_idle, off.offload_unet) == (False, False)
    for full in (DevicePolicy(device=cpu, offload=True), DevicePolicy(device=cpu, offload="full")):
        assert (full.offload_idle, full.offload_unet) == (True, True)
    enc = DevicePolicy(device=cpu, offload="encoders")
    assert (enc.offload_idle, enc.offload_unet) == (True, False)


def test_offload_invalid_mode_rejected():
    with pytest.raises(ValueError):
        DevicePolicy(device=torch.device("cpu"), offload="sometimes")


def test_on_device_parks_module_back_on_cpu(vae):
    cpu = torch.device("cpu")
    with on_device(vae, cpu) as m:
        assert next(m.parameters()).device == cpu
    assert next(vae.parameters()).device == cpu  # parked back after the stage


# --- Sequential offload on a real model (R3) -- needs SDXL + CUDA ------------

def _require_sdxl_cuda():
    if not torch.cuda.is_available():
        pytest.skip("offload verification needs CUDA")
    if not _SDXL_CKPT.exists():
        pytest.skip(f"SDXL checkpoint not found at {_SDXL_CKPT}")


def _free():
    """Drop any freed pipeline's weights from host RAM and the GPU caching
    allocator before the next load — a 6.6 GB SDXL checkpoint won't fit twice."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gen_fresh(offload, **kw):
    """Load one SDXL pipeline in ``offload`` mode, generate, then free it. Loading
    one mode at a time is the whole point: holding the resident copy (6.6 GB on the
    GPU) alongside an encoders-mode copy (5 GB UNet also resident) overflows the
    12 GB card before a single step runs."""
    from diffucore import TextToImage, load_checkpoint

    policy = DevicePolicy(device=torch.device("cuda"), compute_dtype=torch.float16, offload=offload)
    pipe = TextToImage(load_checkpoint(str(_SDXL_CKPT), policy=policy))
    try:
        return np.asarray(pipe("a red cube on a table", negative_prompt="blurry",
                               steps=4, cfg_scale=7.0, seed=0, **kw))
    finally:
        del pipe
        _free()


@pytest.fixture(scope="module")
def sdxl_reference():
    """The 1024² resident-mode image as a (tiny) numpy array; the pipeline that
    produced it is freed immediately. Cached for the module so the offload
    comparisons don't each reload + regenerate the reference."""
    _require_sdxl_cuda()
    return _gen_fresh(False, width=1024, height=1024)


@pytest.mark.parametrize("mode", [True, "encoders"])
def test_offload_is_byte_identical(sdxl_reference, mode):
    """Offload only moves weights between devices — it must not change a pixel, in
    either full (``True``) or encoders-only mode."""
    out = _gen_fresh(mode, width=1024, height=1024)
    assert np.array_equal(sdxl_reference, out), f"offload={mode!r} diverged from resident"


def test_offload_peak_vram_under_budget():
    """offload + tiled VAE at 1024 should keep the peak well under the ~10 GB
    all-resident untiled figure — measured ~6.6 GB on the RTX 2060 (the floor is
    the UNet stage; see docs/RUNTIME_SPEC.md)."""
    _require_sdxl_cuda()
    from diffucore import TextToImage, load_checkpoint

    policy = DevicePolicy(device=torch.device("cuda"), compute_dtype=torch.float16, offload=True)
    pipe = TextToImage(load_checkpoint(str(_SDXL_CKPT), policy=policy))
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pipe("a red cube on a table", negative_prompt="blurry",
             steps=4, cfg_scale=7.0, seed=0, width=1024, height=1024)
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        assert peak_gb <= 7.0, f"peak {peak_gb:.2f} GB exceeds budget"
    finally:
        del pipe
        _free()


def test_encoders_offload_fits_8gb_cap():
    """R4: a 1024² SDXL generation must complete under an emulated 8 GB cap with
    encoders-mode offload + tiled VAE. The pipeline is loaded in isolation (no
    resident copy holding GPU memory) so the cap reflects this run alone."""
    _require_sdxl_cuda()
    _free()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < 8.0:
        pytest.skip(f"card has {total_gb:.1f} GB; cannot emulate an 8 GB cap")
    from diffucore import TextToImage, load_checkpoint

    policy = DevicePolicy(device=torch.device("cuda"), compute_dtype=torch.float16, offload="encoders")
    pipe = TextToImage(load_checkpoint(str(_SDXL_CKPT), policy=policy))
    try:
        torch.cuda.set_per_process_memory_fraction(8.0 / total_gb, 0)
        img = pipe("a red cube on a table", negative_prompt="blurry",
                   steps=4, cfg_scale=7.0, seed=0, width=1024, height=1024)
        assert img.size == (1024, 1024)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        del pipe
        _free()
