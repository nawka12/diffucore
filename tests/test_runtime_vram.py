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
from diffucore.runtime import (
    DevicePolicy,
    can_decode_untiled,
    maybe_compile_backbone,
    on_device,
    perf_context,
    tiled_vae_decode,
    to_channels_last,
)

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


# --- Smart tile-vs-untiled decision -----------------------------------------

def test_can_decode_untiled_cpu_always_true(vae):
    """CPU has no VRAM constraint — system RAM is plentiful, so the smart
    check must always allow untiled regardless of resolution."""
    cpu = torch.device("cpu")
    # Use an absurd latent size that would never fit on a GPU.
    assert can_decode_untiled(vae, (1, 4, 512, 512), cpu) is True


def test_can_decode_untiled_tiles_when_short_on_vram(vae):
    """Estimated decode cost > free * margin → False (tile). At 1024² SDXL the
    decode needs ~6 MB/px = 6 GB with the safety constant; if the caller passes
    only 2 GB free, the check must fall on the tile side."""
    cuda = torch.device("cuda")
    two_gb = 2 * 1024**3
    assert can_decode_untiled(vae, (1, 4, 128, 128), cuda, free_bytes=two_gb) is False


def test_can_decode_untiled_untiled_when_vram_is_ample(vae):
    """With plenty of free VRAM the check must allow untiled — 24 GB free at
    1024² leaves > 6 GB / 0.85 headroom."""
    cuda = torch.device("cuda")
    twenty_four_gb = 24 * 1024**3
    assert can_decode_untiled(vae, (1, 4, 128, 128), cuda, free_bytes=twenty_four_gb) is True


def test_can_decode_untiled_qwen_uses_lower_per_pixel_cost():
    """The Qwen-Image VAE family has lower per-pixel decode cost than SDXL's
    AutoencoderKL (3D-conv path but smaller activation peak — see f10771f).
    At a free budget where SDXL would tile, Qwen must still go untiled. A stub
    class with the right ``__name__`` is enough — the helper only reads
    ``type(vae).__name__``."""
    cuda = torch.device("cuda")
    qwen_stub = type("QwenImageVAE", (torch.nn.Module,), {})()
    sdxl_stub = type("AutoencoderKL", (torch.nn.Module,), {})()
    # 1024² latent (128×128 latent → 1024×1024 px). SDXL needs 6 GB, Qwen 5 GB.
    # Pick free = 6.5 GB so the 0.85 margin (≈5.5 GB) lands between the two.
    free = (13 * 1024**3) // 2  # 6.5 GB
    shape = (1, 16, 128, 128)
    assert can_decode_untiled(sdxl_stub, shape, cuda, free_bytes=free) is False
    assert can_decode_untiled(qwen_stub, shape, cuda, free_bytes=free) is True


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


# --- Perf flags (PR-A) -- CPU-runnable ---------------------------------------

def test_perf_flag_defaults():
    """cudnn_benchmark defaults on (bit-exact, free win); the not-bit-exact /
    hardware-sensitive flags default off."""
    p = DevicePolicy(device=torch.device("cpu"))
    assert p.cudnn_benchmark is True
    assert p.tf32 is False
    assert p.channels_last is False


def test_perf_context_is_no_op_when_off():
    """With all perf flags explicitly off, perf_context must not touch global
    torch state."""
    p = DevicePolicy(device=torch.device("cpu"), cudnn_benchmark=False)
    prev_bench = torch.backends.cudnn.benchmark
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    with perf_context(p):
        assert torch.backends.cudnn.benchmark == prev_bench
        assert torch.backends.cuda.matmul.allow_tf32 == prev_matmul_tf32
        assert torch.backends.cudnn.allow_tf32 == prev_cudnn_tf32
    assert torch.backends.cudnn.benchmark == prev_bench
    assert torch.backends.cuda.matmul.allow_tf32 == prev_matmul_tf32
    assert torch.backends.cudnn.allow_tf32 == prev_cudnn_tf32


def test_perf_context_sets_and_restores_when_on():
    """With flags on, perf_context flips the global flags inside and restores
    the previous values on exit — even if they were True or False to start."""
    p = DevicePolicy(device=torch.device("cpu"), cudnn_benchmark=True, tf32=True)
    # Force a known-baseline OFF state so we can observe the flip and restore.
    prev_bench = torch.backends.cudnn.benchmark
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with perf_context(p):
            assert torch.backends.cudnn.benchmark is True
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


def test_perf_context_fp16_accumulation_sets_and_restores():
    """fp16_accumulation defaults off; when on, perf_context flips the cuBLAS
    fp16-accumulate flag inside the context and restores it on exit."""
    assert DevicePolicy(device=torch.device("cpu")).fp16_accumulation is False
    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        pytest.skip("torch < 2.7: no allow_fp16_accumulation backend flag")
    p = DevicePolicy(device=torch.device("cpu"), cudnn_benchmark=False,
                     fp16_accumulation=True)
    prev = torch.backends.cuda.matmul.allow_fp16_accumulation
    try:
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        with perf_context(p):
            assert torch.backends.cuda.matmul.allow_fp16_accumulation is True
        assert torch.backends.cuda.matmul.allow_fp16_accumulation is False
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = prev


def test_compile_flag_defaults_off():
    """``compile`` must default off so the current behavior is unchanged."""
    p = DevicePolicy(device=torch.device("cpu"))
    assert p.compile is False


def test_compile_with_full_offload_raises():
    """compile=True is incompatible with offload modes that move the backbone."""
    cpu = torch.device("cpu")
    backbone = torch.nn.Linear(4, 4)
    for offload in (True, "full"):
        p = DevicePolicy(device=cpu, compile=True, offload=offload)
        with pytest.raises(ValueError, match="incompatible"):
            maybe_compile_backbone(backbone, p)


def test_compile_with_encoders_offload_is_allowed():
    """offload='encoders' keeps the backbone resident — compile is fine."""
    cpu = torch.device("cpu")
    backbone = torch.nn.Linear(4, 4)
    p = DevicePolicy(device=cpu, compile=True, offload="encoders")
    # Should not raise — return value is either the eager module or an
    # OptimizedModule depending on the platform; both are callable.
    out = maybe_compile_backbone(backbone, p)
    assert callable(out)


def test_maybe_compile_is_identity_when_off():
    """With compile=False the module is returned unchanged (same identity)."""
    backbone = torch.nn.Linear(4, 4)
    p = DevicePolicy(device=torch.device("cpu"))
    assert maybe_compile_backbone(backbone, p) is backbone


def test_cuda_graphs_default_off():
    """PR-C's cuda_graphs flag must default off."""
    p = DevicePolicy(device=torch.device("cpu"))
    assert p.cuda_graphs is False


def test_cuda_graphs_requires_compile():
    """cuda_graphs=True without compile=True is a config error — CUDA Graphs are
    captured by torch.compile's reduce-overhead mode, not by us directly."""
    p = DevicePolicy(device=torch.device("cpu"), cuda_graphs=True)
    with pytest.raises(ValueError, match="requires policy.compile=True"):
        maybe_compile_backbone(torch.nn.Linear(4, 4), p)


def test_cuda_graphs_with_compile_is_allowed():
    """compile=True + cuda_graphs=True is the supported combo. Returns a callable
    (compile may eagerly wrap or defer; either is fine)."""
    backbone = torch.nn.Linear(4, 4)
    p = DevicePolicy(device=torch.device("cpu"), compile=True, cuda_graphs=True)
    out = maybe_compile_backbone(backbone, p)
    assert callable(out)


def test_to_channels_last_preserves_conv_output_cpu():
    """``to_channels_last`` is a layout change, not a math change: a Conv2d
    forward must give the same output (within tight tolerance) whether the
    module + input are channels-last or contiguous. CPU-runnable; CUDA cuDNN
    correctness is well-tested upstream."""
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(8, 16, 3, padding=1).eval()
    x = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        ref = conv(x)
        conv_cl = to_channels_last(torch.nn.Conv2d(8, 16, 3, padding=1).eval())
        # copy weights so the two convs are identical apart from layout
        conv_cl.load_state_dict(conv.state_dict())
        conv_cl = to_channels_last(conv_cl)
        x_cl = x.contiguous(memory_format=torch.channels_last)
        out_cl = conv_cl(x_cl)
    assert torch.allclose(ref, out_cl, atol=1e-5, rtol=1e-5)


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
