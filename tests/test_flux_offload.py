"""FLUX memory orchestration: the ``offload="stream"`` block-streaming mode and
the ``can_decode_untiled`` device-index fix."""
import pytest
import torch

from diffucore.models.flux_dit import Flux, FluxConfig
from diffucore.runtime import (
    DevicePolicy,
    can_decode_untiled,
    maybe_compile_backbone,
    stream_blocks,
)


def _tiny_flux():
    cfg = FluxConfig(
        in_channels=16, context_in_dim=32, vec_in_dim=8,
        hidden_size=256, num_heads=2, depth=2, depth_single_blocks=2,
        guidance_embed=False,
    )
    torch.manual_seed(0)
    return Flux(cfg).eval(), cfg


def _inputs(cfg, device="cpu"):
    g = torch.Generator().manual_seed(1)
    return dict(
        img=torch.randn(1, 4, cfg.in_channels, generator=g).to(device),
        img_ids=torch.zeros(1, 4, 3, device=device),
        txt=torch.randn(1, 3, cfg.context_in_dim, generator=g).to(device),
        txt_ids=torch.zeros(1, 3, 3, device=device),
        timesteps=torch.tensor([0.5], device=device),
        y=torch.randn(1, cfg.vec_in_dim, generator=g).to(device),
    )


# ---- DevicePolicy: the new mode ------------------------------------------------

def test_policy_accepts_stream_mode():
    p = DevicePolicy(device=torch.device("cpu"), offload="stream")
    assert p.offload_stream is True
    assert p.offload_idle is True      # encoders + VAE still parked between stages
    assert p.offload_unet is False     # backbone is streamed, never moved whole


def test_policy_rejects_unknown_offload():
    with pytest.raises(ValueError, match="offload must be"):
        DevicePolicy(device=torch.device("cpu"), offload="sometimes")


def test_compile_incompatible_with_stream():
    p = DevicePolicy(device=torch.device("cpu"), offload="stream", compile=True)
    with pytest.raises(ValueError, match="incompatible"):
        maybe_compile_backbone(torch.nn.Linear(2, 2), p)


# ---- stream_blocks: numerically transparent + correct placement ----------------

def test_stream_blocks_transparent_and_streams_all():
    model, cfg = _tiny_flux()
    x = _inputs(cfg)
    with torch.no_grad():
        ref = model(**x)

    n = stream_blocks(model, ("double_blocks", "single_blocks"),
                      torch.device("cpu"), torch.device("cpu"))
    assert n == cfg.depth + cfg.depth_single_blocks

    with torch.no_grad():
        out = model(**x)
    assert torch.allclose(ref, out, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_blocks_matches_resident_on_cuda():
    dev = torch.device("cuda")
    model, cfg = _tiny_flux()
    x = _inputs(cfg, device="cuda")
    with torch.no_grad():
        ref = model.to(dev)(**x)

    model = model.cpu()
    stream_blocks(model, ("double_blocks", "single_blocks"), dev, torch.device("cpu"))
    # small modules resident on GPU, blocks parked on CPU
    assert model.img_in.weight.device.type == "cuda"
    assert next(model.double_blocks[0].parameters()).device.type == "cpu"

    with torch.no_grad():
        out = model(**x)
    assert out.device.type == "cuda"
    assert torch.allclose(ref, out, atol=1e-4)
    # each block returns to CPU after its forward (resident VRAM stays bounded)
    assert next(model.single_blocks[-1].parameters()).device.type == "cpu"


# ---- can_decode_untiled: indexless cuda device must not raise -------------------

def test_can_decode_untiled_cpu_is_true():
    assert can_decode_untiled(torch.nn.Identity(), (1, 16, 8, 8),
                              torch.device("cpu")) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_can_decode_untiled_indexless_cuda_device():
    # Regression: a bare torch.device("cuda") (no index) used to crash
    # torch.cuda.mem_get_info; it should resolve to the current device instead.
    out = can_decode_untiled(torch.nn.Identity(), (1, 16, 8, 8), torch.device("cuda"))
    assert isinstance(out, bool)


# ---- FLUX.2 VAE/latent bridge -------------------------------------------------

def test_flux2_unpatchify_inverts_reference_patchify():
    from diffucore.pipelines._flux import _flux2_unpatchify_latents
    torch.manual_seed(0)
    z = torch.randn(2, 32, 16, 16)  # VAE latent (32-ch)
    # reference Flux2 pipeline _patchify_latents: 32-ch -> 128-ch (2x2 pixel-shuffle)
    b, c, h, w = z.shape
    zp = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    assert tuple(zp.shape) == (2, 128, 8, 8)
    back = _flux2_unpatchify_latents(zp)
    assert tuple(back.shape) == (2, 32, 16, 16)
    assert torch.allclose(back, z, atol=1e-6)


def test_lift_quant_convs_promotes_flux2_nested_keys():
    from diffucore.bundle import _lift_quant_convs
    sd = {
        "encoder.conv_in.weight": 1, "decoder.conv_in.weight": 1,
        "encoder.quant_conv.weight": 2, "encoder.quant_conv.bias": 3,
        "decoder.post_quant_conv.weight": 4, "decoder.post_quant_conv.bias": 5,
        "bn.running_mean": 6,
    }
    out = _lift_quant_convs(sd)
    assert out["quant_conv.weight"] == 2 and out["quant_conv.bias"] == 3
    assert out["post_quant_conv.weight"] == 4 and out["post_quant_conv.bias"] == 5
    assert "encoder.quant_conv.weight" not in out and "decoder.post_quant_conv.weight" not in out
    assert out["encoder.conv_in.weight"] == 1 and out["bn.running_mean"] == 6  # untouched


def test_infer_vae_config_detects_nested_quant_conv():
    from diffucore.bundle import _infer_vae_config, _lift_quant_convs
    # minimal LDM-ish header with FLUX.2-style nested quant convs
    sd = {
        "encoder.conv_in.weight": torch.zeros(128, 3, 3, 3),
        "encoder.down.0.block.0.norm1.weight": torch.zeros(128),
        "encoder.down.0.block.0.conv2.weight": torch.zeros(128, 128, 3, 3),
        "encoder.conv_out.weight": torch.zeros(64, 512, 3, 3),
        "encoder.quant_conv.weight": torch.zeros(64, 64, 1, 1),
        "decoder.post_quant_conv.weight": torch.zeros(32, 32, 1, 1),
    }
    cfg = _infer_vae_config(_lift_quant_convs(sd), scale_factor=1.0, shift_factor=0.0)
    assert cfg.use_quant_conv is True
    assert cfg.z_channels == 32
