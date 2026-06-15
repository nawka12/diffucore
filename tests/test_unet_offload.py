"""SD/SDXL UNet block streaming: ``offload="stream"`` shuttles the UNet's
down/mid/up blocks per-forward so SDXL fits a 4 GB card (the ComfyUI --lowvram
analog). Mirrors test_flux_offload.py for the conv backbone."""
import pytest
import torch

from diffucore.models.unet import UNetConfig, UNetModel
from diffucore.runtime import stream_blocks

_UNET_BLOCK_ATTRS = ("input_blocks", "middle_block", "output_blocks")


def _tiny_unet():
    cfg = UNetConfig(
        in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1,
        attention_resolutions=(1,), channel_mult=(1, 2), num_heads=2,
        context_dim=64, transformer_depth=1,
    )
    torch.manual_seed(0)
    return UNetModel(cfg).eval(), cfg


def _inputs(cfg, device="cpu"):
    g = torch.Generator().manual_seed(1)
    return dict(
        x=torch.randn(1, cfg.in_channels, 16, 16, generator=g).to(device),
        timesteps=torch.tensor([0.5], device=device),
        context=torch.randn(1, 4, cfg.context_dim, generator=g).to(device),
    )


def _streamed_count(model):
    # one streamed unit per block in each list; middle_block (an nn.Sequential)
    # streams at sub-layer granularity, so its len() is its sub-module count.
    return sum(len(getattr(model, a)) for a in _UNET_BLOCK_ATTRS)


def test_stream_blocks_unet_transparent_and_streams_all():
    model, cfg = _tiny_unet()
    x = _inputs(cfg)
    with torch.no_grad():
        ref = model(**x)

    n = stream_blocks(model, _UNET_BLOCK_ATTRS, torch.device("cpu"), torch.device("cpu"))
    assert n == _streamed_count(model)

    with torch.no_grad():
        out = model(**x)
    assert torch.allclose(ref, out, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_blocks_unet_matches_resident_on_cuda():
    dev = torch.device("cuda")
    model, cfg = _tiny_unet()
    x = _inputs(cfg, device="cuda")
    with torch.no_grad():
        ref = model.to(dev)(**x)

    model = model.cpu()
    stream_blocks(model, _UNET_BLOCK_ATTRS, dev, torch.device("cpu"))
    # small modules resident on GPU, blocks parked on CPU
    assert next(model.time_embed.parameters()).device.type == "cuda"
    assert next(model.out.parameters()).device.type == "cuda"
    assert next(model.input_blocks[0].parameters()).device.type == "cpu"

    with torch.no_grad():
        out = model(**x)
    assert out.device.type == "cuda"
    assert torch.allclose(ref, out, atol=1e-4)
    # each block returns to CPU after its forward (resident VRAM stays bounded)
    assert next(model.output_blocks[-1].parameters()).device.type == "cpu"
    assert next(model.middle_block[0].parameters()).device.type == "cpu"
