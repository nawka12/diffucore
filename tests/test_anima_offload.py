"""Anima DiT block streaming: ``offload="stream"`` shuttles the DiT blocks
per-forward so Anima's ~4 GB DiT fits a small card (the ComfyUI --lowvram
analog). Block 0 stays resident because TeaCache probes its
``modulated_self_attn_input`` directly. Mirrors test_flux_offload.py."""
import pytest
import torch

from diffucore.models.anima_dit import CosmosDiT, CosmosDiTConfig, TeaCache
from diffucore.runtime import stream_blocks


def _tiny_dit():
    cfg = CosmosDiTConfig(model_channels=128, num_blocks=3, num_heads=4, head_dim=32)
    torch.manual_seed(0)
    return CosmosDiT(cfg).float().eval(), cfg


def _inputs(cfg, device="cpu"):
    torch.manual_seed(1)
    return dict(
        x=torch.randn(1, cfg.in_channels, 1, 8, 8).to(device),
        timesteps=torch.tensor([42.0], device=device),
        context=torch.randn(1, 16, cfg.crossattn_emb_channels).to(device),
    )


def test_stream_blocks_anima_transparent_keeps_block0_resident():
    model, cfg = _tiny_dit()
    x = _inputs(cfg)
    with torch.no_grad():
        ref = model(**x)

    n = stream_blocks(model, ("blocks",), torch.device("cpu"), torch.device("cpu"),
                      keep_resident=(model.blocks[0],))
    assert n == cfg.num_blocks - 1   # block 0 kept resident, the rest streamed

    with torch.no_grad():
        out = model(**x)
    assert torch.allclose(ref, out, atol=1e-6)


def test_stream_blocks_grouped_transparent_keeps_block0_resident():
    """``num_blocks_per_group`` shuttles blocks in chunks but is still transparent
    and still returns the streamed-block count (kept-resident block 0 excluded)."""
    model, cfg = _tiny_dit()
    x = _inputs(cfg)
    with torch.no_grad():
        ref = model(**x)

    n = stream_blocks(model, ("blocks",), torch.device("cpu"), torch.device("cpu"),
                      keep_resident=(model.blocks[0],), num_blocks_per_group=2)
    assert n == cfg.num_blocks - 1

    with torch.no_grad():
        out = model(**x)
    assert torch.allclose(ref, out, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_blocks_prefetch_matches_resident_on_cuda():
    """Side-stream prefetch is numerically transparent across repeated forwards
    (repeats shake out any cross-stream free/reuse hazard), and parks the streamed
    blocks back on CPU with block 0 kept resident."""
    dev = torch.device("cuda")
    model, cfg = _tiny_dit()
    x = _inputs(cfg, device="cuda")
    with torch.no_grad():
        ref = model.to(dev)(**x)

    model = model.cpu()
    stream_blocks(model, ("blocks",), dev, torch.device("cpu"),
                  keep_resident=(model.blocks[0],), prefetch=True)
    with torch.no_grad():
        for _ in range(3):
            out = model(**x)
            assert torch.allclose(ref, out, atol=1e-4)
    assert next(model.blocks[-1].parameters()).device.type == "cpu"
    assert next(model.blocks[0].parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_blocks_prefetch_grouped_with_teacache_on_cuda():
    """Prefetch + grouping + TeaCache together: block 0 stays resident for the
    TeaCache probe while the grouped rest streams via the side stream."""
    dev = torch.device("cuda")
    model, cfg = _tiny_dit()
    x = _inputs(cfg, device="cuda")
    with torch.no_grad():
        ref = model.to(dev)(**x)

    model = model.cpu()
    stream_blocks(model, ("blocks",), dev, torch.device("cpu"),
                  keep_resident=(model.blocks[0],), num_blocks_per_group=2, prefetch=True)
    tc = TeaCache(0.1)
    with torch.no_grad():
        out = model(**x, teacache=tc)   # would raise if block 0 were parked on CPU
    assert torch.allclose(ref, out, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_blocks_anima_matches_resident_on_cuda():
    dev = torch.device("cuda")
    model, cfg = _tiny_dit()
    x = _inputs(cfg, device="cuda")
    with torch.no_grad():
        ref = model.to(dev)(**x)

    model = model.cpu()
    stream_blocks(model, ("blocks",), dev, torch.device("cpu"),
                  keep_resident=(model.blocks[0],))
    # small modules + block 0 resident on GPU; the rest parked on CPU
    assert next(model.x_embedder.parameters()).device.type == "cuda"
    assert next(model.final_layer.parameters()).device.type == "cuda"
    assert next(model.blocks[0].parameters()).device.type == "cuda"
    assert next(model.blocks[1].parameters()).device.type == "cpu"

    with torch.no_grad():
        out = model(**x)
    assert out.device.type == "cuda"
    assert torch.allclose(ref, out, atol=1e-4)
    # streamed blocks return to CPU after their forward; block 0 stays resident
    assert next(model.blocks[-1].parameters()).device.type == "cpu"
    assert next(model.blocks[0].parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_with_teacache_probes_resident_block0_on_cuda():
    """Regression: TeaCache calls blocks[0].modulated_self_attn_input directly
    (outside __call__, so the stream hooks don't fire). If block 0 were parked
    on CPU this would be a CPU-weight × GPU-input device mismatch."""
    dev = torch.device("cuda")
    model, cfg = _tiny_dit()
    model = model.cpu()
    stream_blocks(model, ("blocks",), dev, torch.device("cpu"),
                  keep_resident=(model.blocks[0],))
    x = _inputs(cfg, device="cuda")
    tc = TeaCache(0.1)
    with torch.no_grad():
        out = model(**x, teacache=tc)   # would raise if block 0 were on CPU
    assert out.device.type == "cuda"
