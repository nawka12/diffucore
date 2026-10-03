"""Fused Anima glue: stamping, eager fallback off-GPU, and (CUDA + Triton)
agreement with the eager block to fp16 rounding."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffucore.models import _fused
from diffucore.models.anima_dit import CosmosDiTConfig, _Attention, _Block, _VideoRoPE3D

_CFG = CosmosDiTConfig(model_channels=256, num_heads=2, head_dim=128, adaln_lora_dim=32,
                       crossattn_emb_channels=64)


def _block_inputs(device, dtype=torch.float16, h=8, w=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    blk = _Block(_CFG)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    blk = blk.to(device, dtype).eval()
    D = _CFG.model_channels
    x = (torch.randn(1, 1, h, w, D, generator=g) * 2).to(device)
    emb = torch.randn(1, 1, D, generator=g).to(device, dtype)
    ctx = torch.randn(1, 16, _CFG.crossattn_emb_channels, generator=g).to(device, dtype)
    lora = (torch.randn(1, 1, 3 * D, generator=g) * 0.1).to(device, dtype)
    rope = _VideoRoPE3D(_CFG).to(device)(x).unsqueeze(1).unsqueeze(0)
    return blk, x, emb, ctx, rope, lora


def test_stamp_reaches_blocks_and_attention():
    blk = _Block(_CFG)
    assert _fused.set_fused_glue(blk, True) == 3   # block + self + cross attention
    assert blk.fused_glue and blk.self_attn.fused_glue and blk.cross_attn.fused_glue


def test_stamped_block_unchanged_on_cpu():
    blk, x, emb, ctx, rope, lora = _block_inputs("cpu", torch.float32)
    with torch.no_grad():
        before = blk(x, emb, ctx, rope, lora)
        _fused.set_fused_glue(blk, True)
        after = blk(x, emb, ctx, rope, lora)
    assert torch.equal(before, after)


_needs_fused = pytest.mark.skipif(not _fused.fused_glue_available(), reason="needs CUDA and Triton")


@_needs_fused
def test_fused_block_matches_eager():
    blk, x, emb, ctx, rope, lora = _block_inputs("cuda")
    with torch.no_grad():
        ref = blk(x, emb, ctx, rope, lora)
        _fused.set_fused_glue(blk, True)
        x_in = x.clone()
        out = blk(x, emb, ctx, rope, lora)
    assert torch.equal(x, x_in)                    # the input residual is never written
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert float((out - ref).norm() / ref.norm()) < 1e-4


@_needs_fused
def test_ln_mod_kernels_round_like_eager():
    torch.manual_seed(0)
    N, D = 512, 2048
    x = torch.randn(N, D, device="cuda") * 3 + 0.5
    sc1 = (1 + torch.randn(1, D, device="cuda") * 0.3).half()
    s = (torch.randn(1, D, device="cuda") * 0.3).half()
    ref = (F.layer_norm(x, (D,), eps=1e-6) * sc1 + s).half()
    out = _fused.ln_mod(x, sc1, s, N, 1e-6)
    assert (out != ref).float().mean() < 1e-3
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)

    h = torch.randn(N, D, device="cuda").half()
    g = torch.randn(1, D, device="cuda").half()
    xo = torch.empty_like(x)
    _fused.res_ln_mod(x, h, g, sc1, s, N, 1e-6, xo)
    assert torch.equal(xo, x + g.float() * h.float())


@_needs_fused
def test_qk_norm_rope_matches_eager():
    attn = _Attention(256, None, 2, 128, eps=1e-6).cuda().half().eval()
    torch.manual_seed(0)
    x = torch.randn(1, 96, 256, device="cuda").half()
    rope = _VideoRoPE3D(_CFG).cuda()(torch.empty(1, 1, 8, 12, 256, device="cuda")).unsqueeze(1).unsqueeze(0)
    with torch.no_grad():
        ref = attn(x, None, rope)
        _fused.set_fused_glue(attn, True)
        out = attn(x, None, rope)
    assert float((out.float() - ref.float()).norm() / ref.float().norm()) < 1e-3
