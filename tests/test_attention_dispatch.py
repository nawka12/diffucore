"""Attention dispatch semantics: bit-exact SDPA default, silent "auto" fallback,
loud explicit failure, and the per-call eligibility guard. The FA2-Turing
kernel itself is sm75-only and not exercised here; the INT8 kernel's accuracy
tests run only on an sm75 GPU with a CUDA toolchain.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffucore.models._attention import (
    attention_bhld,
    attention_blhd,
    int8_turing_available,
    resolve_attention_backend,
    set_attention_backend,
)
from diffucore.runtime import DevicePolicy

_CPU = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32)


def _ref_blhd(q, k, v):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


# --- dispatch math (sdpa branch is the pre-dispatch code, bit-exact) ---------

def test_blhd_sdpa_matches_manual():
    torch.manual_seed(0)
    q = torch.randn(2, 64, 4, 32)
    k = torch.randn(2, 48, 4, 32)
    v = torch.randn(2, 48, 4, 32)
    assert torch.equal(attention_blhd(q, k, v, "sdpa"), _ref_blhd(q, k, v))


def test_bhld_sdpa_matches_manual():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 64, 32)   # (B, H, L, D)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    ref = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 64, -1)
    assert torch.equal(attention_bhld(q, k, v, "sdpa"), ref)


def test_fa2_backend_falls_back_off_gpu():
    """A stamped module must still run on CPU/fp32 inputs (test forwards,
    detours): the per-call guard reroutes to SDPA, same numbers."""
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 64)
    assert torch.equal(
        attention_blhd(q, q, q, "fa2_turing"), attention_blhd(q, q, q, "sdpa")
    )


# --- policy resolution -------------------------------------------------------

def test_resolve_sdpa_is_default_and_inert():
    assert resolve_attention_backend(_CPU) == "sdpa"


def test_resolve_auto_falls_back_silently_on_cpu():
    policy = DevicePolicy(device=torch.device("cpu"),
                          compute_dtype=torch.float32, attention="auto")
    assert resolve_attention_backend(policy) == "sdpa"


def test_resolve_explicit_fa2_raises_on_cpu():
    policy = DevicePolicy(device=torch.device("cpu"),
                          compute_dtype=torch.float32, attention="fa2_turing")
    with pytest.raises(ValueError, match="fa2_turing"):
        resolve_attention_backend(policy)


def test_resolve_explicit_fa2_raises_with_compile():
    policy = DevicePolicy(device=torch.device("cpu"),
                          compute_dtype=torch.float16,
                          attention="fa2_turing", compile=True)
    with pytest.raises(ValueError, match="compile"):
        resolve_attention_backend(policy)


def test_policy_rejects_unknown_attention_value():
    with pytest.raises(ValueError, match="attention"):
        DevicePolicy(device=torch.device("cpu"), attention="flash3")


# --- stamping ----------------------------------------------------------------

def test_stamp_reaches_anima_and_flux_attention_modules():
    from diffucore.models.anima_dit import _Attention
    from diffucore.models.flux_dit import DoubleStreamBlock, SingleStreamBlock

    anima_attn = _Attention(query_dim=64, context_dim=None, n_heads=2,
                            head_dim=32, eps=1e-6)
    flux_double = DoubleStreamBlock(64, 2, 2.0, qkv_bias=True)
    flux_single = SingleStreamBlock(64, 2, 2.0)
    holder = torch.nn.ModuleList([anima_attn, flux_double, flux_single])

    assert all(m.attn_backend == "sdpa"
               for m in (anima_attn, flux_double, flux_single))
    n = set_attention_backend(holder, "fa2_turing")
    assert n == 3
    assert all(m.attn_backend == "fa2_turing"
               for m in (anima_attn, flux_double, flux_single))


def test_stamped_anima_attention_forward_unchanged_on_cpu():
    """End-to-end through the Anima attention module: stamping fa2 on a CPU
    module changes nothing (guard reroutes), so offline tests stay green."""
    from diffucore.models.anima_dit import _Attention

    torch.manual_seed(0)
    attn = _Attention(query_dim=64, context_dim=None, n_heads=2,
                      head_dim=32, eps=1e-6).eval()
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        before = attn(x, None, None)
        set_attention_backend(attn, "fa2_turing")
        after = attn(x, None, None)
    assert torch.equal(before, after)


# --- INT8 backend ------------------------------------------------------------

def test_int8_backend_falls_back_off_gpu():
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 128)
    assert torch.equal(
        attention_blhd(q, q, q, "int8_turing"), attention_blhd(q, q, q, "sdpa")
    )


def test_resolve_explicit_int8_raises_on_cpu():
    policy = DevicePolicy(device=torch.device("cpu"),
                          compute_dtype=torch.float32, attention="int8_turing")
    with pytest.raises(ValueError, match="int8_turing"):
        resolve_attention_backend(policy)


def test_resolve_explicit_int8_raises_with_compile():
    policy = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float16,
                          attention="int8_turing", compile=True)
    with pytest.raises(ValueError, match="compile"):
        resolve_attention_backend(policy)


def test_int8_stamps_self_attention_only():
    """Anima cross-attention keeps an exact backend; FLUX's joint attention and
    Anima self-attention get the INT8 kernel."""
    from diffucore.models.anima_dit import _Attention
    from diffucore.models.flux_dit import SingleStreamBlock

    self_attn = _Attention(query_dim=64, context_dim=None, n_heads=2, head_dim=32, eps=1e-6)
    cross_attn = _Attention(query_dim=64, context_dim=48, n_heads=2, head_dim=32, eps=1e-6)
    flux_single = SingleStreamBlock(64, 2, 2.0)
    holder = torch.nn.ModuleList([self_attn, cross_attn, flux_single])
    assert set_attention_backend(holder, "int8_turing") == 3
    assert self_attn.attn_backend == "int8_turing"
    assert flux_single.attn_backend == "int8_turing"
    assert cross_attn.attn_backend in ("fa2_turing", "sdpa")


_needs_int8 = pytest.mark.skipif(not int8_turing_available(),
                                 reason="needs an sm75 GPU and a CUDA toolchain")


def _rel_err(out, ref):
    return float((out.float() - ref).norm() / ref.norm())


@_needs_int8
@pytest.mark.parametrize("B,Lq,Lk,H", [(1, 1024, 1024, 4), (2, 1000, 1000, 2),
                                       (1, 700, 77, 4), (1, 70, 3953, 2)])
def test_int8_blhd_close_to_reference(B, Lq, Lk, H):
    """Unaligned lengths exercise the query/key tails; the error budget is the
    INT8 Q/K quantization (~1% on random data), not fp16 rounding."""
    torch.manual_seed(0)
    q = torch.randn(B, Lq, H, 128, device="cuda").half()
    k = (torch.randn(B, Lk, H, 128, device="cuda") + 0.5).half()
    v = torch.randn(B, Lk, H, 128, device="cuda").half()
    ref = _ref_blhd(q.float(), k.float(), v.float())
    out = attention_blhd(q, k, v, "int8_turing")
    assert out.shape == ref.shape and out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert _rel_err(out, ref) < 3e-2
    assert F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0) > 0.9995


@_needs_int8
def test_int8_bhld_and_strided_inputs():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 3, 600, 128, device="cuda").half() for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).transpose(1, 2).reshape(1, 600, -1)
    assert _rel_err(attention_bhld(q, k, v, "int8_turing"), ref) < 3e-2

    qkv = torch.randn(1, 500, 2, 3 * 128, device="cuda").half()   # packed: stride(2) = 384
    q, k, v = qkv[..., :128], qkv[..., 128:256], qkv[..., 256:]
    ref = _ref_blhd(q.float(), k.float(), v.float())
    assert _rel_err(attention_blhd(q, k, v, "int8_turing"), ref) < 3e-2


@_needs_int8
def test_int8_falls_back_for_other_head_dims():
    torch.manual_seed(0)
    q = torch.randn(1, 256, 2, 64, device="cuda").half()
    assert torch.equal(attention_blhd(q, q, q, "int8_turing"), attention_blhd(q, q, q, "sdpa"))
