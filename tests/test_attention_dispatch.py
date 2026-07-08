"""Attention dispatch (models/_attention.py) — resolution, stamping, fallback.

The FA2-Turing kernel itself is GPU-only (sm75) and exercised by the
scratchpad harness / e2e A/B; these tests pin the *dispatch* semantics that
must hold everywhere: bit-exact SDPA default, silent "auto" fallback, loud
explicit failure, and the per-call eligibility guard.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffucore.models._attention import (
    attention_bhld,
    attention_blhd,
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
    detours) — the per-call guard reroutes to SDPA, same numbers."""
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
