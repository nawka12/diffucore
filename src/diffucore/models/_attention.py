"""Attention dispatch: PyTorch SDPA (default) or the optional FA2-Turing port.

Flash SDPA needs sm80+, so Turing (sm75) falls back to the mem-efficient kernel.
The community sm75 FlashAttention-2 port (github.com/ssiu/flash-attention-turing,
package ``flash_attn_turing``) is ×1.3–1.6 faster there on DiT shapes. It is a
locally built optional extra; without it everything resolves to plain SDPA,
and ``auto`` never picks it off sm75.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

_CHOICES = ("sdpa", "auto", "fa2_turing")


@functools.cache
def _fa2_module():
    try:
        import flash_attn_turing
    except Exception:  # noqa: BLE001  absent or broken build: "not installed"
        return None
    return flash_attn_turing


def fa2_turing_available(device=None) -> bool:
    """Whether the FA2-Turing kernel can run here: package installed and the
    CUDA device is sm75 (the extension is compiled with ``-arch=sm_75`` only)."""
    if _fa2_module() is None or not torch.cuda.is_available():
        return False
    if device is not None and getattr(device, "type", None) == "cpu":
        return False
    return torch.cuda.get_device_capability(device) == (7, 5)


def _requirements(policy):
    """(message, satisfied) pairs for running FA2-Turing under ``policy``."""
    yield "flash_attn_turing is not installed", _fa2_module() is not None
    yield "device is not CUDA", policy.device.type == "cuda"
    yield ("GPU is not Turing; the kernel is compiled for sm_75 only",
           policy.device.type == "cuda" and torch.cuda.is_available()
           and torch.cuda.get_device_capability(policy.device) == (7, 5))
    yield "compute dtype is not fp16", policy.compute_dtype == torch.float16
    yield ("incompatible with compile=True (the custom op graph-breaks in "
           "every block)", not policy.compile)


def resolve_attention_backend(policy) -> str:
    """Map ``policy.attention`` to a backend: ``"sdpa"`` as is, ``"auto"`` to
    FA2-Turing only when every requirement holds, and an explicit
    ``"fa2_turing"`` raises on the first unmet requirement (so an A/B never
    silently runs the same backend twice).
    """
    if policy.attention == "sdpa":
        return "sdpa"
    failures = [msg for msg, ok in _requirements(policy) if not ok]
    if not failures:
        return "fa2_turing"
    if policy.attention == "auto":
        return "sdpa"
    raise ValueError(
        "policy.attention='fa2_turing' can't run: " + "; ".join(failures)
    )


def set_attention_backend(module: torch.nn.Module, backend: str) -> int:
    """Stamp ``backend`` on every submodule that declares an ``attn_backend``
    slot (the Anima/FLUX attention modules). Returns the number stamped."""
    n = 0
    for m in module.modules():
        if hasattr(m, "attn_backend"):
            m.attn_backend = backend
            n += 1
    return n


def _fa2_eligible(q: torch.Tensor) -> bool:
    """Per-call guard so a stamped module still runs correctly off the happy
    path (CPU test forwards, fp32 detours, foreign head_dims)."""
    return q.is_cuda and q.dtype == torch.float16 and q.shape[-1] in (64, 128)


def attention_blhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   backend: str = "sdpa") -> torch.Tensor:
    """Attention over ``(B, L, H, D)`` q/k/v → ``(B, Lq, H·D)``. FA2 takes that
    layout natively but ignores strides, so its inputs are made contiguous.
    """
    B, Lq = q.shape[0], q.shape[1]
    if backend == "fa2_turing" and _fa2_eligible(q):
        out, _ = _fa2_module().fwd(
            q.contiguous(), k.contiguous(), v.contiguous(),
            q.shape[-1] ** -0.5, False,
        )
        return out.reshape(B, Lq, -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).reshape(B, Lq, -1)


def attention_bhld(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   backend: str = "sdpa") -> torch.Tensor:
    """Attention over ``(B, H, L, D)`` q/k/v → ``(B, Lq, H·D)`` (FLUX layout,
    post-RoPE). The FA2 branch pays a transpose-copy per tensor (the kernel
    assumes packed ``(B, L, H, D)``); small next to the kernel win."""
    B, Lq = q.shape[0], q.shape[2]
    if backend == "fa2_turing" and _fa2_eligible(q):
        out, _ = _fa2_module().fwd(
            q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(), q.shape[-1] ** -0.5, False,
        )
        return out.reshape(B, Lq, -1)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(B, Lq, -1)
