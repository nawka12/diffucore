"""Attention dispatch: PyTorch SDPA (default) or the optional FA2-Turing port.

PyTorch's flash SDPA backend needs sm80+; on Turing (sm75) every DiT attention
call falls back to the slower mem-efficient kernel. The community
FlashAttention-2 port for sm75 (https://github.com/ssiu/flash-attention-turing,
package ``flash_attn_turing``) measures ×1.3–1.6 over that fallback on DiT
shapes (RTX 2060, fp16, head_dim 64/128). It is a locally-built optional
extra: when it isn't installed, everything here resolves to plain SDPA,
bit-exact with the pre-dispatch code. Newer GPUs are untouched — the kernel is
compiled for sm_75 only, and ``auto`` never picks it elsewhere (sm80+ keeps
its native flash SDPA).
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
    except Exception:  # noqa: BLE001 — absence (or a broken build) means "not installed"
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
    yield ("GPU is not Turing — the kernel is compiled for sm_75 only",
           policy.device.type == "cuda" and torch.cuda.is_available()
           and torch.cuda.get_device_capability(policy.device) == (7, 5))
    yield "compute dtype is not fp16", policy.compute_dtype == torch.float16
    yield ("incompatible with compile=True (the custom op graph-breaks in "
           "every block)", not policy.compile)


def resolve_attention_backend(policy) -> str:
    """Map ``policy.attention`` to the backend the DiTs should run.

    ``"sdpa"`` never changes anything (the bit-exact default). ``"auto"`` opts
    into FA2-Turing only when every requirement holds, silently staying on
    SDPA otherwise. An explicit ``"fa2_turing"`` raises on the first unmet
    requirement instead, so an A/B never silently runs the same backend twice.
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
    """Attention over ``(B, L, H, D)`` q/k/v → ``(B, Lq, H·D)``.

    The SDPA branch reproduces the exact layout dance the DiTs did before
    dispatch existed (transpose in, transpose+reshape out). The FA2 branch
    feeds the kernel its native ``(B, L, H, D)`` layout, skipping both
    transposes; the kernel ignores input strides (it assumes packed layout),
    so inputs are made contiguous — a no-op for Anima's projections.
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
