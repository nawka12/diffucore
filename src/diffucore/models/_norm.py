"""Shared normalization layers and small helpers.

``RMSNorm`` is used by the DiT, Qwen3, and LLM-Adapter; ``_rotate_half`` is the
common sub-expression in both RoPE ``_apply_rope`` variants.

All are implemented in fp32 for numerical fidelity, matching the oracle
references.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm: ``x · rsqrt(mean(x²) + ε) · γ`` computed in fp32."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(in_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Negate the second half of the last dim and swap halves — the RoPE
    ``(−sin, cos)`` mixing operator applied per pair of channels."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
