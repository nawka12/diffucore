"""T5 v1.1 XXL encoder — FLUX.1's primary text encoder.

Implements the encoder half of the T5 text-to-text transformer (Raffel et al.,
2020) in its v1.1 form (Shazeer's gated-GELU FFN, no embedding/output tying, no
attention scaling). FLUX.1 conditions its DiT on the T5-XXL ``last_hidden_state``
(4096-d) alongside a pooled CLIP-L vector.

Submodule and parameter names mirror the HuggingFace ``T5EncoderModel`` on-disk
keys (``shared.weight``, ``encoder.block.{i}.layer.{0,1}.*``,
``encoder.final_layer_norm.weight``) so a ``strict=True`` load is the correctness
check. The vendored ``conditioning/t5_tokenizer.json`` is google-t5's SentencePiece
vocab (32128), the same T5 FLUX inherits.

Contract:
    forward(input_ids: LongTensor[B, L]) -> last_hidden_state: FloatTensor[B, L, 4096]

T5 specifics this implements faithfully:
    - T5LayerNorm == RMSNorm (no mean-subtraction, no bias) — reuses ``_norm.RMSNorm``.
    - Relative position bias (32 buckets, max distance 128), computed once in the
      first block and shared across all layers.
    - No 1/sqrt(d) attention scaling (folded into the trained weights).
    - Gated-GELU FFN: ``wo(gelu(wi_0(x)) * wi_1(x))`` with the tanh GELU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._norm import RMSNorm


@dataclass
class T5Config:
    vocab_size: int = 32128
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_eps: float = 1e-6


def _relative_position_bucket(
    relative_position: torch.Tensor, num_buckets: int, max_distance: int
) -> torch.Tensor:
    """Bidirectional bucketing of relative positions (T5, Raffel et al. 2020).

    Half the buckets encode the sign (memory-before vs memory-after query); within
    a sign, the first ``num_buckets//4`` positions map exactly and the rest fall on
    a logarithmic scale out to ``max_distance``.
    """
    num_buckets //= 2
    ret = (relative_position > 0).to(torch.long) * num_buckets
    n = relative_position.abs()

    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_large = torch.minimum(val_large, torch.full_like(val_large, num_buckets - 1))
    return ret + torch.where(is_small, n, val_large)


class T5Attention(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.d_kv = cfg.d_kv
        inner = cfg.num_heads * cfg.d_kv
        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                cfg.relative_attention_num_buckets, cfg.num_heads
            )
        self.num_buckets = cfg.relative_attention_num_buckets
        self.max_distance = cfg.relative_attention_max_distance

    def compute_bias(self, length: int, device: torch.device) -> torch.Tensor:
        ctx = torch.arange(length, dtype=torch.long, device=device)[:, None]
        mem = torch.arange(length, dtype=torch.long, device=device)[None, :]
        buckets = _relative_position_bucket(mem - ctx, self.num_buckets, self.max_distance)
        values = self.relative_attention_bias(buckets)          # [L, L, heads]
        return values.permute(2, 0, 1).unsqueeze(0)             # [1, heads, L, L]

    def forward(self, x: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape

        def shape(t):
            return t.view(b, length, self.num_heads, self.d_kv).transpose(1, 2)

        q, k, v = shape(self.q(x)), shape(self.k(x)), shape(self.v(x))
        # T5 does not scale queries by 1/sqrt(d_kv); the position bias is added to
        # the raw scores. Softmax in fp32 — T5 activations overflow fp16 otherwise.
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) + position_bias.float()
        attn = scores.softmax(dim=-1).to(v.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, length, -1)
        return self.o(out)


class T5LayerSelfAttention(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.SelfAttention = T5Attention(cfg, has_relative_attention_bias)
        self.layer_norm = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        return x + self.SelfAttention(self.layer_norm(x), position_bias)


class T5DenseGatedActDense(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wi_1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(F.gelu(self.wi_0(x), approximate="tanh") * self.wi_1(x))


class T5LayerFF(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(cfg)
        self.layer_norm = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Block(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.layer = nn.ModuleList(
            [T5LayerSelfAttention(cfg, has_relative_attention_bias), T5LayerFF(cfg)]
        )

    def forward(self, x: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        x = self.layer[0](x, position_bias)
        return self.layer[1](x)


class T5Stack(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        # Only the first block carries the (shared) relative-position-bias table.
        self.block = nn.ModuleList(
            T5Block(cfg, has_relative_attention_bias=(i == 0)) for i in range(cfg.num_layers)
        )
        self.final_layer_norm = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)


class T5TextEncoder(nn.Module):
    """T5 v1.1 XXL encoder. Loads from HF ``T5EncoderModel`` keys."""

    def __init__(self, config: T5Config | None = None):
        super().__init__()
        self.config = config or T5Config()
        self.shared = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.encoder = T5Stack(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.shared(input_ids)
        # Bias is computed once (first block holds the table) and reused per layer.
        position_bias = self.encoder.block[0].layer[0].SelfAttention.compute_bias(
            input_ids.shape[-1], input_ids.device
        )
        for block in self.encoder.block:
            x = block(x, position_bias)
        return self.encoder.final_layer_norm(x)
