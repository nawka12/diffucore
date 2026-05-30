"""OpenCLIP ViT-bigG/14 text transformer — SDXL's second text encoder.

Implements the OpenCLIP text tower (Ilharco et al., 2021; Radford et al., 2021)
used as SDXL's ``conditioner.embedders.1``. Submodule and parameter names mirror
the on-disk OpenCLIP keys (under ``conditioner.embedders.1.model.``) so a
``strict=True`` load is the correctness check.

SDXL consumes two things from this encoder:
  - the **penultimate** hidden state (1280-d, no ``ln_final``), concatenated with
    CLIP-L's penultimate hidden state to form the 2048-d cross-attention context;
  - the **pooled** text embedding (``ln_final`` at the EOS position, projected by
    ``text_projection``), which feeds the UNet's added (size/pooled) conditioning.

Contract:
    forward(token_ids: LongTensor[B, 77], clip_skip: int = 2)
        -> (hidden: FloatTensor[B, 77, 1280], pooled: FloatTensor[B, 1280])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OpenCLIPTextConfig:
    vocab_size: int = 49408
    width: int = 1280
    num_layers: int = 32
    num_heads: int = 20
    mlp_dim: int = 5120
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5


class MultiheadAttention(nn.Module):
    """torch-style MHA with a fused ``in_proj`` (q/k/v stacked), as OpenCLIP stores it."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape(t):
            return t.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, seq, -1)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: OpenCLIPTextConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.width, cfg.mlp_dim)
        self.c_proj = nn.Linear(cfg.mlp_dim, cfg.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg: OpenCLIPTextConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.attn = MultiheadAttention(cfg.width, cfg.num_heads)
        self.ln_2 = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, config: OpenCLIPTextConfig | None = None):
        super().__init__()
        cfg = config or OpenCLIPTextConfig()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(cfg.max_position_embeddings, cfg.width))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            ResidualAttentionBlock(cfg) for _ in range(cfg.num_layers)
        )
        self.ln_final = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.text_projection = nn.Parameter(torch.empty(cfg.width, cfg.width))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, token_ids: torch.Tensor, clip_skip: int = 2):
        x = self.token_embedding(token_ids) + self.positional_embedding[: token_ids.shape[-1]]

        seq = token_ids.shape[-1]
        causal_mask = torch.full((seq, seq), float("-inf"), device=x.device, dtype=x.dtype).triu(1)

        blocks = self.transformer.resblocks
        penult_at = len(blocks) - clip_skip  # grab the hidden state after this block index
        hidden = None
        for i, block in enumerate(blocks):
            x = block(x, causal_mask)
            if i == penult_at:
                hidden = x

        final = self.ln_final(x)
        eos = token_ids.argmax(dim=-1)
        pooled = final[torch.arange(final.shape[0], device=final.device), eos] @ self.text_projection
        return hidden, pooled
