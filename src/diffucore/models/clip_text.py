"""CLIP ViT-L/14 text transformer — the SD1.5 conditioner.

Implements the CLIP text encoder (Radford et al., 2021) as used by Stable
Diffusion. Submodule and parameter names mirror the on-disk HF CLIP keys (under
``cond_stage_model.transformer.``, i.e. ``text_model.*``) so a ``strict=True``
load is the correctness check.

Contract:
    forward(token_ids: LongTensor[B, 77], clip_skip: int = 1)
        -> hidden_states: FloatTensor[B, 77, 768]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLIPTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class CLIPEmbeddings(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        # Non-persistent: position_ids is the constant arange(0..max), not a learned
        # weight. Many SDXL finetunes drop it from their state dict, so requiring it
        # would reject otherwise-valid checkpoints; it is regenerated here identically.
        self.register_buffer(
            "position_ids",
            torch.arange(cfg.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq = token_ids.shape[-1]
        positions = self.position_ids[:, :seq]
        return self.token_embedding(token_ids) + self.position_embedding(positions)


class CLIPAttention(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape

        def shape(t):
            return t.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = shape(self.q_proj(x)), shape(self.k_proj(x)), shape(self.v_proj(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        out = out.transpose(1, 2).reshape(b, seq, -1)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(quick_gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = CLIPAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = CLIPMLP(cfg)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), causal_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layers = nn.ModuleList(CLIPEncoderLayer(cfg) for _ in range(cfg.num_layers))


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.embeddings = CLIPEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)


class CLIPTextEncoder(nn.Module):
    def __init__(self, config: CLIPTextConfig | None = None):
        super().__init__()
        self.config = config or CLIPTextConfig()
        self.text_model = CLIPTextTransformer(self.config)

    def forward(self, token_ids: torch.Tensor, clip_skip: int = 1) -> torch.Tensor:
        tm = self.text_model
        x = tm.embeddings(token_ids)

        seq = token_ids.shape[-1]
        causal_mask = torch.full((seq, seq), float("-inf"), device=x.device, dtype=x.dtype).triu(1)

        # Run all but the last `clip_skip - 1` layers; clip_skip=1 runs all 12.
        num_run = len(tm.encoder.layers) - (clip_skip - 1)
        for layer in tm.encoder.layers[:num_run]:
            x = layer(x, causal_mask)

        if clip_skip == 1:
            x = tm.final_layer_norm(x)
        return x
