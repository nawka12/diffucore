"""Mistral-3 decoder LM — FLUX.2's text encoder.

FLUX.2 (Black Forest Labs, 2025) drops the FLUX.1 T5-XXL + CLIP pair and instead
conditions on the hidden states of a Mistral-Small-3 (24B) language model. This
implements that LM as a hidden-state encoder: a standard pre-norm decoder
transformer with RMSNorm, rotary position embeddings, grouped-query attention,
and a SwiGLU MLP (Touvron et al. / Jiang et al., the Llama/Mistral lineage).

Submodule and parameter names mirror the HuggingFace ``MistralForCausalLM`` keys
(``model.embed_tokens``, ``model.layers.{i}.{self_attn,mlp,*layernorm}``,
``model.norm``) so a ``strict=True`` load is the correctness check; the unused
``lm_head`` is dropped by the loader. Widths/depths are derived from the
checkpoint (:meth:`MistralConfig.from_state_dict`).

⚠ Build-to-spec: this is implemented against the published Mistral-3 architecture
and Mistral-Small-3.1-24B config (head_dim 128, rope θ 1e6, full causal attention),
but has **not** been numerically verified against FLUX.2 weights. The exact layer
whose hidden state FLUX.2 consumes (here: final-norm ``last_hidden_state``) is the
most likely choice and the first thing to confirm against a reference. Mistral's
Tekken tokenizer is not vendored — supply ``tokenizer.json`` via the tokenizer.

Contract:
    forward(input_ids: LongTensor[B, L]) -> last_hidden_state: FloatTensor[B, L, dim]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._norm import RMSNorm, _rotate_half


@dataclass
class MistralConfig:
    vocab_size: int = 131072
    hidden_size: int = 5120
    intermediate_size: int = 32768
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-5

    @classmethod
    def from_state_dict(
        cls,
        sd: Mapping[str, torch.Tensor],
        prefix: str = "",
        *,
        head_dim: int = 128,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-5,
    ) -> "MistralConfig":
        emb = sd[prefix + "model.embed_tokens.weight"]
        n_layers = 0
        while f"{prefix}model.layers.{n_layers}.self_attn.q_proj.weight" in sd:
            n_layers += 1
        q_rows = sd[prefix + "model.layers.0.self_attn.q_proj.weight"].shape[0]
        kv_rows = sd[prefix + "model.layers.0.self_attn.k_proj.weight"].shape[0]
        ffn = sd[prefix + "model.layers.0.mlp.gate_proj.weight"].shape[0]
        return cls(
            vocab_size=emb.shape[0],
            hidden_size=emb.shape[1],
            intermediate_size=ffn,
            num_hidden_layers=n_layers,
            num_attention_heads=q_rows // head_dim,
            num_key_value_heads=kv_rows // head_dim,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )


def _rope_tables(positions: torch.Tensor, head_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]       # [L, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                      # [L, head_dim]
    return emb.cos()[None, None], emb.sin()[None, None]          # [1, 1, L, head_dim]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


class MistralAttention(nn.Module):
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        q = self.q_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, length, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        # GQA: expand the kv heads to the query-head count.
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, length, -1)
        return self.o_proj(out)


class MistralMLP(nn.Module):
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralLayer(nn.Module):
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        self.self_attn = MistralAttention(cfg)
        self.mlp = MistralMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MistralModel(nn.Module):
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(MistralLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.cfg = cfg

    def forward(self, input_ids: torch.Tensor, hidden_layers: list[int] | None = None):
        x = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
        cos, sin = _rope_tables(positions, self.cfg.head_dim, self.cfg.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        targets = set(hidden_layers or [])
        captured: dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, cos, sin)
            if idx in targets:
                captured[idx] = x
        if hidden_layers is not None:
            return [captured[i] for i in hidden_layers]
        return self.norm(x)


class MistralTextEncoder(nn.Module):
    """Mistral-3 LM used as a hidden-state text encoder for FLUX.2."""

    def __init__(self, config: MistralConfig | None = None):
        super().__init__()
        self.config = config or MistralConfig()
        self.model = MistralModel(self.config)

    def forward(self, input_ids: torch.Tensor, hidden_layers: list[int] | None = None):
        return self.model(input_ids, hidden_layers=hidden_layers)
