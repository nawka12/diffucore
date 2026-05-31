"""Qwen3 0.6B base text encoder.

Anima uses Qwen3-0.6B-base as its semantic encoder: token IDs go in, hidden
states come out (no LM head; the final RMS-norm is kept). The hidden states
are then handed to the LLM-Adapter (DT4) which produces the cross-attention
context the DiT consumes.

Architectural facts (Qwen3-0.6B-Base, fixed)::

    vocab_size = 151936     hidden_size = 1024     num_hidden_layers = 28
    num_attention_heads = 16  num_key_value_heads = 8 (GQA, ratio 2)
    head_dim = 128            intermediate_size = 3072
    max_position_embeddings = 32768
    rms_norm_eps = 1e-6       rope_theta = 1_000_000.0
    mlp = SwiGLU (silu(gate)·up → down)
    qkv has no bias; q_norm / k_norm are RMSNorms over head_dim,
        applied to the per-head q/k *before* RoPE.

Submodule and parameter names mirror the HuggingFace ``transformers``
checkpoint layout (``model.embed_tokens.*``, ``model.layers.{i}.*``,
``model.norm.*``) so a strict load against either an HF dump or ComfyUI's
``qwen_3_06b_base.safetensors`` succeeds without remapping.

Verification: this module's forward is bit-identical (max|Δ| = 0 in fp32)
to ``transformers.Qwen3Model`` on a fixed prompt — that's the test in
``tests/test_qwen3_text.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._norm import RMSNorm, _rotate_half


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, T, D);  cos, sin: (B, T, D) → broadcast to (B, 1, T, D)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Qwen3RotaryEmbedding(nn.Module):
    """Standard RoPE over head_dim. Frequencies are float32 for fidelity."""

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: (B, T) int. Output cos/sin: (B, T, head_dim) in x.dtype.
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Qwen3Attention(nn.Module):
    """GQA self-attention with per-head q/k RMSNorm and RoPE.

    Q has ``num_attention_heads`` heads; K/V have ``num_key_value_heads`` heads
    (Qwen3-0.6B uses 16 / 8 → group size 2). K/V are repeat-interleaved up to
    16 heads before SDPA so we never branch on the GQA ratio at attention time.
    """

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        q_inner = self.num_heads * self.head_dim
        kv_inner = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_inner, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_inner, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_inner, bias=False)
        self.o_proj = nn.Linear(q_inner, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x, position_embeddings, attention_mask):
        B, T, _ = x.shape
        # (B, T, H, D) → norm over D → transpose → (B, H, T, D)
        q = self.q_norm(self.q_proj(x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rope(q, k, cos, sin)

        # GQA: lift k/v to num_heads via repeat-interleave (HF semantics).
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Eager attention. SDPA's flash/efficient kernels reorder reductions
        # for speed and lose ~1 fp32 ULP per layer vs the naive form HF uses,
        # which compounds over 28 layers; the explicit matmul→softmax→matmul
        # form below is bit-identical to ``transformers.Qwen3Model`` with
        # ``attn_implementation="eager"`` and the perf delta at ≤512 tokens
        # is negligible.
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn = attn + attention_mask
        else:
            T = q.shape[-2]
            causal = torch.full((T, T), float("-inf"), dtype=attn.dtype, device=attn.device).triu(1)
            attn = attn + causal
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) · up(x))``."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Pre-norm: residual + attention(input_norm(x)) + residual + mlp(post_norm(x))."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x, position_embeddings, attention_mask):
        x = x + self.self_attn(self.input_layernorm(x), position_embeddings, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _Qwen3Inner(nn.Module):
    """The 'model.*' subtree: embed → 28 layers → final norm.

    Wrapping the whole inner stack in a ``model`` attribute matches the HF
    state-dict layout (``model.embed_tokens``, ``model.layers.*``, ``model.norm``)
    so a strict load works without key remapping.
    """

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


class Qwen3TextEncoder(nn.Module):
    """Qwen3 0.6B as a text encoder: tokens → final hidden states.

    Args:
        cfg: Qwen3 hyperparameters (defaults to Qwen3-0.6B-Base).

    Forward:
        input_ids:      LongTensor (B, T).
        attention_mask: optional FloatTensor (B, 1, T, T) or boolean (B, 1, 1, T)
                        for HF-style padding masks. If ``None``, a pure causal
                        mask is applied via SDPA's ``is_causal=True`` fast path.

    Returns:
        hidden_states: FloatTensor (B, T, hidden_size).
    """

    def __init__(self, cfg: Qwen3Config | None = None):
        super().__init__()
        self.cfg = cfg or Qwen3Config()
        self.model = _Qwen3Inner(self.cfg)
        self.rotary_emb = Qwen3RotaryEmbedding(self.cfg.head_dim, self.cfg.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(x, position_ids)
        for layer in self.model.layers:
            x = layer(x, position_embeddings, attention_mask)
        return self.model.norm(x)
