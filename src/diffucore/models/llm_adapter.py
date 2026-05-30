"""LLM-Adapter — Anima's 6-block bridge between Qwen3 hidden states and the DiT.

Anima conditions on its prompt twice. The Qwen3 0.6B encoder (DT3) consumes
the Qwen-tokenized prompt and produces *source hidden states* (semantics).
Separately, the same prompt is tokenized with the T5 tokenizer to produce
*target token IDs* (positional/word-aware tokens). The LLM-Adapter is a small
transformer that uses the T5 tokens as a "query stream" (embedded via its own
32128-row table) and cross-attends to the Qwen3 hidden states; its output is
the 1024-dim context the DiT cross-attends to.

Architecture::

    embed(target_input_ids)               -> x (B, L_t5, 1024)
    in_proj = Identity  (model_dim == target_dim for Anima)
    for 6 blocks:
        x += self_attn (RMSNorm(x), q_rope=target_pos, mask=target_mask)
        x += cross_attn(RMSNorm(x), context=src_hidden,
                        q_rope=target_pos, k_rope=source_pos,
                        mask=source_mask)
        x += mlp       (RMSNorm(x))            # Linear(bias) -> GELU -> Linear(bias)
    return final_norm(out_proj(x))             # both at target_dim

Attention details:
    n_heads = 16, head_dim = 64 → inner_dim = 1024 (= model_dim).
    per-head q_norm / k_norm (RMSNorm over head_dim) applied **before** RoPE.
    o_proj inside each Attention has no bias; the outer ``out_proj`` does.
    RoPE θ = 10_000 (Anima's adapter uses the standard θ, not Qwen3's 1e6).
    Eager attention (matmul → softmax → matmul) to leave room for bit-identity
    once an in-process oracle is available.

Submodule and parameter names mirror the on-disk ``net.llm_adapter.*`` keys
so the standard ``_load_sub(module, sd, "net.llm_adapter.")`` strict-load
works without remapping.

Verification status (DT4): key-set match + strict load + forward-shape +
determinism. Numerical bit-match against the ComfyUI reference is deferred
to DT7 (end-to-end pipeline) because ComfyUI's local install has native deps
that can't be imported in this venv; the end-to-end image comparison is a
stronger correctness signal anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMAdapterConfig:
    target_vocab: int = 32128       # T5 vocab — Anima embeds T5 token IDs here
    target_dim: int = 1024          # adapter output dim (= DiT cross-attn ctx)
    source_dim: int = 1024          # Qwen3 hidden dim consumed by cross-attn
    model_dim: int = 1024           # internal hidden dim
    num_layers: int = 6
    num_heads: int = 16
    head_dim: int = 64
    mlp_ratio: float = 4.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6


class _RMSNorm(nn.Module):
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
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D);  cos, sin: (1, T, D) → (1, 1, T, D)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class _RotaryEmbedding(nn.Module):
    """RoPE over head_dim, with configurable base (θ = 10_000 for Anima)."""
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        # position_ids: (1, T) int.  Returns (cos, sin) each (1, T, head_dim).
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class _Attention(nn.Module):
    """Multi-head attention with per-head q/k RMSNorm + RoPE.

    Self-attn when ``context is None`` (k=v=x, both q and k get the same
    positional embedding). Cross-attn when context is given (k=v from
    context; q uses ``q_pe`` and k uses ``k_pe`` — the adapter applies RoPE
    to cross-attn keys based on the *source* positions, not the target).
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, head_dim: int, eps: float):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)
        self.q_norm = _RMSNorm(head_dim, eps=eps)
        self.k_norm = _RMSNorm(head_dim, eps=eps)

    def forward(self, x, context, q_pe, k_pe, key_mask):
        ctx = x if context is None else context
        B, Tq, _ = x.shape
        Tk = ctx.shape[1]

        q = self.q_norm(self.q_proj(x).view(B, Tq, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(ctx).view(B, Tk, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(ctx).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, *q_pe)
        k = _apply_rope(k, *k_pe)

        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        if key_mask is not None:
            # key_mask: (B, 1, 1, Tk) bool — True = keep, False = mask out.
            attn = attn.masked_fill(~key_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class _Block(nn.Module):
    """One LLM-Adapter layer: self-attn, cross-attn, MLP — all pre-norm with
    RMSNorm; MLP is ``Linear(bias) → GELU → Linear(bias)`` (default ``nn.Linear``
    bias matches the checkpoint's ``mlp.0.bias`` / ``mlp.2.bias``)."""

    def __init__(self, cfg: LLMAdapterConfig):
        super().__init__()
        self.norm_self_attn = _RMSNorm(cfg.model_dim, eps=cfg.rms_norm_eps)
        self.self_attn = _Attention(cfg.model_dim, cfg.model_dim, cfg.num_heads, cfg.head_dim, cfg.rms_norm_eps)
        self.norm_cross_attn = _RMSNorm(cfg.model_dim, eps=cfg.rms_norm_eps)
        self.cross_attn = _Attention(cfg.model_dim, cfg.source_dim, cfg.num_heads, cfg.head_dim, cfg.rms_norm_eps)
        self.norm_mlp = _RMSNorm(cfg.model_dim, eps=cfg.rms_norm_eps)
        intermediate = int(cfg.model_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.model_dim, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, cfg.model_dim),
        )

    def forward(self, x, context, target_pe, source_pe, target_mask, source_mask):
        # Self-attn: q and k both use target positions; mask keys via target_mask.
        x = x + self.self_attn(
            self.norm_self_attn(x), None, target_pe, target_pe, target_mask,
        )
        # Cross-attn: q from target stream, k/v from source; q gets target RoPE,
        # k gets source RoPE; mask keys via source_mask.
        x = x + self.cross_attn(
            self.norm_cross_attn(x), context, target_pe, source_pe, source_mask,
        )
        # MLP.
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    """Anima's 6-block bridge: ``(qwen3_hidden, t5_token_ids) → DiT context``."""

    def __init__(self, cfg: LLMAdapterConfig | None = None):
        super().__init__()
        self.cfg = cfg or LLMAdapterConfig()
        cfg = self.cfg

        self.embed = nn.Embedding(cfg.target_vocab, cfg.target_dim)
        self.in_proj = nn.Identity() if cfg.model_dim == cfg.target_dim else nn.Linear(cfg.target_dim, cfg.model_dim)
        self.rotary_emb = _RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.num_layers)])
        self.out_proj = nn.Linear(cfg.model_dim, cfg.target_dim)
        self.norm = _RMSNorm(cfg.target_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        source_hidden_states: torch.Tensor,        # (B, L_qwen, source_dim)
        target_input_ids: torch.Tensor,            # (B, L_t5) int
        target_attention_mask: torch.Tensor | None = None,  # (B, L_t5) bool/{0,1}
        source_attention_mask: torch.Tensor | None = None,  # (B, L_qwen) bool/{0,1}
    ) -> torch.Tensor:                              # (B, L_t5, target_dim)
        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        device = x.device

        target_pos = torch.arange(x.shape[1], device=device).unsqueeze(0)
        source_pos = torch.arange(context.shape[1], device=device).unsqueeze(0)
        target_pe = self.rotary_emb(target_pos, dtype=x.dtype)
        source_pe = self.rotary_emb(source_pos, dtype=x.dtype)

        # 2D (B, L) masks → 4D (B, 1, 1, L) bool for the key dim.
        def _key_mask(m):
            if m is None:
                return None
            return m.bool().view(m.shape[0], 1, 1, -1)

        target_mask = _key_mask(target_attention_mask)
        source_mask = _key_mask(source_attention_mask)

        for block in self.blocks:
            x = block(x, context, target_pe, source_pe, target_mask, source_mask)
        return self.norm(self.out_proj(x))
