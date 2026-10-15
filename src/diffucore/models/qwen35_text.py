"""Qwen3.5 hybrid (Gated DeltaNet + gated attention) text encoder, experimental.

The ``cosmos-qwen3.5`` swap for Anima's Qwen3-0.6B encoder: every 4th layer is
gated self-attention, the rest Gated DeltaNet linear-attention blocks. Two
variants, told apart by :meth:`Qwen35Config.from_state_dict`: the 4B Anima
encoder (32 layers, hidden 2560, a projection head to 1024) and the 0.8B base
(24 layers, hidden 1024, plain final RMSNorm). Both emit 1024-d, like
Qwen3-0.6B.

The math follows the Qwen3.5 text model in Hugging Face transformers
(``models/qwen3_5/modeling_qwen3_5.py``, Apache-2.0), text-only, no KV cache.
Parameter names mirror the checkpoint so the backbone strict-loads. The 4B
projection head (``norm.0/1/3`` with :class:`ExpRMSNorm`) comes from
``GumGum10/comfyui-qwen35-anima`` (MIT).

The 0.8B path is verified layer by layer against transformers; the 4B path is
only shape-verified (no weights at hand).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._norm import _rotate_half

# Linear-attention key head dim. Not derivable from shapes (conv_dim only fixes
# num_k_heads * head_k_dim); 128 for every Qwen3.5 size (``linear_key_head_dim``).
_LINEAR_KEY_HEAD_DIM = 128


@dataclass
class Qwen35Config:
    # Defaults describe the 4B encoder; ``from_state_dict`` derives the others.
    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    output_dim: int = 1024              # context width handed to the LLM-Adapter
    output_projection: bool = True       # 4B head; False = plain final RMSNorm (0.8B)
    num_hidden_layers: int = 32
    # Layer roles (the rest are linear attention; the listed layers are gated self-attention).
    self_attn_layers: tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31)
    no_mlp_layers: tuple[int, ...] = (31,)   # final attn layer has no MLP
    # Gated self-attention.
    num_attention_heads: int = 16
    num_key_value_heads: int = 4         # GQA ratio 4
    head_dim: int = 256
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25  # only this fraction of head_dim is rotated
    # Gated DeltaNet. conv_dim = 2·key_dim + value_dim = 2·16·128 + 32·128 = 8192.
    ssm_d_ssm: int = 4096                # value_dim = n_groups · ssm_head_dim
    ssm_conv_dim: int = 8192
    ssm_n_groups: int = 32               # value heads (A_log / dt_bias width)
    ssm_head_dim: int = 128              # value head dim
    ssm_d_state: int = _LINEAR_KEY_HEAD_DIM   # key head dim; key heads = key_dim / this
    ssm_conv_kernel: int = 4
    rms_norm_eps: float = 1e-6

    @property
    def ssm_key_dim(self) -> int:
        return (self.ssm_conv_dim - self.ssm_d_ssm) // 2

    @classmethod
    def from_state_dict(cls, sd) -> "Qwen35Config":
        """Derive the config from a bare-key backbone state dict (prefix stripped,
        vision/MTP heads dropped). ``rope_theta``, rotary fraction, eps and the
        linear key head dim are family constants."""
        vocab_size, hidden_size = sd["embed_tokens.weight"].shape
        n = 0
        while f"layers.{n}.input_layernorm.weight" in sd:
            n += 1
        self_attn = tuple(i for i in range(n) if f"layers.{i}.self_attn.q_proj.weight" in sd)
        no_mlp = tuple(i for i in range(n) if f"layers.{i}.mlp.gate_proj.weight" not in sd)
        mlp_i = next(i for i in range(n) if f"layers.{i}.mlp.gate_proj.weight" in sd)
        ai, si = self_attn[0], next(i for i in range(n) if f"layers.{i}.linear_attn.A_log" in sd)

        head_dim = sd[f"layers.{ai}.self_attn.q_norm.weight"].shape[0]
        q_inner = sd[f"layers.{ai}.self_attn.q_proj.weight"].shape[0] // 2   # gated: q + gate
        kv_inner = sd[f"layers.{ai}.self_attn.k_proj.weight"].shape[0]
        n_groups = sd[f"layers.{si}.linear_attn.A_log"].shape[0]
        conv_dim = sd[f"layers.{si}.linear_attn.in_proj_qkv.weight"].shape[0]
        d_ssm = sd[f"layers.{si}.linear_attn.in_proj_z.weight"].shape[0]
        has_proj = "norm.0.weight" in sd
        return cls(
            vocab_size=vocab_size, hidden_size=hidden_size,
            intermediate_size=sd[f"layers.{mlp_i}.mlp.gate_proj.weight"].shape[0],
            output_dim=(sd["norm.3.weight"].shape[0] if has_proj else hidden_size),
            output_projection=has_proj,
            num_hidden_layers=n, self_attn_layers=self_attn, no_mlp_layers=no_mlp,
            num_attention_heads=q_inner // head_dim, num_key_value_heads=kv_inner // head_dim,
            head_dim=head_dim,
            ssm_d_ssm=d_ssm, ssm_conv_dim=conv_dim, ssm_n_groups=n_groups,
            ssm_head_dim=sd[f"layers.{si}.linear_attn.norm.weight"].shape[0],
            ssm_d_state=_LINEAR_KEY_HEAD_DIM,
            ssm_conv_kernel=sd[f"layers.{si}.linear_attn.conv1d.weight"].shape[-1],
        )


class Qwen35RMSNorm(nn.Module):
    """RMSNorm scaled by ``1 + weight`` (checkpoint weights are stored centered
    on zero), fp32 like the reference. Not :class:`._norm.RMSNorm`, whose plain
    ``weight`` scale would read these as "scale to ~0".
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class _RMSNormGated(nn.Module):
    """Per-head RMSNorm with a plain ``weight`` scale, then ``· silu(gate)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(in_dtype)
        return (x * F.silu(gate.float())).to(in_dtype)


class ExpRMSNorm(nn.Module):
    """RMSNorm scaled by ``exp(weight)``. The learned weights sit at ~-0.003, which
    a plain RMSNorm would read as "scale to ~0". fp32, like :class:`RMSNorm`.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (torch.exp(self.weight.float()) * x).to(in_dtype)


def _rope_cos_sin(rot_dim: int, seq_len: int, theta: float, device, dtype):
    """RoPE tables of shape ``(1, 1, seq_len, rot_dim)`` for [B, H, T, D] q/k.

    Text-only: the three mrope position streams are equal, so the interleaved
    mrope reduces to plain RoPE over the first ``rot_dim`` channels.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                 # (T, rot/2)
    emb = torch.cat([freqs, freqs], dim=-1)          # (T, rot)
    cos = emb.cos()[None, None].to(dtype)
    sin = emb.sin()[None, None].to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D);  cos/sin: (1, 1, T, rot). Channels past ``rot`` pass through.
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    return torch.cat([x_rot * cos + _rotate_half(x_rot) * sin, x_pass], dim=-1)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size: int = 64):
    """Gated delta rule, chunked (the reference's ``torch_chunk_gated_delta_rule``
    with a triangular solve in place of its row-by-row loop).

    Per token: ``S <- exp(g)·S``; ``S += k ⊗ (beta·(v − Sᵀk))``; ``out = Sᵀq``,
    with q, k L2-normalized and q scaled by ``1/sqrt(d_k)``. Always fp32.

    q, k: (B, T, H, Dk);  v: (B, T, H, Dv);  g, beta: (B, T, H) -> (B, T, H, Dv)
    """
    in_dtype = q.dtype
    q, k = _l2norm(q.float()), _l2norm(k.float())
    q, k, v, g, beta = (x.transpose(1, 2).float() for x in (q, k, v, g, beta))
    B, H, T, Dk = k.shape
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    q = q * Dk ** -0.5

    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    q, k, v, k_beta, v_beta = (x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta))
    g = g.reshape(B, H, -1, chunk_size).cumsum(-1)

    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
    a = ((k_beta @ k.transpose(-1, -2)) * decay).tril(-1)
    eye = torch.eye(chunk_size, device=q.device, dtype=q.dtype)
    # (I + A)⁻¹ for strictly lower A: the WY-form correction that folds the
    # within-chunk delta updates into v and k.
    t = torch.linalg.solve_triangular(eye + a, eye.expand_as(a), upper=False, unitriangular=True)
    v_new_base = t @ v_beta
    k_cumdecay = t @ (k_beta * g.exp()[..., None])

    state = q.new_zeros(B, H, Dk, v.shape[-1])
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    out = torch.empty_like(v_new_base)
    for i in range(q.shape[2]):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill(causal, 0)
        v_new = v_new_base[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + attn @ v_new
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    out = out.reshape(B, H, -1, out.shape[-1])[:, :, :T]
    return out.transpose(1, 2).contiguous().to(in_dtype)


class GatedDeltaNet(nn.Module):
    """Qwen3.5 linear-attention block: ``in_proj_qkv`` -> causal depthwise conv
    -> SiLU -> (q, k, v); ``in_proj_b`` / ``in_proj_a`` give the delta-rule
    write strength ``beta`` and decay ``g``; ``in_proj_z`` gates the per-head
    output norm.
    """

    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.num_v_heads = cfg.ssm_n_groups
        self.head_v_dim = cfg.ssm_head_dim
        self.head_k_dim = cfg.ssm_d_state
        self.key_dim = cfg.ssm_key_dim
        self.value_dim = cfg.ssm_d_ssm
        self.num_k_heads = self.key_dim // self.head_k_dim
        conv_dim = cfg.ssm_conv_dim

        self.in_proj_qkv = nn.Linear(cfg.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)   # -> decay g
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)   # -> write strength
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, cfg.ssm_conv_kernel, groups=conv_dim,
            padding=cfg.ssm_conv_kernel - 1, bias=False,
        )
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)
        self.norm = _RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        if keep is not None:
            x = x * keep                                  # padded positions feed nothing
        B, L, _ = x.shape
        z = self.in_proj_z(x).reshape(B, L, self.num_v_heads, self.head_v_dim)
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[..., :L]).transpose(1, 2)

        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, L, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, L, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, L, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)

        y = _chunk_gated_delta_rule(q, k, v, g, beta)
        y = self.norm(y.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(y.reshape(B, L, -1))


class GatedSelfAttention(nn.Module):
    """GQA self-attention with per-head q/k RMSNorm and partial RoPE; ``q_proj``
    also emits, per head, a sigmoid gate applied to the output before ``o_proj``.
    """

    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.gqa = self.num_heads // self.num_kv_heads
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, 2 * inner, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, cfg.hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        B, L, _ = x.shape
        # q and gate interleave per head: (.., H, 2·head_dim) -> halves.
        q, gate = self.q_proj(x).view(B, L, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.gqa, dim=1)
        v = v.repeat_interleave(self.gqa, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None)
        )
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


class _MLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) · up(x))``."""

    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """One decoder layer: pre-norm linear attention *or* gated attention, then (optional) MLP."""

    def __init__(self, cfg: Qwen35Config, use_ssm: bool, has_mlp: bool):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp
        self.input_layernorm = Qwen35RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if use_ssm:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = GatedSelfAttention(cfg)
        if has_mlp:
            self.post_attention_layernorm = Qwen35RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.mlp = _MLP(cfg)

    def forward(self, x, cos, sin, attn_mask, keep=None):
        h = self.input_layernorm(x)
        if self.use_ssm:
            x = x + self.linear_attn(h, keep)
        else:
            x = x + self.self_attn(h, cos, sin, attn_mask)
        if self.has_mlp:
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen35TextEncoder(nn.Module):
    """Qwen3.5 hybrid text encoder: ``input_ids`` (B, T) → (B, T, 1024) hidden
    states. ``attention_mask=None`` (Anima's unpadded path) runs pure causal SDPA.
    """

    def __init__(self, cfg: Qwen35Config | None = None):
        super().__init__()
        self.cfg = cfg or Qwen35Config()
        cfg = self.cfg
        self_attn = set(cfg.self_attn_layers)
        no_mlp = set(cfg.no_mlp_layers)

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            HybridBlock(cfg, use_ssm=(i not in self_attn), has_mlp=(i not in no_mlp))
            for i in range(cfg.num_hidden_layers)
        ])
        # Output head: the 4B encoder's ``norm.0/1/3`` projection (SiLU at index
        # 2 has no params), or a plain final ``norm.weight`` on the base models.
        if cfg.output_projection:
            self.norm = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.output_dim, bias=True),
                ExpRMSNorm(cfg.output_dim, eps=cfg.rms_norm_eps),
                nn.SiLU(),
                nn.Linear(cfg.output_dim, cfg.output_dim, bias=True),
            )
        else:
            self.norm = Qwen35RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def run_layers(self, x: torch.Tensor, lo: int, hi: int,
                   attention_mask: torch.Tensor | None = None, caps=()):
        """Residual stream ``x`` through layers ``[lo, hi)`` (position 0 at the
        first token of ``x``). Returns ``(x, {i + 1: residual after layer i})``
        for the layer counts listed in ``caps``."""
        B, L, _ = x.shape
        cfg = self.cfg
        cos, sin = _rope_cos_sin(
            int(cfg.head_dim * cfg.partial_rotary_factor), L, cfg.rope_theta, x.device, x.dtype
        )

        # None → pure causal (SDPA is_causal). A padding mask becomes an additive
        # float mask combining causality with the padded key positions.
        attn_mask = keep = None
        if attention_mask is not None:
            neg = torch.finfo(x.dtype).min / 4
            causal = torch.empty(L, L, dtype=x.dtype, device=x.device).fill_(neg).triu_(1)
            pad = (1.0 - attention_mask.to(x.dtype)).view(B, 1, 1, L) * neg
            attn_mask = causal + pad
            keep = attention_mask.to(x.dtype).view(B, L, 1)

        out = {}
        for i in range(lo, hi):
            x = self.layers[i](x, cos, sin, attn_mask, keep)
            if i + 1 in caps:
                out[i + 1] = x
        return x, out

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x, _ = self.run_layers(self.embed_tokens(input_ids), 0, len(self.layers), attention_mask)
        return self.norm(x)
