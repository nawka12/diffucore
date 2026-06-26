"""Qwen3.5 hybrid (Mamba2 SSM + gated attention) text encoder — experimental.

Anima normally conditions on Qwen3-0.6B-base (see ``qwen3_text.py``). The
``cosmos-qwen3.5`` project swaps in a Qwen3.5 text backbone as a newer semantic
encoder. Unlike Qwen3 these models are *hybrid*: a 3:1 stack where every 4th
decoder layer is gated self-attention and the rest are Mamba2-style selective
state-space (SSM) blocks. Two variants are supported, told apart from the
checkpoint by :meth:`Qwen35Config.from_state_dict`:

  * **4B Anima encoder** (``qwen35_4b.safetensors``): 32 layers (attention at
    3,7,…,31; layer 31 has no MLP), hidden 2560, and a baked-in projection head
    ``Linear 2560->1024 -> ExpRMSNorm -> SiLU -> Linear 1024->1024``.
  * **0.8B base** (``qwen_3_5_08b_base``): 24 layers (attention at 3,7,…,23),
    hidden 1024, and *no* projection — just a plain final RMSNorm, so its output
    is already the 1024-d the adapter expects (same convention as Qwen3-0.6B).

Either way the output is 1024-d — the *same* interface Qwen3-0.6B exposes, so
these encoders are drop-in interchangeable at the Anima boundary.

Ported from the MIT-licensed reference node ``GumGum10/comfyui-qwen35-anima``
(which reverse-engineered ``nightknocker/cosmos-qwen3.5``). Submodule and
parameter names mirror the on-disk checkpoint (``embed_tokens.*``,
``layers.{i}.*``, and ``norm.{0,1,3}.*`` or plain ``norm.*``) so a strict load
of the backbone succeeds without remapping (the loader strips any
``model.language_model.`` prefix and drops the base model's vision/MTP heads).

Two reference details are load-bearing:
  * The late norm uses an ``exp(weight)`` parameterization (:class:`ExpRMSNorm`).
    Its learned weights are ~-0.003; read as a plain RMSNorm scale they collapse
    every token to the same vector. ``exp(-0.003) ~ 1`` keeps near-identity
    scaling and preserves token diversity.
  * The SSM block's input projections are named off-by-one vs intuition:
    ``in_proj_b`` feeds the input-dependent timestep ``dt`` and ``in_proj_a``
    feeds the ``D`` skip term (verified against the checkpoint, not a typo).

Experimental: not bit-verified against an oracle (the 4B weights are fp8 and
need a GPU). It is shape- and strict-load-verified offline. The reference's
optional calibration / Procrustes-alignment refinements and its image ViT are
out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._norm import RMSNorm, _rotate_half


@dataclass
class Qwen35Config:
    # Defaults describe the 4B Anima encoder; ``from_state_dict`` derives the rest
    # of the family (e.g. the 0.8B base) from a checkpoint.
    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    output_dim: int = 1024              # context width handed to the LLM-Adapter
    output_projection: bool = True       # 4B: Linear→ExpRMSNorm→SiLU→Linear head.
                                         # False (0.8B base): plain final RMSNorm.
    num_hidden_layers: int = 32
    # Layer roles (the rest are SSM; the listed layers are gated self-attention).
    self_attn_layers: tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31)
    no_mlp_layers: tuple[int, ...] = (31,)   # final attn layer has no MLP
    # Gated self-attention.
    num_attention_heads: int = 16
    num_key_value_heads: int = 4         # GQA ratio 4
    head_dim: int = 256
    rope_theta: float = 1_000_000.0
    # Mamba2 SSM. conv_dim = d_ssm + 2·n_groups·d_state = 4096 + 2·32·64 = 8192.
    ssm_d_ssm: int = 4096                # = n_groups · ssm_head_dim (the gated path)
    ssm_conv_dim: int = 8192
    ssm_n_groups: int = 32               # == n_heads (1 head per group)
    ssm_head_dim: int = 128
    ssm_d_state: int = 64
    ssm_conv_kernel: int = 4
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_state_dict(cls, sd) -> "Qwen35Config":
        """Derive the config from a bare-key Qwen3.5 backbone state dict (prefix
        already stripped, vision/MTP heads already dropped). Sizes, the layer
        roles (SSM vs attention, which layers carry an MLP) and the output-head
        kind are all read off the tensors; ``rope_theta`` / eps are family
        constants the checkpoint doesn't record."""
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
            ssm_d_state=(conv_dim - d_ssm) // (2 * n_groups),
            ssm_conv_kernel=sd[f"layers.{si}.linear_attn.conv1d.weight"].shape[-1],
        )


class ExpRMSNorm(nn.Module):
    """RMSNorm whose scale is ``exp(weight)`` instead of ``weight``.

    The late-norm's learned weights sit at ~-0.003; a plain RMSNorm would read
    that as "scale to ~0" and collapse every token to the same vector. With the
    exponential parameterization, ~0 means ``exp(0) ~ 1`` (near-identity), which
    preserves token diversity. Computed in fp32, matching :class:`RMSNorm`.
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


def _rope_cos_sin(head_dim: int, seq_len: int, theta: float, device, dtype):
    """RoPE tables of shape ``(1, 1, seq_len, head_dim)`` for [B, H, T, D] q/k."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                 # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)          # (T, D)
    cos = emb.cos()[None, None].to(dtype)
    sin = emb.sin()[None, None].to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D);  cos/sin: (1, 1, T, D).
    return x * cos + _rotate_half(x) * sin


class SSMBlock(nn.Module):
    """Mamba2-style selective state-space block (ref: ``state-spaces/mamba``).

    ``in_proj_qkv`` → conv1d → SiLU → split into ``x`` (d_ssm) + ``B`` + ``C``;
    ``in_proj_z`` is the gate that *bypasses* the conv; ``in_proj_b`` / ``in_proj_a``
    produce the per-head input-dependent ``dt`` / ``D`` (skip). The recurrence is
    a proper outer-product state of width ``d_state``.
    """

    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.n_groups = cfg.ssm_n_groups          # == n_heads
        self.head_dim = cfg.ssm_head_dim
        self.d_ssm = cfg.ssm_d_ssm
        self.d_state = cfg.ssm_d_state
        conv_dim = cfg.ssm_conv_dim

        self.in_proj_qkv = nn.Linear(cfg.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.d_ssm, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.n_groups, bias=False)   # → D skip
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.n_groups, bias=False)   # → dt
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, cfg.ssm_conv_kernel, groups=conv_dim,
            padding=cfg.ssm_conv_kernel - 1, bias=False,
        )
        self.out_proj = nn.Linear(self.d_ssm, cfg.hidden_size, bias=False)
        self.norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)   # per-head, gated
        self.A_log = nn.Parameter(torch.zeros(self.n_groups))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_groups))

    def _scan(self, x, B_state, C_state, dt_in, D_in):
        """Sequential selective scan (fp32).

        x:       (B, L, H, P)   H heads, P head_dim
        B_state: (B, L, G, S)   G groups (== H), S d_state
        C_state: (B, L, G, S)
        dt_in:   (B, L, H)      input-dependent step
        D_in:    (B, L, H)      input-dependent skip
        returns: (B, L, H, P)
        """
        Bsz, L, H, P = x.shape
        S = B_state.shape[-1]
        A = -torch.exp(self.A_log.float())                # (H,) negative
        dt_bias = self.dt_bias.float()                    # (H,)
        h = torch.zeros(Bsz, H, P, S, device=x.device, dtype=torch.float32)

        x_f, B_f, C_f = x.float(), B_state.float(), C_state.float()
        dt_f, D_f = dt_in.float(), D_in.float()
        out = []
        for t in range(L):
            dt_t = F.softplus(dt_f[:, t] + dt_bias)        # (B, H)
            dA = torch.exp(dt_t * A)                        # (B, H)
            dBx = dt_t[..., None, None] * torch.einsum("bhp,bhs->bhps", x_f[:, t], B_f[:, t])
            h = dA[..., None, None] * h + dBx              # (B, H, P, S)
            y_t = torch.einsum("bhps,bhs->bhp", h, C_f[:, t]) + D_f[:, t][..., None] * x_f[:, t]
            out.append(y_t)
        return torch.stack(out, dim=1).to(x.dtype)         # (B, L, H, P)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        z = self.in_proj_z(hidden_states)                  # gate, bypasses conv
        xBC = self.in_proj_qkv(hidden_states)
        dt_in = self.in_proj_b(hidden_states)              # (B, L, H)
        D_in = self.in_proj_a(hidden_states)               # (B, L, H)

        xBC = self.conv1d(xBC.transpose(1, 2))[..., :L]    # causal depthwise conv
        xBC = F.silu(xBC.transpose(1, 2))

        x, B_conv, C_conv = torch.split(
            xBC, [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1
        )
        x = x.reshape(B, L, self.n_groups, self.head_dim)
        B_state = B_conv.reshape(B, L, self.n_groups, self.d_state)
        C_state = C_conv.reshape(B, L, self.n_groups, self.d_state)

        y = self._scan(x, B_state, C_state, dt_in, D_in)   # (B, L, H, P)
        y = self.norm(y)                                   # per-head RMSNorm
        y = y.reshape(B, L, -1) * F.silu(z)                # RMSNormGated
        return self.out_proj(y)


class GatedSelfAttention(nn.Module):
    """GQA self-attention with a SiLU-gated output and per-head q/k RMSNorm.

    ``q_proj`` emits ``2·inner`` = query + gate; the gate multiplies the attention
    output (after re-flattening) before ``o_proj``.
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
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        B, L, _ = x.shape
        q, gate = self.q_proj(x).chunk(2, dim=-1)
        q = self.q_norm(q.view(B, L, self.num_heads, self.head_dim)).transpose(1, 2)
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
        out = out * F.silu(gate)
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
    """One decoder layer: pre-norm SSM *or* gated attention, then (optional) MLP."""

    def __init__(self, cfg: Qwen35Config, use_ssm: bool, has_mlp: bool):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if use_ssm:
            self.linear_attn = SSMBlock(cfg)
        else:
            self.self_attn = GatedSelfAttention(cfg)
        if has_mlp:
            self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.mlp = _MLP(cfg)

    def forward(self, x, cos, sin, attn_mask):
        h = self.input_layernorm(x)
        if self.use_ssm:
            x = x + self.linear_attn(h)
        else:
            x = x + self.self_attn(h, cos, sin, attn_mask)
        if self.has_mlp:
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen35TextEncoder(nn.Module):
    """Qwen3.5 4B hybrid text encoder: tokens → 1024-dim source hidden states.

    Forward:
        input_ids:      LongTensor (B, T).
        attention_mask: optional (B, T) {0,1} padding mask. ``None`` (Anima's
                        path — the tokenizer doesn't pad) runs pure causal
                        attention via SDPA's fast path.

    Returns:
        hidden_states: FloatTensor (B, T, output_dim=1024).
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
        # Output head. The 4B Anima encoder bakes in a projection (``norm.0/1/3``:
        # Linear 2560→1024 → ExpRMSNorm → SiLU → Linear, SiLU at index 2 carrying
        # no params). The raw base models (e.g. 0.8B) have no projection — just a
        # plain final RMSNorm (``norm.weight``) at hidden width, which is already
        # the 1024-d the LLM-Adapter expects (same convention as Qwen3-0.6B-base).
        if cfg.output_projection:
            self.norm = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.output_dim, bias=True),
                ExpRMSNorm(cfg.output_dim, eps=cfg.rms_norm_eps),
                nn.SiLU(),
                nn.Linear(cfg.output_dim, cfg.output_dim, bias=True),
            )
        else:
            self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed_tokens(input_ids)
        B, L, _ = x.shape
        cos, sin = _rope_cos_sin(self.cfg.head_dim, L, self.cfg.rope_theta, x.device, x.dtype)

        # None → pure causal (SDPA is_causal). A padding mask becomes an additive
        # float mask combining causality with the padded key positions.
        attn_mask = None
        if attention_mask is not None:
            neg = torch.finfo(x.dtype).min / 4
            causal = torch.empty(L, L, dtype=x.dtype, device=x.device).fill_(neg).triu_(1)
            pad = (1.0 - attention_mask.to(x.dtype)).view(B, 1, 1, L) * neg
            attn_mask = causal + pad

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        return self.norm(x)
