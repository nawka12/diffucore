"""Krea 2 DiT: single-stream rectified-flow transformer (Raw and Turbo).

Follows krea-ai/krea-2 (Apache-2.0); module names mirror it, so the official,
ComfyUI bf16 and ComfyUI INT8 files load strictly with no key remap. Text and
image tokens share one stream of 28 blocks (6144 wide, 48 query / 12 KV heads
of 128, per-head QK RMSNorm, sigmoid-gated attention output, SwiGLU 16384,
3-axis RoPE (32, 48, 48) at θ=1000 with text at the origin). One timestep
projection ``tproj`` feeds every block, each adding its own learned offset.
The text path fuses 12 Qwen3-VL-4B hidden layers: 2 blocks attend across the
layer axis, a linear collapses it, 2 more blocks refine across tokens, then an
MLP lifts 2560 to 6144. It depends only on the prompt, so the pipeline runs
:meth:`Krea2DiT.encode_text` once per generation.

The reference pads the joint sequence to a multiple of 256 and masks the pad;
here no padding exists, which is exact (padded keys are always masked).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._attention import attention_blhd


@dataclass
class Krea2Config:
    features: int = 6144
    layers: int = 28
    heads: int = 48
    kvheads: int = 12
    mlp_dim: int = 16384
    channels: int = 16
    patch: int = 2
    tdim: int = 256
    txtdim: int = 2560
    txtheads: int = 20
    txtkvheads: int = 20
    txt_mlp_dim: int = 6912
    txtlayers: int = 12
    theta: float = 1000.0

    @classmethod
    def from_state_dict(cls, sd) -> "Krea2Config":
        """Derive the config from a checkpoint's (prefix-stripped) keys; works on
        int8 files too, whose quantized weights keep their shapes."""
        feats, in_dim = sd["first.weight"].shape
        layers = 0
        while f"blocks.{layers}.mod.lin" in sd:
            layers += 1
        hd = sd["blocks.0.attn.qknorm.qnorm.scale"].shape[0]
        thd = sd["txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale"].shape[0]
        txtdim = sd["txtmlp.0.scale"].shape[0]
        channels = in_dim // 4
        return cls(
            features=feats, layers=layers, heads=feats // hd,
            kvheads=sd["blocks.0.attn.wk.weight"].shape[0] // hd,
            mlp_dim=sd["blocks.0.mlp.gate.weight"].shape[0], channels=channels, patch=2,
            tdim=sd["tmlp.0.weight"].shape[1], txtdim=txtdim, txtheads=txtdim // thd,
            txtkvheads=sd["txtfusion.layerwise_blocks.0.attn.wk.weight"].shape[0] // thd,
            txt_mlp_dim=sd["txtfusion.layerwise_blocks.0.mlp.gate.weight"].shape[0],
            txtlayers=sd["txtfusion.projector.weight"].shape[1],
        )


class _RMSNorm(nn.Module):
    """Zero-initialized ``scale`` applied as ``scale + 1``, in fp32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.scale.float() + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=self.eps).to(x.dtype)


class _QKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.qnorm = _RMSNorm(head_dim)
        self.knorm = _RMSNorm(head_dim)


class _Param(nn.Module):
    """Holds one learned tensor named ``lin`` (the checkpoint's modulation keys)."""

    def __init__(self, *shape: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(*shape))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _rope_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs of ``x`` (B, L, H, D) by the per-token
    angles ``cos``/``sin`` (L, D/2), in fp32 like the reference."""
    xe, xo = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((c * xe - s * xo, s * xe + c * xo), dim=-1).flatten(-2).to(x.dtype)


class _Attention(nn.Module):
    """Self-attention with GQA, per-head QK RMSNorm, optional RoPE and a sigmoid
    output gate. ``backend_slot`` exposes ``attn_backend`` for the fa2/int8
    dispatch (main blocks only; the text-fusion attentions are tiny)."""

    def __init__(self, dim: int, heads: int, kvheads: int, backend_slot: bool):
        super().__init__()
        self.heads, self.kvheads = heads, kvheads
        self.head_dim = dim // heads
        self.wq = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, kvheads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, kvheads * self.head_dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.qknorm = _QKNorm(self.head_dim)
        self.wo = nn.Linear(dim, dim, bias=False)
        if backend_slot:
            self.attn_backend = "sdpa"

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, L, _ = x.shape
        q = self.qknorm.qnorm(self.wq(x).view(B, L, self.heads, self.head_dim))
        k = self.qknorm.knorm(self.wk(x).view(B, L, self.kvheads, self.head_dim))
        v = self.wv(x).view(B, L, self.kvheads, self.head_dim)
        if rope is not None:
            q, k = _rope_pairs(q, *rope), _rope_pairs(k, *rope)
        if self.kvheads != self.heads:
            # SDPA's enable_gqa semantics: query head h reads KV head h // group.
            group = self.heads // self.kvheads
            k = k.repeat_interleave(group, dim=2)
            v = v.repeat_interleave(group, dim=2)
        out = attention_blhd(q, k, v, getattr(self, "attn_backend", "sdpa"))
        return self.wo(out * torch.sigmoid(self.gate(x)))


class _TextFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int, mlp_dim: int):
        super().__init__()
        self.prenorm = _RMSNorm(dim)
        self.postnorm = _RMSNorm(dim)
        self.attn = _Attention(dim, heads, kvheads, backend_slot=False)
        self.mlp = _SwiGLU(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class _TextFusion(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        mk = lambda: _TextFusionBlock(cfg.txtdim, cfg.txtheads, cfg.txtkvheads, cfg.txt_mlp_dim)  # noqa: E731
        self.layerwise_blocks = nn.ModuleList([mk() for _ in range(2)])
        self.projector = nn.Linear(cfg.txtlayers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([mk() for _ in range(2)])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, L, n, d = feats.shape
        x = feats.reshape(B * L, n, d)
        for blk in self.layerwise_blocks:
            x = blk(x)
        x = self.projector(x.view(B, L, n, d).transpose(-1, -2)).squeeze(-1)
        for blk in self.refiner_blocks:
            x = blk(x)
        return x


class _Block(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.mod = _Param(6 * cfg.features)
        self.prenorm = _RMSNorm(cfg.features)
        self.postnorm = _RMSNorm(cfg.features)
        self.attn = _Attention(cfg.features, cfg.heads, cfg.kvheads, backend_slot=True)
        self.mlp = _SwiGLU(cfg.features, cfg.mlp_dim)

    def forward(self, x, tvec, rope, residual_scale: float = 1.0):
        prescale, preshift, pregate, postscale, postshift, postgate = (tvec + self.mod.lin).chunk(6, dim=-1)
        if residual_scale != 1.0:
            pregate, postgate = pregate * residual_scale, postgate * residual_scale
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, rope)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class _LastLayer(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.norm = _RMSNorm(cfg.features)
        self.linear = nn.Linear(cfg.features, cfg.patch * cfg.patch * cfg.channels, bias=True)
        self.modulation = _Param(2, cfg.features)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = (temb + self.modulation.lin[None]).chunk(2, dim=1)
        return self.linear((1 + scale) * self.norm(x) + shift)


def _rope_axis(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim))
    return pos.double()[:, None] * omega[None]


class Krea2DiT(nn.Module):
    """``forward(x (B, C, H, W), t (B,), ctx (B, L, F)) -> v`` with ``ctx`` from
    :meth:`encode_text`; ``t`` is σ in [0, 1].

    ``residual_scale`` carries the residual stream at that fraction of its true
    size, for fp16: every reader of the stream is an RMSNorm (scale-invariant),
    so scaling the two stream inputs and the per-block gates is exact up to the
    norm's ε. The bf16 reference peaks at |x| ≈ 2.3e4, a third of fp16's max."""

    def __init__(self, cfg: Krea2Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or Krea2Config()
        hd = cfg.features // cfg.heads
        self.axes = (hd - 12 * (hd // 16), 6 * (hd // 16), 6 * (hd // 16))
        self.residual_scale = 1.0
        self.first = nn.Linear(cfg.channels * cfg.patch ** 2, cfg.features, bias=True)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.layers)])
        self.tmlp = nn.Sequential(nn.Linear(cfg.tdim, cfg.features), nn.GELU(approximate="tanh"),
                                  nn.Linear(cfg.features, cfg.features))
        self.txtfusion = _TextFusion(cfg)
        self.txtmlp = nn.Sequential(_RMSNorm(cfg.txtdim), nn.Linear(cfg.txtdim, cfg.features),
                                    nn.GELU(approximate="tanh"), nn.Linear(cfg.features, cfg.features))
        self.last = _LastLayer(cfg)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(cfg.features, 6 * cfg.features))
        self._rope_cache: dict = {}

    def encode_text(self, feats: torch.Tensor) -> torch.Tensor:
        """Selected Qwen3-VL hidden states ``(B, L, n_layers, txtdim)`` -> the
        stream's text tokens ``(B, L, features)``."""
        dtype = self.first.weight.dtype if self.first.weight.is_floating_point() else feats.dtype
        return self.txtmlp(self.txtfusion(feats.to(dtype)))

    def time_embedding(self, t: torch.Tensor, dtype: torch.dtype):
        """``(temb, tvec)``: the ``tmlp`` output (B, 1, F) used by the last layer
        and its ``tproj`` (B, 1, 6F) shared by the blocks."""
        half = self.cfg.tdim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (t.float() * 1000.0)[:, None, None] * freqs
        temb = self.tmlp(torch.cat((torch.cos(args), torch.sin(args)), dim=-1).to(dtype))
        return temb, self.tproj(temb)

    def rope(self, txt_len: int, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token ``(cos, sin)`` (txt_len + h·w, head_dim/2): text tokens at
        (0, 0, 0), image token (i, j) at (0, i, j)."""
        key = (txt_len, h, w, str(device))
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        n = txt_len + h * w
        pos = torch.zeros(n, 3, dtype=torch.float64, device=device)
        pos[txt_len:, 1] = torch.arange(h, dtype=torch.float64, device=device).repeat_interleave(w)
        pos[txt_len:, 2] = torch.arange(w, dtype=torch.float64, device=device).repeat(h)
        ang = torch.cat([_rope_axis(pos[:, i], d, self.cfg.theta) for i, d in enumerate(self.axes)], dim=-1)
        hit = (torch.cos(ang).float(), torch.sin(ang).float())
        if len(self._rope_cache) > 8:
            self._rope_cache.clear()
        self._rope_cache[key] = hit
        return hit

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch
        B, C, H, W = x.shape
        h, w = H // p, W // p
        dtype = ctx.dtype
        img = x.to(dtype).view(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)
        temb, tvec = self.time_embedding(t, dtype)
        rs = self.residual_scale
        img = self.first(img)
        stream = torch.cat((ctx, img), dim=1)
        if rs != 1.0:
            stream = stream * rs
        rope = self.rope(ctx.shape[1], h, w, x.device)
        for blk in self.blocks:
            stream = blk(stream, tvec, rope, rs)
        out = self.last(stream[:, ctx.shape[1]:], temb)
        return out.view(B, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def int8_layer_names(num_blocks: int) -> list[str]:
    """What the ``int8`` load option quantizes: every block's attention
    projections and MLP linears, as in the ComfyUI Krea 2 INT8 files."""
    names = []
    for i in range(num_blocks):
        names += [f"blocks.{i}.attn.{p}" for p in ("wq", "wk", "wv", "gate", "wo")]
        names += [f"blocks.{i}.mlp.{p}" for p in ("gate", "up", "down")]
    return names
