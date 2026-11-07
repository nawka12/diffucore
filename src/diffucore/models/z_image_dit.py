"""Z-Image DiT (Turbo and Base): single-stream rectified-flow transformer.

Ports ``ZImageTransformer2DModel`` from diffusers (Apache-2.0, Alibaba Z-Image
Team and Hugging Face). Module names follow the original / ComfyUI layout
(fused ``attention.qkv``, ``x_embedder``, ``final_layer``), so the ComfyUI bf16
and INT8 files load strictly; :func:`from_diffusers` renames the diffusers
folder and fuses its q/k/v.

The image latent is cut into 2x2 patches (64 values per token) and, together
with the caption features, run through 30 blocks of 3840 width (30 heads of
128, per-head q/k RMSNorm, SwiGLU 10240). Two modulated ``noise_refiner`` blocks
act on the image tokens and two unmodulated ``context_refiner`` blocks on the
caption tokens before the joint stream ``[image, caption]``. Each block applies
``x += tanh(gate) * RMSNorm(branch(RMSNorm(x) * (1 + scale)))`` with the four
modulation vectors coming from the timestep alone. Image and caption sequences
are padded to a multiple of 32 with learned pad tokens that take part in
attention. RoPE has three axes (32, 48, 48) at theta 256: caption token i sits
at (1 + i, 0, 0), image token (r, c) at (L + 1, r, c) with L the padded caption
length, pad tokens of the image at the origin.

In fp16 the branch internals overflow (bf16-trained weights; Base at 768^2
peaks at 1.3e6 in the qkv projection, 3e6 in the attention output, 7e6 in the
feed-forward hidden and 6e7 in its output, fp16 tops out at 65504, while the
stream itself stays near 1.4e4). Both branches end in an RMSNorm that discards
scale, so :class:`ZImageDiT` carries them at fractions ``attn_scale`` and
``ffn_scale``: the attention input is scaled (q and k meet an RMSNorm, v and the
output just carry the factor) and the feed-forward's ``w3`` output is scaled,
which scales its hidden state and output. Exact up to the norms' epsilon.

The caption path (:meth:`ZImageDiT.encode_text`) depends only on the prompt, so
the pipeline runs it once per generation. The model's timestep is ``1 - sigma``
and its output is minus the flow velocity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._attention import attention_blhd

_SEQ_MULTI_OF = 32
_ADALN_DIM = 256
_QK_EPS = 1e-5
_FINAL_EPS = 1e-6


@dataclass
class ZImageConfig:
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    ffn_dim: int = 10240
    cap_feat_dim: int = 2560
    in_channels: int = 16
    patch: int = 2
    norm_eps: float = 1e-5
    rope_theta: float = 256.0
    axes_dims: tuple = (32, 48, 48)
    t_scale: float = 1000.0
    t_mid: int = 1024

    @classmethod
    def from_state_dict(cls, sd) -> "ZImageConfig":
        """Derive the config from a checkpoint's keys (fused ``attention.qkv``
        layout); works on int8 files too, whose quantized weights keep their
        shapes. ``norm_eps``, the RoPE axes and theta are family constants."""
        dim, patch_dim = sd["x_embedder.weight"].shape
        head_dim = sd["layers.0.attention.q_norm.weight"].shape[0]
        patch = 2
        cfg = cls(
            dim=dim, n_layers=_count(sd, "layers.{}.attention.qkv.weight"),
            n_refiner_layers=_count(sd, "noise_refiner.{}.attention.qkv.weight"),
            n_heads=dim // head_dim, ffn_dim=sd["layers.0.feed_forward.w1.weight"].shape[0],
            cap_feat_dim=sd["cap_embedder.0.weight"].shape[0], in_channels=patch_dim // (patch * patch),
            patch=patch, t_mid=sd["t_embedder.mlp.0.weight"].shape[0],
        )
        if sum(cfg.axes_dims) != head_dim:
            raise ValueError(f"head_dim {head_dim} does not match the RoPE axes {cfg.axes_dims}")
        return cfg


def _count(sd, template: str) -> int:
    n = 0
    while template.format(n) in sd:
        n += 1
    return n


class _RMSNorm(nn.Module):
    """fp32 normalization cast back to the input dtype, then the ``weight``
    scale. ``eps`` is per call so a scaled branch output can use ``eps * c^2``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + (self.eps if eps is None else eps))
        return xf.to(x.dtype) * self.weight


def _rope_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs of ``x`` (B, L, H, D) by the per-token
    angles ``cos``/``sin`` (L, D/2), in fp32 like the reference."""
    xe, xo = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((c * xe - s * xo, s * xe + c * xo), dim=-1).flatten(-2).to(x.dtype)


class _Attention(nn.Module):
    """Fused-qkv self-attention with per-head q/k RMSNorm and RoPE, no mask:
    padding tokens are real tokens. ``attn_backend`` is the fa2/int8 slot."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = _RMSNorm(head_dim, _QK_EPS)
        self.k_norm = _RMSNorm(head_dim, _QK_EPS)
        self.attn_backend = "sdpa"

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], scale: float = 1.0) -> torch.Tensor:
        """``x`` arrives already multiplied by ``scale``, so v and the output
        carry it; the q/k norms take ``eps * scale^2``."""
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, -1).unbind(2)
        eps = _QK_EPS * scale * scale
        q = _rope_pairs(self.q_norm(q, eps), *rope)
        k = _rope_pairs(self.k_norm(k, eps), *rope)
        return self.out(attention_blhd(q, k, v, self.attn_backend))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        up = self.w3(x)
        return self.w2(F.silu(self.w1(x)) * (up * scale if scale != 1.0 else up))


class _Block(nn.Module):
    """Transformer block; ``modulation`` blocks (noise refiner and main layers)
    own the ``adaLN_modulation`` that turns the timestep into
    ``(scale_msa, gate_msa, scale_mlp, gate_mlp)``."""

    def __init__(self, cfg: ZImageConfig, modulation: bool):
        super().__init__()
        self.attention = _Attention(cfg.dim, cfg.n_heads)
        self.feed_forward = _FeedForward(cfg.dim, cfg.ffn_dim)
        self.attention_norm1 = _RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm1 = _RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention_norm2 = _RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm2 = _RMSNorm(cfg.dim, cfg.norm_eps)
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(_ADALN_DIM, 4 * cfg.dim, bias=True))
        self.modulation = modulation

    def forward(self, x, rope, adaln=None, scales: tuple[float, float] = (1.0, 1.0)):
        """``scales`` are the attention and feed-forward branch scales (see
        :class:`ZImageDiT`): the branches come out at that fraction of their size
        and the norm after each uses ``eps * scale^2``."""
        a, f = scales
        eps_a, eps_f = self.attention_norm2.eps * a * a, self.ffn_norm2.eps * f * f
        if self.modulation:
            mod = self.adaLN_modulation(adaln).unsqueeze(1)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            attn = self.attention(self.attention_norm1(x) * ((1.0 + scale_msa) * a), rope, a)
            x = x + gate_msa * self.attention_norm2(attn, eps_a)
            ffn = self.feed_forward(self.ffn_norm1(x) * (1.0 + scale_mlp), f)
            return x + gate_mlp * self.ffn_norm2(ffn, eps_f)
        attn = self.attention(self.attention_norm1(x) * a if a != 1.0 else self.attention_norm1(x), rope, a)
        x = x + self.attention_norm2(attn, eps_a)
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x), f), eps_f)


class _FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(_ADALN_DIM, dim, bias=True))

    def forward(self, x: torch.Tensor, adaln: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(adaln).unsqueeze(1)
        return self.linear(F.layer_norm(x.float(), x.shape[-1:], eps=_FINAL_EPS).to(x.dtype) * scale)


class _TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int, freq_size: int = _ADALN_DIM):
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(nn.Linear(freq_size, mid_size, bias=True), nn.SiLU(),
                                 nn.Linear(mid_size, out_size, bias=True))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


def _axis_angles(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """fp32 angles ``pos * theta^(-2i/dim)`` (L, dim/2); the product is formed in
    fp64 and rounded to fp32 once, like the reference's precomputed table."""
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim))
    return (pos.double()[:, None] * omega[None]).float()


class ZImageDiT(nn.Module):
    """``forward(x (B, C, H, W), t (B,), ctx) -> out`` with ``ctx`` from
    :meth:`encode_text`; ``t`` is the model timestep ``1 - sigma`` in [0, 1] and
    ``out`` is minus the velocity (the pipeline negates it).

    ``attn_scale`` and ``ffn_scale`` are the fractions at which the attention and
    feed-forward branches are carried (1.0 = the bf16 behaviour; fp16 loads use
    1/256 and 1/4096)."""

    def __init__(self, cfg: ZImageConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ZImageConfig()
        if cfg.dim % cfg.n_heads or sum(cfg.axes_dims) != cfg.dim // cfg.n_heads:
            raise ValueError("Z-Image needs head_dim equal to the sum of the RoPE axes")
        self.attn_scale = 1.0
        self.ffn_scale = 1.0
        patch_dim = cfg.patch * cfg.patch * cfg.in_channels
        self.x_embedder = nn.Linear(patch_dim, cfg.dim, bias=True)
        self.final_layer = _FinalLayer(cfg.dim, patch_dim)
        self.noise_refiner = nn.ModuleList([_Block(cfg, True) for _ in range(cfg.n_refiner_layers)])
        self.context_refiner = nn.ModuleList([_Block(cfg, False) for _ in range(cfg.n_refiner_layers)])
        self.t_embedder = _TimestepEmbedder(min(cfg.dim, _ADALN_DIM), cfg.t_mid)
        self.cap_embedder = nn.Sequential(_RMSNorm(cfg.cap_feat_dim, cfg.norm_eps),
                                          nn.Linear(cfg.cap_feat_dim, cfg.dim, bias=True))
        self.x_pad_token = nn.Parameter(torch.zeros(1, cfg.dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, cfg.dim))
        self.layers = nn.ModuleList([_Block(cfg, True) for _ in range(cfg.n_layers)])
        self._rope_cache: dict = {}

    def _rope(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` (L, head_dim/2) for integer positions ``pos`` (L, 3)."""
        ang = torch.cat([_axis_angles(pos[:, i], d, self.cfg.rope_theta)
                         for i, d in enumerate(self.cfg.axes_dims)], dim=-1)
        return torch.cos(ang), torch.sin(ang)

    @staticmethod
    def _cap_positions(cap_len: int, device) -> torch.Tensor:
        pos = torch.zeros(cap_len, 3, dtype=torch.int64, device=device)
        pos[:, 0] = torch.arange(1, cap_len + 1, device=device)
        return pos

    def rope(self, cap_len: int, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` for the joint sequence ``[image, caption]`` with the
        caption padded to ``cap_len``; ``h = w = 0`` gives the caption alone."""
        key = (cap_len, h, w, str(device))
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        pos = self._cap_positions(cap_len, device)
        if h:
            n = h * w
            img = torch.zeros(n + (-n) % _SEQ_MULTI_OF, 3, dtype=torch.int64, device=device)
            img[:n, 0] = cap_len + 1
            img[:n, 1] = torch.arange(h, device=device).repeat_interleave(w)
            img[:n, 2] = torch.arange(w, device=device).repeat(h)
            pos = torch.cat((img, pos), 0)
        hit = self._rope(pos)
        if len(self._rope_cache) > 8:
            self._rope_cache.clear()
        self._rope_cache[key] = hit
        return hit

    def _pad(self, tokens: torch.Tensor, pad_token: torch.Tensor) -> torch.Tensor:
        pad = (-tokens.shape[1]) % _SEQ_MULTI_OF
        if not pad:
            return tokens
        # block streaming moves the child modules, not these two bare parameters
        fill = pad_token.to(tokens.device, tokens.dtype)
        return torch.cat((tokens, fill.expand(tokens.shape[0], pad, -1)), 1)

    def encode_text(self, feats: torch.Tensor) -> torch.Tensor:
        """Text-encoder features ``(B, L, cap_feat_dim)`` -> the caption tokens
        ``(B, L padded to 32, dim)`` after the context refiner. Prompt-only, run
        once per prompt."""
        lin = self.cap_embedder[1]
        dtype = lin.weight.dtype if lin.weight.is_floating_point() else feats.dtype
        cap = self.cap_embedder(feats.to(dtype))
        cap = self._pad(cap, self.cap_pad_token)
        rope = self.rope(cap.shape[1], 0, 0, feats.device)
        rope = rope[0].to(cap.device), rope[1].to(cap.device)
        for blk in self.context_refiner:
            cap = blk(cap, rope, None, (self.attn_scale, self.ffn_scale))
        return cap

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch
        B, C, H, W = x.shape
        h, w = H // p, W // p
        n = h * w
        dtype = ctx.dtype
        scales = (self.attn_scale, self.ffn_scale)
        adaln = self.t_embedder(t.float().reshape(-1) * self.cfg.t_scale).to(dtype)
        patches = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, n, p * p * C)
        img = self.x_embedder(patches.to(self.x_embedder.weight.dtype)).to(dtype)
        img = self._pad(img, self.x_pad_token)
        cos, sin = self.rope(ctx.shape[1], h, w, x.device)
        cos, sin = cos.to(x.device), sin.to(x.device)
        n_pad = img.shape[1]
        for blk in self.noise_refiner:
            img = blk(img, (cos[:n_pad], sin[:n_pad]), adaln, scales)
        stream = torch.cat((img, ctx), 1)
        for blk in self.layers:
            stream = blk(stream, (cos, sin), adaln, scales)
        y = self.final_layer(stream[:, :n], adaln)
        return y.view(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)


def int8_layer_names(cfg: ZImageConfig) -> list[str]:
    """What the ``int8`` load option quantizes: the attention and feed-forward
    linears of every main and refiner block (the ComfyUI INT8 files also
    quantize the per-block modulation, which stays fp here)."""
    names = []
    lists = (("layers", cfg.n_layers), ("noise_refiner", cfg.n_refiner_layers),
             ("context_refiner", cfg.n_refiner_layers))
    for attr, count in lists:
        for i in range(count):
            names += [f"{attr}.{i}.attention.qkv", f"{attr}.{i}.attention.out"]
            names += [f"{attr}.{i}.feed_forward.{w}" for w in ("w1", "w2", "w3")]
    return names


# diffusers (ZImageTransformer2DModel) names -> the original ones, in order.
_DIFFUSERS_RULES = [
    (r"^all_x_embedder\.2-1\b", "x_embedder"),
    (r"^all_final_layer\.2-1\b", "final_layer"),
    (r"\.attention\.to_out\.0\b", ".attention.out"),
    (r"\.attention\.norm_q\b", ".attention.q_norm"),
    (r"\.attention\.norm_k\b", ".attention.k_norm"),
]
_DIFFUSERS_RES = [(re.compile(p), r) for p, r in _DIFFUSERS_RULES]


def from_diffusers_name(name: str) -> str:
    """A diffusers Z-Image tensor or module name in the original naming
    (``attention.to_q/k/v`` are the row thirds of ``attention.qkv`` and stay)."""
    for pattern, repl in _DIFFUSERS_RES:
        name = pattern.sub(repl, name)
    return name


def from_diffusers(state_dict: dict) -> dict:
    """A diffusers-format Z-Image transformer state dict in the original, fused
    layout: each block's ``attention.to_q/k/v`` concatenated (rows q, k, v) into
    ``attention.qkv``. The q/k/v tensors are popped from ``state_dict`` as they
    are fused."""
    out = {}
    for k in list(state_dict):
        if k.endswith(".attention.to_q.weight"):
            base = k[: -len("to_q.weight")]
            out[from_diffusers_name(base) + "qkv.weight"] = torch.cat(
                [state_dict.pop(base + n + ".weight") for n in ("to_q", "to_k", "to_v")], 0)
        elif not k.endswith((".attention.to_k.weight", ".attention.to_v.weight")):
            out[from_diffusers_name(k)] = state_dict[k]
    return out
