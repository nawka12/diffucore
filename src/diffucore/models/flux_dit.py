"""FLUX DiT — Black Forest Labs' rectified-flow transformer (FLUX.1 / FLUX.2).

A double-stream + single-stream MMDiT (Esser et al. SD3, 2024; FLUX, BFL 2024).
Image and text tokens first flow through ``depth`` *double-stream* blocks that
keep separate weights per modality but attend jointly, then are concatenated and
flow through ``depth_single_blocks`` *single-stream* blocks (one fused
qkv+MLP linear). Positions use an axial RoPE; every block is AdaLN-modulated; QK
is RMSNorm'd per head.

Submodule and parameter names mirror Black Forest Labs' ``flux`` repository
(Apache-2.0; the same lineage ComfyUI follows) so a ``strict=True`` load against
an official transformer file is the correctness check.

The same module covers two configs:

* **FLUX.1** (``image_model="flux"``): per-block ``img_mod``/``txt_mod``
  modulation, GELU-tanh MLP, biases on, RoPE axes ``(16,56,56)`` θ=10000,
  ``qkv_bias=True``. 2×2 patchify happens in the pipeline (in_channels=64).
* **FLUX.2** (``image_model="flux2"``): a single *global* set of three shared
  modulators (``double_stream_modulation_img``/``_txt``,
  ``single_stream_modulation``, bias-free) drives every block; SiLU-gated MLP;
  **no biases** anywhere; RoPE axes ``(32,32,32,32)`` θ=2000. patch_size=1, so
  the pipeline feeds the latent channels directly (in_channels=128) and the text
  ids carry positions on axis 3.

Config is driven from the checkpoint shapes (:meth:`FluxConfig.from_state_dict`);
the family constants the shapes don't reveal are passed by the loader.

Contract:
    forward(img:[B,L_img,in_ch], img_ids:[B,L_img,A], txt:[B,L_txt,ctx],
            txt_ids:[B,L_txt,A], timesteps:[B], y:[B,vec]|None, guidance:[B]|None)
        -> [B, L_img, in_ch]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class FluxConfig:
    in_channels: int = 64
    context_in_dim: int = 4096
    vec_in_dim: int | None = 768          # pooled-vector dim; None => no vector_in
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: Sequence[int] = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    guidance_embed: bool = True
    # FLUX.2 differences (defaults keep FLUX.1 behaviour):
    global_modulation: bool = False       # shared 3-modulator AdaLN instead of per-block
    mlp_silu_act: bool = False            # SiLU-gated MLP instead of GELU-tanh
    ops_bias: bool = True                 # biases on the linear layers

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_state_dict(
        cls,
        sd: Mapping[str, torch.Tensor],
        prefix: str = "",
        *,
        num_heads: int = 24,
        axes_dim: Sequence[int] = (16, 56, 56),
        theta: int = 10_000,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        global_modulation: bool = False,
        mlp_silu_act: bool = False,
        ops_bias: bool = True,
    ) -> "FluxConfig":
        """Derive the config from a transformer state dict. Widths/depths come
        from tensor shapes; the family flags come from the caller (detection)."""
        def shape(name):
            return sd[prefix + name].shape

        has_vec = (prefix + "vector_in.in_layer.weight") in sd
        return cls(
            in_channels=shape("img_in.weight")[1],
            context_in_dim=shape("txt_in.weight")[1],
            vec_in_dim=shape("vector_in.in_layer.weight")[1] if has_vec else None,
            hidden_size=shape("img_in.weight")[0],
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            depth=_count_blocks(sd, prefix, "double_blocks", "img_attn.qkv.weight"),
            depth_single_blocks=_count_blocks(sd, prefix, "single_blocks", "linear1.weight"),
            axes_dim=tuple(axes_dim),
            theta=theta,
            qkv_bias=qkv_bias,
            guidance_embed=(prefix + "guidance_in.in_layer.weight") in sd,
            global_modulation=global_modulation,
            mlp_silu_act=mlp_silu_act,
            ops_bias=ops_bias,
        )


def _count_blocks(sd: Mapping[str, torch.Tensor], prefix: str, group: str, leaf: str) -> int:
    i = 0
    while f"{prefix}{group}.{i}.{leaf}" in sd:
        i += 1
    return i


# --------------------------------------------------------------------------- #
# embeddings + RoPE
# --------------------------------------------------------------------------- #

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000, time_factor: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of a (rectified-flow time) scalar, scaled by 1000."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        emb = emb.to(t.dtype)
    return emb


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    """FLUX QK-norm: ``(x·rsqrt(mean(x²)+ε))·scale``. Parameter named ``scale`` to
    match the on-disk ``...norm.query_norm.scale`` keys."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Per-axis RoPE rotation tensor for positions ``pos`` -> [..., L, dim/2, 2, 2]."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: Sequence[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3
        )
        return emb.unsqueeze(1)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    q, k = apply_rope(q, k, pe)
    x = F.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "B H L D -> B L (H D)")


# --------------------------------------------------------------------------- #
# modulation + MLP
# --------------------------------------------------------------------------- #

@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, bias: bool = True):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=bias)

    def forward(self, vec: torch.Tensor):
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(F.silu(vec)).chunk(self.multiplier, dim=-1)
        first = ModulationOut(*out[:3])
        second = ModulationOut(*out[3:]) if self.is_double else None
        return first, second


class SiLUActivation(nn.Module):
    """Gated SiLU: split the last dim in half, ``silu(a)·b`` (FLUX.2 MLP)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


def build_mlp(hidden_size: int, mlp_hidden: int, mlp_silu_act: bool) -> nn.Module:
    if mlp_silu_act:
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden, hidden_size, bias=False),
        )
    return nn.Sequential(
        nn.Linear(hidden_size, mlp_hidden, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(mlp_hidden, hidden_size, bias=True),
    )


class SelfAttention(nn.Module):
    """qkv projection + per-head QK-norm + output projection."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_bias: bool):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias, *,
                 modulation=True, mlp_silu_act=False, proj_bias=True):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.modulation = modulation
        if modulation:
            self.img_mod = Modulation(hidden_size, double=True)
            self.txt_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias, proj_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = build_mlp(hidden_size, mlp_hidden, mlp_silu_act)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias, proj_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = build_mlp(hidden_size, mlp_hidden, mlp_silu_act)

    def forward(self, img, txt, vec, pe):
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, *,
                 modulation=True, mlp_silu_act=False, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # SiLU-gated MLP needs twice the first-linear width (it halves on gating).
        self.mlp_first = self.mlp_hidden_dim * 2 if mlp_silu_act else self.mlp_hidden_dim
        self.mlp_act = SiLUActivation() if mlp_silu_act else nn.GELU(approximate="tanh")
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_first, bias=bias)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=bias)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if modulation:
            self.modulation = Modulation(hidden_size, double=False)
        else:
            self.modulation = None

    def forward(self, x, vec, pe):
        mod = self.modulation(vec)[0] if self.modulation is not None else vec
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_first], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        out = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        return x + mod.gate * out


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, vec):
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift
        return self.linear(x)


# --------------------------------------------------------------------------- #
# top-level model
# --------------------------------------------------------------------------- #

class Flux(nn.Module):
    """FLUX rectified-flow transformer (image-token in, velocity out)."""

    def __init__(self, config: FluxConfig | None = None):
        super().__init__()
        cfg = config or FluxConfig()
        self.config = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.guidance_embed = cfg.guidance_embed
        self.global_modulation = cfg.global_modulation
        bias = cfg.ops_bias

        pe_dim = cfg.head_dim
        if sum(cfg.axes_dim) != pe_dim:
            raise ValueError(f"axes_dim {tuple(cfg.axes_dim)} must sum to head_dim {pe_dim}")

        self.pe_embedder = EmbedND(dim=pe_dim, theta=cfg.theta, axes_dim=cfg.axes_dim)
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size, bias=bias)
        self.time_in = MLPEmbedder(256, cfg.hidden_size, bias=bias)
        self.vector_in = MLPEmbedder(cfg.vec_in_dim, cfg.hidden_size) if cfg.vec_in_dim else None
        self.guidance_in = MLPEmbedder(256, cfg.hidden_size, bias=bias) if cfg.guidance_embed else None
        self.txt_in = nn.Linear(cfg.context_in_dim, cfg.hidden_size, bias=bias)

        self.double_blocks = nn.ModuleList(
            DoubleStreamBlock(
                cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias,
                modulation=not cfg.global_modulation, mlp_silu_act=cfg.mlp_silu_act, proj_bias=bias,
            )
            for _ in range(cfg.depth)
        )
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(
                cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio,
                modulation=not cfg.global_modulation, mlp_silu_act=cfg.mlp_silu_act, bias=bias,
            )
            for _ in range(cfg.depth_single_blocks)
        )
        self.final_layer = LastLayer(cfg.hidden_size, 1, self.out_channels, bias=bias)

        if cfg.global_modulation:
            # Three shared, bias-free modulators drive every block (FLUX.2).
            self.double_stream_modulation_img = Modulation(cfg.hidden_size, double=True, bias=False)
            self.double_stream_modulation_txt = Modulation(cfg.hidden_size, double=True, bias=False)
            self.single_stream_modulation = Modulation(cfg.hidden_size, double=False, bias=False)

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_in is not None:
            if guidance is None:
                raise ValueError("guidance-distilled FLUX requires a `guidance` tensor")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        if self.vector_in is not None:
            vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if self.global_modulation:
            double_vec = (self.double_stream_modulation_img(vec),
                          self.double_stream_modulation_txt(vec))
            single_vec, _ = self.single_stream_modulation(vec)
        else:
            double_vec = vec
            single_vec = vec

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=double_vec, pe=pe)

        x = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            x = block(x, vec=single_vec, pe=pe)
        img = x[:, txt.shape[1] :, ...]

        return self.final_layer(img, vec)
