"""Anima DiT: Cosmos-Predict2-family adaLN transformer (image-only path).

A 28-block transformer adapted from NVIDIA's Cosmos-Predict2-2B (Apache-2.0).
Each block has three adaLN-LoRA modulators (self-attention, cross-attention,
MLP); self-attention uses a 3D RoPE. Anima: 16+1 input channels (latent +
padding mask), patch 2, 2048 channels, 16 heads of 128, 1024-d cross-attention
context from the LLM-Adapter, adaLN-LoRA dim 256, RoPE split 42/42/44.

Latents are treated as T=1 video (``(B, C, H, W)`` reshaped on entry/exit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ._attention import attention_blhd
from ._norm import RMSNorm
from .llm_adapter import LLMAdapter, LLMAdapterConfig


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class CosmosDiTConfig:
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    concat_padding_mask: bool = True
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    head_dim: int = 128
    mlp_ratio: float = 4.0
    crossattn_emb_channels: int = 1024
    adaln_lora_dim: int = 256
    # 3D RoPE setup (Anima 16-ch defaults)
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    base_fps: int = 24
    rope_h_extrapolation_ratio: float = 4.0
    rope_w_extrapolation_ratio: float = 4.0
    rope_t_extrapolation_ratio: float = 1.0
    rope_enable_fps_modulation: bool = True
    rms_norm_eps: float = 1e-6


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _pad_to_patch_size(x: torch.Tensor, patch: Tuple[int, int, int]) -> torch.Tensor:
    """Reflect-pad the last 3 dims (T, H, W) to multiples of the patch size.
    Upstream uses circular padding, which leaks the opposite edge into the seam;
    this only fires on sizes not divisible by 2.
    """
    pads = []
    for i in range(x.ndim - 2):
        n = x.shape[i + 2]
        p = patch[i]
        pads = [0, (p - n % p) % p] + pads
    return F.pad(x, pads, mode="reflect") if any(pads) else x


# --------------------------------------------------------------------------- #
# time embedding
# --------------------------------------------------------------------------- #

class _Timesteps(nn.Module):
    """Sinusoidal timestep embedding, no parameters."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t_B_T: torch.Tensor) -> torch.Tensor:
        assert t_B_T.ndim == 2, f"expected (B, T), got shape {tuple(t_B_T.shape)}"
        t = t_B_T.flatten().float()
        half = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        emb = t[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.view(t_B_T.shape[0], t_B_T.shape[1], self.dim)


class _TimestepEmbedding(nn.Module):
    """``Linear(D→D) · SiLU · Linear(D→3·D)``. The second output is the per-block
    adaLN-LoRA delta added to each ``adaln_modulation_*`` output."""
    def __init__(self, dim: int):
        super().__init__()
        # adaln_lora mode: linear_1 has no bias, linear_2 produces 3·dim
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return sample, emb   # (emb_B_T_D, adaln_lora_B_T_3D)


# --------------------------------------------------------------------------- #
# patch embed / unpatchify
# --------------------------------------------------------------------------- #

class _PatchEmbed(nn.Module):
    """Rearrange ``(B, C, T, H, W) → (B, T/r, H/m, W/n, C·r·m·n)``, then Linear to
    ``model_channels`` (stored as ``proj.1`` to match the checkpoint).
    """
    def __init__(self, patch_t: int, patch_s: int, in_channels: int, model_channels: int):
        super().__init__()
        self.patch_t = patch_t
        self.patch_s = patch_s
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=patch_t, m=patch_s, n=patch_s,
            ),
            nn.Linear(in_channels * patch_s * patch_s * patch_t, model_channels, bias=False),
        )

    def forward(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        return self.proj(x_B_C_T_H_W)


# --------------------------------------------------------------------------- #
# 3D RoPE (Apache-2.0 algorithm from NVIDIA Cosmos)
# --------------------------------------------------------------------------- #

class _VideoRoPE3D(nn.Module):
    """Three-axis (T, H, W) RoPE returning a ``(L, head_dim/2, 2, 2)`` rotation
    tensor for :func:`_apply_rope`. head_dim splits ``head_dim//6·2`` per
    spatial axis, the rest temporal (42/42/44 for Anima); NTK factors scale θ
    per axis.
    """
    def __init__(self, cfg: CosmosDiTConfig):
        super().__init__()
        d = cfg.head_dim
        dim_h = d // 6 * 2
        dim_w = dim_h
        dim_t = d - 2 * dim_h
        self.dim_h, self.dim_w, self.dim_t = dim_h, dim_w, dim_t
        self.base_fps = cfg.base_fps
        self.enable_fps_modulation = cfg.rope_enable_fps_modulation
        self.h_ntk = cfg.rope_h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk = cfg.rope_w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk = cfg.rope_t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2).float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2).float() / dim_t,
            persistent=False,
        )

    def forward(self, x_B_T_H_W_D: torch.Tensor, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H, W, _ = x_B_T_H_W_D.shape
        device = x_B_T_H_W_D.device
        fps_key = fps.item() if fps is not None else None
        # Cached on device (keyed by device, since .to() won't move a plain
        # attribute). Bypassed under torch.compile: a tensor made inside a CUDA
        # Graphs run lives in the static pool the next replay overwrites.
        compiling = torch.compiler.is_compiling()
        cache_key = (H, W, T, fps_key, device)
        if not compiling:
            cached = getattr(self, "_rope_cache", None)
            if cached is not None and cached[0] == cache_key:
                return cached[1]
        h_theta = 10_000.0 * self.h_ntk
        w_theta = 10_000.0 * self.w_ntk
        t_theta = 10_000.0 * self.t_ntk
        h_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device))
        w_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device))
        t_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device))

        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        h_e = torch.outer(seq[:H], h_freqs)
        w_e = torch.outer(seq[:W], w_freqs)
        if fps is None or not self.enable_fps_modulation:
            t_e = torch.outer(seq[:T], t_freqs)
        else:
            t_e = torch.outer(seq[:T] / fps * self.base_fps, t_freqs)

        # [cos, -sin, sin, cos] per (pos, freq): a 2x2 rotation matrix.
        def _rot(e):
            return torch.stack([torch.cos(e), -torch.sin(e), torch.sin(e), torch.cos(e)], dim=-1)
        h_r = _rot(h_e)
        w_r = _rot(w_e)
        t_r = _rot(t_e)

        em = torch.cat(
            [
                repeat(t_r, "t d x -> t h w d x", h=H, w=W),
                repeat(h_r, "h d x -> t h w d x", t=T, w=W),
                repeat(w_r, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )
        result = rearrange(em, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()
        if not compiling:
            self._rope_cache = (cache_key, result)
        return result


def _apply_rope_eager(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply the 2×2 rotation encoded in ``freqs`` (shape ``(L, D/2, 2, 2)``)
    to each pair of channels of ``t`` (shape ``(B, ..., L, D)``)."""
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    out = out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return out


_apply_rope_cuda = None  # compiled lazily on first CUDA call; False = compile unusable


def _apply_rope(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """RoPE apply: eager, except on CUDA where the chain is ``torch.compile``'d
    once (0.29 ms vs 1.37 ms at 4096 tokens, ~7% of a step).
    ``emulate_precision_casts`` keeps rounding identical to eager, so images are
    bit-equal; ``dynamic=True`` compiles once for all resolutions. Under an
    outer compile, ``is_compiling`` hands Dynamo the eager body to fuse. Any
    compile failure falls back to eager for the process.
    """
    global _apply_rope_cuda
    if torch.compiler.is_compiling() or not t.is_cuda:
        return _apply_rope_eager(t, freqs)
    if _apply_rope_cuda is None:
        try:
            compiled = torch.compile(_apply_rope_eager, dynamic=True,
                                     options={"emulate_precision_casts": True})
            out = compiled(t, freqs)  # compile here so a broken backend is caught once
            _apply_rope_cuda = compiled
            return out
        except Exception as e:  # noqa: BLE001  any backend failure: no compile here
            print(f"[rope] torch.compile unavailable ({type(e).__name__}); keeping eager apply", flush=True)
            _apply_rope_cuda = False
    if _apply_rope_cuda is False:
        return _apply_rope_eager(t, freqs)
    return _apply_rope_cuda(t, freqs)


# --------------------------------------------------------------------------- #
# attention and MLP
# --------------------------------------------------------------------------- #

class _Attention(nn.Module):
    """Cosmos-style attention with per-head q/k RMSNorm. Self-attention
    (``context_dim is None``) applies 3D RoPE to q and k.
    """
    def __init__(self, query_dim: int, context_dim: Optional[int], n_heads: int, head_dim: int, eps: float):
        super().__init__()
        inner = n_heads * head_dim
        self.is_selfattn = context_dim is None
        ctx = query_dim if context_dim is None else context_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(ctx, inner, bias=False)
        self.v_proj = nn.Linear(ctx, inner, bias=False)
        self.v_norm = nn.Identity()       # carried for state-dict parity
        self.output_proj = nn.Linear(inner, query_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        # Kernel choice; the loader stamps "fa2_turing" when the policy opts in.
        self.attn_backend = "sdpa"

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor], rope_emb: Optional[torch.Tensor]) -> torch.Tensor:
        ctx = x if context is None else context
        q = rearrange(self.q_proj(x), "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
        k = rearrange(self.k_proj(ctx), "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
        v = rearrange(self.v_proj(ctx), "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.is_selfattn and rope_emb is not None:
            q = _apply_rope(q, rope_emb)
            k = _apply_rope(k, rope_emb)
        # SDPA by default; "fa2_turing" swaps in the sm75 FlashAttention-2 port.
        out = attention_blhd(q, k, v, self.attn_backend)
        return self.output_proj(out)


class _GPT2FeedForward(nn.Module):
    """Linear → GELU → Linear, no bias (``layer1`` / ``layer2``)."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


# --------------------------------------------------------------------------- #
# block (adaLN-LoRA × 3 stages)
# --------------------------------------------------------------------------- #

class _AdaLNLoRA(nn.Sequential):
    """``SiLU → Linear(D→r) → Linear(r→3D)``, keyed ``.1`` / ``.2`` like the
    checkpoint's ``adaln_modulation_X``."""
    def __init__(self, dim: int, r: int):
        super().__init__(
            nn.SiLU(),
            nn.Linear(dim, r, bias=False),
            nn.Linear(r, 3 * dim, bias=False),
        )


class _Block(nn.Module):
    """One adaLN-LoRA block. The layer norms have no affine params; the adaLN
    modulators provide all per-token affine shaping."""
    def __init__(self, cfg: CosmosDiTConfig):
        super().__init__()
        d = cfg.model_channels
        r = cfg.adaln_lora_dim
        h = cfg.num_heads
        hd = cfg.head_dim
        self.layer_norm_self_attn = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.rms_norm_eps)
        self.self_attn = _Attention(d, None, h, hd, eps=cfg.rms_norm_eps)
        self.layer_norm_cross_attn = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.rms_norm_eps)
        self.cross_attn = _Attention(d, cfg.crossattn_emb_channels, h, hd, eps=cfg.rms_norm_eps)
        self.layer_norm_mlp = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.rms_norm_eps)
        self.mlp = _GPT2FeedForward(d, int(d * cfg.mlp_ratio))
        # Three independent modulators, named to match the checkpoint.
        self.adaln_modulation_self_attn = _AdaLNLoRA(d, r)
        self.adaln_modulation_cross_attn = _AdaLNLoRA(d, r)
        self.adaln_modulation_mlp = _AdaLNLoRA(d, r)

    def forward(
        self,
        x: torch.Tensor,                # (B, T, H, W, D), residual dtype (fp32 for fp16 inference)
        emb: torch.Tensor,              # (B, T, D), compute dtype
        ctx: torch.Tensor,              # (B, L, ctx_dim)
        rope_emb: torch.Tensor,         # ((T·H·W), head_dim/2, 2, 2)
        adaln_lora: torch.Tensor,       # (B, T, 3·D)
    ) -> torch.Tensor:
        residual_dtype = x.dtype
        compute_dtype = emb.dtype
        sa = self.adaln_modulation_self_attn(emb) + adaln_lora
        ca = self.adaln_modulation_cross_attn(emb) + adaln_lora
        ml = self.adaln_modulation_mlp(emb) + adaln_lora
        sa_s, sa_sc, sa_g = sa.chunk(3, dim=-1)
        ca_s, ca_sc, ca_g = ca.chunk(3, dim=-1)
        ml_s, ml_sc, ml_g = ml.chunk(3, dim=-1)
        # (B, T, D) -> (B, T, 1, 1, D) so it broadcasts over H, W.
        def _expand(v):
            return rearrange(v, "b t d -> b t 1 1 d")
        sa_s, sa_sc, sa_g = map(_expand, (sa_s, sa_sc, sa_g))
        ca_s, ca_sc, ca_g = map(_expand, (ca_s, ca_sc, ca_g))
        ml_s, ml_sc, ml_g = map(_expand, (ml_s, ml_sc, ml_g))

        B, T, H, W, D = x.shape
        # Normalize in the residual dtype, attend in compute dtype, and cast back
        # before the gated residual add so accumulation stays fp32.
        h = self.layer_norm_self_attn(x) * (1 + sa_sc) + sa_s
        h_seq = rearrange(h, "b t h w d -> b (t h w) d").to(compute_dtype)
        h_seq = self.self_attn(h_seq, None, rope_emb)
        h = rearrange(h_seq, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype)
        x = x + sa_g.to(residual_dtype) * h

        h = self.layer_norm_cross_attn(x) * (1 + ca_sc) + ca_s
        h_seq = rearrange(h, "b t h w d -> b (t h w) d").to(compute_dtype)
        h_seq = self.cross_attn(h_seq, ctx, None)
        h = rearrange(h_seq, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype)
        x = x + ca_g.to(residual_dtype) * h

        h = self.layer_norm_mlp(x) * (1 + ml_sc) + ml_s
        h = self.mlp(h.to(compute_dtype)).to(residual_dtype)
        x = x + ml_g.to(residual_dtype) * h
        return x

    def modulated_self_attn_input(
        self, x: torch.Tensor, emb: torch.Tensor, adaln_lora: torch.Tensor
    ) -> torch.Tensor:
        """The timestep-modulated tokens entering self-attention, TeaCache's
        cheap probe. Same modulation as :meth:`forward`, without the
        attention/MLP."""
        sa = self.adaln_modulation_self_attn(emb) + adaln_lora
        sa_s, sa_sc, _ = sa.chunk(3, dim=-1)
        sa_s = rearrange(sa_s, "b t d -> b t 1 1 d")
        sa_sc = rearrange(sa_sc, "b t d -> b t 1 1 d")
        return self.layer_norm_self_attn(x) * (1 + sa_sc) + sa_s


# --------------------------------------------------------------------------- #
# final layer
# --------------------------------------------------------------------------- #

class _FinalLayer(nn.Module):
    """Two-chunk adaLN (shift, scale; no gate), then Linear to
    ``patch_t · patch_s² · out_channels`` for :meth:`unpatchify`."""
    def __init__(self, cfg: CosmosDiTConfig):
        super().__init__()
        d = cfg.model_channels
        r = cfg.adaln_lora_dim
        patch_out = cfg.patch_temporal * cfg.patch_spatial**2 * cfg.out_channels
        self.layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.rms_norm_eps)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, r, bias=False),
            nn.Linear(r, 2 * d, bias=False),
        )
        self.linear = nn.Linear(d, patch_out, bias=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, adaln_lora: torch.Tensor) -> torch.Tensor:
        # The adaLN output gets the first 2·D slice of adaln_lora added.
        D = x.shape[-1]
        delta = (self.adaln_modulation(emb) + adaln_lora[:, :, : 2 * D])
        shift, scale = delta.chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")
        h = self.layer_norm(x) * (1 + scale) + shift
        return self.linear(h)


# --------------------------------------------------------------------------- #
# TeaCache
# --------------------------------------------------------------------------- #

class TeaCache:
    """Timestep-Embedding-Aware Cache for one DiT denoising stream (Liu et al.,
    2024, arXiv:2411.19108).

    Caches the blocks' residual (``stack(x) - x``) and, on a step where the
    input has drifted little, forecasts it instead of running the blocks.

    * **Decision** (``rule``). ``"drift"``: the accumulated rescaled rel-L1
      change of the block-0 modulated input, recomputing once it crosses
      ``rel_l1_thresh``. ``"easy"`` (EasyCache, arXiv:2507.02860): the
      accumulated predicted output change ``k · ‖Δx‖ / ‖v_{t−1}‖`` with the
      measured transformation rate ``k = ‖Δv‖ / ‖Δx‖`` (mean-abs norms);
      ``warmup`` calls always compute, and ``coefficients``/``record`` don't
      apply.
    * **Forecast.** A skip extrapolates the residual from finite differences
      over activation steps, up to ``max_order`` (0 = reuse, the original
      TeaCache). ``basis="taylor"`` (TaylorSeer, arXiv:2503.06923) uses
      ``k^i``; ``"hermite"`` (HiCache, arXiv:2508.16984) uses damped Hermite
      ``σ^i · H_i(σk)``, which at ``sigma = 2**-0.5`` and order 1 equals taylor.
    * **Floor.** Below noise level ``sigma_floor`` every step computes, under
      either rule. Late steps drift least, so an unguarded accumulator skips
      exactly the calls that clear the last injected noise.

    One instance per stream: CFG's cond and uncond passes have near-identical
    modulated inputs, so a shared accumulator would read zero drift.
    ``coefficients`` rescale the raw rel-L1 (``numpy.poly1d`` order; identity by
    default) and come from offline calibration.
    """

    def __init__(self, rel_l1_thresh: float, coefficients: Sequence[float] = (1.0, 0.0),
                 *, record: bool = False, max_order: int = 1,
                 basis: str = "taylor", sigma: float = 0.5,
                 rule: str = "drift", warmup: int = 3, sigma_floor: float = 0.0):
        if basis not in ("taylor", "hermite"):
            raise ValueError(f"basis must be 'taylor' or 'hermite'; got {basis!r}")
        if rule not in ("drift", "easy"):
            raise ValueError(f"rule must be 'drift' or 'easy'; got {rule!r}")
        if record and rule != "drift":
            raise ValueError("record mode calibrates the drift rule only; got rule='easy'")
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.record = record
        self.max_order = int(max_order)
        self.basis = basis
        self.sigma = float(sigma)
        self.sigma_floor = float(sigma_floor)
        self.prev_modulated: Optional[torch.Tensor] = None
        self.accumulated = 0.0
        self.calls = 0   # forwards seen
        self.skips = 0   # forwards whose blocks were reused
        self.rel_history: list[float] = []   # raw per-step rel-L1 (record mode)
        # ``taylor[i]``: i-th finite difference of the residual over activation
        # steps. ``last_activated``: the ``calls`` index of the last computed step.
        self.taylor: dict[int, torch.Tensor] = {}
        self.last_activated = -1
        # EasyCache state (rule == "easy"), kept in fp32 against fp16
        # cancellation. ``pending_dx`` is the denominator of ``k`` for the output
        # ``record_output`` sees next.
        self.rule = rule
        self.warmup = int(warmup)
        self.prev_x: Optional[torch.Tensor] = None
        self.prev_out: Optional[torch.Tensor] = None
        self.k: Optional[float] = None
        self.pending_dx = 0.0
        self.last_computed = True   # decision of the most recent call
        self.follow_misses = 0      # instrumentation, filled in by the pipeline

    def _rescale(self, x: float) -> float:
        out = 0.0
        for c in self.coefficients:  # Horner, highest degree first
            out = out * x + c
        return out

    def _below_floor(self, sigma: Optional[float]) -> bool:
        return sigma is not None and sigma < self.sigma_floor

    def should_compute(self, modulated: torch.Tensor, sigma: Optional[float] = None) -> bool:
        """Decide whether this step runs the blocks, folding ``modulated`` into
        the accumulator; the first call and any call below ``sigma_floor``
        always compute. ``record`` mode computes every step and logs the raw
        rel-L1 for calibration.
        """
        self.calls += 1
        if self.prev_modulated is None:
            self.accumulated = 0.0
            self.prev_modulated = modulated
            self.last_computed = True
            return True
        denom = self.prev_modulated.abs().mean().clamp_min(1e-8)
        rel = ((modulated - self.prev_modulated).abs().mean() / denom).item()
        self.prev_modulated = modulated
        if self.record:
            self.rel_history.append(rel)
            self.last_computed = True
            return True
        if self._below_floor(sigma):
            self.accumulated = 0.0
            self.last_computed = True
            return True
        self.accumulated += self._rescale(rel)
        if self.accumulated < self.rel_l1_thresh:
            self.skips += 1
            self.last_computed = False
            return False
        self.accumulated = 0.0
        self.last_computed = True
        return True

    def should_compute_easy(self, x: torch.Tensor, sigma: Optional[float] = None) -> bool:
        """EasyCache decision on the padded ``(B, C, T, H, W)`` model input. The
        first ``warmup`` calls (``k`` needs two outputs) and any call below
        ``sigma_floor`` always compute.
        """
        self.calls += 1
        x = x.detach().to(torch.float32, copy=True)
        if self.prev_x is None:
            self.prev_x = x
            self.pending_dx = 0.0
            self.accumulated = 0.0
            self.last_computed = True
            return True
        dx = (x - self.prev_x).abs().mean().item()   # mean-abs, as in the reference
        self.prev_x = x
        self.pending_dx = dx
        if (self.calls <= self.warmup or self.k is None or self.prev_out is None
                or self._below_floor(sigma)):
            decision = True                          # warm-up / rate not measurable / floor
        else:
            v_norm = self.prev_out.abs().mean().item()
            self.accumulated += self.k * dx / max(v_norm, 1e-8)
            decision = self.accumulated >= self.rel_l1_thresh
        if decision:
            self.accumulated = 0.0
        else:
            self.skips += 1
        self.last_computed = decision
        return decision

    def record_output(self, out: torch.Tensor) -> None:
        """Feed the step's output (forecast or real) to the EasyCache rule. The
        rate ``k`` refreshes only after a step that ran the blocks.
        """
        out = out.detach().to(torch.float32, copy=True)
        if self.last_computed and self.prev_out is not None and self.pending_dx > 0:
            self.k = (out - self.prev_out).abs().mean().item() / self.pending_dx
        self.prev_out = out

    def update(self, residual: torch.Tensor) -> None:
        """Record a computed residual and refresh the finite-difference factors,
        each divided by the (possibly uneven) gap between the last two
        activations. At ``max_order == 0`` or on the first activation only the
        residual itself is kept.
        """
        dist = self.calls - self.last_activated if self.last_activated >= 0 else 1
        prev, new = self.taylor, {0: residual}
        for i in range(self.max_order):
            if i in prev:
                new[i + 1] = (new[i] - prev[i]) / dist
            else:
                break
        self.taylor = new
        self.last_activated = self.calls

    def _basis_weight(self, i: int, k: int) -> float:
        """Weight of factor ``i`` (≥ 1) at horizon ``k``: ``k^i`` (taylor) or
        ``σ^i · H_i(σk)`` (hermite, physicists' recurrence)."""
        if self.basis == "taylor":
            return float(k ** i)
        x = self.sigma * k
        h_prev, h = 1.0, 2.0 * x                    # H_0, H_1
        for n in range(1, i):
            h_prev, h = h, 2.0 * x * h - 2.0 * n * h_prev
        return self.sigma ** i * h

    def forecast(self) -> torch.Tensor:
        """Extrapolate the residual to the current skipped step. With only the
        0-th factor this returns the last residual unchanged."""
        k = self.calls - self.last_activated
        out = None
        for i, factor in self.taylor.items():
            term = factor if i == 0 else factor * (self._basis_weight(i, k) / math.factorial(i))
            out = term if out is None else out + term
        return out


# --------------------------------------------------------------------------- #
# base DiT
# --------------------------------------------------------------------------- #

class CosmosDiT(nn.Module):
    """Cosmos-Predict2-style DiT backbone (the inner net of Anima)."""

    def __init__(self, cfg: CosmosDiTConfig | None = None):
        super().__init__()
        self.cfg = cfg or CosmosDiTConfig()
        cfg = self.cfg

        in_ch = cfg.in_channels + (1 if cfg.concat_padding_mask else 0)
        self.x_embedder = _PatchEmbed(cfg.patch_temporal, cfg.patch_spatial, in_ch, cfg.model_channels)
        # Sequential so the checkpoint's ``t_embedder.1.*`` keys match.
        self.t_embedder = nn.Sequential(
            _Timesteps(cfg.model_channels),
            _TimestepEmbedding(cfg.model_channels),
        )
        self.t_embedding_norm = RMSNorm(cfg.model_channels, eps=cfg.rms_norm_eps)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.num_blocks)])
        self.final_layer = _FinalLayer(cfg)
        self.pos_embedder = _VideoRoPE3D(cfg)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def _embed(self, x_B_C_T_H_W: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        if self.cfg.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device,
                )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        return x_B_T_H_W_D

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.cfg.patch_spatial, p2=self.cfg.patch_spatial, t=self.cfg.patch_temporal,
        )

    def forward(
        self,
        x: torch.Tensor,                 # (B, C, T, H, W) latent
        timesteps: torch.Tensor,         # (B,) or (B, T)
        context: torch.Tensor,           # (B, L, crossattn_emb_channels)
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        teacache: Optional["TeaCache"] = None,
    ) -> torch.Tensor:
        orig_shape = list(x.shape)
        x = _pad_to_patch_size(x, (self.cfg.patch_temporal, self.cfg.patch_spatial, self.cfg.patch_spatial))

        x_B_T_H_W_D = self._embed(x, padding_mask)
        # (L, D/2, 2, 2) → (1, L, 1, D/2, 2, 2) to broadcast over (B, head).
        rope_emb = self.pos_embedder(x_B_T_H_W_D, fps=fps).unsqueeze(1).unsqueeze(0)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        sample_emb = self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](sample_emb)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D)

        # The residual stream can exceed fp16's range over 28 blocks, so keep it
        # in fp32; blocks cast to the compute dtype around attention/MLP.
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        # TeaCache: on a low-drift step reuse the cached block residual. The
        # timestep embedding and final layer always run. Only "drift" pays for
        # the block-0 modulation probe. σ is read (a host sync) only when a floor
        # is set.
        sigma = (float(timesteps.flatten()[0])
                 if teacache is not None and teacache.sigma_floor > 0 else None)
        if teacache is None:
            compute = True
        elif teacache.rule == "easy":
            compute = teacache.should_compute_easy(x, sigma)
        else:
            compute = teacache.should_compute(
                self.blocks[0].modulated_self_attn_input(x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D),
                sigma,
            )

        if not compute:
            x_B_T_H_W_D = x_B_T_H_W_D + teacache.forecast()
        else:
            residual_in = x_B_T_H_W_D
            for block in self.blocks:
                x_B_T_H_W_D = block(x_B_T_H_W_D, emb_B_T_D, context, rope_emb, adaln_lora_B_T_3D)
            if teacache is not None:
                teacache.update(x_B_T_H_W_D - residual_in)

        out = self.final_layer(x_B_T_H_W_D.to(context.dtype), emb_B_T_D, adaln_lora_B_T_3D)
        out = self._unpatchify(out)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
        if teacache is not None and teacache.rule == "easy":
            teacache.record_output(out)
        return out


# --------------------------------------------------------------------------- #
# AnimaDiT: base + LLMAdapter
# --------------------------------------------------------------------------- #

class AnimaDiT(CosmosDiT):
    """Cosmos-Predict2 base plus a 6-block LLM-Adapter mapping Qwen3 hidden
    states to the 1024-d cross-attention context. With ``t5xxl_ids``,
    ``context`` is Qwen3 hidden states routed through the adapter (optional
    ``t5xxl_weights``, padded to ≥512 tokens); without, it is used directly.
    """

    def __init__(self, cfg: CosmosDiTConfig | None = None, adapter_cfg: LLMAdapterConfig | None = None):
        super().__init__(cfg)
        if adapter_cfg is None:
            adapter_cfg = LLMAdapterConfig(
                source_dim=self.cfg.crossattn_emb_channels,
                target_dim=self.cfg.crossattn_emb_channels,
                model_dim=self.cfg.crossattn_emb_channels,
            )
        self.llm_adapter = LLMAdapter(adapter_cfg)

    def preprocess_text_embeds(
        self,
        source_hidden: torch.Tensor,
        t5xxl_ids: torch.Tensor,
        t5xxl_weights: Optional[torch.Tensor] = None,
        min_seq: int = 512,
    ) -> torch.Tensor:
        out = self.llm_adapter(source_hidden, t5xxl_ids)
        if t5xxl_weights is not None:
            out = out * t5xxl_weights
        if out.shape[1] < min_seq:
            out = F.pad(out, (0, 0, 0, min_seq - out.shape[1]))
        return out

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        t5xxl_ids: Optional[torch.Tensor] = None,
        t5xxl_weights: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        teacache: Optional["TeaCache"] = None,
    ) -> torch.Tensor:
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=t5xxl_weights)
        return super().forward(x, timesteps, context, fps=fps, padding_mask=padding_mask, teacache=teacache)
