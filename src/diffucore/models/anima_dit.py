"""Anima DiT — Cosmos-Predict2-family adaLN transformer (image-only path).

The Anima backbone is a 28-block adaLN-modulated transformer adapted from
NVIDIA's Cosmos-Predict2-2B (Apache-2.0). Each block carries three
independent adaLN-LoRA modulators — one each for self-attention,
cross-attention, and the MLP — so a single time embedding controls three
distinct (shift, scale, gate) triples per block. Positions on
self-attention come from a 3D RoPE; cross-attention has no positional
encoding on the source side.

Anima specifically:

    - in_channels = 16 (Qwen-Image VAE latent) + 1 (concat padding-mask)
    - patch_spatial = 2,  patch_temporal = 1
    - model_channels = 2048,  num_blocks = 28,  num_heads = 16, head_dim = 128
    - crossattn_emb_channels = 1024 (the LLM-Adapter output)
    - use_adaln_lora = True,  adaln_lora_dim = 256
    - rope3d head split:  dim_h = head_dim//6*2 = 42,
                          dim_w = 42, dim_t = head_dim - 84 = 44
    - extra_per_block_abs_pos_emb = False

The forward path treats T=1 single-frame video tensors throughout (the
``(B, C, H, W)`` latent is reshaped on entry / exit) so the same module
covers future video extensions without an early architectural lock-in.

Verification (DT5): key-set + strict-load against ``anima-base-v1.0.safetensors``
+ behavioural shape/determinism/conditioning-sensitivity tests. Numerical
bit-match is deferred to DT7 (end-to-end image vs ComfyUI reference) for
the same reason as DT4 — ComfyUI is not importable in this venv.
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
    """Reflect-pad the last 3 dims (T, H, W) so each is a multiple of its patch.

    The original code uses ``"circular"`` for non-trace builds; reflect is the
    safer fallback (a circular pad on a slightly off-divisible image leaks
    pixels from the opposite edge into the seam). For images the difference is
    only at the last row/column, and Anima latents at 1024² are already
    cleanly divisible by patch 2 — this branch only fires on odd sizes.
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
    """Sinusoidal timestep embedding — no parameters."""
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
    """``Linear(D→D) · SiLU · Linear(D→3·D)``. The second Linear's output is
    the per-block adaLN-LoRA "delta" added to each ``adaln_modulation_*``
    output before chunking; the carried ``emb`` (the SiLU input/sample) is
    what the per-stage SiLU + Linear pair operates on."""
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
    """Rearrange ``(B, C, T, H, W) → (B, T/r, H/m, W/n, C·r·m·n)`` then Linear
    to ``model_channels``. Stored under ``proj.1`` (the Rearrange is index 0;
    Linear is index 1) so it matches the checkpoint's ``x_embedder.proj.1.weight``.
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
# 3D RoPE — Apache-2.0 algorithm from NVIDIA Cosmos
# --------------------------------------------------------------------------- #

class _VideoRoPE3D(nn.Module):
    """Three-axis (T, H, W) RoPE that returns a ``(L, head_dim/2, 2, 2)``
    rotation-matrix tensor consumed by :func:`_apply_rope`.

    The head_dim is split into (dim_h, dim_w, dim_t) with the spatial axes
    getting ``head_dim//6·2`` each and the temporal axis getting the
    remainder. For Anima head_dim=128 → 42/42/44. NTK extrapolation factors
    scale the base θ per axis.
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
        cache_key = (H, W, T, fps_key)
        cached = getattr(self, "_rope_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1].to(device)
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

        # [cos, -sin, sin, cos] per (pos, freq) → encodes a 2x2 rotation matrix.
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
        self._rope_cache = (cache_key, result.cpu())
        return result.to(device)


def _apply_rope(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply the 2×2 rotation encoded in ``freqs`` (shape ``(L, D/2, 2, 2)``)
    to each pair of channels of ``t`` (shape ``(B, ..., L, D)``)."""
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    out = out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return out


# --------------------------------------------------------------------------- #
# attention and MLP
# --------------------------------------------------------------------------- #

class _Attention(nn.Module):
    """Cosmos-style attention with per-head q/k RMSNorm.

    Cross-attn passes ``context_dim < query_dim``; self-attn has
    ``context_dim is None`` and applies 3D RoPE to q and k.
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
        # SDPA: this attention is not validated against an oracle in DT5;
        # using SDPA here is a perf win that DT7 will check end-to-end. For
        # MLP-free (large head_dim, image-style) attention the math/flash
        # paths land at the same answer within fp16 tolerance.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.output_proj(out)


class _GPT2FeedForward(nn.Module):
    """Linear → GELU → Linear, no bias. Stored under ``layer1`` / ``layer2``."""
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
    """``SiLU → Linear(D→r) → Linear(r→3D)`` — keyed ``.1.weight``, ``.2.weight``
    to match the checkpoint's ``adaln_modulation_X.{1,2}.weight`` (index 0 is
    the SiLU and carries no parameters)."""
    def __init__(self, dim: int, r: int):
        super().__init__(
            nn.SiLU(),
            nn.Linear(dim, r, bias=False),
            nn.Linear(r, 3 * dim, bias=False),
        )


class _Block(nn.Module):
    """One adaLN-LoRA block. ``layer_norm_*`` carry no learnable params
    (``elementwise_affine=False``); the adaLN modulators provide all
    per-token affine shaping."""
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
        x: torch.Tensor,                # (B, T, H, W, D) — residual_dtype (fp32 in fp16-inference mode)
        emb: torch.Tensor,              # (B, T, D)        — compute_dtype (fp16 in fp16-inference mode)
        ctx: torch.Tensor,              # (B, L, ctx_dim)
        rope_emb: torch.Tensor,         # ((T·H·W), head_dim/2, 2, 2)
        adaln_lora: torch.Tensor,       # (B, T, 3·D)
    ) -> torch.Tensor:
        residual_dtype = x.dtype
        compute_dtype = emb.dtype       # whatever the model was loaded in (fp16 on CUDA)
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
        # Self-attn: normalize in residual_dtype (so the modulation stays
        # numerically gentle), cast to compute_dtype for attention, cast back
        # before the gated residual add so accumulation stays in fp32.
        h = self.layer_norm_self_attn(x) * (1 + sa_sc) + sa_s
        h_seq = rearrange(h, "b t h w d -> b (t h w) d").to(compute_dtype)
        h_seq = self.self_attn(h_seq, None, rope_emb)
        h = rearrange(h_seq, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype)
        x = x + sa_g.to(residual_dtype) * h

        # Cross-attn: same pattern.
        h = self.layer_norm_cross_attn(x) * (1 + ca_sc) + ca_s
        h_seq = rearrange(h, "b t h w d -> b (t h w) d").to(compute_dtype)
        h_seq = self.cross_attn(h_seq, ctx, None)
        h = rearrange(h_seq, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype)
        x = x + ca_g.to(residual_dtype) * h

        # MLP: same pattern.
        h = self.layer_norm_mlp(x) * (1 + ml_sc) + ml_s
        h = self.mlp(h.to(compute_dtype)).to(residual_dtype)
        x = x + ml_g.to(residual_dtype) * h
        return x

    def modulated_self_attn_input(
        self, x: torch.Tensor, emb: torch.Tensor, adaln_lora: torch.Tensor
    ) -> torch.Tensor:
        """The timestep-modulated tokens entering self-attention — TeaCache's
        cheap proxy for "has the model's input changed". Reproduces exactly the
        self-attn modulation from :meth:`forward` (adaLN affine over the
        normalized tokens) without the attention/MLP that follow."""
        sa = self.adaln_modulation_self_attn(emb) + adaln_lora
        sa_s, sa_sc, _ = sa.chunk(3, dim=-1)
        sa_s = rearrange(sa_s, "b t d -> b t 1 1 d")
        sa_sc = rearrange(sa_sc, "b t d -> b t 1 1 d")
        return self.layer_norm_self_attn(x) * (1 + sa_sc) + sa_s


# --------------------------------------------------------------------------- #
# final layer
# --------------------------------------------------------------------------- #

class _FinalLayer(nn.Module):
    """Two-chunk adaLN (shift, scale — no gate), then Linear to
    ``patch_t · patch_s² · out_channels``. The output is the pre-unpatchify
    tensor that the DiT's :meth:`unpatchify` reshapes into a (B, C, T, H, W)
    latent."""
    def __init__(self, cfg: CosmosDiTConfig):
        super().__init__()
        d = cfg.model_channels
        r = cfg.adaln_lora_dim
        patch_out = cfg.patch_temporal * cfg.patch_spatial**2 * cfg.out_channels
        self.layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.rms_norm_eps)
        # n_adaln_chunks = 2 (shift+scale only); .1 and .2 = Linear pair.
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, r, bias=False),
            nn.Linear(r, 2 * d, bias=False),
        )
        self.linear = nn.Linear(d, patch_out, bias=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, adaln_lora: torch.Tensor) -> torch.Tensor:
        # The 4096-d adaLN output gets the first 2·D slice of adaln_lora added.
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
    """Timestep-Embedding-Aware Cache for one DiT denoising stream
    (Liu et al., 2024, arXiv:2411.19108).

    The transformer blocks are the bulk of a DiT forward, yet their output
    changes slowly between adjacent timesteps. TeaCache caches the blocks'
    *residual* (``stack(x) - x``) and, on a step where the timestep-modulated
    input has drifted little from the last computed step, *forecasts* that
    residual instead of running the blocks — a ~28×-cheaper step.

    "Little" is measured as the accumulated rescaled relative-L1 change of the
    block-0 modulated input; once it crosses ``rel_l1_thresh`` a real recompute
    is forced and the accumulator resets. Larger threshold = more skipped steps
    = faster but lower fidelity.

    **Forecast (TaylorSeer, arXiv:2503.06923).** The residual is not frozen on a
    skipped step: it is Taylor-extrapolated from its own recent history. Each
    computed step refreshes finite-difference derivatives of the residual over
    the *activation* steps (the steps that actually ran the blocks); a skipped
    step ``k`` steps past the last activation returns
    ``Σ_i residual^(i) · k^i / i!``. ``max_order`` caps the derivative order:
    ``0`` reproduces the original cache-then-reuse (a held constant), ``1`` a
    linear extrapolation, higher a polynomial one. Order ``1`` is the default —
    it tracks the residual's drift between activations, so the same skip
    *decision* yields a markedly better skip *output* than freezing, with no
    change to calibration or thresholds (those govern only *when* to skip).
    Forecasting only engages once ≥2 activations exist; before that (and at
    order 0) a skip reuses the last residual exactly, as before.

    One instance tracks one stream. Classifier-free guidance needs two (the
    conditioned and unconditioned passes are separate Anima forwards whose
    modulated inputs coincide, so a shared accumulator would read zero drift
    between them).

    ``coefficients`` are a polynomial (highest-degree first, ``numpy.poly1d``
    convention) that rescales the raw relative-L1 into an output-change
    estimate. They are model-specific and require offline calibration; Anima
    has none published, so the default is the identity ``f(x) = x``.
    """

    def __init__(self, rel_l1_thresh: float, coefficients: Sequence[float] = (1.0, 0.0),
                 *, record: bool = False, max_order: int = 1):
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.record = record
        self.max_order = int(max_order)
        self.prev_modulated: Optional[torch.Tensor] = None
        self.accumulated = 0.0
        self.calls = 0   # forwards seen
        self.skips = 0   # forwards whose blocks were reused from cache
        self.rel_history: list[float] = []   # raw per-step rel-L1 (record mode)
        # Taylor forecast state: ``taylor[i]`` is the i-th finite difference of
        # the block residual over activation steps (``taylor[0]`` the residual
        # itself). ``last_activated`` is the ``calls`` index of the last computed
        # step, so a skip at ``calls`` extrapolates ``calls - last_activated``
        # steps forward. Empty until the first computed step.
        self.taylor: dict[int, torch.Tensor] = {}
        self.last_activated = -1

    def _rescale(self, x: float) -> float:
        out = 0.0
        for c in self.coefficients:  # Horner, poly1d order (highest degree first)
            out = out * x + c
        return out

    def should_compute(self, modulated: torch.Tensor) -> bool:
        """Decide whether this step must run the blocks, and fold ``modulated``
        into the accumulator. The first call (no history) always computes.

        In ``record`` mode the accumulator/threshold are bypassed: every step
        computes and the raw relative-L1 is logged to ``rel_history`` — that's
        the ``x`` of the (input-drift -> output-drift) fit done by calibration.
        """
        self.calls += 1
        if self.prev_modulated is None:
            self.accumulated = 0.0
            self.prev_modulated = modulated
            return True
        denom = self.prev_modulated.abs().mean().clamp_min(1e-8)
        rel = ((modulated - self.prev_modulated).abs().mean() / denom).item()
        self.prev_modulated = modulated
        if self.record:
            self.rel_history.append(rel)
            return True
        self.accumulated += self._rescale(rel)
        if self.accumulated < self.rel_l1_thresh:
            self.skips += 1
            return False
        self.accumulated = 0.0
        return True

    def update(self, residual: torch.Tensor) -> None:
        """Record a freshly computed block residual and refresh the Taylor
        finite-difference factors used to forecast skipped steps.

        The i-th factor is the i-th finite difference of the residual over
        activation steps, divided by the (possibly uneven) step gap between the
        last two activations — so ``forecast`` reads a per-step derivative. At
        ``max_order == 0``, or on the first activation, only the 0-th factor (the
        residual itself) is kept, which makes a subsequent skip reuse it exactly.
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

    def forecast(self) -> torch.Tensor:
        """Taylor-extrapolate the block residual to the current (skipped) step
        from the factors :meth:`update` last recorded. With only the 0-th factor
        (order 0, or fewer than two activations) this returns the last residual
        unchanged — the original cache-then-reuse behavior."""
        k = self.calls - self.last_activated
        out = None
        for i, factor in self.taylor.items():
            term = factor if i == 0 else factor * (k ** i / math.factorial(i))
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
        # Sequential([Timesteps, TimestepEmbedding]) so the checkpoint's
        # ``t_embedder.1.linear_*.weight`` keys land on the right submodule.
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
        # pos_embedder returns (L, D/2, 2, 2); insert singleton dims so the
        # L axis broadcasts over (B, head) when applied to a (B, L, H, D)
        # query in _apply_rope. Shape becomes (1, L, 1, D/2, 2, 2).
        rope_emb = self.pos_embedder(x_B_T_H_W_D, fps=fps).unsqueeze(1).unsqueeze(0)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        sample_emb = self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](sample_emb)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D)

        # The Cosmos residual stream has large magnitude — over 28 blocks the
        # accumulated norm can overshoot fp16's ±65504 ceiling and saturate
        # to inf/NaN. Promote x to fp32 here so additions across the block
        # chain stay representable; each block re-casts to compute_dtype on
        # the way into attention/MLP and back out before the residual add.
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        # TeaCache: on a low-drift step, reuse the cached block residual instead
        # of running the 28-block stack. The timestep embedding (above) and the
        # final layer (below) are cheap and always run; only the blocks skip.
        compute = teacache.should_compute(
            self.blocks[0].modulated_self_attn_input(x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D)
        ) if teacache is not None else True

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
        return out


# --------------------------------------------------------------------------- #
# AnimaDiT — base + LLMAdapter
# --------------------------------------------------------------------------- #

class AnimaDiT(CosmosDiT):
    """Cosmos-Predict2 base + a 6-block LLM-Adapter that bridges Qwen3 hidden
    states (DT3) into the 1024-d cross-attention context the DiT consumes.

    ``forward`` accepts both the already-prepared ``context`` (1024-d) and the
    optional ``t5xxl_ids`` path used by the actual generation pipeline: when
    ``t5xxl_ids`` is given, the supplied ``context`` is treated as Qwen3 source
    hidden states and is routed through the adapter (with optional
    ``t5xxl_weights`` per-token scaling and padding to ≥512 tokens) before
    the DiT cross-attends to it. When ``t5xxl_ids`` is None the ``context``
    is used directly (useful for unit tests and ablations).
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
