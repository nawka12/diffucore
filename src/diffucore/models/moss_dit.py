"""MOSS-Image DiT (Base and SFT): single-stream rectified-flow transformer.

Ports the modules of the reference ``pipeline_moss_image.py`` from
OpenMOSS-Team/MOSS-Image-SFT (Apache-2.0); module names mirror it, so the
official diffusers shards (after :func:`from_diffusers` fuses q/k/v) and the
Comfy-Org bf16 / INT8 files (which ship the fused ``proj.qkv``) load strictly.

The DiT takes the 4-D latent ``[B, 128, H/16, W/16]`` directly (patch 1) and
runs 42 blocks (3072 wide, 24 heads of 128, SITU feed-forward) over the joint
sequence image tokens, then text. One shared modulation feeds every block; the
blocks own no modulation weights. RoPE has four axes (r, f, h, w) and the image
tokens sit at ``f = L``, the text length, so image RoPE depends on the prompt.
No padding or mask exists anywhere: a sample runs at its true text length.

Everything that depends on the prompt alone (:meth:`MossDiT.encode_text`) or on
the step alone (:meth:`MossDiT.time_vectors`) is exposed separately so the
pipeline computes it once for both CFG branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ._attention import attention_blhd

_EPS = 1e-6


@dataclass
class MossConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0

    @classmethod
    def from_state_dict(cls, sd) -> "MossConfig":
        """Derive the config from a checkpoint's keys (fused ``proj.qkv`` layout);
        works on int8 files too, whose quantized weights keep their shapes."""
        dim, latent = sd["video_embedder.weight"].shape
        layers = 0
        while f"layers.{layers}.proj.qkv.weight" in sd:
            layers += 1
        return cls(
            dim=dim, ffn_dim=sd["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
            num_heads=dim // sd["layers.0.q_norm.weight"].shape[0], num_layers=layers,
            text_dim=sd["text_embedder.proj.weight"].shape[1],
            freq_dim=sd["time_embedding.0.weight"].shape[1], latent_channels=latent,
        )


class _RMSNorm(nn.Module):
    """fp32 normalization cast back to the input dtype, then the plain
    ``weight`` scale (not ``1 + weight``). ``eps`` is per call so a scaled
    residual stream can use ``eps * c^2``."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * self.weight


def _situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on ``x`` (B, L, H, D) with per-token ``cos``/``sin``
    (L, D), in the activation dtype like the reference."""
    a, b = x.chunk(2, -1)
    return x * cos[None, :, None] + torch.cat((-b, a), -1) * sin[None, :, None]


def _attend(q, k, v, cos, sin, backend: str = "sdpa") -> torch.Tensor:
    return attention_blhd(_rotate(q, cos, sin), _rotate(k, cos, sin), v, backend)


def _split_qkv(qkv: torch.Tensor, heads: int):
    B, L, _ = qkv.shape
    return qkv.view(B, L, 3, heads, -1).unbind(2)


class _TextBlock(nn.Module):
    """Pre-norm refiner block of the text embedder (biased linears, not modulated).
    ``rs`` is the residual-stream scale: branch outputs are scaled by it and
    the norms use ``eps * rs^2``."""

    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, cos, sin, rs: float = 1.0):
        eps = _EPS * rs * rs
        q, k, v = _split_qkv(self.qkv(self.norm1(x, eps)), self.heads)
        out = self.proj(_attend(self.q_norm(q), self.k_norm(k), v, cos, sin))
        x = x + (out * rs if rs != 1.0 else out)
        y = self.norm2(x, eps)
        out = self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))
        return x + (out * rs if rs != 1.0 else out)


class _TextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, cos, sin, rs: float = 1.0):
        x = self.proj(x.to(self.proj.weight.dtype))
        if rs != 1.0:
            x = x * rs
        for block in self.blocks:
            x = block(x, cos, sin, rs)
        return x


class _Block(nn.Module):
    """Joint block. ``proj`` holds the four bias-free linears under the
    checkpoint's names (``qkv`` fused; the reference keeps q/k/v apart).
    ``attn_backend`` is the fa2/int8 dispatch slot."""

    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            "qkv": nn.Linear(dim, 3 * dim, bias=False),
            "o": nn.Linear(dim, dim, bias=False),
            "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
            "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
        })
        self.attn_backend = "sdpa"

    def forward(self, x, cos, sin, mod, eps: float = _EPS):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.unbind(1)
        y = self.norm1(x, eps) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = _split_qkv(self.proj["qkv"](y), self.heads)
        attn = _attend(self.q_norm(q), self.k_norm(k), v, cos, sin, self.attn_backend)
        x = x + self.proj["o"](attn) * gate_a[:, None]
        y = self.norm2(x, eps) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class MossDiT(nn.Module):
    """``forward(x (B, C, h, w), mod, head, text) -> v`` with ``mod``/``head``
    from :meth:`time_vectors` (given σ) and ``text`` from :meth:`encode_text`.

    ``residual_scale`` carries the joint stream at that fraction of its true
    size, for fp16 (the bf16-trained stream peaks near 6.2e4 at 1024², fp16's
    max is 65504): the two stream writers (``video_embedder``, the text
    embedder) and both gates are scaled, and every reader is an RMSNorm
    (scale-invariant), so the result is exact up to ε, which uses ``eps * c^2``.
    Checkpoints that already carry the 1/16 in their weights run at 1.0."""

    def __init__(self, cfg: MossConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or MossConfig()
        if cfg.dim % cfg.num_heads or (cfg.dim // cfg.num_heads) % 8 or cfg.freq_dim % 2:
            raise ValueError("MOSS needs dim divisible by heads, head_dim divisible by 8 and an even freq_dim")
        self.residual_scale = 1.0
        self.head_dim = cfg.dim // cfg.num_heads
        self.video_embedder = nn.Linear(cfg.latent_channels, cfg.dim)
        self.text_embedder = _TextProjection(cfg.text_dim, cfg.dim, cfg.ffn_dim, cfg.num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(cfg.freq_dim, cfg.dim), nn.SiLU(),
                                            nn.Linear(cfg.dim, cfg.dim))
        self.shared_block_modulation = nn.Linear(cfg.dim, 6 * cfg.dim)
        self.layers = nn.ModuleList([_Block(cfg.dim, cfg.ffn_dim, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.final_norm_video = _RMSNorm(cfg.dim)
        self.final_linear_video = nn.Linear(cfg.dim, cfg.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(cfg.dim, 2 * cfg.dim)
        self._rope_cache: dict = {}

    def _phase_tables(self, coords: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ``(cos, sin)`` of shape (S, head_dim) for ``coords`` (S, 4): the
        4 axes x head_dim/8 bands phase, axis-major, duplicated for rotate-half.
        The bands are built on the CPU in fp32 like the reference."""
        n = self.head_dim // 8
        bands = 1 / (self.cfg.rope_base ** (torch.arange(n, dtype=torch.float32) / n))
        phase = (coords.unsqueeze(-1) * bands).flatten(-2).to(device)
        sin, cos = phase.sin(), phase.cos()
        return torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)

    @staticmethod
    def _text_coords(txt_len: int) -> torch.Tensor:
        text = torch.zeros(txt_len, 4)
        text[:, 1] = torch.arange(txt_len)
        return text

    def rope(self, txt_len: int, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` for the joint sequence, image tokens first (h = w = 0:
        the text tokens alone). Image token (i, j) sits at ``(0, L, i - (h-1)/2,
        j - (w-1)/2)``, text token t at ``(0, t, 0, 0)``."""
        key = (txt_len, h, w, str(device))
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        gi, gj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        image = torch.stack((torch.zeros_like(gi), torch.full_like(gi, txt_len), gi, gj), -1).flatten(0, 1).float()
        image[:, 2] -= (h - 1) / 2
        image[:, 3] -= (w - 1) / 2
        hit = self._phase_tables(torch.cat((image, self._text_coords(txt_len)), 0), device)
        if len(self._rope_cache) > 8:
            self._rope_cache.clear()
        self._rope_cache[key] = hit
        return hit

    def encode_text(self, feats: torch.Tensor) -> torch.Tensor:
        """Text-encoder features ``(B, L, text_dim)`` -> the stream's text tokens
        ``(B, L, dim)`` (at the residual scale). Prompt-only, run once per prompt."""
        cos, sin = self.rope(feats.shape[1], 0, 0, feats.device)
        dtype = self.text_embedder.proj.weight.dtype
        if not dtype.is_floating_point:
            dtype = feats.dtype
        return self.text_embedder(feats, cos.to(dtype), sin.to(dtype), self.residual_scale)

    def time_vectors(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mod (B, 6, dim), head (B, 2, dim))`` for σ ``t`` (B,) in [0, 1]; the
        model's timestep is σ * 1000. The sinusoid is float64 like the reference.
        The two gate rows come out pre-scaled by the residual scale."""
        half = self.cfg.freq_dim // 2
        times = t.float().reshape(-1) * 1000.0
        freq = 10000 ** (-torch.arange(half, device=t.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        emb = torch.cat((phase.cos(), phase.sin()), -1).float()
        dtype = self.time_embedding[0].weight.dtype
        if not dtype.is_floating_point:
            dtype = torch.float32
        time = self.time_embedding(emb.to(dtype)).float()
        B = time.shape[0]
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(B, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(B, 2, -1)
        rs = self.residual_scale
        if rs != 1.0:
            mod = torch.cat((mod[:, :2], mod[:, 2:3] * rs, mod[:, 3:5], mod[:, 5:] * rs), 1)
        return mod, head

    def forward(self, x: torch.Tensor, mod: torch.Tensor, head: torch.Tensor, text: torch.Tensor,
                skip_layers=()) -> torch.Tensor:
        B, C, h, w = x.shape
        n_img = h * w
        dtype = text.dtype
        skip = frozenset(skip_layers)
        bad = [i for i in skip if not 0 <= i < len(self.layers)]
        if bad:
            raise ValueError(f"skip_layers must be block indices in [0, {len(self.layers)}); got {sorted(bad)}")
        rs = self.residual_scale
        eps = _EPS * rs * rs
        video = self.video_embedder(x.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())
        if rs != 1.0:
            video = video * rs
        cos, sin = self.rope(text.shape[1], h, w, x.device)
        cos, sin = cos.to(dtype), sin.to(dtype)
        stream = torch.cat((video.to(dtype), text), 1)
        for i, blk in enumerate(self.layers):
            if i not in skip:
                stream = blk(stream, cos, sin, mod, eps)
        y = self.final_norm_video(stream[:, :n_img], eps)
        y = y * (1 + head[:, 1, None]) + head[:, 0, None]
        y = self.final_linear_video(y.to(self.final_linear_video.weight.dtype))
        return y.transpose(1, 2).reshape(B, -1, h, w)


def int8_layer_names(num_blocks: int) -> list[str]:
    """What the ``int8`` load option quantizes: the four linears of every joint
    block, as in the ComfyUI MOSS INT8 file (the text embedder stays bf16)."""
    return [f"layers.{i}.proj.{p}" for i in range(num_blocks) for p in ("qkv", "o", "ffn_up_gate", "ffn_down")]


def from_diffusers(state_dict: dict) -> dict:
    """An official diffusers-format MOSS transformer state dict in the fused
    layout: each block's ``proj.q/k/v`` concatenated (rows q, k, v) into
    ``proj.qkv``. The q/k/v tensors are popped from ``state_dict`` as they are fused."""
    out = {}
    for k in list(state_dict):
        if k.endswith(".proj.q.weight"):
            base = k[: -len("q.weight")]
            out[base + "qkv.weight"] = torch.cat([state_dict.pop(base + n + ".weight") for n in "qkv"], 0)
        elif not k.endswith((".proj.k.weight", ".proj.v.weight")):
            out[k] = state_dict[k]
    return out


def drop_unused_keys(state_dict: dict) -> dict:
    """The checkpoint entries our modules don't own: the RoPE bands (re-derived
    in fp32) and the Comfy attention-backend hints."""
    return {k: v for k, v in state_dict.items()
            if k != "rope.bands" and not k.endswith(".comfy_attention.config")}
