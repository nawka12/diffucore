"""Qwen-Image VAE: 16-channel, 8× 3D causal autoencoder (Wan2.1 family), image
path only.

At T=1 the temporal feature cache is omitted; the ``Resample`` temporal convs
keep their weights for a strict load but never run. Names mirror the on-disk
keys (no prefix). Channels: encoder 3 → 96 → 192 → 384 → 384 → 32 (μ + logσ²),
quant_conv 32 → 32, post_quant_conv 16 → 16, decoder 16 → 384 → 192 → 96 → 3.
Latent normalization is per-channel (Wan2.1 stats) via :meth:`process_in` /
:meth:`process_out`, not a scalar scale.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Per-channel latent statistics (Wan2.1 / Qwen-Image family).
_WAN21_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
)
_WAN21_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


class CausalConv3d(nn.Conv3d):
    """Conv3d with causal temporal padding: ``2·padding_t`` zeros on the past side
    only, so no future frame leaks in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_pad = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._causal_pad > 0:
            # F.pad order is (W_l, W_r, H_l, H_r, T_l, T_r) for a 5D tensor.
            x = F.pad(x, (0, 0, 0, 0, self._causal_pad, 0))
        return super().forward(x)


class RMSNorm(nn.Module):
    """Wan-style norm: L2-normalize, then scale by ``√dim·γ`` with ``γ`` broadcast
    over the spatial (and time) dims. ``has_time_dim`` picks the 5-D residual
    shape ``[C, 1, 1, 1]`` or the 4-D attention shape ``[C, 1, 1]``.
    """

    def __init__(self, dim: int, has_time_dim: bool = True):
        super().__init__()
        broadcast = (1, 1, 1) if has_time_dim else (1, 1)
        self.gamma = nn.Parameter(torch.ones(dim, *broadcast))
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x.dtype)


class Resample(nn.Module):
    """2D/3D up- or down-sample block: nearest ×2 + 3×3 conv halving channels, or
    right/bottom zero-pad + 3×3 stride-2 conv. The ``*3d`` modes also carry a
    temporal ``time_conv``, built for the strict load but unused at T=1.
    """

    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            raise ValueError(f"unknown resample mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Image path: collapse T into batch, apply 2D resample, restore.
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class ResidualBlock(nn.Module):
    """RMSNorm → SiLU → CausalConv3d, twice, with a 1×1×1 shortcut on width change."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, has_time_dim=True),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, has_time_dim=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention in the bottleneck (T folded into batch,
    hence the 2-D ``γ``).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, has_time_dim=False)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        # Single-head attention: treat (H·W) as the sequence and C as head_dim.
        # Contiguous so SDPA can take the memory-efficient kernel; a strided
        # last dim falls back to math, which materializes the full fp32
        # (H·W)² matrix (5.2 GiB at 1024x1536).
        q = rearrange(q, "n c h w -> n 1 (h w) c").contiguous()
        k = rearrange(k, "n c h w -> n 1 (h w) c").contiguous()
        v = rearrange(v, "n c h w -> n 1 (h w) c").contiguous()
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "n 1 (h w) c -> n c h w", h=h, w=w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x + identity


def _make_middle(dim: int) -> nn.Sequential:
    return nn.Sequential(
        ResidualBlock(dim, dim),
        AttentionBlock(dim),
        ResidualBlock(dim, dim),
    )


class Encoder3d(nn.Module):
    """3 → 32-channel encoder. Output is concatenated (μ, logσ²) of the latent."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,
        input_channels: int = 3,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True),
    ):
        super().__init__()
        dims = [dim * u for u in (1, *dim_mult)]

        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = _make_middle(dims[-1])

        self.head = nn.Sequential(
            RMSNorm(dims[-1], has_time_dim=True),
            nn.SiLU(),
            CausalConv3d(dims[-1], z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        return x


class Decoder3d(nn.Module):
    """16-channel latent → 3-channel image decoder."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        output_channels: int = 3,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_upsample: tuple[bool, ...] = (True, True, False),
    ):
        super().__init__()
        # Decoder starts wide and tapers: dims[0] is the bottleneck channel
        # count (dim·dim_mult[-1]); subsequent entries mirror the encoder.
        dims = [dim * dim_mult[-1]] + [dim * u for u in reversed(dim_mult)]

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = _make_middle(dims[0])

        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Every stage after the first follows a Resample that halved channels.
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMSNorm(dims[-1], has_time_dim=True),
            nn.SiLU(),
            CausalConv3d(dims[-1], output_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv1(z)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class QwenImageVAE(nn.Module):
    """Image-only Qwen-Image VAE on 4-D ``(B, C, H, W)`` tensors (T=1 inside).
    ``encode`` returns ``μ``; ``process_in`` / ``process_out`` apply the Wan2.1
    per-channel latent normalization.
    """

    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(dim=dim, z_dim=z_dim * 2)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim=dim, z_dim=z_dim)
        self.register_buffer(
            "latents_mean", torch.tensor(_WAN21_LATENTS_MEAN).view(1, z_dim, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std", torch.tensor(_WAN21_LATENTS_STD).view(1, z_dim, 1, 1),
            persistent=False,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels.unsqueeze(2)  # (B, 3, H, W) → (B, 3, 1, H, W)
        x = self.encoder(x)
        mu, _logvar = self.conv1(x).chunk(2, dim=1)
        return mu.squeeze(2)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.unsqueeze(2)  # (B, 16, h, w) → (B, 16, 1, h, w)
        z = self.conv2(z)
        x = self.decoder(z)
        return x.squeeze(2)

    def process_in(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latent → DiT-space (zero-mean, unit-std per channel)."""
        return (latents - self.latents_mean.to(latents)) / self.latents_std.to(latents)

    def process_out(self, latents: torch.Tensor) -> torch.Tensor:
        """DiT-space → VAE latent."""
        return latents * self.latents_std.to(latents) + self.latents_mean.to(latents)


_QWEN2D_RESNET = {"norm1": "residual.0", "conv1": "residual.2", "norm2": "residual.3",
                  "conv2": "residual.6", "conv_shortcut": "shortcut"}


def _qwen2d_key(k: str) -> str:
    k = re.sub(r"^quant_conv\.", "conv1.", k)
    k = re.sub(r"^post_quant_conv\.", "conv2.", k)
    k = re.sub(r"^(encoder|decoder)\.conv_in\.", r"\1.conv1.", k)
    k = re.sub(r"^(encoder|decoder)\.norm_out\.", r"\1.head.0.", k)
    k = re.sub(r"^(encoder|decoder)\.conv_out\.", r"\1.head.2.", k)
    k = re.sub(r"\.mid_block\.resnets\.(\d)\.", lambda m: f".middle.{2 * int(m[1])}.", k)
    k = re.sub(r"\.mid_block\.attentions\.0\.", ".middle.1.", k)
    k = re.sub(r"^encoder\.down_blocks\.", "encoder.downsamples.", k)
    # Each decoder up block is 3 resnets + 1 upsampler in the flat Wan list.
    k = re.sub(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.",
               lambda m: f"decoder.upsamples.{4 * int(m[1]) + int(m[2])}.", k)
    k = re.sub(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.",
               lambda m: f"decoder.upsamples.{4 * int(m[1]) + 3}.", k)
    return re.sub(r"(\.(?:downsamples|upsamples|middle)\.\d+)\.(norm1|conv1|norm2|conv2|conv_shortcut)\.",
                  lambda m: f"{m[1]}.{_QWEN2D_RESNET[m[2]]}.", k)


def convert_qwen2d_state_dict(
    sd: dict[str, torch.Tensor], like: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map a Qwen2D checkpoint (Anzhc's image-only Qwen-Image VAE and its tunes,
    https://github.com/Anzhc: diffusers names, temporal axis dropped) onto the
    3-D layout of ``like`` (a :class:`QwenImageVAE` state dict).
    """
    out = {}
    for k, v in sd.items():
        name = _qwen2d_key(k)
        if name not in like:
            raise ValueError(f"unexpected Qwen2D VAE key {k!r}")
        ref = like[name]
        if v.dim() == 4 and ref.dim() == 5:
            # Causal padding means a single frame only meets the last temporal tap.
            w = v.new_zeros(ref.shape)
            w[:, :, -1] = v
            v = w
        elif v.dim() == 3 and ref.dim() == 4:
            v = v.unsqueeze(-1)
        if v.shape != ref.shape:
            raise ValueError(f"Qwen2D VAE key {k!r}: shape {tuple(v.shape)} != {tuple(ref.shape)}")
        out[name] = v
    for name, ref in like.items():
        if name not in out and ".time_conv." in name:
            out[name] = torch.zeros(ref.shape, dtype=ref.dtype)   # ref may be meta
    return out
