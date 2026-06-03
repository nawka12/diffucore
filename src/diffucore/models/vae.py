"""AutoencoderKL — the SD1.5 VAE (pixels <-> 4-channel latents).

Implements the original LDM/CompVis autoencoder (Rombach et al., 2022, building
on Esser et al., 2021). Submodule and parameter names mirror the on-disk
``first_stage_model.*`` keys so a ``strict=True`` load is the correctness check.

Contracts (scale factor 0.18215 applied at the boundary, not inside the nets):
    encode(image: FloatTensor[B, 3, H, W]) -> latent: FloatTensor[B, 4, H/8, W/8]
    decode(latent: FloatTensor[B, 4, h, w]) -> image: FloatTensor[B, 3, 8h, 8w]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    in_channels: int = 3
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 4
    scale_factor: float = 0.18215
    # FLUX's autoencoder applies a per-tensor *shift* before the scale and ships
    # without the quant / post-quant 1x1 convs. Both default to the SD behaviour
    # (shift 0, convs present) so SD1.5/SDXL loads and numerics are unchanged.
    shift_factor: float = 0.0
    use_quant_conv: bool = True


def Normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """GroupNorm->SiLU->conv3x3 twice, with a 1x1 shortcut when channels change."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Spatial self-attention over H*W with 1x1 q/k/v/proj convs."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = Normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, height, width = q.shape
        # [B, C, H, W] -> [B, HW, C] so each spatial location is a token.
        q = q.reshape(b, c, height * width).transpose(1, 2)
        k = k.reshape(b, c, height * width).transpose(1, 2)
        v = v.reshape(b, c, height * width).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)  # scale = 1/sqrt(c)
        h = h.transpose(1, 2).reshape(b, c, height, width)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    """Strided conv with asymmetric (right/bottom) padding, as in LDM."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour x2 followed by a 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = cfg.base_channels
        num_resolutions = len(cfg.channel_mult)
        self.conv_in = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        in_ch_mult = (1,) + tuple(cfg.channel_mult)
        self.down = nn.ModuleList()
        for i in range(num_resolutions):
            block_in = ch * in_ch_mult[i]
            block_out = ch * cfg.channel_mult[i]
            level = nn.Module()
            level.block = nn.ModuleList(
                ResnetBlock(block_in if j == 0 else block_out, block_out)
                for j in range(cfg.num_res_blocks)
            )
            if i != num_resolutions - 1:
                level.downsample = Downsample(block_out)
            self.down.append(level)

        mid_ch = ch * cfg.channel_mult[-1]
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(mid_ch, mid_ch)
        self.mid.attn_1 = AttnBlock(mid_ch)
        self.mid.block_2 = ResnetBlock(mid_ch, mid_ch)

        self.norm_out = Normalize(mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, 2 * cfg.z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = cfg.base_channels
        num_resolutions = len(cfg.channel_mult)
        block_in = ch * cfg.channel_mult[-1]
        self.conv_in = nn.Conv2d(cfg.z_channels, block_in, 3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # Built ascending by level so up[i] matches level i; forward walks it in
        # reverse (deepest first). num_res_blocks + 1 blocks per level.
        self.up = nn.ModuleList([nn.Module() for _ in range(num_resolutions)])
        cur_in = block_in
        for i in reversed(range(num_resolutions)):
            block_out = ch * cfg.channel_mult[i]
            level = self.up[i]
            blocks = []
            for j in range(cfg.num_res_blocks + 1):
                blocks.append(ResnetBlock(cur_in if j == 0 else block_out, block_out))
            level.block = nn.ModuleList(blocks)
            if i != 0:
                level.upsample = Upsample(block_out)
            cur_in = block_out

        self.norm_out = Normalize(cur_in)
        self.conv_out = nn.Conv2d(cur_in, cfg.in_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i in reversed(range(len(self.up))):
            level = self.up[i]
            for block in level.block:
                h = block(h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class AutoencoderKL(nn.Module):
    def __init__(self, config: VAEConfig | None = None):
        super().__init__()
        self.config = config or VAEConfig()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        if self.config.use_quant_conv:
            self.quant_conv = nn.Conv2d(2 * self.config.z_channels, 2 * self.config.z_channels, 1)
            self.post_quant_conv = nn.Conv2d(self.config.z_channels, self.config.z_channels, 1)
        else:
            self.quant_conv = None
            self.post_quant_conv = None

    def encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        moments = self.encoder(image)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments)
        mean, logvar = moments.chunk(2, dim=1)
        if sample:
            logvar = logvar.clamp(-30.0, 20.0)
            std = (0.5 * logvar).exp()
            noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
            z = mean + std * noise
        else:
            z = mean
        # latent = scale·(z − shift); shift defaults to 0 (SD), so this stays
        # the plain scale for SD1.5/SDXL and adds FLUX's pre-shift when set.
        return self.config.scale_factor * (z - self.config.shift_factor)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        z = latent / self.config.scale_factor + self.config.shift_factor
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)
