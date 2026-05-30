"""SD1.5 UNet — the epsilon-prediction diffusion backbone.

Implements the LDM/Stable-Diffusion UNet (Rombach et al., 2022) with the
cross-attention conditioning of Vaswani et al. (2017). Submodule and parameter
names mirror the on-disk ``model.diffusion_model.*`` keys (the original LDM
naming) so a ``strict=True`` load is the correctness check.

Contract:
    forward(x: FloatTensor[B, 4, h, w],
            timesteps: FloatTensor[B],
            context: FloatTensor[B, 77, 768]) -> eps: FloatTensor[B, 4, h, w]

``timesteps`` are the continuous indices produced by
``DiscreteSchedule.sigma_to_t``; they are embedded with a sinusoidal embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 4
    model_channels: int = 320
    out_channels: int = 4
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (4, 2, 1)
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_heads: int = 8
    context_dim: int = 768
    transformer_depth: int = 1


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (cos then sin), as in LDM."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that normalizes in fp32 for stability under fp16 weights."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x.float(), self.num_groups, self.weight.float(), self.bias.float(), self.eps
        ).type(x.dtype)


def normalization(channels: int) -> GroupNorm32:
    return GroupNorm32(32, channels)


class TimestepEmbedSequential(nn.Sequential):
    """Sequential that routes the time embedding to ResBlocks and the text
    context to SpatialTransformers; everything else gets only ``x``."""

    def forward(self, x, emb, context):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int, emb_channels: int, out_channels: int):
        super().__init__()
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if channels != out_channels:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        b = x.shape[0]

        def split(t):
            return t.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        out = F.scaled_dot_product_attention(q, k, v)  # scale = 1/sqrt(dim_head)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, context_dim: int):
        super().__init__()
        dim_head = dim // num_heads
        self.attn1 = CrossAttention(dim, dim, num_heads, dim_head)        # self-attention
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, num_heads, dim_head)  # cross-attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, num_heads: int, context_dim: int, depth: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock(channels, num_heads, context_dim) for _ in range(depth)
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = x.reshape(b, c, h * w).transpose(1, 2)  # b, hw, c
        for block in self.transformer_blocks:
            x = block(x, context)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.proj_out(x) + x_in


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNetModel(nn.Module):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        cfg = config or UNetConfig()
        self.config = cfg
        mc = cfg.model_channels
        time_embed_dim = mc * 4

        self.time_embed = nn.Sequential(
            nn.Linear(mc, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        def make_transformer(channels):
            return SpatialTransformer(channels, cfg.num_heads, cfg.context_dim, cfg.transformer_depth)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(cfg.in_channels, mc, 3, padding=1))]
        )
        input_block_chans = [mc]
        ch = mc
        ds = 1
        for level, mult in enumerate(cfg.channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, mult * mc)]
                ch = mult * mc
                if ds in cfg.attention_resolutions:
                    layers.append(make_transformer(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(cfg.channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, ch),
            make_transformer(ch),
            ResBlock(ch, time_embed_dim, ch),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, mult * mc)]
                ch = mult * mc
                if ds in cfg.attention_resolutions:
                    layers.append(make_transformer(ch))
                if level != 0 and i == cfg.num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv2d(mc, cfg.out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.config.model_channels)
        emb = self.time_embed(t_emb.to(x.dtype))

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)
