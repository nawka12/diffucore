"""Qwen-Image VAE — 3D causal autoencoder; image-only path.

The Qwen-Image VAE is a 16-channel, 8× spatial autoencoder shared with Wan2.1's
video VAE family (Alibaba). It is built around 3D causal convolutions and an
RMS-norm + single-head attention "middle" block. Anima ships this VAE under a
``qwen_image_vae.safetensors`` file with no key prefix.

Diffucore targets *still images* (T=1). The video-style temporal feature-cache
machinery in the upstream Wan implementation is unused at T=1 and is omitted
here; the temporal convolutions inside ``Resample`` blocks still carry their
weights so a strict load works, but they are not invoked on the image path.

Submodule and parameter names mirror the on-disk keys so a ``strict=True`` load
is the correctness check. The channel hierarchy is::

    encoder:  3 → 96 → 192 → 384 → 384 → 32   (32 = z_dim·2, mean+logvar)
    conv1:    32 → 32                          (quant_conv, kernel 1)
    chunk into μ (16) and logσ² (16); μ is the latent
    conv2:    16 → 16                          (post_quant_conv, kernel 1)
    decoder: 16 → 384 → 192 → 96 → 3

Latent normalization is *per-channel* (Wan2.1 statistics), not a scalar
``latent_scale`` — :meth:`process_in` shifts and scales before the DiT,
:meth:`process_out` undoes it after.
"""

from __future__ import annotations

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
    """Conv3d with *causal* temporal padding (zero-padded only on the past).

    The original 3D-conv would pad symmetrically on the time axis; we strip
    that padding and apply ``2·padding_t`` zeros on the past side instead, so a
    kernel-3 conv with ``padding=1`` still preserves the temporal length but
    cannot leak future frames into the present. At T=1 this is equivalent to
    a Conv3d seeing a zero-padded clip of length kernel-size.
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
    """Wan-style RMS-ish norm: L2-normalize then rescale by ``√dim·γ``.

    Differs from standard :class:`nn.RMSNorm` in two ways: it normalizes via
    :func:`torch.nn.functional.normalize` (true L2, not RMS), and the learnable
    gain ``γ`` is broadcast across spatial (and time) dimensions so it can be
    applied to feature maps without rearranging.

    ``has_time_dim=True`` matches the residual blocks (5D input, γ shape
    ``[C, 1, 1, 1]``); ``False`` matches the attention norm (operates after a
    rearrange to 4D, γ shape ``[C, 1, 1]``).
    """

    def __init__(self, dim: int, has_time_dim: bool = True):
        super().__init__()
        broadcast = (1, 1, 1) if has_time_dim else (1, 1)
        self.gamma = nn.Parameter(torch.ones(dim, *broadcast))
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x.dtype)


class Resample(nn.Module):
    """2D/3D up- or down-sample block.

    Mode semantics:

    - ``upsample2d``: nearest-neighbor ×2 spatial, followed by a 3×3 Conv2d
      that halves channels (``dim → dim/2``).
    - ``upsample3d``: same spatial path; additionally carries a temporal
      ``time_conv`` (kernel 3) that doubles channels for frame interleaving.
      Skipped on the image path (T=1, no feature cache).
    - ``downsample2d``: zero-pad on the right/bottom, then 3×3 stride-2 Conv2d
      keeping channels.
    - ``downsample3d``: same spatial path plus a temporal ``time_conv``
      (kernel 3, stride 2). Skipped on the image path.

    ``time_conv`` weights are still constructed so a strict checkpoint load
    succeeds, even though they're unused at T=1.
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
    """Single-head spatial self-attention in the autoencoder bottleneck.

    The norm carries a 2D ``γ`` shape (``[C, 1, 1]``) because we collapse T
    into the batch before applying it.
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
        q = rearrange(q, "n c h w -> n 1 (h w) c")
        k = rearrange(k, "n c h w -> n 1 (h w) c")
        v = rearrange(v, "n c h w -> n 1 (h w) c")
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
            # Each non-initial stage starts after a Resample that halved
            # channels (Conv2d(dim, dim/2)); reflect that in the first block's
            # in-width so a strict load matches.
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
    """Image-only Qwen-Image VAE.

    ``encode(pixels)`` returns the latent ``μ`` (the encoder's logσ² head is
    discarded at inference). ``decode(latents)`` returns RGB pixels.

    Both APIs accept and return *4D* tensors ``(B, C, H, W)`` — the underlying
    3D modules see T=1 internally. ``process_in`` / ``process_out`` apply the
    per-channel Wan2.1 latent normalization the DiT expects.
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
