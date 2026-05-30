"""AutoencoderKL — the SD1.5 VAE (pixels <-> 4-channel latents). [SKELETON]

Implement per ``docs/IMPLEMENTATION_SPEC.md`` §VAE. Contracts (scale factor
0.18215 applied at the boundary):
    encode(image: FloatTensor[B, 3, H, W]) -> latent: FloatTensor[B, 4, H/8, W/8]
    decode(latent: FloatTensor[B, 4, h, w]) -> image: FloatTensor[B, 3, 8h, 8w]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class VAEConfig:
    in_channels: int = 3
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 4
    scale_factor: float = 0.18215


class AutoencoderKL(nn.Module):
    def __init__(self, config: VAEConfig | None = None):
        super().__init__()
        self.config = config or VAEConfig()

    def encode(self, image: torch.Tensor, sample: bool = True, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError("AutoencoderKL.encode — see docs/IMPLEMENTATION_SPEC.md §VAE")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AutoencoderKL.decode — see docs/IMPLEMENTATION_SPEC.md §VAE")
