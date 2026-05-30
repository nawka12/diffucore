"""SD1.5 UNet — the epsilon-prediction diffusion backbone. [SKELETON]

Implement per ``docs/IMPLEMENTATION_SPEC.md`` §UNet. Contract:
    forward(x: FloatTensor[B, 4, h, w],
            timesteps: FloatTensor[B],
            context: FloatTensor[B, 77, 768]) -> eps: FloatTensor[B, 4, h, w]

``timesteps`` are the continuous indices produced by
``DiscreteSchedule.sigma_to_t``; the UNet embeds them with a sinusoidal
embedding internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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


class UNetModel(nn.Module):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        self.config = config or UNetConfig()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("UNetModel.forward — see docs/IMPLEMENTATION_SPEC.md §UNet")
