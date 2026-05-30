"""CLIP ViT-L/14 text transformer — the SD1.5 conditioner. [SKELETON]

Implement per ``docs/IMPLEMENTATION_SPEC.md`` §CLIP. Contract:
    forward(token_ids: LongTensor[B, 77], clip_skip: int = 1)
        -> hidden_states: FloatTensor[B, 77, 768]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CLIPTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5


class CLIPTextEncoder(nn.Module):
    def __init__(self, config: CLIPTextConfig | None = None):
        super().__init__()
        self.config = config or CLIPTextConfig()

    def forward(self, token_ids: torch.Tensor, clip_skip: int = 1) -> torch.Tensor:
        raise NotImplementedError("CLIPTextEncoder.forward — see docs/IMPLEMENTATION_SPEC.md §CLIP")
