"""Detect a checkpoint's architecture from its tensor keys and shapes.

Stable Diffusion checkpoints follow a de-facto layout (the original LDM naming):
the diffusion UNet lives under ``model.diffusion_model.``, the VAE under
``first_stage_model.``, and the text encoder under a ``cond_stage_model.`` /
``conditioner.`` prefix. The text *context dimension* — read off the UNet's
cross-attention key projection (``attn2.to_k``) — distinguishes the families:
768 = SD1.x (CLIP ViT-L), 1024 = SD2.x (OpenCLIP ViT-H), 2048 = SDXL.

Reading shapes is enough to identify the model, so this works on a header map
without loading any weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

UNET_PREFIX = "model.diffusion_model."
_INPUT_CONV = UNET_PREFIX + "input_blocks.0.0.weight"
_ATTN2_TO_K = re.compile(r"transformer_blocks\.\d+\.attn2\.to_k\.weight$")

Shape = Sequence[int]


@dataclass
class ModelSpec:
    """Everything the engine needs to know about a checkpoint before building it."""

    architecture: str          # e.g. "sd15"
    prediction: str            # "eps" | "v"
    latent_channels: int       # VAE latent channel count
    context_dim: int           # text-encoder hidden size seen by cross-attention
    image_size: int = 512      # native training resolution
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"


def _context_dim(shapes: Mapping[str, Shape]) -> int | None:
    for key, shape in shapes.items():
        if key.startswith(UNET_PREFIX) and _ATTN2_TO_K.search(key):
            return int(shape[1])  # to_k.weight is [inner_dim, context_dim]
    return None


def detect_architecture(shapes: Mapping[str, Shape]) -> ModelSpec:
    """Infer a :class:`ModelSpec` from a ``{key: shape}`` mapping.

    Raises ``ValueError`` if it isn't a recognizable diffusion checkpoint and
    ``NotImplementedError`` for a recognized-but-unsupported family.
    """
    if _INPUT_CONV not in shapes:
        raise ValueError(f"no UNet found (missing {_INPUT_CONV!r}); not a supported checkpoint")

    context_dim = _context_dim(shapes)
    if context_dim is None:
        raise ValueError("could not determine text context dim (no attn2.to_k weight found)")

    if context_dim == 768:
        return ModelSpec(
            architecture="sd15",
            prediction="eps",
            latent_channels=4,
            context_dim=768,
            image_size=512,
        )

    raise NotImplementedError(
        f"recognized a diffusion checkpoint with context_dim={context_dim}, but only "
        "SD1.5 (context_dim=768) is implemented so far"
    )
