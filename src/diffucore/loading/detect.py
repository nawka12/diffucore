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

    architecture: str          # e.g. "sd15", "sdxl"
    prediction: str            # "eps" | "v"
    zero_terminal_snr: bool    # rescale the schedule to zero terminal SNR (ZTSNR)
    latent_channels: int       # VAE latent channel count
    context_dim: int           # text-encoder hidden size seen by cross-attention
    image_size: int = 512      # native training resolution
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    latent_scale: float = 0.18215   # VAE latent scale factor (SDXL uses 0.13025)


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

    # v-prediction checkpoints flag themselves with a bare ``v_pred`` marker tensor
    # (the NoobAI / A1111 / reForge convention); a ``ztsnr`` marker often rides
    # along to request zero-terminal-SNR sampling. The weights are otherwise
    # identical to an eps model, so these flags are the only signal. Absent them,
    # assume epsilon (the SD default) and the standard schedule.
    prediction = "v" if "v_pred" in shapes else "eps"
    zero_terminal_snr = "ztsnr" in shapes

    if context_dim == 768:
        return ModelSpec(
            architecture="sd15",
            prediction=prediction,
            zero_terminal_snr=zero_terminal_snr,
            latent_channels=4,
            context_dim=768,
            image_size=512,
            latent_scale=0.18215,
        )

    if context_dim == 2048:
        return ModelSpec(
            architecture="sdxl",
            prediction=prediction,
            zero_terminal_snr=zero_terminal_snr,
            latent_channels=4,
            context_dim=2048,
            image_size=1024,
            latent_scale=0.13025,
        )

    raise NotImplementedError(
        f"recognized a diffusion checkpoint with context_dim={context_dim}, but only "
        "SD1.5 (768) and SDXL (2048) are implemented so far"
    )
