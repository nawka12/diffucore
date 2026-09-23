"""Detect a checkpoint's architecture from its tensor keys and shapes (header
only, no weights).

SD checkpoints use the LDM layout; the UNet cross-attention context width
(``attn2.to_k``) tells the families apart: 768 = SD1.x, 1024 = SD2.x,
2048 = SDXL. Anima (Cosmos-Predict2 + Qwen3 + Qwen-Image VAE) is found by its
LLM-adapter fingerprint, FLUX by its double-stream blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

UNET_PREFIX = "model.diffusion_model."
_INPUT_CONV = UNET_PREFIX + "input_blocks.0.0.weight"
_ATTN2_TO_K = re.compile(r"transformer_blocks\.\d+\.attn2\.to_k\.weight$")
# Anima's keys are bare (``net.*``) natively but prefixed with
# ``model.diffusion_model.`` in an all-in-one file; match on the suffix.
_ANIMA_FINGERPRINT = "llm_adapter.blocks.0.cross_attn.q_proj.weight"
# FLUX double-stream blocks: bare in a BFL file, prefixed all-in-one.
_FLUX_FINGERPRINT = "double_blocks.0.img_attn.qkv.weight"

Shape = Sequence[int]


@dataclass
class ModelSpec:
    """Everything the engine needs to know about a checkpoint before building it."""

    architecture: str          # e.g. "sd15", "sdxl", "anima", "flux1", "flux2"
    prediction: str            # "eps" | "v" | "flow"
    zero_terminal_snr: bool    # rescale the schedule to zero terminal SNR (ZTSNR)
    latent_channels: int       # VAE latent channel count
    context_dim: int           # text-encoder hidden size seen by cross-attention
    image_size: int = 512      # native training resolution
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    latent_scale: float = 0.18215   # VAE latent scale factor (SDXL uses 0.13025)
    latent_shift: float = 0.0       # VAE pre-scale shift (FLUX uses 0.1159; 0 for SD)
    guidance_distilled: bool = False  # FLUX-dev style baked-in guidance embedding


def _context_dim(shapes: Mapping[str, Shape]) -> int | None:
    for key, shape in shapes.items():
        if key.startswith(UNET_PREFIX) and _ATTN2_TO_K.search(key):
            return int(shape[1])  # to_k.weight is [inner_dim, context_dim]
    return None


def _anima_prefix(shapes: Mapping[str, Shape]) -> str | None:
    """The on-disk prefix in front of Anima's DiT keys (``"net."`` for a native
    export, ``"model.diffusion_model."`` inside an all-in-one ComfyUI checkpoint).
    Recovered from the LLM-adapter fingerprint."""
    for key in shapes:
        if key.endswith(_ANIMA_FINGERPRINT):
            return key[: -len(_ANIMA_FINGERPRINT)]
    return None


def _flux_prefix(shapes: Mapping[str, Shape]) -> str | None:
    """The on-disk prefix in front of the FLUX transformer keys (``""`` for a bare
    transformer, ``"model.diffusion_model."`` inside an all-in-one checkpoint)."""
    for key in shapes:
        if key.endswith(_FLUX_FINGERPRINT):
            return key[: -len(_FLUX_FINGERPRINT)]
    return None


def _detect_flux(shapes: Mapping[str, Shape], prefix: str) -> ModelSpec:
    """FLUX.1 vs FLUX.2: FLUX.2's shared ``double_stream_modulation_img.lin`` is
    the discriminator. FLUX.1 latents are 16-ch / 8× with a
    ``(0.3611, 0.1159)`` scale/shift; FLUX.2's are 128-ch / 16×, unnormalised.
    Other family constants come from ``bundle._flux_arch``.
    """
    txt_in = shapes.get(prefix + "txt_in.weight")
    img_in = shapes.get(prefix + "img_in.weight")
    context_dim = int(txt_in[1]) if txt_in is not None else 0
    guidance = (prefix + "guidance_in.in_layer.weight") in shapes
    is_flux2 = (prefix + "double_stream_modulation_img.lin.weight") in shapes

    if is_flux2:
        in_channels = int(img_in[1]) if img_in is not None else 128
        return ModelSpec(
            architecture="flux2",
            prediction="flow",
            zero_terminal_snr=False,
            latent_channels=in_channels,        # patch_size 1: token width == latent channels
            context_dim=context_dim,
            image_size=1024,
            latent_scale=1.0,                   # FLUX.2 latent: no scale/shift
            latent_shift=0.0,
            guidance_distilled=guidance,
        )

    in_channels = int(img_in[1]) if img_in is not None else 64
    return ModelSpec(
        architecture="flux1",
        prediction="flow",
        zero_terminal_snr=False,
        latent_channels=in_channels // 4,       # 2×2 patch fold
        context_dim=context_dim,
        image_size=1024,
        latent_scale=0.3611,
        latent_shift=0.1159,
        guidance_distilled=guidance,
    )


def detect_architecture(shapes: Mapping[str, Shape]) -> ModelSpec:
    """Infer a :class:`ModelSpec` from a ``{key: shape}`` mapping.

    Raises ``ValueError`` if it isn't a recognizable diffusion checkpoint and
    ``NotImplementedError`` for a recognized-but-unsupported family.
    """
    # Anima: the first LLM-adapter block's q_proj is enough to disambiguate.
    if _anima_prefix(shapes) is not None:
        return ModelSpec(
            architecture="anima",
            prediction="flow",
            zero_terminal_snr=False,
            latent_channels=16,
            context_dim=1024,
            image_size=1024,
            latent_scale=1.0,
        )

    # FLUX (BFL rectified-flow DiT): double-stream blocks, bare or under the
    # ``model.diffusion_model.`` prefix of an all-in-one checkpoint.
    flux_prefix = _flux_prefix(shapes)
    if flux_prefix is not None:
        return _detect_flux(shapes, flux_prefix)

    if _INPUT_CONV not in shapes:
        raise ValueError(f"no UNet found (missing {_INPUT_CONV!r}); not a supported checkpoint")

    context_dim = _context_dim(shapes)
    if context_dim is None:
        raise ValueError("could not determine text context dim (no attn2.to_k weight found)")

    # v-prediction checkpoints carry a bare ``v_pred`` marker tensor (NoobAI /
    # A1111 / reForge), often with ``ztsnr``; otherwise assume eps.
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
