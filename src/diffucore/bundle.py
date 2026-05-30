"""Loading a checkpoint into a ready-to-run :class:`ModelBundle`.

Architecture detection (implemented, M3) runs here; building the modules and
loading their weights is the M4–M6 work described in
``docs/IMPLEMENTATION_SPEC.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .conditioning import CLIPTokenizer
from .loading import ModelSpec, detect_architecture, load_state_dict, read_header
from .models import AutoencoderKL, CLIPTextEncoder, OpenCLIPTextEncoder, UNetModel, VAEConfig
from .models.unet import sdxl_unet_config
from .sampling import DiscreteSchedule, make_betas

# On-disk prefixes (minus the top-level architecture prefix). SDXL keeps CLIP-L
# under embedders.0 and adds OpenCLIP bigG under embedders.1.
_VAE_PREFIX = "first_stage_model."
_UNET_PREFIX = "model.diffusion_model."
_SD15_CLIP = "cond_stage_model.transformer."
_SDXL_CLIP_L = "conditioner.embedders.0.transformer."
_SDXL_CLIP_G = "conditioner.embedders.1.model."


def _load_sub(module, state_dict, prefix):
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.load_state_dict(sub, strict=True)
    return module


@dataclass
class ModelBundle:
    """A loaded model, ready for a pipeline."""

    spec: ModelSpec
    schedule: DiscreteSchedule
    tokenizer: object               # conditioning.CLIPTokenizer
    text_encoder: object            # CLIPTextEncoder (CLIP-L)
    backbone: object                # models.UNetModel
    vae: object                     # models.AutoencoderKL
    text_encoder_2: object = None   # SDXL only: OpenCLIPTextEncoder (bigG)


def load_checkpoint(path: str, device: str = "cpu", dtype: torch.dtype = torch.float16) -> ModelBundle:
    """Detect, build, and weight-load a checkpoint into a :class:`ModelBundle`.

    Supports SD1.5 and SDXL. Text encoder(s) and UNet run in ``dtype`` (fp16 on
    CUDA); the VAE stays fp32 (fp16 decode produces artifacts/NaNs).
    """
    spec = detect_architecture(read_header(path))
    if spec.architecture not in ("sd15", "sdxl"):
        raise NotImplementedError(f"unsupported architecture {spec.architecture!r}")

    # The training schedule is fully determined by the spec; keep its sigma table
    # (fp32) on `device` so sigma<->t stays on the same device as the latents.
    schedule = DiscreteSchedule(make_betas(spec.beta_schedule, spec.num_train_timesteps))
    schedule.sigmas = schedule.sigmas.to(device)
    schedule.log_sigmas = schedule.log_sigmas.to(device)

    state_dict = load_state_dict(path, device="cpu")

    vae = _load_sub(AutoencoderKL(VAEConfig(scale_factor=spec.latent_scale)), state_dict, _VAE_PREFIX)
    vae = vae.to(device, torch.float32).eval()

    text_encoder_2 = None
    if spec.architecture == "sd15":
        text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _SD15_CLIP)
        backbone = _load_sub(UNetModel(), state_dict, _UNET_PREFIX)
    else:  # sdxl
        text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _SDXL_CLIP_L)
        text_encoder_2 = _load_sub(OpenCLIPTextEncoder(), state_dict, _SDXL_CLIP_G)
        text_encoder_2 = text_encoder_2.to(device, dtype).eval()
        backbone = _load_sub(UNetModel(sdxl_unet_config()), state_dict, _UNET_PREFIX)

    text_encoder = text_encoder.to(device, dtype).eval()
    backbone = backbone.to(device, dtype).eval()

    return ModelBundle(
        spec=spec,
        schedule=schedule,
        tokenizer=CLIPTokenizer(),
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
        text_encoder_2=text_encoder_2,
    )


__all__ = ["ModelBundle", "load_checkpoint"]
