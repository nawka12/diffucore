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
from .models import AutoencoderKL, CLIPTextEncoder, UNetModel
from .sampling import DiscreteSchedule, make_betas

_PREFIXES = {
    "backbone": "model.diffusion_model.",
    "vae": "first_stage_model.",
    "text_encoder": "cond_stage_model.transformer.",
}


def _load_sub(module, state_dict, prefix):
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.load_state_dict(sub, strict=True)
    return module


@dataclass
class ModelBundle:
    """A loaded model, ready for a pipeline."""

    spec: ModelSpec
    schedule: DiscreteSchedule
    tokenizer: object       # conditioning.CLIPTokenizer
    text_encoder: object    # models.CLIPTextEncoder
    backbone: object        # models.UNetModel
    vae: object             # models.AutoencoderKL


def load_checkpoint(path: str, device: str = "cpu", dtype: torch.dtype = torch.float16) -> ModelBundle:
    """Detect, build, and weight-load a checkpoint.

    The detection step is implemented and tested; module construction + weight
    mapping is pending (M4–M6). See ``docs/IMPLEMENTATION_SPEC.md``.
    """
    spec = detect_architecture(read_header(path))
    if spec.architecture != "sd15":
        raise NotImplementedError(f"only SD1.5 is implemented; detected {spec.architecture!r}")

    # The training schedule is fully determined by the spec; keep its sigma table
    # (fp32) on `device` so sigma<->t stays on the same device as the latents.
    schedule = DiscreteSchedule(make_betas(spec.beta_schedule, spec.num_train_timesteps))
    schedule.sigmas = schedule.sigmas.to(device)
    schedule.log_sigmas = schedule.log_sigmas.to(device)

    state_dict = load_state_dict(path, device="cpu")

    text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _PREFIXES["text_encoder"])
    backbone = _load_sub(UNetModel(), state_dict, _PREFIXES["backbone"])
    vae = _load_sub(AutoencoderKL(), state_dict, _PREFIXES["vae"])

    # UNet and CLIP run in `dtype` (fp16 on the 2060); the VAE stays fp32 because
    # fp16 decode produces artifacts/NaNs on many SD1.5 weights.
    text_encoder = text_encoder.to(device, dtype).eval()
    backbone = backbone.to(device, dtype).eval()
    vae = vae.to(device, torch.float32).eval()

    return ModelBundle(
        spec=spec,
        schedule=schedule,
        tokenizer=CLIPTokenizer(),
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
    )


__all__ = ["ModelBundle", "load_checkpoint"]
