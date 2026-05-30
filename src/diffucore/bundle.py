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
from .runtime import DevicePolicy
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
    # position_ids is a derived constant (non-persistent buffer); drop it if a
    # checkpoint ships one so the strict load neither misses nor rejects it.
    sub = {k: v for k, v in sub.items() if not k.endswith("position_ids")}
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
    policy: DevicePolicy = None      # placement authority; None -> all-resident


def load_checkpoint(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    policy: DevicePolicy | None = None,
) -> ModelBundle:
    """Detect, build, and weight-load a checkpoint into a :class:`ModelBundle`.

    Supports SD1.5 and SDXL. Text encoder(s) and UNet run in ``dtype`` (fp16 on
    CUDA); the VAE stays fp32 (fp16 decode produces artifacts/NaNs).

    ``policy`` is the single placement authority. When omitted, one is built from
    ``device``/``dtype`` with offload off (current all-resident behavior). When
    ``policy.offload`` is set, modules are left on CPU; the pipeline shuttles each
    onto the GPU around its stage. The sigma schedule always lives on the compute
    device — it is never offloaded.
    """
    if policy is None:
        policy = DevicePolicy(device=torch.device(device), compute_dtype=dtype)

    spec = detect_architecture(read_header(path))
    if spec.architecture not in ("sd15", "sdxl"):
        raise NotImplementedError(f"unsupported architecture {spec.architecture!r}")

    # The training schedule is fully determined by the spec; keep its sigma table
    # (fp32) on the compute device so sigma<->t stays with the latents.
    schedule = DiscreteSchedule(
        make_betas(spec.beta_schedule, spec.num_train_timesteps),
        zero_terminal_snr=spec.zero_terminal_snr,
    )
    schedule.sigmas = schedule.sigmas.to(policy.device)
    schedule.log_sigmas = schedule.log_sigmas.to(policy.device)

    state_dict = load_state_dict(path, device="cpu")

    # With offload, modules wait on CPU and the pipeline moves them per stage. The
    # "idle" group (text encoders + VAE) offloads in every mode; the UNet only in
    # full offload ("encoders" mode keeps it resident — see RUNTIME_SPEC.md R4).
    idle_target = policy.offload_device if policy.offload_idle else policy.device
    unet_target = policy.offload_device if policy.offload_unet else policy.device

    vae = _load_sub(AutoencoderKL(VAEConfig(scale_factor=spec.latent_scale)), state_dict, _VAE_PREFIX)
    vae = vae.to(idle_target, policy.vae_dtype).eval()

    text_encoder_2 = None
    if spec.architecture == "sd15":
        text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _SD15_CLIP)
        backbone = _load_sub(UNetModel(), state_dict, _UNET_PREFIX)
    else:  # sdxl
        text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _SDXL_CLIP_L)
        text_encoder_2 = _load_sub(OpenCLIPTextEncoder(), state_dict, _SDXL_CLIP_G)
        text_encoder_2 = text_encoder_2.to(idle_target, policy.compute_dtype).eval()
        backbone = _load_sub(UNetModel(sdxl_unet_config()), state_dict, _UNET_PREFIX)

    text_encoder = text_encoder.to(idle_target, policy.compute_dtype).eval()
    backbone = backbone.to(unet_target, policy.compute_dtype).eval()

    return ModelBundle(
        spec=spec,
        schedule=schedule,
        tokenizer=CLIPTokenizer(),
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
        text_encoder_2=text_encoder_2,
        policy=policy,
    )


__all__ = ["ModelBundle", "load_checkpoint"]
