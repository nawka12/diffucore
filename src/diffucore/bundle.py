"""Loading a checkpoint into a ready-to-run :class:`ModelBundle`.

Architecture detection (implemented, M3) runs here; building the modules and
loading their weights is the M4–M6 work described in
``docs/IMPLEMENTATION_SPEC.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .loading import ModelSpec, detect_architecture, read_header
from .sampling import DiscreteSchedule, make_betas


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

    # The training schedule is fully determined by the spec and is ready to use:
    schedule = DiscreteSchedule(make_betas(spec.beta_schedule, spec.num_train_timesteps))

    # TODO(M4–M6): build CLIPTextEncoder / UNetModel / AutoencoderKL, remap the
    # checkpoint keys (see spec §"Checkpoint key map"), load weights onto `device`
    # in `dtype`, and populate the bundle. Until then:
    raise NotImplementedError(
        "module construction pending; implement on CUDA per docs/IMPLEMENTATION_SPEC.md "
        f"(detection succeeded: {spec})"
    )


__all__ = ["ModelBundle", "load_checkpoint"]
