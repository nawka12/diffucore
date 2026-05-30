"""Text-to-image pipeline. [SKELETON]

Orchestrates the verified sampling core (already implemented) with the model
components (M4–M6). The denoising loop, schedules, CFG, and parameterizations it
relies on are done and tested; only the model forwards remain. See
``docs/IMPLEMENTATION_SPEC.md`` §Pipeline for the exact wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoid importing the bundle (and torch-heavy models) eagerly
    from ..bundle import ModelBundle


class TextToImage:
    def __init__(self, model: "ModelBundle"):
        self.model = model

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        steps: int = 20,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        sampler: str = "euler",
        scheduler: str = "karras",
        seed: int | None = None,
    ):
        """Return a ``PIL.Image``.

        Reference wiring (see spec): build a ``CFGDenoiser`` from
        ``model.backbone`` + ``EpsScaling`` + ``model.schedule`` with the
        conditioner's cond/uncond context; build the sigma schedule between
        ``model.schedule.sigma_min/max``; seed an initial latent
        ``randn * sigmas[0]``; run the chosen sampler; VAE-decode; to PIL.
        """
        raise NotImplementedError("TextToImage.__call__ — see docs/IMPLEMENTATION_SPEC.md §Pipeline")
