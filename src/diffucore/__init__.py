"""Diffucore: a clean, from-scratch diffusion inference engine.

The sampling foundation (``diffucore.sampling``) and checkpoint detection
(``diffucore.loading``) are implemented and tested. The model backbones behind
``load_checkpoint`` / ``TextToImage`` are skeletons pending milestones M4–M6 —
see ``docs/IMPLEMENTATION_SPEC.md`` and ``docs/HANDOFF.md``.
"""

__version__ = "0.0.1"

from .bundle import ModelBundle, load_anima_checkpoint, load_checkpoint
from .pipelines import ImageToImage, Inpaint, TextToImage

__all__ = [
    "__version__", "ModelBundle",
    "load_checkpoint", "load_anima_checkpoint",
    "ImageToImage", "Inpaint", "TextToImage",
]
