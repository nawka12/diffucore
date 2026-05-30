"""Diffucore: a clean, from-scratch diffusion inference engine.

End-to-end text-to-image, image-to-image, and inpainting for Stable Diffusion
1.5 and SDXL (epsilon / v-prediction, optional zero-terminal-SNR), plus Anima
(the first DiT family) via flow matching. ``load_checkpoint`` /
``load_anima_checkpoint`` build a ``ModelBundle``; ``TextToImage`` /
``ImageToImage`` / ``Inpaint`` drive generation; ``apply_lora`` fuses
LoRA / LoKr adapters. See ``docs/ARCHITECTURE.md`` and ``docs/ROADMAP.md``.
"""

__version__ = "0.1.0"

from .bundle import ModelBundle, load_anima_checkpoint, load_checkpoint
from .lora import LoraReport, apply_lora, clear_loras, remove_lora
from .pipelines import ImageToImage, Inpaint, TextToImage

__all__ = [
    "__version__", "ModelBundle",
    "load_checkpoint", "load_anima_checkpoint",
    "apply_lora", "remove_lora", "clear_loras", "LoraReport",
    "ImageToImage", "Inpaint", "TextToImage",
]
