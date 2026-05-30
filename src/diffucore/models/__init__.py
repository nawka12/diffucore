"""Neural-network backbones (skeletons).

These are interface placeholders for milestones M4–M6. Each ``forward`` raises
``NotImplementedError``; implement the bodies on a CUDA box and verify with real
SD1.5 weights, following ``docs/IMPLEMENTATION_SPEC.md``.
"""

from .clip_text import CLIPTextConfig, CLIPTextEncoder
from .open_clip_text import OpenCLIPTextConfig, OpenCLIPTextEncoder
from .vae import VAEConfig, AutoencoderKL
from .unet import UNetConfig, UNetModel
from .qwen_image_vae import QwenImageVAE
from .qwen3_text import Qwen3Config, Qwen3TextEncoder
from .llm_adapter import LLMAdapter, LLMAdapterConfig
from .anima_dit import AnimaDiT, CosmosDiT, CosmosDiTConfig

__all__ = [
    "CLIPTextConfig",
    "CLIPTextEncoder",
    "OpenCLIPTextConfig",
    "OpenCLIPTextEncoder",
    "VAEConfig",
    "AutoencoderKL",
    "UNetConfig",
    "UNetModel",
    "QwenImageVAE",
    "Qwen3Config",
    "Qwen3TextEncoder",
    "LLMAdapter",
    "LLMAdapterConfig",
    "AnimaDiT",
    "CosmosDiT",
    "CosmosDiTConfig",
]
