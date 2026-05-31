"""Neural-network backbones, implemented from their original publications.

Text encoders (CLIP ViT-L/14, OpenCLIP bigG, Qwen3), the SD1.5/SDXL UNet, the
AutoencoderKL and Qwen-Image VAEs, and the Anima DiT (+ its LLM-Adapter). Each
is strict-loadable from real checkpoints and verified on CUDA against HF
oracles — see ``docs/ROADMAP.md``.
"""

from ._norm import RMSNorm
from .clip_text import CLIPTextConfig, CLIPTextEncoder
from .open_clip_text import OpenCLIPTextConfig, OpenCLIPTextEncoder
from .vae import VAEConfig, AutoencoderKL
from .unet import UNetConfig, UNetModel
from .qwen_image_vae import QwenImageVAE
from .qwen3_text import Qwen3Config, Qwen3TextEncoder
from .llm_adapter import LLMAdapter, LLMAdapterConfig
from .anima_dit import AnimaDiT, CosmosDiT, CosmosDiTConfig

__all__ = [
    "RMSNorm",
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
