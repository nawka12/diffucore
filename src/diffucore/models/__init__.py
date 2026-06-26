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
from .qwen35_text import Qwen35Config, Qwen35TextEncoder
from .llm_adapter import LLMAdapter, LLMAdapterConfig
from .anima_dit import AnimaDiT, CosmosDiT, CosmosDiTConfig
from .t5_text import T5Config, T5TextEncoder
from .flux_dit import Flux, FluxConfig
from .mistral_text import MistralConfig, MistralTextEncoder

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
    "Qwen35Config",
    "Qwen35TextEncoder",
    "LLMAdapter",
    "LLMAdapterConfig",
    "AnimaDiT",
    "CosmosDiT",
    "CosmosDiTConfig",
    "T5Config",
    "T5TextEncoder",
    "Flux",
    "FluxConfig",
    "MistralConfig",
    "MistralTextEncoder",
]
