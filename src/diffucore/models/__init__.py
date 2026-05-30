"""Neural-network backbones (skeletons).

These are interface placeholders for milestones M4–M6. Each ``forward`` raises
``NotImplementedError``; implement the bodies on a CUDA box and verify with real
SD1.5 weights, following ``docs/IMPLEMENTATION_SPEC.md``.
"""

from .clip_text import CLIPTextConfig, CLIPTextEncoder
from .vae import VAEConfig, AutoencoderKL
from .unet import UNetConfig, UNetModel

__all__ = [
    "CLIPTextConfig",
    "CLIPTextEncoder",
    "VAEConfig",
    "AutoencoderKL",
    "UNetConfig",
    "UNetModel",
]
