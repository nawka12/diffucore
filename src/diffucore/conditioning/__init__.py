"""Text conditioning: tokenizer + text encoder -> cross-attention embeddings. [SKELETON]

Implement per ``docs/IMPLEMENTATION_SPEC.md`` §Conditioning.
"""

from __future__ import annotations

import torch


class CLIPTokenizer:
    """CLIP BPE tokenizer. Wraps the standard CLIP vocab/merges.

    Contract: ``encode(text) -> LongTensor[77]`` with BOS(49406) … EOS(49407)
    padded to 77 with EOS.
    """

    def __init__(self, vocab_path: str | None = None):
        self.vocab_path = vocab_path

    def encode(self, text: str) -> torch.Tensor:
        raise NotImplementedError("CLIPTokenizer.encode — see IMPLEMENTATION_SPEC.md §Conditioning")


class Conditioner:
    """Turns prompt strings into UNet cross-attention context.

    Contract: ``__call__(prompt: str, batch: int = 1) -> FloatTensor[batch, 77, 768]``.
    """

    def __init__(self, tokenizer: "CLIPTokenizer", text_encoder, clip_skip: int = 1):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.clip_skip = clip_skip

    def __call__(self, prompt: str, batch: int = 1) -> torch.Tensor:
        raise NotImplementedError("Conditioner.__call__ — see IMPLEMENTATION_SPEC.md §Conditioning")


__all__ = ["CLIPTokenizer", "Conditioner"]
