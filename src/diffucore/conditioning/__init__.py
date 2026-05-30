"""Text conditioning: tokenizer + text encoder -> cross-attention embeddings.

Implements the SD1.5 conditioning path per ``docs/IMPLEMENTATION_SPEC.md``
§Conditioning. The CLIP BPE vocab/merges are vendored as ``clip_tokenizer.json``
(OpenAI CLIP, MIT-licensed) and driven through the ``tokenizers`` library.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

BOS_TOKEN = 49406
EOS_TOKEN = 49407
MAX_LENGTH = 77

_DEFAULT_VOCAB = Path(__file__).with_name("clip_tokenizer.json")


class CLIPTokenizer:
    """CLIP BPE tokenizer.

    Contract: ``encode(text) -> LongTensor[77]`` with BOS(49406) … EOS(49407)
    padded to 77 with EOS. The vendored tokenizer's post-processor already wraps
    the sequence in BOS/EOS; we only truncate and pad to 77 here.
    """

    def __init__(self, vocab_path: str | None = None):
        self.vocab_path = str(vocab_path or _DEFAULT_VOCAB)
        self._tokenizer = Tokenizer.from_file(self.vocab_path)

    def encode(self, text: str) -> torch.Tensor:
        ids = self._tokenizer.encode(text).ids
        ids = ids[:MAX_LENGTH]
        if len(ids) < MAX_LENGTH:
            ids = ids + [EOS_TOKEN] * (MAX_LENGTH - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class Conditioner:
    """Turns prompt strings into UNet cross-attention context.

    Contract: ``__call__(prompt: str, batch: int = 1) -> FloatTensor[batch, 77, 768]``.
    """

    def __init__(self, tokenizer: "CLIPTokenizer", text_encoder, clip_skip: int = 1):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.clip_skip = clip_skip

    def __call__(self, prompt: str, batch: int = 1) -> torch.Tensor:
        param = next(self.text_encoder.parameters())
        ids = self.tokenizer.encode(prompt).unsqueeze(0).to(param.device)
        embeddings = self.text_encoder(ids, clip_skip=self.clip_skip)
        return embeddings.expand(batch, -1, -1)


__all__ = ["CLIPTokenizer", "Conditioner"]
