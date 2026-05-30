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

    def encode(self, text: str, pad_token: int = EOS_TOKEN) -> torch.Tensor:
        """Tokenize to ``LongTensor[77]``. ``pad_token`` is the fill for the unused
        tail: SD1.5/CLIP-L pad with EOS (49407); SDXL's OpenCLIP bigG pads with 0."""
        ids = self._tokenizer.encode(text).ids
        ids = ids[:MAX_LENGTH]
        if len(ids) < MAX_LENGTH:
            ids = ids + [pad_token] * (MAX_LENGTH - len(ids))
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


class SDXLConditioner:
    """SDXL dual text conditioning.

    Runs CLIP-L and OpenCLIP bigG on the prompt (both at ``clip_skip=2`` =
    penultimate hidden, no final norm), concatenates their hidden states into the
    2048-d cross-attention context, and returns bigG's pooled embedding (1280-d)
    for the UNet's added conditioning.

    The two encoders pad differently: CLIP-L with EOS (49407), bigG with 0.

    Contract: ``__call__(prompt, batch=1) -> (context[batch, 77, 2048],
    pooled[batch, 1280])``.
    """

    def __init__(self, tokenizer: "CLIPTokenizer", clip_l, clip_g, clip_skip: int = 2):
        self.tokenizer = tokenizer
        self.clip_l = clip_l
        self.clip_g = clip_g
        self.clip_skip = clip_skip

    def __call__(self, prompt: str, batch: int = 1):
        device = next(self.clip_g.parameters()).device
        ids_l = self.tokenizer.encode(prompt, pad_token=EOS_TOKEN).unsqueeze(0).to(device)
        ids_g = self.tokenizer.encode(prompt, pad_token=0).unsqueeze(0).to(device)

        hidden_l = self.clip_l(ids_l, clip_skip=self.clip_skip)        # [1, 77, 768]
        hidden_g, pooled = self.clip_g(ids_g, clip_skip=self.clip_skip)  # [1, 77, 1280], [1, 1280]

        context = torch.cat([hidden_l, hidden_g], dim=-1)              # [1, 77, 2048]
        return context.expand(batch, -1, -1), pooled.expand(batch, -1)


__all__ = ["CLIPTokenizer", "Conditioner", "SDXLConditioner"]
