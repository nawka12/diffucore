"""Krea 2 prompt tokenization: Qwen2 BPE (the vendored Qwen3 vocab, which
Qwen3-VL shares) in Krea's Qwen-Image chat template.

The reference right-pads ``prefix + prompt`` to a fixed length, appends the
assistant suffix after the padding and gives padding no position; with the
padding left out, positions run 0..n-1, which is the same thing. The 34 prefix
tokens are dropped from the encoder output, leaving at most 512 text tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

_QWEN_VOCAB = Path(__file__).with_name("qwen3_tokenizer.json")

PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
          "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34
MAX_LENGTH = 512
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


@dataclass
class Krea2Tokenized:
    ids: torch.Tensor          # (1, n) prefix + prompt + suffix
    truncated: bool            # the prompt lost tokens to the 512 limit


class Krea2Tokenizer:
    def __init__(self, path: Optional[str] = None):
        self._tok = Tokenizer.from_file(str(path or _QWEN_VOCAB))
        self._suffix = self._tok.encode(SUFFIX).ids

    def __call__(self, prompt: str) -> Krea2Tokenized:
        ids = self._tok.encode(PREFIX + prompt).ids
        keep = MAX_LENGTH + PREFIX_TOKENS - len(self._suffix)
        return Krea2Tokenized(torch.tensor([ids[:keep] + self._suffix], dtype=torch.long), len(ids) > keep)
