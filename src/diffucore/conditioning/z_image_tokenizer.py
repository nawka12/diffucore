"""Z-Image prompt tokenization: the Qwen3 BPE (the vendored Qwen3 vocab, byte
identical to Z-Image's own ``tokenizer.json``) in the reference's chat template.

The reference renders ``apply_chat_template([user prompt],
add_generation_prompt=True, enable_thinking=True)``: no system turn, the content
verbatim, and the assistant turn left open (no ``<think>`` block). It
tokenizes to a fixed 512, truncating the whole template from the right, and
keeps only the attended tokens; with padding left out, positions are the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

_QWEN_VOCAB = Path(__file__).with_name("qwen3_tokenizer.json")

MAX_LENGTH = 512


def render_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class ZImageTokenized:
    ids: torch.Tensor          # (1, n) the rendered template, n <= 512
    truncated: bool            # the template lost its tail to the 512 limit


class ZImageTokenizer:
    def __init__(self, path: Optional[str] = None):
        self._tok = Tokenizer.from_file(str(path or _QWEN_VOCAB))

    def __call__(self, prompt: str) -> ZImageTokenized:
        ids = self._tok.encode(render_prompt(prompt)).ids
        return ZImageTokenized(torch.tensor([ids[:MAX_LENGTH]], dtype=torch.long), len(ids) > MAX_LENGTH)
