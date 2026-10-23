"""MOSS-Image prompt tokenization: the Qwen3.5 BPE (the vendored 248320-token
vocab, which has the chat special tokens as single ids) in the reference's
chat template.

The reference renders ``apply_chat_template([user prompt],
add_generation_prompt=True, enable_thinking=True)``: no system turn, the
content trimmed, and the assistant turn opened with ``<think>\\n``. It
tokenizes one prompt per forward, unpadded, truncated to 4096 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Regex, Tokenizer, pre_tokenizers

_QWEN35_VOCAB = Path(__file__).with_name("qwen35_tokenizer.json")
# Qwen3.5-9B's tokenizer.json splits on ``\p{L}+``; the vendored file (same
# vocab and merges) has ``[\p{L}\p{M}]+``, which keeps combining marks (Thai,
# Devanagari) inside words and changes their ids.
_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
          r"|\s*[\r\n]+|\s+(?!\S)|\s+")

MAX_LENGTH = 4096


def render_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"


@dataclass
class MossTokenized:
    ids: torch.Tensor          # (1, n) the rendered template, n <= 4096
    truncated: bool            # the template lost its tail to the 4096 limit


class MossTokenizer:
    def __init__(self, path: Optional[str] = None):
        self._tok = Tokenizer.from_file(str(path or _QWEN35_VOCAB))
        if path is None:
            self._tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(_SPLIT), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])

    def __call__(self, prompt: str) -> MossTokenized:
        ids = self._tok.encode(render_prompt(prompt)).ids
        return MossTokenized(torch.tensor([ids[:MAX_LENGTH]], dtype=torch.long), len(ids) > MAX_LENGTH)
