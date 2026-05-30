"""Anima's dual tokenizer: Qwen2.5 (semantic) + T5 (positional).

Anima conditions on its prompt twice. The Qwen3 0.6B encoder consumes a
Qwen2.5-tokenized stream and produces source hidden states; the LLM-Adapter
separately consumes a T5-tokenized stream as its target IDs (which it embeds
through its own 32128-row table) and cross-attends to the Qwen3 hidden states
to produce the DiT's conditioning context.

This module deliberately keeps a thin wrapping around the tokenizer files.
For DT7 we load them via ``transformers`` from a directory that already exists
on the user's disk (the ComfyUI install ships ``qwen25_tokenizer/`` and
``t5_tokenizer/`` under ``comfy/text_encoders/``). Proper ``tokenizer.json``
vendoring under ``conditioning/`` — to drop the runtime ``transformers``
dependency on the Anima path — is a follow-up; the existing CLIP tokenizer
under ``conditioning/clip_tokenizer.json`` is the template.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Anima's pad token in Qwen2.5 vocab (``<|endoftext|>``).
QWEN_PAD_ID = 151643


# Best-effort discovery so a user with ComfyUI in the default install location
# doesn't have to wire paths through; explicit args always win.
_COMFY_HINT = "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/comfy/text_encoders"


def _default_qwen_dir() -> str:
    return os.environ.get("DIFFUCORE_QWEN2_TOKENIZER_DIR", f"{_COMFY_HINT}/qwen25_tokenizer")


def _default_t5_dir() -> str:
    return os.environ.get("DIFFUCORE_T5_TOKENIZER_DIR", f"{_COMFY_HINT}/t5_tokenizer")


@dataclass
class AnimaTokenized:
    """Token IDs for one prompt across both encoders.

    Shapes (single-prompt forward; the pipeline batches by replication when
    needed)::

        qwen_ids:   LongTensor (1, L_q)  — Qwen2.5 BPE
        qwen_mask:  LongTensor (1, L_q)  — 1 = real token, 0 = pad
        t5_ids:     LongTensor (1, L_t)  — T5 BPE
    """
    qwen_ids: torch.Tensor
    qwen_mask: torch.Tensor
    t5_ids: torch.Tensor


class AnimaTokenizer:
    """Lazy dual tokenizer for Anima.

    Both directories must contain the files ``transformers`` expects for the
    respective tokenizer class (``Qwen2Tokenizer`` and ``T5TokenizerFast``):
    ``vocab.json`` + ``merges.txt`` + ``tokenizer_config.json`` for Qwen,
    ``tokenizer.json`` (+ optional config) for T5.

    No tokenizer is loaded until the first call — keeps construction cheap
    and lets us fall back to env-driven paths only when needed.
    """

    def __init__(self, qwen2_dir: Optional[str] = None, t5_dir: Optional[str] = None):
        self.qwen2_dir = qwen2_dir or _default_qwen_dir()
        self.t5_dir = t5_dir or _default_t5_dir()
        self._qwen = None
        self._t5 = None

    def _ensure_loaded(self):
        if self._qwen is None or self._t5 is None:
            try:
                from transformers import Qwen2Tokenizer, T5TokenizerFast
            except ImportError as e:
                raise ImportError(
                    "AnimaTokenizer currently uses transformers to load the Qwen2.5 "
                    "and T5 tokenizers. Install with `pip install transformers` or "
                    "wait for the planned vendored-tokenizer.json follow-up."
                ) from e
            for d in (self.qwen2_dir, self.t5_dir):
                if not Path(d).exists():
                    raise FileNotFoundError(
                        f"tokenizer directory not found: {d!r}. Set "
                        "DIFFUCORE_QWEN2_TOKENIZER_DIR / DIFFUCORE_T5_TOKENIZER_DIR "
                        "or pass paths to AnimaTokenizer(...)."
                    )
            self._qwen = Qwen2Tokenizer.from_pretrained(self.qwen2_dir)
            self._t5 = T5TokenizerFast.from_pretrained(self.t5_dir)

    def __call__(self, prompt: str, max_length: int = 512) -> AnimaTokenized:
        self._ensure_loaded()
        q = self._qwen(
            prompt, return_tensors="pt",
            padding=False, truncation=True, max_length=max_length,
        )
        t = self._t5(
            prompt, return_tensors="pt",
            padding=False, truncation=True, max_length=max_length,
        )
        return AnimaTokenized(
            qwen_ids=q["input_ids"].to(torch.long),
            qwen_mask=q["attention_mask"].to(torch.long),
            t5_ids=t["input_ids"].to(torch.long),
        )
