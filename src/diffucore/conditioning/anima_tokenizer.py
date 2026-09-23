"""Anima's dual tokenizer: Qwen2.5 BPE for the Qwen3 encoder, T5 for the
LLM-Adapter's target ids. Both vocabs are vendored (Apache-2.0) and
bit-identical to ComfyUI's ``qwen25_tokenizer/`` + ``t5_tokenizer/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

# Anima's pad token in Qwen2.5 vocab (``<|endoftext|>``).
QWEN_PAD_ID = 151643

_QWEN_VOCAB = Path(__file__).with_name("qwen3_tokenizer.json")
# Qwen3.5's BPE (vocab 248320) for the experimental Qwen3.5 encoders; not
# interchangeable with Qwen3's (151936).
_QWEN35_VOCAB = Path(__file__).with_name("qwen35_tokenizer.json")
_T5_VOCAB = Path(__file__).with_name("t5_tokenizer.json")


@dataclass
class AnimaTokenized:
    """Token IDs for one prompt across both encoders.

    Shapes (single-prompt forward; the pipeline batches by replication when
    needed)::

        qwen_ids:   LongTensor (1, L_q)  Qwen2.5 BPE
        qwen_mask:  LongTensor (1, L_q)  1 = real token, 0 = pad
        t5_ids:     LongTensor (1, L_t)  T5 BPE
    """
    qwen_ids: torch.Tensor
    qwen_mask: torch.Tensor
    t5_ids: torch.Tensor


class AnimaTokenizer:
    """Lazy dual tokenizer for Anima (vocabs load on first call).
    ``qwen_path`` / ``t5_path`` override the vendored files.
    """

    def __init__(self, qwen_path: Optional[str] = None, t5_path: Optional[str] = None):
        self.qwen_path = str(qwen_path or _QWEN_VOCAB)
        self.t5_path = str(t5_path or _T5_VOCAB)
        self._qwen = None
        self._t5 = None

    @classmethod
    def qwen35(cls, t5_path: Optional[str] = None) -> "AnimaTokenizer":
        """Variant that drives the Qwen3.5 BPE (vocab 248320) for the semantic
        stream; the T5 target stream is unchanged. Used when the Anima checkpoint
        ships the experimental Qwen3.5-4B encoder instead of Qwen3-0.6B."""
        return cls(qwen_path=str(_QWEN35_VOCAB), t5_path=t5_path)

    def _ensure_loaded(self, max_length: int):
        if self._qwen is None or self._t5 is None:
            self._qwen = Tokenizer.from_file(self.qwen_path)
            self._t5 = Tokenizer.from_file(self.t5_path)
        self._qwen.enable_truncation(max_length)
        self._t5.enable_truncation(max_length)

    def __call__(self, prompt: str, max_length: int = 512) -> AnimaTokenized:
        self._ensure_loaded(max_length)
        q = self._qwen.encode(prompt)
        t = self._t5.encode(prompt)
        return AnimaTokenized(
            qwen_ids=torch.tensor([q.ids], dtype=torch.long),
            qwen_mask=torch.tensor([q.attention_mask], dtype=torch.long),
            t5_ids=torch.tensor([t.ids], dtype=torch.long),
        )
