"""Anima's dual tokenizer: Qwen2.5 (semantic) + T5 (positional).

Anima conditions on its prompt twice. The Qwen3 0.6B encoder consumes a
Qwen2.5-tokenized stream and produces source hidden states; the LLM-Adapter
separately consumes a T5-tokenized stream as its target IDs (which it embeds
through its own 32128-row table) and cross-attends to the Qwen3 hidden states
to produce the DiT's conditioning context.

Both tokenizers are vendored as ``tokenizer.json`` and driven through the
``tokenizers`` library, exactly like the CLIP tokenizer under
``conditioning/clip_tokenizer.json``. The Qwen vocab is Qwen3-0.6B's (Apache-2.0,
the Qwen2 BPE shared across Qwen2.5/Qwen3); the T5 vocab is ``google-t5/t5-11b``'s
(Apache-2.0), the same T5 Anima inherits from Cosmos-Predict2. Both vendored files
are bit-identical to the ComfyUI ``qwen25_tokenizer/`` + ``t5_tokenizer/`` they
replace (see ``tests/test_anima_pipeline.py``).
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
# Qwen3.5's BPE (vocab 248320) for the experimental Qwen3.5-4B encoder — a
# different vocab from Qwen3 (151936), so the IDs are not interchangeable.
_QWEN35_VOCAB = Path(__file__).with_name("qwen35_tokenizer.json")
_T5_VOCAB = Path(__file__).with_name("t5_tokenizer.json")


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

    Loads the two vendored ``tokenizer.json`` files on first call — keeps
    construction cheap. ``qwen_path`` / ``t5_path`` override the vendored
    defaults (mirrors :class:`CLIPTokenizer`'s ``vocab_path``).
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
