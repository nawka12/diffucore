"""FLUX tokenizers.

FLUX.1 conditions on two encoders: CLIP-L (pooled vector, 77 tokens, EOS-padded)
and T5-XXL (sequence context, padded to at least 256). Both vocabs are already
vendored: ``clip_tokenizer.json`` (OpenAI CLIP, MIT) and ``t5_tokenizer.json``
(google-t5, Apache-2.0).

FLUX.2 conditions on a single decoder LM, and uses the **concatenation of three
intermediate layers** of that LM as the DiT context (no final norm). Two families:

* **Klein** — Qwen3-4B/8B, layers ``[9, 18, 27]``, with the Qwen chat template.
  The Qwen2.5 vocab is the already-vendored ``qwen3_tokenizer.json`` (pad 151643).
* **Dev** — Mistral-3 24B, layers ``[10, 20, 30]``, with a SYSTEM_PROMPT template.
  Mistral's Tekken vocab is not vendored — supply a ``tokenizer.json`` path.

The templates and layer indices mirror ComfyUI's FLUX.2 text-encoder definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

from . import CLIPTokenizer

_T5_VOCAB = Path(__file__).with_name("t5_tokenizer.json")
_QWEN_VOCAB = Path(__file__).with_name("qwen3_tokenizer.json")
_T5_PAD_ID = 0  # T5 pad token (<pad>)

# FLUX.2 chat templates + the LM layers whose hidden states are concatenated.
_QWEN_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_MISTRAL_TEMPLATE = (
    "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give "
    "structured responses focusing on object relationships, object\nattribution and "
    "actions without speculation.[/SYSTEM_PROMPT][INST]{}[/INST]"
)
_KLEIN_LAYERS = [9, 18, 27]
_MISTRAL_LAYERS = [10, 20, 30]


@dataclass
class FluxTokenized:
    """Token IDs for one prompt::

        clip_ids: LongTensor (1, 77)        — CLIP-L BPE, EOS-padded
        t5_ids:   LongTensor (1, >=256)     — T5 BPE, pad(0)-padded
    """
    clip_ids: torch.Tensor
    t5_ids: torch.Tensor


class FluxTokenizer:
    """FLUX.1 dual tokenizer (CLIP-L + T5-XXL)."""

    def __init__(self, clip_path: Optional[str] = None, t5_path: Optional[str] = None):
        self.clip = CLIPTokenizer(clip_path)
        self.t5_path = str(t5_path or _T5_VOCAB)
        self._t5: Optional[Tokenizer] = None

    def __call__(self, prompt: str, t5_min_length: int = 256) -> FluxTokenized:
        clip_ids = self.clip.encode(prompt).unsqueeze(0)  # [1, 77]
        if self._t5 is None:
            self._t5 = Tokenizer.from_file(self.t5_path)
        # FLUX pads the T5 stream to at least 256 (no truncation), matching ComfyUI.
        self._t5.no_truncation()
        self._t5.enable_padding(length=t5_min_length, pad_id=_T5_PAD_ID, pad_token="<pad>")
        t = self._t5.encode(prompt)
        return FluxTokenized(
            clip_ids=clip_ids,
            t5_ids=torch.tensor([t.ids], dtype=torch.long),
        )


class Flux2Tokenizer:
    """FLUX.2 LM tokenizer (Klein/Qwen3 or Dev/Mistral).

    ``kind`` is ``"qwen3_4b"`` / ``"qwen3_8b"`` (Klein) or ``"mistral3_24b"`` (Dev).
    Applies the family chat template and pads to ``min_length``. ``hidden_layers``
    lists the LM layers the pipeline concatenates into the DiT context.
    """

    def __init__(self, kind: str = "qwen3_4b", tokenizer_path: Optional[str] = None):
        self.kind = kind
        if kind in ("qwen3_4b", "qwen3_8b"):
            self.template = _QWEN_TEMPLATE
            self.hidden_layers = list(_KLEIN_LAYERS)
            self.pad_id = 151643
            self.min_length = 512
            self._path = str(tokenizer_path or _QWEN_VOCAB)
        elif kind == "mistral3_24b":
            self.template = _MISTRAL_TEMPLATE
            self.hidden_layers = list(_MISTRAL_LAYERS)
            self.pad_id = 11
            self.min_length = 1
            self._path = str(tokenizer_path) if tokenizer_path else None
        else:
            raise ValueError(f"unknown FLUX.2 text-encoder kind {kind!r}")
        self._tok: Optional[Tokenizer] = None

    def __call__(self, prompt: str) -> torch.Tensor:
        if self._tok is None:
            if not self._path or not Path(self._path).exists():
                raise FileNotFoundError(
                    f"FLUX.2 ({self.kind}) tokenizer not found at {self._path!r}; "
                    "Klein uses the vendored Qwen vocab, Dev needs Mistral's tokenizer.json"
                )
            self._tok = Tokenizer.from_file(self._path)
        text = self.template.format(prompt)
        self._tok.no_truncation()
        if self.min_length > 1:
            self._tok.enable_padding(length=self.min_length, pad_id=self.pad_id, pad_token="<pad>")
        else:
            self._tok.no_padding()
        return torch.tensor([self._tok.encode(text).ids], dtype=torch.long)
