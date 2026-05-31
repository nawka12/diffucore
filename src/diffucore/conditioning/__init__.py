"""Text conditioning: tokenizer + text encoder -> cross-attention embeddings.

Implements the SD1.5 conditioning path per ``docs/IMPLEMENTATION_SPEC.md``
§Conditioning. The CLIP BPE vocab/merges are vendored as ``clip_tokenizer.json``
(OpenAI CLIP, MIT-licensed) and driven through the ``tokenizers`` library.
"""

from __future__ import annotations

import re
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

    def encode_raw(self, text: str) -> list[int]:
        """Token ids with no BOS/EOS and no padding — the bare content tokens.
        Used by the LPW path, which adds its own per-chunk BOS/EOS and padding."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids


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


# --- LPW (long prompt weighting) -------------------------------------------
# AUTOMATIC1111-style prompt attention: `(word)` -> 1.1, `[word]` -> 1/1.1,
# `(word:1.3)` -> explicit weight, nested brackets multiply, `\(` escapes a
# literal paren. Long prompts are split into 75-token chunks (each wrapped with
# BOS/EOS into a 77-token window) and encoded separately, lifting CLIP's 77-token
# cap. Token weights scale the per-token embeddings with A1111's mean-preserving
# renorm (multiply, then rescale so the chunk's mean magnitude is unchanged).

_RE_ATTENTION = re.compile(
    r"""
    \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|
    :\s*([+-]?[.\d]+)\s*\)|
    \)|]|
    [^\\()\[\]:]+|
    :
    """,
    re.X,
)

_ROUND_MULT = 1.1
_SQUARE_MULT = 1.0 / 1.1


def parse_prompt_attention(text: str) -> list[list]:
    """Parse A1111 attention syntax into ``[[chunk_text, weight], ...]``."""
    res: list[list] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []

    def multiply_range(start: int, mult: float):
        for i in range(start, len(res)):
            res[i][1] *= mult

    for m in _RE_ATTENTION.finditer(text):
        token, weight = m.group(0), m.group(1)
        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), _ROUND_MULT)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), _SQUARE_MULT)
        else:
            res.append([token, 1.0])

    for pos in round_brackets:
        multiply_range(pos, _ROUND_MULT)
    for pos in square_brackets:
        multiply_range(pos, _SQUARE_MULT)

    if not res:
        return [["", 1.0]]

    # Merge adjacent runs that share a weight so re-tokenization sees whole words.
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def _tokenize_with_weights(tokenizer: "CLIPTokenizer", prompt: str):
    """Flatten the parsed prompt into parallel ``(token_ids, per_token_weights)``."""
    tokens: list[int] = []
    weights: list[float] = []
    for text, weight in parse_prompt_attention(prompt):
        ids = tokenizer.encode_raw(text)
        tokens.extend(ids)
        weights.extend([weight] * len(ids))
    return tokens, weights


def _chunk(tokens: list[int], weights: list[float], pad_token: int):
    """Group into 75-token windows, each wrapped BOS…EOS and padded to 77.
    Always yields at least one (empty) chunk so an empty prompt still encodes."""
    chunk_len = MAX_LENGTH - 2
    chunk_ids: list[list[int]] = []
    chunk_weights: list[list[float]] = []
    for i in range(0, max(len(tokens), 1), chunk_len):
        grp = tokens[i : i + chunk_len]
        wgrp = weights[i : i + chunk_len]
        ids = [BOS_TOKEN] + grp + [EOS_TOKEN]
        w = [1.0] + wgrp + [1.0]
        pad = MAX_LENGTH - len(ids)
        chunk_ids.append(ids + [pad_token] * pad)
        chunk_weights.append(w + [1.0] * pad)
    return chunk_ids, chunk_weights


def _apply_weights(ctx: torch.Tensor, weights: list[float]) -> torch.Tensor:
    """A1111 mean-preserving weighting on ``[1, 77, dim]`` chunk embeddings."""
    w = torch.tensor(weights, device=ctx.device, dtype=ctx.dtype)
    orig_mean = ctx.mean()
    ctx = ctx * w[None, :, None]
    return ctx * (orig_mean / ctx.mean())


class SDXLConditioner:
    """SDXL dual text conditioning with long-prompt weighting (LPW).

    Runs CLIP-L and OpenCLIP bigG on the prompt (both at ``clip_skip=2`` =
    penultimate hidden, no final norm), concatenates their hidden states into the
    2048-d cross-attention context, and returns bigG's pooled embedding (1280-d)
    for the UNet's added conditioning.

    The prompt is parsed for A1111 attention syntax and split into 75-token
    chunks; each chunk is encoded as its own 77-token window and the chunk
    contexts are concatenated along the sequence axis, so prompts longer than 77
    tokens are supported. Per-token weights scale the context with A1111's
    mean-preserving renorm. The pooled embedding is taken from the final chunk.
    A short prompt with no weighting collapses to a single chunk and reproduces
    the plain (non-LPW) encoding exactly.

    The two encoders pad differently: CLIP-L with EOS (49407), bigG with 0.

    Contract: ``__call__(prompt, batch=1) -> (context[batch, 77*n, 2048],
    pooled[batch, 1280])``.
    """

    def __init__(self, tokenizer: "CLIPTokenizer", clip_l, clip_g, clip_skip: int = 2):
        self.tokenizer = tokenizer
        self.clip_l = clip_l
        self.clip_g = clip_g
        self.clip_skip = clip_skip

    def __call__(self, prompt: str, batch: int = 1):
        device = next(self.clip_g.parameters()).device
        tokens, weights = _tokenize_with_weights(self.tokenizer, prompt)
        chunks_l, chunk_weights = _chunk(tokens, weights, pad_token=EOS_TOKEN)
        chunks_g, _ = _chunk(tokens, weights, pad_token=0)

        contexts = []
        pooled = None
        for ids_l, ids_g, w in zip(chunks_l, chunks_g, chunk_weights):
            t_l = torch.tensor([ids_l], dtype=torch.long, device=device)
            t_g = torch.tensor([ids_g], dtype=torch.long, device=device)
            hidden_l = self.clip_l(t_l, clip_skip=self.clip_skip)          # [1, 77, 768]
            hidden_g, pooled = self.clip_g(t_g, clip_skip=self.clip_skip)  # [1, 77, 1280], [1, 1280]
            ctx = torch.cat([hidden_l, hidden_g], dim=-1)                  # [1, 77, 2048]
            contexts.append(_apply_weights(ctx, w))

        context = torch.cat(contexts, dim=1)                               # [1, 77*n, 2048]
        return context.expand(batch, -1, -1), pooled.expand(batch, -1)


from .anima_tokenizer import AnimaTokenized, AnimaTokenizer

__all__ = [
    "CLIPTokenizer", "Conditioner", "SDXLConditioner", "parse_prompt_attention",
    "AnimaTokenizer", "AnimaTokenized",
]
