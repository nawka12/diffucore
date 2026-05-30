"""LLM-Adapter (DT4) verification.

Without an in-process ComfyUI oracle (its native deps don't install in this
venv), this milestone verifies structurally and behaviorally:

1. Key-set match — module params match the ``net.llm_adapter.*`` slice
   of Anima's checkpoint exactly.
2. Strict load — real weights load cleanly via the project's ``_load_sub``
   prefix-stripping convention.
3. Forward shape on a realistic input.
4. Determinism — same inputs → same output, and a non-trivial change in
   inputs produces a non-trivial change in output (catches identity bugs
   like "ignored cross-attn context").

Numerical bit-match against the ComfyUI reference is deferred to DT7 (the
end-to-end image comparison is a stronger correctness signal anyway).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from diffucore.models import LLMAdapter

_ANIMA_CKPT = Path(os.environ.get(
    "DIFFUCORE_ANIMA_CKPT",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
))
_PREFIX = "net.llm_adapter."


@pytest.fixture(scope="module")
def loaded_adapter():
    if not _ANIMA_CKPT.exists():
        pytest.skip(f"anima checkpoint not at {_ANIMA_CKPT}")
    from safetensors.torch import load_file
    full = load_file(str(_ANIMA_CKPT))
    sub = {k[len(_PREFIX):]: v for k, v in full.items() if k.startswith(_PREFIX)}
    adapter = LLMAdapter()
    adapter.load_state_dict(sub, strict=True)
    return adapter.float().eval()


def test_llm_adapter_key_set_matches_checkpoint():
    from safetensors import safe_open
    if not _ANIMA_CKPT.exists():
        pytest.skip(f"anima checkpoint not at {_ANIMA_CKPT}")
    with safe_open(str(_ANIMA_CKPT), framework="pt") as f:
        ckpt_keys = {k[len(_PREFIX):] for k in f.keys() if k.startswith(_PREFIX)}
    mod_keys = set(LLMAdapter().state_dict().keys())
    assert mod_keys == ckpt_keys, (
        f"key mismatch — missing: {sorted(ckpt_keys - mod_keys)[:5]} | "
        f"extra: {sorted(mod_keys - ckpt_keys)[:5]}"
    )


def test_llm_adapter_forward_shape(loaded_adapter):
    """T5 tokens (B, L_t5) + Qwen3 hidden (B, L_qwen, 1024) -> (B, L_t5, 1024)."""
    torch.manual_seed(0)
    B, L_t5, L_qwen = 1, 11, 13
    src = torch.randn(B, L_qwen, 1024)
    tgt = torch.randint(0, 32128, (B, L_t5))
    with torch.no_grad():
        out = loaded_adapter(src, tgt)
    assert out.shape == (B, L_t5, 1024)


def test_llm_adapter_is_deterministic(loaded_adapter):
    """Same inputs → bit-identical outputs (no hidden randomness)."""
    torch.manual_seed(0)
    src = torch.randn(1, 7, 1024)
    tgt = torch.randint(0, 32128, (1, 5))
    with torch.no_grad():
        a = loaded_adapter(src, tgt)
        b = loaded_adapter(src, tgt)
    assert torch.equal(a, b)


def test_llm_adapter_attends_to_source(loaded_adapter):
    """Changing the source (Qwen3) hidden states changes the output. Catches a
    bug where cross-attn would silently ignore its context (e.g. a wrong
    routing or zeroed q_proj)."""
    torch.manual_seed(0)
    src1 = torch.randn(1, 9, 1024)
    src2 = torch.randn(1, 9, 1024)
    tgt = torch.randint(0, 32128, (1, 5))
    with torch.no_grad():
        a = loaded_adapter(src1, tgt)
        b = loaded_adapter(src2, tgt)
    # Output should differ substantially when context differs.
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel > 1e-2, f"output insensitive to source (relative max|Δ| = {rel:.3e})"


def test_llm_adapter_attends_to_target(loaded_adapter):
    """Changing the target tokens changes the output. Catches a bug where the
    embedding/self-attn path would be silently bypassed."""
    src = torch.randn(1, 9, 1024)
    tgt1 = torch.randint(0, 32128, (1, 5))
    tgt2 = (tgt1 + 100) % 32128
    with torch.no_grad():
        a = loaded_adapter(src, tgt1)
        b = loaded_adapter(src, tgt2)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel > 1e-2, f"output insensitive to target (relative max|Δ| = {rel:.3e})"


def test_llm_adapter_mask_changes_output(loaded_adapter):
    """A non-trivial attention mask must change the output."""
    torch.manual_seed(0)
    src = torch.randn(1, 9, 1024)
    tgt = torch.randint(0, 32128, (1, 7))
    with torch.no_grad():
        no_mask = loaded_adapter(src, tgt)
        # Mask out the last half of source positions.
        smask = torch.ones(1, 9, dtype=torch.bool)
        smask[:, 5:] = False
        masked = loaded_adapter(src, tgt, source_attention_mask=smask)
    rel = (no_mask - masked).abs().max().item() / no_mask.abs().max().item()
    assert rel > 1e-3, f"mask had no effect (relative max|Δ| = {rel:.3e})"
