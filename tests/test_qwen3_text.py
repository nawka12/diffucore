"""Qwen3 0.6B text encoder (DT3) verification.

Three layers:

1. Key-set match against the on-disk Qwen3 safetensors header.
2. Strict load + forward shape check.
3. Bit-identity (max|Δ| = 0 in fp32) against ``transformers.Qwen3Model`` —
   the Apache-2.0 oracle we use, same pattern as SDXL's CLIP-L/bigG match.

Tests 2 and 3 are skipped if the Qwen3 checkpoint isn't on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from diffucore.models import Qwen3Config, Qwen3TextEncoder

_TE_CKPT = Path(os.environ.get(
    "DIFFUCORE_QWEN3_TE",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors",
))


@pytest.fixture(scope="module")
def loaded_te():
    if not _TE_CKPT.exists():
        pytest.skip(f"qwen3 TE checkpoint not at {_TE_CKPT}")
    from safetensors.torch import load_file
    te = Qwen3TextEncoder()
    te.load_state_dict(load_file(str(_TE_CKPT)), strict=True)
    return te.float().eval()


def test_qwen3_key_set_matches_checkpoint():
    from safetensors import safe_open
    if not _TE_CKPT.exists():
        pytest.skip(f"qwen3 TE checkpoint not at {_TE_CKPT}")

    with safe_open(str(_TE_CKPT), framework="pt") as f:
        ckpt_keys = set(f.keys())
    mod_keys = set(Qwen3TextEncoder().state_dict().keys())
    assert mod_keys == ckpt_keys, (
        f"key mismatch — missing: {sorted(ckpt_keys - mod_keys)[:5]} | "
        f"extra: {sorted(mod_keys - ckpt_keys)[:5]}"
    )


def test_qwen3_strict_load_and_forward_shape(loaded_te):
    """A fixed token sequence flows through and produces (B, T, 1024)."""
    input_ids = torch.tensor([[151643, 9707, 11, 1879, 0]])
    with torch.no_grad():
        out = loaded_te(input_ids)
    assert out.shape == (1, 5, Qwen3Config().hidden_size)


def test_qwen3_bit_identical_to_transformers(loaded_te):
    """Final hidden states bit-identical (max|Δ| = 0) to HF Qwen3Model in fp32
    with eager attention. This is the diffucore bar set by SDXL CLIP-L/bigG."""
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3Model
    from transformers import Qwen3Config as HFConfig
    from safetensors.torch import load_file

    cfg = Qwen3Config()
    hf_cfg = HFConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    hf = Qwen3Model(hf_cfg)
    sd = load_file(str(_TE_CKPT))
    sd_hf = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    hf.load_state_dict(sd_hf, strict=True)
    hf = hf.float().eval()

    # A few tokens drawn from the Qwen2.5 vocab — the exact IDs don't matter
    # for the comparison; we just need a non-trivial sequence.
    input_ids = torch.tensor([[151643, 9707, 11, 1879, 0, 358, 1079, 264, 4128, 1614, 13]])
    with torch.no_grad():
        o = loaded_te(input_ids)
        h = hf(input_ids).last_hidden_state

    assert o.shape == h.shape
    diff = (o - h).abs().max().item()
    assert torch.equal(o, h), f"not bit-identical: max|Δ| = {diff:.3e}"
