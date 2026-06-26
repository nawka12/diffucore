"""Qwen3.5 hybrid text encoder (experimental) verification.

The encoder covers two checkpoints — the Anima-packaged 4B (with a baked-in
projection head) and the raw 0.8B base (plain final norm, no projection). Both
are multi-GB, so the offline tests build the module on the ``meta`` device (no
allocation) and check it against the vendored backbone headers
(``qwen35_4b_header.json`` / ``qwen35_08b_base_header.json`` — the key→shape maps
read off the real files, prefix already stripped / vision+MTP dropped):

1. ``from_state_dict`` derives the right config and the module's keys+shapes
   match the real backbone — i.e. a strict load of the real file succeeds.
2. The default config reproduces the 4B (keeps the defaults honest).
3. A small-config forward exercises every path (SSM scan, gated attention,
   hybrid routing, RoPE, ExpRMSNorm / plain-norm head) and checks shape/finite.
4. Strict load + forward on a real checkpoint — skipped unless one is on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from diffucore.bundle import _extract_component
from diffucore.models import Qwen35Config, Qwen35TextEncoder

_HEADERS = {
    "4b": Path(__file__).with_name("qwen35_4b_header.json"),
    "08b_base": Path(__file__).with_name("qwen35_08b_base_header.json"),
}
# Real checkpoints, keyed the same way; skipped when absent.
_CKPTS = {
    "4b": Path(os.environ.get(
        "DIFFUCORE_QWEN35_TE",
        "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/text_encoders/qwen35_4b.safetensors",
    )),
    "08b_base": Path(os.environ.get(
        "DIFFUCORE_QWEN35_08B_TE",
        "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/diffucore-ui/models/text-encoders/qwen_3_5_08b_base.safetensors",
    )),
}


def _header(name: str) -> dict[str, tuple]:
    return {k: tuple(v) for k, v in json.loads(_HEADERS[name].read_text()).items()}


def _meta_state_dict(cfg: Qwen35Config) -> dict[str, tuple]:
    with torch.device("meta"):
        model = Qwen35TextEncoder(cfg)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


@pytest.mark.parametrize("name", list(_HEADERS))
def test_qwen35_derived_config_matches_real_header(name):
    real = _header(name)
    dummy = {k: torch.empty(s, device="meta") for k, s in real.items()}
    cfg = Qwen35Config.from_state_dict(dummy)
    mod = _meta_state_dict(cfg)
    assert set(mod) == set(real), (
        f"[{name}] key mismatch — missing: {sorted(set(real) - set(mod))[:5]} | "
        f"extra: {sorted(set(mod) - set(real))[:5]}"
    )
    bad = {k: (mod[k], real[k]) for k in real if mod[k] != real[k]}
    assert not bad, f"[{name}] shape mismatch (module, ckpt): {dict(list(bad.items())[:5])}"


def test_qwen35_08b_base_uses_plain_norm_head():
    """The 0.8B base has no projection head — a plain final RMSNorm at hidden
    width, not the 4B's Linear→ExpRMSNorm→SiLU→Linear (``norm.0/1/3``)."""
    cfg = Qwen35Config.from_state_dict(
        {k: torch.empty(s, device="meta") for k, s in _header("08b_base").items()}
    )
    assert cfg.output_projection is False
    assert cfg.output_dim == cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.self_attn_layers == (3, 7, 11, 15, 19, 23)
    assert cfg.no_mlp_layers == ()           # every layer keeps its MLP
    keys = set(_meta_state_dict(cfg))
    assert "norm.weight" in keys and "norm.0.weight" not in keys


def test_qwen35_default_config_is_4b():
    assert _meta_state_dict(Qwen35Config()) == _header("4b")


def _tiny_cfg(**over):
    base = dict(
        vocab_size=512, hidden_size=64, intermediate_size=128, output_dim=32,
        output_projection=True, num_hidden_layers=4, self_attn_layers=(1, 3),
        no_mlp_layers=(3,), num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        ssm_d_ssm=32, ssm_conv_dim=96, ssm_n_groups=4, ssm_head_dim=8, ssm_d_state=8,
    )
    base.update(over)
    return Qwen35Config(**base)


@pytest.mark.parametrize("projection", [True, False])
def test_qwen35_forward_small_config(projection):
    """Tiny config — both head kinds run and yield the expected output width."""
    out_dim = 32 if projection else 64  # plain norm keeps hidden width (64)
    cfg = _tiny_cfg(output_projection=projection)
    model = Qwen35TextEncoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (2, 7, out_dim)
    assert torch.isfinite(out).all()


def test_qwen35_exp_rms_norm_preserves_diversity():
    """The 4B projection head must use exp(weight): near-zero weights keep
    token-distinct outputs (a plain RMSNorm scale would collapse them)."""
    cfg = _tiny_cfg(num_hidden_layers=2, self_attn_layers=(1,), no_mlp_layers=())
    model = Qwen35TextEncoder(cfg).eval()
    with torch.no_grad():
        model.norm[1].weight.fill_(-0.003)   # the checkpoint's near-zero regime
        a = model(torch.tensor([[1, 2, 3, 4]]))
        b = model(torch.tensor([[5, 6, 7, 8]]))
    cos = torch.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    assert cos < 0.999, f"outputs collapsed (cos={cos:.4f}) — ExpRMSNorm not applied?"


def test_qwen_encode_runs_without_autograd_graph():
    """``_qwen_encode`` must run under no_grad — the hybrid encoder's unrolled SSM
    scan would otherwise retain every per-timestep state and OOM. Guards both
    encoders against a regression that drops the wrapper."""
    from diffucore.pipelines._anima import _qwen_encode
    model = Qwen35TextEncoder(_tiny_cfg()).eval()
    out = _qwen_encode(model, torch.randint(0, 512, (1, 6)), None, torch.device("cpu"), torch.float32)
    assert out.grad_fn is None and not out.requires_grad


@pytest.mark.parametrize("name", list(_CKPTS))
def test_qwen35_strict_load_and_forward_shape(name):
    ckpt = _CKPTS[name]
    if not ckpt.exists():
        pytest.skip(f"qwen3.5 {name} checkpoint not at {ckpt}")
    from safetensors.torch import load_file
    # Same path the loader takes: strip prefix + drop vision/MTP, then derive cfg.
    sd = _extract_component(load_file(str(ckpt)), "embed_tokens.weight")
    assert any("linear_attn.A_log" in k for k in sd), "not a Qwen3.5 hybrid checkpoint"
    model = Qwen35TextEncoder(Qwen35Config.from_state_dict(sd))
    model.load_state_dict(sd, strict=True)
    model = model.float().eval()
    ids = torch.tensor([[9707, 11, 1879, 0]])
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (1, 4, 1024)
    assert torch.isfinite(out).all()
