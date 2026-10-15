"""Qwen3.5 hybrid text encoder (experimental) verification.

The encoder covers two checkpoints: the Anima-packaged 4B (with a baked-in
projection head) and the raw 0.8B base (plain final norm, no projection). Both
are multi-GB, so the offline tests build the module on the ``meta`` device (no
allocation) and check it against the vendored backbone headers
(``qwen35_4b_header.json`` / ``qwen35_08b_base_header.json``, the key→shape maps
read off the real files, prefix already stripped / vision+MTP dropped):

1. ``from_state_dict`` derives the right config and the module's keys+shapes
   match the real backbone, so a strict load of the real file succeeds.
2. The default config reproduces the 4B (keeps the defaults honest).
3. A small-config forward exercises every path (Gated DeltaNet, gated attention,
   hybrid routing, partial RoPE, ExpRMSNorm / plain-norm head) and checks shape/finite.
4. The chunked delta rule equals a naive per-token recurrence.
5. Strict load + forward on a real checkpoint, skipped unless one is on disk.
6. Per-layer and next-token-loss agreement with Hugging Face transformers on the
   0.8B base weights, skipped unless the file and transformers' qwen3_5 exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from diffucore.bundle import _extract_component
from diffucore.models import Qwen35Config, Qwen35TextEncoder
from diffucore.models.qwen35_text import _chunk_gated_delta_rule

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
        f"[{name}] key mismatch; missing: {sorted(set(real) - set(mod))[:5]} | "
        f"extra: {sorted(set(mod) - set(real))[:5]}"
    )
    bad = {k: (mod[k], real[k]) for k in real if mod[k] != real[k]}
    assert not bad, f"[{name}] shape mismatch (module, ckpt): {dict(list(bad.items())[:5])}"


def test_qwen35_08b_base_uses_plain_norm_head():
    """The 0.8B base has no projection head, just a plain final RMSNorm at hidden
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
    """Tiny config: both head kinds run and yield the expected output width."""
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
    assert cos < 0.999, f"outputs collapsed (cos={cos:.4f}); ExpRMSNorm not applied?"


def _naive_gated_delta_rule(q, k, v, g, beta):
    """Per-token recurrence the chunked form must reproduce (fp32)."""
    q, k = F.normalize(q.float(), dim=-1, eps=1e-12), F.normalize(k.float(), dim=-1, eps=1e-12)
    q = q * q.shape[-1] ** -0.5
    B, T, H, Dk = k.shape
    S = torch.zeros(B, H, Dk, v.shape[-1])
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (S * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        S = S + k[:, t, :, :, None] * delta[:, :, None, :]
        out.append((S * q[:, t, :, :, None]).sum(-2))
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("T", [5, 64, 150])
def test_chunked_delta_rule_matches_naive_recurrence(T):
    gen = torch.Generator().manual_seed(0)
    B, H, Dk, Dv = 2, 3, 16, 24
    q, k = (torch.randn(B, T, H, Dk, generator=gen) for _ in range(2))
    v = torch.randn(B, T, H, Dv, generator=gen)
    g = -torch.rand(B, T, H, generator=gen) * 0.5            # log-decay <= 0
    beta = torch.rand(B, T, H, generator=gen)
    got = _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
    want = _naive_gated_delta_rule(q, k, v, g, beta)
    assert got.shape == want.shape
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-4, f"chunked vs naive delta rule rel diff {rel:.2e}"


def test_qwen_encode_runs_without_autograd_graph():
    """``_qwen_encode`` must run under no_grad: the chunked delta rule would
    otherwise retain every chunk's state for backward. Guards both encoders
    against a regression that drops the wrapper."""
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


# Official Qwen/Qwen3.5-0.8B-Base text_config (what the checkpoint was trained with).
_HF_08B_TEXT_CONFIG = dict(
    attention_bias=False, attention_dropout=0.0, attn_output_gate=True, head_dim=256,
    hidden_act="silu", hidden_size=1024, intermediate_size=3584,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=16,
    linear_num_value_heads=16, linear_value_head_dim=128, max_position_embeddings=262144,
    num_attention_heads=8, num_hidden_layers=24, num_key_value_heads=2, rms_norm_eps=1e-6,
    tie_word_embeddings=True, vocab_size=248320,
    rope_parameters=dict(
        mrope_interleaved=True, mrope_section=[11, 11, 10], rope_type="default",
        rope_theta=10000000, partial_rotary_factor=0.25,
    ),
)


def test_qwen35_08b_matches_huggingface():
    """Oracle: same weights through transformers' Qwen3.5 and ours, fp32."""
    ckpt = _CKPTS["08b_base"]
    if not ckpt.exists():
        pytest.skip(f"qwen3.5 0.8B base checkpoint not at {ckpt}")
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    except ImportError:
        pytest.skip("transformers with qwen3_5 not installed")
    from safetensors.torch import load_file
    from diffucore.conditioning.anima_tokenizer import AnimaTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bare = _extract_component(load_file(str(ckpt)), "embed_tokens.weight")
    ours = Qwen35TextEncoder(Qwen35Config.from_state_dict(bare))
    ours.load_state_dict(bare, strict=True)
    ours = ours.float().to(dev).eval()

    cfg = dict(_HF_08B_TEXT_CONFIG)
    cfg["layer_types"] = cfg["layer_types"] * 6
    hf = M.Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg))
    hf.load_state_dict({"model." + k: v for k, v in bare.items()}, strict=False)
    hf.tie_weights()
    hf = hf.float().to(dev).eval()

    tok = AnimaTokenizer.qwen35()
    texts = [
        "The quick brown fox jumps over the lazy dog. It was a sunny afternoon, and the children were playing in the park near the river.",
        "In 1969, Neil Armstrong became the first person to walk on the Moon, a moment watched by millions of people around the world.",
    ]
    n = len(ours.layers)
    for text in texts:
        ids = tok(text).qwen_ids.to(dev)
        with torch.no_grad():
            ref = hf(ids, labels=ids, output_hidden_states=True)
            x, caps = ours.run_layers(ours.embed_tokens(ids), 0, n, caps=range(1, n + 1))
            logits = ours.norm(x) @ bare["embed_tokens.weight"].float().to(dev).T
            loss = F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
        for i in range(n):
            h = ref.hidden_states[i + 1]
            mine = ours.norm(x) if i + 1 == n else caps[i + 1]   # HF's last state is post-norm
            rel = ((mine - h).norm() / h.norm()).item()
            assert rel < 1e-4, f"layer {i + 1}: rel diff {rel:.2e}"
        assert abs(loss - ref.loss.item()) < 1e-3, (loss, ref.loss.item())
