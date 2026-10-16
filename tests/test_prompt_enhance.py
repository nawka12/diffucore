"""Prompt enhancer: the KV-cached decode matches a full recompute through
``Qwen3TextEncoder`` with the tied head, sampling filters behave, and
``parked_on_cpu`` round-trips a module's CUDA tensors."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffucore.models import Qwen3Config, Qwen3TextEncoder
from diffucore.pipelines.prompt_enhance import PromptEnhancer, _sample, load_prompt_enhancer
from diffucore.runtime import DevicePolicy, parked_on_cpu

_TINY = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=5_000_000.0)


def _enhancer() -> PromptEnhancer:
    torch.manual_seed(0)
    lm = Qwen3TextEncoder(_TINY).float().eval()
    # Fp32 RoPE tables in the reference too, matching the enhancer's.
    lm.rotary_emb.inv_freq = 1.0 / (_TINY.rope_theta ** (
        torch.arange(0, _TINY.head_dim, 2, dtype=torch.float64) / _TINY.head_dim)).float()
    return PromptEnhancer(lm, DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32))


def _full_logits(lm: Qwen3TextEncoder, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        h = lm(torch.tensor([ids]))
    return F.linear(h[0, -1], lm.model.embed_tokens.weight)


def test_kv_cache_decode_matches_full_recompute():
    enh = _enhancer()
    prompt = [151644, 8948, 198, 13048, 151645, 198, 151644, 77091, 198]
    greedy = [t for t in enh._generate(prompt, 6, 0.0, 0.8, 20, torch.Generator(), None) if t not in enh._stop]
    ids = list(prompt)
    for tok in greedy:
        assert tok == int(_full_logits(enh.lm, ids).argmax())
        ids.append(tok)
    # Logits too, not just the argmax, at the last step.
    total = len(ids) + 1
    inv = 1.0 / (_TINY.rope_theta ** (torch.arange(0, 16, 2, dtype=torch.float64) / 16))
    ang = torch.arange(total, dtype=torch.float64)[:, None] * inv
    ang = torch.cat([ang, ang], -1)
    kv = torch.empty(3, 2, 1, 2, total, 16)
    with torch.no_grad():
        enh._forward(torch.tensor([ids[:-1]]), 0, ang.cos().float(), ang.sin().float(), kv)
        step = enh._forward(torch.tensor([ids[-1:]]), len(ids) - 1, ang.cos().float(), ang.sin().float(), kv)
    torch.testing.assert_close(step, _full_logits(enh.lm, ids), rtol=1e-4, atol=1e-4)


def test_generation_stops_at_im_end():
    enh = _enhancer()
    with torch.no_grad():
        enh.lm.model.embed_tokens.weight[151645] *= 50   # make <|im_end|> the argmax
    assert enh._generate([151644, 8948, 198], 10, 0.0, 0.8, 20, torch.Generator(), None) == [151645]


def test_sample_filters():
    logits = torch.tensor([5.0, 4.0, 1.0, 0.0])
    assert _sample(logits, 0.0, 0.8, 20, torch.Generator()) == 0
    g = torch.Generator().manual_seed(0)
    # top_k 2 never leaves the first two.
    assert {_sample(logits, 1.0, 1.0, 2, g) for _ in range(200)} <= {0, 1}
    # top_p small keeps only the head token.
    assert {_sample(logits, 1.0, 0.1, 0, g) for _ in range(50)} == {0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_parked_on_cpu_round_trip():
    lin = torch.nn.Linear(4, 4).cuda()
    w = lin.weight.detach().clone()
    with parked_on_cpu([lin]):
        assert lin.weight.device.type == "cpu"
    assert lin.weight.device.type == "cuda"
    torch.testing.assert_close(lin.weight, w)


def test_expand_keeps_the_prompt_and_stops_at_the_newline(monkeypatch):
    """The answer starts with the prompt verbatim (trailing separators
    normalized); the model's text is cut at its newline, keeping what came
    before it in the same token."""
    enh = _enhancer()
    seen = {}
    def fake_run(ids, max_new_tokens, stop, **kw):
        seen["text"], seen["stop"] = enh.tok.decode(ids), stop
        return enh.tok.encode(" Snow falls on her hat.\nShort prompt: x").ids
    monkeypatch.setattr(enh, "_run", fake_run)
    out = enh.expand("1girl,  @artist, witch hat, ", examples=(("a", "B."),))
    assert out == "1girl, @artist, witch hat. Snow falls on her hat."
    assert seen["text"].endswith("Short prompt: 1girl, @artist, witch hat\nDetailed prompt: 1girl, @artist, witch hat.")
    assert enh.tok.encode(".\n").ids[0] in seen["stop"] and 151645 in seen["stop"]


def test_loader_reads_qwen3_and_rejects_others(tmp_path):
    from safetensors.torch import save_file
    lm = Qwen3TextEncoder(_TINY)
    sd = {k: v.contiguous() for k, v in lm.state_dict().items()}
    vl = {"model.language_model." + k[len("model."):]: v for k, v in sd.items()}
    vl["model.visual.patch_embed.proj.bias"] = torch.zeros(4)
    save_file(sd, str(tmp_path / "plain.safetensors"))
    save_file(vl, str(tmp_path / "vl.safetensors"))
    save_file({"layers.0.linear_attn.A_log": torch.zeros(4)}, str(tmp_path / "hybrid.safetensors"))
    pol = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32)
    assert load_prompt_enhancer(str(tmp_path / "plain.safetensors"), pol).lm.cfg.rope_theta == 1_000_000.0
    assert load_prompt_enhancer(str(tmp_path / "vl.safetensors"), pol).lm.cfg.rope_theta == 5_000_000.0
    with pytest.raises(ValueError, match="not a Qwen3"):
        load_prompt_enhancer(str(tmp_path / "hybrid.safetensors"), pol)
