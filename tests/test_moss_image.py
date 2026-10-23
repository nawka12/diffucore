"""MOSS-Image (Base + SFT) verification, offline, CPU.

1. The DiT, VAE and text-encoder loaders match the real files' keys and shapes
   (``moss_headers.json``: ComfyUI bf16 / INT8 ConvRot DiT, the official
   diffusers key names, both FLUX.2 VAEs, Qwen3.5-9B bf16 / INT8).
2. A tiny random DiT matches the reference ``MOSSImageTransformer`` (vendored in
   ``_moss_reference.py``) in fp32, with and without skipped blocks and at
   residual scales 1 and 1/16.
3. The ``moss`` schedule is the reference scheduler's, the tokenizer renders the
   reference's chat template, the int8 embedding equals the dequantized table,
   the Qwen3.5 loader never reads the vision / MTP / lm_head tensors.
4. A tiny end-to-end text-to-image run exercises CFG, the uncond skip and the
   conditioning cache.

The tokenizer oracle needs ``MOSS_TOKENIZER_DIR`` (the official ``tokenizer``
folder: ``tokenizer.json`` and ``chat_template.jinja``) and is skipped without it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import _moss_reference as ref  # noqa: E402

from diffucore.bundle import (  # noqa: E402
    ModelBundle, _build, _build_flux_vae, _infer_vae_config, _moss_dit_state, _moss_te_layout,
    _moss_text_encoder, load_moss_checkpoint, load_moss_dit,
)
from diffucore.conditioning import MossTokenizer  # noqa: E402
from diffucore.conditioning.moss_tokenizer import MAX_LENGTH, MossTokenized, render_prompt  # noqa: E402
from diffucore.loading import ModelSpec, detect_architecture  # noqa: E402
from diffucore.models import (  # noqa: E402
    AutoencoderKL, MossConfig, MossDiT, Qwen35Config, Qwen35TextEncoder, VAEConfig,
)
from diffucore.models._int8 import (  # noqa: E402
    Int8Embedding, adopt_int8_embedding, adopt_int8_layers, dequantize_weight, int8_layers, quantize_weight,
)
from diffucore.models.moss_dit import drop_unused_keys, from_diffusers, int8_layer_names  # noqa: E402
from diffucore.models.vae import convert_diffusers_vae_state_dict  # noqa: E402
from diffucore.pipelines import TextToImage  # noqa: E402
from diffucore.pipelines._moss import moss_schedule, moss_sigmas, moss_text_to_image  # noqa: E402
from diffucore.runtime import ConditioningCache, DevicePolicy  # noqa: E402

_FIX = json.loads(Path(__file__).with_name("moss_headers.json").read_text())
_TOKENIZER_DIR = os.environ.get("MOSS_TOKENIZER_DIR")
_TAG = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                                     "convrot_groupsize": 256}).encode()), dtype=torch.uint8)


def _expand(entry: dict) -> dict:
    """``{key: (dtype, shape)}`` from a fixture entry (layers stored as templates)."""
    out = {k: (v[0], tuple(v[1])) for k, v in entry["global"].items()}
    for indices, keys in entry["layers"]:
        for i in indices:
            out.update({f"{entry['prefix']}{i}.{k}": (v[0], tuple(v[1])) for k, v in keys.items()})
    return out


def _shapes(hdr: dict) -> dict:
    return {k: s for k, (_, s) in hdr.items()}


def _meta_sd(hdr: dict) -> dict:
    """Meta tensors for a header; ``comfy_quant`` tags become real JSON tags."""
    return {k: (_TAG.clone() if k.endswith(".comfy_quant") else torch.empty(s, device="meta"))
            for k, s in _shapes(hdr).items()}


def _official_header() -> dict:
    """The official diffusers shards: the bf16 header with q/k/v split apart."""
    out = {}
    for k, (dt, s) in _expand(_FIX["comfy_bf16"]).items():
        if k.endswith(".proj.qkv.weight"):
            base = k[: -len("qkv.weight")]
            out.update({base + n + ".weight": (dt, (s[0] // 3, s[1])) for n in "qkv"})
        else:
            out[k] = (dt, s)
    return out


# ─── keys and shapes ─────────────────────────────────────────────────────────


def test_official_key_names_match_the_fixture():
    assert sorted(_official_header()) == _FIX["official_keys"]


def test_detects_moss_in_every_layout_and_ignores_the_control_lora():
    for hdr in (_expand(_FIX["comfy_bf16"]), _expand(_FIX["comfy_int8_convrot"]), _official_header()):
        spec = detect_architecture(_shapes(hdr))
        assert (spec.architecture, spec.prediction, spec.latent_channels, spec.context_dim, spec.image_size,
                spec.latent_scale) == ("moss_image", "flow", 128, 4096, 1024, 1.0)
    prefixed = {"model.diffusion_model." + k: s for k, s in _shapes(_expand(_FIX["comfy_bf16"])).items()}
    assert detect_architecture(prefixed).architecture == "moss_image"
    with pytest.raises(ValueError):
        detect_architecture(_shapes(_FIX["control_lora_sample"]))


def test_config_is_derived_from_the_keys():
    for name in ("comfy_bf16", "comfy_int8_convrot"):
        sd = _meta_sd(_expand(_FIX[name]))
        assert MossConfig.from_state_dict(sd) == MossConfig()


def test_dit_keys_and_shapes_match_the_comfy_files():
    model = _build(MossDiT, MossConfig())
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    bf16 = {k: s for k, s in _shapes(_expand(_FIX["comfy_bf16"])).items() if k != "rope.bands"}
    assert ours == bf16
    int8 = _expand(_FIX["comfy_int8_convrot"])
    assert {k for k in int8 if not k.endswith((".weight_scale", ".comfy_quant", ".comfy_attention.config"))
            and k != "rope.bands"} == set(bf16)


def test_int8_file_loads_strictly_with_every_block_linear_quantized():
    sd = _meta_sd(_expand(_FIX["comfy_int8_convrot"]))
    model = _build(MossDiT, MossConfig())
    sd = drop_unused_keys(sd)
    int8 = int8_layers(sd)
    assert sorted(int8) == sorted(int8_layer_names(42)) and set(int8.values()) == {256}
    sd = adopt_int8_layers(model, sd, int8)
    model.load_state_dict(sd, strict=True, assign=True)


def test_official_shards_fuse_into_the_comfy_layout():
    sd = _meta_sd(_official_header())
    fused = drop_unused_keys(from_diffusers(sd))
    want = {k: s for k, s in _shapes(_expand(_FIX["comfy_bf16"])).items() if k != "rope.bands"}
    assert {k: tuple(v.shape) for k, v in fused.items()} == want
    assert not any(".proj.q." in k or ".proj.k." in k or ".proj.v." in k for k in sd if k in fused)


def test_fusion_stacks_q_k_v_in_that_order():
    q, k, v = (torch.full((4, 8), float(i)) for i in range(3))
    sd = {"layers.0.proj.k.weight": k, "layers.0.proj.o.weight": torch.zeros(1),
          "layers.0.proj.q.weight": q, "layers.0.proj.v.weight": v}
    fused = from_diffusers(sd)
    assert sorted(fused) == ["layers.0.proj.o.weight", "layers.0.proj.qkv.weight"]
    assert torch.equal(fused["layers.0.proj.qkv.weight"], torch.cat([q, k, v]))


def test_moss_dit_state_reads_a_prefixed_official_shard(tmp_path):
    torch.manual_seed(0)
    q, k, v = (torch.randn(8, 4) for _ in range(3))
    save_file({"model.diffusion_model.layers.0.proj.q.weight": q, "model.diffusion_model.layers.0.proj.k.weight": k,
               "model.diffusion_model.layers.0.proj.v.weight": v,
               "model.diffusion_model.shared_block_modulation.weight": torch.zeros(2, 2),
               "model.diffusion_model.rope.bands": torch.zeros(16)}, str(tmp_path / "a.safetensors"))
    sd = _moss_dit_state(str(tmp_path / "a.safetensors"))
    assert sorted(sd) == ["layers.0.proj.qkv.weight", "shared_block_modulation.weight"]
    assert torch.equal(sd["layers.0.proj.qkv.weight"], torch.cat([q, k, v]))


# ─── DiT against the reference ───────────────────────────────────────────────


def _tiny_pair(seed: int = 0):
    cfg = dict(dim=256, ffn_dim=512, num_heads=2, num_layers=3, text_dim=64)
    torch.manual_seed(seed)
    rmodel = ref.MOSSImageTransformer(**cfg).float().eval()
    with torch.no_grad():
        for name, p in rmodel.named_parameters():
            p.copy_(1.0 + 0.2 * torch.randn_like(p) if name.endswith("norm.weight") or ".norm" in name
                    or "_norm." in name else 0.05 * torch.randn_like(p))
    sd = dict(rmodel.state_dict())
    for i in range(cfg["num_layers"]):
        base = f"layers.{i}.proj."
        sd[base + "qkv.weight"] = torch.cat([sd.pop(base + n + ".weight") for n in "qkv"], 0)
    del sd["rope.bands"]
    ours = MossDiT(MossConfig.from_state_dict(sd)).float().eval()
    ours.load_state_dict(sd, strict=True)
    return rmodel, ours


@pytest.mark.parametrize("scale", [1.0, 1 / 16, 0.3])
@pytest.mark.parametrize("skip", [(), (1,), (0, 2)])
@pytest.mark.parametrize("text_len", [5, 11])
def test_dit_matches_the_reference_forward(text_len, skip, scale):
    rmodel, ours = _tiny_pair()
    ours.residual_scale = scale
    torch.manual_seed(1)
    x, feats, sigma = torch.randn(1, 128, 4, 6), torch.randn(1, text_len, 64), torch.tensor([0.37])
    with torch.no_grad():
        want = rmodel(x, sigma * 1000.0, feats, skip_layers=skip).sample
        mod, head = ours.time_vectors(sigma)
        got = ours(x, mod, head, ours.encode_text(feats), skip)
    assert got.shape == want.shape == (1, 128, 4, 6)
    assert (got - want).abs().max() <= 1e-5


def test_image_rope_depends_on_the_text_length():
    _, ours = _tiny_pair()
    short, long = ours.rope(5, 4, 6, "cpu")[0], ours.rope(11, 4, 6, "cpu")[0]
    assert not torch.allclose(short[:24], long[:24])


def test_skip_layers_out_of_range_is_rejected():
    _, ours = _tiny_pair()
    mod, head = ours.time_vectors(torch.tensor([0.5]))
    with pytest.raises(ValueError, match="skip_layers"):
        ours(torch.randn(1, 128, 2, 2), mod, head, ours.encode_text(torch.randn(1, 3, 64)), (3,))


# ─── schedule ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("steps", [1, 8, 28, 50, 200])
def test_moss_schedule_is_the_reference_scheduler(steps):
    got, want = moss_sigmas(steps), ref.reference_sigmas(steps)
    assert torch.equal(got, want)
    assert got[-1] == 0 and got[0] == 1.0


def test_moss_schedule_tail_is_below_the_plain_flow_schedule():
    sigmas, shift = moss_schedule("moss", 50, 1024, 1024, 3.0, "cpu")
    assert shift == 5.0 and abs(float(sigmas[-2]) - 0.024) < 1e-3
    flow, _ = moss_schedule("flow", 50, 1024, 1024, 5.0, "cpu")
    assert float(flow[-2]) > 0.09


# ─── tokenizer ───────────────────────────────────────────────────────────────

_PROMPTS = [
    "A cat", "a very long prompt, " * 80, "猫が窓辺で眠っている。夕焼けの空", "a fox 🦊 in the snow 😀",
    "  leading and trailing whitespace \n\n", "line one\nline two\n\n  indented line", '{"layout": [1, 2, 3]}', "",
    "สวัสดีครับ ยินดีต้อนรับ", "नमस्ते दुनिया",
]


def test_template_and_special_tokens():
    assert render_prompt("  a cat\n") == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
    tok = Tokenizer.from_file(str(Path(__file__).parent.parent / "src/diffucore/conditioning/qwen35_tokenizer.json"))
    ids = MossTokenizer()("a cat").ids[0].tolist()
    for special, count in (("<|im_start|>", 2), ("<|im_end|>", 1), ("<think>", 1)):
        assert ids.count(tok.token_to_id(special)) == count
    assert ids[0] == tok.token_to_id("<|im_start|>") and ids[-2] == tok.token_to_id("<think>")
    assert ids[-1] == tok.encode("\n").ids[0]
    assert MossTokenizer()("  a cat\n").ids.tolist() == MossTokenizer()("a cat").ids.tolist()
    assert MossTokenizer()("").ids.shape[1] > 0
    # Combining marks split from their letters, as in the official tokenizer.json.
    assert MossTokenizer()("नमस्ते").ids.shape[1] > len(tok.encode(render_prompt("नमस्ते")).ids)


def test_truncation_keeps_the_head_and_flags_it():
    out = MossTokenizer()("word " * 6000)
    assert out.ids.shape == (1, MAX_LENGTH) and out.truncated
    assert not MossTokenizer()("word").truncated


@pytest.mark.skipif(not _TOKENIZER_DIR, reason="MOSS_TOKENIZER_DIR (official tokenizer folder) not set")
@pytest.mark.parametrize("prompt", _PROMPTS)
def test_ids_match_the_official_tokenizer(prompt):
    official = Tokenizer.from_file(str(Path(_TOKENIZER_DIR) / "tokenizer.json"))
    want = official.encode(render_prompt(prompt)).ids[:MAX_LENGTH]
    assert MossTokenizer()(prompt).ids[0].tolist() == want


@pytest.mark.skipif(not _TOKENIZER_DIR, reason="MOSS_TOKENIZER_DIR (official tokenizer folder) not set")
@pytest.mark.parametrize("prompt", _PROMPTS)
def test_template_matches_the_hf_chat_template(prompt):
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoTokenizer.from_pretrained(_TOKENIZER_DIR)
    formatted = hf.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    assert formatted == render_prompt(prompt)


# ─── int8 embedding ──────────────────────────────────────────────────────────


def test_int8_embedding_rows_equal_the_dequantized_table():
    torch.manual_seed(0)
    table = torch.randn(50, 512) * 0.02
    q, scale = quantize_weight(table, 256)
    emb = Int8Embedding(50, 512, 256)
    emb.weight.copy_(q)
    emb.scale_bits.copy_(scale.view(torch.int32))
    ids = torch.tensor([[3, 7, 3], [0, 49, 10]])
    want = dequantize_weight(q, scale, 256)[ids]
    torch.testing.assert_close(emb(ids), want, rtol=1e-5, atol=1e-8)
    assert emb(ids).shape == (2, 3, 512)
    assert ((emb(ids) - table[ids]).norm() / table[ids].norm()) < 0.01
    emb.out_dtype = torch.float16
    assert emb(ids).dtype == torch.float16


def test_int8_embedding_adopts_a_comfy_table():
    torch.manual_seed(0)
    table = torch.randn(40, 256) * 0.02
    q, scale = quantize_weight(table, 256)
    holder = torch.nn.Module()
    holder.embed_tokens = torch.nn.Embedding(40, 256, device="meta")
    sd = {"embed_tokens.weight": q, "embed_tokens.weight_scale": scale.unsqueeze(1), "embed_tokens.comfy_quant": _TAG.clone()}
    group = int8_layers(sd).pop("embed_tokens")
    sd = adopt_int8_embedding(holder, sd, "embed_tokens", group)
    holder.load_state_dict(sd, strict=True, assign=True)
    ids = torch.tensor([[5, 39, 0]])
    assert isinstance(holder.embed_tokens, Int8Embedding)
    torch.testing.assert_close(holder.embed_tokens(ids), dequantize_weight(q, scale, 256)[ids], rtol=1e-5, atol=1e-8)


# ─── VAE ─────────────────────────────────────────────────────────────────────

_SPEC = ModelSpec(architecture="moss_image", prediction="flow", zero_terminal_snr=False, latent_channels=128,
                  context_dim=4096, latent_scale=1.0)


def _vae_sd(name: str) -> dict:
    return {k: torch.empty(s, device="meta") for k, s in _shapes(_FIX[name]).items()}


def _strict_vae_load(name: str):
    vae, sd = _build_flux_vae(_vae_sd(name), _SPEC)
    state = {k: v for k, v in sd.items() if not k.startswith("bn.")}
    vae.load_state_dict(state, strict=True, assign=True)
    return vae, sd


def test_small_diffusers_vae_loads_with_its_own_decoder_widths():
    vae, sd = _strict_vae_load("vae_small")
    cfg = vae.config
    assert (cfg.base_channels, cfg.channel_mult, cfg.num_res_blocks, cfg.z_channels) == (128, (1, 2, 4, 4), 2, 32)
    assert cfg.decoder_channels == (96, 192, 384, 384) and cfg.use_quant_conv
    assert tuple(vae.flux2_latent_mean.shape) == tuple(vae.flux2_latent_std.shape) == (1, 128, 1, 1)
    assert "decoder.up.0.block.0.nin_shortcut.weight" in sd and "decoder.mid.attn_1.q.weight" in sd
    assert tuple(sd["decoder.mid.attn_1.q.weight"].shape) == (384, 384, 1, 1)


def test_full_ldm_vae_still_loads_with_the_encoder_widths():
    vae, _ = _strict_vae_load("vae_full")
    assert vae.config.decoder_channels is None and vae.config.base_channels == 128
    assert vae.decoder.conv_in.weight.shape[0] == 512


def test_diffusers_conversion_keeps_the_encoder_and_flips_the_decoder_levels():
    sd = _vae_sd("vae_small")
    out = convert_diffusers_vae_state_dict(sd)
    assert out["decoder.up.3.block.0.conv1.weight"].shape == sd["decoder.up_blocks.0.resnets.0.conv1.weight"].shape
    assert out["decoder.up.0.block.0.nin_shortcut.weight"].shape == (96, 192, 1, 1)
    assert out["encoder.down.1.block.0.nin_shortcut.weight"].shape == (256, 128, 1, 1)
    assert out["decoder.up.3.upsample.conv.weight"].shape == (384, 384, 3, 3)
    assert "decoder.up.0.upsample.conv.weight" not in out
    assert out["encoder.mid.attn_1.proj_out.weight"].shape == (512, 512, 1, 1)
    assert out["decoder.norm_out.weight"].shape == (96,)


def test_decoder_widths_are_inferred_and_round_trip():
    for dec in (None, (32, 64)):
        cfg = VAEConfig(base_channels=64, channel_mult=(1, 2), num_res_blocks=1, z_channels=4,
                        use_quant_conv=False, decoder_channels=dec)
        sd = AutoencoderKL(cfg).state_dict()
        got = _infer_vae_config(sd, scale_factor=1.0, shift_factor=0.0)
        assert got.decoder_channels == dec and got.base_channels == 64 and got.channel_mult == (1, 2)


def test_small_decoder_decodes():
    cfg = VAEConfig(base_channels=64, channel_mult=(1, 2), num_res_blocks=1, z_channels=4, use_quant_conv=False,
                    decoder_channels=(32, 64))
    out = AutoencoderKL(cfg).decode(torch.randn(1, 4, 4, 4))
    assert out.shape == (1, 3, 8, 8)


# ─── text encoder loading ────────────────────────────────────────────────────

_TE_PREFIX = "model.language_model."


def _te_cfg() -> Qwen35Config:
    return Qwen35Config(vocab_size=300, hidden_size=256, intermediate_size=256, output_dim=256,
                        output_projection=False, num_hidden_layers=4, self_attn_layers=(3,), no_mlp_layers=(),
                        num_attention_heads=4, num_key_value_heads=2, head_dim=64, ssm_d_ssm=256,
                        ssm_conv_dim=512, ssm_n_groups=4, ssm_head_dim=64, ssm_d_state=128)


def _random_te(seed: int = 0):
    torch.manual_seed(seed)
    model = Qwen35TextEncoder(_te_cfg()).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(0.05 * torch.randn_like(p) if p.ndim > 1 or name.endswith("A_log") else 0.1 * torch.randn_like(p))
    return model


def _junk() -> dict:
    return {"model.visual.patch_embed.proj.weight": torch.randn(4, 3), "mtp.fc.weight": torch.randn(4, 4),
            "lm_head.weight": torch.randn(300, 256)}


class _SpyOpen:
    """``safe_open`` that records every tensor key actually read."""

    reads: list = []

    def __init__(self, *args, **kwargs):
        from safetensors import safe_open
        self._cm = safe_open(*args, **kwargs)

    def __enter__(self):
        self._f = self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._f, name)

    def get_tensor(self, key):
        _SpyOpen.reads.append(key)
        return self._f.get_tensor(key)


@pytest.fixture
def spy_reads(monkeypatch):
    import diffucore.loading.state_dict as sdmod
    _SpyOpen.reads = []
    monkeypatch.setattr(sdmod, "safe_open", _SpyOpen)
    return _SpyOpen.reads


def _cpu_policy(dtype=torch.float32) -> DevicePolicy:
    return DevicePolicy(device=torch.device("cpu"), compute_dtype=dtype, vae_dtype=torch.float32)


def test_bf16_te_loads_without_reading_vision_mtp_or_lm_head(tmp_path, spy_reads):
    model = _random_te()
    save_file({**{_TE_PREFIX + k: v for k, v in model.state_dict().items()}, **_junk()}, str(tmp_path / "te.safetensors"))
    prefix, hidden = _moss_te_layout(str(tmp_path / "te.safetensors"))
    assert (prefix, hidden) == (_TE_PREFIX, 256)
    encoder, _ = _moss_text_encoder(str(tmp_path / "te.safetensors"), prefix, _cpu_policy())
    assert set(spy_reads) == {_TE_PREFIX + k for k in model.state_dict()}
    ids = torch.randint(0, 300, (1, 9))
    with torch.no_grad():
        assert torch.equal(encoder(ids), model(ids))


def test_int8_te_with_int8_embedding_matches_its_dequantized_weights(tmp_path, spy_reads):
    model = _random_te()
    sd, deq = {}, {}
    for k, v in model.state_dict().items():
        mod = k.rsplit(".", 1)[0]
        quant = k.endswith(".weight") and v.ndim == 2 and not mod.endswith(("in_proj_a", "in_proj_b"))
        if quant:
            q, s = quantize_weight(v, 256)
            sd.update({_TE_PREFIX + k: q, _TE_PREFIX + mod + ".weight_scale": s.unsqueeze(1),
                       _TE_PREFIX + mod + ".comfy_quant": _TAG.clone()})
            deq[k] = dequantize_weight(q, s, 256)
        else:
            sd[_TE_PREFIX + k] = deq[k] = v
    save_file({**sd, **_junk(), "lm_head.weight_scale": torch.ones(300, 1), "lm_head.comfy_quant": _TAG.clone()},
              str(tmp_path / "te.safetensors"))
    encoder, _ = _moss_text_encoder(str(tmp_path / "te.safetensors"), _TE_PREFIX, _cpu_policy())
    assert isinstance(encoder.embed_tokens, Int8Embedding)
    assert not any(k.startswith(("model.visual", "mtp.", "lm_head")) for k in spy_reads)
    want = Qwen35TextEncoder(_te_cfg()).eval()
    want.load_state_dict(deq, strict=True)
    ids = torch.randint(0, 300, (1, 9))
    with torch.no_grad():
        got, expect = encoder(ids), want(ids)
    assert (got - expect).abs().max() < 1e-3 * expect.abs().max()


def test_a_non_qwen35_encoder_is_rejected(tmp_path):
    save_file({"model.embed_tokens.weight": torch.zeros(10, 8), "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8)},
              str(tmp_path / "te.safetensors"))
    prefix, hidden = _moss_te_layout(str(tmp_path / "te.safetensors"))
    assert hidden == 8
    with pytest.raises(ValueError, match="Qwen3.5"):
        _moss_text_encoder(str(tmp_path / "te.safetensors"), prefix, _cpu_policy())
    with pytest.raises(ValueError, match="no language model"):
        save_file({"x.weight": torch.zeros(1)}, str(tmp_path / "x.safetensors"))
        _moss_te_layout(str(tmp_path / "x.safetensors"))


# ─── DiT loading: residual scale ─────────────────────────────────────────────


def _save_tiny_dit(path, metadata=None, official=False):
    _, ours = _tiny_pair()
    sd = dict(ours.state_dict())
    if official:
        for i in range(3):
            base = f"layers.{i}.proj."
            q, k, v = sd.pop(base + "qkv.weight").chunk(3)
            sd.update({base + "q.weight": q, base + "k.weight": k, base + "v.weight": v})
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path), metadata=metadata)
    return ours


def test_residual_scale_is_only_applied_to_fp16_without_a_prescaled_file(tmp_path):
    quiet = lambda *a, **k: None  # noqa: E731
    plain, fixed = tmp_path / "plain.safetensors", tmp_path / "fixed.safetensors"
    _save_tiny_dit(plain)
    _save_tiny_dit(fixed, {"kohakumoss_fp16_fix": json.dumps({"c": 0.0625, "via": "gate", "eps": "stock"})})
    fp16, fp32 = _cpu_policy(torch.float16), _cpu_policy(torch.float32)
    assert load_moss_dit(str(plain), fp16, quiet).residual_scale == 0.0625
    assert load_moss_dit(str(plain), fp32, quiet).residual_scale == 1.0
    assert load_moss_dit(str(fixed), fp16, quiet).residual_scale == 1.0


def test_official_shards_and_the_fused_file_load_to_the_same_weights(tmp_path):
    ours = _save_tiny_dit(tmp_path / "fused.safetensors")
    _save_tiny_dit(tmp_path / "official.safetensors", official=True)
    a = load_moss_dit(str(tmp_path / "fused.safetensors"), _cpu_policy(), lambda *a, **k: None)
    b = load_moss_dit(str(tmp_path / "official.safetensors"), _cpu_policy(), lambda *a, **k: None)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
        assert torch.equal(va, ours.state_dict()[ka])


# ─── end to end, tiny ────────────────────────────────────────────────────────


class _StubTokenizer:
    def __call__(self, prompt):
        n = 4 + len(prompt) % 5
        return MossTokenized(torch.arange(1, n + 1)[None] * 7, False)


def _tiny_bundle(cache=None) -> ModelBundle:
    _, dit = _tiny_pair()
    cfg = Qwen35Config(vocab_size=512, hidden_size=64, intermediate_size=128, output_dim=64,
                       output_projection=False, num_hidden_layers=4, self_attn_layers=(1, 3), no_mlp_layers=(),
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, ssm_d_ssm=32,
                       ssm_conv_dim=96, ssm_n_groups=4, ssm_head_dim=8, ssm_d_state=8)
    torch.manual_seed(3)
    te = Qwen35TextEncoder(cfg).eval()
    vae = AutoencoderKL(VAEConfig(base_channels=32, channel_mult=(1, 2), num_res_blocks=1, z_channels=32,
                                  scale_factor=1.0, use_quant_conv=False)).eval()
    vae.register_buffer("flux2_latent_mean", torch.zeros(1, 128, 1, 1), persistent=False)
    vae.register_buffer("flux2_latent_std", torch.ones(1, 128, 1, 1), persistent=False)
    spec = ModelSpec(architecture="moss_image", prediction="flow", zero_terminal_snr=False, latent_channels=128,
                     context_dim=64, latent_scale=1.0)
    return ModelBundle(spec=spec, schedule=None, tokenizer=_StubTokenizer(), text_encoder=te, backbone=dit, vae=vae,
                       policy=_cpu_policy(), cond_cache=cache)


def _spy_forward(bundle):
    calls = []
    original = bundle.backbone.forward

    def forward(x, mod, head, text, skip_layers=()):
        calls.append((text.shape[1], tuple(skip_layers)))
        return original(x, mod, head, text, skip_layers)

    bundle.backbone.forward = forward
    return calls


def test_text_to_image_runs_cfg_with_the_uncond_branch_skipping_blocks():
    bundle = _tiny_bundle()
    calls = _spy_forward(bundle)
    image = moss_text_to_image(bundle, "a cat", "bad", steps=3, cfg_scale=3.0, width=64, height=64, seed=1,
                               uncond_skip_layers=(1,))
    assert image.size[0] > 0
    assert len(calls) == 6 and [c[1] for c in calls] == [(), (1,)] * 3
    assert calls[0][0] != calls[1][0]   # cond and uncond run at their own text lengths


def test_default_skip_layer_is_dropped_on_a_shallow_net_and_cfg_one_skips_uncond():
    bundle = _tiny_bundle()
    calls = _spy_forward(bundle)
    moss_text_to_image(bundle, "a cat", "", steps=2, cfg_scale=3.0, width=64, height=64, seed=1)
    assert [c[1] for c in calls] == [(), ()] * 2
    calls.clear()
    moss_text_to_image(bundle, "a cat", "", steps=2, cfg_scale=1.0, width=64, height=64, seed=1)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="uncond_skip_layers"):
        moss_text_to_image(bundle, "a cat", "", steps=2, width=64, height=64, uncond_skip_layers=(9,))


def test_cfg_interval_limits_the_uncond_forwards():
    bundle = _tiny_bundle()
    calls = _spy_forward(bundle)
    moss_text_to_image(bundle, "a cat", "", steps=4, cfg_scale=3.0, cfg_interval_start=0.5, width=64, height=64,
                       seed=1, uncond_skip_layers=())
    assert len(calls) == 4 + 2


def test_seeded_runs_repeat_and_the_conditioning_cache_skips_the_encoder():
    cache = ConditioningCache()
    bundle = _tiny_bundle(cache)
    encodes = []
    te_forward = bundle.text_encoder.forward
    bundle.text_encoder.forward = lambda ids, *a, **k: (encodes.append(1), te_forward(ids, *a, **k))[1]
    first = moss_text_to_image(bundle, "a cat", "bad", steps=2, width=64, height=64, seed=5, uncond_skip_layers=())
    assert len(encodes) == 2
    second = moss_text_to_image(bundle, "a cat", "bad", steps=2, width=64, height=64, seed=5, uncond_skip_layers=())
    assert len(encodes) == 2 and first.tobytes() == second.tobytes()


def test_text_to_image_dispatches_on_the_architecture_and_falls_back_to_moss_schedule():
    bundle = _tiny_bundle()
    image, info = TextToImage(bundle)("a cat", "", steps=2, cfg_scale=1.0, scheduler="karras", width=64, height=64,
                                      seed=2, return_info=True)
    assert image.size[0] > 0 and info.vae_decode_mode == "untiled"


def _tiny_files(tmp_path):
    """A loadable tiny checkpoint set: DiT (text_dim 256), Qwen3.5-style TE, FLUX.2-style VAE."""
    torch.manual_seed(0)
    dit = MossDiT(MossConfig(dim=256, ffn_dim=512, num_heads=2, num_layers=3, text_dim=256)).float()
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(tmp_path / "dit.safetensors"))
    save_file({_TE_PREFIX + k: v for k, v in _random_te().state_dict().items()}, str(tmp_path / "te.safetensors"))
    vae = AutoencoderKL(VAEConfig(base_channels=32, channel_mult=(1, 2), num_res_blocks=1, z_channels=32,
                                  use_quant_conv=False))
    save_file({**vae.state_dict(), "bn.running_mean": torch.zeros(128), "bn.running_var": torch.ones(128),
               "bn.num_batches_tracked": torch.zeros((), dtype=torch.int64)}, str(tmp_path / "vae.safetensors"))
    return [str(tmp_path / n) for n in ("dit.safetensors", "te.safetensors", "vae.safetensors")]


def test_load_moss_checkpoint_builds_a_bundle_that_generates(tmp_path):
    bundle = load_moss_checkpoint(*_tiny_files(tmp_path), policy=_cpu_policy())
    assert bundle.spec.architecture == "moss_image" and isinstance(bundle.backbone, MossDiT)
    assert isinstance(bundle.tokenizer, MossTokenizer) and bundle.vae.flux2_latent_std.shape == (1, 128, 1, 1)
    bundle.tokenizer = _StubTokenizer()   # the tiny encoder has a 300-token vocab
    image = TextToImage(bundle)("a cat", "", steps=2, cfg_scale=3.0, width=64, height=64, seed=1)
    assert image.size[0] > 0


@pytest.mark.parametrize("encoder", ["qwen35_4b_header.json", "qwen35_08b_base_header.json"])
def test_a_smaller_qwen35_is_rejected_by_width(tmp_path, encoder):
    hdr = json.loads(Path(__file__).with_name(encoder).read_text())
    key = next(k for k in hdr if k.endswith("embed_tokens.weight"))
    dit, _, vae = _tiny_files(tmp_path)
    save_file({key: torch.zeros(2, hdr[key][-1])}, str(tmp_path / "te.safetensors"))
    with pytest.raises(ValueError, match="needs Qwen3.5-9B"):
        load_moss_checkpoint(dit, str(tmp_path / "te.safetensors"), vae, policy=_cpu_policy())


def test_the_dit_file_must_be_moss(tmp_path):
    dit, te, vae = _tiny_files(tmp_path)
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}, str(tmp_path / "unet.safetensors"))
    with pytest.raises((ValueError, NotImplementedError)):
        load_moss_checkpoint(str(tmp_path / "unet.safetensors"), te, vae, policy=_cpu_policy())


# ─── img2img and inpaint, tiny ───────────────────────────────────────────────

from PIL import Image  # noqa: E402

from diffucore.pipelines import ImageToImage, Inpaint  # noqa: E402
from diffucore.pipelines._flux import _flux2_patchify_latents, _flux2_unpatchify_latents  # noqa: E402
from diffucore.pipelines._moss import _encode_image, moss_img2img  # noqa: E402


def _tiny_i2i_bundle(cache=None) -> ModelBundle:
    """The tiny bundle with an f8 VAE, so a 64x64 image is a 4x4 token grid."""
    bundle = _tiny_bundle(cache)
    torch.manual_seed(5)
    vae = AutoencoderKL(VAEConfig(base_channels=32, channel_mult=(1, 1, 1, 1), num_res_blocks=1, z_channels=32,
                                  scale_factor=1.0, use_quant_conv=False)).eval()
    vae.register_buffer("flux2_latent_mean", torch.zeros(1, 128, 1, 1), persistent=False)
    vae.register_buffer("flux2_latent_std", torch.ones(1, 128, 1, 1), persistent=False)
    bundle.vae = vae
    return bundle


def _init_image() -> Image.Image:
    g = torch.Generator().manual_seed(9)
    return Image.fromarray((torch.rand(64, 64, 3, generator=g) * 255).byte().numpy())


def test_encode_is_the_exact_inverse_of_the_decode_bridge(monkeypatch):
    bundle = _tiny_i2i_bundle()
    z = torch.randn(1, 32, 8, 8)
    bundle.vae.flux2_latent_mean = torch.randn(1, 128, 1, 1)
    bundle.vae.flux2_latent_std = torch.rand(1, 128, 1, 1) + 0.5
    seen = {}

    def encode(pixels, sample=True):
        seen["sample"] = sample
        return z

    monkeypatch.setattr(bundle.vae, "encode", encode)
    lat = _encode_image(bundle, _init_image(), 64, 64)
    assert seen["sample"] is False and lat.shape == (1, 128, 4, 4)
    back = _flux2_unpatchify_latents(lat * bundle.vae.flux2_latent_std + bundle.vae.flux2_latent_mean)
    assert torch.allclose(back, z, atol=1e-6)


def test_img2img_runs_the_last_steps_of_a_longer_schedule():
    bundle = _tiny_i2i_bundle()
    sigmas_seen = []
    time_vectors = bundle.backbone.time_vectors
    bundle.backbone.time_vectors = lambda t: (sigmas_seen.append(float(t[0])), time_vectors(t))[1]
    image = ImageToImage(bundle)("a cat", _init_image(), "", strength=0.5, steps=4, cfg_scale=1.0, width=64,
                                 height=64, seed=3)
    assert image.size == (64, 64)
    assert sigmas_seen == pytest.approx(moss_sigmas(8)[-5:-1].tolist())


def test_img2img_with_full_strength_matches_the_whole_schedule():
    bundle = _tiny_i2i_bundle()
    seen = []
    time_vectors = bundle.backbone.time_vectors
    bundle.backbone.time_vectors = lambda t: (seen.append(float(t[0])), time_vectors(t))[1]
    moss_img2img(bundle, "a cat", _init_image(), strength=1.0, steps=5, cfg_scale=1.0, width=64, height=64, seed=3)
    assert seen == pytest.approx(moss_sigmas(5)[:-1].tolist())


def test_img2img_shares_cfg_skip_layers_and_the_conditioning_cache_with_text_to_image():
    cache = ConditioningCache()
    bundle = _tiny_i2i_bundle(cache)
    calls = _spy_forward(bundle)
    encodes = []
    te_forward = bundle.text_encoder.forward
    bundle.text_encoder.forward = lambda ids, *a, **k: (encodes.append(1), te_forward(ids, *a, **k))[1]
    moss_text_to_image(bundle, "a cat", "bad", steps=2, cfg_scale=3.0, width=64, height=64, seed=1,
                       uncond_skip_layers=(1,))
    assert len(encodes) == 2
    calls.clear()
    ImageToImage(bundle)("a cat", _init_image(), "bad", strength=0.5, steps=2, cfg_scale=3.0, width=64, height=64,
                         seed=1, uncond_skip_layers=(1,))
    assert len(encodes) == 2   # cond and uncond ctx came from the cache
    assert [c[1] for c in calls] == [(), (1,)] * 2
    with pytest.raises(ValueError, match="uncond_skip_layers"):
        ImageToImage(bundle)("a cat", _init_image(), "bad", steps=2, width=64, height=64, uncond_skip_layers=(7,))


def test_img2img_rejects_a_bad_strength_and_a_size_off_the_token_grid():
    bundle = _tiny_i2i_bundle()
    with pytest.raises(ValueError, match="strength"):
        ImageToImage(bundle)("a cat", _init_image(), steps=2, strength=0.0, width=64, height=64)
    with pytest.raises(ValueError, match="divisible by 16"):
        moss_img2img(bundle, "a cat", _init_image(), steps=2, width=72, height=64)


def test_inpaint_keeps_the_masked_out_pixels_and_repaints_the_rest():
    bundle = _tiny_i2i_bundle()
    init = _init_image()
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (32, 0, 64, 64))     # right half repainted
    out = Inpaint(bundle)("a cat", init, mask, "", strength=1.0, steps=3, cfg_scale=1.0, width=64, height=64, seed=4)
    got, want = np.asarray(out), np.asarray(init)
    assert np.array_equal(got[:, :32], want[:, :32])
    assert not np.array_equal(got[:, 32:], want[:, 32:])


def test_inpaint_pins_the_keep_tokens_to_the_init_latent_at_every_step():
    bundle = _tiny_i2i_bundle()
    init = _init_image()
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (32, 0, 64, 64))
    z0 = _encode_image(bundle, init, 64, 64)
    estimates = []
    from diffucore.sampling import get_sampler
    import diffucore.pipelines._moss as moss_mod

    def spy(name):
        sampler = get_sampler(name)

        def run(denoise, x, sigmas, **kw):
            def wrapped(x_in, s):
                x0 = denoise(x_in, s)
                estimates.append(x0)
                return x0
            return sampler(wrapped, x, sigmas, **kw)
        return run

    moss_mod.get_sampler = spy
    try:
        moss_img2img(bundle, "a cat", init, mask_image=mask, strength=1.0, steps=3, cfg_scale=1.0, width=64,
                     height=64, seed=4)
    finally:
        moss_mod.get_sampler = get_sampler
    assert len(estimates) == 3
    for x0 in estimates:
        assert torch.equal(x0[..., :1], z0[..., :1])     # the soft mask edge spreads over the next column


# ─── LoRA: forward-hook side branches ────────────────────────────────────────

from types import SimpleNamespace  # noqa: E402

from diffucore.lora import apply_lora, clear_loras, remove_lora  # noqa: E402
from diffucore.models._int8 import Int8Linear, quantize_linears  # noqa: E402

_LORA_HDR = json.loads(Path(__file__).with_name("moss_lora_headers.json").read_text())
_LORA_RANK, _LORA_ALPHA = 4, 8.0
# Reference-layout module names (q/k/v apart) the synthetic LoRAs touch.
_LORA_MODULES = ["layers.0.proj.q", "layers.0.proj.k", "layers.0.proj.v", "layers.0.proj.o",
                 "layers.1.proj.ffn_up_gate", "layers.2.proj.ffn_down", "text_embedder.blocks.0.qkv",
                 "text_embedder.blocks.1.ffn_gate", "text_embedder.proj", "video_embedder"]


def _lora_bundle(model, cache=None, policy=None):
    return SimpleNamespace(spec=SimpleNamespace(architecture="moss_image"), backbone=model, cond_cache=cache,
                           policy=policy or DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32))


def _make_factors(rmodel, seed=1):
    torch.manual_seed(seed)
    out = {}
    for name in _LORA_MODULES:
        w = rmodel.get_submodule(name).weight
        out[name] = (torch.randn(_LORA_RANK, w.shape[1]) * 0.1, torch.randn(w.shape[0], _LORA_RANK) * 0.1)
    return out


def _write_format(path, factors, fmt):
    tensors, meta = {}, {}
    for name, (down, up) in factors.items():
        if fmt in ("bare", "transformer", "comfy"):
            base = {"bare": "", "transformer": "transformer.", "comfy": "diffusion_model."}[fmt] + name
            tensors[base + ".lora_A.weight"], tensors[base + ".lora_B.weight"] = down, up
        else:
            base = "diffusion_model." + name if fmt == "kohya_dotted" else "lora_unet_" + name.replace(".", "_")
            tensors[base + ".lora_down.weight"], tensors[base + ".lora_up.weight"] = down, up
            tensors[base + ".alpha"] = torch.tensor(_LORA_ALPHA)
    if fmt == "bare":
        meta = {"alpha": str(int(_LORA_ALPHA)), "rank": str(_LORA_RANK)}
    elif fmt in ("transformer", "comfy"):
        meta = {"lora_adapter_metadata": json.dumps({"r": _LORA_RANK, "lora_alpha": _LORA_ALPHA})}
    elif fmt == "kohya_dotted":
        meta = {"alpha": "99"}      # the per-module .alpha tensor wins over the file's
    save_file(tensors, str(path), metadata=meta or None)


def _reference_with_lora(rmodel, factors, multiplier):
    scale = _LORA_ALPHA / _LORA_RANK * multiplier
    with torch.no_grad():
        for name, (down, up) in factors.items():
            rmodel.get_submodule(name).weight += scale * up @ down


@pytest.mark.parametrize("fmt", ["bare", "transformer", "comfy", "kohya_dotted", "kohya_mangled"])
def test_lora_hooks_match_the_separate_qkv_reference_in_every_format(tmp_path, fmt):
    rmodel, ours = _tiny_pair()
    _, plain = _tiny_pair()
    factors = _make_factors(rmodel)
    _write_format(tmp_path / "l.safetensors", factors, fmt)
    torch.manual_seed(2)
    x, feats, sigma = torch.randn(1, 128, 4, 6), torch.randn(1, 7, 64), torch.tensor([0.4])

    def run(model):
        mod, head = model.time_vectors(sigma)
        return model(x, mod, head, model.encode_text(feats), (1,))

    with torch.no_grad():
        base = run(ours)
        report = apply_lora(_lora_bundle(ours), str(tmp_path / "l.safetensors"), 0.7)
        assert report.applied == len(_LORA_MODULES) and report.unmatched == []
        got = run(ours)
        _reference_with_lora(rmodel, factors, 0.7)
        want = rmodel(x, sigma * 1000.0, feats, skip_layers=(1,)).sample
    assert not torch.allclose(got, base, atol=1e-3)
    assert (got - want).abs().max() <= 1e-4
    assert torch.equal(run(plain), base)


def test_qkv_slices_equal_three_separate_lora_applications(tmp_path):
    rmodel, ours = _tiny_pair()
    factors = {k: v for k, v in _make_factors(rmodel).items() if k.startswith("layers.0.proj.") and k[-1] in "qkv"}
    _write_format(tmp_path / "l.safetensors", factors, "bare")
    layer = ours.layers[0].proj["qkv"]
    x = torch.randn(1, 9, 256)
    with torch.no_grad():
        before = layer(x)
        apply_lora(_lora_bundle(ours), str(tmp_path / "l.safetensors"), 1.0)
        delta = layer(x) - before
        for i, c in enumerate("qkv"):
            down, up = factors["layers.0.proj." + c]
            assert torch.allclose(delta[..., i * 256:(i + 1) * 256], 2.0 * (x @ down.T) @ up.T, atol=1e-5)


def test_lora_applies_to_int8_layers(tmp_path):
    rmodel, ours = _tiny_pair()
    quantize_linears(ours, int8_layer_names(3), torch.device("cpu"))
    factors = {k: v for k, v in _make_factors(rmodel).items() if k.startswith(("layers.0.proj.", "layers.1."))}
    _write_format(tmp_path / "l.safetensors", factors, "comfy")
    qkv, up_gate = ours.layers[0].proj["qkv"], ours.layers[1].proj["ffn_up_gate"]
    assert isinstance(qkv, Int8Linear) and isinstance(up_gate, Int8Linear)
    x = torch.randn(1, 6, 256)
    with torch.no_grad():
        before_qkv, before_ug = qkv(x), up_gate(x)
        report = apply_lora(_lora_bundle(ours), str(tmp_path / "l.safetensors"), 1.0)
        assert report.applied == 5 and report.unmatched == []
        d = qkv(x) - before_qkv
        down, up = factors["layers.0.proj.v"]
        assert torch.allclose(d[..., 512:], 2.0 * (x @ down.T) @ up.T, atol=1e-4)
        down, up = factors["layers.1.proj.ffn_up_gate"]
        assert torch.allclose(up_gate(x) - before_ug, 2.0 * (x @ down.T) @ up.T, atol=1e-4)


def test_lora_apply_remove_and_clear_invalidate_the_conditioning_cache(tmp_path):
    rmodel, ours = _tiny_pair()
    _write_format(tmp_path / "l.safetensors", _make_factors(rmodel), "bare")
    cache = ConditioningCache()
    bundle = _lora_bundle(ours, cache)
    feats = torch.randn(1, 5, 64)
    base = ours.encode_text(feats)

    def put():
        cache.put(("moss", "p", ""), {"cond_ctx": base, "uncond_ctx": None})

    put()
    apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
    assert cache.get(("moss", "p", "")) is None
    assert not torch.allclose(ours.encode_text(feats), base, atol=1e-4)   # the refiner sees the LoRA
    put()
    remove_lora(bundle, str(tmp_path / "l.safetensors"))
    assert cache.get(("moss", "p", "")) is None and torch.equal(ours.encode_text(feats), base)
    apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
    put()
    clear_loras(bundle)
    assert cache.get(("moss", "p", "")) is None and torch.equal(ours.encode_text(feats), base)
    assert not any(m._forward_hooks for m in ours.modules())


def test_lora_stacks_and_removes_in_order(tmp_path):
    rmodel, ours = _tiny_pair()
    f1, f2 = _make_factors(rmodel, 1), _make_factors(rmodel, 2)
    _write_format(tmp_path / "a.safetensors", f1, "bare")
    _write_format(tmp_path / "b.safetensors", f2, "comfy")
    bundle = _lora_bundle(ours)
    layer, x = ours.layers[0].proj["o"], torch.randn(1, 4, 256)
    with torch.no_grad():
        base = layer(x)
        apply_lora(bundle, str(tmp_path / "a.safetensors"), 1.0)
        only_a = layer(x)
        apply_lora(bundle, str(tmp_path / "b.safetensors"), 1.0)
        remove_lora(bundle, str(tmp_path / "b.safetensors"))
        assert torch.equal(layer(x), only_a)
        remove_lora(bundle, str(tmp_path / "a.safetensors"))
        assert torch.equal(layer(x), base)
    with pytest.raises(ValueError, match="not in the active LoRA stack"):
        remove_lora(bundle, str(tmp_path / "a.safetensors"))


def test_lora_that_changes_a_weights_shape_is_rejected_and_hooks_nothing(tmp_path):
    rmodel, ours = _tiny_pair()
    factors = _make_factors(rmodel)
    tensors = {}
    for name, (down, up) in factors.items():
        tensors[name + ".lora_A.weight"], tensors[name + ".lora_B.weight"] = down, up
    tensors["video_embedder.weight"] = torch.zeros(256, 384)     # the control adapter's widened input
    tensors["video_embedder.bias"] = torch.zeros(256)
    save_file(tensors, str(tmp_path / "control.safetensors"))
    with pytest.raises(ValueError, match="changes the shape of video_embedder.weight"):
        apply_lora(_lora_bundle(ours), str(tmp_path / "control.safetensors"), 1.0)
    # a factor that doesn't fit its layer is just as wrong
    bad = {"layers.0.proj.o.lora_A.weight": torch.zeros(4, 384), "layers.0.proj.o.lora_B.weight": torch.zeros(256, 4)}
    save_file(bad, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="doesn't fit the layer"):
        apply_lora(_lora_bundle(ours), str(tmp_path / "bad.safetensors"), 1.0)
    assert not any(m._forward_hooks for m in ours.modules())


def test_unknown_modules_are_reported_not_hooked(tmp_path):
    rmodel, ours = _tiny_pair()
    save_file({"layers.0.proj.o.lora_A.weight": torch.zeros(4, 256), "layers.0.proj.o.lora_B.weight": torch.zeros(256, 4),
               "layers.9.proj.o.lora_A.weight": torch.zeros(4, 256), "layers.9.proj.o.lora_B.weight": torch.zeros(256, 4),
               "layers.0.proj.o.lokr_w1": torch.zeros(2, 2)}, str(tmp_path / "l.safetensors"))
    report = apply_lora(_lora_bundle(ours), str(tmp_path / "l.safetensors"), 1.0)
    assert report.applied == 1 and sorted(report.unmatched) == ["layers.9.proj.o"]


def _write_header_lora(path, name, drop=()):
    entry = _LORA_HDR[name]
    save_file({k: torch.zeros(s, dtype=torch.bfloat16) for k, (_, s) in entry["tensors"].items()
               if not k.startswith(tuple(drop))}, str(path), metadata=entry["metadata"])


def _full_size_bundle():
    with torch.device("meta"):
        return _lora_bundle(MossDiT(MossConfig()))


@pytest.mark.parametrize("name", ["pixelart", "comfy", "diffusers"])
def test_real_lora_headers_map_with_zero_unmapped(tmp_path, name):
    _write_header_lora(tmp_path / "l.safetensors", name)
    report = apply_lora(_full_size_bundle(), str(tmp_path / "l.safetensors"), 1.0)
    assert report.unmatched == [] and report.applied == 12      # q k v o up_gate down on layers 0 and 41


def test_the_control_lora_maps_its_plain_part_but_is_rejected_for_the_widened_embedder(tmp_path):
    _write_header_lora(tmp_path / "control.safetensors", "control")
    with pytest.raises(ValueError, match=r"changes the shape of video_embedder.weight \(3072, 128\) -> \(3072, 384\)"):
        apply_lora(_full_size_bundle(), str(tmp_path / "control.safetensors"), 1.0)
    _write_header_lora(tmp_path / "plain.safetensors", "control", drop=("video_embedder",))
    report = apply_lora(_full_size_bundle(), str(tmp_path / "plain.safetensors"), 1.0)
    assert report.unmatched == [] and report.applied == 12 + 10    # plus the text embedder's two blocks


# ─── official text_encoder folder ────────────────────────────────────────────


def _official_te_shards(folder, n_shards=2):
    """The official ``text_encoder/`` layout: shards + index, with the vision
    tower / MTP / lm_head riding along."""
    folder.mkdir(parents=True, exist_ok=True)
    sd = {_TE_PREFIX + k: v for k, v in _random_te().state_dict().items()}
    keys = sorted(sd)
    weight_map = {}
    for i in range(n_shards):
        part = {k: sd[k] for k in keys[i::n_shards]}
        if i == 0:
            part.update(_junk())
        name = f"model-0000{i + 1}-of-0000{n_shards}.safetensors"
        save_file(part, str(folder / name))
        weight_map.update({k: name for k in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return sd


@pytest.mark.parametrize("pick", ["folder", "repo_root", "index"])
def test_the_official_text_encoder_folder_loads_in_every_spelling(tmp_path, pick, spy_reads):
    sd = _official_te_shards(tmp_path / "repo" / "text_encoder")
    path = {"folder": tmp_path / "repo" / "text_encoder", "repo_root": tmp_path / "repo",
            "index": tmp_path / "repo" / "text_encoder" / "model.safetensors.index.json"}[pick]
    prefix, hidden = _moss_te_layout(str(path))
    assert (prefix, hidden) == (_TE_PREFIX, 256)
    encoder, _ = _moss_text_encoder(str(path), prefix, _cpu_policy())
    assert set(spy_reads) == set(sd)     # no vision / mtp / lm_head read, nothing missed across shards
    ids = torch.randint(0, 300, (1, 9))
    with torch.no_grad():
        assert torch.equal(encoder(ids), _random_te()(ids))


def test_load_moss_checkpoint_takes_an_official_text_encoder_folder(tmp_path):
    dit, _, vae = _tiny_files(tmp_path)
    _official_te_shards(tmp_path / "text_encoder")
    bundle = load_moss_checkpoint(dit, str(tmp_path), vae, policy=_cpu_policy())
    bundle.tokenizer = _StubTokenizer()
    assert TextToImage(bundle)("a cat", "", steps=2, cfg_scale=3.0, width=64, height=64, seed=1).size[0] > 0
