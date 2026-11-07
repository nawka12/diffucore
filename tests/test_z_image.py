"""Z-Image (Turbo + Base) verification, offline, CPU.

1. The DiT, VAE and text-encoder loaders match the real files' keys and shapes
   (``z_image_headers.json``: ComfyUI bf16 / INT8 ConvRot DiT, the diffusers
   transformer, Qwen3-4B, real community LoRAs).
2. A tiny random DiT matches the reference ``ZImageTransformer2DModel``
   (``_zimage_reference.py``, diffusers 0.42 trimmed) in fp32 at several text
   lengths, image sizes (with and without image padding) and residual scales.
3. The sigma schedule equals the official scheduler's
   (``z_image_schedule.json``) and the tokenizer renders the reference's chat
   template (``z_image_tokenizer_cases.json``, ids from transformers).
4. LoRA hooks (every key spelling) match the reference with the delta merged.
5. A tiny end-to-end text-to-image / img2img / inpaint run exercises CFG, its
   normalization, the conditioning cache and the loaders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import _zimage_reference as ref  # noqa: E402

from diffucore.bundle import (  # noqa: E402
    ModelBundle, _build, _zimage_dit_state, _zimage_text_encoder, _zimage_variant, load_zimage_checkpoint,
    load_zimage_dit,
)
from diffucore.conditioning import ZImageTokenizer  # noqa: E402
from diffucore.conditioning.z_image_tokenizer import MAX_LENGTH, ZImageTokenized, render_prompt  # noqa: E402
from diffucore.loading import ModelSpec, detect_architecture  # noqa: E402
from diffucore.lora import _zimage_targets, apply_lora, clear_loras, remove_lora  # noqa: E402
from diffucore.models import AutoencoderKL, Qwen3Config, Qwen3TextEncoder, VAEConfig, ZImageConfig, ZImageDiT  # noqa: E402
from diffucore.models._int8 import adopt_int8_layers, int8_layers  # noqa: E402
from diffucore.models.z_image_dit import from_diffusers, from_diffusers_name, int8_layer_names  # noqa: E402
from diffucore.pipelines import ImageToImage, Inpaint, TextToImage  # noqa: E402
from diffucore.pipelines import _zimage as zmod  # noqa: E402
from diffucore.pipelines._zimage import ZIMAGE_DEFAULTS, zimage_sigmas, zimage_text_to_image  # noqa: E402
from diffucore.runtime import ConditioningCache, DevicePolicy  # noqa: E402

_HERE = Path(__file__).parent
_FIX = json.loads((_HERE / "z_image_headers.json").read_text())
_TAG = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                                     "convrot_groupsize": 256}).encode()), dtype=torch.uint8)
_OFFICIAL = Path("/home/kayfa/diffucore-weights/hf/Tongyi-MAI/Z-Image-Turbo")


def _expand(entry: dict) -> dict:
    """``{key: (dtype, shape)}`` from a fixture entry (blocks stored as templates)."""
    out = {k: (v[0], tuple(v[1])) for k, v in entry["global"].items()}
    for prefix, b in entry["blocks"].items():
        for i in range(b["n"]):
            out.update({f"{prefix}.{i}.{k}": (v[0], tuple(v[1])) for k, v in b["keys"].items()})
    return out


def _shapes(hdr: dict) -> dict:
    return {k: s for k, (_, s) in hdr.items()}


def _meta_sd(hdr: dict) -> dict:
    """Meta tensors for a header; ``comfy_quant`` tags become real JSON tags."""
    return {k: (_TAG.clone() if k.endswith(".comfy_quant") else torch.empty(s, device="meta"))
            for k, s in _shapes(hdr).items()}


def _cpu_policy(dtype=torch.float32) -> DevicePolicy:
    return DevicePolicy(device=torch.device("cpu"), compute_dtype=dtype, vae_dtype=torch.float32)


# ─── real file layouts ───────────────────────────────────────────────────────


def test_detects_z_image_in_every_layout():
    for name in ("comfy_bf16", "comfy_int8_convrot_turbo", "diffusers"):
        spec = detect_architecture(_shapes(_expand(_FIX[name])))
        assert (spec.architecture, spec.prediction, spec.latent_channels, spec.context_dim, spec.image_size,
                spec.latent_scale, spec.latent_shift) == ("z_image", "flow", 16, 2560, 1024, 0.3611, 0.1159)
    prefixed = {"model.diffusion_model." + k: s for k, s in _shapes(_expand(_FIX["comfy_bf16"])).items()}
    assert detect_architecture(prefixed).architecture == "z_image"


def test_other_families_are_not_taken_for_z_image():
    data = json.loads((_HERE / "moss_headers.json").read_text())["comfy_bf16"]
    shapes = {k: tuple(v[1]) for k, v in data["global"].items()}
    for indices, keys in data["layers"]:
        shapes.update({f"{data['prefix']}{indices[0]}.{k}": tuple(v[1]) for k, v in keys.items()})
    assert detect_architecture(shapes).architecture == "moss_image"


def test_lumina2_is_rejected_not_taken_for_z_image():
    # Lumina 2 shares the pad tokens; its block modulation is SiLU + Linear.
    shapes = {k.replace(".adaLN_modulation.0.", ".adaLN_modulation.1."): s
              for k, s in _shapes(_expand(_FIX["comfy_bf16"])).items()}
    with pytest.raises(NotImplementedError, match="Lumina 2"):
        detect_architecture(shapes)


def test_config_is_derived_from_the_keys():
    for name in ("comfy_bf16", "comfy_int8_convrot_turbo"):
        assert ZImageConfig.from_state_dict(_meta_sd(_expand(_FIX[name]))) == ZImageConfig()


def test_dit_keys_and_shapes_match_the_comfy_files():
    model = _build(ZImageDiT, ZImageConfig())
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    bf16 = _shapes(_expand(_FIX["comfy_bf16"]))
    assert ours == bf16
    int8 = _expand(_FIX["comfy_int8_convrot_turbo"])
    assert {k for k in int8 if not k.endswith((".weight_scale", ".comfy_quant"))} == set(bf16)


def test_int8_file_loads_strictly_with_the_block_linears_and_modulations_quantized():
    sd = _meta_sd(_expand(_FIX["comfy_int8_convrot_turbo"]))
    cfg = ZImageConfig.from_state_dict(sd)
    model = _build(ZImageDiT, cfg)
    int8 = int8_layers(sd)
    modulations = {f"{a}.{i}.adaLN_modulation.0"
                   for a, n in (("layers", cfg.n_layers), ("noise_refiner", cfg.n_refiner_layers))
                   for i in range(n)}
    assert set(int8) == set(int8_layer_names(cfg)) | modulations and set(int8.values()) == {256}
    sd = adopt_int8_layers(model, sd, int8)
    model.load_state_dict(sd, strict=True, assign=True)


def test_diffusers_transformer_fuses_into_the_comfy_layout():
    sd = _meta_sd(_expand(_FIX["diffusers"]))
    fused = from_diffusers(dict(sd))
    assert {k: tuple(v.shape) for k, v in fused.items()} == _shapes(_expand(_FIX["comfy_bf16"]))


def test_fusion_stacks_q_k_v_in_that_order():
    q, k, v = (torch.full((4, 8), float(i)) for i in range(3))
    sd = {"layers.0.attention.to_k.weight": k, "layers.0.attention.to_out.0.weight": torch.zeros(1),
          "layers.0.attention.to_q.weight": q, "layers.0.attention.to_v.weight": v,
          "layers.0.attention.norm_q.weight": torch.zeros(2), "all_x_embedder.2-1.bias": torch.zeros(3)}
    fused = from_diffusers(sd)
    assert sorted(fused) == ["layers.0.attention.out.weight", "layers.0.attention.q_norm.weight",
                             "layers.0.attention.qkv.weight", "x_embedder.bias"]
    assert torch.equal(fused["layers.0.attention.qkv.weight"], torch.cat([q, k, v]))


@pytest.mark.parametrize("name, want", [
    ("layers.3.attention.to_out.0", "layers.3.attention.out"),
    ("all_final_layer.2-1.adaLN_modulation.1", "final_layer.adaLN_modulation.1"),
    ("all_x_embedder.2-1", "x_embedder"),
    ("layers.0.attention.norm_k.weight", "layers.0.attention.k_norm.weight"),
    ("layers.0.feed_forward.w1", "layers.0.feed_forward.w1"),
])
def test_diffusers_names_in_the_original_naming(name, want):
    assert from_diffusers_name(name) == want


def test_z_image_dit_state_reads_a_prefixed_diffusers_shard_and_rejects_omni(tmp_path):
    torch.manual_seed(0)
    q, k, v = (torch.randn(8, 4) for _ in range(3))
    pre = "model.diffusion_model."
    save_file({pre + "layers.0.attention.to_q.weight": q, pre + "layers.0.attention.to_k.weight": k,
               pre + "layers.0.attention.to_v.weight": v, pre + "all_x_embedder.2-1.weight": torch.zeros(2, 2),
               pre + "cap_pad_token": torch.zeros(1, 2), pre + "x_pad_token": torch.zeros(1, 2)},
              str(tmp_path / "a.safetensors"))
    sd = _zimage_dit_state(str(tmp_path / "a.safetensors"))
    assert sorted(sd) == ["cap_pad_token", "layers.0.attention.qkv.weight", "x_embedder.weight", "x_pad_token"]
    assert torch.equal(sd["layers.0.attention.qkv.weight"], torch.cat([q, k, v]))
    save_file({"cap_pad_token": torch.zeros(1, 2), "siglip_embedder.1.weight": torch.zeros(2, 2)},
              str(tmp_path / "omni.safetensors"))
    with pytest.raises(NotImplementedError, match="Omni"):
        _zimage_dit_state(str(tmp_path / "omni.safetensors"))


def test_qwen3_4b_file_has_the_layout_the_loader_expects():
    keys = _expand(_FIX["qwen3_4b"])
    assert keys["model.embed_tokens.weight"][1] == (151936, 2560)
    assert "lm_head.weight" not in keys and "model.layers.35.self_attn.q_norm.weight" in keys
    assert "model.layers.36.self_attn.q_norm.weight" not in keys


# ─── DiT against the reference ───────────────────────────────────────────────

_TINY = dict(dim=256, n_layers=3, n_refiner_layers=2, n_heads=2, cap_feat_dim=64, in_channels=16)


def _tiny_cfg() -> ZImageConfig:
    return ZImageConfig(dim=256, n_layers=3, n_refiner_layers=2, n_heads=2, ffn_dim=int(256 / 3 * 8),
                        cap_feat_dim=64, in_channels=16)


def _tiny_pair(seed: int = 0):
    torch.manual_seed(seed)
    rmodel = ref.ZImageTransformer2DModel(**_TINY).float().eval()
    with torch.no_grad():
        for name, p in rmodel.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif name.endswith("pad_token"):
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
    ours = ZImageDiT(_tiny_cfg()).float().eval()
    ours.load_state_dict(from_diffusers(dict(rmodel.state_dict())), strict=True)
    return rmodel, ours


def _reference_forward(rmodel, x, t, feats):
    return torch.stack(rmodel(list(x.unsqueeze(2).unbind(0)), t, list(feats.unbind(0)))).squeeze(2)


@pytest.mark.parametrize("scale", [(1.0, 1.0), (1 / 16, 1 / 256), (0.3, 0.7)])
@pytest.mark.parametrize("latent", [(12, 10), (16, 16), (8, 8)])
@pytest.mark.parametrize("text_len", [5, 32, 40])
def test_dit_matches_the_reference_forward(text_len, latent, scale):
    rmodel, ours = _tiny_pair()
    ours.attn_scale, ours.ffn_scale = scale
    torch.manual_seed(1)
    x = torch.randn(1, 16, *latent)
    feats, t = torch.randn(1, text_len, 64), torch.tensor([0.37])
    with torch.no_grad():
        want = _reference_forward(rmodel, x, t, feats)
        got = ours(x, t, ours.encode_text(feats))
    assert got.shape == want.shape == (1, 16, *latent)
    assert (got - want).abs().max() <= 1e-5


def test_dit_batch_of_two_equal_length_prompts_matches_the_reference():
    rmodel, ours = _tiny_pair()
    torch.manual_seed(2)
    x, feats, t = torch.randn(2, 16, 12, 10), torch.randn(2, 9, 64), torch.tensor([0.8, 0.2])
    with torch.no_grad():
        want = _reference_forward(rmodel, x, t, feats)
        got = ours(x, t, ours.encode_text(feats))
    assert (got - want).abs().max() <= 1e-5


def test_caption_tokens_are_padded_to_a_multiple_of_32_and_image_rope_follows():
    _, ours = _tiny_pair()
    assert ours.encode_text(torch.randn(1, 5, 64)).shape == (1, 32, 256)
    assert ours.encode_text(torch.randn(1, 33, 64)).shape == (1, 64, 256)
    a, _ = ours.rope(32, 4, 4, torch.device("cpu"))
    b, _ = ours.rope(64, 4, 4, torch.device("cpu"))
    assert a.shape[0] == 16 + 16 + 32 and b.shape[0] == 16 + 16 + 64
    assert not torch.equal(a[0], b[0])


def test_timestep_is_one_minus_sigma_and_a_different_t_changes_the_output():
    _, ours = _tiny_pair()
    x, ctx = torch.randn(1, 16, 8, 8), ours.encode_text(torch.randn(1, 5, 64))
    with torch.no_grad():
        assert not torch.allclose(ours(x, torch.tensor([0.2]), ctx), ours(x, torch.tensor([0.8]), ctx))


# ─── schedule and tokenizer ──────────────────────────────────────────────────

_SCHED = json.loads((_HERE / "z_image_schedule.json").read_text())


@pytest.mark.parametrize("n", [1, 2, 4, 8, 9, 28, 50])
@pytest.mark.parametrize("variant, repo", [("turbo", "Z-Image-Turbo"), ("base", "Z-Image")])
def test_zimage_schedule_is_the_official_scheduler(variant, repo, n):
    for size in (512, 1024):
        sigmas, shift = zimage_sigmas("zimage", n, variant, size, size, 99.0, "cpu")
        want = torch.tensor(_SCHED["sigmas"][repo][str(n)])
        assert shift == _SCHED["shift"][repo] == ZIMAGE_DEFAULTS[variant]["shift"]
        assert sigmas.shape == want.shape and (sigmas - want).abs().max() <= 1e-6
    assert sigmas[-1] == 0


def test_flow_and_dynamic_schedules_take_their_own_shift():
    s, shift = zimage_sigmas("flow", 8, "turbo", 1024, 1024, 5.0, "cpu")
    assert shift == 5.0 and torch.allclose(s, zimage_sigmas("flow", 8, "base", 512, 512, 5.0, "cpu")[0])
    _, dyn = zimage_sigmas("flow_dyn", 8, "turbo", 1024, 1024, 5.0, "cpu")
    assert dyn != 5.0


_TOK = json.loads((_HERE / "z_image_tokenizer_cases.json").read_text())["cases"]


def test_template_and_special_tokens():
    assert render_prompt("x") == "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n"
    assert render_prompt("  keep  ") == "<|im_start|>user\n  keep  <|im_end|>\n<|im_start|>assistant\n"
    ids = ZImageTokenizer()("hi").ids[0].tolist()
    assert ids[0] == 151644 and ids[-5:] == [151645, 198, 151644, 77091, 198]   # <|im_end|> \n <|im_start|> assistant \n


@pytest.mark.parametrize("case", _TOK, ids=[f"case{i}" for i in range(len(_TOK))])
def test_ids_match_the_transformers_pipeline_encoding(case):
    got = ZImageTokenizer()(case["prompt"])
    assert got.ids.tolist() == [case["ids"]]
    assert got.truncated == (len(case["ids"]) == MAX_LENGTH and case["prompt"].startswith("word"))


@pytest.mark.skipif(not (_OFFICIAL / "tokenizer" / "tokenizer.json").exists(), reason="official tokenizer absent")
def test_vendored_vocab_is_the_official_tokenizer_file():
    path = _OFFICIAL / "tokenizer" / "tokenizer.json"
    from tokenizers import Tokenizer
    official = Tokenizer.from_file(str(path))
    ours = ZImageTokenizer()
    for p in (c["prompt"] for c in _TOK[:6]):
        assert official.encode(render_prompt(p)).ids == ours(p).ids[0].tolist()


def test_truncation_keeps_the_head_and_flags_it():
    tok = ZImageTokenizer()("word " * 700)
    assert tok.ids.shape == (1, MAX_LENGTH) and tok.truncated
    assert tok.ids[0, 0].item() == 151644


# ─── text encoder and loaders ────────────────────────────────────────────────

_TE_CFG = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=36,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)


def _random_te(seed: int = 0, layers: int = 36) -> Qwen3TextEncoder:
    torch.manual_seed(seed)
    from dataclasses import replace
    te = Qwen3TextEncoder(replace(_TE_CFG, num_hidden_layers=layers))
    with torch.no_grad():
        for p in te.parameters():
            p.copy_(0.1 * torch.randn_like(p) + (1.0 if p.dim() == 1 else 0.0))
    return te.eval()


def test_the_text_encoder_drops_the_layers_the_dit_never_reads(tmp_path):
    full = _random_te()
    save_file({k: v.contiguous() for k, v in full.state_dict().items() if "inv_freq" not in k},
              str(tmp_path / "te.safetensors"))
    te, sd = _zimage_text_encoder(str(tmp_path / "te.safetensors"))
    assert len(te.model.layers) == 35 and "model.layers.35.mlp.up_proj.weight" not in sd
    ids = torch.randint(0, 300, (1, 11))
    with torch.no_grad():
        want = full(ids, hidden_layers=[35])[0]
        got = te(ids, hidden_layers=[35])[0]
    assert torch.equal(got, want)


@pytest.mark.parametrize("pick", ["folder", "repo_root", "index"])
def test_the_diffusers_text_encoder_folder_loads_in_every_spelling(tmp_path, pick):
    full = _random_te()
    sd = {k[len("model."):]: v.contiguous() for k, v in full.state_dict().items() if "inv_freq" not in k}
    folder = tmp_path / "text_encoder"
    folder.mkdir()
    keys = sorted(sd)
    weight_map = {}
    for i, part in enumerate((keys[::2], keys[1::2])):
        name = f"model-0000{i + 1}-of-00002.safetensors"
        save_file({k: sd[k] for k in part}, str(folder / name))
        weight_map.update({k: name for k in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    path = {"folder": folder, "repo_root": tmp_path, "index": folder / "model.safetensors.index.json"}[pick]
    te, _ = _zimage_text_encoder(str(path))
    ids = torch.randint(0, 300, (1, 9))
    with torch.no_grad():
        assert torch.equal(te(ids, hidden_layers=[35])[0], full(ids, hidden_layers=[35])[0])


def test_a_text_encoder_with_too_few_layers_is_rejected(tmp_path):
    small = _random_te(layers=30)
    save_file({k: v.contiguous() for k, v in small.state_dict().items() if "inv_freq" not in k},
              str(tmp_path / "te.safetensors"))
    with pytest.raises(ValueError, match="needs Qwen3-4B"):
        _zimage_text_encoder(str(tmp_path / "te.safetensors"))


@pytest.mark.parametrize("name, want", [
    ("z_image_turbo_bf16.safetensors", "turbo"), ("Z_Image_Turbo_int8_convrot.safetensors", "turbo"),
    ("z_image_int8_convrot.safetensors", "base"), ("z_image_bf16.safetensors", "base"),
    ("something.safetensors", "base"),
])
def test_variant_is_read_off_the_file_name(name, want):
    assert _zimage_variant(name, None) == want


def test_variant_override_and_validation_and_diffusers_folder(tmp_path):
    assert _zimage_variant("z_image_turbo_bf16.safetensors", "base") == "base"
    with pytest.raises(ValueError, match="variant"):
        _zimage_variant("x.safetensors", "raw")
    folder = tmp_path / "Z-Image-Turbo" / "transformer"
    folder.mkdir(parents=True)
    assert _zimage_variant(str(folder), None) == "turbo"


class _StubTokenizer:
    def __call__(self, prompt):
        n = 4 + len(prompt) % 5
        return ZImageTokenized(torch.arange(1, n + 1)[None] * 7, False)


def _tiny_bundle(cache=None, variant="base") -> ModelBundle:
    _, dit = _tiny_pair()
    vae = AutoencoderKL(VAEConfig(base_channels=32, channel_mult=(1, 2), num_res_blocks=1, z_channels=16,
                                  scale_factor=0.3611, shift_factor=0.1159, use_quant_conv=False)).eval()
    spec = ModelSpec(architecture="z_image", prediction="flow", zero_terminal_snr=False, latent_channels=16,
                     context_dim=64, latent_scale=0.3611, latent_shift=0.1159, variant=variant)
    return ModelBundle(spec=spec, schedule=None, tokenizer=_StubTokenizer(), text_encoder=_random_te(),
                       backbone=dit, vae=vae, policy=_cpu_policy(), cond_cache=cache)


def _tiny_files(tmp_path, diffusers: bool = False):
    """A loadable tiny checkpoint set: DiT (cap width 64), 36-layer Qwen3 TE, 16-channel VAE."""
    rmodel, ours = _tiny_pair()
    if diffusers:
        folder = tmp_path / "transformer"
        folder.mkdir()
        sd = {k: v.contiguous() for k, v in rmodel.state_dict().items()}
        keys = sorted(sd)
        save_file({k: sd[k] for k in keys[::2]}, str(folder / "diffusion_pytorch_model-00001-of-00002.safetensors"))
        save_file({k: sd[k] for k in keys[1::2]}, str(folder / "diffusion_pytorch_model-00002-of-00002.safetensors"))
        dit = str(tmp_path)
    else:
        save_file({k: v.contiguous() for k, v in ours.state_dict().items()}, str(tmp_path / "dit.safetensors"))
        dit = str(tmp_path / "dit.safetensors")
    te = _random_te()
    save_file({k: v.contiguous() for k, v in te.state_dict().items() if "inv_freq" not in k},
              str(tmp_path / "te.safetensors"))
    vae = AutoencoderKL(VAEConfig(base_channels=32, channel_mult=(1, 2), num_res_blocks=1, z_channels=16,
                                  use_quant_conv=False))
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(tmp_path / "vae.safetensors"))
    return dit, str(tmp_path / "te.safetensors"), str(tmp_path / "vae.safetensors")


def test_load_zimage_checkpoint_builds_a_bundle_that_generates(tmp_path):
    bundle = load_zimage_checkpoint(*_tiny_files(tmp_path), policy=_cpu_policy())
    assert bundle.spec.architecture == "z_image" and bundle.spec.variant == "base"
    assert isinstance(bundle.backbone, ZImageDiT) and isinstance(bundle.tokenizer, ZImageTokenizer)
    assert len(bundle.text_encoder.model.layers) == 35
    bundle.tokenizer = _StubTokenizer()
    image = TextToImage(bundle)("a cat", "", steps=2, cfg_scale=3.0, width=64, height=64, seed=1)
    assert image.size[0] > 0


def test_a_diffusers_folder_loads_to_the_same_weights_as_the_fused_file(tmp_path):
    fused = tmp_path / "fused"
    fused.mkdir()
    a = load_zimage_dit(_tiny_files(fused)[0], _cpu_policy())
    folder = tmp_path / "folder"
    folder.mkdir()
    files = _tiny_files(folder, diffusers=True)
    b = load_zimage_dit(files[0], _cpu_policy())
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    bundle = load_zimage_checkpoint(*files, variant="turbo", policy=_cpu_policy())
    assert bundle.spec.variant == "turbo"


def test_branch_scales_are_only_applied_to_fp16(tmp_path):
    from diffucore.bundle import _ZIMAGE_FP16_ATTN_SCALE, _ZIMAGE_FP16_FFN_SCALE
    dit = _tiny_files(tmp_path)[0]
    m = load_zimage_dit(dit, _cpu_policy(torch.float32))
    assert (m.attn_scale, m.ffn_scale) == (1.0, 1.0)
    m = load_zimage_dit(dit, _cpu_policy(torch.float16))
    assert (m.attn_scale, m.ffn_scale) == (_ZIMAGE_FP16_ATTN_SCALE, _ZIMAGE_FP16_FFN_SCALE)


def test_the_dit_file_must_be_z_image(tmp_path):
    dit, te, vae = _tiny_files(tmp_path)
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}, str(tmp_path / "unet.safetensors"))
    with pytest.raises((ValueError, NotImplementedError)):
        load_zimage_checkpoint(str(tmp_path / "unet.safetensors"), te, vae, policy=_cpu_policy())


def test_a_vae_with_the_wrong_latent_width_is_rejected(tmp_path):
    dit, te, _ = _tiny_files(tmp_path)
    vae = AutoencoderKL(VAEConfig(base_channels=32, channel_mult=(1, 2), num_res_blocks=1, z_channels=32,
                                  use_quant_conv=False))
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(tmp_path / "vae32.safetensors"))
    with pytest.raises(ValueError, match="FLUX.1 VAE"):
        load_zimage_checkpoint(dit, te, str(tmp_path / "vae32.safetensors"), policy=_cpu_policy())


# ─── end to end, tiny ────────────────────────────────────────────────────────


def _spy_forward(bundle):
    calls = []
    original = bundle.backbone.forward

    def forward(x, t, ctx):
        calls.append((ctx.shape[1], round(float(t[0]), 4)))
        return original(x, t, ctx)

    bundle.backbone.forward = forward
    return calls


def test_variant_defaults_pick_steps_and_cfg():
    turbo, base = _tiny_bundle(variant="turbo"), _tiny_bundle(variant="base")
    ct, cb = _spy_forward(turbo), _spy_forward(base)
    zimage_text_to_image(turbo, "a cat", width=64, height=64, seed=1)
    zimage_text_to_image(base, "a cat", "blur", steps=3, width=64, height=64, seed=1)
    assert len(ct) == 8                                   # 8 steps, CFG 1: no uncond forward
    assert len(cb) == 6                                   # 3 steps x (cond + uncond)
    assert ZIMAGE_DEFAULTS["base"] == {"steps": 50, "cfg_scale": 5.0, "shift": 6.0}


def test_timestep_fed_to_the_model_is_one_minus_sigma():
    bundle = _tiny_bundle(variant="turbo")
    calls = _spy_forward(bundle)
    zimage_text_to_image(bundle, "a cat", steps=4, width=64, height=64, seed=1)
    sigmas, _ = zimage_sigmas("zimage", 4, "turbo", 64, 64, 3.0, "cpu")
    assert [t for _, t in calls] == [round(1 - float(s), 4) for s in sigmas[:-1]]


def test_cfg_runs_cond_and_uncond_at_their_own_lengths_and_matches_the_formula(monkeypatch):
    bundle = _tiny_bundle()
    calls = _spy_forward(bundle)
    image = zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=4.0, width=64, height=64, seed=3)
    assert image.size[0] > 0
    # "a cat" -> 4 + 5 % 5 = 4 tokens, "no" -> 4 + 2 % 5 = 6; both pad to 32
    assert [c[0] for c in calls] == [32, 32, 32, 32]
    # hand-rolled euler with the reference's combination: pos + g (pos - neg), g = cfg - 1
    bundle2 = _tiny_bundle()
    dit = bundle2.backbone
    with torch.no_grad():
        ctx_c = dit.encode_text(zmod.zimage_encode(bundle2, "a cat", torch.device("cpu")))
        ctx_u = dit.encode_text(zmod.zimage_encode(bundle2, "no", torch.device("cpu")))
        gen = torch.Generator().manual_seed(3)
        x = torch.randn(1, 16, 8, 8, generator=gen)
        sigmas, _ = zimage_sigmas("zimage", 2, "base", 64, 64, 0.0, "cpu")
        for i in range(2):
            t = torch.tensor([1 - float(sigmas[i])])
            pos, neg = dit(x, t, ctx_c), dit(x, t, ctx_u)
            noise_pred = -(pos + 3.0 * (pos - neg))
            x = x + (sigmas[i + 1] - sigmas[i]) * noise_pred
    out = []
    monkeypatch.setattr(zmod, "_decode", lambda m, z: (out.append(z.clone()), (Image.new("RGB", (8, 8)), "x"))[1])
    zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=4.0, width=64, height=64, seed=3)
    assert (out[0] - x).abs().max() <= 1e-4


def test_cfg_normalization_caps_the_guided_norm_at_the_conditional_one(monkeypatch):
    out = []
    monkeypatch.setattr(zmod, "_decode", lambda m, z: (out.append(z.clone()), (Image.new("RGB", (8, 8)), "x"))[1])
    bundle = _tiny_bundle()
    zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=8.0, width=64, height=64, seed=3)
    zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=8.0, cfg_normalization=1.0, width=64,
                         height=64, seed=3)
    zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=8.0, cfg_normalization=1e6, width=64,
                         height=64, seed=3)
    plain, capped, loose = out
    assert not torch.allclose(plain, capped) and torch.equal(plain, loose)


def test_cfg_interval_limits_the_uncond_forwards():
    bundle = _tiny_bundle()
    calls = _spy_forward(bundle)
    zimage_text_to_image(bundle, "a cat", "no", steps=4, cfg_scale=4.0, cfg_interval_start=0.0,
                         cfg_interval_end=0.5, width=64, height=64, seed=1)
    assert len(calls) == 4 + 2


def test_seeded_runs_repeat_and_the_conditioning_cache_skips_the_encoder():
    cache = ConditioningCache()
    bundle = _tiny_bundle(cache)
    n = []
    original = zmod.zimage_encode
    zmod.zimage_encode = lambda *a, **k: (n.append(1), original(*a, **k))[1]
    try:
        a = zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=3.0, width=64, height=64, seed=7)
        b = zimage_text_to_image(bundle, "a cat", "no", steps=2, cfg_scale=3.0, width=64, height=64, seed=7)
    finally:
        zmod.zimage_encode = original
    assert np.array_equal(np.asarray(a), np.asarray(b)) and len(n) == 2


def test_text_to_image_dispatches_on_the_architecture_and_falls_back_to_the_zimage_schedule():
    bundle = _tiny_bundle(variant="turbo")
    image, info = TextToImage(bundle)("a cat", "", steps=2, cfg_scale=1.0, scheduler="karras", width=64, height=64,
                                      seed=2, return_info=True)
    assert image.size[0] > 0 and info.vae_decode_mode == "untiled"


def test_sizes_off_the_16_grid_and_bad_names_are_rejected():
    bundle = _tiny_bundle()
    with pytest.raises(ValueError, match="divisible by 16"):
        zimage_text_to_image(bundle, "a", width=72, height=64)
    with pytest.raises(ValueError, match="sampler"):
        zimage_text_to_image(bundle, "a", sampler="nope", width=64, height=64)
    with pytest.raises(ValueError, match="scheduler"):
        zimage_text_to_image(bundle, "a", scheduler="nope", width=64, height=64)


# ─── img2img and inpaint, tiny ───────────────────────────────────────────────


def _init_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


def test_encode_is_the_inverse_scaling_of_the_decode(monkeypatch):
    bundle = _tiny_bundle()
    z = zmod._encode_image(bundle, _init_image(), 64, 64)
    assert z.shape == (1, 16, 32, 32)   # the tiny VAE only halves
    mean = bundle.vae.encode(zmod.preprocess_image(_init_image(), 64, 64), sample=False)
    assert torch.equal(z, mean.float())


def test_img2img_runs_the_last_steps_of_a_longer_schedule():
    bundle = _tiny_bundle()
    calls = _spy_forward(bundle)
    ImageToImage(bundle)("a cat", _init_image(), "", strength=0.5, steps=4, cfg_scale=1.0, width=64, height=64, seed=1)
    full, _ = zimage_sigmas("zimage", 8, "base", 64, 64, 0.0, "cpu")
    assert [t for _, t in calls] == [round(1 - float(s), 4) for s in full[-5:-1]]


def test_img2img_rejects_a_bad_strength():
    bundle = _tiny_bundle()
    with pytest.raises(ValueError, match="strength"):
        ImageToImage(bundle)("a", _init_image(), strength=0.0, width=64, height=64)


def test_inpaint_keeps_the_masked_out_pixels_and_repaints_the_rest():
    bundle = _tiny_bundle()
    init = _init_image()
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (32, 0, 64, 64))
    out = Inpaint(bundle)("a cat", init, mask, "", strength=1.0, steps=2, cfg_scale=1.0, width=64, height=64, seed=1)
    a, b = np.asarray(init), np.asarray(out)
    assert np.array_equal(a[:, :32], b[:, :32]) and not np.array_equal(a[:, 32:], b[:, 32:])


def test_inpaint_pins_the_keep_tokens_to_the_init_latent_at_every_step(monkeypatch):
    bundle = _tiny_bundle()
    init = _init_image()
    z0 = zmod._encode_image(bundle, init, 64, 64)
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (32, 0, 64, 64))
    seen = []
    monkeypatch.setattr(zmod, "_decode", lambda m, z: (seen.append(z.clone()), (Image.new("RGB", (64, 64)), "x"))[1])
    Inpaint(bundle)("a cat", init, mask, "", strength=1.0, steps=2, cfg_scale=1.0, width=64, height=64, seed=1,
                    sampler="heun")
    assert torch.allclose(seen[0][..., :14], z0[..., :14], atol=1e-5)


# ─── LoRA ────────────────────────────────────────────────────────────────────

_LORA = json.loads((_HERE / "z_image_headers.json").read_text())["loras"]


def _lora_bundle(cache=None):
    rmodel, ours = _tiny_pair()
    spec = ModelSpec(architecture="z_image", prediction="flow", zero_terminal_snr=False, latent_channels=16,
                     context_dim=64, latent_scale=0.3611, latent_shift=0.1159, variant="base")
    return rmodel, ModelBundle(spec=spec, schedule=None, tokenizer=_StubTokenizer(), text_encoder=None,
                               backbone=ours, vae=None, policy=_cpu_policy(), cond_cache=cache)


# (diffusers module path, in, out) of the layers the LoRA files touch
def _tiny_modules(dim=256, ffn=682):
    out = []
    for blk in ("layers.0", "layers.2", "noise_refiner.1", "context_refiner.0"):
        for n in ("to_q", "to_k", "to_v"):
            out.append((f"{blk}.attention.{n}", dim, dim))
        out.append((f"{blk}.attention.to_out.0", dim, dim))
        out += [(f"{blk}.feed_forward.w1", dim, ffn), (f"{blk}.feed_forward.w2", ffn, dim),
                (f"{blk}.feed_forward.w3", dim, ffn)]
        if not blk.startswith("context"):
            out.append((f"{blk}.adaLN_modulation.0", 256, 4 * dim))
    out += [("all_x_embedder.2-1", 64, dim), ("all_final_layer.2-1.linear", dim, 64),
            ("all_final_layer.2-1.adaLN_modulation.1", 256, dim), ("cap_embedder.1", 64, dim)]
    return out


def _make_factors(rank=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {name: (0.3 * torch.randn(rank, i, generator=g), 0.3 * torch.randn(o, rank, generator=g))
            for name, i, o in _tiny_modules()}


def _spelling(name: str, fmt: str) -> str:
    if fmt == "kohya_mangled":
        return "lora_unet_" + from_diffusers_name(name).replace(".", "_")
    if fmt == "kohya_diffusers":
        return "lora_unet_" + name.replace(".", "_")
    if fmt == "original_names":
        return "diffusion_model." + from_diffusers_name(name)
    prefix = {"bare": "", "transformer": "transformer.", "diffusion_model": "diffusion_model."}[fmt]
    return prefix + name


def _write_lora(path, factors, fmt, alpha=None):
    tensors = {}
    for name, (down, up) in factors.items():
        if fmt in ("original_names", "kohya_mangled"):
            # the fused names carry no per-matrix q/k/v split, so only unfused layers are written
            if name.endswith(("to_q", "to_k", "to_v")):
                continue
        base = _spelling(name, fmt)
        if fmt.startswith("kohya"):
            tensors[base + ".lora_down.weight"], tensors[base + ".lora_up.weight"] = down, up
            tensors[base + ".alpha"] = torch.tensor(float(alpha if alpha is not None else down.shape[0]))
        else:
            tensors[base + ".lora_A.weight"], tensors[base + ".lora_B.weight"] = down, up
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    return tensors


def _reference_with_lora(rmodel, factors, multiplier, alpha=None, skip=()):
    import copy
    merged = copy.deepcopy(rmodel)
    params = dict(merged.named_parameters())
    for name, (down, up) in factors.items():
        if name.endswith(skip):
            continue
        scale = (alpha if alpha is not None else down.shape[0]) / down.shape[0] * multiplier
        w = params[name + ".weight"]
        with torch.no_grad():
            w.add_(up @ down * scale)
    return merged


@pytest.mark.parametrize("fmt", ["bare", "transformer", "diffusion_model", "original_names", "kohya_mangled",
                                 "kohya_diffusers"])
def test_lora_hooks_match_the_reference_with_the_delta_merged_in_every_format(tmp_path, fmt):
    rmodel, bundle = _lora_bundle()
    factors = _make_factors()
    skip = ("to_q", "to_k", "to_v") if fmt in ("original_names", "kohya_mangled") else ()
    alpha = 2.0 if fmt.startswith("kohya") else None
    _write_lora(tmp_path / "l.safetensors", factors, fmt, alpha)
    report = apply_lora(bundle, str(tmp_path / "l.safetensors"), 0.8)
    n_tensors = len(factors) - (12 if skip else 0)
    assert report.applied == n_tensors and report.unmatched == []
    want_model = _reference_with_lora(rmodel, factors, 0.8, alpha, skip)
    torch.manual_seed(5)
    x, feats, t = torch.randn(1, 16, 12, 10), torch.randn(1, 7, 64), torch.tensor([0.4])
    with torch.no_grad():
        want = _reference_forward(want_model, x, t, feats)
        got = bundle.backbone(x, t, bundle.backbone.encode_text(feats))
    assert (got - want).abs().max() <= 1e-4
    clear_loras(bundle)
    with torch.no_grad():
        base = _reference_forward(rmodel, x, t, feats)
        assert (bundle.backbone(x, t, bundle.backbone.encode_text(feats)) - base).abs().max() <= 1e-4


def test_qkv_slices_equal_three_separate_lora_applications(tmp_path):
    rmodel, bundle = _lora_bundle()
    factors = {k: v for k, v in _make_factors().items() if k.startswith("layers.0.attention.to_")
               and not k.endswith("out.0")}
    assert len(factors) == 3
    _write_lora(tmp_path / "l.safetensors", factors, "transformer")
    apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
    fused_in = torch.randn(1, 5, 256)
    qkv = bundle.backbone.layers[0].attention.qkv
    with torch.no_grad():
        got = qkv(fused_in)
        base = torch.nn.functional.linear(fused_in, qkv.weight)
    for i, n in enumerate(("to_q", "to_k", "to_v")):
        down, up = factors["layers.0.attention." + n]
        want = fused_in @ down.T @ up.T
        assert torch.allclose((got - base)[..., i * 256:(i + 1) * 256], want, atol=1e-5)


def test_lora_alpha_comes_from_the_tensor_then_the_metadata_then_the_rank(tmp_path):
    rmodel, bundle = _lora_bundle()
    factors = {"layers.0.feed_forward.w1": _make_factors()["layers.0.feed_forward.w1"]}
    down, up = factors["layers.0.feed_forward.w1"]
    path = tmp_path / "l.safetensors"
    save_file({"layers.0.feed_forward.w1.lora_A.weight": down.contiguous(),
               "layers.0.feed_forward.w1.lora_B.weight": up.contiguous()}, str(path), metadata={"alpha": "2"})
    apply_lora(bundle, str(path), 1.0)
    x = torch.randn(1, 3, 256)
    w1 = bundle.backbone.layers[0].feed_forward.w1
    with torch.no_grad():
        delta = w1(x) - torch.nn.functional.linear(x, w1.weight)
    assert torch.allclose(delta, x @ down.T @ up.T * (2.0 / 4), atol=1e-5)


def test_lora_apply_remove_and_clear_invalidate_the_conditioning_cache(tmp_path):
    cache = ConditioningCache()
    _, bundle = _lora_bundle(cache)
    _write_lora(tmp_path / "l.safetensors", _make_factors(), "bare")
    for action in ("apply", "remove", "clear"):
        cache.put(("z_image", "a", ""), {"cond_ctx": torch.zeros(1), "uncond_ctx": None})
        if action == "apply":
            apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
        elif action == "remove":
            remove_lora(bundle, str(tmp_path / "l.safetensors"))
        else:
            apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
            cache.put(("z_image", "a", ""), {"cond_ctx": torch.zeros(1), "uncond_ctx": None})
            clear_loras(bundle)
        assert cache.get(("z_image", "a", "")) is None


def test_lora_stacks_and_removes_in_order(tmp_path):
    rmodel, bundle = _lora_bundle()
    f1, f2 = _make_factors(seed=1), _make_factors(seed=2)
    _write_lora(tmp_path / "a.safetensors", f1, "bare")
    _write_lora(tmp_path / "b.safetensors", f2, "transformer")
    apply_lora(bundle, str(tmp_path / "a.safetensors"), 1.0)
    apply_lora(bundle, str(tmp_path / "b.safetensors"), 0.5)
    x, feats, t = torch.randn(1, 16, 8, 8), torch.randn(1, 6, 64), torch.tensor([0.5])
    with torch.no_grad():
        both = bundle.backbone(x, t, bundle.backbone.encode_text(feats))
        remove_lora(bundle, str(tmp_path / "a.safetensors"))
        only_b = bundle.backbone(x, t, bundle.backbone.encode_text(feats))
        want_b = _reference_forward(_reference_with_lora(rmodel, f2, 0.5), x, t, feats)
        want_ab = _reference_forward(_reference_with_lora(_reference_with_lora(rmodel, f1, 1.0), f2, 0.5), x, t, feats)
    assert (only_b - want_b).abs().max() <= 1e-4 and (both - want_ab).abs().max() <= 1e-4


def test_unknown_modules_are_reported_not_hooked(tmp_path):
    _, bundle = _lora_bundle()
    save_file({"diffusion_model.layers.0.nope.lora_A.weight": torch.zeros(2, 4),
               "diffusion_model.layers.0.nope.lora_B.weight": torch.zeros(4, 2)}, str(tmp_path / "l.safetensors"))
    report = apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
    assert report.applied == 0 and report.unmatched == ["diffusion_model.layers.0.nope"]


def test_lora_that_changes_a_weights_shape_is_rejected_and_hooks_nothing(tmp_path):
    _, bundle = _lora_bundle()
    save_file({"diffusion_model.x_embedder.weight": torch.zeros(256, 80),
               "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": torch.zeros(4, 256),
               "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": torch.zeros(682, 4)},
              str(tmp_path / "l.safetensors"))
    with pytest.raises(ValueError, match="Z-Image"):
        apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
    assert not bundle.backbone.layers[0].feed_forward.w1._forward_hooks


def test_lora_applies_to_int8_layers(tmp_path):
    from diffucore.models._int8 import Int8Linear, quantize_linears
    torch.manual_seed(0)
    cfg = ZImageConfig(dim=768, n_layers=1, n_refiner_layers=1, n_heads=6, ffn_dim=2048, cap_feat_dim=64)
    model = ZImageDiT(cfg).float().eval()
    spec = ModelSpec(architecture="z_image", prediction="flow", zero_terminal_snr=False, latent_channels=16,
                     context_dim=64)
    bundle = ModelBundle(spec=spec, schedule=None, tokenizer=None, text_encoder=None, backbone=model, vae=None,
                         policy=_cpu_policy())
    quantize_linears(model, int8_layer_names(cfg), torch.device("cpu"))
    qkv = model.layers[0].attention.qkv
    assert isinstance(qkv, Int8Linear)
    g = torch.Generator().manual_seed(1)
    names = {"layers.0.attention.to_q": 768, "layers.0.attention.to_k": 768, "layers.0.attention.to_v": 768}
    factors = {n: (torch.randn(4, 768, generator=g), torch.randn(768, 4, generator=g)) for n in names}
    _write_lora(tmp_path / "l.safetensors", factors, "transformer")
    x = torch.randn(1, 5, 768)
    with torch.no_grad():
        before = qkv(x)
        report = apply_lora(bundle, str(tmp_path / "l.safetensors"), 1.0)
        after = qkv(x)
    assert report.unmatched == [] and report.applied == 3
    want = torch.cat([x @ d.T @ u.T for d, u in factors.values()], dim=-1)
    assert torch.allclose(after - before, want, atol=1e-3)


@pytest.mark.parametrize("name", ["pixel_art_r32", "coloring_book_r6"])
def test_real_lora_headers_map_with_zero_unmapped(name):
    backbone = _build(ZImageDiT, ZImageConfig())
    targets = _zimage_targets(backbone)
    keys = _expand(_LORA[name])
    bases = {k.rsplit(".lora_", 1)[0] for k in keys}
    assert len(bases) == 240 and all(b in targets for b in bases)
    rank = keys["diffusion_model.layers.0.attention.to_q.lora_A.weight"][1][0]
    for b in bases:
        module, rows = targets[b]
        down = keys[b + ".lora_A.weight"][1]
        up = keys[b + ".lora_B.weight"][1]
        out = module.out_features if rows is None else rows[1] - rows[0]
        assert down == (rank, module.in_features) and up == (out, rank)


def test_every_spelling_resolves_to_the_same_layer():
    targets = _zimage_targets(_build(ZImageDiT, ZImageConfig()))
    module, rows = targets["transformer.layers.4.attention.to_v"]
    assert module is targets["diffusion_model.layers.4.attention.qkv"][0] and rows == (2 * 3840, 3 * 3840)
    assert targets["lora_unet_layers_4_attention_to_out_0"][0] is targets["layers.4.attention.out"][0]
    assert targets["all_x_embedder.2-1"][0] is targets["x_embedder"][0]
    assert targets["lora_unet_all_final_layer_2-1_linear"][0] is targets["final_layer.linear"][0]
