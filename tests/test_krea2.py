"""Krea 2 (Raw + Turbo) verification, offline.

1. The DiT and text-encoder modules match the real checkpoints' keys and
   shapes (``krea2_headers.json``: ComfyUI bf16, ComfyUI INT8 ConvRot, and the
   Qwen3-VL-4B language model), with the config derived from the keys.
2. A tiny random DiT matches the krea-ai/krea-2 reference forward in fp32,
   the reference run padded + masked and ours unpadded (proves dropping the
   padding is exact, and the GQA head mapping). Skipped unless ``KREA2_REF``
   points at a krea-ai/krea-2 checkout.
3. ``residual_scale`` leaves the output unchanged.
4. Tokenizer ids and the ``krea2`` schedule match the diffusers golden bundle
   (``KREA2_GOLDEN``, skipped when absent).
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from diffucore.bundle import _build, _krea2_text_encoder
from diffucore.conditioning import Krea2Tokenizer
from diffucore.loading import detect_architecture
from diffucore.models import Krea2Config, Krea2DiT
from diffucore.pipelines._krea2 import krea2_shift, krea2_sigmas

_HEADERS = json.loads(Path(__file__).with_name("krea2_headers.json").read_text())
_REF = os.environ.get("KREA2_REF")
_GOLDEN = Path(os.environ.get(
    "KREA2_GOLDEN",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/krea2-golden-20261012/golden.safetensors",
))
_QUANT_SUFFIXES = (".weight_scale", ".comfy_quant")


def _meta_sd(header: dict) -> dict:
    return {k: torch.empty(shape, device="meta") for k, (_, shape) in header.items()}


def test_detects_krea2_and_derives_the_released_config():
    for name in ("comfy_bf16", "comfy_int8_convrot"):
        shapes = {k: tuple(s) for k, (_, s) in _HEADERS[name].items()}
        assert detect_architecture(shapes).architecture == "krea2"
        assert Krea2Config.from_state_dict(_meta_sd(_HEADERS[name])) == Krea2Config()


def test_dit_keys_and_shapes_match_the_checkpoint():
    model = _build(Krea2DiT, Krea2Config())
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    theirs = {k: tuple(s) for k, (_, s) in _HEADERS["comfy_bf16"].items()}
    assert ours == theirs
    int8_keys = {k for k in _HEADERS["comfy_int8_convrot"] if not k.endswith(_QUANT_SUFFIXES)}
    assert int8_keys == set(theirs)


def test_text_encoder_is_35_layers_of_qwen3vl_at_theta_5e6():
    te_sd = _meta_sd(_HEADERS["qwen3vl_4b_language_model"])
    encoder, sd = _krea2_text_encoder(te_sd)
    cfg = encoder.cfg
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.head_dim) == (35, 2560, 32, 8, 128)
    assert cfg.rope_theta == 5_000_000.0
    assert not any(k.startswith("model.layers.35.") for k in sd)


def _tiny_ref_model(ref):
    cfg = ref.SingleMMDiTConfig(features=256, tdim=64, txtdim=256, heads=2, kvheads=1, multiplier=4,
                                layers=2, patch=2, channels=16, txtheads=2, txtkvheads=2, txtlayers=3)
    torch.manual_seed(0)
    model = ref.SingleStreamDiT(cfg).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            std = 0.3 if name.endswith((".scale", ".lin")) else 0.05
            p.copy_(torch.randn_like(p) * std)
    return model


def _safe_masked_sdpa(q, k, v, mask=None, scale=None, gqa=False):
    """The reference forces cuDNN SDPA (CUDA only); on CPU, fully masked pad
    rows would turn into NaN and leak through the zero attention weights."""
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)
    return torch.nan_to_num(out).transpose(1, 2).flatten(2)


@pytest.mark.skipif(not _REF, reason="KREA2_REF (a krea-ai/krea-2 checkout) not set")
def test_matches_reference_forward_with_padding_dropped(monkeypatch):
    import torch._dynamo
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.syspath_prepend(_REF)
    ref = importlib.import_module("mmdit")
    monkeypatch.setattr(ref, "attention", _safe_masked_sdpa)
    rmodel = _tiny_ref_model(ref)
    ours = Krea2DiT(Krea2Config.from_state_dict(rmodel.state_dict())).float()
    ours.load_state_dict(rmodel.state_dict(), strict=True)

    torch.manual_seed(1)
    n_txt, n_pad, H, W = 5, 3, 8, 12
    feats = torch.randn(1, n_txt + n_pad, 3, 256)
    x = torch.randn(1, 16, H, W)
    t = torch.tensor([0.7])
    h, w = H // 2, W // 2
    img = x.view(1, 16, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, h * w, 64)
    pos = torch.zeros(1, n_txt + n_pad + h * w, 3, dtype=torch.float64)
    pos[0, n_txt + n_pad:, 1] = torch.arange(h).repeat_interleave(w).double()
    pos[0, n_txt + n_pad:, 2] = torch.arange(w).repeat(h).double()
    mask = torch.ones(1, n_txt + n_pad + h * w, dtype=torch.bool)
    mask[0, n_txt:n_txt + n_pad] = False
    with torch.no_grad():
        want = rmodel(img=img, context=feats, t=t, pos=pos, mask=mask)
        ctx = ours.encode_text(feats[:, :n_txt])
        got = ours(x, t, ctx)
    got = got.view(1, 16, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, h * w, 64)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), (got - want).abs().max()


def test_residual_scale_is_exact():
    """Algebraically exact; with ε = 0 so a tiny random stream (ms ~1e-4, far
    below the real model's) doesn't turn the norm's ε into the difference."""
    torch.manual_seed(0)
    cfg = Krea2Config(features=256, layers=2, heads=2, kvheads=1, mlp_dim=768, tdim=64, txtdim=256,
                      txtheads=2, txtkvheads=2, txt_mlp_dim=768, txtlayers=3)
    model = Krea2DiT(cfg).float()
    for m in model.modules():
        if hasattr(m, "eps"):
            m.eps = 0.0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
        ctx = model.encode_text(torch.randn(1, 4, 3, 256))
        x, t = torch.randn(1, 16, 8, 8), torch.tensor([0.4])
        base = model(x, t, ctx)
        model.residual_scale = 0.25
        scaled = model(x, t, ctx)
    assert torch.allclose(base, scaled, rtol=1e-4, atol=1e-5)


@pytest.fixture(scope="module")
def golden():
    if not _GOLDEN.exists():
        pytest.skip(f"golden bundle not at {_GOLDEN}")
    from safetensors.torch import load_file
    return load_file(str(_GOLDEN)), json.loads((_GOLDEN.parent / "meta.json").read_text())


def test_tokenizer_matches_the_reference_ids(golden):
    g, meta = golden
    tok = Krea2Tokenizer()
    for name, prompt in meta["prompts"].items():
        ids, mask = g[f"tok.{name}.input_ids"].long(), g[f"tok.{name}.attention_mask"].bool()
        got = tok(prompt)
        assert got.ids[0].tolist() == ids[mask].tolist(), name
        assert got.truncated == (name == "long")
        pos = g[f"tok.{name}.position_ids"].long()[mask]
        assert pos.tolist() == list(range(int(mask.sum())))


def test_krea2_schedule_matches_the_reference_sigmas(golden):
    g, meta = golden
    for tag, run in meta["runs"].items():
        variant = "turbo" if tag.startswith("turbo") else "raw"
        sigmas, shift = krea2_sigmas("krea2", run["steps"], variant, run["w"], run["h"], 3.0, "cpu")
        assert torch.allclose(sigmas, g[f"{tag}.sigmas"], atol=2e-6), tag
    assert krea2_shift("turbo", 1024, 1024) == pytest.approx(math.exp(1.15))


_VAE = Path(os.environ.get(
    "DIFFUCORE_QWEN_IMAGE_VAE",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/diffucore-ui/models/vae/qwen_image_vae.safetensors",
))


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    """``load_krea2_checkpoint`` on tiny random DiT / TE files (TE keyed like
    the ComfyUI Qwen3-VL file, DiT named ``*turbo*``) and the real VAE."""
    if not _VAE.exists():
        pytest.skip(f"Qwen-Image VAE not at {_VAE}")
    from safetensors.torch import save_file
    from diffucore import load_krea2_checkpoint
    from diffucore.models import Qwen3Config, Qwen3TextEncoder
    d = tmp_path_factory.mktemp("krea2")
    torch.manual_seed(0)
    te = Qwen3TextEncoder(Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=36,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    te_sd = {"model.language_model." + k[len("model."):]: v for k, v in te.state_dict().items()}
    te_sd["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 4)
    save_file(te_sd, str(d / "te.safetensors"))
    dit = Krea2DiT(Krea2Config(features=128, layers=2, heads=2, kvheads=1, mlp_dim=256, tdim=32, txtdim=32,
                               txtheads=2, txtkvheads=2, txt_mlp_dim=64, txtlayers=12))
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(d / "tiny_turbo.safetensors"))
    return load_krea2_checkpoint(str(d / "tiny_turbo.safetensors"), str(d / "te.safetensors"), str(_VAE),
                                 device="cpu", dtype=torch.float32)


def test_tiny_bundle_generates_and_skips_unused_uncond(tiny_bundle, monkeypatch):
    from diffucore import TextToImage
    from diffucore.pipelines import _krea2
    from diffucore.runtime import ConditioningCache
    assert tiny_bundle.spec.variant == "turbo"
    assert tiny_bundle.text_encoder.cfg.num_hidden_layers == 35
    calls = []
    real = _krea2.krea2_encode
    monkeypatch.setattr(_krea2, "krea2_encode", lambda m, p, d: calls.append(p) or real(m, p, d))
    tiny_bundle.cond_cache = ConditioningCache()
    pipe = TextToImage(tiny_bundle)
    img = pipe("a cat", steps=2, width=64, height=48, seed=0, cfg_scale=1.0)   # no uncond pass
    assert img.size == (64, 48) and calls == ["a cat"]
    img2 = pipe("a cat", steps=2, width=64, height=48, seed=0, cfg_scale=5.5, sampler="dpmpp_2m")
    assert img2.size == (64, 48) and calls == ["a cat", "a cat", ""]
    pipe("a cat", steps=2, width=64, height=48, seed=0, cfg_scale=5.5)
    assert len(calls) == 3                                            # both contexts cached


def test_tiny_bundle_img2img_and_inpaint_keep_the_unmasked_pixels(tiny_bundle):
    import numpy as np
    from PIL import Image
    from diffucore import ImageToImage, Inpaint
    init = Image.fromarray((np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8))
    out = ImageToImage(tiny_bundle)("a cat", init, steps=2, strength=0.5, width=64, height=48, seed=0, cfg_scale=1.0)
    assert out.size == (64, 48)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[:, 32:] = 255
    out = Inpaint(tiny_bundle)("a cat", init, Image.fromarray(mask), steps=2, strength=1.0, width=64, height=48,
                               seed=0, cfg_scale=5.5, sampler="dpmpp_2m")
    assert np.array_equal(np.asarray(out)[:, :32], np.asarray(init)[:, :32])


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@_cuda
def test_file_backed_module_uploads_casts_and_repoints():
    from diffucore.runtime import file_backed, on_device
    lin = torch.nn.Linear(8, 4).to(torch.bfloat16)
    masters = {id(t): t.data for t in lin.parameters()}
    file_backed(lin, torch.float16)
    x = torch.randn(2, 8, device="cuda", dtype=torch.float16)
    with on_device(lin, torch.device("cuda")):
        assert lin.weight.device.type == "cuda" and lin.weight.dtype == torch.float16
        got = lin(x)
    want = torch.nn.functional.linear(x, masters[id(lin.weight)].cuda().half(), masters[id(lin.bias)].cuda().half())
    assert torch.equal(got, want)
    assert all(t.data_ptr() == masters[id(t)].data_ptr() and t.dtype == torch.bfloat16 for t in lin.parameters())


def test_file_masters_are_read_ahead_in_window_sized_chunks(monkeypatch):
    import mmap
    from diffucore import runtime
    calls = []
    monkeypatch.setattr(runtime, "_MADVISE", lambda p, n, advice: calls.append((p, n, advice)))
    lin = torch.nn.Linear(256, 300)    # 300 KB weight: three chunks
    runtime.file_backed(lin)
    with runtime.on_device(lin, torch.device("cpu")):
        pass
    assert {advice for _, _, advice in calls} == {mmap.MADV_WILLNEED}
    assert all(p % mmap.PAGESIZE == 0 and n <= runtime._READAHEAD_CHUNK for p, n, _ in calls)
    for t in lin.parameters():
        start, end = t.data_ptr(), t.data_ptr() + t.nbytes
        spans = sorted((p, p + n) for p, n, _ in calls if p < end and p + n > start)
        assert spans[0][0] <= start and spans[-1][1] == end
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


@_cuda
def test_file_master_streaming_matches_resident():
    from diffucore.runtime import stream_blocks
    torch.manual_seed(0)
    cfg = Krea2Config(features=256, layers=3, heads=2, kvheads=1, mlp_dim=768, tdim=64, txtdim=256,
                      txtheads=2, txtkvheads=2, txt_mlp_dim=768, txtlayers=3)
    model = Krea2DiT(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    model = model.to(torch.bfloat16).eval()
    ref = Krea2DiT(cfg).half().cuda().eval()
    ref.load_state_dict(model.state_dict())
    block_masters = [t.data for t in model.blocks[1:].parameters()]
    stream_blocks(model, ("blocks",), torch.device("cuda"), keep_resident=(model.blocks[0],),
                  masters="file", dtype=torch.float16)
    assert model.blocks[0].mod.lin.device.type == "cuda"
    feats = torch.randn(1, 4, 3, 256, device="cuda")
    x, t = torch.randn(1, 16, 8, 8, device="cuda"), torch.tensor([0.5], device="cuda")
    with torch.inference_mode():
        want = ref(x, t, ref.encode_text(feats))
        got = model(x, t, model.encode_text(feats))
    assert torch.equal(got, want)
    assert all(t.data_ptr() == m.data_ptr() and t.dtype == torch.bfloat16
               for t, m in zip(model.blocks[1:].parameters(), block_masters))


@_cuda
def test_layer_streamed_file_backed_text_encoder_matches_resident():
    from diffucore.models import Qwen3Config, Qwen3TextEncoder
    from diffucore.runtime import file_backed, on_device, stream_layers
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=32)
    te = Qwen3TextEncoder(cfg).eval()
    for p in te.parameters():          # bf16 weights like the file; RoPE inv_freq stays fp32
        p.data = p.data.to(torch.bfloat16)
    ref = Qwen3TextEncoder(cfg).half().cuda().eval()
    ref.load_state_dict(te.state_dict())
    masters = [t.data for t in te.model.layers.parameters()]
    file_backed(te, torch.float16, exclude=(te.model.layers,))
    stream_layers(te.model.layers, torch.device("cuda"), torch.float16)
    ids = torch.randint(0, 1000, (1, 12), device="cuda")
    with torch.no_grad(), on_device(te, torch.device("cuda")):
        assert te.model.embed_tokens.weight.device.type == "cuda"
        assert te.model.layers[0].self_attn.q_proj.weight.device.type == "cpu"
        got = te(ids, hidden_layers=[2, 5])
    want = ref(ids, hidden_layers=[2, 5])
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert te.model.embed_tokens.weight.device.type == "cpu"
    assert all(t.data_ptr() == m.data_ptr() for t, m in zip(te.model.layers.parameters(), masters))


def test_auto_masters_go_to_file_when_ram_is_short(monkeypatch, tmp_path):
    from diffucore import bundle
    from diffucore.runtime import DevicePolicy
    f = tmp_path / "w.bin"
    f.write_bytes(b"\0" * 1024)
    policy = DevicePolicy(device=torch.device("cpu"), offload="stream")
    monkeypatch.setattr(bundle, "_available_ram", lambda: 2 * 2**30)
    assert bundle._krea2_masters(policy, str(f)) == "file"
    monkeypatch.setattr(bundle, "_available_ram", lambda: 64 * 2**30)
    assert bundle._krea2_masters(policy, str(f)) == "pinned"
    assert bundle._krea2_masters(DevicePolicy(device=torch.device("cpu"), stream_masters="file"), str(f)) == "file"


def _tiny_krea2(seed: int = 0) -> Krea2DiT:
    torch.manual_seed(seed)
    cfg = Krea2Config(features=256, layers=2, heads=2, kvheads=1, mlp_dim=768, tdim=64, txtdim=256,
                      txtheads=2, txtkvheads=2, txt_mlp_dim=768, txtlayers=3)
    model = Krea2DiT(cfg).float()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return model.eval()


def test_diffusers_names_map_onto_the_original_layout():
    from diffucore.models.krea2_dit import from_diffusers_name
    ours = set(_build(Krea2DiT, Krea2Config()).state_dict())
    assert {from_diffusers_name(k) for k in _HEADERS["diffusers_keys"]} == ours


def test_krea2_lora_hooks_match_a_fused_lora_in_all_three_formats(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from diffucore.lora import apply_lora, clear_loras, remove_lora
    from diffucore.runtime import ConditioningCache, DevicePolicy
    model, fused = _tiny_krea2(), _tiny_krea2()
    torch.manual_seed(1)
    r = 4
    layers = {  # original layer name -> the LoRA key base naming it
        "blocks.0.attn.wq": "diffusion_model.blocks.0.attn.wq",
        "blocks.1.mlp.down": "lora_unet_blocks_1_mlp_down",
        "txtfusion.refiner_blocks.0.attn.wo": "transformer.text_fusion.refiner_blocks.0.attn.to_out.0",
        "tproj.1": "transformer.time_mod_proj",
    }
    lora = {}
    for name, key in layers.items():
        lin = model.get_submodule(name)
        down, up = torch.randn(r, lin.in_features) * 0.1, torch.randn(lin.out_features, r) * 0.1
        if key.startswith("transformer."):
            lora[key + ".lora_A.weight"], lora[key + ".lora_B.weight"] = down, up
        else:
            lora[key + ".lora_down.weight"], lora[key + ".lora_up.weight"] = down, up
            lora[key + ".alpha"] = torch.tensor(2.0)
        scale = 0.7 * (2.0 / r if not key.startswith("transformer.") else 1.0)
        with torch.no_grad():
            fused.get_submodule(name).weight += scale * up @ down
    lora["diffusion_model.blocks.0.attn.nonexistent.lora_down.weight"] = torch.zeros(r, 4)
    lora["diffusion_model.blocks.0.attn.nonexistent.lora_up.weight"] = torch.zeros(4, r)
    save_file(lora, str(tmp_path / "l.safetensors"))

    cache = ConditioningCache()
    cache.put(("krea2", "p", ""), {"cond_ctx": torch.zeros(1), "uncond_ctx": None})
    bundle = SimpleNamespace(spec=SimpleNamespace(architecture="krea2"), backbone=model, cond_cache=cache,
                             policy=DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32))
    feats, x, t = torch.randn(1, 4, 3, 256), torch.randn(1, 16, 8, 8), torch.tensor([0.6])
    with torch.no_grad():
        base = model(x, t, model.encode_text(feats))
        report = apply_lora(bundle, str(tmp_path / "l.safetensors"), 0.7)
        assert report.applied == 4 and report.unmatched == ["diffusion_model.blocks.0.attn.nonexistent"]
        assert cache.get(("krea2", "p", "")) is None
        got = model(x, t, model.encode_text(feats))
        want = fused(x, t, fused.encode_text(feats))
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)
        remove_lora(bundle, str(tmp_path / "l.safetensors"))
        assert torch.equal(model(x, t, model.encode_text(feats)), base)
        apply_lora(bundle, str(tmp_path / "l.safetensors"), 0.7)
        clear_loras(bundle)
        assert torch.equal(model(x, t, model.encode_text(feats)), base)


_TO_DIFFUSERS = [(re.compile(p), r) for p, r in [
    (r"\.mod\.lin$", ".scale_shift_table"),
    (r"\.prenorm\.scale$", ".norm1.weight"), (r"\.postnorm\.scale$", ".norm2.weight"),
    (r"\.mlp\.", ".ff."),
    (r"\.attn\.qknorm\.qnorm\.scale$", ".attn.norm_q.weight"),
    (r"\.attn\.qknorm\.knorm\.scale$", ".attn.norm_k.weight"),
    (r"\.attn\.wo\b", ".attn.to_out.0"), (r"\.attn\.gate\b", ".attn.to_gate"),
    (r"\.attn\.wv\b", ".attn.to_v"), (r"\.attn\.wk\b", ".attn.to_k"), (r"\.attn\.wq\b", ".attn.to_q"),
    (r"^txtmlp\.3\b", "txt_in.linear_2"), (r"^txtmlp\.1\b", "txt_in.linear_1"),
    (r"^txtmlp\.0\.scale$", "txt_in.norm.weight"),
    (r"^tproj\.1\b", "time_mod_proj"), (r"^tmlp\.2\b", "time_embed.linear_2"), (r"^tmlp\.0\b", "time_embed.linear_1"),
    (r"^last\.modulation\.lin$", "final_layer.scale_shift_table"),
    (r"^last\.norm\.scale$", "final_layer.norm.weight"), (r"^last\.linear\b", "final_layer.linear"),
    (r"^first\b", "img_in"), (r"^txtfusion\.", "text_fusion."), (r"^blocks\.", "transformer_blocks."),
]]


def _to_diffusers(name: str) -> str:
    for pattern, repl in _TO_DIFFUSERS:
        name = pattern.sub(repl, name)
    return name


def test_loads_a_sharded_diffusers_transformer(tmp_path):
    """The same weights as diffusers shards (renamed, [6, F] block modulation,
    split over two files + an index) load to the same model, from the folder,
    its parent repo or the index."""
    from safetensors.torch import save_file
    from diffucore.bundle import load_krea2_dit, _krea2_dit_state
    from diffucore.models.krea2_dit import from_diffusers_name
    from diffucore.runtime import DevicePolicy
    model = _tiny_krea2()
    sd = model.state_dict()
    assert all(from_diffusers_name(_to_diffusers(k)) == k for k in _build(Krea2DiT, Krea2Config()).state_dict())
    dsd = {}
    for k, v in sd.items():
        dk = _to_diffusers(k)
        dsd[dk] = v.reshape(6, -1) if dk.startswith("transformer_blocks") and dk.endswith("scale_shift_table") else v
    d = tmp_path / "Krea-2-Turbo" / "transformer"
    d.mkdir(parents=True)
    keys = sorted(dsd)
    shards = {"m-1.safetensors": keys[: len(keys) // 2], "m-2.safetensors": keys[len(keys) // 2:]}
    for f, ks in shards.items():
        save_file({k: dsd[k].contiguous() for k in ks}, str(d / f))
    (d / "m.safetensors.index.json").write_text(json.dumps({"weight_map": {k: f for f, ks in shards.items() for k in ks}}))
    assert set(_krea2_dit_state(str(d))) == set(sd)
    feats, x, t = torch.randn(1, 4, 3, 256), torch.randn(1, 16, 8, 8), torch.tensor([0.3])
    for path in (str(d), str(d.parent), str(d / "m.safetensors.index.json")):
        loaded = load_krea2_dit(path, DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32),
                                lambda m: None)
        with torch.no_grad():
            assert torch.equal(loaded(x, t, loaded.encode_text(feats)), model(x, t, model.encode_text(feats)))
