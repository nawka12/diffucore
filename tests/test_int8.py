"""INT8 (W8A8, ConvRot) linears: ComfyUI's int8_tensorwise format, load-time
quantization, LoRA re-quantization, and (CUDA + Triton) agreement of the fused
kernels with the torch-op path."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from diffucore.bundle import ModelBundle
from diffucore.loading import ModelSpec
from diffucore.lora import apply_lora, clear_loras
from diffucore.models import _int8
from diffucore.models._int8 import (Int8Linear, adopt_int8_layers, dequantize_linears, dequantize_weight,
                                    int8_layers, int8_mlp, quantize_linears, quantize_weight)
from diffucore.models.anima_dit import CosmosDiT, CosmosDiTConfig

fast = pytest.mark.skipif(not _int8.int8_fast_available(), reason="needs CUDA sm75+ and Triton")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _comfy_layer(w: torch.Tensor, convrot: bool) -> dict:
    """One layer as ComfyUI writes it: int8 weight, [N, 1] fp32 scale, JSON tag."""
    q, s = quantize_weight(w, 256 if convrot else 0)
    meta = {"format": "int8_tensorwise"}
    if convrot:
        meta.update(convrot=True, convrot_groupsize=256)
    return {"weight": q, "weight_scale": s.unsqueeze(1),
            "comfy_quant": torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)}


def test_hadamard_is_symmetric_and_self_inverse():
    h = _int8._hadamard(256, "cpu")
    torch.testing.assert_close(h, h.T)
    torch.testing.assert_close(h @ h, torch.eye(256), atol=1e-6, rtol=0)


def test_weight_round_trip():
    w = torch.randn(64, 512) * 0.02
    for group in (0, 256):
        q, s = quantize_weight(w, group)
        assert _rel(dequantize_weight(q, s, group), w) < 0.01


def test_forward_matches_float_linear():
    torch.manual_seed(0)
    lin = nn.Linear(512, 96, bias=True)
    x = torch.randn(2, 40, 512)
    m = Int8Linear.from_linear(lin)
    assert _rel(m(x), lin(x)) < 0.02


def test_scale_survives_dtype_cast():
    m = Int8Linear.from_linear(nn.Linear(256, 32))
    s = m.scale.clone()
    m.to(torch.float16)
    assert m.weight.dtype == torch.int8 and torch.equal(m.scale, s)


def test_short_input_is_padded_for_int_mm():
    lin = nn.Linear(256, 64, bias=False)
    x = torch.randn(3, 256)
    assert _rel(Int8Linear.from_linear(lin)(x), lin(x)) < 0.02


def test_adopts_comfy_int8_checkpoint():
    torch.manual_seed(1)
    ref = nn.Sequential(nn.Linear(512, 256, bias=False), nn.Linear(256, 128, bias=True))
    sd = {"1.bias": ref[1].bias.detach().clone()}
    for name, mod, convrot in (("0", ref[0], True), ("1", ref[1], False)):
        sd.update({f"{name}.{k}": v for k, v in _comfy_layer(mod.weight.detach(), convrot).items()})

    layers = int8_layers(sd)
    assert layers == {"0": 256, "1": 0}
    with torch.device("meta"):
        model = nn.Sequential(nn.Linear(512, 256, bias=False), nn.Linear(256, 128, bias=True))
    model.load_state_dict(adopt_int8_layers(model, sd, layers), strict=True, assign=True)
    assert isinstance(model[0], Int8Linear) and model[0].group == 256 and model[1].group == 0

    x = torch.randn(8, 512)
    assert _rel(model(x), ref(x)) < 0.03
    dequantize_linears(model, torch.float32)
    assert type(model[0]) is nn.Linear and _rel(model(x), ref(x)) < 0.02


def test_rejects_other_quantized_formats():
    tag = torch.tensor(list(json.dumps({"format": "nvfp4"}).encode()), dtype=torch.uint8)
    with pytest.raises(ValueError, match="nvfp4"):
        int8_layers({"blocks.0.mlp.layer1.comfy_quant": tag})


def _tiny_anima():
    torch.manual_seed(2)
    return CosmosDiT(CosmosDiTConfig(model_channels=256, num_blocks=1, num_heads=2, head_dim=128,
                                     crossattn_emb_channels=256, adaln_lora_dim=4, mlp_ratio=2.0,
                                     max_img_h=16, max_img_w=16, max_frames=1))


def test_lora_requantizes_unbiased_and_restores_bytes(tmp_path):
    """A LoRA far below one int8 step still lands (stochastic rounding keeps
    its mean); clearing restores the original int8 bytes and scales."""
    dit = _tiny_anima()
    name = "blocks.0.mlp.layer1"
    quantize_linears(dit, [name], "cpu")
    layer = dit.get_submodule(name)
    q0, s0 = layer.weight.clone(), layer.scale.clone()
    before = dequantize_weight(q0, s0, layer.group)

    rank = 4
    g = torch.Generator().manual_seed(3)
    down = torch.randn(rank, layer.in_features, generator=g)
    up = torch.randn(layer.out_features, rank, generator=g)
    step = s0.mean().item()
    k = 0.2 * step / (up @ down).std().item()          # delta ~0.2 int8 steps
    path = str(tmp_path / "lora.safetensors")
    save_file({f"diffusion_model.{name}.lora_A.weight": down * k,
               f"diffusion_model.{name}.lora_B.weight": up}, path)
    spec = ModelSpec(architecture="anima", prediction="flow", zero_terminal_snr=False,
                     latent_channels=16, context_dim=1024)
    bundle = ModelBundle(spec=spec, schedule=None, tokenizer=None, text_encoder=None,
                         backbone=dit, vae=None)

    assert apply_lora(bundle, path).applied == 1
    delta = (up @ down) * k
    change = dequantize_weight(layer.weight, layer.scale, layer.group) - before
    gain = (change * delta).sum() / (delta * delta).sum()
    assert 0.9 < gain.item() < 1.1

    clear_loras(bundle)
    assert torch.equal(layer.weight, q0) and torch.equal(layer.scale, s0)


@fast
def test_kernels_match_torch_ops(monkeypatch):
    torch.manual_seed(4)
    for k, n in ((2048, 2048), (768, 512), (1024, 8192)):
        lin = nn.Linear(k, n, bias=n == 512).cuda().half()
        m = Int8Linear.from_linear(lin)
        x = torch.randn(3, 333, k, device="cuda", dtype=torch.float16)
        y = m(x)
        with monkeypatch.context() as mp:
            mp.setattr(_int8, "_fast", lambda t: False)
            y_ref = m(x)
        assert y.dtype == torch.float16 and _rel(y, y_ref) < 2e-3
        assert _rel(y, lin(x)) < 0.02


@fast
def test_fused_mlp_matches_unfused(monkeypatch):
    torch.manual_seed(5)
    l1 = Int8Linear.from_linear(nn.Linear(1024, 4096, bias=False).cuda().half())
    l2 = Int8Linear.from_linear(nn.Linear(4096, 1024, bias=False).cuda().half())
    x = torch.randn(2, 500, 1024, device="cuda", dtype=torch.float16)
    y = int8_mlp(x, l1, l2)
    with monkeypatch.context() as mp:
        mp.setattr(_int8, "_fast", lambda t: False)
        y_ref = l2(F.gelu(l1(x)))
    assert _rel(y, y_ref) < 2e-3


fused_gemm = pytest.mark.skipif(
    not (_int8.int8_fast_available() and _int8.fused_gemm_applies("cuda") and not _int8.fused_gemm_error("cuda")),
    reason="needs an sm75 GPU, the CUTLASS headers and a CUDA toolchain")


@fused_gemm
@pytest.mark.parametrize("m,n,k", [(33, 2048, 1024), (300, 512, 768), (4096, 2048, 2048), (1000, 8192, 2048)])
def test_fused_gemm_is_bit_exact_with_rescale(m, n, k):
    """The epilogue rounds exactly like the int32 GEMM + rescale pass."""
    torch.manual_seed(6)
    q = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    sx, sw = torch.rand(m, device="cuda") * 1e-2, torch.rand(n, device="cuda") * 1e-3
    out = _int8.scaled_mm(q, w, sx, sw, torch.float16)
    assert out is not None and torch.equal(out, _int8.rescale(_int8.int_mm(q, w), sx, sw, None, torch.float16))


@fused_gemm
def test_fused_gemm_mlp_matches_triton_gelu(monkeypatch):
    torch.manual_seed(5)
    l1 = Int8Linear.from_linear(nn.Linear(1024, 4096, bias=False).cuda().half())
    l2 = Int8Linear.from_linear(nn.Linear(4096, 1024, bias=False).cuda().half())
    x = torch.randn(2, 500, 1024, device="cuda", dtype=torch.float16)
    y = int8_mlp(x, l1, l2)
    with monkeypatch.context() as mp:
        mp.setattr(_int8, "scaled_mm", lambda *a, **k: None)
        y_ref = int8_mlp(x, l1, l2)
    assert torch.equal(y, y_ref)


@fast
def test_selfattn_int8_shares_one_quantized_input():
    from diffucore.models.anima_dit import _Attention
    torch.manual_seed(7)
    attn = _Attention(512, None, 4, 128, 1e-6).cuda().half().eval()
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(attn, name, Int8Linear.from_linear(getattr(attn, name)))
    x = torch.randn(1, 300, 512, device="cuda", dtype=torch.float16)
    for got, ref in zip(attn._qkv(x, x), (attn.q_proj(x), attn.k_proj(x), attn.v_proj(x))):
        assert torch.equal(got, ref)
