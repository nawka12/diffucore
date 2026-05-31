"""LoRA fusing (kohya format) verified against tiny real-structure models.

Full SD/SDXL backbones are too heavy for the CPU suite, so these build the same
module structure at small dimensions — the kohya key names depend on structure,
not size, so the name-mangling is exercised for real.
"""

import torch
from safetensors.torch import save_file

from diffucore.bundle import ModelBundle
from diffucore.loading import ModelSpec
from diffucore.lora import apply_lora
from diffucore.models.clip_text import CLIPTextConfig, CLIPTextEncoder
from diffucore.models.open_clip_text import OpenCLIPTextConfig, OpenCLIPTextEncoder
from diffucore.models.anima_dit import CosmosDiT, CosmosDiTConfig
from diffucore.models.unet import UNetConfig, UNetModel


def _tiny_unet():
    return UNetModel(UNetConfig(model_channels=32, channel_mult=(1, 2),
                                num_res_blocks=1, context_dim=64, num_heads=4))


def _tiny_clip():
    return CLIPTextEncoder(CLIPTextConfig(vocab_size=10, hidden_size=64, num_layers=1,
                                          num_heads=4, intermediate_size=128,
                                          max_position_embeddings=16))


def _tiny_bigg():
    return OpenCLIPTextEncoder(OpenCLIPTextConfig(vocab_size=10, width=64, num_layers=1,
                                                  num_heads=4, mlp_dim=128,
                                                  max_position_embeddings=16))


def _bundle(backbone, text_encoder, text_encoder_2=None):
    spec = ModelSpec(architecture="sd15", prediction="eps", zero_terminal_snr=False,
                     latent_channels=4, context_dim=64)
    return ModelBundle(spec=spec, schedule=None, tokenizer=None,
                       text_encoder=text_encoder, backbone=backbone, vae=None,
                       text_encoder_2=text_encoder_2)


def _pair(weight_shape, rank, gen):
    """A (down, up) LoRA pair for a target of ``weight_shape`` and its ΔW (unscaled)."""
    out = weight_shape[0]
    if len(weight_shape) == 4:                     # conv
        _, cin, kh, kw = weight_shape
        down = torch.randn(rank, cin, kh, kw, generator=gen)
        up = torch.randn(out, rank, 1, 1, generator=gen)
    else:                                          # linear
        cin = weight_shape[1]
        down = torch.randn(rank, cin, generator=gen)
        up = torch.randn(out, rank, generator=gen)
    delta = (up.reshape(out, -1) @ down.reshape(rank, -1)).reshape(weight_shape)
    return down, up, delta


def _write_lora(tmp_path, entries, rank, gen):
    """Build a kohya LoRA file from ``{name: weight_shape}``; return (path, {name: ΔW})."""
    tensors, deltas = {}, {}
    for name, shape in entries.items():
        down, up, delta = _pair(shape, rank, gen)
        tensors[f"{name}.lora_down.weight"] = down
        tensors[f"{name}.lora_up.weight"] = up
        tensors[f"{name}.alpha"] = torch.tensor([float(rank)])  # alpha=rank -> scale=mult
        deltas[name] = delta
    path = str(tmp_path / "lora.safetensors")
    save_file(tensors, path)
    return path, deltas


def test_fuses_unet_and_clip_deltas(tmp_path):
    unet, clip = _tiny_unet(), _tiny_clip()
    bundle = _bundle(unet, clip)

    unet_q = unet.input_blocks[1][1].transformer_blocks[0].attn1.to_q
    clip_q = clip.text_model.encoder.layers[0].self_attn.q_proj
    before_unet = unet_q.weight.detach().clone()
    before_clip = clip_q.weight.detach().clone()

    gen = torch.Generator().manual_seed(0)
    path, deltas = _write_lora(tmp_path, {
        "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q": tuple(before_unet.shape),
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj": tuple(before_clip.shape),
    }, rank=4, gen=gen)

    report = apply_lora(bundle, path)

    assert report.applied == 2 and report.unmatched == []
    torch.testing.assert_close(
        unet_q.weight - before_unet,
        deltas["lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q"])
    torch.testing.assert_close(
        clip_q.weight - before_clip,
        deltas["lora_te_text_model_encoder_layers_0_self_attn_q_proj"])


def test_fuses_through_compile_wrapper(tmp_path):
    """LoRA must find its targets even when the backbone is wrapped by
    ``torch.compile`` (which yields an OptimizedModule whose ``named_modules``
    prefixes everything with ``_orig_mod.``). We simulate the wrapper directly
    so the test stays platform-independent (Inductor isn't required here)."""
    class _OptimizedShim(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self._orig_mod = orig

    unet = _tiny_unet()
    wrapped = _OptimizedShim(unet)
    bundle = _bundle(wrapped, None)

    unet_q = unet.input_blocks[1][1].transformer_blocks[0].attn1.to_q
    before = unet_q.weight.detach().clone()
    gen = torch.Generator().manual_seed(7)
    path, deltas = _write_lora(
        tmp_path,
        {"lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q": tuple(before.shape)},
        rank=4, gen=gen,
    )
    report = apply_lora(bundle, path)
    assert report.applied == 1 and report.unmatched == []
    torch.testing.assert_close(
        unet_q.weight - before,
        deltas["lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q"],
    )


def test_fuses_unet_conv_proj_in(tmp_path):
    unet = _tiny_unet()
    bundle = _bundle(unet, None)
    conv = unet.input_blocks[1][1].proj_in   # SD1.5: 1x1 Conv2d
    assert isinstance(conv, torch.nn.Conv2d)
    before = conv.weight.detach().clone()

    gen = torch.Generator().manual_seed(1)
    path, deltas = _write_lora(
        tmp_path,
        {"lora_unet_input_blocks_1_1_proj_in": tuple(before.shape)}, rank=4, gen=gen)

    report = apply_lora(bundle, path)
    assert report.applied == 1
    torch.testing.assert_close(conv.weight - before, deltas["lora_unet_input_blocks_1_1_proj_in"])


def test_bigg_q_proj_targets_in_proj_row_slice(tmp_path):
    bigg = _tiny_bigg()
    bundle = _bundle(_tiny_unet(), _tiny_clip(), text_encoder_2=bigg)
    in_proj = bigg.transformer.resblocks[0].attn.in_proj_weight
    # in_proj_weight is torch.empty (normally filled by load_state_dict); give it
    # finite values so the unchanged-rows comparison isn't NaN - NaN.
    torch.nn.init.normal_(in_proj, std=0.02)
    width = in_proj.shape[1]
    before = in_proj.detach().clone()

    gen = torch.Generator().manual_seed(2)
    name = "lora_te2_text_model_encoder_layers_0_self_attn_q_proj"
    path, deltas = _write_lora(tmp_path, {name: (width, width)}, rank=4, gen=gen)

    report = apply_lora(bundle, path)
    assert report.applied == 1 and report.unmatched == []
    # The q delta lands in the first `width` rows; k/v rows are untouched.
    torch.testing.assert_close(in_proj[:width] - before[:width], deltas[name])
    torch.testing.assert_close(in_proj[width:], before[width:])


def test_multiplier_and_alpha_scale_the_delta(tmp_path):
    unet = _tiny_unet()
    bundle = _bundle(unet, None)
    target = unet.input_blocks[1][1].transformer_blocks[0].attn1.to_q
    before = target.weight.detach().clone()

    gen = torch.Generator().manual_seed(3)
    name = "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q"
    # alpha = 2*rank, so scale = (alpha/rank) * multiplier = 2 * 0.5 = 1.0
    down, up, delta = _pair(tuple(before.shape), rank=4, gen=gen)
    tensors = {f"{name}.lora_down.weight": down, f"{name}.lora_up.weight": up,
               f"{name}.alpha": torch.tensor([8.0])}
    path = str(tmp_path / "lora.safetensors")
    save_file(tensors, path)

    apply_lora(bundle, path, multiplier=0.5)
    torch.testing.assert_close(target.weight - before, delta)


def test_fuses_lokr_full_matrices(tmp_path):
    """LoKr with full w1/w2: ΔW = kron(w1, w2), no alpha scaling (scale 1)."""
    unet, clip = _tiny_unet(), _tiny_clip()
    bundle = _bundle(unet, clip)
    target = clip.text_model.encoder.layers[0].self_attn.q_proj  # [64, 64]
    before = target.weight.detach().clone()

    gen = torch.Generator().manual_seed(5)
    name = "lora_te_text_model_encoder_layers_0_self_attn_q_proj"
    # 64 = 8 * 8: pick w1 (8,8), w2 (8,8) so kron -> (64, 64).
    w1 = torch.randn(8, 8, generator=gen)
    w2 = torch.randn(8, 8, generator=gen)
    tensors = {f"{name}.lokr_w1": w1, f"{name}.lokr_w2": w2,
               f"{name}.alpha": torch.tensor(8.0)}  # ignored: no _b factor
    path = str(tmp_path / "lokr.safetensors")
    save_file(tensors, path)

    report = apply_lora(bundle, path)
    assert report.applied == 1 and report.unmatched == []
    torch.testing.assert_close(target.weight - before, torch.kron(w1, w2))


def test_fuses_lokr_conv_kernel(tmp_path):
    """LoKr on a 3x3 conv: w2 carries the kernel dims, w1 is lifted to 4-D."""
    unet = _tiny_unet()
    bundle = _bundle(unet, None)
    conv = unet.input_blocks[1][0].in_layers[2]   # ResBlock conv, [32, 32, 3, 3]
    assert isinstance(conv, torch.nn.Conv2d) and conv.weight.shape == (32, 32, 3, 3)
    before = conv.weight.detach().clone()

    gen = torch.Generator().manual_seed(6)
    name = "lora_unet_input_blocks_1_0_in_layers_2"
    w1 = torch.randn(8, 8, generator=gen)              # (out1, in1)
    w2 = torch.randn(4, 4, 3, 3, generator=gen)        # (out2, in2, kh, kw)
    tensors = {f"{name}.lokr_w1": w1, f"{name}.lokr_w2": w2, f"{name}.alpha": torch.tensor(8.0)}
    path = str(tmp_path / "lokr.safetensors")
    save_file(tensors, path)

    report = apply_lora(bundle, path)
    assert report.applied == 1
    expected = torch.kron(w1.reshape(8, 8, 1, 1), w2)
    torch.testing.assert_close(conv.weight - before, expected)


def test_lokr_low_rank_factor_uses_alpha_over_dim(tmp_path):
    """When w2 is low-rank (w2_a @ w2_b), scale = alpha / dim with dim = rank."""
    unet, clip = _tiny_unet(), _tiny_clip()
    bundle = _bundle(unet, clip)
    target = clip.text_model.encoder.layers[0].self_attn.q_proj  # [64, 64]
    before = target.weight.detach().clone()

    gen = torch.Generator().manual_seed(7)
    name = "lora_te_text_model_encoder_layers_0_self_attn_q_proj"
    w1 = torch.randn(8, 8, generator=gen)
    rank = 2
    w2_a = torch.randn(8, rank, generator=gen)
    w2_b = torch.randn(rank, 8, generator=gen)
    alpha = 4.0
    tensors = {f"{name}.lokr_w1": w1, f"{name}.lokr_w2_a": w2_a,
               f"{name}.lokr_w2_b": w2_b, f"{name}.alpha": torch.tensor(alpha)}
    path = str(tmp_path / "lokr.safetensors")
    save_file(tensors, path)

    report = apply_lora(bundle, path)
    assert report.applied == 1
    expected = torch.kron(w1, w2_a @ w2_b) * (alpha / rank)
    torch.testing.assert_close(target.weight - before, expected)


def test_anima_dotted_lora_a_b(tmp_path):
    """Anima DiT LoRA: dotted module paths under ``diffusion_model.`` with
    lora_A/lora_B factors and no alpha (scale 1)."""
    dit = CosmosDiT(CosmosDiTConfig(model_channels=48, num_blocks=1, num_heads=2,
                                    head_dim=24, crossattn_emb_channels=12,
                                    adaln_lora_dim=4, mlp_ratio=2.0,
                                    max_img_h=16, max_img_w=16, max_frames=1))
    spec = ModelSpec(architecture="anima", prediction="flow", zero_terminal_snr=False,
                     latent_channels=16, context_dim=1024)
    bundle = ModelBundle(spec=spec, schedule=None, tokenizer=None, text_encoder=None,
                         backbone=dit, vae=None)

    target = dit.blocks[0].self_attn.q_proj      # Linear(48, 48), bias=False
    before = target.weight.detach().clone()
    dim = target.weight.shape[0]

    gen = torch.Generator().manual_seed(8)
    rank = 4
    name = "diffusion_model.blocks.0.self_attn.q_proj"
    down = torch.randn(rank, dim, generator=gen)
    up = torch.randn(dim, rank, generator=gen)
    tensors = {f"{name}.lora_A.weight": down, f"{name}.lora_B.weight": up}
    path = str(tmp_path / "anima_lora.safetensors")
    save_file(tensors, path)

    report = apply_lora(bundle, path)
    assert report.applied == 1 and report.unmatched == []
    torch.testing.assert_close(target.weight - before, up @ down)


def test_anima_kohya_mangled_lokr(tmp_path):
    """Anima LoKr in the LyCORIS/kohya convention: lora_unet_<mangled> + lokr_w1/w2."""
    dit = CosmosDiT(CosmosDiTConfig(model_channels=48, num_blocks=1, num_heads=2,
                                    head_dim=24, crossattn_emb_channels=12,
                                    adaln_lora_dim=4, mlp_ratio=2.0,
                                    max_img_h=16, max_img_w=16, max_frames=1))
    spec = ModelSpec(architecture="anima", prediction="flow", zero_terminal_snr=False,
                     latent_channels=16, context_dim=1024)
    bundle = ModelBundle(spec=spec, schedule=None, tokenizer=None, text_encoder=None,
                         backbone=dit, vae=None)

    target = dit.blocks[0].self_attn.q_proj      # Linear(48, 48)
    before = target.weight.detach().clone()

    gen = torch.Generator().manual_seed(9)
    name = "lora_unet_blocks_0_self_attn_q_proj"   # mangled DiT path
    w1 = torch.randn(6, 6, generator=gen)          # 48 = 6 * 8
    w2 = torch.randn(8, 8, generator=gen)
    tensors = {f"{name}.lokr_w1": w1, f"{name}.lokr_w2": w2, f"{name}.alpha": torch.tensor(6.0)}
    path = str(tmp_path / "anima_lokr.safetensors")
    save_file(tensors, path)

    report = apply_lora(bundle, path)
    assert report.applied == 1 and report.unmatched == []
    torch.testing.assert_close(target.weight - before, torch.kron(w1, w2))


def test_unmatched_keys_reported_not_applied(tmp_path):
    unet = _tiny_unet()
    bundle = _bundle(unet, None)

    gen = torch.Generator().manual_seed(4)
    path, _ = _write_lora(tmp_path, {"lora_unet_does_not_exist": (8, 8)}, rank=4, gen=gen)

    report = apply_lora(bundle, path)
    assert report.applied == 0
    assert report.unmatched == ["lora_unet_does_not_exist"]
