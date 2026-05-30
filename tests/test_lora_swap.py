"""LoRA removal / swapping — verifying the snapshot-and-refuse unload path.

Builds on the same tiny real-structure models as ``test_lora.py``: ``apply_lora``
fuses deltas in place, and ``remove_lora`` / ``clear_loras`` undo a fuse by
restoring a pristine snapshot and replaying the remaining stack, so every result
is bit-identical to a fresh apply of just the surviving LoRAs.
"""

import pytest
import torch
from safetensors.torch import save_file

from diffucore.bundle import ModelBundle
from diffucore.loading import ModelSpec
from diffucore.lora import apply_lora, clear_loras, remove_lora
from diffucore.models.clip_text import CLIPTextConfig, CLIPTextEncoder
from diffucore.models.open_clip_text import OpenCLIPTextConfig, OpenCLIPTextEncoder
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


_Q_PROJ = "lora_te_text_model_encoder_layers_0_self_attn_q_proj"


def _write_lora(path, name, shape, rank, seed):
    """A kohya LoRA file for one target, alpha=rank so scale == multiplier."""
    gen = torch.Generator().manual_seed(seed)
    out, cin = shape
    down = torch.randn(rank, cin, generator=gen)
    up = torch.randn(out, rank, generator=gen)
    save_file({f"{name}.lora_down.weight": down, f"{name}.lora_up.weight": up,
               f"{name}.alpha": torch.tensor([float(rank)])}, str(path))


def _clip_q(bundle):
    return bundle.text_encoder.text_model.encoder.layers[0].self_attn.q_proj


def test_remove_lora_matches_fresh_apply_of_remainder(tmp_path):
    """Applying A+B then removing A leaves exactly orig + B (bit-identical to a
    bundle that only ever saw B)."""
    bundle = _bundle(_tiny_unet(), _tiny_clip())
    target = _clip_q(bundle)
    orig = target.weight.detach().clone()
    path_a, path_b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _write_lora(path_a, _Q_PROJ, tuple(orig.shape), rank=4, seed=1)
    _write_lora(path_b, _Q_PROJ, tuple(orig.shape), rank=2, seed=2)

    apply_lora(bundle, str(path_a), multiplier=0.8)
    apply_lora(bundle, str(path_b), multiplier=0.5)
    remove_lora(bundle, str(path_a))

    ref = _bundle(_tiny_unet(), _tiny_clip())
    _clip_q(ref).weight.data.copy_(orig)
    apply_lora(ref, str(path_b), multiplier=0.5)
    torch.testing.assert_close(target.weight, _clip_q(ref).weight)


def test_clear_loras_restores_original_exactly(tmp_path):
    bundle = _bundle(_tiny_unet(), _tiny_clip())
    target = _clip_q(bundle)
    orig = target.weight.detach().clone()
    path = tmp_path / "a.safetensors"
    _write_lora(path, _Q_PROJ, tuple(orig.shape), rank=4, seed=3)

    apply_lora(bundle, str(path), multiplier=0.8)
    apply_lora(bundle, str(path), multiplier=0.3)
    clear_loras(bundle)

    torch.testing.assert_close(target.weight, orig)
    assert bundle._lora_state == {"stack": [], "base": {}}


def test_clear_loras_is_byte_identical_in_fp16(tmp_path):
    """Snapshot/restore is a plain copy, so unload is exact even in fp16 — no
    add/subtract drift (the reason we snapshot rather than recompute-and-subtract)."""
    bundle = _bundle(_tiny_unet().half(), _tiny_clip().half())
    target = _clip_q(bundle)
    orig = target.weight.detach().clone()
    path_a, path_b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _write_lora(path_a, _Q_PROJ, tuple(orig.shape), rank=4, seed=4)
    _write_lora(path_b, _Q_PROJ, tuple(orig.shape), rank=2, seed=5)

    apply_lora(bundle, str(path_a), multiplier=0.8)
    apply_lora(bundle, str(path_b), multiplier=0.5)
    clear_loras(bundle)

    assert torch.equal(target.weight, orig)


def test_swapping_many_loras_stays_bounded_and_exact(tmp_path):
    """Swap 20 LoRAs (drop previous, add next). The snapshot dict never grows
    beyond the touched module, and every swap matches a fresh single apply."""
    bundle = _bundle(_tiny_unet(), _tiny_clip())
    target = _clip_q(bundle)
    orig = target.weight.detach().clone()
    paths = []
    for i in range(20):
        p = tmp_path / f"l{i}.safetensors"
        _write_lora(p, _Q_PROJ, tuple(orig.shape), rank=2 + (i % 3), seed=100 + i)
        paths.append(p)

    for i, path in enumerate(paths):
        if i > 0:
            remove_lora(bundle, str(paths[i - 1]))
        apply_lora(bundle, str(path), multiplier=0.7)

        assert len(bundle._lora_state["base"]) == 1   # one touched module, snapshot reused
        ref = _bundle(_tiny_unet(), _tiny_clip())
        _clip_q(ref).weight.data.copy_(orig)
        apply_lora(ref, str(path), multiplier=0.7)
        torch.testing.assert_close(target.weight, _clip_q(ref).weight)


def test_shared_in_proj_snapshotted_once(tmp_path):
    """bigG's q/k/v share one ``in_proj_weight``; LoRA keys for q and k target
    its row-slices, so it must be snapshotted once and fully restored on clear."""
    bundle = _bundle(_tiny_unet(), _tiny_clip(), text_encoder_2=_tiny_bigg())
    in_proj = bundle.text_encoder_2.transformer.resblocks[0].attn.in_proj_weight
    torch.nn.init.normal_(in_proj, std=0.02)
    width = in_proj.shape[1]
    before = in_proj.detach().clone()

    gen = torch.Generator().manual_seed(7)
    q = "lora_te2_text_model_encoder_layers_0_self_attn_q_proj"
    k = "lora_te2_text_model_encoder_layers_0_self_attn_k_proj"
    path = tmp_path / "bigg.safetensors"
    save_file({f"{q}.lora_down.weight": torch.randn(4, width, generator=gen),
               f"{q}.lora_up.weight": torch.randn(width, 4, generator=gen),
               f"{q}.alpha": torch.tensor([4.0]),
               f"{k}.lora_down.weight": torch.randn(4, width, generator=gen),
               f"{k}.lora_up.weight": torch.randn(width, 4, generator=gen),
               f"{k}.alpha": torch.tensor([4.0])}, str(path))

    apply_lora(bundle, str(path))
    assert len(bundle._lora_state["base"]) == 1
    clear_loras(bundle)
    torch.testing.assert_close(in_proj, before)


def test_remove_unknown_lora_raises(tmp_path):
    bundle = _bundle(_tiny_unet(), _tiny_clip())
    with pytest.raises(ValueError):
        remove_lora(bundle, "never_applied.safetensors")
