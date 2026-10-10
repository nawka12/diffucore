"""The loaders build each model with meta parameters and adopt the checkpoint
tensors (``bundle._build`` / ``_place``) instead of random-initializing fp32
weights and copying over them. The result must equal the plain build + load +
cast exactly, buffers included, and must not alias the checkpoint's storage.
"""

import pytest
import torch

from diffucore import bundle
from diffucore.models import (
    AnimaDiT, AutoencoderKL, CLIPTextEncoder, CosmosDiTConfig, Flux, FluxConfig,
    MistralConfig, MistralTextEncoder, OpenCLIPTextEncoder, Qwen35Config,
    Qwen35TextEncoder, Qwen3Config, Qwen3TextEncoder, QwenImageVAE, T5TextEncoder,
    UNetModel, VAEConfig,
)
from diffucore.models.clip_text import CLIPTextConfig
from diffucore.models.llm_adapter import LLMAdapterConfig
from diffucore.models.open_clip_text import OpenCLIPTextConfig
from diffucore.models.t5_text import T5Config
from diffucore.models.unet import UNetConfig

_CASES = {
    "anima_dit": (AnimaDiT, (
        CosmosDiTConfig(model_channels=128, num_blocks=2, num_heads=4, head_dim=32,
                        crossattn_emb_channels=64, adaln_lora_dim=32),
        LLMAdapterConfig(target_vocab=64, target_dim=64, source_dim=64, model_dim=64,
                         num_layers=1, num_heads=4, head_dim=16),
    )),
    "qwen_image_vae": (QwenImageVAE, (32,)),
    "qwen3": (Qwen3TextEncoder, (Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                                             num_hidden_layers=2, num_attention_heads=4,
                                             num_key_value_heads=2, head_dim=16),)),
    "qwen35": (Qwen35TextEncoder, (Qwen35Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, output_dim=32,
        output_projection=True, num_hidden_layers=4, self_attn_layers=(1, 3),
        no_mlp_layers=(3,), num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        ssm_d_ssm=32, ssm_conv_dim=96, ssm_n_groups=4, ssm_head_dim=8, ssm_d_state=8),)),
    "mistral": (MistralTextEncoder, (MistralConfig(vocab_size=64, hidden_size=64,
                                                   intermediate_size=128, num_hidden_layers=2,
                                                   num_attention_heads=4, num_key_value_heads=2,
                                                   head_dim=16),)),
    "t5": (T5TextEncoder, (T5Config(vocab_size=64, d_model=64, d_kv=16, d_ff=128,
                                    num_layers=2, num_heads=4),)),
    "clip": (CLIPTextEncoder, (CLIPTextConfig(vocab_size=10, hidden_size=64, num_layers=1,
                                              num_heads=4, intermediate_size=128,
                                              max_position_embeddings=16),)),
    "open_clip": (OpenCLIPTextEncoder, (OpenCLIPTextConfig(vocab_size=10, width=64, num_layers=1,
                                                           num_heads=4, mlp_dim=128,
                                                           max_position_embeddings=16),)),
    "unet": (UNetModel, (UNetConfig(model_channels=32, channel_mult=(1, 2), num_res_blocks=1,
                                    context_dim=64, num_heads=4),)),
    "vae": (AutoencoderKL, (VAEConfig(base_channels=32, channel_mult=(1, 2), num_res_blocks=1),)),
    "flux": (Flux, (FluxConfig(in_channels=16, context_in_dim=32, vec_in_dim=8, hidden_size=256,
                               num_heads=2, depth=1, depth_single_blocks=1,
                               guidance_embed=False),)),
}


def _tensors(module):
    """Every parameter and buffer, non-persistent buffers included."""
    return {**dict(module.named_parameters()), **dict(module.named_buffers())}


@pytest.mark.parametrize("name", sorted(_CASES))
@pytest.mark.parametrize("ckpt_dtype,dtype", [
    (torch.bfloat16, torch.float16),    # the usual Anima/FLUX case: a cast
    (torch.float16, torch.float16),     # same dtype on CPU: the move is a no-op
    (torch.float32, torch.float32),
])
def test_meta_build_matches_plain_load(name, ckpt_dtype, dtype):
    cls, args = _CASES[name]
    g = torch.Generator().manual_seed(0)
    # Fresh values: some modules leave weights as uninitialized torch.empty.
    sd = {k: torch.randn(v.shape, generator=g).to(ckpt_dtype) if v.is_floating_point() else v
          for k, v in cls(*args).state_dict().items()}

    plain = cls(*args)
    plain.load_state_dict(sd, strict=True)
    plain = plain.to("cpu", dtype)

    built = bundle._build(cls, *args)
    assert all(p.is_meta for p in built.parameters())
    built.load_state_dict(sd, strict=True, assign=True)
    built = bundle._place(built, "cpu", dtype, sd)

    want, got = _tensors(plain), _tensors(built)
    assert want.keys() == got.keys()
    for k, w in want.items():
        g = got[k]
        assert not g.is_meta, k
        assert g.dtype == w.dtype and torch.equal(g, w), k

    mapped = {t.untyped_storage().data_ptr() for t in sd.values()}
    assert not any(t.untyped_storage().data_ptr() in mapped for t in got.values())


def test_meta_parameters_is_off_outside_the_block():
    bundle._build(torch.nn.Linear, 4, 4)
    assert not torch.nn.Linear(4, 4).weight.is_meta
