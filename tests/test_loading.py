import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from diffucore.loading import load_state_dict, read_header, detect_architecture

_ANIMA_CKPT = Path(os.environ.get(
    "DIFFUCORE_ANIMA_CKPT",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
))


def sd15_shapes():
    """Minimal subset of an SD1.5 checkpoint's keys that detection inspects."""
    return {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
        "first_stage_model.decoder.conv_out.weight": (3, 128, 3, 3),
    }


def test_detect_sd15():
    spec = detect_architecture(sd15_shapes())
    assert spec.architecture == "sd15"
    assert spec.prediction == "eps"
    assert spec.context_dim == 768
    assert spec.latent_channels == 4
    assert spec.beta_schedule == "scaled_linear"


def test_detect_missing_unet_raises():
    with pytest.raises(ValueError):
        detect_architecture({"some.random.key": (1, 2)})


def test_detect_no_context_dim_raises():
    with pytest.raises(ValueError):
        detect_architecture({"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3)})


def test_detect_sdxl():
    shapes = dict(sd15_shapes())
    shapes["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"] = (640, 2048)
    spec = detect_architecture(shapes)
    assert spec.architecture == "sdxl"
    assert spec.prediction == "eps"
    assert spec.context_dim == 2048
    assert spec.image_size == 1024
    assert spec.latent_scale == 0.13025


def test_detect_v_prediction_from_marker():
    """A bare ``v_pred`` marker tensor flips the prediction type to v; without it
    the same checkpoint is read as epsilon."""
    shapes = dict(sd15_shapes())
    shapes["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"] = (640, 2048)
    assert detect_architecture(shapes).prediction == "eps"
    shapes["v_pred"] = ()
    spec = detect_architecture(shapes)
    assert spec.architecture == "sdxl"
    assert spec.prediction == "v"


def test_detect_ztsnr_from_marker():
    """A bare ``ztsnr`` marker tensor requests zero-terminal-SNR sampling."""
    shapes = dict(sd15_shapes())
    assert detect_architecture(shapes).zero_terminal_snr is False
    shapes["ztsnr"] = ()
    assert detect_architecture(shapes).zero_terminal_snr is True


def test_detect_unsupported_family_raises():
    # SD2.x (context_dim 1024, OpenCLIP ViT-H) is recognized but not implemented.
    shapes = dict(sd15_shapes())
    shapes["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"] = (640, 1024)
    with pytest.raises(NotImplementedError):
        detect_architecture(shapes)


def test_safetensors_load_and_header_roundtrip(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"a": torch.randn(2, 3), "b": torch.zeros(4)}, str(path))

    sd = load_state_dict(str(path))
    assert set(sd) == {"a", "b"}
    assert tuple(sd["a"].shape) == (2, 3)

    header = read_header(str(path))
    assert header == {"a": (2, 3), "b": (4,)}


def test_detect_from_real_header(tmp_path):
    # End-to-end: write a file with SD1.5-shaped tensors, read only the header,
    # and detect — no full load required.
    path = tmp_path / "fake_sd15.safetensors"
    save_file({k: torch.zeros(*v) for k, v in sd15_shapes().items()}, str(path))
    spec = detect_architecture(read_header(str(path)))
    assert spec.architecture == "sd15"


def test_load_rejects_pickle_checkpoint():
    with pytest.raises(ValueError):
        load_state_dict("model.ckpt")


def test_detect_anima_from_fingerprint():
    """Anima carries bare ``net.*`` keys; the LLM-adapter fingerprint is enough
    to identify it without the LDM-style ``model.diffusion_model.`` prefix."""
    shapes = {
        "net.llm_adapter.blocks.0.cross_attn.q_proj.weight": (1024, 1024),
        "net.blocks.0.adaln_modulation_self_attn.1.weight": (256, 2048),
        "net.x_embedder.proj.1.weight": (2048, 68),
    }
    spec = detect_architecture(shapes)
    assert spec.architecture == "anima"
    assert spec.prediction == "flow"
    assert spec.zero_terminal_snr is False
    assert spec.latent_channels == 16          # Qwen-Image VAE
    assert spec.context_dim == 1024            # LLM-adapter output dim
    assert spec.image_size == 1024


def test_detect_anima_takes_precedence_over_missing_unet():
    """Without the Anima marker, an unrelated key set raises ValueError. The
    marker alone identifies the family — there is no ``model.diffusion_model.``
    prefix to find."""
    shapes_no_marker = {"net.blocks.0.adaln_modulation_self_attn.1.weight": (256, 2048)}
    with pytest.raises(ValueError):
        detect_architecture(shapes_no_marker)


@pytest.mark.skipif(not _ANIMA_CKPT.exists(), reason=f"anima checkpoint not at {_ANIMA_CKPT}")
def test_detect_anima_from_real_header():
    """End-to-end: read the actual Anima checkpoint's header and detect."""
    spec = detect_architecture(read_header(str(_ANIMA_CKPT)))
    assert spec.architecture == "anima"
    assert spec.prediction == "flow"
    assert spec.latent_channels == 16
    assert spec.context_dim == 1024
