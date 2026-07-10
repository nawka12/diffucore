"""Anima DiT (DT5) verification — backbone + adapter wrapping.

Layered structural + behavioral tests. Numerical bit-match against ComfyUI
is deferred to DT7 (end-to-end image comparison against a ComfyUI-generated
reference) because the local ComfyUI install can't import in this venv —
the per-component oracle would be more fragile than the end-to-end check.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from diffucore.models.anima_dit import AnimaDiT, CosmosDiT, CosmosDiTConfig

_ANIMA_CKPT = Path(os.environ.get(
    "DIFFUCORE_ANIMA_CKPT",
    "/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/comfy/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
))
_PREFIX = "net."


@pytest.fixture(scope="module")
def loaded_dit():
    if not _ANIMA_CKPT.exists():
        pytest.skip(f"anima checkpoint not at {_ANIMA_CKPT}")
    from safetensors.torch import load_file
    sd = load_file(str(_ANIMA_CKPT))
    sub = {k[len(_PREFIX):]: v for k, v in sd.items() if k.startswith(_PREFIX)}
    dit = AnimaDiT()
    dit.load_state_dict(sub, strict=True)
    return dit.float().eval()


# --- 1. structural ----------------------------------------------------------

def test_anima_dit_key_set_matches_checkpoint():
    """All 685 net.* keys land on a module parameter, no missing/extra."""
    from safetensors import safe_open
    if not _ANIMA_CKPT.exists():
        pytest.skip(f"anima checkpoint not at {_ANIMA_CKPT}")
    with safe_open(str(_ANIMA_CKPT), framework="pt") as f:
        ckpt_keys = {k[len(_PREFIX):] for k in f.keys() if k.startswith(_PREFIX)}
    mod_keys = set(AnimaDiT().state_dict().keys())
    assert mod_keys == ckpt_keys, (
        f"key mismatch — missing: {sorted(ckpt_keys - mod_keys)[:5]} | "
        f"extra: {sorted(mod_keys - ckpt_keys)[:5]}"
    )


def test_anima_dit_parameter_count_matches_2B():
    """~2B parameters total (the architecture's headline figure)."""
    n = sum(p.numel() for p in AnimaDiT().parameters())
    # 2.05B–2.15B band — pre-empts an off-by-one stage that would silently
    # change channel hierarchy.
    assert 2.05e9 < n < 2.15e9, f"unexpected param count: {n/1e9:.3f}B"


# --- 2. forward (shapes / paths) -------------------------------------------

def test_dit_forward_shape_image_path(loaded_dit):
    """Latent → latent at the same (B, C, T, H, W) shape (the t2i path)."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 16, 16)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 32, 1024)
    with torch.no_grad():
        out = loaded_dit(x, t, ctx)
    assert out.shape == x.shape


def test_dit_t5xxl_path_routes_through_adapter(loaded_dit):
    """Passing ``t5xxl_ids`` produces a different output than feeding the
    pre-adapter context directly — proves the LLM-Adapter is actually in the
    forward path when its input keys are supplied."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 16, 16)
    t = torch.tensor([500.0])
    src_hidden = torch.randn(1, 32, 1024)
    t5_ids = torch.randint(0, 32128, (1, 7))
    with torch.no_grad():
        a = loaded_dit(x, t, src_hidden)
        b = loaded_dit(x, t, src_hidden, t5xxl_ids=t5_ids)
    assert not torch.equal(a, b)


def test_rope_cache_lives_on_compute_device():
    """The RoPE table is cached on the device it was built for, so a cache hit
    returns the same tensor with no copy — it used to be parked on CPU and
    re-uploaded (4 MB H2D at 1024²) on every forward."""
    from diffucore.models.anima_dit import _VideoRoPE3D
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rope = _VideoRoPE3D(CosmosDiTConfig()).to(device)
    x = torch.empty(1, 1, 8, 8, 2048, device=device)
    a = rope(x)
    b = rope(x)
    assert rope._rope_cache[1].device.type == device.type
    assert b is a


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_apply_rope_compiled_is_bit_equal_to_eager():
    """The CUDA rope apply is torch.compile'd (×4.8 measured on an RTX 2060)
    with ``emulate_precision_casts`` so the fused kernel rounds exactly like
    eager — the default path must stay bit-identical. Pins that property across
    shapes and batch sizes; if a torch upgrade breaks it, this fails rather
    than images silently drifting. (When compile is unusable the dispatch falls
    back to eager, and the assertion holds trivially.)"""
    from diffucore.models.anima_dit import _VideoRoPE3D, _apply_rope, _apply_rope_eager
    device = torch.device("cuda")
    rope = _VideoRoPE3D(CosmosDiTConfig()).to(device)
    for (h, w), b in (((64, 64), 1), ((96, 64), 1), ((32, 48), 2)):
        freqs = rope(torch.empty(1, 1, h, w, 2048, device=device)).unsqueeze(1).unsqueeze(0)
        q = torch.randn(b, h * w, 16, 128, device=device, dtype=torch.float16)
        assert torch.equal(_apply_rope(q, freqs), _apply_rope_eager(q, freqs)), (h, w, b)


# --- 3. conditioning sensitivity (catches "ignored input" bugs) ------------

def test_dit_sensitive_to_timesteps(loaded_dit):
    """The DiT must respond to the timestep — adaLN-LoRA is where this lives."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 16, 16)
    ctx = torch.randn(1, 32, 1024)
    with torch.no_grad():
        a = loaded_dit(x, torch.tensor([100.0]), ctx)
        b = loaded_dit(x, torch.tensor([900.0]), ctx)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel > 1e-2, f"DiT insensitive to t (relative max|Δ| = {rel:.3e})"


def test_dit_sensitive_to_context(loaded_dit):
    """Changing the cross-attn context must change the output — guards
    against a broken cross-attn wiring."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 16, 16)
    t = torch.tensor([500.0])
    c1 = torch.randn(1, 32, 1024)
    c2 = torch.randn(1, 32, 1024)
    with torch.no_grad():
        a = loaded_dit(x, t, c1)
        b = loaded_dit(x, t, c2)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel > 1e-2, f"DiT insensitive to context (relative max|Δ| = {rel:.3e})"


def test_dit_sensitive_to_latent(loaded_dit):
    """Different noise latents must produce different outputs."""
    torch.manual_seed(0)
    x1 = torch.randn(1, 16, 1, 16, 16)
    x2 = torch.randn(1, 16, 1, 16, 16)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 32, 1024)
    with torch.no_grad():
        a = loaded_dit(x1, t, ctx)
        b = loaded_dit(x2, t, ctx)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel > 1e-2, f"DiT insensitive to x (relative max|Δ| = {rel:.3e})"


def test_dit_determinism(loaded_dit):
    """Same inputs → bit-identical output."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 16, 16)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 32, 1024)
    with torch.no_grad():
        a = loaded_dit(x, t, ctx)
        b = loaded_dit(x, t, ctx)
    assert torch.equal(a, b)


# --- 4. base CosmosDiT (image-only) -----------------------------------------

def test_cosmos_dit_runs_without_adapter():
    """The base DiT (no LLM-Adapter) constructs and forwards on random
    weights. Cheaper than instantiating the 2B-param AnimaDiT; verifies the
    image-only forward path works for any image size that divides the patch."""
    cfg = CosmosDiTConfig(model_channels=128, num_blocks=2, num_heads=4, head_dim=32)
    dit = CosmosDiT(cfg).float().eval()
    torch.manual_seed(0)
    x = torch.randn(1, cfg.in_channels, 1, 8, 8)
    t = torch.tensor([42.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    with torch.no_grad():
        out = dit(x, t, ctx)
    assert out.shape == x.shape
