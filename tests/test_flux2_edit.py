"""FLUX.2 reference-image editing: ref ids/tokens, the "flux2" scheduler, the
reference preprocessing, and that the no-reference path is unchanged. CPU,
tiny random-weight DiT + stub VAE/text encoder."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from diffucore.models.flux_dit import Flux, FluxConfig
from diffucore.pipelines._flux import (
    _flux2_empirical_mu,
    _flux2_patchify_latents,
    _flux2_reference_geometry,
    _flux2_reference_ids,
    _flux2_prepare_reference,
    _img_ids,
    _patchify,
    _txt_ids,
    _unpatchify,
    flux_img2img,
    flux_text_to_image,
)
from diffucore.pipelines.text_to_image import TextToImage
from diffucore.runtime import DevicePolicy
from diffucore.sampling import flow_matching_schedule

CTX_DIM = 24  # per-layer width; the context is 3 layers wide


class _StubVAE(nn.Module):
    """8x pixel pooling -> 32 channels; ``sample=True`` adds noise so a missed
    ``sample=False`` shows up as non-determinism."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, 32 * 3).view(32, 3), requires_grad=False)
        self.register_buffer("flux2_latent_mean", torch.linspace(-0.3, 0.3, 128).view(1, -1, 1, 1),
                             persistent=False)
        self.register_buffer("flux2_latent_std", torch.linspace(0.8, 1.4, 128).view(1, -1, 1, 1),
                             persistent=False)

    def encode(self, x, sample=True, generator=None):
        z = torch.einsum("bchw,kc->bkhw", torch.nn.functional.avg_pool2d(x, 8), self.w)
        return z + 0.5 * torch.randn_like(z) if sample else z

    def decode(self, z):
        img = torch.einsum("bkhw,kc->bchw", z, self.w) / 32
        return torch.nn.functional.interpolate(img, scale_factor=8)


class _RecordingFlux(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = []

    def forward(self, img, img_ids, txt, txt_ids, timesteps, y=None, guidance=None):
        self.calls.append(dict(n_tokens=img.shape[1], ids=img_ids.clone(),
                               t=float(timesteps[0])))
        return self.inner(img, img_ids, txt, txt_ids, timesteps, y, guidance)


def _tiny_klein(arch="flux2"):
    cfg = FluxConfig(
        in_channels=128, context_in_dim=3 * CTX_DIM, vec_in_dim=None,
        hidden_size=256, num_heads=2, depth=1, depth_single_blocks=1,
        axes_dim=(32, 32, 32, 32), theta=2000, mlp_ratio=3.0, qkv_bias=False,
        guidance_embed=False, global_modulation=True, mlp_silu_act=True, ops_bias=False,
    )
    torch.manual_seed(0)
    backbone = _RecordingFlux(Flux(cfg).eval())

    class _Tok:
        hidden_layers = (1, 2, 3)

        def __call__(self, prompt):
            ids = torch.tensor([[ord(c) % 50 for c in prompt][:6] or [0]])
            return SimpleNamespace(to=lambda device: ids)

    def text_encoder(ids, hidden_layers):
        g = torch.Generator().manual_seed(int(ids.sum()))
        return [torch.randn(1, ids.shape[1], CTX_DIM, generator=g) for _ in hidden_layers]

    spec = SimpleNamespace(architecture=arch, latent_channels=128, guidance_distilled=False,
                           image_size=1024)
    return SimpleNamespace(
        spec=spec, backbone=backbone, vae=_StubVAE(), tokenizer=_Tok(),
        text_encoder=text_encoder, text_encoder_2=None, cond_cache=None,
        policy=DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32),
    )


def _img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _pixels(img):
    return np.asarray(img, dtype=np.int16)


# ---- "flux2" scheduler ---------------------------------------------------------

def _diffusers_mu(image_seq_len, num_steps):
    """Independent restatement of diffusers' ``compute_empirical_mu``."""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return a2 * image_seq_len + b2
    m200, m10 = a2 * image_seq_len + b2, a1 * image_seq_len + b1
    a = (m200 - m10) / 190.0
    return a * num_steps + (m200 - 200.0 * a)


def _diffusers_sigmas(seq_len, steps):
    """FlowMatchEulerDiscreteScheduler, exponential time shift, terminal 0."""
    mu = _diffusers_mu(seq_len, steps)
    s = np.linspace(1.0, 1 / steps, steps)
    s = math.exp(mu) / (math.exp(mu) + (1 / s - 1))
    return np.append(s, 0.0)


@pytest.mark.parametrize("seq_len,steps", [
    (256, 4), (1024, 4), (4096, 4), (1024, 20), (4096, 50), (4300, 8), (4301, 4), (6000, 30), (9216, 4),
])
def test_flux2_sigmas_match_diffusers_formula(seq_len, steps):
    assert _flux2_empirical_mu(seq_len, steps) == pytest.approx(_diffusers_mu(seq_len, steps), abs=1e-12)
    sig = flow_matching_schedule(steps, shift=math.exp(_flux2_empirical_mu(seq_len, steps)))
    np.testing.assert_allclose(sig.numpy(), _diffusers_sigmas(seq_len, steps), atol=2e-6)


def test_flux2_scheduler_drives_the_backbone_timesteps():
    model = _tiny_klein()
    flux_text_to_image(model, "p", steps=4, width=64, height=64, seed=1, scheduler="flux2")
    seq_len = (64 // 16) ** 2
    got = [c["t"] for c in model.backbone.calls]
    np.testing.assert_allclose(got, _diffusers_sigmas(seq_len, 4)[:-1], atol=2e-6)


def test_flux2_scheduler_uses_noisy_token_count_not_refs():
    model = _tiny_klein()
    flux_text_to_image(model, "p", steps=4, width=64, height=64, seed=1, scheduler="flux2",
                       reference_images=[_img(128, 128)])
    got = [c["t"] for c in model.backbone.calls]
    np.testing.assert_allclose(got, _diffusers_sigmas(16, 4)[:-1], atol=2e-6)


def test_flux_scheduler_unchanged_and_flux2_rejected_for_flux1():
    model = _tiny_klein()
    flux_text_to_image(model, "p", steps=4, width=64, height=64, seed=1, scheduler="flux")
    ref = flow_matching_schedule(4, shift=math.exp(2.02))[:-1]
    np.testing.assert_allclose([c["t"] for c in model.backbone.calls], ref.numpy(), atol=1e-7)
    with pytest.raises(ValueError, match="FLUX.2"):
        flux_text_to_image(_tiny_klein("flux1"), "p", steps=2, width=64, height=64, scheduler="flux2")


# ---- reference preprocessing ---------------------------------------------------

def test_reference_geometry_area_cap_and_floor():
    # over the cap: scaled to ~1 MP (int-truncated), then floored to /16
    assert _flux2_reference_geometry((2048, 1536)) == ((1182, 886), (1168, 880))
    assert _flux2_reference_geometry((1024, 1024)) == ((1024, 1024), (1024, 1024))
    # under the cap: no scaling, only the floor
    assert _flux2_reference_geometry((1000, 700)) == ((1000, 700), (992, 688))
    assert _flux2_reference_geometry((512, 512)) == ((512, 512), (512, 512))


def test_reference_geometry_rejects_bad_inputs():
    with pytest.raises(ValueError, match="too small"):
        _flux2_reference_geometry((32, 512))
    with pytest.raises(ValueError, match="aspect"):
        _flux2_reference_geometry((2048, 128))


def test_prepare_reference_centre_crops_without_resampling():
    img = _img(100, 70)  # floors to 96x64: crop 2 px left/right, 3 px top, 3 bottom
    t = _flux2_prepare_reference(img)
    assert tuple(t.shape) == (1, 3, 64, 96)
    expect = _pixels(img)[3:67, 2:98].astype(np.float32) / 127.5 - 1.0
    np.testing.assert_allclose(t[0].permute(1, 2, 0).numpy(), expect, atol=1e-6)
    assert t.min() >= -1 and t.max() <= 1


def test_prepare_reference_resizes_over_area_images():
    t = _flux2_prepare_reference(Image.new("RGB", (2048, 1536), (10, 200, 30)))
    assert tuple(t.shape) == (1, 3, 880, 1168)


def test_default_output_size_follows_first_reference():
    model = _tiny_klein()
    out = flux_text_to_image(model, "p", steps=1, seed=1,
                             reference_images=[_img(100, 70), _img(64, 64)])
    assert out.size == (96, 64)
    # an explicit size wins
    out = flux_text_to_image(model, "p", steps=1, seed=1, width=64, height=48,
                             reference_images=[_img(100, 70)])
    assert out.size == (64, 48)


# ---- ids / tokens --------------------------------------------------------------

def test_reference_ids_layout():
    ids = _flux2_reference_ids([(2, 3), (1, 2)], "cpu")
    assert tuple(ids.shape) == (1, 8, 4)
    first, second = ids[0, :6], ids[0, 6:]
    assert (first[:, 0] == 10).all() and (second[:, 0] == 20).all()
    assert first[:, 1].tolist() == [0, 0, 0, 1, 1, 1] and first[:, 2].tolist() == [0, 1, 2, 0, 1, 2]
    assert second[:, 1].tolist() == [0, 0] and second[:, 2].tolist() == [0, 1]
    assert (ids[..., 3] == 0).all()
    # the noisy latent keeps T = 0
    assert (_img_ids(2, 3, 1, 4, "cpu")[..., 0] == 0).all()


def test_backbone_sees_noisy_plus_reference_tokens_and_ids():
    model = _tiny_klein()
    flux_text_to_image(model, "p", steps=3, width=64, height=64, seed=2,
                       reference_images=[_img(128, 64), _img(64, 64)])
    n_noisy, n_ref = 16, (64 // 16) * (128 // 16) + (64 // 16) ** 2  # 4x8 + 4x4
    calls = model.backbone.calls
    assert len(calls) == 3
    for c in calls:
        assert c["n_tokens"] == n_noisy + n_ref
        assert c["ids"].shape == (1, n_noisy + n_ref, 4)
        assert (c["ids"][0, :n_noisy, 0] == 0).all()
        assert (c["ids"][0, n_noisy:n_noisy + 32, 0] == 10).all()
        assert (c["ids"][0, n_noisy + 32:, 0] == 20).all()
    # the noisy part is the plain t2i grid
    assert torch.equal(calls[0]["ids"][:, :n_noisy], _img_ids(4, 4, 1, 4, "cpu"))


def test_references_are_encoded_once_and_deterministically():
    model = _tiny_klein()
    encodes = []
    orig = model.vae.encode
    model.vae.encode = lambda *a, **k: (encodes.append(k.get("sample")), orig(*a, **k))[1]
    kw = dict(steps=3, width=64, height=64, seed=2, reference_images=[_img(64, 64)])
    a = flux_text_to_image(model, "p", **kw)
    b = flux_text_to_image(model, "p", **kw)
    assert len(encodes) == 2 and all(s is False for s in encodes)  # once per call, mode not sample
    assert np.array_equal(_pixels(a), _pixels(b))


def test_reference_tokens_use_pixel_shuffle_and_batch_norm():
    from diffucore.pipelines._flux import _flux2_encode_references
    model = _tiny_klein()
    img = _img(128, 64)
    tokens, ids = _flux2_encode_references(model, [img], torch.device("cpu"), torch.float32)
    z = _flux2_patchify_latents(model.vae.encode(_flux2_prepare_reference(img), sample=False))
    z = (z - model.vae.flux2_latent_mean) / model.vae.flux2_latent_std
    assert torch.allclose(tokens, _patchify(z, 1), atol=1e-6)
    assert tuple(tokens.shape) == (1, 8 * 4, 128) and tuple(ids.shape) == (1, 32, 4)


def test_reference_changes_the_result_and_euler_sampler_variants_run():
    model = _tiny_klein()
    kw = dict(steps=3, width=64, height=64, seed=2)
    plain = flux_text_to_image(model, "p", **kw)
    edited = flux_text_to_image(model, "p", reference_images=[_img(64, 64)], **kw)
    assert not np.array_equal(_pixels(plain), _pixels(edited))
    # a non-euler sampler must also slice the output back to the noisy tokens
    flux_text_to_image(model, "p", sampler="heun", reference_images=[_img(64, 64)], **kw)


# ---- the no-reference path is unchanged ----------------------------------------

def test_no_reference_path_matches_manual_t2i_loop():
    """Re-derive the pre-change t2i op by op; the output must be identical."""
    model = _tiny_klein()
    steps, size, seed = 4, 64, 7
    out = flux_text_to_image(model, "a prompt", steps=steps, width=size, height=size, seed=seed)

    dtype, device = torch.float32, torch.device("cpu")
    ids = model.tokenizer("a prompt").to(device)
    ctx = torch.cat(model.text_encoder(ids, hidden_layers=(1, 2, 3)), dim=-1)
    h = size // 16
    img_ids = _img_ids(h, h, 1, 4, device)
    txt_ids = _txt_ids(ctx.shape[1], 4, 3, device)
    sigmas = flow_matching_schedule(steps, shift=math.exp(2.02))
    gen = torch.Generator(device=device).manual_seed(seed)
    x = _patchify(torch.randn(1, 128, h, h, generator=gen, device=device, dtype=dtype), 1)
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((1,), float(sigmas[i]), dtype=dtype)
            v = model.backbone.inner(x, img_ids, ctx, txt_ids, t, None, None)
            x = x + (sigmas[i + 1] - sigmas[i]).to(dtype) * v
        x = _unpatchify(x, h, h, 1)
        x = x * model.vae.flux2_latent_std + model.vae.flux2_latent_mean
        from diffucore.pipelines._flux import _flux2_unpatchify_latents, _to_pil
        want = _to_pil(model.vae.decode(_flux2_unpatchify_latents(x)))
    assert np.array_equal(_pixels(out), _pixels(want))


def test_default_size_without_references_is_1024_behaviour():
    # width/height now default to None -> 1024 when there are no references
    model = _tiny_klein()
    seen = {}
    orig = model.backbone.forward
    model.backbone.forward = lambda img, *a, **k: (seen.setdefault("n", img.shape[1]), orig(img, *a, **k))[1]
    with pytest.raises(Exception):  # too heavy to finish on CPU; only the token count matters
        model.vae.decode = lambda z: (_ for _ in ()).throw(RuntimeError("stop"))
        flux_text_to_image(model, "p", steps=1, seed=1)
    assert seen["n"] == (1024 // 16) ** 2


# ---- guards and dispatch -------------------------------------------------------

def test_references_rejected_for_flux1():
    with pytest.raises(ValueError, match="FLUX.2"):
        flux_text_to_image(_tiny_klein("flux1"), "p", steps=1, width=64, height=64,
                           reference_images=[_img(64, 64)])


def test_text_to_image_rejects_references_for_other_families():
    for arch in ("anima", "flux1", "sdxl"):
        t2i = TextToImage(SimpleNamespace(spec=SimpleNamespace(architecture=arch), policy=None))
        with pytest.raises(ValueError, match="FLUX.2"):
            t2i("p", reference_images=[_img(64, 64)])


def test_text_to_image_forwards_references_and_follows_first_reference_size():
    model = _tiny_klein()
    out = TextToImage(model)("p", steps=1, cfg_scale=1.0, seed=1, scheduler="flux2",
                             reference_images=[_img(100, 70)])
    assert out.size == (96, 64)


def test_img2img_accepts_flux2_scheduler():
    model = _tiny_klein()
    flux_img2img(model, "p", _img(64, 64), strength=1.0, steps=4, width=64, height=64,
                 seed=1, scheduler="flux2")
    got = [c["t"] for c in model.backbone.calls]
    np.testing.assert_allclose(got, _diffusers_sigmas(16, 4)[:-1], atol=2e-6)


# ---- fp16 saturation ------------------------------------------------------------

def test_fp16_blocks_saturate_instead_of_overflowing_to_inf():
    from diffucore.models.flux_dit import (DoubleStreamBlock, EmbedND, ModulationOut,
                                           SingleStreamBlock)
    torch.manual_seed(0)
    dim, heads, L = 64, 2, 6
    pe = EmbedND(dim // heads, 2000, (8, 8, 8, 8))(torch.zeros(1, L, 4))
    big = ModulationOut(torch.zeros(1, 1, dim), torch.zeros(1, 1, dim), torch.full((1, 1, dim), 6e4))
    x = torch.full((1, L, dim), 6e4).half()

    single = SingleStreamBlock(dim, heads, 3.0, modulation=False, mlp_silu_act=True, bias=False).half().eval()
    big16 = ModulationOut(*[t.half() for t in (big.shift, big.scale, big.gate)])
    out = single(x, vec=big16, pe=pe)
    assert torch.isfinite(out).all() and out.abs().max() <= 65504

    double = DoubleStreamBlock(dim, heads, 3.0, False, modulation=False, mlp_silu_act=True,
                               proj_bias=False).half().eval()
    off = ModulationOut(*[t.half() for t in (big.shift, big.scale, torch.zeros(1, 1, dim))])
    mod = (off, big16)  # attention residual untouched, MLP residual overflows
    pe2 = EmbedND(dim // heads, 2000, (8, 8, 8, 8))(torch.zeros(1, 2 * L, 4))  # txt + img
    img, txt = double(x, x.clone(), vec=(mod, mod), pe=pe2)
    assert torch.isfinite(txt).all() and txt.abs().max() <= 65504
