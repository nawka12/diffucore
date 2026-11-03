"""Anima Skip Layer Guidance, offline, CPU: the DiT's block skip, the guidance
formula on all three sampling paths, its window, and the TeaCache / CUDA Graphs
guards. The MOSS-Image reference pins the same formula in ``test_moss_image.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from diffucore.models.anima_dit import AnimaDiT, CosmosDiT, CosmosDiTConfig
from diffucore.pipelines import ImageToImage, Inpaint, TextToImage, _anima
from diffucore.pipelines._anima import anima_img2img, anima_text_to_image
from diffucore.runtime import ConditioningCache, DevicePolicy, stream_blocks
from diffucore.sampling import flow_matching_schedule

_C, _U, _K = 1.0, 0.25, 0.5   # the cond, uncond and skipped branches' constant velocities


def _dit():
    cfg = CosmosDiTConfig(model_channels=128, num_blocks=4, num_heads=4, head_dim=32)
    torch.manual_seed(0)
    return CosmosDiT(cfg).float().eval(), cfg


def _inputs(cfg):
    torch.manual_seed(1)
    return (torch.randn(1, cfg.in_channels, 1, 8, 8), torch.tensor([0.5]),
            torch.randn(1, 16, cfg.crossattn_emb_channels))


@pytest.mark.parametrize("skip", [(0,), (1,), (3,), (1, 2), (0, 3)])
def test_skip_layers_is_exactly_removing_those_blocks(skip):
    dit, cfg = _dit()
    x, t, ctx = _inputs(cfg)
    with torch.no_grad():
        out = dit(x, t, ctx, skip_layers=skip)
        full = dit(x, t, ctx)
        dit.blocks = nn.ModuleList(b for i, b in enumerate(dit.blocks) if i not in skip)
        assert torch.equal(out, dit(x, t, ctx))
    assert not torch.equal(out, full)


def test_empty_skip_layers_is_the_plain_forward_and_anima_passes_it_through():
    dit, cfg = _dit()
    x, t, ctx = _inputs(cfg)
    with torch.no_grad():
        assert torch.equal(dit(x, t, ctx), dit(x, t, ctx, skip_layers=()))
    anima = AnimaDiT(cfg).float().eval()
    with torch.no_grad():
        assert torch.equal(anima(x, t, ctx, skip_layers=(1,)), CosmosDiT.forward(anima, x, t, ctx, skip_layers=(1,)))
        assert not torch.equal(anima(x, t, ctx, skip_layers=(1,)), anima(x, t, ctx))


def test_skipping_a_streamed_block_is_transparent():
    dit, cfg = _dit()
    x, t, ctx = _inputs(cfg)
    with torch.no_grad():
        want = dit(x, t, ctx, skip_layers=(2,))
    stream_blocks(dit, ("blocks",), torch.device("cpu"), torch.device("cpu"), keep_resident=(dit.blocks[0],))
    with torch.no_grad():
        assert torch.equal(dit(x, t, ctx, skip_layers=(2,)), want)


class _ConstantDiT(nn.Module):
    """Constant velocity per branch, with the block list the pipeline validates
    against. ``calls`` records ``(sigma, branch)`` per forward."""

    def __init__(self, cond_ctx):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(6))
        self.cond_ctx = cond_ctx
        self.calls = []

    def forward(self, x, t, ctx, teacache=None, skip_layers=()):
        is_cond = torch.equal(ctx, self.cond_ctx)
        assert not skip_layers or is_cond
        branch = "skip" if skip_layers else "cond" if is_cond else "uncond"
        self.calls.append((float(t.flatten()[0]), branch))
        return torch.full_like(x, {"skip": _K, "cond": _C, "uncond": _U}[branch])


class _StubVAE(nn.Module):
    def process_out(self, z):
        return z

    def process_in(self, z):
        return z

    def encode(self, pixels):
        return torch.zeros(1, 16, pixels.shape[2] // 8, pixels.shape[3] // 8)


def _bundle(monkeypatch, cuda_graphs=False):
    """A bundle whose conditioning cache spares the text encoder. Returns it and
    the list of final latents the pipeline decodes."""
    cond, uncond = torch.ones(1, 4, 8), torch.zeros(1, 4, 8)
    cache = ConditioningCache()
    cache.put(("p", "n"), {"cond_ctx": cond, "uncond_ctx": uncond})
    policy = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32, vae_dtype=torch.float32)
    if cuda_graphs:
        policy = SimpleNamespace(cuda_graphs=True, device=torch.device("cpu"), compute_dtype=torch.float32)
    model = SimpleNamespace(spec=SimpleNamespace(architecture="anima", image_size=64), backbone=_ConstantDiT(cond),
                            vae=_StubVAE(), tokenizer=None, text_encoder=None, policy=policy, cond_cache=cache)
    latents = []
    monkeypatch.setattr(_anima, "vae_decode_safe", lambda vae, z, policy: (latents.append(z.clone()), "full")[::-1])
    monkeypatch.setattr(_anima, "_to_pil", lambda image: Image.new("RGB", (64, 64)))
    return model, latents


def _moved(latents) -> float:
    """How far sampling moved the latent from its seeded start."""
    start = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(5))
    return (latents[0] - start).mean().item()


def _euler_move(sigmas, velocities) -> float:
    return sum(float(sigmas[i + 1] - sigmas[i]) * v for i, v in enumerate(velocities))


CFG_ON = _U + 3.0 * (_C - _U)
BRANCHES = {"cfg_slg": CFG_ON + 2.0 * (_C - _K), "cfg": CFG_ON, "cond_slg": _C + 2.0 * (_C - _K)}


@pytest.mark.parametrize("sampler", ["euler", "dpmpp_2m"])
def test_formula_with_cfg_inside_and_outside_the_window_and_the_cfg_band(monkeypatch, sampler):
    model, latents = _bundle(monkeypatch)
    anima_text_to_image(model, "p", "n", steps=4, cfg_scale=3.0, cfg_interval_end=0.75, width=64, height=64, seed=5,
                        sampler=sampler, slg_layers=(2,), slg_scale=2.0, slg_start=0.25)
    branches = [b for _, b in model.backbone.calls]
    #        step 0: no SLG      steps 1, 2: CFG + SLG              step 3: off the CFG band, SLG on plain cond
    assert branches == ["cond", "uncond"] + ["cond", "uncond", "skip"] * 2 + ["cond", "skip"]
    if sampler == "euler":
        sigmas = flow_matching_schedule(4, shift=3.0, device="cpu", dtype=torch.float32)
        want = _euler_move(sigmas, [BRANCHES["cfg"], BRANCHES["cfg_slg"], BRANCHES["cfg_slg"], BRANCHES["cond_slg"]])
        assert _moved(latents) == pytest.approx(want, rel=1e-4)


@pytest.mark.parametrize("scale, want", [(2.0, _K + 2.0 * (_C - _K)), (1.0, _C)])
def test_at_cfg_one_slg_builds_on_the_skipped_branch(monkeypatch, scale, want):
    model, latents = _bundle(monkeypatch)
    anima_text_to_image(model, "p", "n", steps=2, cfg_scale=1.0, width=64, height=64, seed=5,
                        slg_layers=(2,), slg_scale=scale)
    assert [b for _, b in model.backbone.calls] == ["cond", "skip"] * 2
    sigmas = flow_matching_schedule(2, shift=3.0, device="cpu", dtype=torch.float32)
    assert _moved(latents) == pytest.approx(_euler_move(sigmas, [want, want]), rel=1e-4)


@pytest.mark.parametrize("off", [dict(slg_layers=(2,), slg_scale=0.0), dict(slg_layers=(), slg_scale=2.0),
                                 dict(slg_layers=None, slg_scale=2.0), dict(slg_layers=(2,), slg_scale=2.0,
                                                                            slg_start=0.5, slg_stop=0.5)])
def test_slg_off_runs_no_extra_forward_and_is_identical(monkeypatch, off):
    model, latents = _bundle(monkeypatch)
    anima_text_to_image(model, "p", "n", steps=3, cfg_scale=3.0, width=64, height=64, seed=5)
    base = list(model.backbone.calls)
    model.backbone.calls.clear()
    anima_text_to_image(model, "p", "n", steps=3, cfg_scale=3.0, width=64, height=64, seed=5, **off)
    assert model.backbone.calls == base and torch.equal(latents[0], latents[1])


def test_second_order_evaluations_follow_the_window(monkeypatch):
    model, _ = _bundle(monkeypatch)
    anima_text_to_image(model, "p", "n", steps=4, cfg_scale=1.0, width=64, height=64, seed=5, sampler="heun",
                        slg_layers=(2,), slg_scale=2.0, slg_start=0.5)
    sigmas = flow_matching_schedule(4, shift=3.0, device="cpu", dtype=torch.float32)
    limit = float(sigmas[2])
    calls = model.backbone.calls
    assert any(b == "skip" for _, b in calls) and all(s <= limit + 1e-6 for s, b in calls if b == "skip")
    assert {round(s, 5) for s, b in calls if b == "skip"} == {round(s, 5) for s, b in calls if s <= limit + 1e-6}


def _i2i(model, **kw):
    init = Image.new("RGB", (64, 64))
    return anima_img2img(model, "p", init, "n", strength=0.5, steps=4, cfg_scale=3.0, width=64, height=64, seed=5, **kw)


@pytest.mark.parametrize("mask", [None, Image.new("L", (64, 64), 255)])
def test_img2img_and_inpaint_window_covers_the_sliced_schedule(monkeypatch, mask):
    model, _ = _bundle(monkeypatch)
    _i2i(model, mask_image=mask, sampler="dpmpp_2m", slg_layers=(2,), slg_scale=2.0, slg_start=0.5)
    branches = [b for _, b in model.backbone.calls]
    assert branches == ["cond", "uncond"] * 2 + ["cond", "uncond", "skip"] * 2


@pytest.mark.parametrize("cls, args", [
    (TextToImage, ("p", "n")),
    (ImageToImage, ("p", Image.new("RGB", (64, 64)), "n")),
    (Inpaint, ("p", Image.new("RGB", (64, 64)), Image.new("L", (64, 64), 255), "n")),
])
def test_every_pipeline_entry_point_forwards_slg(monkeypatch, cls, args):
    model, _ = _bundle(monkeypatch)
    cls(model)(*args, steps=2, cfg_scale=3.0, width=64, height=64, seed=5, slg_layers=(2,), slg_scale=2.0)
    assert [b for _, b in model.backbone.calls].count("skip") == 2


@pytest.mark.parametrize("bad, match", [
    (dict(slg_layers=(6,), slg_scale=2.0), "slg_layers"),
    (dict(slg_layers=(-1,), slg_scale=2.0), "slg_layers"),
    (dict(slg_scale=-1.0), "slg_scale"),
    (dict(slg_scale=float("nan")), "slg_scale"),
    (dict(slg_start=0.6, slg_stop=0.5), "slg_start"),
    (dict(slg_stop=1.5), "slg_start"),
])
def test_slg_options_are_validated(monkeypatch, bad, match):
    model, _ = _bundle(monkeypatch)
    with pytest.raises(ValueError, match=match):
        anima_text_to_image(model, "p", "n", steps=2, width=64, height=64, **bad)
    with pytest.raises(ValueError, match=match):
        _i2i(model, **bad)


def test_slg_is_rejected_with_teacache_and_cuda_graphs_but_only_when_on(monkeypatch):
    model, _ = _bundle(monkeypatch)
    on = dict(slg_layers=(2,), slg_scale=2.0)
    with pytest.raises(ValueError, match="TeaCache"):
        anima_text_to_image(model, "p", "n", steps=2, width=64, height=64, teacache_thresh=0.3, **on)
    with pytest.raises(ValueError, match="TeaCache"):
        _i2i(model, teacache_thresh=0.3, **on)
    graphs, _ = _bundle(monkeypatch, cuda_graphs=True)
    with pytest.raises(ValueError, match="CUDA Graphs"):
        anima_text_to_image(graphs, "p", "n", steps=2, width=64, height=64, **on)
    with pytest.raises(ValueError, match="CUDA Graphs"):
        _i2i(graphs, **on)
    assert _anima._anima_slg(graphs, (2,), 0.0, 0.0, 1.0, 0.3) == ()


def _probe_sampler(monkeypatch) -> list:
    """Swap the registry sampler for one recording the velocity the closure gives
    at each schedule sigma, so the closures' formula is checked by value."""
    velocities = []

    def sample(denoise, x, sigmas, callback=None, **kwargs):
        for sigma in sigmas[:-1]:
            velocities.append(((x - denoise(x, sigma.reshape(1))) / sigma).mean().item())
        return x

    monkeypatch.setattr(_anima, "get_sampler", lambda name: sample)
    return velocities


def _run_path(path, model, **kw):
    if path == "t2i":
        anima_text_to_image(model, "p", "n", steps=4, width=64, height=64, seed=5, sampler="dpmpp_2m", **kw)
    else:
        mask = Image.new("L", (64, 64), 255) if path == "inpaint" else None
        anima_img2img(model, "p", Image.new("RGB", (64, 64)), "n", mask_image=mask, strength=0.5, steps=4, width=64,
                      height=64, seed=5, sampler="dpmpp_2m", **kw)


@pytest.mark.parametrize("path", ["t2i", "i2i", "inpaint"])
def test_closure_formula_by_value(monkeypatch, path):
    model, _ = _bundle(monkeypatch)
    velocities = _probe_sampler(monkeypatch)
    _run_path(path, model, cfg_scale=3.0, cfg_interval_end=0.75, slg_layers=(2,), slg_scale=2.0, slg_start=0.25)
    assert velocities == pytest.approx([BRANCHES["cfg"], BRANCHES["cfg_slg"], BRANCHES["cfg_slg"], BRANCHES["cond_slg"]])


@pytest.mark.parametrize("path", ["t2i", "i2i", "inpaint"])
@pytest.mark.parametrize("scale, want", [(2.0, _K + 2.0 * (_C - _K)), (1.0, _C)])
def test_closure_at_cfg_one_builds_on_the_skipped_branch(monkeypatch, path, scale, want):
    model, _ = _bundle(monkeypatch)
    velocities = _probe_sampler(monkeypatch)
    _run_path(path, model, cfg_scale=1.0, slg_layers=(2,), slg_scale=scale)
    assert velocities == pytest.approx([want] * 4)


class _Shift(nn.Module):
    def __init__(self, by):
        super().__init__()
        self.by = by

    def forward(self, x, *args, carry=None, defer=False, null_keys=0):
        return x + self.by, None


def test_a_skipped_pass_survives_a_stream_beyond_fp16_range():
    """Removing a block that cancels a large residual leaves the fp32 stream above
    65504. The cast to fp16 would make it inf, so the skipped pass normalizes first."""
    dit, cfg = _dit()
    dit = dit.half()
    dit.blocks = nn.ModuleList([_Shift(1e6), _Shift(-1e6)])
    x = torch.randn(1, cfg.in_channels, 1, 8, 8).half()
    t = torch.tensor([0.5]).half()
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels).half()
    with torch.no_grad():
        assert torch.isfinite(dit(x, t, ctx)).all()
        assert torch.isfinite(dit(x, t, ctx, skip_layers=(1,))).all()
        dit.blocks = nn.ModuleList([_Shift(1e6), _Shift(0.0)])
        assert not torch.isfinite(dit(x, t, ctx)).all()   # what the skip path avoids
