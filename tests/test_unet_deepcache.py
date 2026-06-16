"""DeepCache (arXiv:2312.00858) on the SD/SDXL UNet.

Verifies the cache is a transparent accelerator with a clean correctness anchor:
when it doesn't reuse, the forward is bit-identical to a plain forward; when it
*does* reuse, the splice is geometrically exact — recomputing the shallow level-0
blocks on an unchanged input and pasting back the cached deep feature reproduces
the full forward exactly. Uses tiny UNet configs — no SD weights needed.
"""

from __future__ import annotations

import torch

from diffucore.models.unet import DeepCache, UNetConfig, UNetModel
from diffucore.sampling.denoiser import CFGDenoiser, ModelDenoiser
from diffucore.sampling.parameterization import DiscreteSchedule, EpsScaling, make_betas
from diffucore.sampling.samplers import sample_euler


def _sd_tiny():
    """SD1.5-shaped: scalar transformer_depth, no added (y) conditioning."""
    cfg = UNetConfig(
        model_channels=32, channel_mult=(1, 2, 4), num_res_blocks=1,
        num_heads=4, context_dim=64, transformer_depth=1,
    )
    return UNetModel(cfg).float().eval(), cfg


def _sdxl_tiny():
    """SDXL-shaped: per-level depth, 2 res blocks, pooled+size (y) conditioning."""
    cfg = UNetConfig(
        model_channels=32, channel_mult=(1, 2, 4), num_res_blocks=2,
        num_head_channels=16, context_dim=64, transformer_depth=(0, 2, 2),
        adm_in_channels=128, use_linear_in_transformer=True,
    )
    return UNetModel(cfg).float().eval(), cfg


def _inputs(cfg, *, batch=1, size=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, cfg.in_channels, size, size)
    t = torch.full((batch,), 7.0)
    ctx = torch.randn(batch, 12, cfg.context_dim)
    y = torch.randn(batch, cfg.adm_in_channels) if cfg.adm_in_channels > 0 else None
    return x, t, ctx, y


def test_disabled_matches_plain_forward():
    """No ``_deepcache`` attribute is the unmodified path."""
    unet, cfg = _sd_tiny()
    x, t, ctx, _ = _inputs(cfg)
    with torch.no_grad():
        a = unet(x, t, ctx)
        b = unet(x, t, ctx)
    assert torch.equal(a, b)


def test_interval_one_never_reuses():
    """interval=1 forces a full UNet every step (``calls % 1 == 0`` always), so
    even varying inputs stay bit-exact to the plain forward."""
    unet, cfg = _sd_tiny()
    cache = DeepCache(interval=1)
    unet._deepcache = cache
    with torch.no_grad():
        for i in range(4):
            x, t, ctx, _ = _inputs(cfg, seed=i)
            unet._deepcache = None
            plain = unet(x, t, ctx)
            unet._deepcache = cache
            cached = unet(x, t, ctx)
            assert torch.equal(plain, cached)
    assert cache.calls == 4 and cache.skips == 0


def test_first_eval_always_computes():
    """The first eval has no cached feature, so it computes the full UNet and
    matches the plain forward exactly — never a cache hit."""
    unet, cfg = _sd_tiny()
    x, t, ctx, _ = _inputs(cfg, seed=1)
    cache = DeepCache(interval=2)
    with torch.no_grad():
        plain = unet(x, t, ctx)
        unet._deepcache = cache
        first = unet(x, t, ctx)
    assert cache.calls == 1 and cache.skips == 0
    assert cache.deep_feature is not None
    assert torch.equal(plain, first)


def test_cached_step_on_same_input_reproduces_full():
    """The splice is exact: a cached eval recomputes the shallow level-0 blocks
    on the (unchanged) input and pastes back the cached deep feature, so it
    reproduces the full forward bit-for-bit when the input hasn't moved."""
    for make in (_sd_tiny, _sdxl_tiny):
        unet, cfg = make()
        x, t, ctx, y = _inputs(cfg, seed=2)
        cache = DeepCache(interval=2)
        unet._deepcache = cache
        with torch.no_grad():
            full = unet(x, t, ctx, y)      # call 0: full, populates the cache
            cached = unet(x, t, ctx, y)    # call 1: reuses the deep feature
        assert cache.skips == 1
        assert torch.equal(full, cached)


def test_cached_step_skips_the_deep_blocks():
    """A reuse step must not run the middle block (the deepest, most expensive
    part); a full step must."""
    unet, cfg = _sd_tiny()
    ran: list[int] = []
    unet.middle_block.register_forward_hook(lambda *a: ran.append(1))
    cache = DeepCache(interval=2)
    unet._deepcache = cache
    with torch.no_grad():
        for i in range(4):  # full, cached, full, cached
            x, t, ctx, _ = _inputs(cfg, seed=i)
            unet(x, t, ctx)
    assert cache.calls == 4 and cache.skips == 2
    assert sum(ran) == 2  # middle block ran only on the two full steps


def test_interval_three_cadence():
    """interval=3 recomputes on evals 0, 3, 6 and reuses on the rest."""
    unet, cfg = _sd_tiny()
    cache = DeepCache(interval=3)
    unet._deepcache = cache
    with torch.no_grad():
        for i in range(7):
            x, t, ctx, _ = _inputs(cfg, seed=i)
            unet(x, t, ctx)
    assert cache.calls == 7 and cache.skips == 4  # 7 evals, fulls at 0/3/6


# --- integration: through the real CFG-batched sampler stack ----------------

def _euler_run(unet, cfg, *, interval, scale=2.0):
    """Run sample_euler through ModelDenoiser+CFGDenoiser (the production path:
    cond/uncond batched into one UNet forward), with the cache attached the way
    _Pipeline._sample attaches it. Returns the final latent."""
    den = ModelDenoiser(unet, EpsScaling(), DiscreteSchedule(make_betas("scaled_linear", 1000)))
    torch.manual_seed(11)
    cond = {"context": torch.randn(1, 12, cfg.context_dim)}
    uncond = {"context": torch.randn(1, 12, cfg.context_dim)}
    guided = CFGDenoiser(den, cond, uncond, scale=scale)
    sigmas = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.0])
    x = torch.randn(1, cfg.in_channels, 16, 16) * sigmas[0]
    cache = DeepCache(interval) if interval > 1 else None
    if cache is not None:
        unet._deepcache = cache
    try:
        with torch.no_grad():
            out = sample_euler(guided, x, sigmas)
    finally:
        unet._deepcache = None
    return out, cache


def test_sampler_interval_one_matches_no_cache():
    """Through the full CFG-batched sampler loop, an attached interval=1 cache is
    bit-identical to no cache at all (every eval computes the full UNet)."""
    unet, cfg = _sd_tiny()
    baseline, _ = _euler_run(unet, cfg, interval=1)
    # interval=1 builds no cache in _euler_run; force-attach one to prove the
    # attribute path itself is transparent when every step is a full step.
    unet._deepcache = DeepCache(interval=1)
    try:
        with_cache, _ = _euler_run(unet, cfg, interval=1)
    finally:
        unet._deepcache = None
    assert torch.equal(baseline, with_cache)


def test_sampler_with_cache_skips_and_stays_finite():
    """With interval=2 the cache skips on half the CFG-batched forwards and the
    run still produces a finite latent of the right shape."""
    unet, cfg = _sd_tiny()
    out, cache = _euler_run(unet, cfg, interval=2)
    assert cache.calls == 5 and cache.skips == 2  # 5 euler evals; fulls at 0/2/4
    assert out.shape == (1, cfg.in_channels, 16, 16)
    assert torch.isfinite(out).all()
    assert unet._deepcache is None  # detached after the run
