"""TeaCache (arXiv:2411.19108) on the Cosmos/Anima DiT.

Verifies the cache is a transparent accelerator: when it doesn't skip it must
be bit-identical to a plain forward, and when it does skip it must reuse an
*exact* residual (so a repeated input reproduces its own earlier output). Uses
the cheap small-config CosmosDiT — no Anima weights needed.
"""

from __future__ import annotations

import torch

from diffucore.models.anima_dit import CosmosDiT, CosmosDiTConfig, TeaCache


def _tiny():
    cfg = CosmosDiTConfig(model_channels=128, num_blocks=3, num_heads=4, head_dim=32)
    return CosmosDiT(cfg).float().eval(), cfg


def test_disabled_matches_plain_forward():
    """``teacache=None`` is the unmodified path."""
    dit, cfg = _tiny()
    torch.manual_seed(0)
    x = torch.randn(1, cfg.in_channels, 1, 8, 8)
    t = torch.tensor([7.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    with torch.no_grad():
        a = dit(x, t, ctx)
        b = dit(x, t, ctx, teacache=None)
    assert torch.equal(a, b)


def test_first_step_always_computes_and_is_exact():
    """The first forward has no history, so it must compute the blocks and
    match the plain forward exactly — never a cache hit."""
    dit, cfg = _tiny()
    torch.manual_seed(1)
    x = torch.randn(1, cfg.in_channels, 1, 8, 8)
    t = torch.tensor([3.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    tc = TeaCache(rel_l1_thresh=1e9)  # huge threshold: skip whenever allowed
    with torch.no_grad():
        plain = dit(x, t, ctx)
        first = dit(x, t, ctx, teacache=tc)
    assert tc.calls == 1 and tc.skips == 0
    assert torch.equal(plain, first)


def test_skip_reuses_exact_residual():
    """A second forward with an *identical* input drifts zero, so (with a
    permissive threshold) the blocks are skipped — the reused residual then
    reproduces the first step (to fp rounding of the residual add-back)."""
    dit, cfg = _tiny()
    torch.manual_seed(2)
    x = torch.randn(1, cfg.in_channels, 1, 8, 8)
    t = torch.tensor([5.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    tc = TeaCache(rel_l1_thresh=1e9)
    with torch.no_grad():
        out1 = dit(x, t, ctx, teacache=tc)   # computes (first call)
        out2 = dit(x, t, ctx, teacache=tc)   # identical input -> skips
    assert tc.skips == 1
    assert torch.allclose(out1, out2, atol=1e-5, rtol=1e-4)


def test_zero_threshold_never_skips():
    """A threshold of 0 forces a recompute every step (any drift >= 0 trips it),
    so even varying inputs stay bit-exact to the plain forward."""
    dit, cfg = _tiny()
    torch.manual_seed(3)
    t = torch.tensor([5.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    tc = TeaCache(rel_l1_thresh=0.0)
    with torch.no_grad():
        for i in range(4):
            x = torch.randn(1, cfg.in_channels, 1, 8, 8)
            plain = dit(x, t, ctx)
            cached = dit(x, t, ctx, teacache=tc)
            assert torch.equal(plain, cached)
    assert tc.calls == 4 and tc.skips == 0


def test_record_mode_never_skips_and_logs_rel():
    """Calibration's record mode: every call computes (no skipping) and the raw
    per-step relative-L1 is logged for the (input-drift -> output-drift) fit."""
    tc = TeaCache(rel_l1_thresh=1e9, record=True)
    a = torch.ones(4)
    assert tc.should_compute(a) is True          # first call: no history logged
    assert tc.should_compute(a + 0.2) is True    # records, never skips
    assert tc.should_compute(a + 0.5) is True
    assert tc.skips == 0
    assert len(tc.rel_history) == 2              # one per call after the first


def test_fitted_coefficients_round_trip_through_rescale():
    """A polynomial fit on synthetic (x, y) reproduces y via ``_rescale`` (the
    Horner eval must match ``numpy.poly1d`` order — highest degree first)."""
    np = __import__("numpy")
    xs = np.linspace(0.0, 1.0, 25)
    ys = 0.3 * xs**2 + 0.1 * xs + 0.05
    coeffs = np.polyfit(xs, ys, 4)
    tc = TeaCache(rel_l1_thresh=1.0, coefficients=[float(c) for c in coeffs])
    for x, y in zip(xs, ys):
        assert abs(tc._rescale(float(x)) - y) < 1e-6


def test_threshold_forces_recompute_and_resets():
    """With the default identity coefficients the accumulator is the running sum
    of per-step relative-L1 distances; a small step skips, and a large step that
    crosses the threshold forces a compute and resets the accumulator."""
    tc = TeaCache(rel_l1_thresh=0.5)
    a = torch.ones(4)
    assert tc.should_compute(a) is True          # first call -> compute
    assert tc.accumulated == 0.0
    assert tc.should_compute(a + 0.1) is False   # rel-L1 ~0.1 < 0.5 -> skip
    assert 0.0 < tc.accumulated < 0.5
    assert tc.should_compute(a + 5.0) is True     # big jump -> over threshold
    assert tc.accumulated == 0.0
