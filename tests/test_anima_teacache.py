"""TeaCache (arXiv:2411.19108) on the Cosmos/Anima DiT.

Verifies the cache is a transparent accelerator: when it doesn't skip it must
be bit-identical to a plain forward, and when it does skip it must reuse an
*exact* residual (so a repeated input reproduces its own earlier output). Uses
the cheap small-config CosmosDiT — no Anima weights needed.
"""

from __future__ import annotations

import math

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


def test_default_order_is_linear_forecast():
    """The default cache forecasts (order 1), not freezes."""
    assert TeaCache(rel_l1_thresh=1.0).max_order == 1


def test_order0_reuses_last_residual_exactly():
    """``max_order=0`` is the original cache-then-reuse: a skip returns the last
    computed residual unchanged, however many steps have passed."""
    tc = TeaCache(rel_l1_thresh=1.0, max_order=0)
    r = torch.tensor([2.0, -4.0, 7.0])
    tc.calls = 1
    tc.update(r)                 # single activation at step 1
    for step in (2, 5, 9):       # every later skip reuses r exactly
        tc.calls = step
        assert torch.equal(tc.forecast(), r)


def test_single_activation_forecast_reuses_last_residual():
    """Order 1 can't extrapolate from one data point, so the first skips after a
    lone activation still reuse the residual (matches order-0 until a 2nd
    activation gives a slope)."""
    tc = TeaCache(rel_l1_thresh=1.0, max_order=1)
    r = torch.tensor([1.0, 2.0])
    tc.calls = 1
    tc.update(r)
    tc.calls = 3
    assert torch.equal(tc.forecast(), r)


def test_order1_linear_extrapolation_over_uneven_gap():
    """Two activations set a per-step slope ``(r1-r0)/gap``; later skips are the
    linear extrapolation ``r1 + slope·(step - last_activation)`` — and the gap is
    the actual (uneven) activation spacing, not a fixed 1."""
    tc = TeaCache(rel_l1_thresh=1.0, max_order=1)
    r0 = torch.tensor([0.0, 0.0])
    r1 = torch.tensor([2.0, 4.0])       # activation gap = 2 steps → slope [1, 2]
    tc.calls = 1
    tc.update(r0)
    tc.calls = 3
    tc.update(r1)
    tc.calls = 4                         # 1 step past the last activation
    assert torch.allclose(tc.forecast(), torch.tensor([3.0, 6.0]))
    tc.calls = 5                         # 2 steps past
    assert torch.allclose(tc.forecast(), torch.tensor([4.0, 8.0]))


def test_higher_order_reduces_error_on_a_curved_residual():
    """On a residual that curves in the step index, each extra Taylor order
    tracks it better: order 0 (freeze) is worst, order 2 best. (The Taylor form
    ``Σ rⁱ·kⁱ/i!`` is an approximation, not an exact polynomial fit — hence a
    shrinking error rather than zero.)"""
    f = lambda s: torch.tensor([float(s * s)])   # residual curves as step²
    def error_at_order(order):
        tc = TeaCache(rel_l1_thresh=1.0, max_order=order)
        for step in (1, 2, 3):
            tc.calls = step
            tc.update(f(step))
        tc.calls = 4                             # forecast one step past step 3
        return float((tc.forecast() - f(4)).abs())
    e0, e1, e2 = error_at_order(0), error_at_order(1), error_at_order(2)
    assert e2 < e1 < e0


def _forecast_after(activations, k_at, **teacache_kwargs):
    """Feed ``activations`` = [(step, residual), …] through ``update`` and return
    the forecast at step ``k_at``."""
    tc = TeaCache(rel_l1_thresh=1.0, **teacache_kwargs)
    for step, r in activations:
        tc.calls = step
        tc.update(r)
    tc.calls = k_at
    return tc.forecast()


def test_hermite_at_sigma_inv_sqrt2_order1_is_exactly_taylor():
    """HiCache's degradation guarantee: ``σ·H_1(σk) = 2σ²k``, so at σ = 1/√2 an
    order-1 hermite forecast is bit-identical to the order-1 taylor one."""
    acts = [(1, torch.tensor([0.0, 1.0])), (3, torch.tensor([2.0, 5.0]))]
    for k_at in (4, 5, 8):
        taylor = _forecast_after(acts, k_at, max_order=1)
        hermite = _forecast_after(acts, k_at, max_order=1, basis="hermite", sigma=2 ** -0.5)
        assert torch.equal(taylor, hermite)


def test_hermite_order2_matches_closed_form():
    """Order-2 scaled-Hermite forecast against hand-computed values:
    ``r + Δ¹·σ·H₁(σk) + Δ²/2!·σ²·H₂(σk)`` with H₁(x)=2x, H₂(x)=4x²−2."""
    # Unit-gap activations with residuals 1, 4, 9 → Δ¹ = 5, Δ² = 2 at step 3.
    acts = [(1, torch.tensor([1.0])), (2, torch.tensor([4.0])), (3, torch.tensor([9.0]))]
    sigma, k = 0.5, 2  # forecast at step 5
    x = sigma * k
    expect = 9.0 + 5.0 * (sigma * 2 * x) + (2.0 / 2.0) * (sigma ** 2 * (4 * x * x - 2))
    got = _forecast_after(acts, 5, max_order=2, basis="hermite", sigma=sigma)
    assert torch.allclose(got, torch.tensor([expect]))


def test_hermite_first_skip_and_order0_reuse_residual_exactly():
    """H̃₀ ≡ 1, so hermite changes nothing before a slope exists: a lone
    activation (and order 0 generally) still reuses the residual bit-exactly."""
    r = torch.tensor([1.0, -2.0])
    assert torch.equal(_forecast_after([(1, r)], 4, max_order=2, basis="hermite"), r)
    assert torch.equal(_forecast_after([(1, r), (2, r + 1)], 3, max_order=0, basis="hermite"),
                       r + 1)


def test_hermite_damps_taylor_overshoot_at_turning_points():
    """The HiCache selling point: on a residual that turns (rises then falls),
    Taylor's monotone extrapolation overshoots; the σ-contracted Hermite forecast
    stays closer to the truth."""
    f = lambda s: torch.tensor([math.sin(1.2 * s)])  # turns within a few steps
    acts = [(s, f(s)) for s in (1, 2, 3)]
    truth = f(5)
    taylor = _forecast_after(acts, 5, max_order=2)
    hermite = _forecast_after(acts, 5, max_order=2, basis="hermite", sigma=0.5)
    assert (hermite - truth).abs().item() < (taylor - truth).abs().item()


def test_unknown_basis_and_forecast_mode_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        TeaCache(rel_l1_thresh=1.0, basis="chebyshev")
    with pytest.raises(ValueError):
        TeaCache(rel_l1_thresh=1.0, rule="chebyshev")
    from diffucore.pipelines._anima import _make_teacache
    with pytest.raises(ValueError):
        _make_teacache(0.15, None, 4.0, "cubic")
    with pytest.raises(ValueError):
        _make_teacache(0.15, None, 4.0, "hermite", "bogus")


def test_easy_rejects_record_mode():
    """Calibration fits the drift rule's (input-drift -> output-drift) curve;
    EasyCache has no such curve, so the combination is a programming error."""
    import pytest

    with pytest.raises(ValueError):
        TeaCache(rel_l1_thresh=0.0, record=True, rule="easy")


def test_make_teacache_forecast_modes():
    """The pipeline-level ``teacache_forecast`` string maps to the paper setups:
    hermite = order-2 σ=0.5 scaled-Hermite (HiCache default), taylor = the
    order-1 linear TaylorSeer forecast."""
    from diffucore.pipelines._anima import _make_teacache

    cond, uncond = _make_teacache(0.15, None, 4.0, "hermite")
    assert (cond.basis, cond.max_order, cond.sigma) == ("hermite", 2, 0.5)
    assert (uncond.basis, uncond.max_order, uncond.sigma) == ("hermite", 2, 0.5)
    cond, uncond = _make_teacache(0.15, None, 1.0, "taylor")
    assert (cond.basis, cond.max_order) == ("taylor", 1)
    assert uncond is None  # CFG off -> single stream, as before
    assert _make_teacache(0.0, None, 4.0, "hermite") == (None, None)


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


# --------------------------------------------------------------------------- #
# EasyCache decision rule (arXiv:2507.02860)
# --------------------------------------------------------------------------- #

def test_easy_rule_warmup_always_computes():
    """The first ``warmup`` calls run the blocks whatever the threshold says —
    the transformation rate isn't measurable yet — so they stay bit-exact."""
    dit, cfg = _tiny()
    torch.manual_seed(10)
    t = torch.tensor([5.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    tc = TeaCache(rel_l1_thresh=0.05, rule="easy", warmup=3)
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(1, cfg.in_channels, 1, 8, 8)
            assert torch.equal(dit(x, t, ctx), dit(x, t, ctx, teacache=tc))
    assert (tc.calls, tc.skips) == (3, 0)


def test_easy_rule_identical_input_skips_after_warmup():
    """Once the rate is known, a step whose latent didn't move predicts zero
    output change and skips — reusing the cached residual exactly (order 0)."""
    dit, cfg = _tiny()
    torch.manual_seed(11)
    t = torch.tensor([5.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    a = torch.randn(1, cfg.in_channels, 1, 8, 8)
    b = torch.randn(1, cfg.in_channels, 1, 8, 8)
    # max_order=0 so the skip reuses the residual rather than extrapolating it;
    # this test is about the decision, not the forecast.
    tc = TeaCache(rel_l1_thresh=1e9, rule="easy", warmup=1, max_order=0)
    with torch.no_grad():
        dit(a, t, ctx, teacache=tc)          # call 1: no history
        out2 = dit(b, t, ctx, teacache=tc)   # call 2: k not yet known -> computes
        assert tc.skips == 0 and tc.k is not None
        out3 = dit(b, t, ctx, teacache=tc)   # call 3: dx == 0 -> skip
    assert tc.skips == 1
    assert torch.allclose(out2, out3, atol=1e-5, rtol=1e-4)


def test_easy_rate_updates_only_on_computed_steps():
    """Drive the rule directly: ``k`` is refreshed only after a step that ran
    the blocks, and the predicted-error accumulator carries across skips and
    resets on a recompute."""
    import pytest

    tc = TeaCache(rel_l1_thresh=0.5, rule="easy", warmup=1)
    x = torch.zeros(4)
    assert tc.should_compute_easy(x) is True      # first call -> no history
    tc.record_output(torch.zeros(4))
    assert tc.k is None                            # no previous output to difference

    x = x + 1.0                                    # dx = 1.0
    assert tc.should_compute_easy(x) is True       # k still unknown -> compute
    tc.record_output(torch.full((4,), 2.0))        # k = |2 - 0| / 1.0
    assert tc.k == pytest.approx(2.0)

    x = x + 0.1                                    # eps = 2.0 * 0.1 / 2.0 = 0.1 < 0.5
    assert tc.should_compute_easy(x) is False
    assert tc.accumulated == pytest.approx(0.1)
    tc.record_output(torch.full((4,), 2.5))        # skipped: rate must not move
    assert tc.k == pytest.approx(2.0)

    x = x + 1.0                                    # eps = 2.0 * 1.0 / 2.5 = 0.8 -> 0.9 >= 0.5
    assert tc.should_compute_easy(x) is True
    assert tc.accumulated == 0.0
    tc.record_output(torch.full((4,), 5.5))        # computed: k = |5.5 - 2.5| / 1.0
    assert tc.k == pytest.approx(3.0)
    assert (tc.calls, tc.skips) == (4, 1)


def test_easy_zero_threshold_never_skips():
    """τ = 0 makes every accumulated prediction cross the bar, so the rule
    degrades to a plain forward however far the latent moves."""
    dit, cfg = _tiny()
    torch.manual_seed(12)
    t = torch.tensor([5.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    tc = TeaCache(rel_l1_thresh=0.0, rule="easy", warmup=1)
    with torch.no_grad():
        for _ in range(4):
            x = torch.randn(1, cfg.in_channels, 1, 8, 8)
            assert torch.equal(dit(x, t, ctx), dit(x, t, ctx, teacache=tc))
    assert tc.skips == 0


def test_make_teacache_rule_and_coeffs():
    """``rule="easy"`` reaches both streams and drops the calibration
    coefficients (it has no rescale); ``"drift"`` keeps them."""
    from diffucore.pipelines._anima import _EASY_WARMUP, _make_teacache

    coeffs = (2.0, 0.5)
    cond, uncond = _make_teacache(0.15, coeffs, 4.0, "hermite", "easy")
    assert (cond.rule, cond.warmup, cond.coefficients) == ("easy", _EASY_WARMUP, (1.0, 0.0))
    assert (uncond.rule, uncond.coefficients) == ("easy", (1.0, 0.0))
    assert cond.basis == "hermite"          # the forecast is orthogonal to the rule
    cond, _ = _make_teacache(0.15, coeffs, 4.0, "hermite", "drift")
    assert (cond.rule, cond.coefficients) == ("drift", coeffs)


def test_drift_rule_unchanged():
    """Guard: naming the default rule explicitly changes nothing — same skips,
    same accumulator, bit-identical outputs."""
    dit, cfg = _tiny()
    torch.manual_seed(13)
    t = torch.tensor([5.0])
    ctx = torch.randn(1, 16, cfg.crossattn_emb_channels)
    base = torch.randn(1, cfg.in_channels, 1, 8, 8)
    step = torch.randn(1, cfg.in_channels, 1, 8, 8) * 0.01   # small drift -> some skips
    xs = [base + i * step for i in range(5)]
    old = TeaCache(rel_l1_thresh=0.5)
    new = TeaCache(rel_l1_thresh=0.5, rule="drift")
    with torch.no_grad():
        for x in xs:
            assert torch.equal(dit(x, t, ctx, teacache=old), dit(x, t, ctx, teacache=new))
    assert old.skips == new.skips > 0        # the scenario must actually skip
    assert old.accumulated == new.accumulated
    assert new.prev_x is None and new.k is None   # the easy state stays untouched
