import torch

from diffucore.sampling import optimal_steps as O
from diffucore.sampling import schedules as S


# ── DP core ───────────────────────────────────────────────────────────


def test_optimal_step_schedule_picks_minimum_cost_path():
    # K=4. With 2 steps from 0 to 3, the candidate paths are [0,1,3] and
    # [0,2,3]. Make the second strictly cheaper and check the DP finds it.
    INF = 1e9
    cost = torch.full((4, 4), INF)
    cost[0, 1] = 10.0
    cost[1, 3] = 1.0   # path [0,1,3] -> 11
    cost[0, 2] = 1.0
    cost[2, 3] = 1.0   # path [0,2,3] -> 2
    assert O.optimal_step_schedule(cost, 2) == [0, 2, 3]


def test_optimal_step_schedule_endpoints_and_length():
    torch.manual_seed(0)
    K = 12
    cost = torch.rand(K, K).triu(1) + 0.01   # positive on the upper triangle
    path = O.optimal_step_schedule(cost, 5)
    assert len(path) == 6                      # num_steps + 1
    assert path[0] == 0 and path[-1] == K - 1  # pinned endpoints
    assert all(a < b for a, b in zip(path, path[1:]))   # strictly ascending


def test_optimal_step_schedule_no_worse_than_uniform():
    # The DP is optimal by construction, so its total cost must be <= any other
    # valid path's — in particular the evenly-strided ("uniform") one.
    torch.manual_seed(1)
    K, num_steps = 40, 8
    cost = torch.rand(K, K).triu(1) + 0.01

    path = O.optimal_step_schedule(cost, num_steps)
    dp_total = sum(cost[i, j].item() for i, j in zip(path, path[1:]))

    uniform = sorted(set(torch.linspace(0, K - 1, num_steps + 1).round().int().tolist()))
    # linspace endpoints already include 0 and K-1; rounding keeps them distinct here.
    assert uniform[0] == 0 and uniform[-1] == K - 1 and len(uniform) == num_steps + 1
    uni_total = sum(cost[i, j].item() for i, j in zip(uniform, uniform[1:]))

    assert dp_total <= uni_total + 1e-6


def test_optimal_step_schedule_invalid_args_raise():
    import pytest

    cost = torch.rand(6, 6).triu(1)
    with pytest.raises(ValueError):
        O.optimal_step_schedule(cost, 0)
    with pytest.raises(ValueError):
        O.optimal_step_schedule(cost, 6)          # num_steps must be <= K-1
    with pytest.raises(ValueError):
        O.optimal_step_schedule(torch.rand(3, 4), 2)   # non-square


# ── calibration ───────────────────────────────────────────────────────


def _const_denoise(target):
    def denoise(x, sigma):
        return target.expand_as(x).clone()
    return denoise


def test_calibrate_const_denoiser_is_valid_schedule():
    # A constant-x0 denoiser makes the rectified-flow ODE exactly linear, so
    # every sub-schedule is error-free; the result must still be a well-formed,
    # descending schedule with the conventional endpoints.
    target = torch.full((1, 4, 4, 4), 0.1)
    grid = S.flow_matching_schedule(40, shift=3.0)[:-1]   # drop trailing 0
    out = O.calibrate_oss_schedule(_const_denoise(target), torch.randn(1, 4, 4, 4), grid, num_steps=10)

    assert out.shape[0] == 11                      # num_steps + 1
    assert out[-1].item() == 0.0
    assert torch.isfinite(out).all()
    assert torch.all(out[:-1] >= out[1:])          # descending
    assert abs(out[0].item() - float(grid[0])) < 1e-6     # starts at grid max
    assert abs(out[-2].item() - float(grid[-1])) < 1e-6   # last nonzero == grid min


def test_calibrate_concentrates_steps_where_trajectory_bends():
    # A denoiser whose x0 estimate swings sharply in a narrow σ band creates
    # local curvature there; OSS should place more steps in that band than a
    # uniform schedule would. We check the chosen sigmas cluster around σ≈0.5.
    torch.manual_seed(0)
    target_lo = torch.full((1, 2, 4, 4), -1.0)
    target_hi = torch.full((1, 2, 4, 4), 1.0)

    def denoise(x, sigma):
        s = float(sigma.reshape(-1)[0])
        # x0 flips between two targets across a sharp transition at σ≈0.5.
        w = torch.sigmoid(torch.tensor((s - 0.5) / 0.03))
        return (w * target_hi + (1 - w) * target_lo).expand_as(x).clone()

    grid = S.flow_matching_schedule(60, shift=1.0)[:-1]
    out = O.calibrate_oss_schedule(denoise, torch.randn(1, 2, 4, 4), grid, num_steps=10)[:-1]

    # Count chosen sigmas inside the curved band vs a uniform-stride baseline.
    band = ((out >= 0.4) & (out <= 0.6)).sum().item()
    uniform = grid[torch.linspace(0, len(grid) - 1, 10).round().long()]
    uniform_band = ((uniform >= 0.4) & (uniform <= 0.6)).sum().item()
    assert band > uniform_band


def test_calibrate_fires_progress_callback():
    # One callback per teacher-trajectory step (K-1 of them), ending at (total, total).
    target = torch.zeros(1, 2, 4, 4)
    grid = S.flow_matching_schedule(20, shift=3.0)[:-1]
    seen = []
    O.calibrate_oss_schedule(
        _const_denoise(target), torch.randn(1, 2, 4, 4), grid, num_steps=6,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert len(seen) == len(grid) - 1
    assert seen[0] == (1, len(grid) - 1)
    assert seen[-1] == (len(grid) - 1, len(grid) - 1)


def test_calibrate_invalid_args_raise():
    import pytest

    target = torch.zeros(1, 2, 4, 4)
    grid = S.flow_matching_schedule(20, shift=3.0)[:-1]
    with pytest.raises(ValueError):
        O.calibrate_oss_schedule(_const_denoise(target), torch.randn(1, 2, 4, 4), grid, num_steps=1)
    with pytest.raises(ValueError):
        O.calibrate_oss_schedule(_const_denoise(target), torch.randn(1, 2, 4, 4), grid.flip(0), num_steps=5)
