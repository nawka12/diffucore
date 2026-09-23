"""Optimal-stepsize sampling schedules (OSS; Pu et al., arXiv:2503.21774, related
to GITS).

One dense "teacher" trajectory measures the single-step Euler error between
every pair of candidate noise levels, and a dynamic program picks the ``N``-step
sub-grid with the least total error. Calibration (:func:`calibrate_oss_schedule`)
runs once per (model, resolution) on a GPU; the DP itself is cheap.
"""

from __future__ import annotations

from typing import Callable

import torch

from .schedules import append_zero

__all__ = ["optimal_step_schedule", "calibrate_oss_schedule"]


def optimal_step_schedule(cost: torch.Tensor, num_steps: int) -> list[int]:
    """Dynamic-programming optimal sub-schedule over a descending candidate grid.

    ``cost[i][j]`` (``i < j``) is the error of one step from candidate ``i`` to
    ``j``. Returns ``num_steps + 1`` ascending candidate indices from ``0``
    (highest noise) to ``K - 1`` minimizing the summed error, via
    ``dp[n][j] = min_{i<j} dp[n-1][i] + cost[i][j]``.
    """
    if cost.dim() != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square K×K matrix")
    K = cost.shape[0]
    if not 1 <= num_steps <= K - 1:
        raise ValueError(f"num_steps must be in [1, K-1]=[1, {K - 1}]; got {num_steps}")

    INF = float("inf")
    c = cost.tolist()
    # dp[n][j]: min summed cost to reach candidate j from 0 using exactly n steps.
    dp = [[INF] * K for _ in range(num_steps + 1)]
    prev = [[-1] * K for _ in range(num_steps + 1)]
    dp[0][0] = 0.0
    for n in range(1, num_steps + 1):
        for j in range(1, K):
            best, arg = INF, -1
            for i in range(j):
                if dp[n - 1][i] == INF:
                    continue
                val = dp[n - 1][i] + c[i][j]
                if val < best:
                    best, arg = val, i
            dp[n][j] = best
            prev[n][j] = arg

    if dp[num_steps][K - 1] == INF:
        raise ValueError("no valid path to the clean endpoint; check num_steps vs grid size")

    path = [K - 1]
    j = K - 1
    for n in range(num_steps, 0, -1):
        j = prev[n][j]
        path.append(j)
    path.reverse()
    return path


def calibrate_oss_schedule(
    denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    candidate_sigmas: torch.Tensor,
    num_steps: int,
    *,
    metric: str = "l2",
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Distill an error-optimal σ schedule from a fine reference trajectory.

    ``denoise(x, sigma)`` returns the x0 estimate; ``candidate_sigmas`` is a
    dense descending grid (no trailing zero). Euler-integrates the whole grid
    for teacher states, fills ``cost[i][j]`` with the error of one Euler step
    from ``x*[i]`` to level ``j``, and runs :func:`optimal_step_schedule`. Steps
    land where the trajectory bends. Returns ``num_steps + 1`` descending σ with
    a trailing 0; ``progress_callback(done, total)`` fires per teacher step.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if metric not in ("l2", "l1"):
        raise ValueError("metric must be 'l2' or 'l1'")
    sig = candidate_sigmas.to(dtype=torch.float32)
    K = sig.shape[0]
    if K < num_steps:
        raise ValueError(f"need at least num_steps={num_steps} candidate sigmas; got {K}")
    if not torch.all(sig[:-1] >= sig[1:]):
        raise ValueError("candidate_sigmas must be descending")

    s_in = x_init.new_ones([x_init.shape[0]])
    # Teacher trajectory: states at every candidate level, and the Euler slope d
    # used to leave each level (d[k] = (x - x0)/sigma).
    states = [x_init]
    slopes: list[torch.Tensor] = []
    x = x_init
    for k in range(K - 1):
        x0 = denoise(x, sig[k] * s_in).float()
        d = (x - x0) / sig[k]
        slopes.append(d)
        x = x + d * (sig[k + 1] - sig[k])
        states.append(x)
        if progress_callback is not None:
            progress_callback(k + 1, K - 1)

    cost = torch.zeros(K, K, dtype=torch.float32, device=x_init.device)
    for i in range(K - 1):
        for j in range(i + 1, K):
            pred = states[i] + slopes[i] * (sig[j] - sig[i])
            diff = pred - states[j]
            cost[i, j] = diff.pow(2).mean() if metric == "l2" else diff.abs().mean()

    # num_steps sampling steps == num_steps σ before the trailing 0, i.e. a
    # (num_steps - 1)-transition path through the candidate grid.
    idx = optimal_step_schedule(cost, num_steps - 1)
    chosen = sig[torch.tensor(idx, device=sig.device)]
    return append_zero(chosen)
