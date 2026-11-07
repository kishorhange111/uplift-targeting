"""Evaluating uplift scores and targeting policies on randomised data.

Individual effects are never observed, so each measure compares treated and control customers within a group.
"""

from __future__ import annotations

import numpy as np

GRID = np.linspace(0, 1, 101)


def ranking(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Best first, with ties broken at random so that a constant score behaves like random targeting."""
    tiebreak = np.random.default_rng(seed).permutation(len(score))
    return np.lexsort((tiebreak, -score))


def incremental_curve(score: np.ndarray, t: np.ndarray, y: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """Extra outcomes per customer in the whole population from treating only the top fraction ``phi`` by score.

    For the top k: (treated mean - control mean) within the top k, times k, divided by n. At phi = 1 this is the
    average treatment effect; random targeting gives a straight line from 0 to it.
    """
    n = len(score)
    order = ranking(score)
    yt, yc = np.cumsum((y * t)[order]), np.cumsum((y * (1 - t))[order])
    nt, nc = np.cumsum(t[order]), np.cumsum(1 - t[order])
    out = np.zeros(len(grid))
    for i, phi in enumerate(grid):
        k = int(round(phi * n))
        if k == 0 or nt[k - 1] == 0 or nc[k - 1] == 0:
            continue
        out[i] = (yt[k - 1] / nt[k - 1] - yc[k - 1] / nc[k - 1]) * k / n
    return out


def qini_area(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Area between the curve and the random-targeting line, in outcomes per customer. Random targeting scores 0."""
    curve = incremental_curve(score, t, y)
    return float(np.trapezoid(curve - GRID * curve[-1], GRID))


def uplift_in_top(score: np.ndarray, t: np.ndarray, y: np.ndarray, fraction: float) -> float:
    """Observed treated-minus-control mean outcome among the top ``fraction`` by score."""
    top = ranking(score)[: int(round(fraction * len(score)))]
    tt, yy = t[top], y[top]
    return float(yy[tt == 1].mean() - yy[tt == 0].mean())


def by_decile(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> list[dict]:
    """Predicted against observed uplift per score decile, best first: is the ranking calibrated?"""
    order = ranking(score)
    rows = []
    for d, idx in enumerate(np.array_split(order, 10)):
        tt, yy = t[idx], y[idx]
        rows.append({"decile": d + 1, "predicted": round(float(score[idx].mean()), 4), "observed": round(float(yy[tt == 1].mean() - yy[tt == 0].mean()), 4), "n": len(idx)})
    return rows


def bootstrap(stat, arrays: tuple[np.ndarray, ...], strata: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95 % percentile interval, resampling customers within each arm."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        values.append(stat(*(a[idx] for a in arrays)))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def ipw_value(policy: np.ndarray, arm: np.ndarray, y: np.ndarray, propensity: dict[int, float]) -> float:
    """Mean outcome per customer if everyone followed ``policy``, from customers whose random arm agreed with it.

    Unbiased because the arm was assigned at random with known probabilities (Horvitz-Thompson).
    """
    p = np.array([propensity[a] for a in arm])
    return float(np.mean(y * (policy == arm) / p))
