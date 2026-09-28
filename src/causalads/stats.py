"""Small, dependency-free statistical helpers shared across stages."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

from causalads import config as C


def diff_in_means(y: np.ndarray, t: np.ndarray, alpha: float = 0.05) -> dict:
    """Difference in means (treated - control) with Welch SE, CI, and p-value.
    For binary y this is the two-proportion difference with unpooled SE."""
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(est / se))) if se > 0 else float("nan")
    return {"estimate": float(est), "se": float(se), "ci_low": float(est - z * se),
            "ci_high": float(est + z * se), "p_value": float(p),
            "n_treated": int(len(y1)), "n_control": int(len(y0)),
            "mean_treated": float(y1.mean()), "mean_control": float(y0.mean())}


def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, n_boot: int = C.N_BOOTSTRAP,
                 alpha: float = 0.05, seed: int = C.SEED) -> tuple[float, float]:
    """Percentile bootstrap CI. `stat` receives an index array of length n."""
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """Standardized mean difference (optionally weighted), pooled unweighted SD."""
    w = np.ones_like(x, dtype=float) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1])
    m0 = np.average(x[t == 0], weights=w[t == 0])
    sd = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
    return float((m1 - m0) / sd) if sd > 0 else 0.0
