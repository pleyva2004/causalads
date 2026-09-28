"""Uplift evaluation on a randomized hold-out: Qini/uplift curves with (paired) bootstrap,
GATES, the Chernozhukov-Demirer-Duflo-Fernandez-Val BLP test, Cochran's Q, and IPW
off-policy values. Bootstrap draws are shared multinomial weights so every comparison
between models/policies is paired."""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats as sps

from causalads import config as C
from causalads import stats as S

GRID = np.linspace(0.0, 1.0, 101)
_CHUNK = 250


def boot_weights(n: int, n_boot: int = C.N_BOOTSTRAP, seed: int = C.SEED) -> np.ndarray:
    """(n_boot, n) multinomial count matrix; row b = how often each unit appears in draw b."""
    rng = np.random.default_rng(seed)
    W = np.empty((n_boot, n), dtype=np.uint8)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def order(score: np.ndarray, seed: int = C.SEED) -> np.ndarray:
    """Descending order with random (seeded) tie-breaking, so ties don't inherit row order."""
    tie = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tie, -score))


def _curves(y: np.ndarray, t: np.ndarray, score: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Qini (Radcliffe) and uplift-gain curves on GRID for each weight row.
    qini(phi) = [Y_T - Y_C * N_T/N_C] / N ; gain(phi) = (ybar_T - ybar_C) * N(phi)/N."""
    o = order(score)
    ys, ts = y[o], t[o].astype(float)
    n = len(y)
    starts = np.r_[0, np.round(GRID[1:-1] * n).astype(int)]
    q_out, g_out = [], []
    for i in range(0, len(W), _CHUNK):
        Ws = W[i:i + _CHUNK][:, o].astype(float)
        seg = lambda v: np.add.reduceat(Ws * v, starts, axis=1).cumsum(axis=1)  # noqa: E731
        nt, nc, yt, yc = seg(ts), seg(1 - ts), seg(ys * ts), seg(ys * (1 - ts))
        N = Ws.sum(axis=1, keepdims=True)
        ratio = np.divide(nt, nc, out=np.zeros_like(nt), where=nc > 0)
        mt = np.divide(yt, nt, out=np.zeros_like(yt), where=nt > 0)
        mc = np.divide(yc, nc, out=np.zeros_like(yc), where=nc > 0)
        z = np.zeros((len(Ws), 1))
        q_out.append(np.hstack([z, (yt - yc * ratio) / N]))
        g_out.append(np.hstack([z, (mt - mc) * (nt + nc) / N]))
    return np.vstack(q_out), np.vstack(g_out)


def _area_over_random(curve: np.ndarray) -> np.ndarray:
    return np.trapezoid(curve, GRID, axis=1) - curve[:, -1] / 2


def uplift_eval(y: np.ndarray, t: np.ndarray, score: np.ndarray, W: np.ndarray | None) -> dict:
    """Point curves + (if W) bootstrap draws of Qini coefficient, normalized Qini, AUUC gain.
    Normalized Qini divides by the perfect-ordering Qini area (sklift convention: treated
    responders first, control responders last)."""
    ones = np.ones((1, len(y)), dtype=np.uint8)
    perfect = y * t - y * (1 - t)
    out = {}
    for tag, w in [("point", ones), ("boot", W)]:
        if w is None:
            continue
        q, g = _curves(y, t, score, w)
        qp, _ = _curves(y, t, perfect, w)
        coef = _area_over_random(q)
        out[tag] = {"qini_curve": q, "gain_curve": g, "qini_coef": coef,
                    "norm_qini": coef / _area_over_random(qp), "auuc_gain": _area_over_random(g)}
    return out


def ci(draws: np.ndarray, est: float, scale: float = 1.0) -> dict:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(est * scale), "ci_low": float(lo * scale), "ci_high": float(hi * scale)}


def paired(d_a: np.ndarray, d_b: np.ndarray, est_a: float, est_b: float, scale: float = 1.0) -> dict:
    """a - b with paired bootstrap CI and two-sided bootstrap p-value (sign of draws)."""
    diff = d_a - d_b
    p = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
    return {**ci(diff, est_a - est_b, scale), "p_value": p}


def by_group(y: np.ndarray, t: np.ndarray, groups: np.ndarray) -> list[dict]:
    """Observed treated-minus-control per group (valid: treatment is randomized in every group)."""
    return [{"group": int(g), "n": int((groups == g).sum()), **S.diff_in_means(y[groups == g], t[groups == g])}
            for g in np.unique(groups)]


def quantile_groups(score: np.ndarray, k: int) -> np.ndarray:
    """0 = highest-score group ... k-1 = lowest (ranked with seeded tie-breaking)."""
    rank = np.empty(len(score), dtype=int)
    rank[order(score)] = np.arange(len(score))
    return (rank * k) // len(score)


def gates(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: int = 5) -> dict:
    groups = by_group(y, t, quantile_groups(score, k))
    top, bot = groups[0], groups[-1]
    diff = top["estimate"] - bot["estimate"]
    se = float(np.hypot(top["se"], bot["se"]))
    return {"groups": groups, "top_minus_bottom": {
        "estimate": diff, "se": se, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
        "p_value": float(2 * sps.norm.sf(abs(diff / se)))}}


def blp(y: np.ndarray, t: np.ndarray, cate: np.ndarray, baseline: np.ndarray, p: float = C.P_ANY_EMAIL) -> dict:
    """Best linear predictor of the CATE (Chernozhukov et al. 2018):
    y = a0 + a1 B(X) + b1 (T-p) + b2 (T-p)(S(X) - mean S) + e.
    b1 = ATE; b2 > 0 <=> the score carries real heterogeneity (b2 = 1: well calibrated).
    Weights 1/(p(1-p)) are constant under a known constant propensity, so plain OLS; HC1 SEs."""
    d = t - p
    Z = np.column_stack([np.ones_like(y), baseline, d, d * (cate - cate.mean())])
    fit = sm.OLS(y, Z).fit(cov_type="HC1")
    b, se = fit.params, fit.bse
    return {"beta1_ate": float(b[2]), "beta1_se": float(se[2]),
            "beta2": float(b[3]), "beta2_se": float(se[3]),
            "beta2_ci_low": float(b[3] - 1.96 * se[3]), "beta2_ci_high": float(b[3] + 1.96 * se[3]),
            "beta2_p_one_sided": float(sps.norm.sf(b[3] / se[3]))}


def cochran_q(groups: list[dict]) -> dict:
    """Test that independent group ATEs are equal (chi2 with K-1 df)."""
    est = np.array([g["estimate"] for g in groups])
    w = 1 / np.array([g["se"] for g in groups]) ** 2
    m = (w * est).sum() / w.sum()
    q = float((w * (est - m) ** 2).sum())
    return {"Q": q, "df": len(groups) - 1, "p_value": float(sps.chi2.sf(q, len(groups) - 1))}


def ipw_contrib(y: np.ndarray, a: np.ndarray, pi: np.ndarray, p: float = C.P_ARM) -> np.ndarray:
    """Per-unit Horvitz-Thompson contribution to V(pi) - V(no email). Its mean is unbiased
    for the policy's incremental outcome per customer because P(A=a|X) = 1/3 is known."""
    return y * ((a == pi).astype(float) - (a == 0).astype(float)) / p


def boot_means(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(m, n) contributions -> (m, n_boot) bootstrap means under shared weights W."""
    n = D.shape[1]
    return np.hstack([D @ W[i:i + _CHUNK].T.astype(float) / n for i in range(0, len(W), _CHUNK)])
