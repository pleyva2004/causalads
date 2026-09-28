"""Cross-fitted nuisance models and ATE/ATT estimators for the observational stage.

Every estimator returns (estimate, standard error). Pure numpy/sklearn so one
replicate is cheap enough to repeat a few hundred times.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalads.stats import diff_in_means

PS_CLIP = (0.01, 0.99)  # numerical guard only; no unit is dropped, so the estimand is unchanged
CALIPER_SD = 0.2  # Austin (2011): 0.2 SD of the logit propensity score
N_STRATA = 5
K_FOLDS = 5
# Fixed a priori and never tuned against the RCT benchmark.
GBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_jobs=1, verbose=-1)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _glm(binary: bool):
    # Large C = effectively unpenalized main-effects logistic regression on standardized X.
    return LogisticRegression(C=1e4, max_iter=5000) if binary else LinearRegression()


def _gbm(binary: bool, seed: int):
    cls = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
    return cls(**GBM_PARAMS, random_state=seed)


def _predict(model, X: np.ndarray, binary: bool) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if binary else model.predict(X)


def crossfit(X_glm: np.ndarray, X_gbm: np.ndarray, t: np.ndarray, outcomes: dict[str, np.ndarray],
             folds: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Out-of-fold propensity (logistic, GBM) and per-arm outcome predictions (GLM, GBM T-learner)."""
    n = len(t)
    out = {k: np.empty(n) for k in ("ps_logit", "ps_gbm")}
    for y in outcomes:
        for m in ("glm", "gbm"):
            out[f"{y}_mu0_{m}"], out[f"{y}_mu1_{m}"] = np.empty(n), np.empty(n)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        out["ps_logit"][te] = _glm(True).fit(X_glm[tr], t[tr]).predict_proba(X_glm[te])[:, 1]
        out["ps_gbm"][te] = _gbm(True, seed).fit(X_gbm[tr], t[tr]).predict_proba(X_gbm[te])[:, 1]
        for y, yv in outcomes.items():
            binary = bool(np.isin(yv, (0, 1)).all())
            for arm in (0, 1):
                fit = tr & (t == arm)
                g = _glm(binary).fit(X_glm[fit], yv[fit])
                out[f"{y}_mu{arm}_glm"][te] = _predict(g, X_glm[te], binary)
                b = _gbm(binary, seed).fit(X_gbm[fit], yv[fit])
                out[f"{y}_mu{arm}_gbm"][te] = _predict(b, X_gbm[te], binary)
    return out


def clip(ps: np.ndarray) -> np.ndarray:
    return np.clip(ps, *PS_CLIP)


# ---------- estimators ----------
def naive(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    r = diff_in_means(y, t)
    return r["estimate"], r["se"]


def ols(y: np.ndarray, t: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Coefficient on T in Y ~ 1 + T + X with HC1 standard errors."""
    Z = np.column_stack([np.ones(len(t)), t, X])
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    beta = ZtZ_inv @ Z.T @ y
    u = y - Z @ beta
    Zu = Z * u[:, None]
    V = ZtZ_inv @ (Zu.T @ Zu) @ ZtZ_inv * len(y) / (len(y) - Z.shape[1])
    return float(beta[1]), float(np.sqrt(V[1, 1]))


def stratified(y: np.ndarray, t: np.ndarray, ps: np.ndarray) -> tuple[float, float]:
    """Within-quintile differences in means, weighted by stratum size (ATE)."""
    edges = np.quantile(ps, np.linspace(0, 1, N_STRATA + 1)[1:-1])
    s = np.searchsorted(edges, ps, side="right")
    est, var = 0.0, 0.0
    for k in np.unique(s):
        m = s == k
        y1, y0 = y[m & (t == 1)], y[m & (t == 0)]
        w = m.mean()
        est += w * (y1.mean() - y0.mean())
        var += w**2 * (y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(est), float(np.sqrt(var))


def hajek(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Normalized weighted difference in means and its influence function (weights taken as known)."""
    s1, s0 = (w * t).sum(), (w * (1 - t)).sum()
    m1, m0 = (w * t * y).sum() / s1, (w * (1 - t) * y).sum() / s0
    n = len(y)
    phi = n * (w * t * (y - m1) / s1 - w * (1 - t) * (y - m0) / s0)
    return float(m1 - m0), phi


def ipw(y: np.ndarray, t: np.ndarray, ps: np.ndarray) -> tuple[float, float]:
    """Hajek IPW for the ATE; SE ignores PS estimation (conservative for ATE under a correct MLE PS)."""
    ps = clip(ps)
    est, phi = hajek(y, t, t / ps + (1 - t) / (1 - ps))
    return est, float(phi.std(ddof=1) / np.sqrt(len(y)))


def aipw(y: np.ndarray, t: np.ndarray, ps: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> tuple[float, float]:
    ps = clip(ps)
    psi = mu1 - mu0 + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(y)))


def _nearest(src: np.ndarray, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index into `pool` of the nearest value to each `src` value (1-D), and the distance."""
    order = np.argsort(pool, kind="stable")
    sp = pool[order]
    pos = np.searchsorted(sp, src)
    lo, hi = np.clip(pos - 1, 0, len(sp) - 1), np.clip(pos, 0, len(sp) - 1)
    pick = np.where(np.abs(sp[lo] - src) <= np.abs(sp[hi] - src), lo, hi)
    return order[pick], np.abs(sp[pick] - src)


def match(y: np.ndarray, t: np.ndarray, ps: np.ndarray, sigma2: np.ndarray, estimand: str,
          rng: np.random.Generator) -> tuple[float, float, float]:
    """1:1 nearest-neighbour matching on logit PS, with replacement, caliper 0.2 SD.

    ATE matches every unit to its nearest opposite-arm unit; ATT matches treated units
    only. Variance is Abadie-Imbens (2006) with M=1; sigma2 is a per-unit conditional
    outcome variance. Ignoring PS estimation is conservative for the ATE (Abadie-Imbens
    2016), not guaranteed for the ATT. Returns (estimate, se, share dropped by caliper).
    """
    # Tiny jitter = random tie-breaking; otherwise one of many identical-PS controls absorbs all matches.
    lp = _logit(clip(ps)) + rng.uniform(-1e-9, 1e-9, len(ps))
    cal = CALIPER_SD * lp.std()
    i1, i0 = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    j, d = _nearest(lp[i1], lp[i0])
    m1, k1 = i0[j], d <= cal
    n = len(y)
    if estimand == "att":
        diff = y[i1[k1]] - y[m1[k1]]
        tau = diff.mean()
        K = np.bincount(m1[k1], minlength=n)
        var = (((diff - tau) ** 2).sum() + (K * (K - 1) * sigma2).sum()) / k1.sum() ** 2
        return float(tau), float(np.sqrt(var)), float(1 - k1.mean())
    j, d = _nearest(lp[i0], lp[i1])
    m0, k0 = i1[j], d <= cal
    diff = np.r_[y[i1[k1]] - y[m1[k1]], y[m0[k0]] - y[i0[k0]]]
    tau = diff.mean()
    K = np.bincount(np.r_[m1[k1], m0[k0]], minlength=n)
    var = (((diff - tau) ** 2).sum() + ((K**2 + K) * sigma2).sum()) / len(diff) ** 2
    return float(tau), float(np.sqrt(var)), float(1 - len(diff) / n)
