"""Difference-in-differences on the GeoLift example geo panel (Meta, simulated).

Primary inference is an exact randomization test: "treatment" is reassigned to every
same-size subset of never-treated locations over the same window. With 2 treated
units, cluster-robust SEs are shown but should not be trusted.
"""
from __future__ import annotations

import re
import time
import urllib.request
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize

from causalads import config as cfg
from causalads import io
from causalads.data import load_geolift

VIGNETTE_URL = "https://raw.githubusercontent.com/facebookincubator/GeoLift/main/vignettes/GeoLift_Walkthrough.Rmd"
# Facts below were extracted from the vignette text (verified at build time by
# `fetch_vignette_facts`, which regex-checks each one and raises if the source changed).
TREATED = ["chicago", "portland"]
START_T, END_T = 91, 105  # 1-indexed time stamps of the treatment window
N_PRE = START_T - 1
ALPHA = 0.05
N_MATCH = 10  # matched-control subset size
N_JOBS = 4


# ---------------------------------------------------------------- source facts
def fetch_vignette_facts() -> dict:
    text = urllib.request.urlopen(VIGNETTE_URL, timeout=60).read().decode()
    checks = {
        "simulated": re.search(r"simulated data of 40 US cities across 90 days", text),
        "y_is_conversions": re.search(r"Y \(number of conversions/KPI in each day/location\)", text),
        "budget_and_length": re.search(r"\\\$65,000 and 15-day marketing campaign was executed in the cities of Chicago and Portland", text),
        "window": re.search(r"treatment_start_time = 91,\s*treatment_end_time = 105", text),
        "result": re.search(r"5\.4% lift in units sold corresponding to 4667 incremental units", text),
        "att": re.search(r"Average Estimated Treatment Effect is of 155\.556 units", text),
        "p": re.search(r"only a 1\.1% chance", text),
        "p_summary": re.search(r"p-value which was equal to 0\.01", text),
    }
    missing = [k for k, v in checks.items() if v is None]
    if missing:
        raise RuntimeError(f"vignette no longer matches expected text: {missing}")
    return {
        "source_url": VIGNETTE_URL,
        "y_definition": "number of conversions/KPI in each day/location (vignette wording)",
        "data_is_simulated": True,
        "data_note": "Vignette: 'we will use simulated data of 40 US cities across 90 days'",
        "treated_locations": TREATED,
        "treatment_window_timestamps": [START_T, END_T],
        "campaign_budget_usd": 65000,
        "campaign_length_days": 15,
        "geolift_published": {
            "method": "GeoLift (augmented synthetic control, default settings), 95% level",
            "lift_pct": 5.4, "att_per_location_day": 155.556, "incremental_y": 4667,
            "p_value": 0.01, "p_value_text_pct": 1.1,
            "note": "Rounded values as quoted in the vignette prose; the vignette gives no CI in text.",
        },
    }


# ---------------------------------------------------------------- panel helpers
def prepare(test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = test.copy()
    dates = np.sort(df["date"].unique())
    df["t"] = pd.Series(df["date"]).map({d: i + 1 for i, d in enumerate(dates)}).to_numpy()
    n_loc, n_t = df["location"].nunique(), len(dates)
    per_loc = df.groupby("location")["date"].nunique()
    checks = {
        "n_locations": int(n_loc), "n_days": int(n_t), "rows": int(len(df)),
        "balanced": bool((per_loc == n_t).all() and len(df) == n_loc * n_t),
        "missing_values": int(df[["location", "Y", "date"]].isna().sum().sum()),
        "duplicate_location_date": int(df.duplicated(["location", "date"]).sum()),
        "contiguous_daily": bool((np.diff(dates).astype("timedelta64[D]").astype(int) == 1).all()),
        "date_min": str(pd.Timestamp(dates[0]).date()), "date_max": str(pd.Timestamp(dates[-1]).date()),
        "treatment_start_date": str(pd.Timestamp(dates[START_T - 1]).date()),
        "treatment_end_date": str(pd.Timestamp(dates[END_T - 1]).date()),
        "y_min": int(df["Y"].min()), "y_nonnegative": bool((df["Y"] >= 0).all()),
    }
    if not (checks["balanced"] and checks["missing_values"] == 0 and n_t == END_T):
        raise ValueError(f"panel failed checks: {checks}")
    df["treated"] = df["location"].isin(TREATED).astype(int)
    df["post"] = (df["t"] >= START_T).astype(int)
    df["D"] = df["treated"] * df["post"]
    df["logY"] = np.log(df["Y"].clip(lower=1))
    return df, checks


def wide(df: pd.DataFrame, col: str = "Y") -> pd.DataFrame:
    return df.pivot(index="location", columns="t", values=col)


# Stats take treated series A (k, T) or aggregate rows and control aggregate B; all
# vectorized over the leading (placebo) axis. Windows are column slices.
def ratio_stat(A: np.ndarray, B: np.ndarray, t0: int) -> np.ndarray:
    return (A[..., t0:].mean(-1) / A[..., :t0].mean(-1)) / (B[..., t0:].mean(-1) / B[..., :t0].mean(-1)) - 1


def level_stat(A: np.ndarray, B: np.ndarray, t0: int) -> np.ndarray:
    return (A[..., t0:].mean(-1) - A[..., :t0].mean(-1)) - (B[..., t0:].mean(-1) - B[..., :t0].mean(-1))


def placebo_series(Yc: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """All same-size subsets of the control pool as pseudo-treated: (A, B, subsets)."""
    subs = list(combinations(range(Yc.shape[0]), size))
    idx = np.array(subs)
    A = Yc[idx].mean(1)
    B = (Yc.sum(0) - Yc[idx].sum(1)) / (Yc.shape[0] - size)
    return A, B, subs


def perm_p(obs: float, placebo: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(placebo) >= abs(obs))) / (len(placebo) + 1))


def invert_ci(obs: float, placebo: np.ndarray, kind: str) -> tuple[float, float]:
    """Test-inversion CI: theta is kept if the p-value of the theta-adjusted statistic
    exceeds ALPHA. Ratio stat: adjusting post outcomes by 1/(1+theta); level: shifting."""
    grid = np.linspace(-0.6, 0.8, 14001) if kind == "ratio" else obs + np.linspace(-1, 1, 20001) * 6 * np.abs(placebo).max()
    adj = (1 + obs) / (1 + grid) - 1 if kind == "ratio" else obs - grid
    a = np.sort(np.abs(placebo))
    p = (1 + (len(a) - np.searchsorted(a, np.abs(adj), side="left"))) / (len(a) + 1)
    keep = grid[p > ALPHA]
    return float(keep.min()), float(keep.max())


# ---------------------------------------------------------------- estimators
def scm_weights(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """min ||y - Xw||^2 s.t. w>=0, sum w = 1 (SLSQP, analytic gradient)."""
    s = y.mean()
    y, X = y / s, X / s
    J = X.shape[1]
    f = lambda w: float(np.sum((y - X @ w) ** 2))
    g = lambda w: -2 * X.T @ (y - X @ w)
    r = minimize(f, np.full(J, 1 / J), jac=g, bounds=[(0, 1)] * J, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(J)}],
                 options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def scm_one(a: np.ndarray, X: np.ndarray, t0: int) -> tuple[float, float, np.ndarray]:
    w = scm_weights(a[:t0], X[:t0])
    synth = X @ w
    lift = a[t0:].mean() / synth[t0:].mean() - 1
    rmspe = float(np.sqrt(np.mean((a[:t0] - synth[:t0]) ** 2)))
    return float(lift), rmspe, synth


def matched_stat(A: np.ndarray, Yd: np.ndarray, t0: int, k: int = N_MATCH) -> float:
    """Level & ratio DiD vs the k donors most correlated with A over the pre-period."""
    Ap, Xp = A[:t0] - A[:t0].mean(), Yd[:, :t0] - Yd[:, :t0].mean(1, keepdims=True)
    corr = (Xp @ Ap) / (np.linalg.norm(Xp, axis=1) * np.linalg.norm(Ap))
    sel = np.argsort(-corr)[:k]
    return float(ratio_stat(A, Yd[sel].mean(0), t0))


def matched_placebo(Yc: np.ndarray, sub: tuple[int, ...], t0: int) -> float:
    rest = np.delete(np.arange(Yc.shape[0]), sub)
    return matched_stat(Yc[list(sub)].mean(0), Yc[rest], t0)


def scm_placebo(Yc: np.ndarray, sub: tuple[int, ...], t0: int) -> tuple[float, float, float]:
    rest = np.delete(np.arange(Yc.shape[0]), sub)
    lift, rmspe, synth = scm_one(Yc[list(sub)].mean(0), Yc[rest].T, t0)
    a = Yc[list(sub)].mean(0)
    return lift, rmspe, float(np.sqrt(np.mean((a[t0:] - synth[t0:]) ** 2)))


# ---------------------------------------------------------------- event study
def bins(t: np.ndarray) -> np.ndarray:
    k = np.floor((t - START_T) / 7).astype(int)
    return np.clip(k, -12, 1)  # merge 6-day leftover into -12 and 1-day tail into +1


def es_coefs(logY: np.ndarray, is_t: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Weekly event-study coefficients (ref bin -1) = bin-mean of treated-minus-control
    log gap minus the reference-bin mean. Numerically identical to the TWFE
    regression with treated x bin dummies (asserted against statsmodels once)."""
    k = bins(t)
    gap = logY[is_t].mean(0) - logY[~is_t].mean(0)
    ks = [b for b in sorted(set(k)) if b != -1]
    return np.array([gap[k == b].mean() - gap[k == -1].mean() for b in ks]), ks


def event_study(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["k"] = bins(d["t"].to_numpy())
    ks = [b for b in sorted(d["k"].unique()) if b != -1]
    for b in ks:
        d[f"e{b + 20}"] = ((d["k"] == b) & (d["treated"] == 1)).astype(int)
    names = [f"e{b + 20}" for b in ks]
    m = smf.ols("logY ~ " + " + ".join(names) + " + C(location) + C(date)", d)
    res = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["location"])[0]})
    coef = res.params[names].to_numpy()
    se = res.bse[names].to_numpy()
    lead_idx = [i for i, b in enumerate(ks) if b <= -2]
    pos = {n: j for j, n in enumerate(res.params.index)}
    R = np.zeros((len(lead_idx), len(res.params)))
    for r, i in enumerate(lead_idx):
        R[r, pos[names[i]]] = 1
    wald_c = res.wald_test(R, use_f=True, scalar=True)
    res_h = m.fit(cov_type="HC1")
    wald_h = res_h.wald_test(R, use_f=True, scalar=True)
    piv = wide(df, "logY")
    fast, fks = es_coefs(piv.to_numpy(), piv.index.isin(TREATED), piv.columns.to_numpy())
    assert fks == ks and np.allclose(fast, coef, atol=1e-8), "fast event-study formula != regression"
    return {"ks": ks, "coef": coef, "se_cluster": se, "lead_idx": lead_idx,
            "wald_cluster_F": float(wald_c.statistic), "wald_cluster_p": float(wald_c.pvalue),
            "wald_hc1_F": float(wald_h.statistic), "wald_hc1_p": float(wald_h.pvalue),
            "wald_df": [len(lead_idx), int(res.df_resid)]}


# ---------------------------------------------------------------- power / MDE
def power_curve(pre: pd.DataFrame, effects: np.ndarray, n_sims: int, seed: int) -> pd.DataFrame:
    Y = wide(pre.assign(t=pre.groupby("location").cumcount() + 1)).to_numpy().astype(float)
    L, T = Y.shape
    win = END_T - START_T + 1
    rng = np.random.default_rng(seed)
    # same draws across effect sizes so the curve is monotone up to test noise only
    draws = [(rng.choice(L, 2, replace=False), int(rng.integers(45, T - win + 1))) for _ in range(n_sims)]
    rows = []
    for e in effects:
        hits = 0
        for sub, t0 in draws:
            Yw = Y[:, : t0 + win].copy()
            Yw[sub, t0:] *= 1 + e
            ctrl = np.delete(np.arange(L), sub)
            Yc = Yw[ctrl]
            obs = float(ratio_stat(Yw[sub].mean(0), Yc.mean(0), t0))
            A, B, _ = placebo_series(Yc, 2)
            hits += perm_p(obs, ratio_stat(A, B, t0)) <= ALPHA
        lo, hi = stats.binomtest(hits, n_sims).proportion_ci(0.95, method="wilson")
        rows.append({"effect": float(e), "power": hits / n_sims, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def mde_at(curve: pd.DataFrame, target: float = 0.8) -> float | None:
    x, y = curve["effect"].to_numpy(), curve["power"].to_numpy()
    for i in range(1, len(x)):
        if y[i] >= target > y[i - 1]:
            return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return float(x[0]) if y[0] >= target else None


# ---------------------------------------------------------------- run
def run() -> None:
    io.style()
    t_start = time.time()
    facts = fetch_vignette_facts()
    pre_df, test_df = load_geolift()
    df, checks = prepare(test_df)
    same_as_test = bool(np.array_equal(
        wide(pre_df.assign(t=pre_df.groupby("location").cumcount() + 1)).to_numpy(),
        wide(df[df["t"] <= N_PRE]).to_numpy()))
    checks["pretest_equals_first_90_days_of_test"] = same_as_test
    ntr, ndays = len(TREATED), END_T - START_T + 1

    Y = wide(df)
    locs = Y.index.to_numpy()
    is_t = np.isin(locs, TREATED)
    Yt, Yc = Y.to_numpy()[is_t].astype(float), Y.to_numpy()[~is_t].astype(float)
    A_obs, B_obs = Yt.mean(0), Yc.mean(0)
    t0 = N_PRE

    # ---- classic 2x2 and permutation distributions (exact: C(38,2) = 703 subsets)
    obs_ratio = float(ratio_stat(A_obs, B_obs, t0))
    obs_level = float(level_stat(A_obs, B_obs, t0))
    Ap, Bp, subs = placebo_series(Yc, ntr)
    pl_ratio, pl_level = ratio_stat(Ap, Bp, t0), level_stat(Ap, Bp, t0)
    p_ratio, p_level = perm_p(obs_ratio, pl_ratio), perm_p(obs_level, pl_level)
    ci_ratio, ci_level = invert_ci(obs_ratio, pl_ratio, "ratio"), invert_ci(obs_level, pl_level, "level")
    t_post_total = float(Yt[:, t0:].sum())
    cf_mean = float(A_obs[t0:].mean() / (1 + obs_ratio))  # counterfactual treated per-location-day mean
    inc = lambda th: t_post_total * th / (1 + th)  # incremental conversions implied by ratio lift th
    cell = {
        "treated_pre_mean": float(A_obs[:t0].mean()), "treated_post_mean": float(A_obs[t0:].mean()),
        "control_pre_mean": float(B_obs[:t0].mean()), "control_post_mean": float(B_obs[t0:].mean()),
    }

    # ---- TWFE (levels, log) with clustered SEs
    grp = pd.factorize(df["location"])[0]
    def twfe(y: str) -> dict:
        r = smf.ols(f"{y} ~ D + C(location) + C(date)", df).fit(cov_type="cluster", cov_kwds={"groups": grp})
        ci = r.conf_int(alpha=ALPHA).loc["D"].to_numpy()
        tcrit = stats.t.ppf(1 - ALPHA / 2, df=len(TREATED) - 1)  # G_treated-1 df: most conservative rule of thumb
        return {"coef": float(r.params["D"]), "se_cluster": float(r.bse["D"]), "p_cluster_normal": float(r.pvalues["D"]),
                "ci_cluster_normal": [float(ci[0]), float(ci[1])],
                "ci_cluster_t_1df": [float(r.params["D"] - tcrit * r.bse["D"]), float(r.params["D"] + tcrit * r.bse["D"])]}
    tw_lvl, tw_log = twfe("Y"), twfe("logY")
    tw_log["pct_lift"] = float(np.exp(tw_log["coef"]) - 1)
    tw_log["pct_ci_cluster_normal"] = [float(np.exp(x) - 1) for x in tw_log["ci_cluster_normal"]]

    # ---- event study / parallel trends
    es = event_study(df)
    logY = np.log(Y.clip(lower=1)).to_numpy()
    pl_coefs = np.array([es_coefs(np.vstack([logY[~is_t][list(s)], logY[~is_t][[i for i in range(len(logY[~is_t])) if i not in s]]]),
                                  np.r_[np.ones(len(s), bool), np.zeros(len(logY[~is_t]) - len(s), bool)],
                                  Y.columns.to_numpy())[0] for s in subs])
    lead = es["lead_idx"]
    S_obs, S_pl = float(np.sum(es["coef"][lead] ** 2)), np.sum(pl_coefs[:, lead] ** 2, 1)
    pretrend_perm_p = float((1 + np.sum(S_pl >= S_obs)) / (len(S_pl) + 1))
    es["pretrend_perm_p"], es["lead_rms_log_points"] = pretrend_perm_p, float(np.sqrt(S_obs / len(lead)))
    es["placebo_band"] = np.quantile(pl_coefs, [0.025, 0.975], axis=0)

    # ---- synthetic control (+ in-space placebos, each placebo re-fits weights)
    lift_s, rmspe_s, synth = scm_one(A_obs, Yc.T, t0)
    w = scm_weights(A_obs[:t0], Yc[:, :t0].T)
    res = Parallel(n_jobs=N_JOBS)(delayed(scm_placebo)(Yc, s, t0) for s in subs)
    pl_scm, pl_rm_pre, pl_rm_post = (np.array(x) for x in zip(*res))
    p_scm = perm_p(lift_s, pl_scm)
    ci_scm = invert_ci(lift_s, pl_scm, "ratio")
    post_rmspe = float(np.sqrt(np.mean((A_obs[t0:] - synth[t0:]) ** 2)))
    ratio_rmspe = post_rmspe / rmspe_s
    p_rmspe = float((1 + np.sum(pl_rm_post / pl_rm_pre >= ratio_rmspe)) / (len(pl_rm_pre) + 1))
    top_w = sorted(zip(np.array(Y.index[~is_t]), w), key=lambda x: -x[1])[:5]
    scm_gap_post = float(A_obs[t0:].mean() - synth[t0:].mean())

    # ---- matched-control DiD (matching re-run for every placebo subset)
    m_obs = matched_stat(A_obs, Yc, t0)
    pl_m = np.array(Parallel(n_jobs=N_JOBS)(delayed(matched_placebo)(Yc, s, t0) for s in subs))
    p_m, ci_m = perm_p(m_obs, pl_m), invert_ci(m_obs, pl_m, "ratio")
    Ap_, Xp_ = A_obs[:t0] - A_obs[:t0].mean(), Yc[:, :t0] - Yc[:, :t0].mean(1, keepdims=True)
    corr = (Xp_ @ Ap_) / (np.linalg.norm(Xp_, axis=1) * np.linalg.norm(Ap_))
    matched_names = list(Y.index[~is_t][np.argsort(-corr)[:N_MATCH]])

    # ---- power / MDE on the untreated pre-test panel
    effects = np.round(np.arange(0, 0.1501, 0.01), 3)
    curve = power_curve(pre_df, effects, n_sims=500, seed=cfg.SEED)
    mde = mde_at(curve)
    fpr = float(curve.loc[curve["effect"] == 0, "power"].iloc[0])

    # ---- in-time placebo: same pair, every 15-day window of the pre-period as fake treatment
    g = np.log(A_obs / B_obs)
    win = ndays
    def in_time(s0: int) -> float:
        m = np.zeros(t0, bool)
        m[s0:s0 + win] = True
        return float(g[:t0][m].mean() - g[:t0][~m].mean())
    it_pl = np.array([in_time(s0) for s0 in range(0, t0 - win + 1)])
    it_obs = float(g[t0:].mean() - g[:t0].mean())
    it_p = float((1 + np.sum(np.abs(it_pl) >= abs(it_obs))) / (len(it_pl) + 1))

    # ---- business translation
    inc_pt, inc_ci = inc(obs_ratio), [inc(ci_ratio[0]), inc(ci_ratio[1])]
    budget = facts["campaign_budget_usd"]
    cpic = budget / inc_pt
    cpic_ci = [budget / inc_ci[1], budget / inc_ci[0] if inc_ci[0] > 0 else None]
    gl = facts["geolift_published"]

    def att_from_lift(th: float) -> float:
        return inc(th) / (ntr * ndays)
    rows = [
        {"method": "2x2 DiD, levels", "att": obs_level, "att_ci": ci_level, "lift": obs_level / (A_obs[t0:].mean() - obs_level),
         "lift_ci": None, "p": p_level, "p_kind": "permutation (703 placebo pairs)", "incremental": obs_level * ntr * ndays},
        {"method": "Ratio DiD, % lift (primary)", "att": att_from_lift(obs_ratio), "att_ci": [att_from_lift(x) for x in ci_ratio],
         "lift": obs_ratio, "lift_ci": list(ci_ratio), "p": p_ratio, "p_kind": "permutation (703 placebo pairs)", "incremental": inc_pt},
        {"method": "TWFE, levels", "att": tw_lvl["coef"], "att_ci": tw_lvl["ci_cluster_normal"], "lift": tw_lvl["coef"] / (A_obs[t0:].mean() - tw_lvl["coef"]),
         "lift_ci": None, "p": tw_lvl["p_cluster_normal"], "p_kind": "cluster-robust (2 treated clusters; unreliable)", "incremental": tw_lvl["coef"] * ntr * ndays},
        {"method": "TWFE, log Y", "att": None, "att_ci": None, "lift": tw_log["pct_lift"], "lift_ci": tw_log["pct_ci_cluster_normal"],
         "p": tw_log["p_cluster_normal"], "p_kind": "cluster-robust (unreliable)", "incremental": None},
        {"method": "Synthetic control", "att": scm_gap_post, "att_ci": [att_from_lift(x) for x in ci_scm], "lift": lift_s, "lift_ci": list(ci_scm),
         "p": p_scm, "p_kind": "permutation (703 placebo pairs)", "incremental": scm_gap_post * ntr * ndays},
        {"method": f"DiD, top-{N_MATCH} correlated controls", "att": att_from_lift(m_obs), "att_ci": [att_from_lift(x) for x in ci_m],
         "lift": m_obs, "lift_ci": list(ci_m), "p": p_m, "p_kind": "permutation (703 placebo pairs)", "incremental": inc(m_obs)},
        {"method": "GeoLift published (benchmark)", "att": gl["att_per_location_day"], "att_ci": None, "lift": gl["lift_pct"] / 100, "lift_ci": None,
         "p": gl["p_value"], "p_kind": "GeoLift conformal/permutation, as published", "incremental": gl["incremental_y"]},
    ]

    headline = {
        "did_att_per_location_day": {"estimate": att_from_lift(obs_ratio), "ci95": [att_from_lift(x) for x in ci_ratio], "perm_p": p_ratio,
                                     "note": "ratio-DiD counterfactual; 2x2 level DiD in `estimators`"},
        "in_time_placebo": {"p_value": it_p, "obs_log_points": it_obs, "n_windows": len(it_pl),
                            "placebo_abs_max_log_points": float(np.abs(it_pl).max()),
                            "note": "same treated pair, overlapping pre-period 15-day windows; windows are dependent"},
        "did_att_level_2x2_per_location_day": {"estimate": obs_level, "ci95": list(ci_level), "perm_p": p_level},
        "did_pct_lift": {"estimate": obs_ratio, "ci95": list(ci_ratio), "perm_p": p_ratio},
        "pretrend_test": {"permutation_p": pretrend_perm_p, "wald_cluster_p": es["wald_cluster_p"], "wald_hc1_p": es["wald_hc1_p"]},
        "synthetic_control_pct_lift": {"estimate": lift_s, "ci95": list(ci_scm), "perm_p": p_scm, "pre_rmspe": rmspe_s},
        "geolift_published": gl,
        "incremental_conversions": {"estimate": inc_pt, "ci95": inc_ci},
        "cost_per_incremental_conversion_usd": {"estimate": cpic, "ci95": cpic_ci, "budget_usd": budget},
        "mde_80pct_power_pct": mde * 100 if mde is not None else None,
    }

    # ---------------------------------------------------------------- figures
    dates = np.sort(df["date"].unique())
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    ax[0].plot(dates, A_obs, color=io.SERIES[0], label="Chicago + Portland (mean)")
    ax[0].plot(dates, synth, color=io.SERIES[1], ls="--", label="Synthetic control")
    ax[1].plot(dates, A_obs - synth, color=io.SERIES[0])
    ax[1].axhline(0, color=io.REFERENCE, lw=1)
    for a in ax:
        a.axvline(dates[t0], color=io.REFERENCE, ls=":", lw=1.2)
    ax[0].text(dates[t0], ax[0].get_ylim()[1], " campaign start", va="top", color=io.TEXT_MUTED, fontsize=9)
    ax[0].set_title("Treated cities vs synthetic control (simulated GeoLift example data)", pad=26)
    ax[0].set_ylabel("Conversions per city-day")
    ax[1].set_ylabel("Gap (conversions/city-day)")
    ax[1].set_xlabel("Date (2021)")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax[1].tick_params(axis="x", rotation=30)
    io.save_fig(fig, "did_synthetic_control")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.array(es["ks"])
    band = es["placebo_band"]
    ax.fill_between(x, band[0], band[1], color=io.SERIES[0], alpha=0.15, label="95% range of 703 placebo pairs")
    ax.errorbar(x, es["coef"], yerr=1.96 * es["se_cluster"], fmt="o-", color=io.SERIES[1], capsize=3,
                label="Chicago + Portland (95% cluster CI, unreliable)")
    ax.axhline(0, color=io.REFERENCE, lw=1)
    ax.axvline(-0.5, color=io.REFERENCE, ls=":", lw=1.2)
    ax.set_title("Event study: weekly effect on log conversions vs week -1")
    ax.set_xlabel("Weeks relative to campaign start (week 0 = days 91-97; week 1 = days 98-105)")
    ax.set_ylabel("Effect on log(conversions) (log points)")
    ax.legend(loc="upper left")
    io.save_fig(fig, "did_event_study")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(pl_ratio * 100, bins=40, color=io.SERIES[0], alpha=0.75, label="Placebo pairs (703)")
    ax.axvline(obs_ratio * 100, color=io.SERIES[1], lw=2.2, label=f"Chicago + Portland: {obs_ratio * 100:.1f}%")
    ax.axvline(0, color=io.REFERENCE, lw=1)
    ax.set_title(f"Randomization test: permutation p = {p_ratio:.3f}")
    ax.set_xlabel("DiD % lift over 15-day window (%)")
    ax.set_ylabel("Number of placebo pairs")
    ax.legend()
    io.save_fig(fig, "did_permutation")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(curve["effect"] * 100, curve["power"], "o-", color=io.SERIES[0], label="Power (500 simulations/point)")
    ax.fill_between(curve["effect"] * 100, curve["ci_low"], curve["ci_high"], color=io.SERIES[0], alpha=0.18)
    ax.axhline(0.8, color=io.REFERENCE, ls="--", lw=1, label="80% power")
    ax.axhline(ALPHA, color=io.REFERENCE, ls=":", lw=1, label="5% test size")
    if mde is not None:
        ax.axvline(mde * 100, color=io.SERIES[1], ls="-", lw=1.5, label=f"MDE = {mde * 100:.1f}%")
    ax.set_title("Power of the DiD + permutation test (2 pseudo-treated cities, 15 days)")
    ax.set_xlabel("Injected lift (%)")
    ax.set_ylabel("Power (share of simulations with p <= 0.05)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    io.save_fig(fig, "did_power")

    # ---------------------------------------------------------------- outputs
    runtime = time.time() - t_start
    out = {
        "source_facts": facts, "data_checks": checks, "cell_means_per_location_day": cell,
        "n_placebo_subsets": len(subs), "permutation_lift_pct_quantiles_2.5_97.5": np.quantile(pl_ratio * 100, [0.025, 0.975]).tolist(),
        "estimators": rows,
        "twfe": {"levels": tw_lvl, "log": tw_log,
                 "warning": "2 treated clusters: cluster-robust SEs are badly biased downward; treat as indicative only."},
        "event_study": {**es, "placebo_band": es["placebo_band"].tolist()},
        "synthetic_control": {"pre_rmspe": rmspe_s, "pre_rmspe_pct_of_mean": rmspe_s / float(A_obs[:t0].mean()) * 100,
                              "post_rmspe": post_rmspe, "post_pre_rmspe_ratio": ratio_rmspe, "rmspe_ratio_perm_p": p_rmspe,
                              "post_gap_per_location_day": scm_gap_post, "top_weights": top_w,
                              "placebo_pre_rmspe_median": float(np.median(pl_rm_pre))},
        "matched_controls": matched_names,
        "power": curve.to_dict(orient="records"), "mde_80pct": mde, "false_positive_rate_at_zero": fpr,
        "business": {"incremental_conversions": inc_pt, "incremental_ci95": inc_ci, "budget_usd": budget,
                     "cost_per_incremental_conversion_usd": cpic, "cpic_ci95": cpic_ci,
                     "geolift_implied_cpic": budget / gl["incremental_y"]},
        "headline": headline, "runtime_seconds": runtime,
    }
    io.save_json(out, "did")
    (cfg.RESULTS / "did.md").write_text(render_md(out, rows, curve, mde, es, pl_ratio))
    print(f"did done in {runtime:.0f}s; lift {obs_ratio:.3%} CI {ci_ratio} p={p_ratio:.4f}; SCM {lift_s:.3%}; MDE {mde}")


def f_ci(ci: list | None, fmt: str = "{:.1f}") -> str:
    return "n/a" if ci is None else "[" + ", ".join(fmt.format(x) for x in ci) + "]"


def render_md(o: dict, rows: list[dict], curve: pd.DataFrame, mde: float | None, es: dict, pl_ratio: np.ndarray) -> str:
    f, ck, h, b = o["source_facts"], o["data_checks"], o["headline"], o["business"]
    cm, sc = o["cell_means_per_location_day"], o["synthetic_control"]
    tr = "".join(
        f"| {r['method']} | {'n/a' if r['att'] is None else f'{r['att']:.1f}'} | {f_ci(r['att_ci'])} | {r['lift'] * 100:.1f}% | "
        f"{f_ci(None if r['lift_ci'] is None else [x * 100 for x in r['lift_ci']])}{'' if r['lift_ci'] is None else '%'} | "
        f"{'n/a' if r['incremental'] is None else f'{r['incremental']:,.0f}'} | {r['p']:.3f} ({r['p_kind']}) |\n" for r in rows)
    pw = "".join(f"| {r['effect'] * 100:.0f}% | {r['power']:.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] |\n" for r in o["power"])
    cp = b["cpic_ci95"]
    cpic_ci = f"[${cp[0]:.2f}, {'unbounded' if cp[1] is None else f'${cp[1]:.2f}'}]"
    inc_lo, inc_hi = b["incremental_ci95"]
    mde_txt = "not reached within 0-15%" if mde is None else f"{mde * 100:.1f}%"
    return f"""## Difference-in-Differences on a geo experiment (GeoLift example panel)

**Method.** The customer-level Hillstrom data has no pre/post panel, so DiD uses Meta's GeoLift example panel: {ck['n_locations']} US cities x {ck['n_days']} days ({ck['date_min']} to {ck['date_max']}), Y = daily conversions. Per the [GeoLift walkthrough]({f['source_url']}), Chicago and Portland ran a ${f['campaign_budget_usd']:,}, {f['campaign_length_days']}-day campaign (time stamps {f['treatment_window_timestamps'][0]}-{f['treatment_window_timestamps'][1]}, {ck['treatment_start_date']} to {ck['treatment_end_date']}); the other {ck['n_locations'] - 2} cities are controls. The panel is balanced with no missing values. I estimate a 2x2 DiD, TWFE (location + date fixed effects), a ratio ("% lift") DiD, a synthetic control (non-negative weights summing to 1, fit on the 90 pre-days), and DiD against the {N_MATCH} controls most correlated with the treated pair in the pre-period. **Inference is an exact randomization test**: "treatment" is reassigned to each of the {o['n_placebo_subsets']} = C(38,2) pairs of control cities over the same window (re-fitting weights/matches for every pair), giving permutation p-values and test-inversion CIs. Estimates in the headline use the ratio DiD because city sizes differ ~5x (2.3k to 13.6k conversions/day), so a level-scale placebo distribution is dominated by big-city noise.

![Treated vs synthetic control](results/figures/did_synthetic_control.png)

### Headline

| Quantity | Estimate | 95% CI |
|---|---|---|
| DiD % lift | {h['did_pct_lift']['estimate'] * 100:.1f}% | {f_ci([x * 100 for x in h['did_pct_lift']['ci95']])}% |
| DiD ATT (conversions per city-day) | {h['did_att_per_location_day']['estimate']:.1f} | {f_ci(h['did_att_per_location_day']['ci95'])} |
| Permutation p-value (two-sided) | {h['did_pct_lift']['perm_p']:.3f} (min. attainable {1 / (o['n_placebo_subsets'] + 1):.4f}) | |
| Pre-trend test, permutation p | {h['pretrend_test']['permutation_p']:.3f} | Wald F p: {h['pretrend_test']['wald_cluster_p']:.2g} (cluster, degenerate), {h['pretrend_test']['wald_hc1_p']:.3f} (HC1) |
| Incremental conversions, 15 days | {b['incremental_conversions']:,.0f} | [{inc_lo:,.0f}, {inc_hi:,.0f}] |
| Cost per incremental conversion | ${b['cost_per_incremental_conversion_usd']:.2f} | {cpic_ci} |
| In-time placebo p (same pair, {h['in_time_placebo']['n_windows']} pre-period windows) | {h['in_time_placebo']['p_value']:.3f} | observed {h['in_time_placebo']['obs_log_points'] * 100:.1f} log pts; largest placebo window {h['in_time_placebo']['placebo_abs_max_log_points'] * 100:.1f} |
| MDE at 80% power (this design) | {mde_txt} | |

### All estimators vs GeoLift's published result

| Method | ATT / city-day | 95% CI | % lift | 95% CI | Incremental (15 d) | p (kind) |
|---|---|---|---|---|---|---|
{tr}
Cell means per city-day: treated {cm['treated_pre_mean']:.0f} pre / {cm['treated_post_mean']:.0f} post; controls {cm['control_pre_mean']:.0f} pre / {cm['control_post_mean']:.0f} post. Level-DiD CIs are wide because the level placebo distribution mixes cities of very different size. GeoLift's numbers are the vignette's rounded prose values (it gives no CI in text; it uses augmented synthetic control with fixed effects). The cross-city permutation test (which uses only untreated cities' noise) is much more conservative than GeoLift's published p = 0.01, which is based on the treated pair's own fitted series; the in-time placebo above is the closer analogue, but it only measures the pair's own day-to-day volatility (overlapping windows, so p is approximate) and ignores drift relative to the control pool. Cost per incremental conversion implied by GeoLift's 4,667: ${b['geolift_implied_cpic']:.2f}.

![Permutation distribution](results/figures/did_permutation.png)

### Parallel trends

Weekly event study on log conversions (reference = week -1). Joint test that the {len(es['lead_idx'])} pre-period lead coefficients are zero: permutation p = {es['pretrend_perm_p']:.3f}; Wald F({es['wald_df'][0]}, {es['wald_df'][1]}) p = {es['wald_cluster_p']:.2g} with cluster-robust SEs (degenerate: the clustered covariance rests on 2 treated clusters and should be ignored) and {es['wald_hc1_p']:.3f} with HC1 SEs. RMS lead coefficient: {es['lead_rms_log_points'] * 100:.2f} log points. A non-rejection is weak evidence for parallel trends when power is limited, and the parametric tests are unreliable with 2 treated units, so the placebo band is the fairer yardstick.

![Event study](results/figures/did_event_study.png)

### Synthetic control

Pre-period RMSPE {sc['pre_rmspe']:.1f} conversions/city-day ({sc['pre_rmspe_pct_of_mean']:.2f}% of the treated mean); post-period gap {sc['post_gap_per_location_day']:.1f} per city-day; post/pre RMSPE ratio {sc['post_pre_rmspe_ratio']:.1f} (placebo-rank p = {sc['rmspe_ratio_perm_p']:.3f}). Largest donor weights: {', '.join(f'{n} {w_:.2f}' for n, w_ in sc['top_weights'])}.

### Power and minimum detectable effect

Using the untreated GeoLift_PreTest panel, I inject a multiplicative lift into 2 random pseudo-treated cities over a random 15-day window (pre-period = all earlier days, at least 45; effect grid 0-15%), and run the full ratio-DiD + 703-pair permutation test ({len(o['power']) and 500} simulations per effect size). Size at 0% lift: {o['false_positive_rate_at_zero'] * 100:.1f}% (nominal 5%). **MDE at 80% power: {mde_txt}.**

| Injected lift | Power | 95% Wilson CI |
|---|---|---|
{pw}
![Power curve](results/figures/did_power.png)

### Caveats

- **The panel is simulated example data published by Meta** ("simulated data of 40 US cities" per the GeoLift vignette), not a real advertiser's campaign. Treat all numbers as a methods demonstration.
- Only 2 treated cities: cluster-robust SEs (TWFE, event study Wald) are unreliable and are shown for completeness; the permutation test is the primary inference. With {o['n_placebo_subsets']} placebo pairs the smallest attainable p-value is {1 / (o['n_placebo_subsets'] + 1):.4f}.
- Treated cities were chosen by GeoLift's market-selection algorithm for fit, not randomized; the randomization test assumes exchangeability of city pairs, which selection on pre-period fit only approximately satisfies.
- Level DiD assumes additive parallel trends across cities of very different size; the ratio/log versions assume proportional trends and are preferred here.
- Test-inversion CIs assume a constant multiplicative effect over the window. Budget-based cost per conversion uses the vignette's stated ${f['campaign_budget_usd']:,} and ignores any baseline spend.
- The GeoLift benchmark is the rounded value quoted in the vignette prose, not a re-run of the R package; differences reflect method (fixed-effects augmented SCM vs. plain DiD/SCM).
- Power results are for this design on this panel (pre-period 45-75 days, 2 cities, 15 days) and use the same permutation test, with 500 simulations per effect size (Wilson CIs shown).
"""
