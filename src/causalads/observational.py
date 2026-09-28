"""Stage: propensity-score methods validated against the Hillstrom RCT (LaLonde-style design).

The randomized data are turned into realistic observational samples by letting a
"marketer" targeting rule e(X) decide who we keep: treated customers are retained with
prob c*e(X)/p and controls with c*(1-e(X))/(1-p), p = observed treated share. Retention
is then c for every X, so the retained sample has the population's covariate mix (the
estimand is the population ATE) while P(T=1 | X, retained) = e(X). Selection uses only
(X, T), so unconfoundedness holds by construction and the RCT difference in means on
all 64,000 customers is the ground truth each estimator should recover.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed, parallel_config
from scipy import stats as sps
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

from causalads import config as C
from causalads import data, io
from causalads import observational_estimators as E
from causalads.stats import diff_in_means, smd

N_REPLICATES = 300
N_JOBS = 4
Z = float(sps.norm.ppf(0.975))
OUTCOMES = ["visit", "conversion", "spend"]
OUT_LABEL = {"visit": "Site visit", "conversion": "Conversion", "spend": "Spend"}
# Percentage points for rates, dollars for spend.
SCALE = {"visit": 100.0, "conversion": 100.0, "spend": 1.0}
UNIT = {"visit": "pp", "conversion": "pp", "spend": "$"}

# Marketer targeting rule: an RFM-style score with a VIP interaction, a win-back arm for
# lapsed high-value customers (non-monotone in recency) and a hard $300 value threshold,
# squashed into [0.1, 0.9] so everyone keeps a chance of either arm (positivity by design).
HIGH_VALUE = 300.0
RULE = {"intercept": -1.8, "recent_buyer": 1.6, "high_value": 1.2, "vip": 1.6, "winback": 2.2,
        "ch_multi": 1.0, "both_lines": 1.0, "newbie": -0.8}
RULE_DESC = {"intercept": "Intercept", "recent_buyer": "Recent buyer (recency <= 3 mo)",
             "high_value": f"High value (history >= ${HIGH_VALUE:.0f})",
             "vip": "VIP: recent x high value", "winback": "Win-back: lapsed (recency >= 10) x high value",
             "ch_multi": "Multichannel buyer", "both_lines": "Bought both mens and womens",
             "newbie": "New customer (last 12 mo)"}
E_FLOOR, E_SPAN = 0.1, 0.8

# mens + womens - both_lines == 1 and the zip/channel dummies are full one-hots, so the
# linear models drop one column of each exact dependency.
X_GLM = [f for f in data.FEATURES if f not in ("womens", "zip_rural", "ch_web")]
RULE_TERMS = ["high_value", "lapsed", "vip", "winback"]
BALANCE = data.FEATURES + RULE_TERMS

EST = {  # key: (label, estimand, PS family)
    "naive": ("Naive difference in means", "ATE", "none"),
    "ols": ("OLS adjustment (Y ~ T + X)", "ATE", "none"),
    "strat_logit": ("PS stratification, quintiles (logistic)", "ATE", "logit"),
    "strat_gbm": ("PS stratification, quintiles (GBM)", "ATE", "gbm"),
    "match_logit": ("PS matching 1:1, caliper (logistic)", "ATE", "logit"),
    "match_gbm": ("PS matching 1:1, caliper (GBM)", "ATE", "gbm"),
    "ipw_logit": ("IPW, Hajek (logistic)", "ATE", "logit"),
    "ipw_gbm": ("IPW, Hajek (GBM)", "ATE", "gbm"),
    "aipw_logit_glm": ("AIPW (logistic PS + GLM outcome)", "ATE", "logit"),
    "aipw_gbm_gbm": ("AIPW (GBM PS + GBM outcome)", "ATE", "gbm"),
    "aipw_logit_gbm": ("AIPW (logistic PS + GBM outcome)", "ATE", "logit"),
    "aipw_gbm_glm": ("AIPW (GBM PS + GLM outcome)", "ATE", "gbm"),
    "ipw_oracle": ("IPW, Hajek (true PS, oracle)", "ATE", "none"),
    "att_match_logit": ("PS matching 1:1, caliper (logistic)", "ATT", "logit"),
    "att_match_gbm": ("PS matching 1:1, caliper (GBM)", "ATT", "gbm"),
}
ATE_KEYS = [k for k, v in EST.items() if v[1] == "ATE"]
FEASIBLE = [k for k in EST if k != "ipw_oracle"]  # oracle needs the true PS, unavailable in practice
FAMILY = {"logit": ("Logistic PS", io.SERIES[0]), "gbm": ("GBM PS", io.SERIES[1]),
          "none": ("No PS / true PS", io.SERIES[2])}


# ---------- design ----------
def rule_terms(df: pd.DataFrame) -> pd.DataFrame:
    hv = (df["history"] >= HIGH_VALUE).astype(int)
    lapsed = (df["recency"] >= 10).astype(int)
    return pd.DataFrame({"recent_buyer": df["recent_buyer"], "high_value": hv, "lapsed": lapsed,
                         "vip": df["recent_buyer"] * hv, "winback": lapsed * hv, "ch_multi": df["ch_multi"],
                         "both_lines": df["both_lines"], "newbie": df["newbie"]})


def marketer_propensity(df: pd.DataFrame) -> np.ndarray:
    terms = rule_terms(df)
    z = RULE["intercept"] + sum(RULE[k] * terms[k] for k in RULE if k != "intercept")
    return (E_FLOOR + E_SPAN / (1 + np.exp(-z))).to_numpy()


def retention(e: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-unit keep probability q and the constant c; largest c with every q <= 1."""
    p = float(t.mean())  # observed share, so E[Hajek IPW | data] is exactly the full-sample DiM
    c = min(p / e.max(), (1 - p) / (1 - e.min()))
    q = np.where(t == 1, c * e / p, c * (1 - e) / (1 - p))
    return q, c, p


def benchmarks(df: pd.DataFrame, e: np.ndarray) -> dict:
    """RCT ground truth on all customers: ATE (difference in means), the ATT of the
    observational population (treated share e(X), so E-weighted DiM), and a Lin (2013)
    covariate-adjusted ATE as a robustness check."""
    t = df["treated"].to_numpy()
    Xc = df[X_GLM].to_numpy(float)
    Xc = Xc - Xc.mean(0)
    out = {}
    for y in OUTCOMES:
        yv = df[y].to_numpy(float)
        ate = diff_in_means(yv, t)
        _, phi_ate = E.hajek(yv, t, np.ones_like(yv))
        att, phi_att = E.hajek(yv, t, e)
        n = len(yv)
        se_att = phi_att.std(ddof=1) / np.sqrt(n)
        se_gap = (phi_att - phi_ate).std(ddof=1) / np.sqrt(n)
        lin, lin_se = E.ols(yv, t, np.column_stack([Xc, t[:, None] * Xc]))
        out[y] = {"ate": ate, "att": {"estimate": att, "se": se_att, "ci_low": att - Z * se_att,
                                      "ci_high": att + Z * se_att},
                  "att_minus_ate": {"estimate": att - ate["estimate"], "ci_low": att - ate["estimate"] - Z * se_gap,
                                    "ci_high": att - ate["estimate"] + Z * se_gap},
                  "lin_adjusted_ate": {"estimate": lin, "ci_low": lin - Z * lin_se, "ci_high": lin + Z * lin_se}}
    return out


# ---------- one replicate ----------
def _weights(t: np.ndarray, ps: np.ndarray) -> np.ndarray:
    ps = E.clip(ps)
    return t / ps + (1 - t) / (1 - ps)


def _ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def _replicate(r: int, A: dict, bootstrap: bool, detail: bool = False) -> dict:
    """Draw one observational sample and run every estimator on it.

    bootstrap=True first resamples the 64,000 customers with replacement, so replicate
    spread reflects full sampling variability (not just which customers the rule kept);
    folds are assigned per original customer so duplicates never straddle train/test.
    """
    rng = np.random.default_rng([C.SEED, r] if bootstrap else C.SEED)
    N = len(A["t"])
    base = rng.integers(0, N, N) if bootstrap else np.arange(N)
    idx = base[rng.random(len(base)) < A["q"][base]]
    folds = rng.integers(0, E.K_FOLDS, N)[idx]
    t, e = A["t"][idx], A["e"][idx]
    Xg = A["X_glm"][idx]
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    ys = {y: A[y][idx] for y in OUTCOMES}
    nu = E.crossfit(Xg, A["X_gbm"][idx], t, ys, folds, seed=C.SEED + r)
    ps = {"logit": nu["ps_logit"], "gbm": nu["ps_gbm"], "oracle": e}

    est: dict[str, dict] = {}
    drop: dict[str, float] = {}
    for y, yv in ys.items():
        sig2 = np.where(t == 1, yv - nu[f"{y}_mu1_gbm"], yv - nu[f"{y}_mu0_gbm"]) ** 2
        res = {"naive": E.naive(yv, t), "ols": E.ols(yv, t, Xg), "ipw_oracle": E.ipw(yv, t, e)}
        for m in ("logit", "gbm"):
            res[f"strat_{m}"] = E.stratified(yv, t, ps[m])
            res[f"ipw_{m}"] = E.ipw(yv, t, ps[m])
            for kind, key in (("ate", f"match_{m}"), ("att", f"att_match_{m}")):
                est_, se_, drop[f"{kind}_{m}"] = E.match(yv, t, ps[m], sig2, kind, rng)
                res[key] = (est_, se_)
            for o in ("glm", "gbm"):
                res[f"aipw_{m}_{o}"] = E.aipw(yv, t, ps[m], nu[f"{y}_mu0_{o}"], nu[f"{y}_mu1_{o}"])
        est[y] = {k: tuple(map(float, res[k])) for k in EST}

    B = A["X_bal"][idx]
    smds = {name: np.array([abs(smd(B[:, j], t, w)) for j in range(B.shape[1])])
            for name, w in {"unweighted": None, "logit": _weights(t, ps["logit"]),
                            "gbm": _weights(t, ps["gbm"]), "oracle": _weights(t, e)}.items()}
    diag = {"n": int(len(t)), "treated_share": float(t.mean()), "matching_drop": drop}
    for m, p in ps.items():
        w = _weights(t, p)
        diag[m] = {"auc": float(roc_auc_score(t, p)), "rmse_vs_true": float(np.sqrt(np.mean((p - e) ** 2))),
                   "max_smd": float(smds[m].max()), "max_smd_main": float(smds[m][:len(data.FEATURES)].max()),
                   "share_outside_05_95": float(np.mean((p < 0.05) | (p > 0.95))),
                   "ess_treated": _ess(w[t == 1]), "ess_control": _ess(w[t == 0])}
    diag["unweighted"] = {"max_smd": float(smds["unweighted"].max()),
                          "max_smd_main": float(smds["unweighted"][:len(data.FEATURES)].max())}
    out = {"r": r, "est": est, "diag": diag}
    if detail:
        out["arrays"] = {"t": t, "e": e, "ps_logit": ps["logit"], "ps_gbm": ps["gbm"], "smd": smds}
    return out


# ---------- summaries ----------
def _wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def _projected_coverage(delta: float, k: float = 10.0) -> float:
    """Coverage of an honest-SE 95% CI if n grows k-fold and the bias stays put (bias = delta SDs today)."""
    shift = delta * np.sqrt(k)
    return float(sps.norm.cdf(Z - shift) - sps.norm.cdf(-Z - shift))


def summarize(reps: list[dict], bench: dict, boot: np.ndarray) -> tuple[dict, dict]:
    """Bias (Monte Carlo CI), RMSE and SE ratio (bootstrap-over-replicates CIs), CI coverage (Wilson)."""
    out: dict[str, dict] = {}
    errs: dict[str, dict] = {}
    R = len(reps)
    for y in OUTCOMES:
        out[y], errs[y] = {}, {}
        for k, (label, estimand, _) in EST.items():
            truth = bench[y]["ate" if estimand == "ATE" else "att"]["estimate"]
            est = np.array([rp["est"][y][k][0] for rp in reps])
            se = np.array([rp["est"][y][k][1] for rp in reps])
            err = est - truth
            errs[y][k] = err
            mcse = err.std(ddof=1) / np.sqrt(R)
            rmse_b = np.sqrt((err[boot] ** 2).mean(1))
            ratio_b = se[boot].mean(1) / est[boot].std(1, ddof=1)
            covered = int((np.abs(err) <= Z * se).sum())
            sd = float(est.std(ddof=1))
            out[y][k] = {
                "label": label, "estimand": estimand, "truth": truth, "mean_estimate": float(est.mean()),
                "bias": float(err.mean()), "bias_ci": [float(err.mean() - Z * mcse), float(err.mean() + Z * mcse)],
                "bias_pct": float(100 * err.mean() / truth),
                "bias_pct_ci": [float(100 * (err.mean() - Z * mcse) / truth), float(100 * (err.mean() + Z * mcse) / truth)],
                "rmse": float(np.sqrt((err**2).mean())),
                "rmse_ci": [float(np.quantile(rmse_b, 0.025)), float(np.quantile(rmse_b, 0.975))],
                "empirical_sd": sd, "mean_se": float(se.mean()),
                "se_ratio": float(se.mean() / sd),
                "se_ratio_ci": [float(np.quantile(ratio_b, 0.025)), float(np.quantile(ratio_b, 0.975))],
                "coverage": covered / R, "coverage_ci": list(_wilson(covered, R)),
                "bias_over_sd": float(err.mean() / sd),
                "projected_coverage_10x_n": _projected_coverage(float(err.mean() / sd)),
                "median_ci_halfwidth": float(np.median(Z * se)),
            }
    return out, errs


def _naive_se_check(A: dict, n: int = 3000) -> dict:
    """Welch-SE calibration of the naive visit estimator over many model-free replicates: isolates
    Monte Carlo noise in the SE ratio from genuine SE miscalibration."""
    N = len(A["t"])
    est, se = np.empty(n), np.empty(n)
    for r in range(n):
        rng = np.random.default_rng([C.SEED, r])  # same draws as the first replicates of the simulation
        base = rng.integers(0, N, N)
        idx = base[rng.random(N) < A["q"][base]]
        est[r], se[r] = E.naive(A["visit"][idx], A["t"][idx])
    boot = np.random.default_rng(C.SEED).integers(0, n, (C.N_BOOTSTRAP, n))
    rb = se[boot].mean(1) / est[boot].std(1, ddof=1)
    return {"n": n, "ratio": float(se.mean() / est.std(ddof=1)),
            "ci": [float(np.quantile(rb, 0.025)), float(np.quantile(rb, 0.975))]}


def _rmse_ratio(ea: np.ndarray, eb: np.ndarray, boot: np.ndarray) -> dict:
    """RMSE(a)/RMSE(b) on the same replicates, paired bootstrap CI."""
    rb = np.sqrt((ea[boot] ** 2).mean(1) / (eb[boot] ** 2).mean(1))
    return {"ratio": float(np.sqrt((ea**2).mean() / (eb**2).mean())),
            "ci": [float(np.quantile(rb, 0.025)), float(np.quantile(rb, 0.975))]}


def _diag_summary(reps: list[dict]) -> dict:
    def band(vals: list[float]) -> dict:
        v = np.array(vals)
        return {"mean": float(v.mean()), "p2_5": float(np.quantile(v, 0.025)), "p97_5": float(np.quantile(v, 0.975))}
    out = {"n": band([r["diag"]["n"] for r in reps]),
           "treated_share": band([r["diag"]["treated_share"] for r in reps]),
           "unweighted": {k: band([r["diag"]["unweighted"][k] for r in reps]) for k in ("max_smd", "max_smd_main")}}
    for m in ("logit", "gbm", "oracle"):
        out[m] = {k: band([r["diag"][m][k] for r in reps]) for k in reps[0]["diag"][m]}
    out["matching_drop"] = {k: band([r["diag"]["matching_drop"][k] for r in reps]) for k in reps[0]["diag"]["matching_drop"]}
    return out


def calibration(t: np.ndarray, ps: np.ndarray, n_bins: int = 10) -> dict:
    """Decile reliability table, ECE, Brier, and logistic recalibration slope (1 = calibrated)."""
    edges = np.quantile(ps, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, ps, side="right")
    rows = []
    for k in np.unique(b):
        m = b == k
        lo, hi = _wilson(int(t[m].sum()), int(m.sum()))
        rows.append({"mean_ps": float(ps[m].mean()), "observed": float(t[m].mean()), "lo": lo, "hi": hi, "n": int(m.sum())})
    ece = sum(r["n"] * abs(r["observed"] - r["mean_ps"]) for r in rows) / len(t)
    fit = sm.Logit(t, sm.add_constant(E._logit(E.clip(ps)))).fit(disp=0)
    ci = fit.conf_int()[1]
    return {"bins": rows, "ece": float(ece), "brier": float(brier_score_loss(t, ps)),
            "slope": float(fit.params[1]), "slope_ci": [float(ci[0]), float(ci[1])]}


# ---------- figures ----------
def _fig_overlap(a: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (key, title) in zip(axes, [("e", "True marketer propensity e(X)"), ("ps_logit", "Logistic PS (cross-fitted)"),
                                       ("ps_gbm", "GBM PS (cross-fitted)")], strict=True):
        for arm, name, col in [(0, "Not emailed", io.SERIES[0]), (1, "Emailed", io.SERIES[1])]:
            ax.hist(a[key][a["t"] == arm], bins=bins, density=True, alpha=0.55, color=col, label=name)
        ax.set_title(title)
        ax.set_xlabel("Propensity score")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("Overlap in the observational sample (single replicate)", fontweight="bold")
    io.save_fig(fig, "observational_overlap")


def _fig_calibration(cal: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=io.REFERENCE, ls="--", lw=1, label="Perfect calibration")
    for m, name, col in [("logit", "Logistic PS", io.SERIES[0]), ("gbm", "GBM PS", io.SERIES[1])]:
        rows = cal[m]["bins"]
        x = np.array([r["mean_ps"] for r in rows])
        yv = np.array([r["observed"] for r in rows])
        err = np.array([[r["observed"] - r["lo"] for r in rows], [r["hi"] - r["observed"] for r in rows]])
        ax.errorbar(x, yv, yerr=err, fmt="o-", color=col, ms=4, capsize=2,
                    label=f"{name} (ECE {cal[m]['ece']:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted propensity (decile bin)",
           ylabel="Observed share emailed (95% CI)", title="Propensity-score calibration (out-of-fold)")
    ax.legend(loc="upper left")
    io.save_fig(fig, "observational_calibration")


def _fig_love(smds: dict) -> None:
    order = np.argsort(smds["unweighted"])
    names = np.array(BALANCE)[order]
    ypos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.axvline(0.1, color=io.REFERENCE, ls="--", lw=1, label="|SMD| = 0.1")
    for key, name, col, mk in [("unweighted", "Unweighted", io.SERIES[2], "o"),
                               ("logit", "IPW, logistic PS", io.SERIES[0], "s"),
                               ("gbm", "IPW, GBM PS", io.SERIES[1], "D")]:
        ax.scatter(smds[key][order], ypos, color=col, marker=mk, s=28, label=name, zorder=3)
    ax.set_yticks(ypos, [f"{n} *" if n in RULE_TERMS else n for n in names])
    ax.set(xlabel="|Standardized mean difference|", title="Covariate balance before and after IPW weighting")
    ax.legend(loc="lower right", title="* targeting-rule term (not a model input)", title_fontsize=8)
    io.save_fig(fig, "observational_love_plot")


def _fig_estimates(single: dict, reps: list[dict], bench: dict, y: str) -> None:
    keys = ATE_KEYS[::-1]
    s = SCALE[y]
    truth = bench[y]["ate"]
    ypos = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax in axes:
        ax.axvspan(truth["ci_low"] * s, truth["ci_high"] * s, color=io.REFERENCE, alpha=0.18, lw=0)
        ax.axvline(truth["estimate"] * s, color=io.REFERENCE, lw=1.5)
    handles: dict[str, object] = {}
    for i, k in enumerate(keys):
        fam, col = FAMILY[EST[k][2]]
        est, se = single["est"][y][k]
        handles[fam] = axes[0].errorbar(est * s, i, xerr=Z * se * s, fmt="o", color=col, capsize=3, ms=5)
        vals = np.array([rp["est"][y][k][0] for rp in reps]) * s
        bp = axes[1].boxplot(vals, positions=[i], orientation="horizontal", widths=0.55, patch_artist=True,
                             flierprops={"markersize": 2, "markeredgecolor": col})
        for b in bp["boxes"]:
            b.set(facecolor=col, alpha=0.35, edgecolor=col)
        for part in ("whiskers", "caps", "medians"):
            for b in bp[part]:
                b.set(color=col)
    band = axes[0].fill_between([], [], color=io.REFERENCE, alpha=0.35)
    order = [FAMILY[f][0] for f in ("logit", "gbm", "none")]
    axes[0].legend([handles[f] for f in order] + [band], order + ["RCT benchmark (95% CI)"], loc="lower right", fontsize=8)
    axes[0].set_yticks(ypos, [EST[k][0] for k in keys])
    axes[0].set(xlabel=f"ATE on {OUT_LABEL[y].lower()} ({UNIT[y]}), 95% CI",
                title="One observational sample: estimate and 95% CI")
    axes[1].set(xlabel=f"ATE on {OUT_LABEL[y].lower()} ({UNIT[y]})",
                title=f"Across {len(reps)} simulated observational samples")
    io.save_fig(fig, f"observational_estimates_{y}")


def _fig_metrics(summ: dict) -> None:
    keys = FEASIBLE[::-1]
    ypos = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    off = {"visit": 0.2, "conversion": 0.0, "spend": -0.2}
    shown = [k for k in keys if k != "naive"]  # naive is ~10x off-scale; printed instead
    lim = 1.15 * max(abs(v) for y in OUTCOMES for k in shown for v in summ[y][k]["bias_pct_ci"])
    for j, y in enumerate(OUTCOMES):
        col = io.SERIES[j]
        sel = np.array([k != "naive" for k in keys])
        b = np.array([summ[y][k]["bias_pct"] for k in keys])
        bci = np.array([summ[y][k]["bias_pct_ci"] for k in keys])
        cv = np.array([summ[y][k]["coverage"] for k in keys])
        cci = np.array([summ[y][k]["coverage_ci"] for k in keys])
        axes[0].errorbar(b[sel], ypos[sel] + off[y], xerr=[(b - bci[:, 0])[sel], (bci[:, 1] - b)[sel]], fmt="o",
                         ms=4, capsize=2, color=col, label=OUT_LABEL[y])
        axes[1].errorbar(cv, ypos + off[y], xerr=[cv - cci[:, 0], cci[:, 1] - cv], fmt="o", ms=4, capsize=2,
                         color=col, label=OUT_LABEL[y])
    naive = " / ".join(f"{summ[y]['naive']['bias_pct']:+.0f}%" for y in OUTCOMES)
    axes[0].text(lim * 0.97, keys.index("naive"), f"off-scale: {naive}", ha="right", va="center",
                 fontsize=9, color=io.TEXT_MUTED)
    axes[0].set_xlim(-lim, lim)
    axes[0].axvline(0, color=io.REFERENCE, lw=1)
    axes[1].axvline(0.95, color=io.REFERENCE, lw=1, ls="--")
    axes[0].set_yticks(ypos, [f"{EST[k][0]} [{EST[k][1]}]" for k in keys])
    axes[0].set(xlabel="Bias, % of RCT benchmark (Monte Carlo 95% CI)", title="Bias across replicates")
    axes[1].set(xlabel="Coverage of the RCT benchmark by nominal 95% CIs (Wilson 95% CI)",
                title="Confidence-interval coverage (dashed = 0.95)")
    axes[1].legend(loc="center left")
    io.save_fig(fig, "observational_simulation_metrics")


# ---------- report ----------
def _fmt(v: float, y: str, d: int | None = None) -> str:
    d = (2 if y != "spend" else 3) if d is None else d
    return f"{v * SCALE[y]:.{d}f}"


def _ci(lo: float, hi: float, y: str, d: int | None = None) -> str:
    return f"[{_fmt(lo, y, d)}, {_fmt(hi, y, d)}]"


def _pct(r: dict) -> str:
    return f"{r['bias_pct']:+.1f}% [{r['bias_pct_ci'][0]:+.1f}, {r['bias_pct_ci'][1]:+.1f}]"


def _nobias(r: dict) -> bool:
    return r["bias_ci"][0] <= 0 <= r["bias_ci"][1]


def _valid(r: dict) -> bool:
    """No detectable bias and not significantly under-covering."""
    return _nobias(r) and r["coverage_ci"][1] >= 0.95


def _headline(y: str, bench: dict, summ: dict, errs: dict, single: dict, boot: np.ndarray) -> dict:
    s = summ[y]
    keep = ("bias", "bias_ci", "bias_pct", "bias_pct_ci", "rmse", "rmse_ci", "coverage", "coverage_ci")
    by_rmse = sorted(FEASIBLE, key=lambda k: s[k]["rmse"])
    valid = [k for k in by_rmse if _valid(s[k])]
    best = valid[0] if valid else by_rmse[0]
    lowest = by_rmse[0]
    nxt = next(k for k in by_rmse if k != best)
    ne, nse = single["est"][y]["naive"]
    truth = bench[y]["ate"]["estimate"]
    return {
        "rct_ate": {k: bench[y]["ate"][k] for k in ("estimate", "se", "ci_low", "ci_high")},
        "naive_single_sample": {"estimate": ne, "ci_low": ne - Z * nse, "ci_high": ne + Z * nse,
                                "bias_pct_vs_rct": 100 * (ne - truth) / truth},
        "naive_simulation": {k: s["naive"][k] for k in keep},
        "best_estimator": {"rule": "lowest RMSE among estimators with bias CI containing 0 and coverage not "
                                   "significantly below 0.95", "key": best, "label": s[best]["label"],
                           **{k: s[best][k] for k in keep}},
        "lowest_rmse_overall": {"key": lowest, "label": s[lowest]["label"], **{k: s[lowest][k] for k in keep}},
        "next_best_by_rmse": {"key": nxt, "label": s[nxt]["label"], "rmse": s[nxt]["rmse"],
                              "rmse_ratio_best_vs_next": _rmse_ratio(errs[y][best], errs[y][nxt], boot)},
        "valid_estimators": [s[k]["label"] + f" [{s[k]['estimand']}]" for k in valid],
        "significant_undercoverage": [s[k]["label"] + f" [{s[k]['estimand']}]" for k in FEASIBLE
                                      if s[k]["coverage_ci"][1] < 0.95],
        "significant_bias": [s[k]["label"] + f" [{s[k]['estimand']}]" for k in FEASIBLE
                             if not s[k]["bias_ci"][0] <= 0 <= s[k]["bias_ci"][1]],
        "misspecified_ps_check": {k: {m: s[k][m] for m in ("bias_pct", "bias_pct_ci", "rmse", "coverage", "coverage_ci")}
                                  for k in MISSPEC_ROWS},
        "signal_ok": bool(bench[y]["ate"]["ci_low"] > 0 and s["aipw_gbm_gbm"]["median_ci_halfwidth"] < abs(truth)),
    }


MISSPEC_ROWS = ["ipw_logit", "strat_logit", "match_logit", "aipw_logit_glm", "aipw_logit_gbm", "ipw_gbm", "aipw_gbm_gbm"]


def _md(res: dict) -> str:
    b, s, d, dg, h = res["benchmarks"], res["simulation"], res["design"], res["diagnostics"], res["headline_by_outcome"]
    one = res["single_replicate"]
    R = res["n_replicates"]
    y = "visit"
    hv = h["visit"]
    L: list[str] = []
    L.append("## Observational methods vs. an experimental ground truth")
    L.append("")
    L.append(
        "Marketing data are usually observational: marketers email the customers they already expect to buy, so a "
        "naive emailed-vs-not comparison is confounded. Because Hillstrom is randomized, we can manufacture that "
        "selection bias from real customers and real outcomes and check which estimators recover the experimental "
        "answer (a LaLonde-style design). A targeting rule e(X) (below) plays the marketer. From the RCT, emailed "
        f"customers are kept with probability c*e(X)/p and un-emailed ones with c*(1-e(X))/(1-p), where p = "
        f"{d['p_treated_rct']:.4f} is the RCT's emailed share and c = {d['c']:.3f} is the largest value keeping both "
        "probabilities <= 1. Retention is then c for every X, so the observational sample has the full population's "
        "customer mix (the estimand is the population ATE) while its true propensity is exactly e(X); selection uses "
        "only (X, T), so there is no unmeasured confounding by construction. Ground truth is the RCT difference in "
        f"means on all 64,000 customers. We analyse one observational sample in detail (n = {one['n']:,}), then repeat "
        f"the whole pipeline on {R} simulated samples (each: bootstrap the 64,000 customers, then apply the targeting "
        "draw) and score each estimator's bias, RMSE and 95% CI coverage of the benchmark. Nuisance models are "
        f"cross-fitted ({E.K_FOLDS} folds): a main-effects logistic regression, which is misspecified here by design "
        "(the rule has a VIP interaction, a win-back arm that is non-monotone in recency, a $300 threshold and a "
        "bounded link), and LightGBM. Matching is 1:1 nearest-neighbour on the logit PS with replacement and a 0.2-SD "
        "caliper (Abadie-Imbens variance); it is run for the ATE and, separately, for the ATT, which is scored "
        "against the RCT's ATT for the observational population (the e(X)-weighted RCT difference in means).")
    L.append("")
    L.append("**Targeting rule** (logit score, mapped to e = 0.1 + 0.8 * sigmoid(score))")
    L.append("")
    L.append("| Term | Coefficient | Share of customers |")
    L.append("|---|---|---|")
    for k, v in RULE.items():
        share = "" if k == "intercept" else f"{100 * d['term_share'][k]:.1f}%"
        L.append(f"| {RULE_DESC[k]} | {v:+.1f} | {share} |")
    L.append("")
    L.append(f"True propensity ranges {d['e_min']:.3f}-{d['e_max']:.3f}; the marketer emails "
             f"{100 * d['e_mean']:.1f}% of customers. Observational samples average "
             f"{dg['simulation']['n']['mean']:,.0f} customers ({100 * dg['simulation']['treated_share']['mean']:.1f}% emailed).")
    L.append("")
    L.append("**Ground truth from the RCT (all 64,000 customers)**")
    L.append("")
    L.append("| Outcome | RCT ATE [95% CI] | Covariate-adjusted ATE (Lin) | ATT of observational population [95% CI] | ATT - ATE [95% CI] |")
    L.append("|---|---|---|---|---|")
    for o in OUTCOMES:
        bo = b[o]
        L.append(f"| {OUT_LABEL[o]} ({UNIT[o]}) | {_fmt(bo['ate']['estimate'], o)} {_ci(bo['ate']['ci_low'], bo['ate']['ci_high'], o)} "
                 f"| {_fmt(bo['lin_adjusted_ate']['estimate'], o)} | {_fmt(bo['att']['estimate'], o)} "
                 f"{_ci(bo['att']['ci_low'], bo['att']['ci_high'], o)} | {_fmt(bo['att_minus_ate']['estimate'], o)} "
                 f"{_ci(bo['att_minus_ate']['ci_low'], bo['att_minus_ate']['ci_high'], o)} |")
    L.append("")
    L.append("### Propensity model diagnostics")
    L.append("")
    L.append(f"Single sample (n = {one['n']:,}); bracketed ranges are the 2.5-97.5th percentiles across the {R} simulated samples.")
    L.append("")
    L.append("| | True e(X) | Logistic PS | GBM PS |")
    L.append("|---|---|---|---|")
    sd, simd = dg["single"], dg["simulation"]

    def row(label: str, key: str, fmt: str = ".3f", models: tuple = ("oracle", "logit", "gbm")) -> str:
        cells = ["-" if m not in models else
                 f"{sd[m][key]:{fmt}} [{simd[m][key]['p2_5']:{fmt}}, {simd[m][key]['p97_5']:{fmt}}]"
                 for m in ("oracle", "logit", "gbm")]
        return f"| {label} | " + " | ".join(cells) + " |"
    L.append(row("AUC (out-of-fold)", "auc"))
    L.append(row("RMSE vs true propensity", "rmse_vs_true", models=("logit", "gbm")))
    L.append(row("Share of PS outside [0.05, 0.95]", "share_outside_05_95"))
    L.append(row("Max abs SMD after IPW, model features", "max_smd_main"))
    L.append(row("Max abs SMD after IPW, incl. rule terms", "max_smd"))
    L.append(row("Effective sample size, emailed", "ess_treated", ",.0f"))
    L.append(row("Effective sample size, not emailed", "ess_control", ",.0f"))
    cal = dg["calibration"]
    L.append(f"| Calibration slope [95% CI] | - | {cal['logit']['slope']:.2f} {_ci(*cal['logit']['slope_ci'], 'spend', 2)} "
             f"| {cal['gbm']['slope']:.2f} {_ci(*cal['gbm']['slope_ci'], 'spend', 2)} |")
    L.append(f"| Expected calibration error / Brier | - | {cal['logit']['ece']:.3f} / {cal['logit']['brier']:.4f} "
             f"| {cal['gbm']['ece']:.3f} / {cal['gbm']['brier']:.4f} |")
    L.append("")
    uw = sd["unweighted"]
    L.append(f"Before weighting the max |SMD| is {uw['max_smd_main']:.3f} on model features and {uw['max_smd']:.3f} "
             f"including the rule terms (simulation range {simd['unweighted']['max_smd']['p2_5']:.3f}-"
             f"{simd['unweighted']['max_smd']['p97_5']:.3f}). Logistic-PS weighting leaves the rule's interaction terms "
             f"imbalanced (max |SMD| {sd['logit']['max_smd']:.3f}, above the 0.1 threshold) while GBM-PS weighting "
             f"balances everything ({sd['gbm']['max_smd']:.3f}). No trimming was needed: the true propensity is bounded "
             f"in [0.1, 0.9] by design and estimated scores are only clipped to [{E.PS_CLIP[0]}, {E.PS_CLIP[1]}] as a "
             f"numerical guard (no units dropped, estimand unchanged). The matching caliper dropped at most "
             f"{100 * max(v['p97_5'] for v in simd['matching'].values()):.2f}% of units (97.5th percentile over runs).")
    L.append("")
    L.append("![Propensity overlap](results/figures/observational_overlap.png)")
    L.append("")
    L.append("![Propensity calibration](results/figures/observational_calibration.png)")
    L.append("")
    L.append("![Love plot](results/figures/observational_love_plot.png)")
    L.append("")
    L.append(f"### Simulation results: site visit ({R} observational samples)")
    L.append("")
    L.append(f"Benchmark: RCT ATE {_fmt(b[y]['ate']['estimate'], y)} pp for ATE rows; RCT ATT "
             f"{_fmt(b[y]['att']['estimate'], y)} pp for ATT rows. Bias CIs are Monte Carlo 95% CIs; RMSE and SE-ratio "
             "CIs bootstrap the replicates; coverage CIs are Wilson. Bias/SD is bias in units of the estimator's "
             "sampling SD; the 10x-n column projects coverage if the sample were ten times larger and the bias did not "
             "shrink (honest SEs assumed). SE ratio = mean reported SE / empirical SD (1 = honest).")
    L.append("")
    L.append("| Estimator | Estimand | Bias, pp [95% CI] (% of truth) | Bias/SD | RMSE, pp [95% CI] | Coverage [95% CI] | Coverage at 10x n | SE ratio [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in EST:
        r = s[y][k]
        L.append(f"| {r['label']} | {r['estimand']} | {_fmt(r['bias'], y, 3)} {_ci(*r['bias_ci'], y, 3)} ({r['bias_pct']:+.1f}%) | "
                 f"{r['bias_over_sd']:+.2f} | {_fmt(r['rmse'], y, 3)} {_ci(*r['rmse_ci'], y, 3)} | "
                 f"{r['coverage']:.3f} [{r['coverage_ci'][0]:.3f}, {r['coverage_ci'][1]:.3f}] | "
                 f"{r['projected_coverage_10x_n']:.2f} | {r['se_ratio']:.2f} [{r['se_ratio_ci'][0]:.2f}, {r['se_ratio_ci'][1]:.2f}] |")
    L.append("")
    L.append(f"![Estimates vs RCT benchmark](results/figures/observational_estimates_{y}.png)")
    L.append("")
    L.append("### Conversion and spend")
    L.append("")
    L.append("| Estimator | Estimand | Conversion bias % [95% CI] | Conversion RMSE (pp) | Conversion coverage | Spend bias % [95% CI] | Spend RMSE ($) | Spend coverage |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in EST:
        c_, sp = s["conversion"][k], s["spend"][k]
        L.append(f"| {c_['label']} | {c_['estimand']} | {_pct(c_)} | {_fmt(c_['rmse'], 'conversion', 3)} | "
                 f"{c_['coverage']:.3f} | {_pct(sp)} | {_fmt(sp['rmse'], 'spend', 3)} | {sp['coverage']:.3f} |")
    L.append("")
    L.append("![Bias and coverage](results/figures/observational_simulation_metrics.png)")
    L.append("")
    L.append("### Does double robustness rescue a misspecified propensity model?")
    L.append("")
    L.append("Bias as % of the RCT benchmark [Monte Carlo 95% CI].")
    L.append("")
    L.append("| Estimator | " + " | ".join(OUT_LABEL[o] for o in OUTCOMES) + " |")
    L.append("|---|" + "---|" * len(OUTCOMES))
    for k in MISSPEC_ROWS:
        L.append(f"| {EST[k][0]} | " + " | ".join(_pct(s[o][k]) for o in OUTCOMES) + " |")
    L.append("")
    L.append("### Findings")
    L.append("")
    nv, ns = hv["naive_simulation"], hv["naive_single_sample"]
    L.append(f"- **Naive comparison is badly confounded.** In the single sample the naive visit lift is "
             f"{_fmt(ns['estimate'], y)} pp {_ci(ns['ci_low'], ns['ci_high'], y)} vs the RCT's "
             f"{_fmt(b[y]['ate']['estimate'], y)} pp {_ci(b[y]['ate']['ci_low'], b[y]['ate']['ci_high'], y)}; across "
             f"samples its bias is {_pct(nv)} of the true effect and its CI covers the truth "
             f"{100 * nv['coverage']:.1f}% of the time. For spend it is {_pct(s['spend']['naive'])}.")
    be, lo, nx = hv["best_estimator"], hv["lowest_rmse_overall"], hv["next_best_by_rmse"]
    rr = nx["rmse_ratio_best_vs_next"]
    L.append(f"- **Best estimator (visit; lowest RMSE among those with no detectable bias and nominal coverage): "
             f"{be['label']}** - bias {_pct(be)}, RMSE {_fmt(be['rmse'], y, 3)} pp {_ci(*be['rmse_ci'], y, 3)}, "
             f"coverage {be['coverage']:.3f} [{be['coverage_ci'][0]:.3f}, {be['coverage_ci'][1]:.3f}]. "
             + ("It is also the lowest-RMSE estimator overall. " if be["key"] == lo["key"] else
                f"The lowest RMSE overall is {lo['label']} ({_fmt(lo['rmse'], y, 3)} pp, bias {_pct(lo)}). ")
             + f"Paired RMSE ratio vs the next estimator by RMSE ({nx['label']}): {rr['ratio']:.3f} "
             f"[{rr['ci'][0]:.3f}, {rr['ci'][1]:.3f}], i.e. "
             + ("statistically indistinguishable, so the ranking within the top group is not robust."
                if rr["ci"][0] <= 1 <= rr["ci"][1] else "a detectable precision advantage.")
             + f" Estimators passing both checks: {', '.join(hv['valid_estimators'])}.")
    und = hv["significant_undercoverage"]
    L.append("- **Significant under-coverage (visit; Wilson upper bound < 0.95):** " + (", ".join(und) if und else "none") + ".")
    unbiased_all = [EST[k][0] + f" [{EST[k][1]}]" for k in FEASIBLE
                    if all(s[o][k]["bias_ci"][0] <= 0 <= s[o][k]["bias_ci"][1] for o in OUTCOMES)]
    L.append("- **No detectable bias on any of the three outcomes:** " + (", ".join(unbiased_all) or "none") + ".")
    il, ag = s[y]["ipw_logit"], s[y]["aipw_logit_gbm"]
    cells = [(k, o) for k in ("ipw_logit", "strat_logit", "match_logit", "att_match_logit") for o in OUTCOMES]
    n_biased = sum(not s[o][k]["bias_ci"][0] <= 0 <= s[o][k]["bias_ci"][1] for k, o in cells)
    L.append(f"- **What the misspecified logistic PS costs.** Estimators relying on it alone (IPW, stratification, "
             f"ATE and ATT matching) are significantly biased in {n_biased} of {len(cells)} estimator-outcome cells "
             f"(IPW: visit {il['bias_pct']:+.1f}%, conversion {s['conversion']['ipw_logit']['bias_pct']:+.1f}%, spend "
             f"{s['spend']['ipw_logit']['bias_pct']:+.1f}%), but at n ~ {dg['simulation']['n']['mean']:,.0f} the bias is "
             f"only {abs(il['bias_over_sd']):.2f} SD for visit, so "
             + ("CIs still cover at about the nominal rate" if il["coverage_ci"][1] >= 0.95 else "CIs already under-cover")
             + f" ({il['coverage']:.3f}). The danger grows with scale: with 10x the customers the same bias would cut "
             f"visit coverage to about {il['projected_coverage_10x_n']:.2f}. IPW with the GBM PS "
             + (f"removes the bias (visit {_pct(s[y]['ipw_gbm'])})." if _nobias(s[y]["ipw_gbm"]) else
                f"shrinks the bias but does not remove it (visit {_pct(s[y]['ipw_gbm'])}): plug-in IPW with an ML "
                "propensity keeps some regularization bias, which AIPW's outcome-model correction cancels."))
    dr_ok = all(_nobias(s[o]["aipw_logit_gbm"]) for o in OUTCOMES)
    glm_ok = [OUT_LABEL[o].lower() for o in OUTCOMES if _nobias(s[o]["aipw_logit_glm"])]
    glm_bad = [OUT_LABEL[o].lower() for o in OUTCOMES if not _nobias(s[o]["aipw_logit_glm"])]
    L.append(f"- **Double robustness {'rescues' if dr_ok else 'only partly rescues'} the misspecified PS when the "
             f"outcome model is flexible.** AIPW keeping the logistic PS but adding a GBM outcome model has bias "
             f"{_pct(ag)} (visit), {_pct(s['conversion']['aipw_logit_gbm'])} (conversion), "
             f"{_pct(s['spend']['aipw_logit_gbm'])} (spend). With a GLM outcome model instead (both nuisances "
             f"parametric) there is no detectable bias for {' and '.join(glm_ok) or 'no outcome'}"
             + (f", but it remains for {' and '.join(glm_bad)} ("
                + ", ".join(_pct(s[o]["aipw_logit_glm"]) for o in OUTCOMES if not _nobias(s[o]["aipw_logit_glm"]))
                + "): with both nuisance models misspecified, double robustness carries no guarantee." if glm_bad else "."))
    st, sl = s[y]["strat_gbm"], s[y]["strat_logit"]
    both = not _nobias(st) and not _nobias(sl)
    L.append(f"- **Stratification {'leaves residual confounding regardless of the PS model' if both else 'on quintiles'}** "
             f"(visit bias {_pct(sl)} logistic, {_pct(st)} GBM)"
             + (f": five strata are too coarse for a propensity spanning {d['e_min']:.2f}-{d['e_max']:.2f}." if both else "."))
    hc = h["conversion"]
    if hc["signal_ok"]:
        bc = hc["best_estimator"]
        L.append(f"- **Conversion (about 1% base rate):** RCT ATE {_fmt(b['conversion']['ate']['estimate'], 'conversion', 3)} pp "
                 f"{_ci(b['conversion']['ate']['ci_low'], b['conversion']['ate']['ci_high'], 'conversion', 3)}; naive "
                 f"bias {_pct(hc['naive_simulation'])}, coverage {hc['naive_simulation']['coverage']:.3f}; best by the same "
                 f"rule: {bc['label']} (bias {_pct(bc)}, RMSE {_fmt(bc['rmse'], 'conversion', 3)} pp, coverage "
                 f"{bc['coverage']:.3f}). Even the best estimator's per-sample RMSE is "
                 f"{100 * bc['rmse'] / b['conversion']['ate']['estimate']:.0f}% of the effect, so single observational "
                 "conversion estimates are imprecise.")
    else:
        L.append(f"- **Conversion is too noisy for a headline:** the typical AIPW 95% CI half-width "
                 f"({_fmt(s['conversion']['aipw_gbm_gbm']['median_ci_halfwidth'], 'conversion', 3)} pp) is not smaller "
                 f"than the effect itself ({_fmt(b['conversion']['ate']['estimate'], 'conversion', 3)} pp).")
    L.append("")
    L.append("**Caveats**")
    L.append("")
    L.append("- Semi-synthetic by design: the selection mechanism is known and depends only on observed X, so "
             "unconfoundedness holds exactly. Real targeting also uses signals the analyst never sees (e.g. browsing), "
             "which no estimator here can fix; this study isolates *modelling* error, not hidden confounding.")
    L.append(f"- The simulation's truth is the full-sample RCT estimate (bootstrap world; the oracle IPW's bias of "
             f"{_pct(s[y]['ipw_oracle'])} confirms the benchmark is the right target). The RCT's own sampling error "
             f"(visit CI half-width {_fmt(Z * b[y]['ate']['se'], y)} pp) is not in the bias/coverage numbers; the "
             f"covariate-adjusted RCT estimate differs from the difference in means by "
             f"{_fmt(b[y]['lin_adjusted_ate']['estimate'] - b[y]['ate']['estimate'], y, 3)} pp.")
    L.append(f"- Bootstrap duplicates make ML nuisances slightly noisier than in a fresh sample of the same size (GBM PS "
             f"RMSE vs truth: {sd['gbm']['rmse_vs_true']:.3f} in the real single sample vs "
             f"{simd['gbm']['rmse_vs_true']['p2_5']:.3f}-{simd['gbm']['rmse_vs_true']['p97_5']:.3f} across replicates), "
             "so the simulation is if anything conservative for GBM-based estimators.")
    L.append("- Results are conditional on one targeting rule. Hillstrom outcomes are close to additive in these "
             "features, so the interactions the logistic PS omits carry little outcome signal; a rule keyed to "
             "outcome-relevant nonlinearities would make misspecification costlier.")
    L.append("- Standard errors treat the estimated PS as known: conservative for ATE IPW/matching with a parametric PS, "
             "no guarantee for the GBM PS or ATT matching. OLS with a treatment dummy targets a variance-weighted "
             "average of effects, which equals the ATE only if effects are homogeneous.")
    L.append("- 'Emailed' pools two creatives (Mens/Womens, 50/50 at every X), identically in the RCT and the "
             "observational samples. GBM hyperparameters were fixed up front, not tuned on the benchmark.")
    ck = res["se_check"]
    L.append(f"- Monte Carlo precision: {R} replicates pin coverage to about +/-{100 * Z * np.sqrt(0.95 * 0.05 / R):.1f} pp; "
             "conversion is the least precise outcome. SE ratios share one set of replicates, so they drift together: "
             f"the naive estimator's Welch SE (exact in theory) shows a ratio of {s[y]['naive']['se_ratio']:.2f} here but "
             f"{ck['ratio']:.3f} [{ck['ci'][0]:.3f}, {ck['ci'][1]:.3f}] over {ck['n']:,} cheap replicates with no model "
             "fitting, so SE ratios a few percent above 1 are Monte Carlo noise, not miscalibration.")
    return "\n".join(L) + "\n"


def simulate(n_replicates: int = N_REPLICATES) -> dict:
    """Design, RCT benchmarks, the detailed single sample, and the replicate simulation."""
    t0 = time.time()
    df = data.load()
    t = df["treated"].to_numpy()
    e = marketer_propensity(df)
    q, c, p = retention(e, t)
    terms = rule_terms(df)
    bal = pd.concat([df[data.FEATURES], terms[RULE_TERMS]], axis=1)
    A = {"t": t, "e": e, "q": q, "X_glm": df[X_GLM].to_numpy(float), "X_gbm": df[data.FEATURES].to_numpy(float),
         "X_bal": bal[BALANCE].to_numpy(float), **{y: df[y].to_numpy(float) for y in OUTCOMES}}
    bench = benchmarks(df, e)
    design = {"rule": RULE, "high_value_threshold": HIGH_VALUE, "e_floor": E_FLOOR, "e_span": E_SPAN,
              "e_min": float(e.min()), "e_max": float(e.max()), "e_mean": float(e.mean()), "c": c,
              "p_treated_rct": p, "expected_n": float(c * len(df)),
              "term_share": {k: float(terms[k].mean()) for k in RULE if k != "intercept"}}
    print(f"design: e in [{e.min():.3f}, {e.max():.3f}], c={c:.3f}, expected n={c * len(df):,.0f}", flush=True)

    se_check = _naive_se_check(A)
    single = _replicate(0, A, bootstrap=False, detail=True)  # the real customers, no bootstrap
    arr = single.pop("arrays")
    cal = {m: calibration(arr["t"], arr[f"ps_{m}"]) for m in ("logit", "gbm")}
    print(f"single replicate done ({time.time() - t0:.0f}s); simulating {n_replicates} replicates", flush=True)

    with parallel_config(backend="loky", inner_max_num_threads=1):
        reps = Parallel(n_jobs=N_JOBS)(delayed(_replicate)(r, A, True) for r in range(n_replicates))
    print(f"simulation done ({time.time() - t0:.0f}s)", flush=True)
    return {"design": design, "bench": bench, "single": single, "arr": arr, "cal": cal, "reps": reps,
            "se_check": se_check}


def report(sim: dict, runtime: float) -> None:
    design, bench, single, arr, cal, reps = (sim[k] for k in ("design", "bench", "single", "arr", "cal", "reps"))
    boot = np.random.default_rng(C.SEED).integers(0, len(reps), (C.N_BOOTSTRAP, len(reps)))
    summ, errs = summarize(reps, bench, boot)
    dsum = _diag_summary(reps)
    dsum["matching"] = dsum.pop("matching_drop")
    diags = {"single": single["diag"], "simulation": dsum, "calibration": cal}
    heads = {y: _headline(y, bench, summ, errs, single, boot) for y in OUTCOMES}
    hv = heads["visit"]
    headline = {
        "rct_ate_visit": hv["rct_ate"],
        "naive_visit_single_sample": hv["naive_single_sample"],
        "naive_visit_bias_pct": {"estimate": hv["naive_simulation"]["bias_pct"], "ci": hv["naive_simulation"]["bias_pct_ci"],
                                 "coverage": hv["naive_simulation"]["coverage"]},
        "best_estimator_visit": hv["best_estimator"],
        "lowest_rmse_visit": hv["lowest_rmse_overall"],
        "best_vs_next_rmse_visit": hv["next_best_by_rmse"],
        "max_abs_smd": {
            "unweighted": {"single": single["diag"]["unweighted"]["max_smd"],
                           "sim_range": [dsum["unweighted"]["max_smd"]["p2_5"], dsum["unweighted"]["max_smd"]["p97_5"]]},
            **{f"ipw_{m}": {"single": single["diag"][m]["max_smd"],
                            "sim_range": [dsum[m]["max_smd"]["p2_5"], dsum[m]["max_smd"]["p97_5"]]} for m in ("logit", "gbm")}},
        "visit_significant_undercoverage": hv["significant_undercoverage"],
        "visit_significant_bias": hv["significant_bias"],
        "misspecified_ps_check": {y: heads[y]["misspecified_ps_check"] for y in OUTCOMES},
        "conversion": ({k: heads["conversion"][k] for k in ("rct_ate", "naive_simulation", "best_estimator", "signal_ok")}
                       if heads["conversion"]["signal_ok"] else {"signal_ok": False}),
        "n_replicates": len(reps), "runtime_seconds": runtime,
    }

    _fig_overlap(arr)
    _fig_calibration(cal)
    _fig_love(arr["smd"])
    _fig_estimates(single, reps, bench, "visit")
    _fig_metrics(summ)

    res = {"headline": headline, "design": design, "benchmarks": bench, "n_replicates": len(reps),
           "single_replicate": {"n": single["diag"]["n"], "estimates": single["est"]},
           "simulation": summ, "diagnostics": diags, "headline_by_outcome": heads,
           "se_check": sim["se_check"], "runtime_seconds": runtime}
    io.save_json(res, "observational")
    (C.RESULTS / "observational.md").write_text(_md(res))


def run() -> None:
    io.style()
    t0 = time.time()
    sim = simulate()
    report(sim, time.time() - t0)
    print(f"observational: {time.time() - t0:.0f}s", flush=True)
