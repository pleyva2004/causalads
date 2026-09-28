"""Stage 1: EDA + randomized-experiment readout (balance, ATEs, power, incrementality, segments)."""
from __future__ import annotations

import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as sps
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests

from causalads import config as CFG
from causalads import data, io
from causalads.stats import bootstrap_ci, diff_in_means, smd  # noqa: F401  (bootstrap_ci kept for parity)

OUTS = ["visit", "conversion", "spend"]
LABEL = {"visit": "Site visit", "conversion": "Conversion", "spend": "Spend"}
# (treated arm(s), control arm). Arm ids: 0 none, 1 mens, 2 womens.
COMPS: dict[str, tuple] = {
    "Mens vs No E-Mail": ((1,), (0,)),
    "Womens vs No E-Mail": ((2,), (0,)),
    "Any E-Mail vs No E-Mail": ((1, 2), (0,)),
    "Mens vs Womens": ((1,), (2,)),
}
Z = sps.norm.ppf(0.975)
ADJ = ["recency", "log_history", "history_at_floor", "mens", "womens", "newbie",
       "zip_urban", "zip_suburban", "ch_phone", "ch_web"]
SEG_COLS = {"channel": "channel", "zip_code": "zip_code", "newbie": "newbie",
            "past mens buyer": "mens", "past womens buyer": "womens", "history bucket": "history_segment"}
N_BOOT = CFG.N_BOOTSTRAP


# ---------- estimators ----------
def bin_effect(x1: int, n1: int, x0: int, n0: int) -> dict:
    """Two-proportion difference (unpooled Wald CI, pooled-z p) and relative lift (delta method on log ratio)."""
    p1, p0 = x1 / n1, x0 / n0
    d = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    pp = (x1 + x0) / (n1 + n0)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    out = {"estimate": d, "se": se, "ci_low": d - Z * se, "ci_high": d + Z * se,
           "p_value": float(2 * sps.norm.sf(abs(d) / se0)) if se0 > 0 else 1.0,
           "mean_treated": p1, "mean_control": p0, "n_treated": n1, "n_control": n0}
    if x1 > 0 and x0 > 0:
        lse = np.sqrt((1 - p1) / (n1 * p1) + (1 - p0) / (n0 * p0))
        lr = np.log(p1 / p0)
        out.update(rel_lift=p1 / p0 - 1, rel_ci_low=np.exp(lr - Z * lse) - 1, rel_ci_high=np.exp(lr + Z * lse) - 1)
    else:
        out.update(rel_lift=np.nan, rel_ci_low=np.nan, rel_ci_high=np.nan)
    return out


def spend_effect(y1: np.ndarray, y0: np.ndarray, rng: np.random.Generator) -> dict:
    """Welch t (primary), Mann-Whitney, and percentile bootstrap for the difference and relative lift."""
    t = np.r_[np.ones(len(y1)), np.zeros(len(y0))]
    out = diff_in_means(np.r_[y1, y0], t)
    out["p_welch"] = out["p_value"] = float(sps.ttest_ind(y1, y0, equal_var=False).pvalue)
    out["p_mannwhitney"] = float(sps.mannwhitneyu(y1, y0, alternative="two-sided").pvalue)
    d, r = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        m1 = y1[rng.integers(0, len(y1), len(y1))].mean()
        m0 = y0[rng.integers(0, len(y0), len(y0))].mean()
        d[b], r[b] = m1 - m0, m1 / m0 - 1
    out.update(boot_ci_low=float(np.quantile(d, .025)), boot_ci_high=float(np.quantile(d, .975)),
               rel_lift=out["mean_treated"] / out["mean_control"] - 1,
               rel_ci_low=float(np.quantile(r, .025)), rel_ci_high=float(np.quantile(r, .975)))
    return out


def lin_adjusted(df: pd.DataFrame, treated: np.ndarray, y: str) -> dict:
    """Lin (2013): y ~ 1 + T + Xc + T*Xc with X centred on the analysis-sample mean; HC2 robust SEs."""
    X = df[ADJ].to_numpy(float)
    Xc = X - X.mean(0)
    T = treated.astype(float)[:, None]
    yy = df[y].to_numpy(float)
    adj = sm.OLS(yy, np.c_[np.ones(len(yy)), T, Xc, T * Xc]).fit(cov_type="HC2")
    raw = sm.OLS(yy, np.c_[np.ones(len(yy)), T]).fit(cov_type="HC2")
    return {"adj_estimate": float(adj.params[1]), "adj_se": float(adj.bse[1]),
            "adj_ci_low": float(adj.conf_int()[1, 0]), "adj_ci_high": float(adj.conf_int()[1, 1]),
            "unadj_estimate": float(raw.params[1]), "unadj_se": float(raw.bse[1]),
            "se_reduction_pct": float(100 * (1 - adj.bse[1] / raw.bse[1]))}


def mde_prop(p0: float, n1: int, n0: int, alpha=0.05, power=0.8) -> float:
    k = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    f = lambda d: d - k * np.sqrt(p0 * (1 - p0) / n0 + (p0 + d) * (1 - p0 - d) / n1)
    return float(brentq(f, 1e-7, 0.5))


def n_per_arm(p0: float, p1: float, alpha=0.05, power=0.8) -> float:
    k = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(k**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2)


# ---------- SQL readouts ----------
def sql_readouts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.register("customers", df)
    arms = con.execute((CFG.SQL_DIR / "experiment_arm_readout.sql").read_text()).df()
    seg = con.execute((CFG.SQL_DIR / "experiment_segment_readout.sql").read_text()).df()
    con.close()
    return arms, seg


# ---------- balance ----------
def balance(df: pd.DataFrame) -> dict:
    cats = {"channel": "channel", "zip_code": "zip_code", "newbie": "newbie", "past mens buyer": "mens",
            "past womens buyer": "womens", "history bucket": "history_segment"}
    rows = []
    for name, col in cats.items():
        chi2, p, dof, _ = sps.chi2_contingency(pd.crosstab(df[col], df["arm"]))
        rows.append({"variable": name, "test": "chi-square", "statistic": chi2, "dof": dof, "p_raw": p})
    for col in ["recency", "history"]:
        g = [df.loc[df.arm == a, col].to_numpy() for a in (0, 1, 2)]
        kw = sps.kruskal(*g)
        rows.append({"variable": col, "test": "Kruskal-Wallis", "statistic": kw.statistic, "dof": 2, "p_raw": kw.pvalue,
                     "p_anova": float(sps.f_oneway(*g).pvalue)})
    tab = pd.DataFrame(rows)
    tab["p_holm"] = multipletests(tab["p_raw"], method="holm")[1]

    # Omnibus: can the 10 covariates jointly predict arm? (LR test of multinomial logit vs intercept-only)
    X = sm.add_constant(df[ADJ].astype(float))
    full = sm.MNLogit(df["arm"], X).fit(disp=0, maxiter=200)
    null = sm.MNLogit(df["arm"], X[["const"]]).fit(disp=0)
    lr = 2 * (full.llf - null.llf)
    dof = 2 * len(ADJ)
    omni = {"lr_stat": float(lr), "dof": dof, "p_value": float(sps.chi2.sf(lr, dof))}

    cols = ["recency", "history", "log_history", "history_at_floor", "mens", "womens", "newbie",
            "zip_urban", "zip_suburban", "zip_rural", "ch_phone", "ch_web", "ch_multi"]
    smds = []
    for col in cols:
        r = {"variable": col}
        for a, nm in ((1, "Mens"), (2, "Womens")):
            sub = df[df.arm.isin([0, a])]
            r[f"smd_{nm.lower()}"] = smd(sub[col].to_numpy(float), (sub.arm == a).to_numpy())
        smds.append(r)
    return {"tests": tab, "omnibus": omni, "smd": pd.DataFrame(smds)}


# ---------- main effects ----------
def effects(df: pd.DataFrame, arms: pd.DataFrame) -> dict:
    rng = np.random.default_rng(CFG.SEED)
    cnt = arms.set_index("arm")
    def counts(ids: tuple, col: str) -> tuple[int, int]:
        rows = cnt.loc[[-1]] if ids == (1, 2) else cnt.loc[list(ids)]
        return int(rows[col].sum()), int(rows["n"].sum())

    res: dict = {}
    for name, (tr, co) in COMPS.items():
        m1, m0 = df.arm.isin(tr).to_numpy(), df.arm.isin(co).to_numpy()
        res[name] = {}
        for out in ("visit", "conversion"):
            col = {"visit": "visits", "conversion": "conversions"}[out]
            (x1, n1), (x0, n0) = counts(tr, col), counts(co, col)
            res[name][out] = bin_effect(x1, n1, x0, n0)
        res[name]["spend"] = spend_effect(df.loc[m1, "spend"].to_numpy(), df.loc[m0, "spend"].to_numpy(), rng)
        sub = df[m1 | m0]
        treated = m1[m1 | m0]
        for out in OUTS:
            res[name][out]["lin"] = lin_adjusted(sub, treated, out)
    # Holm over the 12 primary tests (binary: two-proportion z; spend: Welch t).
    keys = [(c, o) for c in COMPS for o in OUTS]
    adj = multipletests([res[c][o]["p_value"] for c, o in keys], method="holm")[1]
    for (c, o), p in zip(keys, adj):
        res[c][o]["p_holm"] = float(p)
    return res


def power_block(df: pd.DataFrame, eff: dict) -> dict:
    n = df.arm.value_counts().to_dict()
    p0 = {o: float(df.loc[df.arm == 0, o].mean()) for o in ("visit", "conversion")}
    s0 = float(df.loc[df.arm == 0, "spend"].std())
    out = {"arm_sizes": {int(k): int(v) for k, v in n.items()}, "alpha": 0.05, "power": 0.8, "rows": []}
    for name, (tr, co) in COMPS.items():
        n1, n0 = int(sum(n[a] for a in tr)), int(sum(n[a] for a in co))
        base = df[df.arm.isin(co)]
        for o in ("visit", "conversion"):
            p = float(base[o].mean())
            d = mde_prop(p, n1, n0)
            obs = eff[name][o]["estimate"]
            out["rows"].append({"comparison": name, "outcome": o, "baseline": p, "mde_abs": d, "mde_rel": d / p,
                                "observed": obs, "n_per_arm_for_observed": n_per_arm(p, p + obs) if obs > 0 else None})
        s = float(base["spend"].std())
        k = sps.norm.ppf(.975) + sps.norm.ppf(.8)
        d = k * np.sqrt(s ** 2 / n0 + df.loc[df.arm.isin(tr), "spend"].var() / n1)
        out["rows"].append({"comparison": name, "outcome": "spend", "baseline": float(base["spend"].mean()),
                            "mde_abs": float(d), "mde_rel": float(d / base["spend"].mean()),
                            "observed": eff[name]["spend"]["estimate"], "n_per_arm_for_observed": None})
    del p0, s0
    return out


def incrementality(df: pd.DataFrame, eff: dict) -> dict:
    out = {}
    for name in ("Mens vs No E-Mail", "Womens vs No E-Mail", "Any E-Mail vs No E-Mail"):
        n1 = eff[name]["visit"]["n_treated"]
        row = {"emails_sent": n1}
        for o in OUTS:
            e = eff[name][o]
            lo, hi = (e["boot_ci_low"], e["boot_ci_high"]) if o == "spend" else (e["ci_low"], e["ci_high"])
            row[o] = {"per_1000": [1000 * e["estimate"], 1000 * lo, 1000 * hi],
                      "total": [n1 * e["estimate"], n1 * lo, n1 * hi]}
        e = eff[name]["spend"]
        row["break_even_cost_per_email"] = [e["estimate"], e["ci_low"], e["ci_high"]]
        row["iroas"] = [e["estimate"] / CFG.COST_PER_CONTACT, e["ci_low"] / CFG.COST_PER_CONTACT, e["ci_high"] / CFG.COST_PER_CONTACT]
        out[name] = row
    return out


# ---------- segments ----------
def segments(df: pd.DataFrame, seg_sql: pd.DataFrame) -> dict:
    rows = []
    for (dim, lvl), g in seg_sql.groupby(["dimension", "level"], sort=True):
        g = g.set_index("arm")
        for arm, nm in ((1, "Mens"), (2, "Womens")):
            for o, col in (("visit", "visits"), ("conversion", "conversions")):
                e = bin_effect(int(g.loc[arm, col]), int(g.loc[arm, "n"]), int(g.loc[0, col]), int(g.loc[0, "n"]))
                rows.append({"dimension": dim, "level": lvl, "arm": nm, "outcome": o, "n_treated": int(g.loc[arm, "n"]),
                             "n_control": int(g.loc[0, "n"]), **{k: e[k] for k in
                             ("estimate", "ci_low", "ci_high", "p_value", "mean_treated", "mean_control", "rel_lift")}})
    tab = pd.DataFrame(rows)
    tab["p_holm"] = multipletests(tab["p_value"], method="holm")[1]

    inter = []
    for dim, col in SEG_COLS.items():
        for arm, nm in ((1, "Mens"), (2, "Womens")):
            sub = df[df.arm.isin([0, arm])].copy()
            sub["T"] = (sub.arm == arm).astype(int)
            sub["seg"] = sub[col].astype(str)
            for o in ("visit", "conversion"):
                r = smf.ols(f"{o} ~ T * C(seg)", sub).fit(cov_type="HC2")
                idx = [i for i, n in enumerate(r.params.index) if n.startswith("T:")]
                w = r.wald_test(np.eye(len(r.params))[idx], use_f=False, scalar=True)
                row = {"dimension": dim, "arm": nm, "outcome": o, "chi2": float(w.statistic), "dof": len(idx),
                       "p_value": float(w.pvalue)}
                if len(idx) == 1:  # binary segment: effect difference (level 2 minus level 1)
                    row.update(diff=float(r.params.iloc[idx[0]]), diff_ci_low=float(r.conf_int().iloc[idx[0], 0]),
                               diff_ci_high=float(r.conf_int().iloc[idx[0], 1]),
                               contrast=r.params.index[idx[0]].split("[T.")[-1].rstrip("]"))
                inter.append(row)
    it = pd.DataFrame(inter)
    it["p_holm"] = multipletests(it["p_value"], method="holm")[1]
    return {"effects": tab, "interactions": it}


# ---------- figures ----------
def fig_eda(df: pd.DataFrame) -> list[str]:
    S = io.SERIES
    names = []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(df.history, bins=np.geomspace(df.history.min(), df.history.max(), 50), color=S[0])
    ax[0].axvline(29.99, color=io.REFERENCE, ls="--", lw=1)
    ax[0].text(34, ax[0].get_ylim()[1] * .92, r"\$29.99 floor", fontsize=8, color=io.TEXT_MUTED)
    ax[0].set_xscale("log")
    ax[0].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[0].set(title="Past-year spend is right-skewed and floor-censored", xlabel="Customer history (USD, log scale)", ylabel="Customers")
    rc = df.recency.value_counts().sort_index()
    ax[1].bar(rc.index, rc.values, color=S[0])
    ax[1].set(title="Recency", xlabel="Months since last purchase", ylabel="Customers")
    ax[1].set_xticks(range(1, 13))
    sp = df.loc[df.conversion == 1, "spend"]
    ax[2].hist(sp, bins=np.geomspace(sp.min(), sp.max(), 40), color=S[0])
    ax[2].axvline(sp.median(), color=S[1], lw=1.5, label=f"median \\${sp.median():.0f}")
    ax[2].axvline(sp.mean(), color=S[2], lw=1.5, label=f"mean \\${sp.mean():.0f}")
    ax[2].set_xscale("log")
    ax[2].set(title=f"Spend among converters (n={len(sp):,}) is heavy-tailed", xlabel="Spend (USD, log scale)", ylabel="Converters")
    ax[2].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[2].legend()
    io.save_fig(fig, "experiment_distributions")
    names.append("experiment_distributions")

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (title, col) in zip(ax, [("Channel", "channel"), ("ZIP type", "zip_code"), ("Newbie", "newbie"), ("History bucket", "history_segment")]):
        sh = pd.crosstab(df[col], df["arm"], normalize="columns") * 100
        x = np.arange(len(sh))
        for j, arm in enumerate((0, 1, 2)):
            a.bar(x + (j - 1) * .27, sh[arm].values, .27, color=S[j], label=CFG.ARMS[arm])
        a.set_xticks(x)
        lab = ["existing" if v == 0 else "new" for v in sh.index] if col == "newbie" else [str(v).split(") ")[-1].replace(" - ", "-").replace("$", "") if col == "history_segment" else v for v in sh.index]
        a.set_xticklabels(lab, rotation=45 if col == "history_segment" else 0, ha="right" if col == "history_segment" else "center", fontsize=8)
        a.set(title=f"{title} mix by arm", ylabel="% of arm", xlabel=title + (" (USD)" if col == "history_segment" else ""))
    ax[3].legend(fontsize=8)
    io.save_fig(fig, "experiment_mix")
    names.append("experiment_mix")

    cols = ADJ
    cm = df[cols].corr()
    cmap = LinearSegmentedColormap.from_list("div", io.DIVERGING)
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(cm, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=io.TEXT)
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=.8)
    ax.set_title("Pre-treatment feature correlations (full sample)")
    io.save_fig(fig, "experiment_correlation")
    names.append("experiment_correlation")
    return names


def fig_outcomes_and_balance(df: pd.DataFrame, arms: pd.DataFrame, bal: dict) -> list[str]:
    S = io.SERIES
    a = arms[arms.arm >= 0].set_index("arm").sort_index()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for k, (o, scale, unit) in enumerate([("visit", 100, "%"), ("conversion", 100, "%"), ("spend", 1, "USD")]):
        for j, arm in enumerate((0, 1, 2)):
            y = df.loc[df.arm == arm, o].to_numpy(float)
            m, se = y.mean(), y.std(ddof=1) / np.sqrt(len(y))
            ax[k].errorbar(j, m * scale, yerr=Z * se * scale, fmt="o", color=S[j], capsize=4, ms=7)
        ax[k].set_xticks(range(3), [CFG.ARMS[i].replace(" E-Mail", "\nE-Mail") for i in range(3)])
        ax[k].set_xlim(-.5, 2.5)
        ax[k].set(title={"visit": "Visit rate", "conversion": "Conversion rate", "spend": "Spend per customer"}[o],
                  ylabel={"%": f"{LABEL[o]} rate (%)", "USD": "Mean spend (USD per customer)"}[unit])
    io.save_fig(fig, "experiment_outcomes")

    s = bal["smd"].set_index("variable")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    y = np.arange(len(s))
    ax.scatter(s.smd_mens, y - .12, color=S[0], label="Mens vs No E-Mail", zorder=3)
    ax.scatter(s.smd_womens, y + .12, color=S[1], label="Womens vs No E-Mail", zorder=3)
    ax.axvline(0, color=io.REFERENCE, lw=1)
    for v in (-.1, .1):
        ax.axvline(v, color=io.REFERENCE, lw=1, ls="--")
    ax.set_yticks(y, s.index)
    ax.invert_yaxis()
    ax.set_xlim(-.1 * 1.6, .1 * 1.6)
    ax.set(title="Covariate balance: all |SMD| far below the 0.1 rule of thumb", xlabel="Standardized mean difference vs control")
    ax.legend(loc="lower right", fontsize=8)
    io.save_fig(fig, "experiment_balance")
    return ["experiment_outcomes", "experiment_balance"]


def fig_ate(eff: dict, inc: dict) -> list[str]:
    S = io.SERIES
    names = list(COMPS)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for k, (o, sc, lab) in enumerate([("visit", 100, "ATE on visit rate (pp)"), ("conversion", 100, "ATE on conversion rate (pp)"),
                                      ("spend", 1, "ATE on spend (USD per customer)")]):
        for j, nm in enumerate(names):
            e = eff[nm][o]
            lo, hi = (e["boot_ci_low"], e["boot_ci_high"]) if o == "spend" else (e["ci_low"], e["ci_high"])
            ax[k].errorbar(e["estimate"] * sc, j, xerr=[[(e["estimate"] - lo) * sc], [(hi - e["estimate"]) * sc]], fmt="o",
                           color=S[0], capsize=4, ms=7)
        ax[k].axvline(0, color=io.REFERENCE, lw=1)
        ax[k].set(title=LABEL[o], xlabel=lab)
    ax[0].set_yticks(range(len(names)), names)
    ax[0].invert_yaxis()
    io.save_fig(fig, "experiment_ate")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    arms = ["Mens vs No E-Mail", "Womens vs No E-Mail"]
    for j, nm in enumerate(arms):
        _, lo, hi = inc[nm]["spend"]["per_1000"]
        est = inc[nm]["spend"]["per_1000"][0]
        ax[0].errorbar(j, est, yerr=[[est - lo], [hi - est]], fmt="o", color=S[j], capsize=5, ms=8)
        e, lo, hi = inc[nm]["conversion"]["per_1000"]
        ax[1].errorbar(j, e, yerr=[[e - lo], [hi - e]], fmt="o", color=S[j], capsize=5, ms=8)
    for a_, t, yl in ((ax[0], "Incremental revenue per 1,000 emails", "USD per 1,000 emails (bootstrap 95% CI)"),
                      (ax[1], "Incremental conversions per 1,000 emails", "Conversions per 1,000 emails (95% CI)")):
        a_.axhline(0, color=io.REFERENCE, lw=1)
        a_.set_xticks(range(2), ["Mens E-Mail", "Womens E-Mail"])
        a_.set_xlim(-.6, 1.6)
        a_.set(title=t, ylabel=yl)
    io.save_fig(fig, "experiment_incrementality")
    return ["experiment_ate", "experiment_incrementality"]


def fig_segments(seg: pd.DataFrame) -> str:
    S = io.SERIES
    t = seg["effects"]
    keys = list(dict.fromkeys(zip(t.dimension, t.level)))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 8.2), sharey=True)
    ypos, prev, y = {}, None, 0.0
    for dim, lvl in keys:
        if prev is not None and dim != prev:
            y += .7
        ypos[(dim, lvl)] = y
        y += 1
        prev = dim
    for k, o in enumerate(("visit", "conversion")):
        for j, arm in enumerate(("Mens", "Womens")):
            s = t[(t.outcome == o) & (t.arm == arm)]
            yy = np.array([ypos[(d, l)] for d, l in zip(s.dimension, s.level)]) + (j - .5) * .32
            est, lo, hi = s.estimate.to_numpy() * 100, s.ci_low.to_numpy() * 100, s.ci_high.to_numpy() * 100
            sig = (s.p_holm < .05).to_numpy()
            ax[k].errorbar(est, yy, xerr=[est - lo, hi - est], fmt="none", ecolor=S[j], capsize=2, lw=1.3)
            ax[k].scatter(est[sig], yy[sig], color=S[j], s=34, zorder=3, label=f"{arm} E-Mail (Holm p<0.05)")
            ax[k].scatter(est[~sig], yy[~sig], facecolor=io.SURFACE, edgecolor=S[j], s=34, zorder=3, label=f"{arm} E-Mail (not significant)")
        ax[k].axvline(0, color=io.REFERENCE, lw=1)
        ax[k].set(title=f"{LABEL[o]}: ATE by segment vs No E-Mail", xlabel="Treatment effect (percentage points, 95% CI)")
    ax[0].set_yticks(list(ypos.values()), [f"{d}: {l}".replace("$", r"\$") for d, l in ypos])
    ax[0].invert_yaxis()
    h, l = ax[1].get_legend_handles_labels()
    ax[1].legend(h[:4], l[:4], fontsize=8, loc="lower right")
    io.save_fig(fig, "experiment_segments")
    return "experiment_segments"


# ---------- formatting ----------
def fp(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < .01 else f"{p:.3f}"


def ci(e: float, lo: float, hi: float, sc: float = 1, d: int = 2) -> str:
    return f"{e * sc:.{d}f} [{lo * sc:.{d}f}, {hi * sc:.{d}f}]"


def md_table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_md(df, arms, bal, eff, pw, inc, seg) -> str:
    n = len(df)
    ai = arms.set_index("arm")
    L = [f"## Experiment readout: randomized email test (Hillstrom, n={n:,})", "",
         f"Each customer was randomized 1/3 each to No E-Mail ({ai.loc[0, 'n']:,}), Mens E-Mail ({ai.loc[1, 'n']:,}) or "
         f"Womens E-Mail ({ai.loc[2, 'n']:,}); outcomes are site visit, purchase and spend over the following two weeks. The full "
         "64,000-row sample is used (this is a readout, not model training). Aggregates come from DuckDB SQL "
         "(`sql/experiment_*.sql`); inference is in Python. Binary outcomes: unpooled Wald difference CI, pooled two-proportion z "
         "p-value, and delta-method CI for the relative lift (log ratio). Spend is heavy-tailed, so the primary test is Welch t with "
         f"a {N_BOOT:,}-draw percentile bootstrap CI (Mann-Whitney shown as a robustness check). The 12 primary tests "
         "(4 comparisons x 3 outcomes) are Holm-adjusted together.", "",
         "![Outcomes by arm](results/figures/experiment_outcomes.png)", "",
         "### Randomization check", ""]
    t = bal["tests"]
    rows = [[r.variable, r.test, f"{r.statistic:.2f}", int(r.dof), fp(r.p_raw), fp(r.p_holm)] for r in t.itertuples()]
    L += [md_table(["Variable", "Test", "Statistic", "df", "p (raw)", "p (Holm)"], rows), "",
          f"Omnibus LR test that the {len(ADJ)} covariates jointly predict arm (multinomial logit): chi2({bal['omnibus']['dof']}) = "
          f"{bal['omnibus']['lr_stat']:.1f}, p = {fp(bal['omnibus']['p_value'])}. Max |SMD| vs control: "
          f"{bal['smd'][['smd_mens', 'smd_womens']].abs().to_numpy().max():.3f} (Mens/Womens).", "",
          "![Covariate balance](results/figures/experiment_balance.png)", "",
          "### Average treatment effects", ""]
    for o, sc, unit, d in (("visit", 100, "pp", 2), ("conversion", 100, "pp", 2), ("spend", 1, "USD", 3)):
        rows = []
        for nm in COMPS:
            e = eff[nm][o]
            lo, hi = (e["boot_ci_low"], e["boot_ci_high"]) if o == "spend" else (e["ci_low"], e["ci_high"])
            rel = f"{e['rel_lift'] * 100:.1f}% [{e['rel_ci_low'] * 100:.1f}, {e['rel_ci_high'] * 100:.1f}]"
            extra = f"{fp(e['p_mannwhitney'])}" if o == "spend" else "-"
            rows.append([nm, f"{e['mean_control'] * (sc if o != 'spend' else 1):.{d}f}", f"{e['mean_treated'] * (sc if o != 'spend' else 1):.{d}f}",
                         ci(e["estimate"], lo, hi, sc, d), rel, fp(e["p_value"]), fp(e["p_holm"]), extra])
        base = "%" if o != "spend" else "USD"
        L += [f"**{LABEL[o]}** (difference in {unit}; 95% CI{' = bootstrap' if o == 'spend' else ''})", "",
              md_table(["Comparison", f"Control mean ({base})", f"Treated mean ({base})", f"ATE ({unit}) [95% CI]", "Relative lift % [95% CI]",
                        "p (raw)", "p (Holm, 12 tests)"] + (["Mann-Whitney p"] if o == "spend" else []), [r[:-1] if o != "spend" else r for r in rows]), ""]
    L += ["![ATE forest](results/figures/experiment_ate.png)", "", "### Variance reduction (Lin 2013 regression adjustment)", "",
          f"OLS of the outcome on treatment, the {len(ADJ)} centred pre-treatment covariates and their interactions with treatment; "
          "HC2 robust SEs. The comparison is against the unadjusted HC2 SE (same estimator, no covariates). Reductions are small because the "
          "covariates predict a rare, mostly-idiosyncratic outcome poorly; in an RCT this only affects precision, not validity.", ""]
    rows = []
    for nm in COMPS:
        for o in OUTS:
            l = eff[nm][o]["lin"]
            rows.append([nm, o, f"{l['unadj_estimate'] * (100 if o != 'spend' else 1):.3f}", f"{l['adj_estimate'] * (100 if o != 'spend' else 1):.3f}",
                         f"{l['unadj_se'] * (100 if o != 'spend' else 1):.4f}", f"{l['adj_se'] * (100 if o != 'spend' else 1):.4f}", f"{l['se_reduction_pct']:.1f}%"])
    L += [md_table(["Comparison", "Outcome", "Unadj. ATE", "Adj. ATE", "Unadj. SE", "Adj. SE", "SE reduction"], rows),
          "(Rates in pp, spend in USD.)", "", "### Power", "",
          f"Two-sided alpha = 0.05, 80% power, arm sizes as run. MDE = smallest true effect detectable; required n is per arm, to "
          "detect the observed effect.", ""]
    rows = []
    for r in pw["rows"]:
        sc = 1 if r["outcome"] == "spend" else 100
        u = "USD" if r["outcome"] == "spend" else "pp"
        rows.append([r["comparison"], r["outcome"], f"{r['baseline'] * sc:.{3 if sc == 1 else 2}f}", f"{r['mde_abs'] * sc:.{3 if sc == 1 else 2}f} {u}",
                     f"{r['mde_rel'] * 100:.1f}%", f"{r['observed'] * sc:.{3 if sc == 1 else 2}f} {u}",
                     f"{r['n_per_arm_for_observed']:,.0f}" if r["n_per_arm_for_observed"] else "-"])
    L += [md_table(["Comparison", "Outcome", "Baseline", "MDE (abs)", "MDE (rel)", "Observed", "n/arm to detect observed"], rows), "",
          "### Incrementality", "",
          f"**Cost assumption:** `COST_PER_CONTACT = ${CFG.COST_PER_CONTACT:.2f}` per email is an ILLUSTRATIVE assumption (the dataset has no cost "
          "column). The break-even cost per email (= incremental revenue per email) does not depend on it. Revenue is the two-week spend only "
          "(no margin, no lifetime value). Totals scale each ATE to the number of emails actually sent in that arm of the test.", ""]
    rows = []
    for nm, r in inc.items():
        rows.append([nm.replace(" vs No E-Mail", ""), f"{r['emails_sent']:,}",
                     ci(*r["visit"]["per_1000"], d=1), ci(*r["conversion"]["per_1000"], d=1), ci(*r["spend"]["per_1000"], d=0),
                     ci(*r["spend"]["total"], d=0), ci(*r["iroas"], d=2), ci(*r["break_even_cost_per_email"], d=3)])
    L += [md_table(["Arm", "Emails sent", "Incr. visits /1k [95% CI]", "Incr. conversions /1k [95% CI]", "Incr. revenue /1k (USD) [95% CI]",
                    "Total incr. revenue (USD)", f"iROAS @ ${CFG.COST_PER_CONTACT:.2f}/email", "Break-even cost / email (USD)"], rows), "",
          "![Incrementality](results/figures/experiment_incrementality.png)", "",
          "### Segment readout and heterogeneity", "",
          "Pre-specified segments (channel, ZIP type, newbie, past mens / womens buyer, history bucket), defined before looking at outcomes. "
          f"Per-segment ATEs vs No E-Mail with 95% Wald CIs; Holm correction over all {len(seg['effects'])} segment x arm x outcome tests. "
          "Interaction tests are linear-probability models `y ~ T x segment` with HC2 SEs (joint Wald test on the interaction terms), "
          f"Holm-adjusted over all {len(seg['interactions'])} interaction tests.", "",
          "![Segment forest](results/figures/experiment_segments.png)", ""]
    te = seg["effects"]
    rows = []
    for (dim, lvl), g in te.groupby(["dimension", "level"], sort=False):
        cells = []
        for arm in ("Mens", "Womens"):
            for o in ("visit", "conversion"):
                r = g[(g.arm == arm) & (g.outcome == o)].iloc[0]
                cells.append(f"{ci(r.estimate, r.ci_low, r.ci_high, 100)} ({fp(r.p_holm)})")
        rows.append([dim, lvl, f"{int(g.n_control.iloc[0]):,}"] + cells)
    L += [md_table(["Segment", "Level", "n control", "Mens: visit pp (Holm p)", "Mens: conv. pp (Holm p)", "Womens: visit pp (Holm p)", "Womens: conv. pp (Holm p)"], rows), "",
          "Interaction tests (treatment x segment):", ""]
    it = seg["interactions"].sort_values("p_value")
    rows = []
    for r in it.itertuples():
        d = f"{r.diff * 100:.2f} [{r.diff_ci_low * 100:.2f}, {r.diff_ci_high * 100:.2f}] ({r.contrast})" if isinstance(r.contrast, str) else "-"
        rows.append([r.dimension, r.arm, r.outcome, f"{r.chi2:.1f}", r.dof, fp(r.p_value), fp(r.p_holm), d])
    L += [md_table(["Segment", "Arm", "Outcome", "Wald chi2", "df", "p (raw)", "p (Holm)", "Effect difference pp [95% CI] (level 1 vs level 0; 1 = yes / new)"], rows), ""]
    return "\n".join(L)


def caveats(df, eff, seg, heter) -> list[str]:
    conv = eff["Any E-Mail vs No E-Mail"]["conversion"]
    return [
        "Outcomes cover a two-week window only: no long-run, cannibalization or unsubscribe effects are measured, so iROAS is a short-window figure.",
        f"COST_PER_CONTACT (${CFG.COST_PER_CONTACT:.2f}) is an illustrative assumption; use the break-even cost per email for decisions. Revenue is not margin.",
        "Spend is zero-inflated and heavy-tailed: the Welch/bootstrap CIs are for the mean; Mann-Whitney tests a different (distributional) hypothesis.",
        "Segment-level CIs are unadjusted Wald intervals; segments with small n (e.g. top history buckets) have wide intervals and low power for effect differences.",
        "Interaction tests use a linear-probability model with robust SEs; logistic-scale heterogeneity may differ. Only pre-specified dimensions were tested, and Holm correction is applied across the full family.",
        f"History is floor-censored at $29.99 ({df.history_at_floor.mean() * 100:.1f}% of customers) and 6,562 exact-duplicate rows are distinct customers (see results/data.json).",
        f"Balance tests: with {len(seg['effects'])} segment tests and 8 balance tests, some raw p<0.05 are expected by chance; conclusions rest on Holm-adjusted values.",
        "Hillstrom is a 2008 single-retailer test; effect sizes should not be assumed to transfer.",
    ] + ([f"Heterogeneity: {heter['summary']}"] if heter.get("summary") else [])


def run() -> dict:
    warnings.filterwarnings("ignore")
    io.style()
    df = data.load()
    arms, seg_sql = sql_readouts(df)
    bal = balance(df)
    eff = effects(df, arms)
    pw = power_block(df, eff)
    inc = incrementality(df, eff)
    seg = segments(df, seg_sql)

    figs = fig_eda(df) + fig_outcomes_and_balance(df, arms, bal) + fig_ate(eff, inc) + [fig_segments(seg)]

    it = seg["interactions"].sort_values("p_value")
    best = it.iloc[0]
    sig = it[it.p_holm < .05]
    if len(sig):
        b = sig.iloc[0]
        heter = {"survived_correction": True, "n_significant": int(len(sig)),
                 "strongest": b.to_dict(),
                 "summary": (f"strongest interaction is {b.dimension} x {b.arm} E-Mail on {b.outcome} "
                             f"(Wald chi2={b.chi2:.1f}, df={int(b.dof)}, raw p={fp(b.p_value)}, Holm p={fp(b.p_holm)}); "
                             f"{len(sig)} of {len(it)} interaction tests survive Holm correction.")}
    else:
        heter = {"survived_correction": False, "n_significant": 0, "strongest": best.to_dict(),
                 "summary": (f"no interaction survived Holm correction (smallest: {best.dimension} x {best.arm} on {best.outcome}, "
                             f"raw p={fp(best.p_value)}, Holm p={fp(best.p_holm)}).")}

    def h(nm: str, o: str) -> dict:
        e = eff[nm][o]
        lo, hi = (e["boot_ci_low"], e["boot_ci_high"]) if o == "spend" else (e["ci_low"], e["ci_high"])
        return {"ate": e["estimate"], "ci95": [lo, hi], "p_raw": e["p_value"], "p_holm": e["p_holm"],
                "rel_lift": e["rel_lift"], "rel_lift_ci95": [e["rel_ci_low"], e["rel_ci_high"]],
                "se_reduction_pct": e["lin"]["se_reduction_pct"], "adj_ate": e["lin"]["adj_estimate"]}

    headline = {
        "ate": {nm: {o: h(nm, o) for o in OUTS} for nm in COMPS},
        "incrementality": inc,
        "cost_per_contact_assumption_usd": CFG.COST_PER_CONTACT,
        "se_reduction_pct_range": [min(eff[c][o]["lin"]["se_reduction_pct"] for c in COMPS for o in OUTS),
                                   max(eff[c][o]["lin"]["se_reduction_pct"] for c in COMPS for o in OUTS)],
        "heterogeneity": heter,
        "randomization_ok": bool(bal["omnibus"]["p_value"] > .05 and (bal["tests"]["p_holm"] > .05).all()),
    }
    md = render_md(df, arms, bal, eff, pw, inc, seg)
    cav = caveats(df, eff, seg, heter)
    top = ["### EDA", "", "![Distributions](results/figures/experiment_distributions.png)", "",
           "![Customer mix by arm](results/figures/experiment_mix.png)", "",
           "![Correlations](results/figures/experiment_correlation.png)", ""]
    md = md.replace("### Randomization check", "\n".join(top) + "\n### Randomization check", 1)
    md += "\n**Caveats**\n\n" + "\n".join(f"- {c}" for c in cav) + "\n"
    (CFG.RESULTS / "experiment.md").write_text(md)
    io.save_json({
        "headline": headline, "arms": arms, "balance": {"tests": bal["tests"], "omnibus": bal["omnibus"], "smd": bal["smd"]},
        "effects": eff, "power": pw, "segment_effects": seg["effects"].to_dict("records"),
        "interactions": seg["interactions"].to_dict("records"), "figures": figs,
    }, "experiment")
    print(f"headline: {heter['summary']}")
    return headline
