"""Stage `uplift`: heterogeneous treatment effects, uplift targeting, interpretable segments,
multi-arm email choice, and budget-constrained policy evaluation on the Hillstrom RCT.

Business question: which customers should receive the campaign, and which email, given a
fixed contact budget? Everything is fit on train, early-stopped / selected on val, and
evaluated once on the randomized test split."""
from __future__ import annotations

import time

import numpy as np
from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causalads import config as C
from causalads import data, io
from causalads import stats as S
from causalads import uplift_learners as L
from causalads import uplift_metrics as M
from causalads import uplift_plots as P
from causalads import uplift_report as R

OUTCOMES = ["visit", "conversion", "spend"]
NAMES = [*L.LEARNERS, L.RESPONSE]
BUDGET_GRID = np.round(np.arange(1, 21) * 0.05, 2)
CLUSTER_FEATURES = ["recency", "log_history", "mens", "womens", "newbie", "ch_web", "ch_phone"]
POLICY_LABELS = {
    "random": "Random targeting",
    "response": "Response model (visit propensity if emailed), send Mens",
    "uplift_binary": "Binary uplift (any-email CATE), send Mens",
    "uplift_mens": "Mens-email uplift (CATE Mens vs none), send Mens",
    "uplift_multiarm": "Multi-arm uplift, send higher-CATE email",
    "uplift_revenue": "Revenue uplift (multi-arm CATE on spend), send higher-CATE email",
    "response_spend": "Response model (expected spend if emailed), send Mens",
}
FEATURE_LABELS = {
    "recency": "months since last purchase", "history": "past-year spend", "log_history": "past-year spend",
    "history_bucket": "spend bucket", "history_at_floor": "spend at $29.99 floor", "mens": "bought men's",
    "womens": "bought women's", "both_lines": "bought both lines", "newbie": "new customer",
    "recent_buyer": "bought in last 3 months", "zip_urban": "urban", "zip_suburban": "suburban",
    "zip_rural": "rural", "ch_phone": "phone channel", "ch_web": "web channel", "ch_multi": "multichannel",
}
FEATURE_GROUPS = {
    "Past spend (history, log, bucket, floor)": ["history", "log_history", "history_bucket", "history_at_floor"],
    "Product line (mens, womens, both)": ["mens", "womens", "both_lines"],
    "Recency (recency, recent_buyer)": ["recency", "recent_buyer"],
    "New customer": ["newbie"],
    "Channel (phone, web, multi)": ["ch_phone", "ch_web", "ch_multi"],
    "Zip type (urban, suburban, rural)": ["zip_urban", "zip_suburban", "zip_rural"],
}


def _fold(df, outcome: str) -> L.Fold:
    return L.Fold(df[data.FEATURES].to_numpy(float), df["treated"].to_numpy(int),
                  df["arm"].to_numpy(int), df[outcome].to_numpy(float))


def _log(msg: str, t0: float) -> None:
    print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- fitting
def _fit_all(folds: dict, t0: float) -> tuple[dict, dict, dict]:
    """preds[seed][outcome][name] = {'val', 'test'}; multi[seed][outcome] = (n_test, 2) CATEs."""
    preds, fits, multi = {}, {}, {}
    for seed in (C.SEED, *C.SEEDS):
        preds[seed], multi[seed] = {}, {}
        for o in OUTCOMES:
            tr, va, te = folds[o]
            fit = L.fit_binary(tr, va, binary=o != "spend", seed=seed)
            preds[seed][o] = {k: {"val": f(va.X), "test": f(te.X)} for k, f in fit["predict"].items()}
            if seed == C.SEED:
                fits[o] = fit
            # multi-arm forests: visit for every seed (policy stability), spend for the main seed
            if o == "visit" or seed == C.SEED:
                cf3 = L.fit_multiarm(tr, seed, fit["n_trees"])
                multi[seed][o] = np.asarray(cf3.const_marginal_effect(te.X)).reshape(len(te.X), 2)
                if seed == C.SEED:
                    fits[f"multi_{o}"] = cf3
        _log(f"seed {seed}: fitted 5 CATE learners + response model x 3 outcomes, multi-arm forest(s)", t0)
    return preds, fits, multi


# ---------------------------------------------------------------- learner evaluation
def _learner_eval(folds: dict, preds: dict, W: np.ndarray) -> tuple[dict, dict, dict]:
    table, curves, selected = {}, {}, {}
    for o in OUTCOMES:
        _, va, te = folds[o]
        p = preds[C.SEED][o]
        ev = {k: M.uplift_eval(te.y, te.t, p[k]["test"], W) for k in NAMES}
        val = {k: M.uplift_eval(va.y, va.t, p[k]["val"], None)["point"] for k in NAMES}
        selected[o] = max(L.LEARNERS, key=lambda k: val[k]["norm_qini"][0])
        rows = {}
        for k in NAMES:
            pt, bt = ev[k]["point"], ev[k]["boot"]
            per_seed = [M.uplift_eval(te.y, te.t, preds[s][o][k]["test"], None)["point"] for s in C.SEEDS]
            nq = np.array([r["norm_qini"][0] for r in per_seed])
            ag = np.array([r["auuc_gain"][0] for r in per_seed]) * 1000
            rows[k] = {
                "val_norm_qini": float(val[k]["norm_qini"][0]),
                "norm_qini": M.ci(bt["norm_qini"], pt["norm_qini"][0]),
                "auuc_gain_per_1000": M.ci(bt["auuc_gain"], pt["auuc_gain"][0], 1000),
                "qini_coef_per_1000": M.ci(bt["qini_coef"], pt["qini_coef"][0], 1000),
                "seeds_norm_qini": {"mean": float(nq.mean()), "std": float(nq.std(ddof=1)), "values": nq},
                "seeds_auuc_gain_per_1000": {"mean": float(ag.mean()), "std": float(ag.std(ddof=1)), "values": ag},
            }
            if k != L.RESPONSE:
                r = ev[L.RESPONSE]
                rows[k]["vs_response_norm_qini"] = M.paired(bt["norm_qini"], r["boot"]["norm_qini"],
                                                            pt["norm_qini"][0], r["point"]["norm_qini"][0])
                rows[k]["vs_response_auuc_gain_per_1000"] = M.paired(
                    bt["auuc_gain"], r["boot"]["auuc_gain"], pt["auuc_gain"][0], r["point"]["auuc_gain"][0], 1000)
        table[o] = rows
        curves[o] = {k: {"gain": ev[k]["point"]["gain_curve"][0],
                         "lo": np.quantile(ev[k]["boot"]["gain_curve"], 0.025, axis=0),
                         "hi": np.quantile(ev[k]["boot"]["gain_curve"], 0.975, axis=0)} for k in NAMES}
    return table, curves, selected


def _heterogeneity(folds: dict, preds: dict, selected: dict) -> dict:
    out = {"blp": {}, "gates": {}, "deciles": {}}
    for o in OUTCOMES:
        te = folds[o][2]
        p = preds[C.SEED][o]
        out["blp"][o] = {k: M.blp(te.y, te.t, p[k]["test"], p["mu0"]["test"]) for k in NAMES}
        out["gates"][o] = {k: M.gates(te.y, te.t, p[k]["test"], 5) for k in (selected[o], L.RESPONSE)}
        out["deciles"][o] = {k: M.by_group(te.y, te.t, M.quantile_groups(p[k]["test"], 10))
                             for k in (selected[o], L.RESPONSE)}
        out.setdefault("ate", {})[o] = S.diff_in_means(te.y, te.t)
        # per-seed BLP beta2 for the selected learner: is the heterogeneity finding seed-robust?
        b2 = np.array([M.blp(te.y, te.t, preds[s][o][selected[o]]["test"], preds[s][o]["mu0"]["test"])["beta2"]
                       for s in C.SEEDS])
        out["blp"][o][selected[o]]["seeds_beta2"] = {"mean": float(b2.mean()), "std": float(b2.std(ddof=1))}
    return out


# ---------------------------------------------------------------- segments
def _rules(tree, is_binary: dict, negation: dict[str, str]) -> dict[int, str]:
    t, names = tree.tree_, data.FEATURES
    leaves: dict[int, list] = {}

    def walk(node: int, conds: list) -> None:
        if t.children_left[node] == -1:
            leaves[node] = conds
            return
        f, thr = names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [(f, "<=", thr)])
        walk(t.children_right[node], conds + [(f, ">", thr)])

    walk(0, [])
    shown = {"log_history": np.expm1}
    text = {}
    for leaf, conds in leaves.items():
        parts, bounds = [], {}
        for f, op, thr in conds:
            if is_binary[f]:
                parts.append((f, FEATURE_LABELS[f] if op == ">" else negation.get(f, f"not {FEATURE_LABELS[f]}")))
            else:
                lo, hi = bounds.get(f, (-np.inf, np.inf))
                bounds[f] = (max(lo, thr), hi) if op == ">" else (lo, min(hi, thr))
        for f, (lo, hi) in bounds.items():
            g = shown.get(f, lambda v: v)
            money = f in ("history", "log_history")
            fmt = (lambda v: f"${g(v):,.0f}") if money else (lambda v: f"{g(v):g}")
            if f in ("recency", "history_bucket"):  # integer-valued: thresholds sit at k + 0.5
                lo_s, hi_s = (f"{int(np.floor(lo)) + 1}" if np.isfinite(lo) else None), (f"{int(np.floor(hi))}" if np.isfinite(hi) else None)
            else:
                lo_s, hi_s = (fmt(lo) if np.isfinite(lo) else None), (fmt(hi) if np.isfinite(hi) else None)
            lab = FEATURE_LABELS[f]
            if lo_s and hi_s:
                parts.append((f, f"{lab} {lo_s}–{hi_s}" if f in ("recency", "history_bucket") else f"{lo_s} < {lab} ≤ {hi_s}"))
            elif lo_s:
                parts.append((f, f"{lab} ≥ {lo_s}" if f in ("recency", "history_bucket") else f"{lab} > {lo_s}"))
            else:
                parts.append((f, f"{lab} ≤ {hi_s}"))
        text[leaf] = " AND ".join(p for _, p in parts) or "all customers"
    return text


def _segments(folds: dict, fits: dict, preds: dict, tr_df) -> dict:
    cf = fits["visit"]["cf"]
    _, va, te = folds["visit"]
    imp = np.asarray(cf.feature_importances_, dtype=float)
    per_feature = dict(zip(data.FEATURES, imp))
    groups = {g: float(sum(per_feature[f] for f in fs)) for g, fs in FEATURE_GROUPS.items()}
    # Tree is fit on VAL-set forest CATEs (forest trained on train), rules are scored on TEST.
    interp = SingleTreeCateInterpreter(max_depth=3, min_samples_leaf=1000, random_state=C.SEED)
    interp.interpret(cf, va.X)
    tree = interp.tree_model_
    is_binary = {f: bool(set(np.unique(tr_df[f])) <= {0, 1}) for f in data.FEATURES}
    # every Hillstrom customer bought at least one line, so "not women's" == men's-only
    neither = bool(((tr_df["mens"] == 0) & (tr_df["womens"] == 0)).any())
    negation = {} if neither else {"womens": "men's-only buyer", "mens": "women's-only buyer"}
    text = _rules(tree, is_binary, negation)
    leaf_te = tree.apply(te.X)
    cate_te = preds[C.SEED]["visit"]["Causal forest"]["test"]
    obs = {o: {g["group"]: g for g in M.by_group(folds[o][2].y, te.t, leaf_te)} for o in OUTCOMES}
    rows = []
    for leaf in np.unique(leaf_te):
        rows.append({"leaf": int(leaf), "rule": text[int(leaf)],
                     "predicted_cate_val": float(tree.tree_.value[leaf].ravel()[0]),
                     "predicted_cate_test": float(cate_te[leaf_te == leaf].mean()),
                     "n_test": int((leaf_te == leaf).sum()), "share": float((leaf_te == leaf).mean()),
                     **{f"ate_{o}": obs[o][leaf] for o in OUTCOMES}})
    rows.sort(key=lambda r: -r["predicted_cate_val"])
    q = {o: M.cochran_q([r[f"ate_{o}"] for r in rows]) for o in OUTCOMES}
    top, bot = rows[0]["ate_visit"], rows[-1]["ate_visit"]
    d = top["estimate"] - bot["estimate"]
    se = float(np.hypot(top["se"], bot["se"]))
    return {"feature_importance": per_feature, "group_importance": groups, "rules": rows,
            "heterogeneity_test": q, "tree_depth": 3, "min_samples_leaf_val": 1000,
            "top_minus_bottom_visit": {"estimate": d, "se": se, "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se}}


def _describe(profile: dict, overall: dict) -> str:
    parts = []
    line = {"men's only": profile["mens_only"], "women's only": profile["womens_only"], "both lines": profile["both_lines"]}
    k, v = max(line.items(), key=lambda kv: kv[1])
    if v > 0.6:
        parts.append(k)
    ch = {"web": profile["ch_web"], "phone": profile["ch_phone"], "multichannel": profile["ch_multi"]}
    k, v = max(ch.items(), key=lambda kv: kv[1])
    if v > 0.6:
        parts.append(k)
    if profile["newbie"] > 0.7:
        parts.append("new")
    elif profile["newbie"] < 0.3:
        parts.append("returning")
    if profile["recency"] < overall["recency"] - 1.0:
        parts.append("recent")
    elif profile["recency"] > overall["recency"] + 1.0:
        parts.append("lapsed")
    if profile["history"] > 1.5 * overall["history"]:
        parts.append("high spend")
    elif profile["history"] < 0.6 * overall["history"]:
        parts.append("low spend")
    return ", ".join(parts) or "mixed"


def _clusters(tr_df, te_df, folds: dict) -> dict:
    scaler = StandardScaler().fit(tr_df[CLUSTER_FEATURES].to_numpy(float))
    Ztr = scaler.transform(tr_df[CLUSTER_FEATURES].to_numpy(float))
    Zte = scaler.transform(te_df[CLUSTER_FEATURES].to_numpy(float))
    sil, models = {}, {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Ztr)
        sil[k] = float(silhouette_score(Ztr, km.labels_, sample_size=5000, random_state=C.SEED))
        models[k] = km
    k = max(sil, key=sil.get)
    lab_tr, lab_te = models[k].labels_, models[k].predict(Zte)

    def profile(df, mask) -> dict:
        d = df[mask]
        return {"recency": float(d["recency"].mean()), "history": float(d["history"].mean()),
                "mens_only": float(((d["mens"] == 1) & (d["womens"] == 0)).mean()),
                "womens_only": float(((d["mens"] == 0) & (d["womens"] == 1)).mean()),
                "both_lines": float(d["both_lines"].mean()), "newbie": float(d["newbie"].mean()),
                "ch_web": float(d["ch_web"].mean()), "ch_phone": float(d["ch_phone"].mean()),
                "ch_multi": float(d["ch_multi"].mean())}

    overall = profile(tr_df, np.ones(len(tr_df), bool))
    te = folds["visit"][2]
    obs = {o: {g["group"]: g for g in M.by_group(folds[o][2].y, te.t, lab_te)} for o in OUTCOMES}
    rows = []
    for c in range(k):
        prof = profile(tr_df, lab_tr == c)
        rows.append({"cluster": c, "label": _describe(prof, overall), "share_train": float((lab_tr == c).mean()),
                     "n_test": int((lab_te == c).sum()), "profile": prof,
                     **{f"ate_{o}": obs[o][c] for o in OUTCOMES}})
    rows.sort(key=lambda r: -r["ate_visit"]["estimate"])
    return {"features": CLUSTER_FEATURES, "silhouette": sil, "k": k, "clusters": rows, "overall_profile": overall,
            "heterogeneity_test": {o: M.cochran_q([r[f"ate_{o}"] for r in rows]) for o in OUTCOMES}}


# ---------------------------------------------------------------- multi-arm
def _multi_arm(folds: dict, multi: dict, mu0: np.ndarray, te_df) -> dict:
    tau = multi[C.SEED]["visit"]
    te = folds["visit"][2]
    pref = 1 + (tau[:, 1] > tau[:, 0]).astype(int)
    mens, womens = te_df["mens"].to_numpy(), te_df["womens"].to_numpy()
    hist = np.where((mens == 1) & (womens == 0), "Men's only", np.where((mens == 0) & (womens == 1), "Women's only", "Both lines"))
    rule = np.where(hist == "Women's only", 2, 1)
    crosstab = {h: {"n": int((hist == h).sum()), "share_prefer_womens": float((pref[hist == h] == 2).mean()),
                    "mean_cate_mens": float(tau[hist == h, 0].mean()), "mean_cate_womens": float(tau[hist == h, 1].mean())}
                for h in ("Men's only", "Women's only", "Both lines")}
    emailed = te.a > 0

    def matched_gain(choice: np.ndarray) -> dict:
        # Among emailed customers the email is randomized 50/50, so this is the causal gain
        # from sending the chosen email instead of the other one.
        out = {"overall": S.diff_in_means(te.y[emailed], (te.a[emailed] == choice[emailed]).astype(int))}
        for arm in (1, 2):
            m = emailed & (choice == arm)
            if m.sum() > 50 and 0 < (te.a[m] == arm).mean() < 1:
                out[C.ARMS[arm]] = S.diff_in_means(te.y[m], (te.a[m] == arm).astype(int))
        return out

    blp = {}
    for arm in (1, 2):
        m = (te.a == 0) | (te.a == arm)
        blp[C.ARMS[arm]] = M.blp(te.y[m], (te.a[m] == arm).astype(float), tau[m, arm - 1], mu0[m], p=0.5)
    seed_share = np.array([(multi[s]["visit"][:, 1] > multi[s]["visit"][:, 0]).mean() for s in C.SEEDS])
    return {"share_prefer": {"Mens E-Mail": float((pref == 1).mean()), "Womens E-Mail": float((pref == 2).mean())},
            "share_prefer_womens_seeds": {"mean": float(seed_share.mean()), "std": float(seed_share.std(ddof=1))},
            "share_both_cates_negative": float((tau.max(1) <= 0).mean()),
            "mean_cate": {"Mens E-Mail": float(tau[:, 0].mean()), "Womens E-Mail": float(tau[:, 1].mean())},
            "by_purchase_history": crosstab,
            "agreement_with_purchase_rule": float((pref == rule).mean()),
            "matched_email_gain_model": matched_gain(pref), "matched_email_gain_rule": matched_gain(rule),
            "blp": blp, "_pref": pref, "_rule": rule}


# ---------------------------------------------------------------- policies
def _topk(score: np.ndarray, arm, frac: float) -> np.ndarray:
    n = len(score)
    pi = np.zeros(n, dtype=int)
    idx = M.order(score)[: int(round(frac * n))]
    pi[idx] = arm[idx] if np.ndim(arm) else arm
    return pi


def _policy_specs(preds_s: dict, multi_s: dict, selected: str, best_arm: dict, spend: bool = True) -> dict:
    tv = multi_s["visit"]
    specs = {
        "response": (preds_s["visit"][L.RESPONSE]["test"], best_arm["visit"]),
        "uplift_binary": (preds_s["visit"][selected]["test"], best_arm["visit"]),
        "uplift_mens": (tv[:, 0], 1),
        "uplift_multiarm": (tv.max(1), 1 + tv.argmax(1)),
    }
    if spend:
        ts = multi_s["spend"]
        specs["uplift_revenue"] = (ts.max(1), 1 + ts.argmax(1))
        specs["response_spend"] = (preds_s["spend"][L.RESPONSE]["test"], best_arm["spend"])
    return specs


def _policies(folds: dict, preds: dict, multi: dict, selected: str, best_arm: dict,
              mm: dict, W: np.ndarray) -> dict:
    te = {o: folds[o][2] for o in OUTCOMES}
    a = te["visit"].a
    specs = _policy_specs(preds[C.SEED], multi[C.SEED], selected, best_arm)
    keys, rows = [], []
    for p, (score, arm) in specs.items():
        for f in BUDGET_GRID:
            pi = _topk(score, arm, f)
            for o in OUTCOMES:
                keys.append((p, float(f), o))
                rows.append(M.ipw_contrib(te[o].y, a, pi))
    full = {"all_mens": np.full(len(a), 1), "all_womens": np.full(len(a), 2),
            "purchase_rule": mm["_rule"], "multiarm_argmax": mm["_pref"]}
    for p, pi in full.items():
        for o in OUTCOMES:
            keys.append((p, 1.0, o))
            rows.append(M.ipw_contrib(te[o].y, a, pi))
    D = np.vstack(rows)
    point, draws = D.mean(axis=1), M.boot_means(D, W)
    idx = {k: i for i, k in enumerate(keys)}
    get = lambda p, f, o: (point[idx[(p, float(f), o)]], draws[idx[(p, float(f), o)]])  # noqa: E731

    curves = {p: {o: {"est": [], "lo": [], "hi": []} for o in OUTCOMES} for p in ["random", *specs]}
    for o in OUTCOMES:
        am_pt, am_dr = get("all_mens" if best_arm["visit"] == 1 else "all_womens", 1.0, o)
        for f in BUDGET_GRID:
            for p in ["random", *specs]:
                pt, dr = (f * am_pt, f * am_dr) if p == "random" else get(p, f, o)
                c = M.ci(dr, pt, 1000)
                curves[p][o]["est"].append(c["estimate"])
                curves[p][o]["lo"].append(c["ci_low"])
                curves[p][o]["hi"].append(c["ci_high"])

    def at(p: str, f: float, o: str) -> tuple[float, np.ndarray]:
        if p == "random":
            am_pt, am_dr = get("all_mens" if best_arm["visit"] == 1 else "all_womens", 1.0, o)
            return f * am_pt, f * am_dr
        return get(p, f, o)

    table, compare, iroas, capture = {}, {}, {}, {}
    full_pt, full_dr = at("random", 1.0, "visit")
    for f in C.BUDGET_FRACS:
        fk = f"{f:.2f}"
        # share of the email-everyone incremental visits captured with only f of the sends
        capture[fk] = {p: M.ci(at(p, f, "visit")[1] / full_dr, at(p, f, "visit")[0] / full_pt, 100)
                       for p in ["random", *specs]}
        table[fk] = {p: {o: M.ci(at(p, f, o)[1], at(p, f, o)[0], 1000) for o in OUTCOMES} for p in ["random", *specs]}
        compare[fk] = {}
        for u, r in [("uplift_binary", "response"), ("uplift_mens", "response"), ("uplift_multiarm", "response"),
                     ("uplift_multiarm", "uplift_binary"), ("uplift_revenue", "response_spend"),
                     ("uplift_revenue", "uplift_multiarm"), ("uplift_binary", "random")]:
            compare[fk][f"{u}_vs_{r}"] = {}
            for o in OUTCOMES:
                (up, ud), (rp, rd) = at(u, f, o), at(r, f, o)
                pct_d = np.divide(ud - rd, rd, out=np.full_like(ud, np.nan), where=rd != 0)
                pct = M.ci(pct_d[np.isfinite(pct_d)], (up - rp) / rp, 100) if rp != 0 else None  # undefined, keep JSON NaN-free
                compare[fk][f"{u}_vs_{r}"][o] = {"diff_per_1000": M.paired(ud, rd, up, rp, 1000),
                                                  "pct_improvement": pct}
        iroas[fk] = {}
        for p in ["random", *specs]:
            rp, rd = at(p, f, "spend")
            iroas[fk][p] = {"iroas": M.ci(rd / (f * C.COST_PER_CONTACT), rp / (f * C.COST_PER_CONTACT)),
                            "break_even_cost_per_contact": M.ci(rd / f, rp / f)}

    full_tbl = {p: {o: M.ci(get(p, 1.0, o)[1], get(p, 1.0, o)[0], 1000) for o in OUTCOMES} for p in full}
    full_cmp = {f"{u}_vs_{r}": {o: M.paired(get(u, 1.0, o)[1], get(r, 1.0, o)[1], get(u, 1.0, o)[0], get(r, 1.0, o)[0], 1000)
                                for o in OUTCOMES}
                for u, r in [("multiarm_argmax", "all_mens"), ("multiarm_argmax", "purchase_rule"), ("purchase_rule", "all_mens")]}

    # seed stability of the policy comparison (point estimates only)
    seeds = {}
    for s in C.SEEDS:
        sp = _policy_specs(preds[s], multi[s], selected, best_arm, spend=False)
        seeds[s] = {p: {f"{f:.2f}": {o: float(M.ipw_contrib(te[o].y, a, _topk(sc, arm, f)).mean() * 1000) for o in OUTCOMES}
                        for f in C.BUDGET_FRACS} for p, (sc, arm) in sp.items()}
    seed_summary = {}
    for p in ("response", "uplift_binary", "uplift_mens", "uplift_multiarm"):
        seed_summary[p] = {}
        for f in C.BUDGET_FRACS:
            fk = f"{f:.2f}"
            seed_summary[p][fk] = {o: {"mean": float(np.mean(v := [seeds[s][p][fk][o] for s in C.SEEDS])),
                                       "std": float(np.std(v, ddof=1))} for o in OUTCOMES}
    for u in ("uplift_binary", "uplift_mens", "uplift_multiarm"):
        seed_summary[f"{u}_minus_response"] = {
            f"{f:.2f}": {o: {"mean": float(np.mean(v := [seeds[s][u][f'{f:.2f}'][o] - seeds[s]["response"][f'{f:.2f}'][o] for s in C.SEEDS])),
                             "std": float(np.std(v, ddof=1))} for o in OUTCOMES} for f in C.BUDGET_FRACS}

    return {"labels": {p: POLICY_LABELS[p] for p in ["random", *specs]}, "best_average_arm": {o: C.ARMS[a_] for o, a_ in best_arm.items()},
            "budget_grid": BUDGET_GRID, "curves": curves, "table_per_1000": table, "comparisons": compare,
            "iroas": iroas, "capture_pct_of_full_campaign_visits": capture, "full_coverage_per_1000": full_tbl, "full_coverage_comparisons": full_cmp,
            "seeds": seed_summary, "cost_per_contact": C.COST_PER_CONTACT}


# ---------------------------------------------------------------- main
def _headline(lt: dict, sel: dict, het: dict, seg: dict, clu: dict, mm: dict, pol: dict) -> dict:
    b = sel["visit"]
    row = lt["visit"][b]
    hl = {
        "selected_learner_visit": b,
        "selection_rule": "highest validation-set normalized Qini (visit) among 5 CATE learners",
        "visit_ate_test": het["ate"]["visit"],
        "norm_qini_visit": row["norm_qini"], "norm_qini_visit_seeds": row["seeds_norm_qini"],
        "norm_qini_visit_response": lt["visit"][L.RESPONSE]["norm_qini"],
        "auuc_gain_per_1000_visit": row["auuc_gain_per_1000"],
        "auuc_gain_per_1000_visit_seeds": row["seeds_auuc_gain_per_1000"],
        "auuc_gain_per_1000_visit_response": lt["visit"][L.RESPONSE]["auuc_gain_per_1000"],
        "margin_over_response_auuc_gain_per_1000": row["vs_response_auuc_gain_per_1000"],
        "margin_over_response_norm_qini": row["vs_response_norm_qini"],
        "blp_visit": {k: het["blp"]["visit"][b][k] for k in ("beta2", "beta2_ci_low", "beta2_ci_high", "beta2_p_one_sided")},
        "segments_top": {"rule": seg["rules"][0]["rule"], "ate_visit": seg["rules"][0]["ate_visit"]},
        "segments_bottom": {"rule": seg["rules"][-1]["rule"], "ate_visit": seg["rules"][-1]["ate_visit"]},
        "segments_heterogeneity_p_visit": seg["heterogeneity_test"]["visit"]["p_value"],
        "clusters_k": clu["k"], "clusters_heterogeneity_p_visit": clu["heterogeneity_test"]["visit"]["p_value"],
        "share_prefer_womens_email": mm["share_prefer"]["Womens E-Mail"],
        "matched_email_gain_visit": mm["matched_email_gain_model"]["overall"],
        "budget": {},
    }
    for fk in ("0.20", "0.30"):
        hl["budget"][fk] = {
            p: pol["table_per_1000"][fk][p] for p in ("random", "response", "uplift_binary", "uplift_mens", "uplift_multiarm")}
        for u in ("uplift_binary", "uplift_mens", "uplift_multiarm"):
            hl["budget"][fk][f"{u}_vs_response"] = pol["comparisons"][fk][f"{u}_vs_response"]
            hl["budget"][fk][f"{u}_vs_response_seeds"] = pol["seeds"][f"{u}_minus_response"][fk]
        hl["budget"][fk]["iroas_uplift_binary"] = pol["iroas"][fk]["uplift_binary"]["iroas"]
        hl["budget"][fk]["iroas_response"] = pol["iroas"][fk]["response"]["iroas"]
    return hl


def run() -> None:
    t0 = time.time()
    io.style()
    with threadpool_limits(limits=L.N_JOBS):
        tr_df, va_df, te_df = data.split_frames()
        folds = {o: tuple(_fold(d, o) for d in (tr_df, va_df, te_df)) for o in OUTCOMES}
        # best average arm on TRAIN (used by the non-arm-aware policies)
        best_arm = {o: int(max((1, 2), key=lambda k: tr_df.loc[tr_df["arm"] == k, o].mean())) for o in OUTCOMES}
        W = M.boot_weights(len(te_df))
        preds, fits, multi = _fit_all(folds, t0)
        lt, curves, sel = _learner_eval(folds, preds, W)
        _log(f"learner evaluation done; val-selected: {sel}", t0)
        het = _heterogeneity(folds, preds, sel)
        seg = _segments(folds, fits, preds, tr_df)
        clu = _clusters(tr_df, te_df, folds)
        mm = _multi_arm(folds, multi, preds[C.SEED]["visit"]["mu0"]["test"], te_df)
        _log("heterogeneity, segments, clusters, multi-arm done", t0)
        pol = _policies(folds, preds, multi, sel["visit"], best_arm, mm, W)
        _log("policy evaluation done", t0)

        figs = P.make_all(lt=lt, curves=curves, sel=sel, het=het, seg=seg, clu=clu, mm=mm, pol=pol)
        runtime = time.time() - t0
        res = {
            "config": {"seed": C.SEED, "seeds": C.SEEDS, "n_bootstrap": C.N_BOOTSTRAP, "budget_fracs": C.BUDGET_FRACS,
                       "cost_per_contact": C.COST_PER_CONTACT, "n_train": len(tr_df), "n_val": len(va_df),
                       "n_test": len(te_df), "features": data.FEATURES,
                       "propensity": "DummyClassifier(prior) inside econml = design probability (2/3 any-email; 1/3 per arm)"},
            "headline": _headline(lt, sel, het, seg, clu, mm, pol),
            "selected_learner": sel, "learners": lt, "heterogeneity": het, "segments": seg, "clusters": clu,
            "multi_arm": {k: v for k, v in mm.items() if not k.startswith("_")}, "policies": pol,
            "figures": figs, "runtime_sec": runtime,
        }
        io.save_json(res, "uplift")
        # render from the JSON round-trip so the .md and .json can never disagree
        (C.RESULTS / "uplift.md").write_text(R.markdown(io.load_json("uplift")))
    print(f"  wrote results/uplift.json, results/uplift.md, {len(figs)} figures in {runtime:.0f}s", flush=True)
