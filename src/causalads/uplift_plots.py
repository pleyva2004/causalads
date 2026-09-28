"""Figures for the uplift stage. Colors follow the entity (learner / policy) across charts."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from causalads import config as C
from causalads import io
from causalads import uplift_learners as L
from causalads.uplift_metrics import GRID

OUTCOME_TITLE = {"visit": "Visit", "conversion": "Conversion", "spend": "Spend"}
PER_1000 = {"visit": "incremental visits per 1,000 customers", "conversion": "incremental conversions per 1,000 customers",
            "spend": "incremental revenue per 1,000 customers ($)"}
EFFECT_UNIT = {"visit": "visit-rate uplift (pp)", "conversion": "conversion-rate uplift (pp)", "spend": "spend uplift ($ per customer)"}
SCALE = {"visit": 100, "conversion": 100, "spend": 1}  # pp for rates, $ for spend
COLOR = {L.RESPONSE: io.SERIES[0], "S-learner": io.SERIES[1], "T-learner": io.SERIES[2],
         "X-learner": io.SERIES[3], "DR-learner": io.SERIES[4], "Causal forest": io.SERIES[5]}


def _policy_color(p: str, sel: str) -> str:
    return {"random": io.REFERENCE, "response": COLOR[L.RESPONSE], "uplift_binary": COLOR[sel],
            "uplift_multiarm": io.SERIES[6], "uplift_mens": io.SERIES[7]}[p]


def uplift_curves(curves: dict, sel: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    x = GRID * 100
    for ax, (o, cv) in zip(axes, curves.items()):
        for k in [L.RESPONSE, *L.LEARNERS]:
            c = cv[k]
            lw = 2.4 if k == sel[o] else 1.4
            ax.plot(x, c["gain"] * 1000, color=COLOR[k], lw=lw,
                    label=f"{k} (val-selected)" if k == sel[o] else k)
            if k == sel[o]:
                ax.fill_between(x, c["lo"] * 1000, c["hi"] * 1000, color=COLOR[k], alpha=0.15, lw=0)
        end = cv[L.RESPONSE]["gain"][-1] * 1000
        ax.plot([0, 100], [0, end], color=io.REFERENCE, ls="--", lw=1.4, label="Random targeting")
        ax.set_title(f"{OUTCOME_TITLE[o]}: cumulative uplift on test")
        ax.set_xlabel("customers emailed, ranked by score (%)")
        ax.set_ylabel(PER_1000[o])
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Uplift (gain) curves, 95% bootstrap band for the val-selected learner", y=1.02, fontsize=11)
    return str(io.save_fig(fig, "uplift_curves").name)


def learner_forest(lt: dict, sel: dict) -> str:
    names = [*L.LEARNERS, L.RESPONSE]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (o, rows) in zip(axes, lt.items()):
        for i, k in enumerate(names):
            r = rows[k]
            est, lo, hi = r["norm_qini"]["estimate"], r["norm_qini"]["ci_low"], r["norm_qini"]["ci_high"]
            ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=COLOR[k], ms=7, capsize=3, lw=1.6,
                        label=f"Seed {C.SEED} estimate, 95% bootstrap CI" if i == 0 else None)
            seeds = np.asarray(r["seeds_norm_qini"]["values"])
            ax.scatter(seeds, np.full(len(seeds), i + 0.25), marker="|", s=60, color=io.TEXT_MUTED,
                       label="Refits over 5 other seeds" if i == 0 else None)
            if k == sel[o]:  # selection differs by outcome, so mark it per panel
                ax.annotate("val-selected", (est, i), xytext=(0, 8), textcoords="offset points", ha="center",
                            fontsize=7.5, color=io.TEXT_MUTED)
        ax.axvline(0, color=io.REFERENCE, lw=1.2, ls="--", label="Random targeting")
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [])
        ax.set_ylim(len(names) - 0.5, -0.7)
        ax.set_title(f"{OUTCOME_TITLE[o]}")
        ax.set_xlabel("normalized Qini coefficient (test)")
    fig.tight_layout()
    # figure-level legend: tight_layout ignores it, so the axes keep their width
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8.5)
    fig.suptitle("Uplift ranking quality by learner (test set; learner used for targeting selected on validation Qini)",
                 y=1.03, fontsize=11)
    return str(io.save_fig(fig, "uplift_learners").name)


def _grouped_effects(ax, groups_by_model: dict, o: str, ate: float) -> None:
    ks = list(groups_by_model)
    offs = np.linspace(-0.18, 0.18, len(ks))
    for off, k in zip(offs, ks):
        g = groups_by_model[k]
        x = np.arange(1, len(g) + 1) + off
        est = np.array([r["estimate"] for r in g]) * SCALE[o]
        lo = np.array([r["ci_low"] for r in g]) * SCALE[o]
        hi = np.array([r["ci_high"] for r in g]) * SCALE[o]
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", color=COLOR[k], ms=6, capsize=2.5, lw=1.4,
                    label=f"ranked by {k}")
    ax.axhline(ate * SCALE[o], color=io.REFERENCE, ls="--", lw=1.2, label="overall ATE (= random targeting)")
    ax.axhline(0, color=io.REFERENCE, lw=0.8)
    ax.set_ylabel(f"observed {EFFECT_UNIT[o]}")


def deciles(het: dict, o: str = "visit") -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _grouped_effects(ax, het["deciles"][o], o, het["ate"][o]["estimate"])
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("predicted-score decile (1 = highest predicted)")
    ax.set_title(f"{OUTCOME_TITLE[o]}: observed treated - control by score decile (test, 95% CI)")
    ax.legend(fontsize=8)
    return str(io.save_fig(fig, f"uplift_deciles_{o}").name)


def gates(het: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, o in zip(axes, ["visit", "conversion", "spend"]):
        _grouped_effects(ax, {k: v["groups"] for k, v in het["gates"][o].items()}, o, het["ate"][o]["estimate"])
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("predicted-score quintile (1 = highest)")
        ax.set_title(f"{OUTCOME_TITLE[o]}: GATES by quintile")
        ax.legend(fontsize=7.5)
    return str(io.save_fig(fig, "uplift_gates").name)


def importance(seg: dict) -> str:
    g = dict(sorted(seg["group_importance"].items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(list(g), [v * 100 for v in g.values()], color=COLOR["Causal forest"], height=0.6)
    for i, v in enumerate(g.values()):
        ax.text(v * 100 + 0.5, i, f"{v * 100:.0f}%", va="center", fontsize=8.5, color=io.TEXT)
    ax.set_xlabel("share of causal-forest split importance (%)")
    ax.set_title("What drives email-effect heterogeneity (visit, causal forest)")
    ax.grid(axis="y", visible=False)
    return str(io.save_fig(fig, "uplift_cf_importance").name)


def _effect_rows(ax, labels: list[str], effects: list[dict], ate: float, color: str, pred: list[float] | None = None) -> None:
    y = np.arange(len(labels))
    est = np.array([e["estimate"] for e in effects]) * 100
    lo = np.array([e["ci_low"] for e in effects]) * 100
    hi = np.array([e["ci_high"] for e in effects]) * 100
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=color, ms=7, capsize=3, lw=1.6,
                label="observed ATE on test (95% CI)")
    if pred is not None:
        ax.scatter(np.array(pred) * 100, y, marker="D", facecolor="none", edgecolor=io.TEXT, s=36, zorder=3,
                   label="forest-predicted CATE (test mean)")
    ax.axvline(ate * 100, color=io.REFERENCE, ls="--", lw=1.2, label="overall ATE")
    ax.axvline(0, color=io.REFERENCE, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("visit-rate uplift from any email (pp)")
    ax.legend(fontsize=7.5, loc="lower right")


def segments(seg: dict, ate: float) -> str:
    rows = seg["rules"]
    labels = [textwrap.fill(f"{r['rule']}  ({r['share'] * 100:.0f}%)", 52).replace("$", r"\$") for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.6))
    _effect_rows(ax, labels, [r["ate_visit"] for r in rows], ate, COLOR["Causal forest"],
                 [r["predicted_cate_test"] for r in rows])
    ax.set_title("Segment rules from a depth-3 CATE tree (fit on val), scored on test")
    return str(io.save_fig(fig, "uplift_segments").name)


def clusters(clu: dict, ate: float) -> str:
    rows = clu["clusters"]
    labels = [textwrap.fill(f"C{r['cluster']}: {r['label']} ({r['share_train'] * 100:.0f}%)", 40) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.6))
    _effect_rows(ax, labels, [r["ate_visit"] for r in rows], ate, io.SERIES[0])
    ax.set_title(f"KMeans behavioral clusters (k={clu['k']}): visit uplift on test")
    return str(io.save_fig(fig, "uplift_clusters").name)


def arm_choice(mm: dict) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1, 1.1]})
    groups = list(mm["by_purchase_history"])
    w = np.array([mm["by_purchase_history"][g]["share_prefer_womens"] for g in groups]) * 100
    y = np.arange(len(groups))
    a1.barh(y, 100 - w, color=io.SERIES[0], height=0.55, label="Mens E-Mail")
    a1.barh(y, w, left=100 - w, color=io.SERIES[1], height=0.55, label="Womens E-Mail")
    for i, g in enumerate(groups):
        a1.text(1, i, f"{100 - w[i]:.0f}%", va="center", color="white", fontsize=8.5)
        a1.text(99, i, f"{w[i]:.0f}%", va="center", ha="right", color="white", fontsize=8.5)
    a1.set_yticks(y, [f"{g} (n={mm['by_purchase_history'][g]['n']:,})" for g in groups])
    a1.set_xlim(0, 100)
    a1.set_xlabel("share of test customers (%)")
    a1.set_title("Email with the higher predicted visit uplift, by past purchases")
    a1.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1, -0.42), ncol=2)
    a1.grid(axis="y", visible=False)

    items = [("Model's pick vs other email (all emailed)", mm["matched_email_gain_model"]["overall"]),
             ("Purchase-line rule vs other email", mm["matched_email_gain_rule"]["overall"])]
    for arm in ("Mens E-Mail", "Womens E-Mail"):
        if arm in mm["matched_email_gain_model"]:
            items.append((f"Model prefers {arm.split()[0]}: {arm.split()[0]} - other", mm["matched_email_gain_model"][arm]))
    est = np.array([e["estimate"] for _, e in items]) * 100
    lo = np.array([e["ci_low"] for _, e in items]) * 100
    hi = np.array([e["ci_high"] for _, e in items]) * 100
    yy = np.arange(len(items))
    a2.errorbar(est, yy, xerr=[est - lo, hi - est], fmt="o", color=io.SERIES[6], ms=7, capsize=3, lw=1.6)
    a2.axvline(0, color=io.REFERENCE, lw=1.0)
    a2.set_yticks(yy, [textwrap.fill(t, 34) for t, _ in items])
    a2.set_ylim(len(items) - 0.4, -0.8)
    a2.set_xlabel("visit-rate difference, chosen vs other email (pp, test, 95% CI)")
    a2.set_title("Does the chosen email actually do better? (randomized check)")
    return str(io.save_fig(fig, "uplift_arm_choice").name)


def budget(pol: dict, sel: str) -> str:
    x = np.asarray(pol["budget_grid"]) * 100
    # revenue-CATE policies are tabulated only: no spend-CATE signal and only 8 palette slots
    order = ["random", "response", "uplift_binary", "uplift_mens", "uplift_multiarm"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, o in zip(axes, ["visit", "conversion", "spend"]):
        for p in order:
            c = pol["curves"][p][o]
            col = _policy_color(p, sel)
            label = pol["labels"][p] + (f" [{sel}]" if p == "uplift_binary" else "")
            ax.plot(x, c["est"], color=col, lw=1.4 if p == "random" else 2.0, ls="--" if p == "random" else "-",
                    label=label)
            ax.fill_between(x, c["lo"], c["hi"], color=col, alpha=0.08 if p == "random" else 0.12, lw=0)
        for f in C.BUDGET_FRACS:
            ax.axvline(f * 100, color=io.GRID, lw=1.0, zorder=0)
        ax.axhline(0, color=io.REFERENCE, lw=0.8)
        ax.set_xlabel("contact budget (% of customers emailed)")
        ax.set_ylabel(PER_1000[o])
        ax.set_title(f"{OUTCOME_TITLE[o]}: IPW policy value on test")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8.5)
    fig.suptitle("Incremental outcome vs sending no email, by targeting policy (95% bootstrap bands)", y=1.02, fontsize=11)
    return str(io.save_fig(fig, "uplift_budget").name)


def make_all(*, lt: dict, curves: dict, sel: dict, het: dict, seg: dict, clu: dict, mm: dict, pol: dict) -> list[str]:
    ate = het["ate"]["visit"]["estimate"]
    return [uplift_curves(curves, sel), learner_forest(lt, sel), deciles(het), gates(het), importance(seg),
            segments(seg, ate), clusters(clu, ate), arm_choice(mm), budget(pol, sel["visit"])]
