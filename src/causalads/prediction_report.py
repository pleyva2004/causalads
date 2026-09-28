"""Figures and README section for the prediction stage."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from causalads import config as C
from causalads import io

MODELS = ["lr", "rf", "xgb", "lgbm"]
LABEL = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
COL = dict(zip(MODELS, io.SERIES))
STRAT_LABEL = {"none": "No reweighting", "weight": "Class weight", "undersample": "Undersample 10:1"}
ST_COL = dict(zip(["none", "weight", "undersample"], io.SERIES))
NAMES = {"recency": "recency", "history": "history ($)", "log_history": "log history", "history_bucket": "history bucket",
         "history_at_floor": "history at $29.99 floor", "mens": "bought men's", "womens": "bought women's",
         "both_lines": "bought both lines", "newbie": "new customer", "recent_buyer": "recent buyer",
         "zip_urban": "zip urban", "zip_suburban": "zip suburban", "zip_rural": "zip rural", "ch_phone": "channel phone",
         "ch_web": "channel web", "ch_multi": "channel multi", "mens_email": "Men's e-mail sent",
         "womens_email": "Women's e-mail sent"}


def _ci(d: dict, k: int = 3, pct: bool = False) -> str:
    f = (lambda v: f"{100 * v:.{k - 1}f}%") if pct else (lambda v: f"{v:.{k}f}")
    return f"{f(d['est'])} [{f(d['ci'][0])}, {f(d['ci'][1])}]"


def figures(R: dict, A: dict, te: pd.DataFrame) -> None:
    c, a = R["conversion"], A["conversion"]
    best, bt = c["best"], c["best_tree"]
    yte, pt = a["yte"], a["pt"]
    rng = np.random.default_rng(C.SEED)

    # 1. repeated-CV spread
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, lab in zip(axs, ("roc_auc", "pr_auc"), ("ROC-AUC", "PR-AUC")):
        for k, m in enumerate(MODELS):
            v = np.array(c["cv_repeated"][m][key])
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=16, alpha=0.55, color=COL[m], label=LABEL[m])
            ax.errorbar(k + 0.3, v.mean(), v.std(ddof=1), fmt="o", color=io.TEXT, capsize=3, ms=4)
        ax.set_xticks(range(4), [LABEL[m].replace(" ", "\n") for m in MODELS])
        ax.set_ylabel(f"{lab} per held-out fold")
        ax.set_title(f"{lab}: 5x3 repeated CV, 15 folds (black = mean ± SD)")
    axs[0].legend(fontsize=8, loc="lower right")
    io.save_fig(fig, "prediction_cv_spread")

    # 2. imbalance strategies (CV on train)
    cv = c["imbalance"]["cv"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, lab in zip(axs, ("pr_auc", "brier", "mean_pred_over_base"),
                            ("PR-AUC", "Brier score", "Mean predicted prob. / true base rate")):
        for j, s in enumerate(STRAT_LABEL):
            ax.bar(np.arange(4) + (j - 1) * 0.27, [cv[f"{m}|{s}"][key]["mean"] for m in MODELS], 0.27,
                   yerr=[cv[f"{m}|{s}"][key]["std"] for m in MODELS], capsize=2, color=ST_COL[s], label=STRAT_LABEL[s])
        ax.set_xticks(range(4), [LABEL[m].replace(" ", "\n") for m in MODELS])
        ax.set_ylabel(lab)
    axs[0].set_title("Discrimination (5-fold CV, train; bars ± SD)")
    axs[1].set_title("Brier score (lower is better)")
    axs[2].set_title("Probability inflation (log scale; 1 = calibrated)")
    axs[2].set_yscale("log")
    axs[2].axhline(1, color=io.REFERENCE, lw=1.2)
    axs[0].legend(fontsize=8)
    io.save_fig(fig, "prediction_imbalance")

    # 3. calibration
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for v, lab, col in zip(("raw", "platt", "isotonic"), ("Uncalibrated", "Platt (fit on val)", "Isotonic (fit on val)"), io.SERIES):
        px, py = _rel(yte, a["cal_variants"][best][v])
        axs[0].plot(px * 100, py * 100, "o-", color=col, label=lab, ms=4)
    for s in ("none", "weight", "undersample"):
        px, py = _rel(a["ytr"], a["oof"][(bt, s)])
        axs[1].plot(px * 100, py * 100, "o-", color=ST_COL[s], label=STRAT_LABEL[s], ms=4)
    for ax, t in zip(axs, (f"{LABEL[best]} on test: before/after calibration", f"{LABEL[bt]} out-of-fold (train): imbalance handling")):
        lo, hi = 0.1, 100
        ax.plot([lo, hi], [lo, hi], color=io.REFERENCE, lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_ylim(0.1, 100)
        ax.set_xlabel("Mean predicted probability (%, equal-mass decile bins)")
        ax.set_ylabel("Observed conversion rate (%)")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    io.save_fig(fig, "prediction_calibration")

    # 4. ROC + PR
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in MODELS:
        f, t, _ = roc_curve(yte, pt[m])
        axs[0].plot(f, t, color=COL[m], label=f"{LABEL[m]} ({c['test']['models'][m]['roc_auc']['est']:.3f})", lw=1.6)
        p, r, _ = precision_recall_curve(yte, pt[m])
        axs[1].plot(r, p * 100, color=COL[m], label=LABEL[m], lw=1.6)
    axs[0].plot([0, 1], [0, 1], color=io.REFERENCE, ls="--", lw=1.2)
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, test (AUC in legend)")
    axs[1].axhline(100 * yte.mean(), color=io.REFERENCE, ls="--", lw=1.2, label=f"Base rate {100 * yte.mean():.2f}%")
    axs[1].set(xlabel="Recall", ylabel="Precision (%)", title=f"Precision-recall, test ({int(yte.sum())} conversions)")
    # Precision hits 100% at recall ~0 (the single top-scored customer); clip so the curves are readable.
    axs[1].set_ylim(0, 8)
    for ax in axs:
        ax.legend(fontsize=8)
    io.save_fig(fig, "prediction_roc_pr")

    # 5. lift + cumulative gains
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    d = c["deciles"]["best"]
    x = np.arange(1, 11)
    lift = np.array([r["lift"] for r in d])
    err = np.array([[r["lift"] - r["lift_ci"][0], r["lift_ci"][1] - r["lift"]] for r in d]).T
    axs[0].bar(x, lift, yerr=err, capsize=3, color=io.SERIES[0])
    axs[0].axhline(1, color=io.REFERENCE, lw=1.2)
    axs[0].set(xlabel="Score decile (1 = highest predicted)", ylabel="Lift vs. base rate (x)",
               title=f"{LABEL[best]}: lift by decile, test (95% CI)")
    axs[0].set_xticks(x)
    for key, lab, col in (("best", LABEL[best], io.SERIES[0]), ("lr", LABEL["lr"], io.SERIES[1])):
        g = [0] + [r["cum_gain"] * 100 for r in c["deciles"][key]]
        if key == "lr" and best == "lr":
            continue
        axs[1].plot(np.arange(0, 101, 10), g, "o-", color=col, label=lab, ms=4)
    axs[1].plot([0, 100], [0, 100], color=io.REFERENCE, ls="--", lw=1.2, label="Random targeting")
    axs[1].set(xlabel="Customers contacted (% of test, by score)", ylabel="Conversions captured (%)",
               title="Cumulative gains, test")
    axs[1].legend(fontsize=8)
    io.save_fig(fig, "prediction_gains")

    # 6. SHAP beeswarm + 7. importance
    import shap
    sv, samp = a["shap"]
    plt.figure(figsize=(7, 5))
    shap.summary_plot(sv, samp.rename(columns=NAMES), show=False, max_display=12, plot_size=None)
    plt.title(f"SHAP values, {LABEL[bt]} (val sample, n={len(samp)})")
    plt.xlabel("SHAP value (impact on model output, log-odds of conversion)")
    io.save_fig(plt.gcf(), "prediction_shap_beeswarm")

    at = c["attribution"]
    cols = list(at["shap_mean_abs"])
    order = sorted(cols, key=lambda k: at["shap_mean_abs"][k])[-12:]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    axs[0].barh([NAMES[k] for k in order], [at["shap_mean_abs"][k] for k in order], color=io.SERIES[0])
    axs[0].set(xlabel="Mean |SHAP| (log-odds)", title="SHAP (train-fit model, val rows)")
    axs[1].barh([NAMES[k] for k in order], [at["perm_importance"][k]["mean"] for k in order],
                xerr=[at["perm_importance"][k]["std"] for k in order], capsize=2, color=io.SERIES[1])
    axs[1].axvline(0, color=io.REFERENCE, lw=1.2)
    axs[1].set(xlabel="Drop in PR-AUC when shuffled (mean ± SD, 10 repeats)", title="Permutation importance (val)")
    io.save_fig(fig, "prediction_importance")

    # 8. subgroup calibration
    sg = pd.DataFrame(c["subgroups"])
    sg["label"] = sg["variable"] + ": " + sg["group"]
    y = np.arange(len(sg))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(sg) + 1.5))
    for off, key, lab, col in ((-0.17, "observed", "Observed (95% CI)", io.SERIES[0]), (0.17, "predicted", "Mean predicted (95% CI)", io.SERIES[1])):
        v = sg[key] * 100
        lo = v - sg[f"{key}_ci"].str[0] * 100
        hi = sg[f"{key}_ci"].str[1] * 100 - v
        ax.errorbar(v, y + off, xerr=[lo, hi], fmt="o", ms=4, capsize=2, color=col, label=lab)
    ax.axvline(100 * yte.mean(), color=io.REFERENCE, ls="--", lw=1.2, label="Overall base rate")
    ax.set_yticks(y, sg["label"])
    ax.set(xlabel="Conversion rate (%)", title=f"{LABEL[best]}: calibration by subgroup, test")
    ax.legend(fontsize=8, loc="lower right")
    io.save_fig(fig, "prediction_subgroups")


def _rel(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from causalads.prediction import reliability
    px, py = reliability(y, p)
    keep = (px > 1e-3) & (py > 0)  # zero-conversion bins cannot be drawn on a log axis
    return px[keep], py[keep]


def markdown(R: dict) -> str:
    c, v = R["conversion"], R["visit"]
    best, bt = c["best"], c["best_tree"]
    n = R["n"]
    L: list[str] = []
    add = L.append
    add("## Prediction: who is likely to convert?")
    add("")
    add(f"Supervised models score each customer's probability of converting (primary target, base rate "
        f"{100 * c['base_rate']['train']:.2f}% in train) or visiting (secondary, {100 * v['base_rate']['train']:.1f}%). "
        f"Features are the {len(R['features']) - 2} pre-treatment features plus two exposure indicators (Men's / Women's e-mail sent), "
        f"so attribution can show the e-mail effect. Splits are the fixed stratified train/val/test = {n['train']:,}/{n['val']:,}/{n['test']:,}. "
        "Models are sklearn Pipelines (median imputer, plus a standard scaler for logistic regression) so no preprocessing leaks across folds. "
        "Hyperparameters were tuned with Optuna (TPE, seed 42; logistic-regression C by grid) on **train only** using stratified 5-fold CV and PR-AUC. "
        f"The 'best' model is chosen by **validation** PR-AUC (**{LABEL[best]}**); calibrators are fit on validation predictions; "
        f"the test split is scored once. All test intervals are percentile bootstrap 95% CIs ({C.N_BOOTSTRAP} resamples of test rows). "
        "This is a *predictive* stage: high scores identify people who convert, not people the e-mail causes to convert (see the uplift stage).")
    add("")
    add("### Tuned configuration (train CV PR-AUC)")
    add("")
    add("| Model | CV PR-AUC (tuned, conversion) | Best hyperparameters (conversion) |")
    add("|---|---|---|")
    for m in MODELS:
        t = c["tuned"][m]
        ps = ", ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in t["params"].items())
        add(f"| {LABEL[m]} | {t['cv_pr_auc']:.4f} ({t['n_trials']} {'grid points' if m == 'lr' else 'trials'}) | {ps} |")
    add("")
    for tgt, T, title in (("conversion", c, "Conversion (primary)"), ("visit", v, "Visit (secondary)")):
        add(f"### Test performance: {title}")
        add("")
        add(f"Test base rate {100 * T['base_rate']['test']:.2f}% ({T['base_rate']['test_positives']} positives out of {n['test']:,}). "
            "A random scorer has PR-AUC equal to the base rate.")
        add("")
        add("| Model | ROC-AUC | PR-AUC | Brier | Log loss | Top-10% lift (x) |")
        add("|---|---|---|---|---|---|")
        for m in MODELS:
            r = T["test"]["models"][m]
            add(f"| {LABEL[m]}{' *' if m == T['best'] else ''} | {_ci(r['roc_auc'])} | {_ci(r['pr_auc'], 4)} | {_ci(r['brier'], 4)} | "
                f"{_ci(r['log_loss'], 4)} | {_ci(r['lift_top10'], 2)} |")
        add("")
        if "diff_roc_auc" in T["test"]:
            d1, d2 = T["test"]["diff_roc_auc"], T["test"]["diff_pr_auc"]
            add(f"\\* best on validation PR-AUC. Paired bootstrap, {LABEL[T['best']]} minus Logistic Regression: "
                f"ΔROC-AUC = {_ci(d1)} (two-sided p = {d1['p_two_sided']:.3f}); ΔPR-AUC = {_ci(d2, 4)} (p = {d2['p_two_sided']:.3f}).")
        else:
            add("\\* best on validation PR-AUC (this is the logistic-regression baseline, so there is no paired difference to test).")
        add("")
        add("Repeated stratified 5x3 CV on train+val (same tuned configs), mean ± SD over 15 folds:")
        add("")
        add("| Model | ROC-AUC | PR-AUC |")
        add("|---|---|---|")
        for m in MODELS:
            r = T["cv_repeated"][m]
            add(f"| {LABEL[m]} | {r['roc_mean']:.3f} ± {r['roc_std']:.3f} | {r['pr_mean']:.4f} ± {r['pr_std']:.4f} |")
        add("")
        if tgt == "conversion":
            add("![CV spread](results/figures/prediction_cv_spread.png)")
            add("")
            add("![ROC and PR curves](results/figures/prediction_roc_pr.png)")
            add("")

    add("### Class imbalance: reweighting does not improve ranking and wrecks probabilities")
    add("")
    add("Same tuned hyperparameters, three treatments of the ~0.9% positive class, 5-fold CV on train (mean ± SD across folds). "
        "The last column is mean predicted probability divided by the true base rate: 1.0 means the average score is an honest probability.")
    add("")
    add("| Model | Strategy | ROC-AUC | PR-AUC | Brier | ECE | Mean pred / base rate |")
    add("|---|---|---|---|---|---|---|")
    im = c["imbalance"]["cv"]
    for m in MODELS:
        for s in STRAT_LABEL:
            r = im[f"{m}|{s}"]
            add(f"| {LABEL[m]} | {STRAT_LABEL[s]} | {r['roc_auc']['mean']:.3f} ± {r['roc_auc']['std']:.3f} | "
                f"{r['pr_auc']['mean']:.4f} ± {r['pr_auc']['std']:.4f} | {r['brier']['mean']:.4f} ± {r['brier']['std']:.4f} | "
                f"{r['ece']['mean']:.4f} ± {r['ece']['std']:.4f} | {r['mean_pred_over_base']['mean']:.2f} ± {r['mean_pred_over_base']['std']:.2f} |")
    add("")
    add("![Imbalance strategies](results/figures/prediction_imbalance.png)")
    add("")

    add("### Calibration (conversion, test)")
    add("")
    add(f"Calibrators are fit on the validation predictions of models trained on train, then scored on test. ECE uses 10 equal-mass bins "
        f"(about {int(0.1 * n['test'])} customers, ~{int(round(0.1 * c['base_rate']['test_positives']))} conversions per bin), so it is noisy and biased upward for small samples (bootstrap ECE CIs can therefore sit above the point estimate). Brier differences between variants are far inside the CIs; ECE differences are not resolvable either.")
    add("")
    add("| Model | Calibration | Brier | ECE | Mean pred (base rate " + f"{100 * c['calibration']['base_rate_test']:.2f}%) |")
    add("|---|---|---|---|---|")
    for m in MODELS:
        for var in ("raw", "platt", "isotonic"):
            r = c["calibration"]["results"][m][var]
            add(f"| {LABEL[m]} | {var.capitalize() if var != 'raw' else 'None'} | {_ci(r['brier'], 5)} | {_ci(r['ece'], 4)} | {100 * r['mean_pred']:.2f}% |")
    add("")
    add("![Calibration](results/figures/prediction_calibration.png)")
    add("")

    add(f"### Targeting: {LABEL[best]}, test")
    add("")
    cf = c["confusion"]
    add(f"Decision rule (fixed before looking at test): {cf['rule']}; threshold = {cf['threshold']:.4f}. It flags {100 * cf['flagged_share']:.1f}% of test customers.")
    add("")
    add("| | Converted | Did not convert |")
    add("|---|---|---|")
    add(f"| Flagged | {cf['tp']} (TP) | {cf['fp']:,} (FP) |")
    add(f"| Not flagged | {cf['fn']} (FN) | {cf['tn']:,} (TN) |")
    add("")
    add(f"Recall {_ci(cf['recall'], 3, True)}; precision {_ci(cf['precision'], 3, True)} (base rate {100 * c['base_rate']['test']:.2f}%).")
    add("")
    add("| Decile | Conversion rate (95% CI) | Lift (95% CI) | Conversions | Cumulative share of conversions |")
    add("|---|---|---|---|---|")
    for r in c["deciles"]["best"]:
        add(f"| {r['decile']} | {100 * r['rate']:.2f}% [{100 * r['rate_ci'][0]:.2f}, {100 * r['rate_ci'][1]:.2f}] | "
            f"{r['lift']:.2f} [{r['lift_ci'][0]:.2f}, {r['lift_ci'][1]:.2f}] | {r['conversions']} | {100 * r['cum_gain']:.0f}% |")
    add("")
    add("![Lift and gains](results/figures/prediction_gains.png)")
    add("")

    at = c["attribution"]
    add(f"### Attribution ({LABEL[bt]}, best tree model)")
    add("")
    top = sorted(at["shap_mean_abs"], key=lambda k: -at["shap_mean_abs"][k])[:8]
    add("| Feature | Mean abs SHAP (log-odds) | Permutation importance, PR-AUC drop (mean ± SD) |")
    add("|---|---|---|")
    for k in top:
        p = at["perm_importance"][k]
        add(f"| {NAMES[k]} | {at['shap_mean_abs'][k]:.4f} | {p['mean']:.5f} ± {p['std']:.5f} |")
    add("")
    sp = at["spearman_shap_vs_perm"]
    agree = len(set(at["top3_shap"]) & set(at["top3_perm"]))
    add(f"Top-3 by SHAP: {', '.join(NAMES[k] for k in at['top3_shap'])}. Top-3 by permutation: {', '.join(NAMES[k] for k in at['top3_perm'])} "
        f"({agree}/3 overlap). Spearman correlation of the two rankings across all {len(at['shap_mean_abs'])} features: {sp['rho']:.2f} (p = {sp['p']:.3f}). "
        f"SHAP used {at['n_shap_rows']:,} validation rows; permutation importance used all {n['val']:,} validation rows, which hold only "
        f"~{int(round(c['base_rate']['test'] * n['val']))} conversions, so its SDs are large. Correlated features (history / log history / bucket, zip dummies, "
        "channel dummies) split credit, so single-feature rankings understate the importance of the underlying concept.")
    add("")
    add("![SHAP beeswarm](results/figures/prediction_shap_beeswarm.png)")
    add("")
    add("![Importance cross-check](results/figures/prediction_importance.png)")
    add("")

    add("### Error analysis (conversion, test)")
    add("")
    add("| Subgroup | n | Conversions | Observed rate (95% CI) | Mean predicted (95% CI) | ROC-AUC (95% CI; shown if >=10 conversions) |")
    add("|---|---|---|---|---|---|")
    for r in c["subgroups"]:
        auc = f"{r['roc_auc']:.3f} [{r['roc_auc_ci'][0]:.3f}, {r['roc_auc_ci'][1]:.3f}]" if r["roc_auc"] is not None else "n/a"
        add(f"| {r['variable']}: {r['group']} | {r['n']:,} | {r['conversions']} | {100 * r['observed']:.2f}% [{100 * r['observed_ci'][0]:.2f}, {100 * r['observed_ci'][1]:.2f}] | "
            f"{100 * r['predicted']:.2f}% [{100 * r['predicted_ci'][0]:.2f}, {100 * r['predicted_ci'][1]:.2f}] | {auc} |")
    add("")
    add("![Subgroup calibration](results/figures/prediction_subgroups.png)")
    add("")
    fn = c["false_negatives"]
    add(f"**Who the model misses.** At the top-20% rule, {fn['n_missed']} of {fn['n_caught'] + fn['n_missed']} test converters are not flagged. "
        f"Mean features of caught vs missed converters (median spend: ${fn['median_spend']['caught']:.2f} caught vs ${fn['median_spend']['missed']:.2f} missed):")
    add("")
    add("| Feature | Caught converters | Missed converters |")
    add("|---|---|---|")
    for k, r in fn["means"].items():
        add(f"| {NAMES.get(k, k if k != 'no_email' else 'No e-mail sent')} | {r['caught']:.3f} | {r['missed']:.3f} |")
    add("")

    h = R["headline"]
    add("### Caveats")
    add("")
    add(f"- Only {c['base_rate']['test_positives']} test conversions: every conversion CI above is wide, and most model-to-model differences are within noise. "
        f"Visits ({v['base_rate']['test_positives']} positives) are much better powered.")
    add("- Model selection used validation PR-AUC on ~115 validation conversions, so the 'best' model is partly a coin flip among near-equals; the paired bootstrap on test is the honest comparison.")
    add("- Calibrators (especially isotonic) are fit on ~115 validation conversions and can overfit; test ECE/Brier show whether they helped.")
    add("- Scores mix baseline propensity and e-mail response. Exposure is a feature, so 'likely to convert' is not 'persuadable'; targeting by these scores is not causal targeting.")
    add("- 2 weeks of outcomes, one 2008 retailer, history floor-censored at $29.99 for 12.4% of customers; hyperparameter tuning saw only train folds, but the same test set is shared across pipeline stages by design (never used for selection here).")
    add("- SHAP explains the model, not the world: attribution to correlated features is arbitrary between them.")
    add("")
    return "\n".join(L) + "\n"
