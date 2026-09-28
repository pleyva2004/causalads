"""Markdown results section for the uplift stage. Every number is read from the results
dict (the JSON round-trip), never typed by hand."""
from __future__ import annotations

LEARNERS = ["S-learner", "T-learner", "X-learner", "DR-learner", "Causal forest"]
RESPONSE = "Response model"
FIG = "results/figures"


def _ci(c: dict | None, scale: float = 1.0, d: int = 1) -> str:
    if c is None:
        return "n/a"
    return f"{c['estimate'] * scale:.{d}f} [{c['ci_low'] * scale:.{d}f}, {c['ci_high'] * scale:.{d}f}]"


def _excl0(c: dict | None) -> bool:
    return c is not None and (c["ci_low"] > 0 or c["ci_high"] < 0)


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _verdict(c: dict, what: str) -> str:
    if not _excl0(c):
        return f"{what} is **not statistically significant** (95% CI includes 0)"
    return f"{what} is significantly {'positive' if c['estimate'] > 0 else 'negative'} (95% CI excludes 0)"


def _learner_section(res: dict) -> list[str]:
    lt, sel, blp = res["learners"], res["selected_learner"], res["heterogeneity"]["blp"]
    v = lt["visit"]
    out = [
        "### 1. Ranking quality: CATE learners vs a response model",
        "",
        "Outcome = visit, treatment = any email. Normalized Qini = area between the model's Qini curve and random, "
        "divided by the same area for a perfect ranking (0 = random). AUUC gain = average over targeting depths of "
        "extra incremental visits per 1,000 customers vs random targeting. BLP β2 is the Chernozhukov et al. test "
        "of whether the score carries real effect heterogeneity (β2 > 0; β2 = 1 = well calibrated). "
        "`*` = selected on validation.",
        "",
        "| Learner | Val norm. Qini | Test norm. Qini [95% CI] | 5-seed mean ± sd | AUUC gain / 1k [95% CI] "
        "| Δ AUUC gain vs response [95% CI] | BLP β2 [95% CI] | β2 one-sided p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in [*LEARNERS, RESPONSE]:
        r, b = v[k], blp["visit"][k]
        s = r["seeds_norm_qini"]
        delta = _ci(r["vs_response_auuc_gain_per_1000"], 1, 2) if k != RESPONSE else "—"
        out.append(f"| {k}{' *' if k == sel['visit'] else ''} | {r['val_norm_qini']:.3f} | {_ci(r['norm_qini'], 1, 3)} "
                   f"| {s['mean']:.3f} ± {s['std']:.3f} | {_ci(r['auuc_gain_per_1000'], 1, 2)} | {delta} "
                   f"| {b['beta2']:.2f} [{b['beta2_ci_low']:.2f}, {b['beta2_ci_high']:.2f}] | {_p(b['beta2_p_one_sided'])} |")
    n_sig = sum(blp["visit"][k]["beta2_p_one_sided"] < 0.05 for k in LEARNERS)
    beats = [k for k in LEARNERS if v[k]["vs_response_auuc_gain_per_1000"]["ci_low"] > 0]
    test_best = max(LEARNERS, key=lambda k: v[k]["norm_qini"]["estimate"])
    vals = [v[k]["val_norm_qini"] for k in LEARNERS]
    out += [
        "",
        f"- **Heterogeneity is real for visits:** {n_sig}/5 CATE learners have BLP β2 > 0 at one-sided p < 0.05 on the held-out test set.",
        "- **But uplift ranking does not beat response ranking:** "
        + (f"{', '.join(beats)} significantly beat the response model on AUUC gain."
           if beats else "no CATE learner's AUUC gain is significantly above the response model's (every paired 95% CI includes 0). "
           f"The response model's own normalized Qini is {_ci(v[RESPONSE]['norm_qini'], 1, 3)}"
           + (f" and its BLP β2 p = {_p(blp['visit'][RESPONSE]['beta2_p_one_sided'])}: customers most likely to visit when "
              "emailed are also, on average, the ones the email moves most."
              if blp["visit"][RESPONSE]["beta2_p_one_sided"] < 0.05 else ".")),
        f"- Validation-set Qini barely separates the learners (range {min(vals):.3f}–{max(vals):.3f}); the val-selected "
        f"learner is **{sel['visit']}**, while the post-hoc test-best is {test_best} "
        f"({v[test_best]['norm_qini']['estimate']:.3f}). Headline numbers use the val-selected learner.",
        "",
        "Conversion and spend (same learners, same protocol):",
        "",
        "| Learner | Conversion norm. Qini [95% CI] | Conversion BLP β2 p | Spend norm. Qini [95% CI] | Spend BLP β2 p |",
        "|---|---|---|---|---|",
    ]
    for k in [*LEARNERS, RESPONSE]:
        c, s = lt["conversion"][k], lt["spend"][k]
        out.append(f"| {k} | {_ci(c['norm_qini'], 1, 3)} | {_p(blp['conversion'][k]['beta2_p_one_sided'])} "
                   f"| {_ci(s['norm_qini'], 1, 3)} | {_p(blp['spend'][k]['beta2_p_one_sided'])} |")
    n_sig_c = sum(blp["conversion"][k]["beta2_p_one_sided"] < 0.05 for k in LEARNERS)
    n_sig_s = sum(blp["spend"][k]["beta2_p_one_sided"] < 0.05 for k in LEARNERS)
    ac = res["heterogeneity"]["ate"]["conversion"]
    out += [
        "",
        f"For conversion {n_sig_c}/5 and for spend {n_sig_s}/5 learners show significant heterogeneity. With "
        f"{ac['n_control'] * ac['mean_control']:.0f} control-group conversions in the test set, conversion and spend CATEs "
        "are not reliable and are not used for targeting claims.",
        "",
        f"![Uplift curves]({FIG}/uplift_curves.png)",
        "",
        f"![Normalized Qini by learner]({FIG}/uplift_learners.png)",
        "",
    ]
    return out


def _heterogeneity_section(res: dict) -> list[str]:
    het, sel = res["heterogeneity"], res["selected_learner"]["visit"]
    g = het["gates"]["visit"]
    b = het["blp"]["visit"][sel]
    out = [
        "### 2. Is the heterogeneity real? GATES and BLP",
        "",
        f"Test-set ATE of any email on visits: {_ci(het['ate']['visit'], 100, 2)} pp. Ranking test customers by the "
        f"{sel} CATE, BLP β2 = {b['beta2']:.2f} [{b['beta2_ci_low']:.2f}, {b['beta2_ci_high']:.2f}] "
        f"(one-sided p = {_p(b['beta2_p_one_sided'])}; across 5 refit seeds β2 = {b['seeds_beta2']['mean']:.2f} ± {b['seeds_beta2']['std']:.2f}). "
        "Group average treatment effects (observed treated − control within each predicted quintile; valid because "
        "assignment is randomized inside every group):",
        "",
        f"| Quintile (1 = highest score) | Ranked by {sel}: uplift pp [95% CI] | Ranked by response model: uplift pp [95% CI] |",
        "|---|---|---|",
    ]
    for i, (a, r) in enumerate(zip(g[sel]["groups"], g[RESPONSE]["groups"])):
        out.append(f"| {i + 1} | {_ci(a, 100, 1)} | {_ci(r, 100, 1)} |")
    tb, tbr = g[sel]["top_minus_bottom"], g[RESPONSE]["top_minus_bottom"]
    out += [
        f"| Top − bottom | {_ci(tb, 100, 1)} (p = {_p(tb['p_value'])}) | {_ci(tbr, 100, 1)} (p = {_p(tbr['p_value'])}) |",
        "",
        f"![Uplift by decile]({FIG}/uplift_deciles_visit.png)",
        "",
        f"![GATES by outcome]({FIG}/uplift_gates.png)",
        "",
    ]
    return out


def _segments_section(res: dict) -> list[str]:
    seg, clu = res["segments"], res["clusters"]
    gi = sorted(seg["group_importance"].items(), key=lambda kv: -kv[1])
    q, qc = seg["heterogeneity_test"], clu["heterogeneity_test"]
    out = [
        "### 3. Who responds? Interpretable segments",
        "",
        "Causal-forest heterogeneity importance (visit), grouped over redundant encodings: "
        + ", ".join(f"{k} {v * 100:.0f}%" for k, v in gi[:3]) + ".",
        "",
        f"![Causal forest importance]({FIG}/uplift_cf_importance.png)",
        "",
        f"A depth-{seg['tree_depth']} `SingleTreeCateInterpreter` tree was fit to the forest's CATEs on the **validation** set "
        f"(min {seg['min_samples_leaf_val']:,} customers per leaf); each leaf rule is then scored on the **test** set:",
        "",
        "| Segment rule | Share | Forest CATE (pp) | Observed visit uplift, pp [95% CI] | Observed spend uplift, $ [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in seg["rules"]:
        out.append(f"| {r['rule']} | {r['share'] * 100:.0f}% | {r['predicted_cate_test'] * 100:.1f} "
                   f"| {_ci(r['ate_visit'], 100, 1)} | {_ci(r['ate_spend'], 1, 2)} |")
    tmb = seg["top_minus_bottom_visit"]
    out += [
        "",
        f"Equal-effects test across segments (Cochran's Q): visit Q = {q['visit']['Q']:.1f}, df = {q['visit']['df']}, "
        f"p = {_p(q['visit']['p_value'])}; spend p = {_p(q['spend']['p_value'])}; conversion p = {_p(q['conversion']['p_value'])}. "
        f"Highest- minus lowest-predicted segment: {_ci(tmb, 100, 1)} pp.",
        "",
        f"![Segment rules]({FIG}/uplift_segments.png)",
        "",
        f"**Behavioral clusters.** KMeans on standardized {', '.join(clu['features'])} (fit on train); k = {clu['k']} maximizes "
        f"the train silhouette ({', '.join(f'k={k}: {v:.3f}' for k, v in clu['silhouette'].items())}).",
        "",
        "| Cluster | Profile | Share (train) | Recency (mo) | Past spend ($) | Visit uplift, pp [95% CI] | Spend uplift, $ [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in clu["clusters"]:
        p = r["profile"]
        out.append(f"| C{r['cluster']} | {r['label']} | {r['share_train'] * 100:.0f}% | {p['recency']:.1f} | {p['history']:.0f} "
                   f"| {_ci(r['ate_visit'], 100, 1)} | {_ci(r['ate_spend'], 1, 2)} |")
    out += [
        "",
        f"Equal-effects test across clusters: visit p = {_p(qc['visit']['p_value'])} (Q = {qc['visit']['Q']:.1f}, df = {qc['visit']['df']}), "
        f"spend p = {_p(qc['spend']['p_value'])}. "
        + ("Unsupervised clusters separate the email effect less sharply than the CATE-tree segments."
           if qc["visit"]["p_value"] > q["visit"]["p_value"] else
           "Clusters separate the email effect at least as sharply as the CATE-tree segments."),
        "",
        f"![Cluster effects]({FIG}/uplift_clusters.png)",
        "",
    ]
    return out


def _multiarm_section(res: dict) -> list[str]:
    mm, pol = res["multi_arm"], res["policies"]
    full, fc = pol["full_coverage_per_1000"], pol["full_coverage_comparisons"]
    sh = mm["share_prefer_womens_seeds"]
    gm, gr = mm["matched_email_gain_model"]["overall"], mm["matched_email_gain_rule"]["overall"]
    out = [
        "### 4. Which email? Multi-arm CATE",
        "",
        "One causal forest over the 3-level arm gives per-customer CATEs for Mens vs none and Womens vs none "
        f"(test means: {mm['mean_cate']['Mens E-Mail'] * 100:.1f} pp and {mm['mean_cate']['Womens E-Mail'] * 100:.1f} pp; "
        f"BLP β2 one-sided p = {_p(mm['blp']['Mens E-Mail']['beta2_p_one_sided'])} and "
        f"{_p(mm['blp']['Womens E-Mail']['beta2_p_one_sided'])}). "
        f"{mm['share_prefer']['Mens E-Mail'] * 100:.0f}% of test customers are predicted to respond more to the Mens email and "
        f"{mm['share_prefer']['Womens E-Mail'] * 100:.0f}% to the Womens email (Womens share over 5 refit seeds: "
        f"{sh['mean'] * 100:.0f}% ± {sh['std'] * 100:.1f}%).",
        "",
        "| Past purchases | Test customers | Predicted to prefer Womens email | Mean CATE Mens (pp) | Mean CATE Womens (pp) |",
        "|---|---|---|---|---|",
    ]
    for h, r in mm["by_purchase_history"].items():
        out.append(f"| {h} | {r['n']:,} | {r['share_prefer_womens'] * 100:.0f}% | {r['mean_cate_mens'] * 100:.1f} | {r['mean_cate_womens'] * 100:.1f} |")
    out += [
        "",
        f"The model's email choice agrees with a one-line rule (\"Womens email to women's-only buyers, Mens otherwise\") for "
        f"{mm['agreement_with_purchase_rule'] * 100:.0f}% of customers. Randomized check among emailed test customers: sending "
        f"the model's pick instead of the other email raises visits by {_ci(gm, 100, 1)} pp; the purchase rule achieves "
        f"{_ci(gr, 100, 1)} pp. Emailing everyone (IPW, incremental outcomes per 1,000 customers vs no email):",
        "",
        "| Policy (100% emailed) | Visits / 1k [95% CI] | Conversions / 1k [95% CI] | Revenue $ / 1k [95% CI] |",
        "|---|---|---|---|",
    ]
    names = {"all_mens": "Everyone gets Mens", "all_womens": "Everyone gets Womens",
             "purchase_rule": "Purchase-line rule", "multiarm_argmax": "Model's higher-CATE email"}
    for k, lab in names.items():
        r = full[k]
        out.append(f"| {lab} | {_ci(r['visit'])} | {_ci(r['conversion'], 1, 2)} | {_ci(r['spend'], 1, 0)} |")
    d1, d2 = fc["multiarm_argmax_vs_all_mens"]["visit"], fc["multiarm_argmax_vs_purchase_rule"]["visit"]
    out += [
        "",
        f"Model vs all-Mens: {_ci(d1)} visits / 1k (p = {_p(d1['p_value'])}); model vs purchase rule: {_ci(d2)} (p = {_p(d2['p_value'])}). "
        f"{_verdict(d2, 'The model’s advantage over the one-line rule')}.",
        "",
        f"![Email choice]({FIG}/uplift_arm_choice.png)",
        "",
    ]
    return out


def _budget_section(res: dict) -> list[str]:
    pol, cfg = res["policies"], res["config"]
    lab, t, cmp = pol["labels"], pol["table_per_1000"], pol["comparisons"]
    arm = pol["best_average_arm"]["visit"]
    out = [
        "### 5. Budget allocation: incremental outcomes under a fixed contact budget",
        "",
        f"Each policy emails a fixed share of test customers; policies that are not arm-aware send the email with the best "
        f"average visit rate on train ({arm}). Values are Horvitz–Thompson IPW estimates on test, "
        "V(π) − V(no email) = mean[Y·(1{A=π(X)} − 1{A=0}) / (1/3)], unbiased because each arm had known probability 1/3. "
        f"iROAS = incremental revenue / (contacts × ${cfg['cost_per_contact']:.2f}); the cost per email is an illustrative "
        "assumption, so break-even cost per contact is reported too. 95% bootstrap CIs (paired draws).",
        "",
        "| Budget | Policy | Visits / 1k | Conversions / 1k | Revenue $ / 1k | iROAS | Break-even $ / contact |",
        "|---|---|---|---|---|---|---|",
    ]
    for fk, rows in t.items():
        for p, r in rows.items():
            io_ = pol["iroas"][fk][p]
            out.append(f"| {float(fk) * 100:.0f}% | {lab[p]} | {_ci(r['visit'])} | {_ci(r['conversion'], 1, 2)} "
                       f"| {_ci(r['spend'], 1, 0)} | {_ci(io_['iroas'], 1, 1)} | {_ci(io_['break_even_cost_per_contact'], 1, 2)} |")
    out += [
        "",
        "Paired differences vs the response-model policy (uplift − response, same bootstrap draws; the last column is the "
        "mean ± sd of the visit difference over 5 refit seeds):",
        "",
        "| Budget | Uplift policy | Δ visits / 1k [95% CI] | % change in visits [95% CI] | Δ conversions / 1k [95% CI] "
        "| Δ revenue $ / 1k [95% CI] | Seeds: Δ visits / 1k |",
        "|---|---|---|---|---|---|---|",
    ]
    for fk in cmp:
        for u in ("uplift_binary", "uplift_mens", "uplift_multiarm"):
            c = cmp[fk][f"{u}_vs_response"]
            sd = pol["seeds"][f"{u}_minus_response"][fk]["visit"]
            out.append(f"| {float(fk) * 100:.0f}% | {lab[u]} | {_ci(c['visit']['diff_per_1000'])} "
                       f"| {_ci(c['visit']['pct_improvement'], 1, 0)} | {_ci(c['conversion']['diff_per_1000'], 1, 2)} "
                       f"| {_ci(c['spend']['diff_per_1000'], 1, 0)} | {sd['mean']:.1f} ± {sd['std']:.1f} |")
    rev = {fk: cmp[fk]["uplift_revenue_vs_response_spend"]["spend"]["diff_per_1000"] for fk in cmp}
    wins = [(fk, u) for fk in cmp for u in ("uplift_binary", "uplift_mens", "uplift_multiarm")
            if cmp[fk][f"{u}_vs_response"]["visit"]["diff_per_1000"]["ci_low"] > 0]
    losses = [(fk, u) for fk in cmp for u in ("uplift_binary", "uplift_mens", "uplift_multiarm")
              if cmp[fk][f"{u}_vs_response"]["visit"]["diff_per_1000"]["ci_high"] < 0]
    fmt = lambda cells: "; ".join(f"{lab[u]} at {float(fk) * 100:.0f}%" for fk, u in cells)  # noqa: E731
    out += [
        "",
        ("**No uplift policy significantly beats response targeting on visits at any budget.** "
         if not wins else f"Uplift beats response on visits (CI excludes 0): {fmt(wins)}. ")
        + (f"Response targeting is significantly better (CI excludes 0) for: {fmt(losses)}. " if losses else "")
        + "Revenue-optimized policy (multi-arm causal forest on spend) vs a spend response model, Δ revenue $ / 1k: "
        + "; ".join(f"{float(fk) * 100:.0f}%: {_ci(c, 1, 0)}" for fk, c in rev.items())
        + ". Spend CATEs carry no detectable signal (Section 1), so that policy is tabulated but not recommended.",
        "",
        f"![Budget curves]({FIG}/uplift_budget.png)",
        "",
    ]
    return out


def _headline_section(res: dict) -> list[str]:
    h = res["headline"]
    sel = h["selected_learner_visit"]
    b = h["blp_visit"]
    out = [
        "**Headline (test set, 95% CIs):**",
        "",
        f"- Val-selected learner **{sel}**: normalized Qini {_ci(h['norm_qini_visit'], 1, 3)} "
        f"(5-seed {h['norm_qini_visit_seeds']['mean']:.3f} ± {h['norm_qini_visit_seeds']['std']:.3f}) vs response model "
        f"{_ci(h['norm_qini_visit_response'], 1, 3)}; AUUC gain {_ci(h['auuc_gain_per_1000_visit'], 1, 2)} vs "
        f"{_ci(h['auuc_gain_per_1000_visit_response'], 1, 2)} extra visits / 1k. "
        + _verdict(h["margin_over_response_auuc_gain_per_1000"],
                   f"The margin over response targeting, {_ci(h['margin_over_response_auuc_gain_per_1000'], 1, 2)} visits / 1k,")
        + ".",
        f"- BLP heterogeneity test: β2 = {b['beta2']:.2f} [{b['beta2_ci_low']:.2f}, {b['beta2_ci_high']:.2f}], "
        f"one-sided p = {_p(b['beta2_p_one_sided'])} — "
        + ("the CATE ranking carries real heterogeneity." if b["beta2_p_one_sided"] < 0.05 else "no significant heterogeneity."),
        f"- Top segment \"{h['segments_top']['rule']}\": visit uplift {_ci(h['segments_top']['ate_visit'], 100, 1)} pp; "
        f"bottom segment \"{h['segments_bottom']['rule']}\": {_ci(h['segments_bottom']['ate_visit'], 100, 1)} pp "
        f"(equal-effects p = {_p(h['segments_heterogeneity_p_visit'])}).",
    ]
    for fk in ("0.20", "0.30"):
        bb = h["budget"][fk]
        c, cm = bb["uplift_binary_vs_response"], bb["uplift_multiarm_vs_response"]
        out.append(
            f"- **{float(fk) * 100:.0f}% budget, per 1,000 customers.** Response policy: {_ci(bb['response']['visit'])} visits, "
            f"{_ci(bb['response']['conversion'], 1, 2)} conversions, ${_ci(bb['response']['spend'], 1, 0)} revenue. "
            f"Binary-uplift policy: {_ci(bb['uplift_binary']['visit'])} visits, {_ci(bb['uplift_binary']['conversion'], 1, 2)} "
            f"conversions, ${_ci(bb['uplift_binary']['spend'], 1, 0)}. Uplift vs response: "
            f"{_ci(c['visit']['pct_improvement'], 1, 0)}% visits, {_ci(c['conversion']['pct_improvement'], 1, 0)}% conversions, "
            f"{_ci(c['spend']['pct_improvement'], 1, 0)}% revenue "
            f"({'CI includes 0' if not _excl0(c['visit']['pct_improvement']) else 'CI excludes 0'} for visits). "
            f"Multi-arm uplift vs response: {_ci(cm['visit']['pct_improvement'], 1, 0)}% visits "
            f"({'CI includes 0' if not _excl0(cm['visit']['pct_improvement']) else 'CI excludes 0'}).")
    return out + [""]


def _bottom_line(res: dict) -> list[str]:
    h, pol, mm = res["headline"], res["policies"], res["multi_arm"]
    cap = pol["capture_pct_of_full_campaign_visits"]["0.30"]
    c30 = h["budget"]["0.30"]["uplift_binary_vs_response"]["visit"]["diff_per_1000"]
    beats = c30["ci_low"] > 0
    rule = pol["full_coverage_comparisons"]["multiarm_argmax_vs_purchase_rule"]["visit"]
    return [
        "**Bottom line.** "
        f"Emailing the top 30% ranked by the response model captures {_ci(cap['response'], 1, 0)}% of the incremental visits "
        "of emailing everyone, using 30% of the sends (random targeting captures 30% by construction). "
        + (f"Uplift models find real heterogeneity (\"{h['segments_top']['rule']}\": "
           f"{_ci(h['segments_top']['ate_visit'], 100, 1)} pp vs {h['visit_ate_test']['estimate'] * 100:.1f} pp on average) "
           if h["blp_visit"]["beta2_p_one_sided"] < 0.05 else "Uplift models find no significant heterogeneity ")
        + ("but do not rank customers better than the simpler response model at a 30% budget "
           f"({_ci(c30)} visits / 1k), so the extra modeling does not pay for itself in targeting on this campaign. "
           if not beats else f"and beat response targeting at a 30% budget by {_ci(c30)} visits / 1k. ")
        + (f"For email choice, the multi-arm model agrees with the rule \"Womens email to women's-only buyers, Mens otherwise\" "
           f"for {mm['agreement_with_purchase_rule'] * 100:.0f}% of customers and does not beat it "
           f"({_ci(rule)} visits / 1k)." if not _excl0(rule) else
           f"For email choice, the multi-arm model differs from the purchase-line rule by {_ci(rule)} visits / 1k."),
        "",
    ]


def markdown(res: dict) -> str:
    cfg = res["config"]
    h = res["heterogeneity"]["ate"]
    lt = res["learners"]["visit"]
    lines = [
        "## Uplift modeling and budget optimization: who should get the email, and which one?",
        "",
        f"**Method.** Five CATE learners for any email vs none — S-, T- and X-learners with LightGBM bases (early-stopped on "
        f"validation), econml's DR-learner and CausalForestDML (3-fold cross-fitting) — are fit on the {cfg['n_train']:,}-customer "
        f"train split and compared with a classic response model that ranks customers by P(visit | X, emailed). The propensity "
        "is supplied, not learned: assignment was randomized, so econml receives a prior-only classifier that returns each "
        "fold's assignment share (≈ 2/3 any email, 1/3 per arm); a covariate propensity model could only add noise. The learner "
        f"used for targeting is chosen by validation-set normalized Qini ({cfg['n_val']:,} customers); the {cfg['n_test']:,}-customer "
        f"test split is used once, with {cfg['n_bootstrap']:,} paired bootstrap draws for CIs and refits over seeds "
        f"{', '.join(str(s) for s in cfg['seeds'])} for model-randomness spread. Budgeted policies are scored off-policy with "
        "inverse-propensity weighting using the known 1/3 arm probabilities.",
        "",
        *_headline_section(res),
        *_bottom_line(res),
        *_learner_section(res),
        *_heterogeneity_section(res),
        *_segments_section(res),
        *_multiarm_section(res),
        *_budget_section(res),
        "**Caveats**",
        "",
        f"- Conversion is rare (test control conversion rate {h['conversion']['mean_control'] * 100:.2f}%, "
        f"≈{h['conversion']['n_control'] * h['conversion']['mean_control']:.0f} control conversions) and spend is zero for "
        f"≈{(1 - h['conversion']['mean_treated']) * 100:.0f}% of emailed customers; their CATE, Qini and revenue CIs are wide and "
        "mostly span 0. Only visit supports heterogeneity claims.",
        "- The pooled any-email CATE averages two different emails; ranking by it and then sending the Mens email is an "
        "estimand/action mismatch (its top customers skew to women's-only buyers). The Mens-email uplift policy was added "
        "after diagnosing this, so it is a post-hoc comparison, not a pre-specified one.",
        f"- Learner selection on validation is noisy (val normalized Qini spans "
        f"{min(lt[k]['val_norm_qini'] for k in LEARNERS):.3f}–{max(lt[k]['val_norm_qini'] for k in LEARNERS):.3f}); test CIs "
        "overlap for all learners.",
        "- One fixed train/val/test split. Chernozhukov et al. recommend repeating BLP/GATES over many splits and taking "
        "medians; seed refits here vary model randomness only, not the split.",
        "- The Horvitz–Thompson IPW estimator is unbiased but high-variance, and many policy × budget × outcome comparisons "
        "are reported; treat any single nominally significant cell outside the pre-specified visit comparison with caution.",
        f"- Cost per email (${cfg['cost_per_contact']:.2f}) is illustrative; iROAS scales inversely with it, break-even cost does not "
        "depend on it. Data are one 2008 two-week campaign; effects may not transfer to other creatives, seasons or lists.",
        "",
    ]
    # "$" pairs would otherwise be rendered as inline math on GitHub
    return "\n".join(lines).replace("$", "\\$")
