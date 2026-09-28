"""Assemble README.md from the results each stage wrote (results/<stage>.json|.md)."""
from __future__ import annotations

from causalads import config as C
from causalads import io

SECTIONS = ["experiment", "prediction", "observational", "uplift", "did"]

INTRO = """# CausalAds

**Which customers should receive a marketing campaign, and what incremental effect does the campaign actually cause?**

CausalAds is an end-to-end statistical ML and causal-inference pipeline on a real randomized email campaign
(Kevin Hillstrom's MineThatData E-Mail Analytics challenge: 64,000 customers, randomized to a Mens email,
a Womens email, or no email). There are no LLMs in it: SQL, pandas, scikit-learn, XGBoost/LightGBM,
statsmodels, and EconML.

| Layer | Question | Methods |
|---|---|---|
| Experiment readout | Did the emails work, and by how much? | Randomization checks, two-proportion / Welch / Mann-Whitney tests, bootstrap and delta-method CIs, Holm correction, regression adjustment (CUPED-style), power/MDE, iROAS |
| Prediction | Who is likely to convert? | Logistic regression, Random Forest, XGBoost, LightGBM; Optuna tuning, repeated stratified CV, class-imbalance handling, Platt/isotonic calibration, SHAP, error analysis |
| Propensity scores | Could we recover the effect without an experiment? | Semi-synthetic confounded samples from the RCT; logistic/GBM propensity models, matching, stratification, IPW, doubly robust (AIPW); bias/RMSE/coverage against the RCT truth |
| Uplift / HTE | Who responds *because* of the email? | S/T/X/DR-learners, causal forests (EconML), Qini/AUUC, GATES + BLP heterogeneity test, interpretable segment rules, KMeans segments |
| Budget optimization | Where should a fixed budget go? | Off-policy (IPW) evaluation of targeting policies on held-out randomized data; uplift vs response-model targeting |
| Difference-in-differences | Did a geo campaign lift conversions? | 2x2 DiD, TWFE, event study / pre-trend test, permutation inference, synthetic control, power simulation |

```
raw CSV (64,000 customers)
  -> DuckDB SQL: typing, cleaning, feature engineering        sql/01_clean.sql, sql/02_features.sql
  -> validation + fixed stratified train/val/test split       src/causalads/data.py
  -> experiment readout (hypothesis tests, CIs, iROAS)        src/causalads/experiment.py
  -> response models (LR / RF / XGBoost / LightGBM)           src/causalads/prediction.py
  -> propensity-score estimators vs RCT ground truth          src/causalads/observational.py
  -> uplift / causal forest / segments / budget optimizer     src/causalads/uplift.py
  -> geo-level difference-in-differences                      src/causalads/did.py
  -> this README                                              src/causalads/report.py
```

## Reproduce

```bash
uv sync
uv run causalads all          # every stage, then regenerates this README
uv run causalads uplift       # or a single stage
```

Raw data is downloaded on first run. Every number below is generated from `results/*.json` by
`src/causalads/report.py`; nothing is typed by hand.
"""


def _data_section() -> str:
    d = io.load_json("data")
    v = d["validation"]
    rows = "\n".join(
        f"| {r['segment']} | {r['customers']:,} | {r['visit_rate']:.2%} | {r['conversion_rate']:.2%} "
        f"| ${r['spend_per_customer']:.3f} |"
        for r in d["arm_summary"]
    )
    s = d["split_sizes"]
    return f"""## Data

Source: MineThatData E-Mail Analytics challenge (Hillstrom, 2008). {v['rows']:,} customers who purchased in the last
12 months were randomly split into three equal groups; outcomes are measured over the following two weeks.

| Arm | Customers | Visit rate | Conversion rate | Spend / customer |
|---|---|---|---|---|
{rows}

Data-quality findings from `data.validate_raw` and the SQL cleaning step:
- {sum(v['missing_values'].values())} missing values. The pipeline still imputes (median / mode plus missing-indicator columns), so it does not rely on that.
- `zip_code` spells *Suburban* as "Surburban" in {v['zip_code_misspelled_surburban']:,} rows. This is fixed in SQL; without the fix the one-hot `zip_suburban` feature is silently all zeros.
- `history` is floor-censored at $29.99 for {v['history_at_floor_29_99']:,} customers ({v['history_at_floor_29_99'] / v['rows']:.1%}). This is kept as a `history_at_floor` flag. These customers explain the {v['duplicate_rows']:,} exact-duplicate rows, which are distinct customers and are kept.
- Fixed split, stratified on arm x visit x conversion: train {s['train']:,} / validation {s['val']:,} / test {s['test']:,}.
"""


def _ci(lo: float, hi: float, scale: float = 1.0, nd: int = 1) -> str:
    return f"[{lo * scale:.{nd}f}, {hi * scale:.{nd}f}]"


def _segment_vs_rest(seg: dict, overall: dict) -> dict:
    """Segment ATE vs the ATE of all *other* test customers (binary outcome, exact from counts)."""
    import math

    from scipy.stats import norm

    groups = {}
    for arm, n_key, m_key in (("t", "n_treated", "mean_treated"), ("c", "n_control", "mean_control")):
        n_s, n = seg[n_key], overall[n_key]
        groups[f"s{arm}"] = (n_s, seg[m_key])
        groups[f"r{arm}"] = (n - n_s, (overall[m_key] * n - seg[m_key] * n_s) / (n - n_s))
    rest = groups["rt"][1] - groups["rc"][1]
    diff = seg["estimate"] - rest
    se = math.sqrt(sum(p * (1 - p) / n for n, p in groups.values()))
    return {"rest": rest, "diff": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
            "p": 2 * (1 - norm.cdf(abs(diff / se)))}


def _key_results() -> str:
    e = io.load_json("experiment")["headline"]
    p = io.load_json("prediction")["headline"]["conversion"]
    o = io.load_json("observational")["headline"]
    u_all = io.load_json("uplift")
    u = u_all["headline"]
    d = io.load_json("did")["headline"]

    mv, mc = e["ate"]["Mens vs No E-Mail"]["visit"], e["ate"]["Mens vs No E-Mail"]["conversion"]
    rev = e["incrementality"]["Mens vs No E-Mail"]["spend"]["per_1000"]
    aucs = [p["test"][m]["roc_auc"]["est"] for m in ("lr", "rf", "xgb", "lgbm")]
    dauc = p["best_vs_lr_diff_roc_auc"]
    best = o["best_estimator_visit"]
    seg = u["segments_top"]
    cap = u_all["policies"]["capture_pct_of_full_campaign_visits"]["0.30"]["response"]
    ma30 = u["budget"]["0.30"]["uplift_multiarm_vs_response"]["visit"]["diff_per_1000"]
    lift = d["did_pct_lift"]
    vr = _segment_vs_rest(seg["ate_visit"], u["visit_ate_test"])
    return f"""## Key results

| Question | Result (95% CI) |
|---|---|
| Did the Mens email work? | Visits **+{mv['ate'] * 100:.1f} pp** {_ci(*mv['ci95'], 100)}, conversions **+{mc['ate'] * 100:.2f} pp** {_ci(*mc['ci95'], 100, 2)} (+{mc['rel_lift'] * 100:.0f}% relative), **${rev[0]:,.0f}** incremental sales per 1,000 emails {_ci(rev[1], rev[2], 1, 0)} (gross customer spend, not margin) |
| Can we predict who converts? | Test ROC-AUC {min(aucs):.2f}-{max(aucs):.2f} across LR / RF / XGBoost / LightGBM. Gradient boosting's gain over logistic regression is not significant (ΔAUC {dauc['est']:+.3f}, p = {dauc['p_two_sided']:.2f}). |
| Can propensity scores replace the experiment? | On confounded samples, the naive estimate is biased **+{o['naive_visit_bias_pct']['estimate']:.0f}%**. Doubly robust AIPW recovers the RCT effect with no detectable bias: the lowest-RMSE variant, {best['label']}, has {best['bias_pct']:+.1f}% bias {_ci(*best['bias_pct_ci'])}% and {best['coverage']:.0%} CI coverage over 300 replicates. The confounding comes from a simulated targeting rule. |
| Is there real heterogeneity? | Yes, for visits: BLP test beta2 = {u['blp_visit']['beta2']:.2f}, one-sided p = {u['blp_visit']['beta2_p_one_sided']:.3f}. Segment "{seg['rule']}" (rule learned on validation, scored on test): **+{seg['ate_visit']['estimate'] * 100:.1f} pp** {_ci(seg['ate_visit']['ci_low'], seg['ate_visit']['ci_high'], 100)} vs +{vr['rest'] * 100:.1f} pp for all other customers (difference +{vr['diff'] * 100:.1f} pp {_ci(vr['lo'], vr['hi'], 100)}, p = {vr['p']:.2f}) |
| Where should a 30% budget go? | Emailing the top 30% by response-model score captures **{cap['estimate']:.0f}%** {_ci(cap['ci_low'], cap['ci_high'], 1, 0)} of the incremental visits of sending the Mens email to everyone. Multi-arm uplift targeting does **not** beat it ({ma30['estimate']:+.1f} visits / 1k {_ci(ma30['ci_low'], ma30['ci_high'])}). |
| Did the geo campaign lift conversions? | DiD **{lift['estimate'] * 100:+.1f}%** {_ci(*lift['ci95'], 100)}, permutation p = {lift['perm_p']:.2f}. The point estimate is consistent with GeoLift's published {d['geolift_published']['lift_pct']}% (p = {d['geolift_published']['p_value']}, augmented synthetic control). Under cross-city permutation inference, though, the minimum detectable lift is {d['mde_80pct_power_pct']:.1f}% at 80% power, so a 5-7% effect cannot be confirmed. |
"""


def run() -> None:
    parts = [INTRO, _key_results(), _data_section()]
    for name in SECTIONS:
        md = C.RESULTS / f"{name}.md"
        parts.append(md.read_text().strip() + "\n" if md.exists() else f"## {name}\n\n_Not run yet._\n")
    parts.append("""## Limitations

- The Hillstrom campaign is from 2008 and has only 8 raw pre-treatment attributes, so the HTE models have little to work with. That limits how much uplift targeting can gain over simpler rules.
- Conversions are rare (~0.9%), so conversion-level heterogeneity estimates are noisy. Visits carry most of the statistical signal.
- Campaign cost is not in the data. iROAS uses an illustrative cost per email (`config.COST_PER_CONTACT`), and the break-even cost per email is reported so no conclusion depends on it.
- The observational study is semi-synthetic: real customers and outcomes, but a selection mechanism we designed. It shows the estimators work when unconfoundedness holds; it cannot show that unconfoundedness holds in the wild.
""")
    (C.ROOT / "README.md").write_text("\n".join(parts))
    print(f"wrote {C.ROOT / 'README.md'}")
