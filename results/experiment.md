## Experiment readout: randomized email test (Hillstrom, n=64,000)

Each customer was randomized 1/3 each to No E-Mail (21,306), Mens E-Mail (21,307) or Womens E-Mail (21,387); outcomes are site visit, purchase and spend over the following two weeks. The full 64,000-row sample is used (this is a readout, not model training). Aggregates come from DuckDB SQL (`sql/experiment_*.sql`); inference is in Python. Binary outcomes: unpooled Wald difference CI, pooled two-proportion z p-value, and delta-method CI for the relative lift (log ratio). Spend is heavy-tailed, so the primary test is Welch t with a 2,000-draw percentile bootstrap CI (Mann-Whitney shown as a robustness check). The 12 primary tests (4 comparisons x 3 outcomes) are Holm-adjusted together.

![Outcomes by arm](results/figures/experiment_outcomes.png)

### EDA

![Distributions](results/figures/experiment_distributions.png)

![Customer mix by arm](results/figures/experiment_mix.png)

![Correlations](results/figures/experiment_correlation.png)

### Randomization check

| Variable | Test | Statistic | df | p (raw) | p (Holm) |
|---|---|---|---|---|---|
| channel | chi-square | 3.63 | 4 | 0.458 | 1.000 |
| zip_code | chi-square | 2.87 | 4 | 0.579 | 1.000 |
| newbie | chi-square | 0.14 | 2 | 0.934 | 1.000 |
| past mens buyer | chi-square | 0.80 | 2 | 0.672 | 1.000 |
| past womens buyer | chi-square | 0.63 | 2 | 0.729 | 1.000 |
| history bucket | chi-square | 13.15 | 12 | 0.358 | 1.000 |
| recency | Kruskal-Wallis | 0.46 | 2 | 0.794 | 1.000 |
| history | Kruskal-Wallis | 0.81 | 2 | 0.667 | 1.000 |

Omnibus LR test that the 10 covariates jointly predict arm (multinomial logit): chi2(20) = 15.1, p = 0.768. Max |SMD| vs control: 0.014 (Mens/Womens).

![Covariate balance](results/figures/experiment_balance.png)

### Average treatment effects

**Site visit** (difference in pp; 95% CI)

| Comparison | Control mean (%) | Treated mean (%) | ATE (pp) [95% CI] | Relative lift % [95% CI] | p (raw) | p (Holm, 12 tests) |
|---|---|---|---|---|---|---|
| Mens vs No E-Mail | 10.62 | 18.28 | 7.66 [7.00, 8.32] | 72.1% [64.0, 80.6] | <0.0001 | <0.0001 |
| Womens vs No E-Mail | 10.62 | 15.14 | 4.52 [3.89, 5.16] | 42.6% [35.6, 50.0] | <0.0001 | <0.0001 |
| Any E-Mail vs No E-Mail | 10.62 | 16.70 | 6.09 [5.54, 6.63] | 57.3% [50.5, 64.5] | <0.0001 | <0.0001 |
| Mens vs Womens | 15.14 | 18.28 | 3.14 [2.43, 3.84] | 20.7% [15.7, 26.0] | <0.0001 | <0.0001 |

**Conversion** (difference in pp; 95% CI)

| Comparison | Control mean (%) | Treated mean (%) | ATE (pp) [95% CI] | Relative lift % [95% CI] | p (raw) | p (Holm, 12 tests) |
|---|---|---|---|---|---|---|
| Mens vs No E-Mail | 0.57 | 1.25 | 0.68 [0.50, 0.86] | 118.8% [76.8, 170.9] | <0.0001 | <0.0001 |
| Womens vs No E-Mail | 0.57 | 0.88 | 0.31 [0.15, 0.47] | 54.3% [23.0, 93.6] | 0.0002 | 0.0006 |
| Any E-Mail vs No E-Mail | 0.57 | 1.07 | 0.50 [0.35, 0.64] | 86.5% [52.9, 127.6] | <0.0001 | <0.0001 |
| Mens vs Womens | 0.88 | 1.25 | 0.37 [0.17, 0.56] | 41.8% [17.8, 70.7] | 0.0002 | 0.0006 |

**Spend** (difference in USD; 95% CI = bootstrap)

| Comparison | Control mean (USD) | Treated mean (USD) | ATE (USD) [95% CI] | Relative lift % [95% CI] | p (raw) | p (Holm, 12 tests) | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Mens vs No E-Mail | 0.653 | 1.423 | 0.770 [0.484, 1.052] | 117.9% [63.9, 196.1] | <0.0001 | <0.0001 | <0.0001 |
| Womens vs No E-Mail | 0.653 | 1.077 | 0.424 [0.171, 0.685] | 65.0% [22.8, 125.8] | 0.0011 | 0.0023 | 0.0002 |
| Any E-Mail vs No E-Mail | 0.653 | 1.250 | 0.597 [0.375, 0.822] | 91.4% [47.7, 154.9] | <0.0001 | <0.0001 | <0.0001 |
| Mens vs Womens | 1.077 | 1.423 | 0.345 [0.028, 0.653] | 32.1% [2.3, 68.7] | 0.030 | 0.030 | 0.0002 |

![ATE forest](results/figures/experiment_ate.png)

### Variance reduction (Lin 2013 regression adjustment)

OLS of the outcome on treatment, the 10 centred pre-treatment covariates and their interactions with treatment; HC2 robust SEs. The comparison is against the unadjusted HC2 SE (same estimator, no covariates). Reductions are small because the covariates predict a rare, mostly-idiosyncratic outcome poorly; in an RCT this only affects precision, not validity.

| Comparison | Outcome | Unadj. ATE | Adj. ATE | Unadj. SE | Adj. SE | SE reduction |
|---|---|---|---|---|---|---|
| Mens vs No E-Mail | visit | 7.659 | 7.622 | 0.3386 | 0.3336 | 1.5% |
| Mens vs No E-Mail | conversion | 0.681 | 0.680 | 0.0921 | 0.0920 | 0.1% |
| Mens vs No E-Mail | spend | 0.770 | 0.770 | 0.1452 | 0.1450 | 0.2% |
| Womens vs No E-Mail | visit | 4.523 | 4.540 | 0.3234 | 0.3189 | 1.4% |
| Womens vs No E-Mail | conversion | 0.311 | 0.312 | 0.0823 | 0.0822 | 0.1% |
| Womens vs No E-Mail | spend | 0.424 | 0.426 | 0.1303 | 0.1304 | -0.1% |
| Any E-Mail vs No E-Mail | visit | 6.088 | 6.083 | 0.2777 | 0.2738 | 1.4% |
| Any E-Mail vs No E-Mail | conversion | 0.495 | 0.496 | 0.0717 | 0.0717 | 0.1% |
| Any E-Mail vs No E-Mail | spend | 0.597 | 0.598 | 0.1126 | 0.1124 | 0.1% |
| Mens vs Womens | visit | 3.136 | 3.064 | 0.3608 | 0.3557 | 1.4% |
| Mens vs Womens | conversion | 0.369 | 0.366 | 0.0995 | 0.0995 | 0.0% |
| Mens vs Womens | spend | 0.345 | 0.342 | 0.1596 | 0.1593 | 0.2% |
(Rates in pp, spend in USD.)

### Power

Two-sided alpha = 0.05, 80% power, arm sizes as run. MDE = smallest true effect detectable; required n is per arm, to detect the observed effect.

| Comparison | Outcome | Baseline | MDE (abs) | MDE (rel) | Observed | n/arm to detect observed |
|---|---|---|---|---|---|---|
| Mens vs No E-Mail | visit | 10.62 | 0.85 pp | 8.0% | 7.66 pp | 327 |
| Mens vs No E-Mail | conversion | 0.57 | 0.22 pp | 39.1% | 0.68 pp | 3,062 |
| Mens vs No E-Mail | spend | 0.653 | 0.407 USD | 62.3% | 0.770 USD | - |
| Womens vs No E-Mail | visit | 10.62 | 0.85 pp | 8.0% | 4.52 pp | 857 |
| Womens vs No E-Mail | conversion | 0.57 | 0.22 pp | 39.0% | 0.31 pp | 11,720 |
| Womens vs No E-Mail | spend | 0.653 | 0.365 USD | 55.9% | 0.424 USD | - |
| Any E-Mail vs No E-Mail | visit | 10.62 | 0.73 pp | 6.9% | 6.09 pp | 496 |
| Any E-Mail vs No E-Mail | conversion | 0.57 | 0.19 pp | 32.6% | 0.50 pp | 5,199 |
| Any E-Mail vs No E-Mail | spend | 0.653 | 0.315 USD | 48.3% | 0.597 USD | - |
| Mens vs Womens | visit | 15.14 | 0.98 pp | 6.5% | 3.14 pp | 2,218 |
| Mens vs Womens | conversion | 0.88 | 0.27 pp | 30.8% | 0.37 pp | 12,156 |
| Mens vs Womens | spend | 1.077 | 0.447 USD | 41.5% | 0.345 USD | - |

### Incrementality

**Cost assumption:** `COST_PER_CONTACT = $0.10` per email is an ILLUSTRATIVE assumption (the dataset has no cost column). The break-even cost per email (= incremental revenue per email) does not depend on it. Revenue is the two-week spend only (no margin, no lifetime value). Totals scale each ATE to the number of emails actually sent in that arm of the test.

| Arm | Emails sent | Incr. visits /1k [95% CI] | Incr. conversions /1k [95% CI] | Incr. revenue /1k (USD) [95% CI] | Total incr. revenue (USD) | iROAS @ $0.10/email | Break-even cost / email (USD) |
|---|---|---|---|---|---|---|---|
| Mens | 21,307 | 76.6 [70.0, 83.2] | 6.8 [5.0, 8.6] | 770 [484, 1052] | 16403 [10313, 22409] | 7.70 [4.85, 10.55] | 0.770 [0.485, 1.055] |
| Womens | 21,387 | 45.2 [38.9, 51.6] | 3.1 [1.5, 4.7] | 424 [171, 685] | 9077 [3662, 14659] | 4.24 [1.69, 6.80] | 0.424 [0.169, 0.680] |
| Any E-Mail | 42,694 | 60.9 [55.4, 66.3] | 5.0 [3.5, 6.4] | 597 [375, 822] | 25480 [16020, 35091] | 5.97 [3.76, 8.17] | 0.597 [0.376, 0.817] |

![Incrementality](results/figures/experiment_incrementality.png)

### Segment readout and heterogeneity

Pre-specified segments (channel, ZIP type, newbie, past mens / womens buyer, history bucket), defined before looking at outcomes. Per-segment ATEs vs No E-Mail with 95% Wald CIs; Holm correction over all 76 segment x arm x outcome tests. Interaction tests are linear-probability models `y ~ T x segment` with HC2 SEs (joint Wald test on the interaction terms), Holm-adjusted over all 24 interaction tests.

![Segment forest](results/figures/experiment_segments.png)

| Segment | Level | n control | Mens: visit pp (Holm p) | Mens: conv. pp (Holm p) | Womens: visit pp (Holm p) | Womens: conv. pp (Holm p) |
|---|---|---|---|---|---|---|
| channel | Multichannel | 2,606 | 8.29 [6.26, 10.33] (<0.0001) | 1.02 [0.42, 1.61] (0.021) | 4.71 [2.76, 6.66] (<0.0001) | 0.71 [0.15, 1.26] (0.233) |
| channel | Phone | 9,327 | 7.56 [6.61, 8.51] (<0.0001) | 0.55 [0.29, 0.80] (0.0011) | 4.46 [3.57, 5.35] (<0.0001) | 0.17 [-0.05, 0.40] (1.000) |
| channel | Web | 9,373 | 7.56 [6.53, 8.59] (<0.0001) | 0.72 [0.45, 0.99] (<0.0001) | 4.57 [3.57, 5.56] (<0.0001) | 0.34 [0.10, 0.59] (0.128) |
| history bucket | 1) $0 - $100 | 7,612 | 6.86 [5.84, 7.87] (<0.0001) | 0.50 [0.24, 0.76] (0.0050) | 4.68 [3.69, 5.66] (<0.0001) | 0.38 [0.13, 0.63] (0.070) |
| history bucket | 2) $100 - $200 | 4,836 | 6.91 [5.59, 8.24] (<0.0001) | 0.48 [0.16, 0.81] (0.085) | 4.25 [2.99, 5.51] (<0.0001) | 0.37 [0.06, 0.68] (0.285) |
| history bucket | 3) $200 - $350 | 4,044 | 8.44 [6.82, 10.06] (<0.0001) | 0.85 [0.39, 1.30] (0.0086) | 3.61 [2.09, 5.13] (0.0001) | -0.07 [-0.42, 0.28] (1.000) |
| history bucket | 4) $350 - $500 | 2,124 | 8.92 [6.54, 11.30] (<0.0001) | 0.87 [0.12, 1.62] (0.312) | 4.04 [1.80, 6.28] (0.013) | -0.22 [-0.82, 0.39] (1.000) |
| history bucket | 5) $500 - $750 | 1,652 | 9.90 [7.40, 12.39] (<0.0001) | 0.83 [0.16, 1.51] (0.234) | 6.61 [4.24, 8.98] (<0.0001) | 1.02 [0.33, 1.71] (0.094) |
| history bucket | 6) $750 - $1,000 | 622 | 6.15 [1.89, 10.41] (0.108) | 2.00 [0.68, 3.32] (0.085) | 5.48 [1.14, 9.81] (0.233) | 0.87 [-0.21, 1.94] (1.000) |
| history bucket | 7) $1,000 + | 416 | 9.71 [4.34, 15.08] (0.014) | 0.93 [-0.87, 2.73] (1.000) | 5.38 [0.07, 10.68] (0.527) | 1.13 [-0.76, 3.01] (1.000) |
| newbie | existing | 10,611 | 7.78 [6.77, 8.79] (<0.0001) | 0.62 [0.34, 0.90] (0.0005) | 4.01 [3.04, 4.98] (<0.0001) | 0.09 [-0.15, 0.34] (1.000) |
| newbie | new (<12mo) | 10,695 | 7.53 [6.68, 8.38] (<0.0001) | 0.74 [0.51, 0.97] (<0.0001) | 5.04 [4.23, 5.86] (<0.0001) | 0.53 [0.31, 0.74] (<0.0001) |
| past mens buyer | no | 9,519 | 7.06 [6.11, 8.01] (<0.0001) | 0.57 [0.32, 0.82] (0.0003) | 7.40 [6.44, 8.35] (<0.0001) | 0.51 [0.26, 0.76] (0.0017) |
| past mens buyer | yes | 11,787 | 8.16 [7.24, 9.08] (<0.0001) | 0.77 [0.51, 1.02] (<0.0001) | 2.18 [1.33, 3.02] (<0.0001) | 0.15 [-0.06, 0.36] (1.000) |
| past womens buyer | no | 9,638 | 6.92 [5.96, 7.88] (<0.0001) | 0.60 [0.34, 0.86] (0.0002) | 1.11 [0.24, 1.97] (0.233) | 0.06 [-0.15, 0.28] (1.000) |
| past womens buyer | yes | 11,668 | 8.26 [7.34, 9.17] (<0.0001) | 0.74 [0.49, 0.99] (<0.0001) | 7.31 [6.41, 8.22] (<0.0001) | 0.51 [0.28, 0.75] (0.0006) |
| zip_code | Rural | 3,139 | 7.34 [5.42, 9.26] (<0.0001) | 0.53 [0.02, 1.04] (0.513) | 2.91 [1.07, 4.75] (0.054) | 0.33 [-0.15, 0.82] (1.000) |
| zip_code | Suburban | 9,625 | 7.43 [6.47, 8.40] (<0.0001) | 0.74 [0.48, 1.01] (<0.0001) | 4.89 [3.96, 5.81] (<0.0001) | 0.26 [0.03, 0.48] (0.312) |
| zip_code | Urban | 8,542 | 7.96 [6.94, 8.99] (<0.0001) | 0.66 [0.38, 0.95] (0.0002) | 4.69 [3.72, 5.67] (<0.0001) | 0.36 [0.10, 0.62] (0.124) |

Interaction tests (treatment x segment):

| Segment | Arm | Outcome | Wald chi2 | df | p (raw) | p (Holm) | Effect difference pp [95% CI] (level 1 vs level 0; 1 = yes / new) |
|---|---|---|---|---|---|---|---|
| past womens buyer | Womens | visit | 94.2 | 1 | <0.0001 | <0.0001 | 6.20 [4.95, 7.46] (1) |
| past mens buyer | Womens | visit | 64.4 | 1 | <0.0001 | <0.0001 | -5.22 [-6.50, -3.95] (1) |
| past womens buyer | Womens | conversion | 7.7 | 1 | 0.0055 | 0.121 | 0.45 [0.13, 0.77] (1) |
| newbie | Womens | conversion | 7.0 | 1 | 0.0084 | 0.176 | 0.43 [0.11, 0.76] (1) |
| past mens buyer | Womens | conversion | 4.8 | 1 | 0.028 | 0.568 | -0.36 [-0.69, -0.04] (1) |
| history bucket | Womens | conversion | 13.5 | 6 | 0.035 | 0.670 | - |
| past womens buyer | Mens | visit | 3.9 | 1 | 0.048 | 0.858 | 1.34 [0.01, 2.67] (1) |
| past mens buyer | Mens | visit | 2.6 | 1 | 0.104 | 1.000 | 1.10 [-0.23, 2.42] (1) |
| newbie | Womens | visit | 2.5 | 1 | 0.111 | 1.000 | 1.03 [-0.24, 2.29] (1) |
| history bucket | Mens | visit | 9.5 | 6 | 0.146 | 1.000 | - |
| zip_code | Womens | visit | 3.6 | 2 | 0.163 | 1.000 | - |
| channel | Womens | conversion | 3.4 | 2 | 0.184 | 1.000 | - |
| history bucket | Mens | conversion | 7.6 | 6 | 0.266 | 1.000 | - |
| past mens buyer | Mens | conversion | 1.2 | 1 | 0.282 | 1.000 | 0.20 [-0.16, 0.56] (1) |
| channel | Mens | conversion | 2.3 | 2 | 0.313 | 1.000 | - |
| past womens buyer | Mens | conversion | 0.6 | 1 | 0.434 | 1.000 | 0.14 [-0.22, 0.50] (1) |
| newbie | Mens | conversion | 0.4 | 1 | 0.519 | 1.000 | 0.12 [-0.24, 0.48] (1) |
| history bucket | Womens | visit | 5.1 | 6 | 0.530 | 1.000 | - |
| newbie | Mens | visit | 0.1 | 1 | 0.708 | 1.000 | -0.25 [-1.58, 1.07] (1) |
| zip_code | Mens | visit | 0.7 | 2 | 0.721 | 1.000 | - |
| zip_code | Mens | conversion | 0.6 | 2 | 0.751 | 1.000 | - |
| channel | Mens | visit | 0.4 | 2 | 0.799 | 1.000 | - |
| zip_code | Womens | conversion | 0.4 | 2 | 0.831 | 1.000 | - |
| channel | Womens | visit | 0.1 | 2 | 0.970 | 1.000 | - |

**Caveats**

- Outcomes cover a two-week window only: no long-run, cannibalization or unsubscribe effects are measured, so iROAS is a short-window figure.
- COST_PER_CONTACT ($0.10) is an illustrative assumption; use the break-even cost per email for decisions. Revenue is not margin.
- Spend is zero-inflated and heavy-tailed: the Welch/bootstrap CIs are for the mean; Mann-Whitney tests a different (distributional) hypothesis.
- Segment-level CIs are unadjusted Wald intervals; segments with small n (e.g. top history buckets) have wide intervals and low power for effect differences.
- Interaction tests use a linear-probability model with robust SEs; logistic-scale heterogeneity may differ. Only pre-specified dimensions were tested, and Holm correction is applied across the full family.
- History is floor-censored at $29.99 (12.4% of customers) and 6,562 exact-duplicate rows are distinct customers (see results/data.json).
- Balance tests: with 76 segment tests and 8 balance tests, some raw p<0.05 are expected by chance; conclusions rest on Holm-adjusted values.
- Hillstrom is a 2008 single-retailer test; effect sizes should not be assumed to transfer.
- Heterogeneity: strongest interaction is past womens buyer x Womens E-Mail on visit (Wald chi2=94.2, df=1, raw p=<0.0001, Holm p=<0.0001); 2 of 24 interaction tests survive Holm correction.
