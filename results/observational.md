## Observational methods vs. an experimental ground truth

Marketing data are usually observational: marketers email the customers they already expect to buy, so a naive emailed-vs-not comparison is confounded. Because Hillstrom is randomized, we can manufacture that selection bias from real customers and real outcomes and check which estimators recover the experimental answer (a LaLonde-style design). A targeting rule e(X) (below) plays the marketer. From the RCT, emailed customers are kept with probability c*e(X)/p and un-emailed ones with c*(1-e(X))/(1-p), where p = 0.6671 is the RCT's emailed share and c = 0.394 is the largest value keeping both probabilities <= 1. Retention is then c for every X, so the observational sample has the full population's customer mix (the estimand is the population ATE) while its true propensity is exactly e(X); selection uses only (X, T), so there is no unmeasured confounding by construction. Ground truth is the RCT difference in means on all 64,000 customers. We analyse one observational sample in detail (n = 25,312), then repeat the whole pipeline on 300 simulated samples (each: bootstrap the 64,000 customers, then apply the targeting draw) and score each estimator's bias, RMSE and 95% CI coverage of the benchmark. Nuisance models are cross-fitted (5 folds): a main-effects logistic regression, which is misspecified here by design (the rule has a VIP interaction, a win-back arm that is non-monotone in recency, a $300 threshold and a bounded link), and LightGBM. Matching is 1:1 nearest-neighbour on the logit PS with replacement and a 0.2-SD caliper (Abadie-Imbens variance); it is run for the ATE and, separately, for the ATT, which is scored against the RCT's ATT for the observational population (the e(X)-weighted RCT difference in means).

**Targeting rule** (logit score, mapped to e = 0.1 + 0.8 * sigmoid(score))

| Term | Coefficient | Share of customers |
|---|---|---|
| Intercept | -1.8 |  |
| Recent buyer (recency <= 3 mo) | +1.6 | 35.0% |
| High value (history >= $300) | +1.2 | 27.8% |
| VIP: recent x high value | +1.6 | 13.5% |
| Win-back: lapsed (recency >= 10) x high value | +2.2 | 3.4% |
| Multichannel buyer | +1.0 | 12.1% |
| Bought both mens and womens | +1.0 | 10.1% |
| New customer (last 12 mo) | -0.8 | 50.2% |

True propensity ranges 0.155-0.892; the marketer emails 37.2% of customers. Observational samples average 25,273 customers (37.2% emailed).

**Ground truth from the RCT (all 64,000 customers)**

| Outcome | RCT ATE [95% CI] | Covariate-adjusted ATE (Lin) | ATT of observational population [95% CI] | ATT - ATE [95% CI] |
|---|---|---|---|---|
| Site visit (pp) | 6.09 [5.54, 6.63] | 6.07 | 6.49 [5.77, 7.21] | 0.40 [0.03, 0.78] |
| Conversion (pp) | 0.50 [0.35, 0.64] | 0.49 | 0.56 [0.36, 0.76] | 0.06 [-0.04, 0.16] |
| Spend ($) | 0.597 [0.376, 0.817] | 0.595 | 0.667 [0.368, 0.965] | 0.070 [-0.080, 0.220] |

### Propensity model diagnostics

Single sample (n = 25,312); bracketed ranges are the 2.5-97.5th percentiles across the 300 simulated samples.

| | True e(X) | Logistic PS | GBM PS |
|---|---|---|---|
| AUC (out-of-fold) | 0.768 [0.763, 0.775] | 0.755 [0.749, 0.761] | 0.765 [0.758, 0.771] |
| RMSE vs true propensity | - | 0.094 [0.094, 0.096] | 0.039 [0.041, 0.047] |
| Share of PS outside [0.05, 0.95] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] |
| Max abs SMD after IPW, model features | 0.021 [0.017, 0.049] | 0.113 [0.044, 0.124] | 0.018 [0.009, 0.022] |
| Max abs SMD after IPW, incl. rule terms | 0.021 [0.020, 0.051] | 0.202 [0.176, 0.212] | 0.018 [0.009, 0.022] |
| Effective sample size, emailed | 6,686 [6,582, 6,833] | 6,255 [6,063, 6,531] | 6,540 [6,315, 6,743] |
| Effective sample size, not emailed | 11,164 [11,008, 11,510] | 10,304 [9,147, 12,191] | 11,158 [10,316, 11,162] |
| Calibration slope [95% CI] | - | 0.99 [0.96, 1.03] | 0.98 [0.95, 1.01] |
| Expected calibration error / Brier | - | 0.028 / 0.1853 | 0.009 / 0.1777 |

Before weighting the max |SMD| is 0.710 on model features and 0.792 including the rule terms (simulation range 0.782-0.834). Logistic-PS weighting leaves the rule's interaction terms imbalanced (max |SMD| 0.202, above the 0.1 threshold) while GBM-PS weighting balances everything (0.018). No trimming was needed: the true propensity is bounded in [0.1, 0.9] by design and estimated scores are only clipped to [0.01, 0.99] as a numerical guard (no units dropped, estimand unchanged). The matching caliper dropped at most 0.14% of units (97.5th percentile over runs).

![Propensity overlap](results/figures/observational_overlap.png)

![Propensity calibration](results/figures/observational_calibration.png)

![Love plot](results/figures/observational_love_plot.png)

### Simulation results: site visit (300 observational samples)

Benchmark: RCT ATE 6.09 pp for ATE rows; RCT ATT 6.49 pp for ATT rows. Bias CIs are Monte Carlo 95% CIs; RMSE and SE-ratio CIs bootstrap the replicates; coverage CIs are Wilson. Bias/SD is bias in units of the estimator's sampling SD; the 10x-n column projects coverage if the sample were ten times larger and the bias did not shrink (honest SEs assumed). SE ratio = mean reported SE / empirical SD (1 = honest).

| Estimator | Estimand | Bias, pp [95% CI] (% of truth) | Bias/SD | RMSE, pp [95% CI] | Coverage [95% CI] | Coverage at 10x n | SE ratio [95% CI] |
|---|---|---|---|---|---|---|---|
| Naive difference in means | ATE | 3.738 [3.690, 3.786] (+61.4%) | +8.80 | 3.762 [3.715, 3.809] | 0.000 [0.000, 0.013] | 0.00 | 1.10 [1.03, 1.19] |
| OLS adjustment (Y ~ T + X) | ATE | 0.095 [0.039, 0.152] (+1.6%) | +0.19 | 0.510 [0.470, 0.551] | 0.950 [0.919, 0.969] | 0.91 | 1.01 [0.94, 1.10] |
| PS stratification, quintiles (logistic) | ATE | 0.315 [0.260, 0.371] (+5.2%) | +0.65 | 0.581 [0.537, 0.628] | 0.927 [0.891, 0.951] | 0.47 | 1.05 [0.97, 1.14] |
| PS stratification, quintiles (GBM) | ATE | 0.317 [0.260, 0.374] (+5.2%) | +0.63 | 0.593 [0.549, 0.638] | 0.913 [0.876, 0.940] | 0.49 | 1.04 [0.96, 1.12] |
| PS matching 1:1, caliper (logistic) | ATE | 0.093 [0.025, 0.161] (+1.5%) | +0.15 | 0.608 [0.558, 0.657] | 0.957 [0.927, 0.975] | 0.92 | 1.10 [1.01, 1.19] |
| PS matching 1:1, caliper (GBM) | ATE | 0.178 [0.107, 0.249] (+2.9%) | +0.28 | 0.651 [0.600, 0.698] | 0.953 [0.923, 0.972] | 0.85 | 1.07 [0.99, 1.16] |
| IPW, Hajek (logistic) | ATE | -0.150 [-0.209, -0.090] (-2.5%) | -0.29 | 0.542 [0.501, 0.585] | 0.943 [0.911, 0.964] | 0.85 | 1.05 [0.97, 1.14] |
| IPW, Hajek (GBM) | ATE | -0.062 [-0.119, -0.004] (-1.0%) | -0.12 | 0.512 [0.475, 0.549] | 0.967 [0.940, 0.982] | 0.93 | 1.07 [0.99, 1.15] |
| AIPW (logistic PS + GLM outcome) | ATE | 0.049 [-0.008, 0.107] (+0.8%) | +0.10 | 0.508 [0.468, 0.548] | 0.967 [0.940, 0.982] | 0.94 | 1.09 [1.01, 1.18] |
| AIPW (GBM PS + GBM outcome) | ATE | -0.036 [-0.094, 0.023] (-0.6%) | -0.07 | 0.515 [0.476, 0.553] | 0.967 [0.940, 0.982] | 0.94 | 1.06 [0.99, 1.15] |
| AIPW (logistic PS + GBM outcome) | ATE | -0.016 [-0.075, 0.043] (-0.3%) | -0.03 | 0.518 [0.478, 0.559] | 0.963 [0.936, 0.979] | 0.95 | 1.06 [0.99, 1.15] |
| AIPW (GBM PS + GLM outcome) | ATE | -0.024 [-0.082, 0.033] (-0.4%) | -0.05 | 0.508 [0.470, 0.544] | 0.970 [0.944, 0.984] | 0.95 | 1.07 [1.00, 1.15] |
| IPW, Hajek (true PS, oracle) | ATE | 0.005 [-0.053, 0.063] (+0.1%) | +0.01 | 0.513 [0.474, 0.551] | 0.953 [0.923, 0.972] | 0.95 | 1.04 [0.97, 1.13] |
| PS matching 1:1, caliper (logistic) | ATT | -0.005 [-0.097, 0.088] (-0.1%) | -0.01 | 0.815 [0.746, 0.886] | 0.953 [0.923, 0.972] | 0.95 | 1.08 [0.99, 1.18] |
| PS matching 1:1, caliper (GBM) | ATT | 0.266 [0.171, 0.362] (+4.1%) | +0.32 | 0.885 [0.816, 0.951] | 0.950 [0.919, 0.969] | 0.83 | 1.08 [1.00, 1.17] |

![Estimates vs RCT benchmark](results/figures/observational_estimates_visit.png)

### Conversion and spend

| Estimator | Estimand | Conversion bias % [95% CI] | Conversion RMSE (pp) | Conversion coverage | Spend bias % [95% CI] | Spend RMSE ($) | Spend coverage |
|---|---|---|---|---|---|---|---|
| Naive difference in means | ATE | +65.7% [+62.5, +68.8] | 0.353 | 0.280 | +54.0% [+50.0, +57.9] | 0.384 | 0.687 |
| OLS adjustment (Y ~ T + X) | ATE | -10.1% [-13.3, -6.8] | 0.152 | 0.927 | -14.5% [-18.4, -10.6] | 0.223 | 0.943 |
| PS stratification, quintiles (logistic) | ATE | -6.0% [-9.3, -2.7] | 0.149 | 0.940 | -7.8% [-11.8, -3.8] | 0.215 | 0.947 |
| PS stratification, quintiles (GBM) | ATE | +7.1% [+3.8, +10.3] | 0.145 | 0.940 | +6.2% [+2.2, +10.2] | 0.214 | 0.957 |
| PS matching 1:1, caliper (logistic) | ATE | -6.7% [-10.6, -2.7] | 0.177 | 0.953 | -6.8% [-11.5, -2.2] | 0.248 | 0.950 |
| PS matching 1:1, caliper (GBM) | ATE | +3.0% [-0.8, +6.8] | 0.167 | 0.947 | +4.4% [-0.7, +9.6] | 0.273 | 0.970 |
| IPW, Hajek (logistic) | ATE | -9.5% [-13.1, -5.9] | 0.164 | 0.950 | -11.4% [-15.4, -7.3] | 0.223 | 0.960 |
| IPW, Hajek (GBM) | ATE | -0.4% [-3.8, +3.0] | 0.148 | 0.943 | +0.1% [-4.0, +4.1] | 0.213 | 0.960 |
| AIPW (logistic PS + GLM outcome) | ATE | -1.8% [-5.5, +1.9] | 0.160 | 0.973 | -10.4% [-14.2, -6.5] | 0.213 | 0.963 |
| AIPW (GBM PS + GBM outcome) | ATE | +0.6% [-2.9, +4.0] | 0.151 | 0.957 | +1.8% [-2.3, +5.9] | 0.216 | 0.967 |
| AIPW (logistic PS + GBM outcome) | ATE | -2.0% [-5.5, +1.5] | 0.152 | 0.960 | -1.7% [-5.6, +2.3] | 0.208 | 0.967 |
| AIPW (GBM PS + GLM outcome) | ATE | +0.4% [-3.0, +3.8] | 0.149 | 0.943 | +0.6% [-3.4, +4.6] | 0.212 | 0.963 |
| IPW, Hajek (true PS, oracle) | ATE | +0.7% [-2.7, +4.1] | 0.147 | 0.927 | +1.4% [-2.5, +5.4] | 0.207 | 0.960 |
| PS matching 1:1, caliper (logistic) | ATT | -9.5% [-14.1, -4.8] | 0.235 | 0.963 | -9.0% [-14.6, -3.3] | 0.337 | 0.967 |
| PS matching 1:1, caliper (GBM) | ATT | +6.7% [+2.1, +11.2] | 0.228 | 0.937 | +4.8% [-1.1, +10.7] | 0.349 | 0.937 |

![Bias and coverage](results/figures/observational_simulation_metrics.png)

### Does double robustness rescue a misspecified propensity model?

Bias as % of the RCT benchmark [Monte Carlo 95% CI].

| Estimator | Site visit | Conversion | Spend |
|---|---|---|---|
| IPW, Hajek (logistic) | -2.5% [-3.4, -1.5] | -9.5% [-13.1, -5.9] | -11.4% [-15.4, -7.3] |
| PS stratification, quintiles (logistic) | +5.2% [+4.3, +6.1] | -6.0% [-9.3, -2.7] | -7.8% [-11.8, -3.8] |
| PS matching 1:1, caliper (logistic) | +1.5% [+0.4, +2.7] | -6.7% [-10.6, -2.7] | -6.8% [-11.5, -2.2] |
| AIPW (logistic PS + GLM outcome) | +0.8% [-0.1, +1.8] | -1.8% [-5.5, +1.9] | -10.4% [-14.2, -6.5] |
| AIPW (logistic PS + GBM outcome) | -0.3% [-1.2, +0.7] | -2.0% [-5.5, +1.5] | -1.7% [-5.6, +2.3] |
| IPW, Hajek (GBM) | -1.0% [-2.0, -0.1] | -0.4% [-3.8, +3.0] | +0.1% [-4.0, +4.1] |
| AIPW (GBM PS + GBM outcome) | -0.6% [-1.5, +0.4] | +0.6% [-2.9, +4.0] | +1.8% [-2.3, +5.9] |

### Findings

- **Naive comparison is badly confounded.** In the single sample the naive visit lift is 9.69 pp [8.77, 10.61] vs the RCT's 6.09 pp [5.54, 6.63]; across samples its bias is +61.4% [+60.6, +62.2] of the true effect and its CI covers the truth 0.0% of the time. For spend it is +54.0% [+50.0, +57.9].
- **Best estimator (visit; lowest RMSE among those with no detectable bias and nominal coverage): AIPW (GBM PS + GLM outcome)** - bias -0.4% [-1.3, +0.5], RMSE 0.508 pp [0.470, 0.544], coverage 0.970 [0.944, 0.984]. It is also the lowest-RMSE estimator overall. Paired RMSE ratio vs the next estimator by RMSE (AIPW (logistic PS + GLM outcome)): 0.999 [0.957, 1.044], i.e. statistically indistinguishable, so the ranking within the top group is not robust. Estimators passing both checks: AIPW (GBM PS + GLM outcome) [ATE], AIPW (logistic PS + GLM outcome) [ATE], AIPW (GBM PS + GBM outcome) [ATE], AIPW (logistic PS + GBM outcome) [ATE], PS matching 1:1, caliper (logistic) [ATT].
- **Significant under-coverage (visit; Wilson upper bound < 0.95):** Naive difference in means [ATE], PS stratification, quintiles (GBM) [ATE].
- **No detectable bias on any of the three outcomes:** AIPW (GBM PS + GBM outcome) [ATE], AIPW (logistic PS + GBM outcome) [ATE], AIPW (GBM PS + GLM outcome) [ATE].
- **What the misspecified logistic PS costs.** Estimators relying on it alone (IPW, stratification, ATE and ATT matching) are significantly biased in 11 of 12 estimator-outcome cells (IPW: visit -2.5%, conversion -9.5%, spend -11.4%), but at n ~ 25,273 the bias is only 0.29 SD for visit, so CIs still cover at about the nominal rate (0.943). The danger grows with scale: with 10x the customers the same bias would cut visit coverage to about 0.85. IPW with the GBM PS shrinks the bias but does not remove it (visit -1.0% [-2.0, -0.1]): plug-in IPW with an ML propensity keeps some regularization bias, which AIPW's outcome-model correction cancels.
- **Double robustness rescues the misspecified PS when the outcome model is flexible.** AIPW keeping the logistic PS but adding a GBM outcome model has bias -0.3% [-1.2, +0.7] (visit), -2.0% [-5.5, +1.5] (conversion), -1.7% [-5.6, +2.3] (spend). With a GLM outcome model instead (both nuisances parametric) there is no detectable bias for site visit and conversion, but it remains for spend (-10.4% [-14.2, -6.5]): with both nuisance models misspecified, double robustness carries no guarantee.
- **Stratification leaves residual confounding regardless of the PS model** (visit bias +5.2% [+4.3, +6.1] logistic, +5.2% [+4.3, +6.1] GBM): five strata are too coarse for a propensity spanning 0.16-0.89.
- **Conversion (about 1% base rate):** RCT ATE 0.495 pp [0.355, 0.636]; naive bias +65.7% [+62.5, +68.8], coverage 0.280; best by the same rule: IPW, Hajek (GBM) (bias -0.4% [-3.8, +3.0], RMSE 0.148 pp, coverage 0.943). Even the best estimator's per-sample RMSE is 30% of the effect, so single observational conversion estimates are imprecise.

**Caveats**

- Semi-synthetic by design: the selection mechanism is known and depends only on observed X, so unconfoundedness holds exactly. Real targeting also uses signals the analyst never sees (e.g. browsing), which no estimator here can fix; this study isolates *modelling* error, not hidden confounding.
- The simulation's truth is the full-sample RCT estimate (bootstrap world; the oracle IPW's bias of +0.1% [-0.9, +1.0] confirms the benchmark is the right target). The RCT's own sampling error (visit CI half-width 0.54 pp) is not in the bias/coverage numbers; the covariate-adjusted RCT estimate differs from the difference in means by -0.017 pp.
- Bootstrap duplicates make ML nuisances slightly noisier than in a fresh sample of the same size (GBM PS RMSE vs truth: 0.039 in the real single sample vs 0.041-0.047 across replicates), so the simulation is if anything conservative for GBM-based estimators.
- Results are conditional on one targeting rule. Hillstrom outcomes are close to additive in these features, so the interactions the logistic PS omits carry little outcome signal; a rule keyed to outcome-relevant nonlinearities would make misspecification costlier.
- Standard errors treat the estimated PS as known: conservative for ATE IPW/matching with a parametric PS, no guarantee for the GBM PS or ATT matching. OLS with a treatment dummy targets a variance-weighted average of effects, which equals the ATE only if effects are homogeneous.
- 'Emailed' pools two creatives (Mens/Womens, 50/50 at every X), identically in the RCT and the observational samples. GBM hyperparameters were fixed up front, not tuned on the benchmark.
- Monte Carlo precision: 300 replicates pin coverage to about +/-2.5 pp; conversion is the least precise outcome. SE ratios share one set of replicates, so they drift together: the naive estimator's Welch SE (exact in theory) shows a ratio of 1.10 here but 1.002 [0.978, 1.027] over 3,000 cheap replicates with no model fitting, so SE ratios a few percent above 1 are Monte Carlo noise, not miscalibration.
