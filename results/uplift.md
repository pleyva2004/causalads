## Uplift modeling and budget optimization: who should get the email, and which one?

**Method.** Five CATE learners for any email vs none — S-, T- and X-learners with LightGBM bases (early-stopped on validation), econml's DR-learner and CausalForestDML (3-fold cross-fitting) — are fit on the 38,400-customer train split and compared with a classic response model that ranks customers by P(visit | X, emailed). The propensity is supplied, not learned: assignment was randomized, so econml receives a prior-only classifier that returns each fold's assignment share (≈ 2/3 any email, 1/3 per arm); a covariate propensity model could only add noise. The learner used for targeting is chosen by validation-set normalized Qini (12,800 customers); the 12,800-customer test split is used once, with 2,000 paired bootstrap draws for CIs and refits over seeds 0, 1, 2, 3, 4 for model-randomness spread. Budgeted policies are scored off-policy with inverse-propensity weighting using the known 1/3 arm probabilities.

**Headline (test set, 95% CIs):**

- Val-selected learner **DR-learner**: normalized Qini 0.033 [0.006, 0.062] (5-seed 0.028 ± 0.003) vs response model 0.031 [0.002, 0.059]; AUUC gain 4.44 [0.98, 8.08] vs 4.09 [0.51, 7.67] extra visits / 1k. The margin over response targeting, 0.35 [-3.20, 3.74] visits / 1k, is **not statistically significant** (95% CI includes 0).
- BLP heterogeneity test: β2 = 0.62 [0.14, 1.11], one-sided p = 0.006 — the CATE ranking carries real heterogeneity.
- Top segment "bought women's AND past-year spend > \$364": visit uplift 10.6 [6.8, 14.4] pp; bottom segment "men's-only buyer AND not bought in last 3 months AND past-year spend ≤ \$255": 4.2 [2.0, 6.3] pp (equal-effects p = 0.003).
- **20% budget, per 1,000 customers.** Response policy: 19.2 [10.5, 28.4] visits, 2.11 [-0.47, 4.69] conversions, \$207 [-256, 690] revenue. Binary-uplift policy: 15.5 [7.0, 23.4] visits, 1.41 [-0.70, 3.52] conversions, \$175 [-211, 569]. Uplift vs response: -20 [-54, 31]% visits, -33 [-200, 200]% conversions, -16 [-503, 644]% revenue (CI includes 0 for visits). Multi-arm uplift vs response: -37 [-73, 7]% visits (CI includes 0).
- **30% budget, per 1,000 customers.** Response policy: 31.4 [21.1, 41.7] visits, 3.52 [0.47, 6.33] conversions, \$513 [-48, 1122] revenue. Binary-uplift policy: 22.7 [13.4, 31.9] visits, 1.64 [-0.70, 3.98] conversions, \$186 [-211, 588]. Uplift vs response: -28 [-52, 3]% visits, -53 [-150, 20]% conversions, -64 [-207, 145]% revenue (CI includes 0 for visits). Multi-arm uplift vs response: -27 [-54, 2]% visits (CI includes 0).

**Bottom line.** Emailing the top 30% ranked by the response model captures 41 [30, 52]% of the incremental visits of emailing everyone, using 30% of the sends (random targeting captures 30% by construction). Uplift models find real heterogeneity ("bought women's AND past-year spend > \$364": 10.6 [6.8, 14.4] pp vs 6.1 pp on average) but do not rank customers better than the simpler response model at a 30% budget (-8.7 [-18.3, 0.7] visits / 1k), so the extra modeling does not pay for itself in targeting on this campaign. For email choice, the multi-arm model agrees with the rule "Womens email to women's-only buyers, Mens otherwise" for 77% of customers and does not beat it (-0.5 [-8.9, 8.0] visits / 1k).

### 1. Ranking quality: CATE learners vs a response model

Outcome = visit, treatment = any email. Normalized Qini = area between the model's Qini curve and random, divided by the same area for a perfect ranking (0 = random). AUUC gain = average over targeting depths of extra incremental visits per 1,000 customers vs random targeting. BLP β2 is the Chernozhukov et al. test of whether the score carries real effect heterogeneity (β2 > 0; β2 = 1 = well calibrated). `*` = selected on validation.

| Learner | Val norm. Qini | Test norm. Qini [95% CI] | 5-seed mean ± sd | AUUC gain / 1k [95% CI] | Δ AUUC gain vs response [95% CI] | BLP β2 [95% CI] | β2 one-sided p |
|---|---|---|---|---|---|---|---|
| S-learner | 0.018 | 0.044 [0.016, 0.074] | 0.044 ± 0.006 | 5.72 [2.07, 9.52] | 1.63 [-1.19, 4.45] | 0.93 [0.33, 1.53] | 0.001 |
| T-learner | 0.016 | 0.027 [-0.001, 0.057] | 0.032 ± 0.004 | 3.58 [0.05, 7.37] | -0.51 [-4.32, 3.36] | 0.51 [0.01, 1.02] | 0.023 |
| X-learner | 0.018 | 0.037 [0.011, 0.066] | 0.040 ± 0.003 | 4.88 [1.53, 8.53] | 0.79 [-3.03, 4.46] | 1.24 [0.35, 2.13] | 0.003 |
| DR-learner * | 0.019 | 0.033 [0.006, 0.062] | 0.028 ± 0.003 | 4.44 [0.98, 8.08] | 0.35 [-3.20, 3.74] | 0.62 [0.14, 1.11] | 0.006 |
| Causal forest | 0.015 | 0.035 [0.008, 0.065] | 0.033 ± 0.003 | 4.57 [1.11, 8.36] | 0.48 [-3.23, 4.24] | 0.74 [0.18, 1.30] | 0.005 |
| Response model | 0.002 | 0.031 [0.002, 0.059] | 0.033 ± 0.001 | 4.09 [0.51, 7.67] | — | 0.23 [-0.01, 0.48] | 0.030 |

- **Heterogeneity is real for visits:** 5/5 CATE learners have BLP β2 > 0 at one-sided p < 0.05 on the held-out test set.
- **But uplift ranking does not beat response ranking:** no CATE learner's AUUC gain is significantly above the response model's (every paired 95% CI includes 0). The response model's own normalized Qini is 0.031 [0.002, 0.059] and its BLP β2 p = 0.030: customers most likely to visit when emailed are also, on average, the ones the email moves most.
- Validation-set Qini barely separates the learners (range 0.015–0.019); the val-selected learner is **DR-learner**, while the post-hoc test-best is S-learner (0.044). Headline numbers use the val-selected learner.

Conversion and spend (same learners, same protocol):

| Learner | Conversion norm. Qini [95% CI] | Conversion BLP β2 p | Spend norm. Qini [95% CI] | Spend BLP β2 p |
|---|---|---|---|---|
| S-learner | -0.066 [-0.181, 0.043] | 0.874 | 0.070 [-0.069, 0.202] | 0.488 |
| T-learner | -0.064 [-0.187, 0.068] | 0.767 | -0.146 [-0.281, -0.007] | 0.835 |
| X-learner | -0.059 [-0.181, 0.062] | 0.888 | 0.114 [-0.045, 0.277] | 0.135 |
| DR-learner | -0.037 [-0.152, 0.081] | 0.807 | -0.059 [-0.212, 0.095] | 0.734 |
| Causal forest | -0.071 [-0.190, 0.046] | 0.908 | -0.070 [-0.211, 0.075] | 0.858 |
| Response model | 0.046 [-0.076, 0.167] | 0.236 | -0.002 [-0.142, 0.135] | 0.473 |

For conversion 0/5 and for spend 0/5 learners show significant heterogeneity. With 24 control-group conversions in the test set, conversion and spend CATEs are not reliable and are not used for targeting claims.

![Uplift curves](results/figures/uplift_curves.png)

![Normalized Qini by learner](results/figures/uplift_learners.png)

### 2. Is the heterogeneity real? GATES and BLP

Test-set ATE of any email on visits: 6.09 [4.88, 7.31] pp. Ranking test customers by the DR-learner CATE, BLP β2 = 0.62 [0.14, 1.11] (one-sided p = 0.006; across 5 refit seeds β2 = 0.54 ± 0.05). Group average treatment effects (observed treated − control within each predicted quintile; valid because assignment is randomized inside every group):

| Quintile (1 = highest score) | Ranked by DR-learner: uplift pp [95% CI] | Ranked by response model: uplift pp [95% CI] |
|---|---|---|
| 1 | 8.4 [5.4, 11.4] | 8.9 [5.7, 12.2] |
| 2 | 8.0 [5.4, 10.6] | 7.1 [4.1, 10.0] |
| 3 | 5.3 [2.5, 8.1] | 4.9 [2.1, 7.6] |
| 4 | 6.0 [3.4, 8.6] | 6.1 [3.7, 8.5] |
| 5 | 3.2 [0.8, 5.7] | 4.0 [2.1, 5.8] |
| Top − bottom | 5.1 [1.3, 9.0] (p = 0.010) | 5.0 [1.2, 8.7] (p = 0.010) |

![Uplift by decile](results/figures/uplift_deciles_visit.png)

![GATES by outcome](results/figures/uplift_gates.png)

### 3. Who responds? Interpretable segments

Causal-forest heterogeneity importance (visit), grouped over redundant encodings: Product line (mens, womens, both) 48%, Past spend (history, log, bucket, floor) 32%, Recency (recency, recent_buyer) 9%.

![Causal forest importance](results/figures/uplift_cf_importance.png)

A depth-3 `SingleTreeCateInterpreter` tree was fit to the forest's CATEs on the **validation** set (min 1,000 customers per leaf); each leaf rule is then scored on the **test** set:

| Segment rule | Share | Forest CATE (pp) | Observed visit uplift, pp [95% CI] | Observed spend uplift, \$ [95% CI] |
|---|---|---|---|---|
| bought women's AND past-year spend > \$364 | 14% | 8.2 | 10.6 [6.8, 14.4] | 2.20 [0.46, 3.95] |
| bought women's AND \$38 < past-year spend ≤ \$364 | 34% | 7.4 | 8.1 [6.0, 10.1] | 0.52 [-0.30, 1.35] |
| bought women's AND past-year spend ≤ \$38 | 8% | 6.5 | 5.5 [1.4, 9.7] | 0.57 [-0.79, 1.94] |
| men's-only buyer AND past-year spend > \$255 | 12% | 5.5 | 1.9 [-1.9, 5.6] | -0.24 [-2.21, 1.72] |
| men's-only buyer AND bought in last 3 months AND past-year spend ≤ \$255 | 9% | 5.0 | 3.6 [-0.6, 7.9] | 1.42 [-0.20, 3.04] |
| men's-only buyer AND not bought in last 3 months AND past-year spend ≤ \$255 | 23% | 3.6 | 4.2 [2.0, 6.3] | -0.31 [-1.17, 0.56] |

Equal-effects test across segments (Cochran's Q): visit Q = 18.3, df = 5, p = 0.003; spend p = 0.120; conversion p = 0.280. Highest- minus lowest-predicted segment: 6.4 [2.1, 10.8] pp.

![Segment rules](results/figures/uplift_segments.png)

**Behavioral clusters.** KMeans on standardized recency, log_history, mens, womens, newbie, ch_web, ch_phone (fit on train); k = 5 maximizes the train silhouette (k=3: 0.239, k=4: 0.290, k=5: 0.320, k=6: 0.301, k=7: 0.301, k=8: 0.303).

| Cluster | Profile | Share (train) | Recency (mo) | Past spend (\$) | Visit uplift, pp [95% CI] | Spend uplift, \$ [95% CI] |
|---|---|---|---|---|---|---|
| C4 | multichannel, new, recent, high spend | 16% | 3.6 | 612 | 8.8 [5.5, 12.1] | 1.92 [0.25, 3.58] |
| C2 | women's only, web | 20% | 6.1 | 162 | 7.9 [5.1, 10.6] | 0.71 [-0.09, 1.52] |
| C0 | women's only, phone | 22% | 6.3 | 177 | 6.2 [3.8, 8.7] | 0.84 [0.26, 1.42] |
| C3 | men's only, phone | 20% | 6.0 | 160 | 4.5 [2.0, 7.0] | 0.22 [-0.95, 1.38] |
| C1 | men's only, web | 22% | 6.3 | 174 | 3.9 [1.2, 6.5] | -0.63 [-2.07, 0.81] |

Equal-effects test across clusters: visit p = 0.079 (Q = 8.4, df = 4), spend p = 0.184. Unsupervised clusters separate the email effect less sharply than the CATE-tree segments.

![Cluster effects](results/figures/uplift_clusters.png)

### 4. Which email? Multi-arm CATE

One causal forest over the 3-level arm gives per-customer CATEs for Mens vs none and Womens vs none (test means: 7.6 pp and 4.7 pp; BLP β2 one-sided p = 0.039 and <0.001). 76% of test customers are predicted to respond more to the Mens email and 24% to the Womens email (Womens share over 5 refit seeds: 23% ± 1.3%).

| Past purchases | Test customers | Predicted to prefer Womens email | Mean CATE Mens (pp) | Mean CATE Womens (pp) |
|---|---|---|---|---|
| Men's only | 5,634 | 0% | 6.9 | 1.4 |
| Women's only | 5,905 | 51% | 7.6 | 7.5 |
| Both lines | 1,261 | 7% | 10.7 | 5.9 |

The model's email choice agrees with a one-line rule ("Womens email to women's-only buyers, Mens otherwise") for 77% of customers. Randomized check among emailed test customers: sending the model's pick instead of the other email raises visits by 4.0 [2.4, 5.6] pp; the purchase rule achieves 4.2 [2.6, 5.8] pp. Emailing everyone (IPW, incremental outcomes per 1,000 customers vs no email):

| Policy (100% emailed) | Visits / 1k [95% CI] | Conversions / 1k [95% CI] | Revenue \$ / 1k [95% CI] |
|---|---|---|---|
| Everyone gets Mens | 76.4 [60.5, 92.3] | 6.80 [3.05, 11.02] | 805 [154, 1508] |
| Everyone gets Womens | 45.9 [30.7, 60.9] | 3.28 [-0.23, 6.80] | 288 [-265, 855] |
| Purchase-line rule | 80.9 [64.4, 96.8] | 6.80 [3.05, 10.78] | 780 [137, 1434] |
| Model's higher-CATE email | 80.4 [64.0, 96.6] | 6.80 [3.05, 10.55] | 841 [183, 1510] |

Model vs all-Mens: 4.0 [-4.5, 12.4] visits / 1k (p = 0.383); model vs purchase rule: -0.5 [-8.9, 8.0] (p = 0.938). The model’s advantage over the one-line rule is **not statistically significant** (95% CI includes 0).

![Email choice](results/figures/uplift_arm_choice.png)

### 5. Budget allocation: incremental outcomes under a fixed contact budget

Each policy emails a fixed share of test customers; policies that are not arm-aware send the email with the best average visit rate on train (Mens E-Mail). Values are Horvitz–Thompson IPW estimates on test, V(π) − V(no email) = mean[Y·(1{A=π(X)} − 1{A=0}) / (1/3)], unbiased because each arm had known probability 1/3. iROAS = incremental revenue / (contacts × \$0.10); the cost per email is an illustrative assumption, so break-even cost per contact is reported too. 95% bootstrap CIs (paired draws).

| Budget | Policy | Visits / 1k | Conversions / 1k | Revenue \$ / 1k | iROAS | Break-even \$ / contact |
|---|---|---|---|---|---|---|
| 10% | Random targeting | 7.6 [6.0, 9.2] | 0.68 [0.30, 1.10] | 80 [15, 151] | 8.0 [1.5, 15.1] | 0.80 [0.15, 1.51] |
| 10% | Response model (visit propensity if emailed), send Mens | 8.7 [1.6, 15.5] | 1.88 [-0.23, 3.98] | 140 [-206, 502] | 14.0 [-20.6, 50.2] | 1.40 [-2.06, 5.02] |
| 10% | Binary uplift (any-email CATE), send Mens | 4.5 [-1.6, 10.3] | 0.47 [-0.94, 2.11] | 57 [-270, 397] | 5.7 [-27.0, 39.7] | 0.57 [-2.70, 3.97] |
| 10% | Mens-email uplift (CATE Mens vs none), send Mens | 7.7 [1.4, 14.1] | 1.88 [0.00, 3.75] | 286 [-114, 726] | 28.6 [-11.4, 72.6] | 2.86 [-1.14, 7.26] |
| 10% | Multi-arm uplift, send higher-CATE email | 6.3 [0.0, 12.4] | 2.11 [0.47, 3.98] | 302 [-88, 735] | 30.2 [-8.8, 73.5] | 3.02 [-0.88, 7.35] |
| 10% | Revenue uplift (multi-arm CATE on spend), send higher-CATE email | 4.2 [-0.9, 9.4] | -0.94 [-2.34, 0.23] | -120 [-397, 124] | -12.0 [-39.7, 12.4] | -1.20 [-3.97, 1.24] |
| 10% | Response model (expected spend if emailed), send Mens | 3.0 [-1.9, 8.4] | 0.00 [-1.17, 1.17] | 125 [-113, 425] | 12.5 [-11.3, 42.5] | 1.25 [-1.13, 4.25] |
| 20% | Random targeting | 15.3 [12.1, 18.5] | 1.36 [0.61, 2.20] | 161 [31, 302] | 8.0 [1.5, 15.1] | 0.80 [0.15, 1.51] |
| 20% | Response model (visit propensity if emailed), send Mens | 19.2 [10.5, 28.4] | 2.11 [-0.47, 4.69] | 207 [-256, 690] | 10.4 [-12.8, 34.5] | 1.04 [-1.28, 3.45] |
| 20% | Binary uplift (any-email CATE), send Mens | 15.5 [7.0, 23.4] | 1.41 [-0.70, 3.52] | 175 [-211, 569] | 8.7 [-10.5, 28.5] | 0.87 [-1.05, 2.85] |
| 20% | Mens-email uplift (CATE Mens vs none), send Mens | 14.3 [6.1, 22.7] | 1.64 [-0.70, 3.98] | 238 [-179, 692] | 11.9 [-8.9, 34.6] | 1.19 [-0.89, 3.46] |
| 20% | Multi-arm uplift, send higher-CATE email | 12.2 [4.2, 20.4] | 1.88 [-0.23, 4.22] | 276 [-140, 722] | 13.8 [-7.0, 36.1] | 1.38 [-0.70, 3.61] |
| 20% | Revenue uplift (multi-arm CATE on spend), send higher-CATE email | 13.6 [6.3, 20.9] | 0.23 [-1.88, 2.34] | -111 [-474, 206] | -5.5 [-23.7, 10.3] | -0.55 [-2.37, 1.03] |
| 20% | Response model (expected spend if emailed), send Mens | 11.0 [3.3, 18.5] | 0.00 [-2.11, 1.88] | 101 [-230, 454] | 5.1 [-11.5, 22.7] | 0.51 [-1.15, 2.27] |
| 30% | Random targeting | 22.9 [18.1, 27.7] | 2.04 [0.91, 3.30] | 241 [46, 453] | 8.0 [1.5, 15.1] | 0.80 [0.15, 1.51] |
| 30% | Response model (visit propensity if emailed), send Mens | 31.4 [21.1, 41.7] | 3.52 [0.47, 6.33] | 513 [-48, 1122] | 17.1 [-1.6, 37.4] | 1.71 [-0.16, 3.74] |
| 30% | Binary uplift (any-email CATE), send Mens | 22.7 [13.4, 31.9] | 1.64 [-0.70, 3.98] | 186 [-211, 588] | 6.2 [-7.0, 19.6] | 0.62 [-0.70, 1.96] |
| 30% | Mens-email uplift (CATE Mens vs none), send Mens | 17.8 [8.0, 27.2] | 2.11 [-0.47, 4.69] | 195 [-279, 698] | 6.5 [-9.3, 23.3] | 0.65 [-0.93, 2.33] |
| 30% | Multi-arm uplift, send higher-CATE email | 23.0 [13.1, 32.8] | 1.88 [-0.47, 4.45] | 282 [-136, 741] | 9.4 [-4.5, 24.7] | 0.94 [-0.45, 2.47] |
| 30% | Revenue uplift (multi-arm CATE on spend), send higher-CATE email | 22.3 [12.9, 31.2] | 1.64 [-0.94, 3.98] | 127 [-301, 557] | 4.2 [-10.0, 18.6] | 0.42 [-1.00, 1.86] |
| 30% | Response model (expected spend if emailed), send Mens | 19.0 [9.6, 28.6] | 1.64 [-0.94, 4.22] | 374 [-124, 892] | 12.5 [-4.1, 29.7] | 1.25 [-0.41, 2.97] |
| 50% | Random targeting | 38.2 [30.2, 46.2] | 3.40 [1.52, 5.51] | 402 [77, 754] | 8.0 [1.5, 15.1] | 0.80 [0.15, 1.51] |
| 50% | Response model (visit propensity if emailed), send Mens | 42.2 [29.1, 55.1] | 5.39 [2.11, 8.91] | 698 [111, 1337] | 14.0 [2.2, 26.7] | 1.40 [0.22, 2.67] |
| 50% | Binary uplift (any-email CATE), send Mens | 34.9 [23.4, 46.4] | 3.05 [0.23, 6.09] | 308 [-119, 728] | 6.2 [-2.4, 14.6] | 0.62 [-0.24, 1.46] |
| 50% | Mens-email uplift (CATE Mens vs none), send Mens | 41.5 [29.8, 53.2] | 4.22 [1.17, 7.50] | 549 [-25, 1171] | 11.0 [-0.5, 23.4] | 1.10 [-0.05, 2.34] |
| 50% | Multi-arm uplift, send higher-CATE email | 43.8 [31.9, 55.5] | 3.75 [0.70, 6.80] | 520 [-62, 1139] | 10.4 [-1.2, 22.8] | 1.04 [-0.12, 2.28] |
| 50% | Revenue uplift (multi-arm CATE on spend), send higher-CATE email | 33.0 [21.3, 45.2] | 2.81 [-0.23, 5.86] | 210 [-317, 739] | 4.2 [-6.3, 14.8] | 0.42 [-0.63, 1.48] |
| 50% | Response model (expected spend if emailed), send Mens | 35.2 [23.4, 47.3] | 3.75 [0.70, 7.03] | 542 [-47, 1183] | 10.8 [-0.9, 23.7] | 1.08 [-0.09, 2.37] |

Paired differences vs the response-model policy (uplift − response, same bootstrap draws; the last column is the mean ± sd of the visit difference over 5 refit seeds):

| Budget | Uplift policy | Δ visits / 1k [95% CI] | % change in visits [95% CI] | Δ conversions / 1k [95% CI] | Δ revenue \$ / 1k [95% CI] | Seeds: Δ visits / 1k |
|---|---|---|---|---|---|---|
| 10% | Binary uplift (any-email CATE), send Mens | -4.2 [-10.8, 2.6] | -49 [-138, 65] | -1.41 [-3.52, 0.70] | -83 [-306, 140] | -3.1 ± 1.4 |
| 10% | Mens-email uplift (CATE Mens vs none), send Mens | -0.9 [-7.3, 5.9] | -11 [-82, 200] | 0.00 [-1.88, 1.88] | 146 [-93, 469] | -0.7 ± 2.1 |
| 10% | Multi-arm uplift, send higher-CATE email | -2.3 [-9.1, 4.5] | -27 [-106, 136] | 0.23 [-1.88, 2.34] | 162 [-131, 532] | -2.6 ± 1.8 |
| 20% | Binary uplift (any-email CATE), send Mens | -3.8 [-11.7, 4.2] | -20 [-54, 31] | -0.70 [-2.81, 1.41] | -33 [-419, 317] | -4.1 ± 2.2 |
| 20% | Mens-email uplift (CATE Mens vs none), send Mens | -4.9 [-12.4, 3.1] | -26 [-60, 22] | -0.47 [-2.58, 1.41] | 31 [-390, 420] | -5.4 ± 1.3 |
| 20% | Multi-arm uplift, send higher-CATE email | -7.0 [-15.2, 0.9] | -37 [-73, 7] | -0.23 [-2.58, 2.11] | 68 [-369, 508] | -6.6 ± 1.5 |
| 30% | Binary uplift (any-email CATE), send Mens | -8.7 [-18.3, 0.7] | -28 [-52, 3] | -1.88 [-4.22, 0.47] | -328 [-839, 136] | -7.4 ± 1.9 |
| 30% | Mens-email uplift (CATE Mens vs none), send Mens | -13.6 [-22.7, -4.5] | -43 [-70, -16] | -1.41 [-3.52, 0.70] | -318 [-719, 35] | -9.0 ± 3.0 |
| 30% | Multi-arm uplift, send higher-CATE email | -8.4 [-18.3, 0.7] | -27 [-54, 2] | -1.64 [-3.98, 0.94] | -232 [-708, 217] | -8.2 ± 2.3 |
| 50% | Binary uplift (any-email CATE), send Mens | -7.3 [-16.4, 1.6] | -17 [-36, 5] | -2.34 [-4.45, -0.47] | -391 [-896, 40] | -6.4 ± 1.5 |
| 50% | Mens-email uplift (CATE Mens vs none), send Mens | -0.7 [-10.1, 8.0] | -2 [-21, 23] | -1.17 [-3.75, 1.17] | -149 [-414, 96] | -1.7 ± 1.2 |
| 50% | Multi-arm uplift, send higher-CATE email | 1.6 [-9.1, 12.2] | 4 [-19, 36] | -1.64 [-4.22, 1.17] | -179 [-491, 126] | -2.5 ± 2.8 |

**No uplift policy significantly beats response targeting on visits at any budget.** Response targeting is significantly better (CI excludes 0) for: Mens-email uplift (CATE Mens vs none), send Mens at 30%. Revenue-optimized policy (multi-arm causal forest on spend) vs a spend response model, Δ revenue \$ / 1k: 10%: -245 [-601, 23]; 20%: -212 [-628, 133]; 30%: -246 [-684, 158]; 50%: -332 [-767, 50]. Spend CATEs carry no detectable signal (Section 1), so that policy is tabulated but not recommended.

![Budget curves](results/figures/uplift_budget.png)

**Caveats**

- Conversion is rare (test control conversion rate 0.56%, ≈24 control conversions) and spend is zero for ≈99% of emailed customers; their CATE, Qini and revenue CIs are wide and mostly span 0. Only visit supports heterogeneity claims.
- The pooled any-email CATE averages two different emails; ranking by it and then sending the Mens email is an estimand/action mismatch (its top customers skew to women's-only buyers). The Mens-email uplift policy was added after diagnosing this, so it is a post-hoc comparison, not a pre-specified one.
- Learner selection on validation is noisy (val normalized Qini spans 0.015–0.019); test CIs overlap for all learners.
- One fixed train/val/test split. Chernozhukov et al. recommend repeating BLP/GATES over many splits and taking medians; seed refits here vary model randomness only, not the split.
- The Horvitz–Thompson IPW estimator is unbiased but high-variance, and many policy × budget × outcome comparisons are reported; treat any single nominally significant cell outside the pre-specified visit comparison with caution.
- Cost per email (\$0.10) is illustrative; iROAS scales inversely with it, break-even cost does not depend on it. Data are one 2008 two-week campaign; effects may not transfer to other creatives, seasons or lists.
