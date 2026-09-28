## Prediction: who is likely to convert?

Supervised models score each customer's probability of converting (primary target, base rate 0.91% in train) or visiting (secondary, 14.7%). Features are the 16 pre-treatment features plus two exposure indicators (Men's / Women's e-mail sent), so attribution can show the e-mail effect. Splits are the fixed stratified train/val/test = 38,400/12,800/12,800. Models are sklearn Pipelines (median imputer, plus a standard scaler for logistic regression) so no preprocessing leaks across folds. Hyperparameters were tuned with Optuna (TPE, seed 42; logistic-regression C by grid) on **train only** using stratified 5-fold CV and PR-AUC. The 'best' model is chosen by **validation** PR-AUC (**LightGBM**); calibrators are fit on validation predictions; the test split is scored once. All test intervals are percentile bootstrap 95% CIs (2000 resamples of test rows). This is a *predictive* stage: high scores identify people who convert, not people the e-mail causes to convert (see the uplift stage).

### Tuned configuration (train CV PR-AUC)

| Model | CV PR-AUC (tuned, conversion) | Best hyperparameters (conversion) |
|---|---|---|
| Logistic Regression | 0.0183 (6 grid points) | C=0.001 |
| Random Forest | 0.0159 (20 trials) | n_estimators=250, max_depth=3, min_samples_leaf=17, max_features=0.3707 |
| XGBoost | 0.0157 (30 trials) | n_estimators=200, learning_rate=0.01388, max_depth=3, min_child_weight=1.116, subsample=0.7791, colsample_bytree=0.8276, reg_lambda=27.98 |
| LightGBM | 0.0162 (30 trials) | n_estimators=150, learning_rate=0.01137, num_leaves=11, min_child_samples=25, subsample=0.616, colsample_bytree=0.5399, reg_lambda=16.33, subsample_freq=1 |

### Test performance: Conversion (primary)

Test base rate 0.90% (115 positives out of 12,800). A random scorer has PR-AUC equal to the base rate.

| Model | ROC-AUC | PR-AUC | Brier | Log loss | Top-10% lift (x) |
|---|---|---|---|---|---|
| Logistic Regression | 0.648 [0.598, 0.699] | 0.0180 [0.0129, 0.0279] | 0.0089 [0.0074, 0.0106] | 0.0504 [0.0435, 0.0582] | 2.26 [1.51, 3.02] |
| Random Forest | 0.640 [0.587, 0.692] | 0.0194 [0.0136, 0.0332] | 0.0089 [0.0074, 0.0105] | 0.0502 [0.0433, 0.0578] | 2.78 [1.96, 3.62] |
| XGBoost | 0.655 [0.608, 0.706] | 0.0185 [0.0134, 0.0288] | 0.0089 [0.0074, 0.0105] | 0.0501 [0.0431, 0.0577] | 2.26 [1.54, 3.07] |
| LightGBM * | 0.650 [0.601, 0.699] | 0.0183 [0.0132, 0.0280] | 0.0089 [0.0074, 0.0106] | 0.0503 [0.0433, 0.0579] | 2.17 [1.42, 2.98] |

\* best on validation PR-AUC. Paired bootstrap, LightGBM minus Logistic Regression: ΔROC-AUC = 0.001 [-0.024, 0.028] (two-sided p = 0.931); ΔPR-AUC = 0.0003 [-0.0061, 0.0065] (p = 0.944).

Repeated stratified 5x3 CV on train+val (same tuned configs), mean ± SD over 15 folds:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.624 ± 0.029 | 0.0173 ± 0.0042 |
| Random Forest | 0.618 ± 0.027 | 0.0154 ± 0.0031 |
| XGBoost | 0.624 ± 0.025 | 0.0152 ± 0.0021 |
| LightGBM | 0.621 ± 0.025 | 0.0151 ± 0.0020 |

![CV spread](results/figures/prediction_cv_spread.png)

![ROC and PR curves](results/figures/prediction_roc_pr.png)

### Test performance: Visit (secondary)

Test base rate 14.67% (1878 positives out of 12,800). A random scorer has PR-AUC equal to the base rate.

| Model | ROC-AUC | PR-AUC | Brier | Log loss | Top-10% lift (x) |
|---|---|---|---|---|---|
| Logistic Regression | 0.654 [0.641, 0.667] | 0.2362 [0.2216, 0.2536] | 0.1205 [0.1164, 0.1246] | 0.3983 [0.3878, 0.4087] | 1.88 [1.73, 2.03] |
| Random Forest | 0.651 [0.638, 0.664] | 0.2341 [0.2195, 0.2502] | 0.1207 [0.1165, 0.1249] | 0.3987 [0.3881, 0.4091] | 1.89 [1.74, 2.03] |
| XGBoost * | 0.656 [0.643, 0.669] | 0.2369 [0.2218, 0.2538] | 0.1205 [0.1163, 0.1246] | 0.3983 [0.3877, 0.4087] | 1.84 [1.69, 2.00] |
| LightGBM | 0.656 [0.644, 0.669] | 0.2371 [0.2222, 0.2539] | 0.1205 [0.1162, 0.1246] | 0.3979 [0.3873, 0.4084] | 1.85 [1.70, 1.99] |

\* best on validation PR-AUC. Paired bootstrap, XGBoost minus Logistic Regression: ΔROC-AUC = 0.002 [-0.001, 0.004] (two-sided p = 0.196); ΔPR-AUC = 0.0007 [-0.0034, 0.0045] (p = 0.796).

Repeated stratified 5x3 CV on train+val (same tuned configs), mean ± SD over 15 folds:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.645 ± 0.006 | 0.2284 ± 0.0069 |
| Random Forest | 0.641 ± 0.004 | 0.2258 ± 0.0061 |
| XGBoost | 0.646 ± 0.005 | 0.2298 ± 0.0065 |
| LightGBM | 0.646 ± 0.004 | 0.2296 ± 0.0064 |

### Class imbalance: reweighting does not improve ranking and wrecks probabilities

Same tuned hyperparameters, three treatments of the ~0.9% positive class, 5-fold CV on train (mean ± SD across folds). The last column is mean predicted probability divided by the true base rate: 1.0 means the average score is an honest probability.

| Model | Strategy | ROC-AUC | PR-AUC | Brier | ECE | Mean pred / base rate |
|---|---|---|---|---|---|---|
| Logistic Regression | No reweighting | 0.613 ± 0.030 | 0.0183 ± 0.0048 | 0.0090 ± 0.0001 | 0.0032 ± 0.0007 | 1.00 ± 0.01 |
| Logistic Regression | Class weight | 0.625 ± 0.029 | 0.0188 ± 0.0075 | 0.2330 ± 0.0026 | 0.4603 ± 0.0044 | 51.80 ± 0.81 |
| Logistic Regression | Undersample 10:1 | 0.610 ± 0.027 | 0.0181 ± 0.0047 | 0.0157 ± 0.0001 | 0.0814 ± 0.0003 | 9.98 ± 0.10 |
| Random Forest | No reweighting | 0.609 ± 0.032 | 0.0152 ± 0.0015 | 0.0090 ± 0.0001 | 0.0028 ± 0.0007 | 1.00 ± 0.01 |
| Random Forest | Class weight | 0.611 ± 0.034 | 0.0150 ± 0.0034 | 0.2210 ± 0.0024 | 0.4548 ± 0.0036 | 51.19 ± 0.72 |
| Random Forest | Undersample 10:1 | 0.607 ± 0.033 | 0.0174 ± 0.0081 | 0.0163 ± 0.0001 | 0.0814 ± 0.0007 | 9.98 ± 0.13 |
| XGBoost | No reweighting | 0.621 ± 0.026 | 0.0149 ± 0.0033 | 0.0090 ± 0.0001 | 0.0032 ± 0.0006 | 0.98 ± 0.02 |
| XGBoost | Class weight | 0.617 ± 0.025 | 0.0144 ± 0.0028 | 0.2156 ± 0.0022 | 0.4439 ± 0.0036 | 49.99 ± 0.74 |
| XGBoost | Undersample 10:1 | 0.614 ± 0.028 | 0.0141 ± 0.0019 | 0.0160 ± 0.0001 | 0.0794 ± 0.0007 | 9.77 ± 0.13 |
| LightGBM | No reweighting | 0.618 ± 0.029 | 0.0147 ± 0.0034 | 0.0090 ± 0.0001 | 0.0029 ± 0.0004 | 0.98 ± 0.01 |
| LightGBM | Class weight | 0.605 ± 0.029 | 0.0141 ± 0.0026 | 0.0927 ± 0.0460 | 0.2646 ± 0.1021 | 30.15 ± 11.17 |
| LightGBM | Undersample 10:1 | 0.609 ± 0.030 | 0.0142 ± 0.0026 | 0.0156 ± 0.0001 | 0.0794 ± 0.0006 | 9.76 ± 0.13 |

![Imbalance strategies](results/figures/prediction_imbalance.png)

### Calibration (conversion, test)

Calibrators are fit on the validation predictions of models trained on train, then scored on test. ECE uses 10 equal-mass bins (about 1280 customers, ~12 conversions per bin), so it is noisy and biased upward for small samples (bootstrap ECE CIs can therefore sit above the point estimate). Brier differences between variants are far inside the CIs; ECE differences are not resolvable either.

| Model | Calibration | Brier | ECE | Mean pred (base rate 0.90%) |
|---|---|---|---|---|
| Logistic Regression | None | 0.00889 [0.00736, 0.01049] | 0.0026 [0.0019, 0.0045] | 0.91% |
| Logistic Regression | Platt | 0.00888 [0.00737, 0.01046] | 0.0011 [0.0013, 0.0034] | 0.89% |
| Logistic Regression | Isotonic | 0.00889 [0.00737, 0.01047] | 0.0013 [0.0013, 0.0035] | 0.89% |
| Random Forest | None | 0.00888 [0.00737, 0.01046] | 0.0024 [0.0018, 0.0046] | 0.91% |
| Random Forest | Platt | 0.00888 [0.00737, 0.01046] | 0.0024 [0.0018, 0.0046] | 0.89% |
| Random Forest | Isotonic | 0.00888 [0.00737, 0.01046] | 0.0028 [0.0017, 0.0041] | 0.89% |
| XGBoost | None | 0.00888 [0.00736, 0.01048] | 0.0019 [0.0017, 0.0040] | 0.89% |
| XGBoost | Platt | 0.00888 [0.00736, 0.01048] | 0.0020 [0.0017, 0.0041] | 0.89% |
| XGBoost | Isotonic | 0.00889 [0.00738, 0.01048] | 0.0019 [0.0017, 0.0039] | 0.90% |
| LightGBM | None | 0.00888 [0.00736, 0.01048] | 0.0023 [0.0018, 0.0043] | 0.89% |
| LightGBM | Platt | 0.00888 [0.00736, 0.01048] | 0.0024 [0.0018, 0.0044] | 0.89% |
| LightGBM | Isotonic | 0.00888 [0.00738, 0.01045] | 0.0018 [0.0015, 0.0035] | 0.90% |

![Calibration](results/figures/prediction_calibration.png)

### Targeting: LightGBM, test

Decision rule (fixed before looking at test): contact customers whose score is >= the VAL 80th percentile of the best model's scores (top 20%); threshold = 0.0111. It flags 20.0% of test customers.

| | Converted | Did not convert |
|---|---|---|
| Flagged | 46 (TP) | 2,511 (FP) |
| Not flagged | 69 (FN) | 10,174 (TN) |

Recall 40.00% [30.91%, 49.02%]; precision 1.80% [1.29%, 2.31%] (base rate 0.90%).

| Decile | Conversion rate (95% CI) | Lift (95% CI) | Conversions | Cumulative share of conversions |
|---|---|---|---|---|
| 1 | 1.95% [1.25, 2.73] | 2.17 [1.39, 3.04] | 25 | 22% |
| 2 | 1.64% [1.02, 2.42] | 1.83 [1.13, 2.70] | 21 | 40% |
| 3 | 0.94% [0.39, 1.56] | 1.04 [0.43, 1.74] | 12 | 50% |
| 4 | 0.78% [0.31, 1.33] | 0.87 [0.35, 1.48] | 10 | 59% |
| 5 | 1.09% [0.55, 1.64] | 1.22 [0.61, 1.83] | 14 | 71% |
| 6 | 0.70% [0.31, 1.17] | 0.78 [0.35, 1.30] | 9 | 79% |
| 7 | 0.55% [0.16, 0.94] | 0.61 [0.17, 1.04] | 7 | 85% |
| 8 | 0.55% [0.23, 1.02] | 0.61 [0.26, 1.13] | 7 | 91% |
| 9 | 0.39% [0.08, 0.70] | 0.43 [0.09, 0.78] | 5 | 96% |
| 10 | 0.39% [0.08, 0.78] | 0.43 [0.09, 0.87] | 5 | 100% |

![Lift and gains](results/figures/prediction_gains.png)

### Attribution (LightGBM, best tree model)

| Feature | Mean abs SHAP (log-odds) | Permutation importance, PR-AUC drop (mean ± SD) |
|---|---|---|
| Men's e-mail sent | 0.1177 | 0.00373 ± 0.00103 |
| history ($) | 0.0977 | 0.00078 ± 0.00257 |
| recency | 0.0839 | 0.00099 ± 0.00192 |
| new customer | 0.0332 | 0.00161 ± 0.00066 |
| log history | 0.0321 | -0.00082 ± 0.00229 |
| channel phone | 0.0310 | -0.00036 ± 0.00108 |
| zip rural | 0.0273 | 0.00014 ± 0.00032 |
| recent buyer | 0.0214 | 0.00003 ± 0.00096 |

Top-3 by SHAP: Men's e-mail sent, history ($), recency. Top-3 by permutation: Men's e-mail sent, new customer, recency (2/3 overlap). Spearman correlation of the two rankings across all 18 features: 0.53 (p = 0.024). SHAP used 6,000 validation rows; permutation importance used all 12,800 validation rows, which hold only ~115 conversions, so its SDs are large. Correlated features (history / log history / bucket, zip dummies, channel dummies) split credit, so single-feature rankings understate the importance of the underlying concept.

![SHAP beeswarm](results/figures/prediction_shap_beeswarm.png)

![Importance cross-check](results/figures/prediction_importance.png)

### Error analysis (conversion, test)

| Subgroup | n | Conversions | Observed rate (95% CI) | Mean predicted (95% CI) | ROC-AUC (95% CI; shown if >=10 conversions) |
|---|---|---|---|---|---|
| channel: Multichannel | 1,544 | 14 | 0.91% [0.45, 1.46] | 1.16% [1.14, 1.17] | 0.675 [0.561, 0.784] |
| channel: Phone | 5,571 | 46 | 0.83% [0.61, 1.06] | 0.82% [0.81, 0.83] | 0.664 [0.580, 0.748] |
| channel: Web | 5,685 | 55 | 0.97% [0.72, 1.25] | 0.89% [0.89, 0.90] | 0.639 [0.562, 0.716] |
| zip_code: Rural | 1,903 | 17 | 0.89% [0.47, 1.26] | 0.98% [0.97, 1.00] | 0.448 [0.314, 0.600] |
| zip_code: Suburban | 5,781 | 46 | 0.80% [0.57, 1.02] | 0.88% [0.87, 0.88] | 0.686 [0.616, 0.764] |
| zip_code: Urban | 5,116 | 52 | 1.02% [0.76, 1.27] | 0.88% [0.87, 0.88] | 0.668 [0.590, 0.751] |
| newbie: existing | 6,387 | 64 | 1.00% [0.76, 1.25] | 0.90% [0.89, 0.91] | 0.623 [0.546, 0.693] |
| newbie: newbie | 6,413 | 51 | 0.80% [0.60, 1.01] | 0.88% [0.88, 0.89] | 0.681 [0.604, 0.758] |
| history_bucket: 1 | 4,618 | 33 | 0.71% [0.48, 0.95] | 0.77% [0.76, 0.77] | 0.602 [0.505, 0.693] |
| history_bucket: 2 | 2,814 | 20 | 0.71% [0.43, 1.07] | 0.75% [0.74, 0.75] | 0.655 [0.557, 0.758] |
| history_bucket: 3 | 2,492 | 20 | 0.80% [0.44, 1.20] | 0.99% [0.98, 1.00] | 0.675 [0.534, 0.817] |
| history_bucket: 4 | 1,279 | 15 | 1.17% [0.55, 1.88] | 1.12% [1.10, 1.13] | 0.644 [0.494, 0.756] |
| history_bucket: 5 | 977 | 12 | 1.23% [0.61, 1.94] | 1.15% [1.13, 1.16] | 0.574 [0.424, 0.720] |
| history_bucket: 6 | 370 | 7 | 1.89% [0.81, 3.24] | 1.23% [1.20, 1.25] | n/a |
| history_bucket: 7 | 250 | 8 | 3.20% [1.20, 5.20] | 1.29% [1.26, 1.31] | n/a |
| arm: Mens E-Mail | 4,261 | 53 | 1.24% [0.95, 1.56] | 1.05% [1.05, 1.06] | 0.656 [0.581, 0.725] |
| arm: No E-Mail | 4,261 | 24 | 0.56% [0.36, 0.80] | 0.80% [0.79, 0.81] | 0.594 [0.474, 0.706] |
| arm: Womens E-Mail | 4,278 | 38 | 0.89% [0.64, 1.17] | 0.82% [0.82, 0.83] | 0.628 [0.538, 0.716] |

![Subgroup calibration](results/figures/prediction_subgroups.png)

**Who the model misses.** At the top-20% rule, 69 of 115 test converters are not flagged. Mean features of caught vs missed converters (median spend: $94.10 caught vs $87.40 missed):

| Feature | Caught converters | Missed converters |
|---|---|---|
| recency | 2.804 | 5.841 |
| history ($) | 582.370 | 200.198 |
| new customer | 0.478 | 0.420 |
| recent buyer | 0.761 | 0.333 |
| zip urban | 0.500 | 0.420 |
| zip suburban | 0.457 | 0.362 |
| zip rural | 0.043 | 0.217 |
| channel phone | 0.304 | 0.464 |
| channel web | 0.413 | 0.522 |
| channel multi | 0.283 | 0.014 |
| bought men's | 0.696 | 0.464 |
| bought women's | 0.696 | 0.623 |
| Men's e-mail sent | 0.696 | 0.304 |
| Women's e-mail sent | 0.152 | 0.449 |
| No e-mail sent | 0.152 | 0.246 |

### Caveats

- Only 115 test conversions: every conversion CI above is wide, and most model-to-model differences are within noise. Visits (1878 positives) are much better powered.
- Model selection used validation PR-AUC on ~115 validation conversions, so the 'best' model is partly a coin flip among near-equals; the paired bootstrap on test is the honest comparison.
- Calibrators (especially isotonic) are fit on ~115 validation conversions and can overfit; test ECE/Brier show whether they helped.
- Scores mix baseline propensity and e-mail response. Exposure is a feature, so 'likely to convert' is not 'persuadable'; targeting by these scores is not causal targeting.
- 2 weeks of outcomes, one 2008 retailer, history floor-censored at $29.99 for 12.4% of customers; hyperparameter tuning saw only train folds, but the same test set is shared across pipeline stages by design (never used for selection here).
- SHAP explains the model, not the world: attribution to correlated features is arbitrary between them.

