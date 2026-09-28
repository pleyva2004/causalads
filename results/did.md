## Difference-in-Differences on a geo experiment (GeoLift example panel)

**Method.** The customer-level Hillstrom data has no pre/post panel, so DiD uses Meta's GeoLift example panel: 40 US cities x 105 days (2021-01-01 to 2021-04-15), Y = daily conversions. Per the [GeoLift walkthrough](https://raw.githubusercontent.com/facebookincubator/GeoLift/main/vignettes/GeoLift_Walkthrough.Rmd), Chicago and Portland ran a $65,000, 15-day campaign (time stamps 91-105, 2021-04-01 to 2021-04-15); the other 38 cities are controls. The panel is balanced with no missing values. I estimate a 2x2 DiD, TWFE (location + date fixed effects), a ratio ("% lift") DiD, a synthetic control (non-negative weights summing to 1, fit on the 90 pre-days), and DiD against the 10 controls most correlated with the treated pair in the pre-period. **Inference is an exact randomization test**: "treatment" is reassigned to each of the 703 = C(38,2) pairs of control cities over the same window (re-fitting weights/matches for every pair), giving permutation p-values and test-inversion CIs. Estimates in the headline use the ratio DiD because city sizes differ ~5x (2.3k to 13.6k conversions/day), so a level-scale placebo distribution is dominated by big-city noise.

![Treated vs synthetic control](results/figures/did_synthetic_control.png)

### Headline

| Quantity | Estimate | 95% CI |
|---|---|---|
| DiD % lift | 7.2% | [-6.5, 25.5]% |
| DiD ATT (conversions per city-day) | 203.3 | [-208.5, 615.2] |
| Permutation p-value (two-sided) | 0.352 (min. attainable 0.0014) | |
| Pre-trend test, permutation p | 0.727 | Wald F p: 2.2e-14 (cluster, degenerate), 0.082 (HC1) |
| Incremental conversions, 15 days | 6,098 | [-6,255, 18,456] |
| Cost per incremental conversion | $10.66 | [$3.52, unbounded] |
| In-time placebo p (same pair, 76 pre-period windows) | 0.013 | observed 6.6 log pts; largest placebo window 5.5 |
| MDE at 80% power (this design) | 10.2% | |

### All estimators vs GeoLift's published result

| Method | ATT / city-day | 95% CI | % lift | 95% CI | Incremental (15 d) | p (kind) |
|---|---|---|---|---|---|---|
| 2x2 DiD, levels | 255.1 | [-357.4, 867.5] | 9.2% | n/a | 7,652 | 0.411 (permutation (703 placebo pairs)) |
| Ratio DiD, % lift (primary) | 203.3 | [-208.5, 615.2] | 7.2% | [-6.5, 25.5]% | 6,098 | 0.352 (permutation (703 placebo pairs)) |
| TWFE, levels | 255.1 | [-267.2, 777.3] | 9.2% | n/a | 7,652 | 0.338 (cluster-robust (2 treated clusters; unreliable)) |
| TWFE, log Y | n/a | n/a | 10.4% | [-7.2, 31.3]% | n/a | 0.263 (cluster-robust (unreliable)) |
| Synthetic control | 122.7 | [-261.9, 507.3] | 4.2% | [-8.0, 20.2]% | 3,682 | 0.480 (permutation (703 placebo pairs)) |
| DiD, top-10 correlated controls | 249.8 | [-152.5, 652.0] | 9.0% | [-4.8, 27.5]% | 7,494 | 0.217 (permutation (703 placebo pairs)) |
| GeoLift published (benchmark) | 155.6 | n/a | 5.4% | n/a | 4,667 | 0.010 (GeoLift conformal/permutation, as published) |

Cell means per city-day: treated 2917 pre / 3024 post; controls 4489 pre / 4341 post. Level-DiD CIs are wide because the level placebo distribution mixes cities of very different size. GeoLift's numbers are the vignette's rounded prose values (it gives no CI in text; it uses augmented synthetic control with fixed effects). The cross-city permutation test (which uses only untreated cities' noise) is much more conservative than GeoLift's published p = 0.01, which is based on the treated pair's own fitted series; the in-time placebo above is the closer analogue, but it only measures the pair's own day-to-day volatility (overlapping windows, so p is approximate) and ignores drift relative to the control pool. Cost per incremental conversion implied by GeoLift's 4,667: $13.93.

![Permutation distribution](results/figures/did_permutation.png)

### Parallel trends

Weekly event study on log conversions (reference = week -1). Joint test that the 11 pre-period lead coefficients are zero: permutation p = 0.727; Wald F(11, 4043) p = 2.2e-14 with cluster-robust SEs (degenerate: the clustered covariance rests on 2 treated clusters and should be ignored) and 0.082 with HC1 SEs. RMS lead coefficient: 3.16 log points. A non-rejection is weak evidence for parallel trends when power is limited, and the parametric tests are unreliable with 2 treated units, so the placebo band is the fairer yardstick.

![Event study](results/figures/did_event_study.png)

### Synthetic control

Pre-period RMSPE 97.7 conversions/city-day (3.35% of the treated mean); post-period gap 122.7 per city-day; post/pre RMSPE ratio 2.4 (placebo-rank p = 0.325). Largest donor weights: miami 0.21, cincinnati 0.21, baton rouge 0.16, dallas 0.11, honolulu 0.09.

### Power and minimum detectable effect

Using the untreated GeoLift_PreTest panel, I inject a multiplicative lift into 2 random pseudo-treated cities over a random 15-day window (pre-period = all earlier days, at least 45; effect grid 0-15%), and run the full ratio-DiD + 703-pair permutation test (500 simulations per effect size). Size at 0% lift: 5.0% (nominal 5%). **MDE at 80% power: 10.2%.**

| Injected lift | Power | 95% Wilson CI |
|---|---|---|
| 0% | 0.05 | [0.03, 0.07] |
| 1% | 0.05 | [0.03, 0.07] |
| 2% | 0.09 | [0.07, 0.12] |
| 3% | 0.14 | [0.11, 0.17] |
| 4% | 0.22 | [0.19, 0.26] |
| 5% | 0.33 | [0.29, 0.37] |
| 6% | 0.43 | [0.39, 0.48] |
| 7% | 0.53 | [0.49, 0.57] |
| 8% | 0.63 | [0.58, 0.67] |
| 9% | 0.71 | [0.67, 0.75] |
| 10% | 0.79 | [0.75, 0.82] |
| 11% | 0.84 | [0.80, 0.87] |
| 12% | 0.88 | [0.85, 0.91] |
| 13% | 0.90 | [0.87, 0.93] |
| 14% | 0.92 | [0.90, 0.94] |
| 15% | 0.94 | [0.92, 0.96] |

![Power curve](results/figures/did_power.png)

### Caveats

- **The panel is simulated example data published by Meta** ("simulated data of 40 US cities" per the GeoLift vignette), not a real advertiser's campaign. Treat all numbers as a methods demonstration.
- Only 2 treated cities: cluster-robust SEs (TWFE, event study Wald) are unreliable and are shown for completeness; the permutation test is the primary inference. With 703 placebo pairs the smallest attainable p-value is 0.0014.
- Treated cities were chosen by GeoLift's market-selection algorithm for fit, not randomized; the randomization test assumes exchangeability of city pairs, which selection on pre-period fit only approximately satisfies.
- Level DiD assumes additive parallel trends across cities of very different size; the ratio/log versions assume proportional trends and are preferred here.
- Test-inversion CIs assume a constant multiplicative effect over the window. Budget-based cost per conversion uses the vignette's stated $65,000 and ignores any baseline spend.
- The GeoLift benchmark is the rounded value quoted in the vignette prose, not a re-run of the R package; differences reflect method (fixed-effects augmented SCM vs. plain DiD/SCM).
- Power results are for this design on this panel (pre-period 45-75 days, 2 cities, 15 days) and use the same permutation test, with 500 simulations per effect size (Wilson CIs shown).
