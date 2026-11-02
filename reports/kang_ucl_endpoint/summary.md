# Extension 1 — UCL endpoint specificity

Completed comparison against the frozen Kang replication. Endpoint/adjudication and resulting case membership/reference dates changed; model, predictors, splitting algorithm, selection, window search and ROC-AUC procedure did not.

## Cohort and endpoint overlap

| outcome   | role     |   games |   pitcher_seasons |   unique_pitchers |   injury_events |   injured_pitchers |   control_games |   case_games |
|:----------|:---------|--------:|------------------:|------------------:|----------------:|-------------------:|----------------:|-------------:|
| elbow     | starter  |   34477 |              1665 |               620 |             123 |                106 |           32975 |         1502 |
| elbow     | reliever |  119590 |              3669 |              1418 |             233 |                208 |          114349 |         5241 |
| ucl       | starter  |   33218 |              1567 |               599 |              36 |                 36 |           32853 |          365 |
| ucl       | reliever |  114915 |              3466 |              1389 |              34 |                 34 |          114304 |          611 |

The benchmark contains **356 broad elbow events**. **55 (15.4%)** remain UCL cases. The same first-event rule also admits **15 confirmed forearm/UCL cases**, yielding **70 UCL events**. The established source contains 150 accepted 2017–2025 UCL events before benchmark eligibility/first-event filtering. Non-UCL elbow cases are excluded, not reclassified as healthy controls. The endpoint-overlap and all-source mapping tables give event-level reasons.

## Prevalence at the common 30-day candidate window

| outcome   | role     |   games |   positive_games |   negative_games |   prevalence |   represented_events |
|:----------|:---------|--------:|-----------------:|-----------------:|-------------:|---------------------:|
| elbow     | starter  |   34477 |              552 |            33925 |     0.016011 |                  120 |
| elbow     | reliever |  119590 |             1916 |           117674 |     0.016021 |                  230 |
| ucl       | starter  |   33218 |              186 |            33032 |     0.005599 |                   36 |
| ucl       | reliever |  114915 |              285 |           114630 |     0.00248  |                   34 |

This common existing candidate window describes class composition; it is not an additional window-selection rule. All 43 windows and earlier-case negatives appear in `tables/window_support.csv`.

## Selected-window results

| outcome   | role     |   cohort_events |   selected_window |   band_lower |   band_upper |   band_segments | selected_day_in_band   |   roc_auc |   ci_low |   ci_high |   test_events |   shared_positive_events |   features | ucl_vs_elbow_window      |
|:----------|:---------|----------------:|------------------:|-------------:|-------------:|----------------:|:-----------------------|----------:|---------:|----------:|--------------:|-------------------------:|-----------:|:-------------------------|
| elbow     | starter  |             123 |                28 |        26.88 |        29.39 |               1 | True                   |    0.9245 |   0.9048 |    0.9442 |           100 |                       99 |         18 | ambiguous_or_unsupported |
| ucl       | starter  |              36 |                26 |        18.26 |        33.98 |               3 | False                  |    0.9559 |   0.9269 |    0.9877 |            30 |                       29 |         18 | ambiguous_or_unsupported |
| elbow     | reliever |             233 |                25 |        23.69 |        26.33 |               1 | True                   |    0.9097 |   0.891  |    0.9256 |           208 |                      206 |         18 | shorter                  |
| ucl       | reliever |              34 |                17 |        15.23 |        19.06 |               1 | True                   |    0.9868 |   0.9797 |    0.9929 |            29 |                       29 |         18 | shorter                  |

**Starters:** UCL AUC 0.956 (95% cluster interval 0.927–0.988), versus elbow 0.925; higher by 0.031. Selected integer window 26 days; band 18.26–33.98 (3 segment(s), midpoint in band: False). Window relation: ambiguous_or_unsupported. 30 UCL events contribute positive held-out games.

Shared top-five predictors: age_at_appearance, kang_release_abs_x_avg, release_z_avg, extension_avg. Rank changes (positive = lower UCL rank): avg_velocity (+4 ranks), kang_velocity_sd (+3 ranks), spin_avg (+3 ranks), kang_spin_max (-4 ranks), kang_spin_sd (+5 ranks), ff_usage_pct (-4 ranks), st_usage_pct (-10 ranks), ch_usage_pct (+3 ranks). Newly selected: none. No longer selected: none.

**Relievers:** UCL AUC 0.987 (95% cluster interval 0.980–0.993), versus elbow 0.910; higher by 0.077. Selected integer window 17 days; band 15.23–19.06 (1 segment(s), midpoint in band: True). Window relation: shorter. 29 UCL events contribute positive held-out games.

Shared top-five predictors: age_at_appearance, kang_release_abs_x_avg, release_z_avg, extension_avg. Rank changes (positive = lower UCL rank): none with a ≥3-rank shift among shared selected features. Newly selected: spin_avg. No longer selected: kang_spin_max.

The intervals on ROC-AUC use the same 500 held-out pitcher-season resamples as the benchmark. The slope bands are not confidence intervals. A disconnected envelope or midpoint outside the band does not establish a unique optimal preinjury window.

## Interpretation and decision

Ranking signal persists for both roles under the same random-game design: both UCL AUC intervals exceed 0.5. This is conditional discrimination, not evidence of ahead-of-time clinical usefulness or generalization to future seasons.

Higher or lower AUC is not automatically improvement or deterioration of a warning system. Fewer and different cases, their timing, prevalence-derived class weights and changed random-game membership all accompany the endpoint change. No paired significance test or operational-warning claim is made. Gain describes model reliance, not tissue damage or causation.

The reduced independent-event count limits precision and makes feature ranks/window shapes sensitive to case composition. Shared pitchers across random train/test games remain intentional. Broad-control unknown diagnoses remain as in the benchmark. The source is retrospective operational adjudication, not an independent clinical review.

The primary table also reports injury events with positive games in **both** training and test. The held-out games are not held-out injury events. Strong AUC can therefore reflect recognizable pitcher/event context; these results cannot isolate short-term warning information from persistent profile information.

There is enough technical support to specify **Extension 2: chronological validation**, with event-support checks and the present benchmark frozen. These random-game findings do not predict that chronological performance will persist. Do not tune this endpoint experiment to make that later comparison favorable. Extension 2 has not been run.

## Reproducibility

The original R script is reused without edits. `experiment_freeze.json` hashes all benchmark artifacts and the authoritative endpoint. `tables/benchmark_equivalence.csv` verifies fixed settings; `tables/validation_checks.csv` independently verifies labels, splits and AUC. Only the two UCL model tasks are fitted. The automatic prior-broad-elbow-history sensitivity is supplemental and never used for selection.

See [methods](../../docs/kang_ucl_endpoint_methods.md), [configuration](../../config/kang_ucl_endpoint.yaml), [window figure](figures/01_endpoint_window_comparison.png), [importance figure](figures/02_endpoint_feature_importance.png), [primary table](tables/primary_comparison.csv), and [importance comparison](tables/feature_importance_comparison.csv).
