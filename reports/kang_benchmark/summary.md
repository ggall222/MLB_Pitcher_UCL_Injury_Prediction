# Kang benchmark foundation

Approximate reference implementation, 2017–2025 regular seasons. No 2026 modeling rows or outcomes were loaded. This random-game benchmark is separate from the final chronological UCL study.

See [method specification](../../docs/kang_benchmark_spec.md), [feature crosswalk](tables/feature_crosswalk.csv), and [comparison table](../../docs/kang_benchmark_comparison.csv).

## Primary cohort support at the fixed 28-day audit window

| endpoint   | role     |   rows |   pitcher_seasons |   unique_pitchers |   case_seasons |   control_seasons |   represented_events |   positive_rows |
|:-----------|:---------|-------:|------------------:|------------------:|---------------:|------------------:|---------------------:|----------------:|
| elbow      | starter  |  21114 |              1021 |               528 |            117 |               904 |                  115 |             496 |
| elbow      | reliever |  86909 |              2674 |              1281 |            217 |              2457 |                  214 |            1705 |
| shoulder   | starter  |  22009 |              1090 |               542 |            186 |               904 |                  184 |             804 |
| shoulder   | reliever |  87989 |              2744 |              1285 |            287 |              2457 |                  286 |            2223 |

Counts use no-IL control seasons. Alternative target-specific controls, each window, season exclusions, ambiguous first events and residual missingness are tabulated separately. The 28-day audit does not select the fitted window.

## Bounded reference fits

| endpoint   | role     |   selected_window_days |   development_cv_auc |   roc_auc |   roc_auc_ci_low |   roc_auc_ci_high |   average_precision |   prevalence |   represented_events |   selected_features |
|:-----------|:---------|-----------------------:|---------------------:|----------:|-----------------:|------------------:|--------------------:|-------------:|---------------------:|--------------------:|
| elbow      | starter  |                     49 |               0.7962 |    0.8465 |           0.816  |            0.8744 |              0.1898 |       0.0357 |                  104 |                  18 |
| elbow      | reliever |                     49 |               0.7406 |    0.7549 |           0.7269 |            0.7819 |              0.0998 |       0.0296 |                  204 |                  18 |
| shoulder   | starter  |                     49 |               0.7494 |    0.7702 |           0.7377 |            0.8046 |              0.1971 |       0.0569 |                  168 |                  18 |
| shoulder   | reliever |                     49 |               0.7215 |    0.7077 |           0.6747 |            0.7358 |              0.1157 |       0.0387 |                  266 |                  18 |

Each window and feature subset was chosen inside the random development partition; only the frozen model was scored on its reserved random-game holdout. Windows maximize mean development CV AUC on the declared weekly grid. They are **not** replications of the published GAM plateau estimates. Bootstrap intervals resample pitcher-seasons, but do not remove overlap of pitchers/events across training and holdout.

These discrimination results do not establish prospective UCL warning, biological onset, calibration, or an actionable alert policy. Class weighting changes the probability scale; Brier/mean-risk are descriptive diagnostics, not claims of calibrated population risk.

4 of 4 tasks selected the 49-day grid boundary. This is a descriptive development result, not evidence for an optimal biological window or replication of the published plateau.

## Next implementation step

Resolve control/label and release-Y/extension ambiguities against author code if obtainable, then freeze the reference specification. Next add an adjudicated UCL endpoint comparison while holding features/model/window policy fixed; subsequently introduce chronological evaluation and a limited family-by-lead-time analysis. Preserve 2026 until development is frozen.

Existing PCA/PLS, consolidation, nonlinear and temporal results remain exploratory context; their selected features and scores are not inputs to this benchmark.
