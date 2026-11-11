# Extension 2 — chronological UCL evaluation

**Conclusion: Temporal discrimination does not clearly persist.**

The only major design change is calendar separation. All fitted choices use 2017–2023. The same frozen models score 2024–2025 and 2026; neither later period chooses a model, window or feature. The random-game comparator is archived Extension 1.

## Cohort and positive-event support

| role     | period     |   pitcher_games |   eligible_games |   censored_games |   pitcher_seasons |   unique_pitchers |   ucl_events |   represented_events |   positive_games |   negative_games |   prevalence |
|:---------|:-----------|----------------:|-----------------:|-----------------:|------------------:|------------------:|-------------:|---------------------:|-----------------:|-----------------:|-------------:|
| starter  | training   |           25675 |            25675 |                0 |              1227 |               502 |           23 |                   23 |               66 |            25609 |     0.002571 |
| starter  | validation |            7543 |             7543 |                0 |               340 |               245 |           13 |                   13 |               36 |             7507 |     0.004773 |
| starter  | test       |            3830 |             3830 |                0 |               166 |               166 |            6 |                    6 |               18 |             3812 |     0.0047   |
| reliever | training   |           88220 |            88220 |                0 |              2698 |              1162 |           28 |                   28 |              139 |            88081 |     0.001576 |
| reliever | validation |           26695 |            26695 |                0 |               768 |               558 |            6 |                    6 |               26 |            26669 |     0.000974 |
| reliever | test       |           13421 |            13421 |                0 |               403 |               403 |            9 |                    9 |               32 |            13389 |     0.002384 |

Pitcher-games are the screened cohort; eligible games have observed labels for the selected window. Cohort events and events represented by positive games are distinct. Source observation ends **2026-10-20**. Late negative horizons beyond that date remain censored. First-event and regular-season injury framing remain inherited.

At the selected 16/15-day horizons, 0 test games require administrative censoring. The last regular-season game is 2026-09-27; the source cutoff is 2026-10-20. Administrative maturity does not guarantee complete diagnosis/procedure confirmation. The source has 21 accepted 2026 UCL episodes, of which 15 enter the benchmark-compatible cohort. The audit also retains 33 probable-supplemental episodes and 6 episodes with timing/date exclusions; these are not promoted to confirmed outcomes. Unresolved IL absence is distinct from uncertain UCL identity. Later source confirmation could change retrospective ascertainment; no such confirmation is invented.

## Windows and discrimination

| role     | period            |   roc_auc |   ci_low |   ci_high |   represented_events |   valid_bootstraps |
|:---------|:------------------|----------:|---------:|----------:|---------------------:|-------------------:|
| starter  | random_comparator |    0.9559 |   0.9269 |    0.9877 |                   30 |                500 |
| starter  | validation        |    0.532  |   0.4271 |    0.6405 |                   13 |                500 |
| starter  | test              |    0.6421 |   0.5051 |    0.7885 |                    6 |                499 |
| reliever | random_comparator |    0.9868 |   0.9797 |    0.9929 |                   29 |                500 |
| reliever | validation        |    0.4753 |   0.1965 |    0.7585 |                    6 |                497 |
| reliever | test              |    0.4016 |   0.3008 |    0.5144 |                    9 |                500 |

**Starter window:** 16 days; slope envelope 11.50–21.36, 3 segment(s), midpoint inside a segment: False. This checks band shape, not window-selection stability across independent samples.

Starter validation: AUC 0.532 (0.427–0.640), -0.424 versus random 0.956; 13 represented events, 36 positive games; 0 administratively censored games.

Starter test: AUC 0.642 (0.505–0.789), -0.314 versus random 0.956; 6 represented events, 18 positive games; 0 administratively censored games.

Starter leading chronological predictors: age_at_appearance, kang_velocity_sd, kang_release_abs_x_avg, extension_avg, release_z_avg. Shared top-five with random: age_at_appearance, kang_release_abs_x_avg, release_z_avg, extension_avg. Gain is model reliance, not causation.

**Reliever window:** 15 days; slope envelope 14.26–15.17, 1 segment(s), midpoint inside a segment: True. This checks band shape, not window-selection stability across independent samples.

Reliever validation: AUC 0.475 (0.196–0.759), -0.512 versus random 0.987; 6 represented events, 26 positive games; 0 administratively censored games.

Reliever test: AUC 0.402 (0.301–0.514), -0.585 versus random 0.987; 9 represented events, 32 positive games; 0 administratively censored games.

Reliever leading chronological predictors: age_at_appearance, kang_release_abs_x_avg, release_z_avg, extension_avg, avg_velocity. Shared top-five with random: age_at_appearance, kang_release_abs_x_avg, release_z_avg, extension_avg. Gain is model reliance, not causation.

The same 500 pitcher-season bootstrap attempts yield conditional intervals; degenerate draws are omitted and their valid count is reported. A small number of events limits inference even when intervals appear narrow. Training CV remains random within training, as required; its window curve is not a later-time performance estimate.

In-sample training AUCs: starter 1.000, reliever 1.000. These are fit diagnostics, not independent evaluation. The comparison with later-period estimates does not isolate a specific cause of any generalization gap.

## Seen and unseen pitchers

| role     | period     | training_pitcher_status   |   scored_games |   eligible_games |   pitchers |   pitcher_seasons |   positive_games |   represented_events |   prevalence |   roc_auc | small_event_support   |
|:---------|:-----------|:--------------------------|---------------:|-----------------:|-----------:|------------------:|-----------------:|---------------------:|-------------:|----------:|:----------------------|
| starter  | validation | seen                      |           5440 |             5440 |        148 |               219 |               21 |                    8 |      0.00386 |   0.4836  | False                 |
| starter  | validation | unseen                    |           2103 |             2103 |         97 |               121 |               15 |                    5 |      0.00713 |   0.57481 | False                 |
| starter  | test       | seen                      |           1887 |             1887 |         72 |                72 |                3 |                    1 |      0.00159 |   0.58953 | True                  |
| starter  | test       | unseen                    |           1943 |             1943 |         94 |                94 |               15 |                    5 |      0.00772 |   0.61909 | False                 |
| reliever | validation | seen                      |          19312 |            19312 |        331 |               484 |               18 |                    4 |      0.00093 |   0.34585 | True                  |
| reliever | validation | unseen                    |           7383 |             7383 |        227 |               284 |                8 |                    2 |      0.00108 |   0.78146 | True                  |
| reliever | test       | seen                      |           7126 |             7126 |        171 |               171 |               12 |                    3 |      0.00168 |   0.39181 | True                  |
| reliever | test       | unseen                    |           6295 |             6295 |        232 |               232 |               20 |                    6 |      0.00318 |   0.39731 | False                 |

Seen status means the pitcher appeared in 2017–2023 training. Pitcher identities can recur across periods, but games, pitcher-seasons and positive events cannot. Subgroup differences are descriptive and may reflect very few events; there is no subgroup model selection.

Unseen-pitcher point estimates are higher in the displayed comparisons, but these do not establish a reliable subgroup advantage. The high validation AUC for unseen relievers is based on only two events; in 2026 both reliever subgroups are near 0.39. The seen-starter test subgroup has only one event. These sparse contrasts should remain descriptive.

## Score, predictor and calibration drift

| role     | period     |   scored_games |     mean |       sd |   q1 |   q10 |   q25 |     q50 |      q75 |      q90 |      q99 |
|:---------|:-----------|---------------:|---------:|---------:|-----:|------:|------:|--------:|---------:|---------:|---------:|
| starter  | training   |          25675 | 0.002961 | 0.050705 |    0 | 1e-06 | 4e-06 | 1.4e-05 | 7.1e-05  | 0.000352 | 0.009339 |
| starter  | validation |           7543 | 0.001345 | 0.014753 |    0 | 2e-06 | 5e-06 | 2.5e-05 | 0.000147 | 0.000944 | 0.024363 |
| starter  | test       |           3830 | 0.00324  | 0.030947 |    0 | 2e-06 | 6e-06 | 3.4e-05 | 0.000224 | 0.001419 | 0.058001 |
| reliever | training   |          88220 | 0.002054 | 0.039972 |    0 | 1e-06 | 3e-06 | 1.6e-05 | 8.9e-05  | 0.000437 | 0.008767 |
| reliever | validation |          26695 | 0.000538 | 0.009075 |    0 | 1e-06 | 3e-06 | 1.4e-05 | 7.2e-05  | 0.000328 | 0.007372 |
| reliever | test       |          13421 | 0.000794 | 0.009323 |    0 | 1e-06 | 3e-06 | 1.4e-05 | 8.2e-05  | 0.000504 | 0.012137 |

Largest descriptive standardized mean shifts (all original fields remain in the full drift table):

| role     | period     | feature       |   standardized_mean_difference |   missing_fraction |
|:---------|:-----------|:--------------|-------------------------------:|-------------------:|
| starter  | test       | st_usage_pct  |                         0.6906 |             0      |
| reliever | test       | st_usage_pct  |                         0.6678 |             0      |
| starter  | test       | extension_avg |                         0.6587 |             0      |
| starter  | validation | extension_avg |                         0.6557 |             0.0005 |
| starter  | validation | st_usage_pct  |                         0.6205 |             0      |
| starter  | test       | max_velocity  |                         0.5787 |             0      |
| reliever | validation | st_usage_pct  |                         0.5193 |             0      |
| reliever | validation | extension_avg |                         0.4332 |             0.0007 |
| reliever | test       | extension_avg |                         0.3857 |             0.0003 |
| starter  | validation | max_velocity  |                         0.3406 |             0      |
| reliever | test       | kang_spin_sd  |                         0.2927 |             0.0042 |
| reliever | validation | max_velocity  |                         0.222  |             0      |

Score distributions include all scored cohort games, including censored test games; training scores are in-sample. Predictor drift is descriptive and does not alter feature selection. Class-weighted XGBoost scores are not assumed to be absolute risks. Reliability uses training-derived bins and observed labels only; no recalibration or updating was applied.

The largest predictor shifts involve sweeper usage and extension, with additional velocity/spin shifts. Such distribution changes may reflect repertoire, population or measurement changes; they are not isolated causes of the AUC decline. Starter median/upper-tail scores rise in later periods, while average scores remain below observed prevalence in every later role/period. This distinguishes score drift and underprediction from discrimination.

| role     | period     |    brier |   mean_risk_bias |   prevalence |
|:---------|:-----------|---------:|-----------------:|-------------:|
| starter  | validation | 0.004985 |        -0.003428 |     0.004773 |
| starter  | test       | 0.005664 |        -0.001459 |     0.0047   |
| reliever | validation | 0.001056 |        -0.000436 |     0.000974 |
| reliever | test       | 0.002472 |        -0.00159  |     0.002384 |

## Scope of the conclusion

The frozen qualitative policy requires later intervals above chance and at least five represented events in every role/period before making a strong overall persistence statement. This is a conservative narrative rule, not an equivalence test or model-selection criterion. Read each role/period estimate alongside its support.

A decline is evidence of weaker later-time performance under this design, not proof that the original experiment was invalid. Temporal drift, case composition and the removal of random cross-set event sharing can all contribute. This experiment does not identify their individual causal contributions.

The endpoint remains retrospective operational adjudication. Historical source-publication archives are not independently verified; generic/unknown conditions and excluded possible-UCL episodes remain limitations. Season-role and eligibility use full-season information as in the parent. Previously analyzed 2026 data are locked for this experiment, but are not globally untouched or prospective confirmation. No clinical, deployment or operational-warning claim follows.

## Reproducibility

The protocol freeze records source hashes, settings and interpretation policy before fitting. The development freeze records training models, features and windows before later-period access. AUC is verified independently in Python; tests enforce date, game, pitcher-season and event separation and unchanged prior artifacts.

See [methods](../../docs/kang_ucl_chronological_methods.md), [config](../../config/kang_ucl_chronological.yaml), [primary comparison](tables/primary_comparison.csv), [validation checks](tables/validation_checks.csv), and [AUC figure](figures/03_auc_comparison.png).
