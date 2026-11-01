# Kang parent-paper replication

This stage reproduces the published experimental structure using existing data. It adds no UCL-specific endpoint, chronological split, new warning metric, learner comparison or project-specific predictor.

## Cohort

| endpoint   | role     | control_definition   |   games |   unique_pitchers |   pitcher_seasons |   events |   control_seasons |   unknown_diagnosis_control_seasons |
|:-----------|:---------|:---------------------|--------:|------------------:|------------------:|---------:|------------------:|------------------------------------:|
| elbow      | starter  | no_shoulder_elbow_il |   34477 |               620 |              1665 |      123 |              1542 |                                 127 |
| elbow      | reliever | no_shoulder_elbow_il |  119590 |              1418 |              3669 |      233 |              3436 |                                 194 |
| shoulder   | starter  | no_shoulder_elbow_il |   35384 |               627 |              1735 |      193 |              1542 |                                 127 |
| shoulder   | reliever | no_shoulder_elbow_il |  120704 |              1421 |              3737 |      301 |              3436 |                                 194 |

The Methods control definition is primary; counts using the conflicting Results no-IL definition are retained in `tables/cohort_counts.csv`. Events and controls are shared across repeated games; endpoint totals must not be summed as unique people.

## Window and discrimination comparison

| task              |   lower |   upper |   band_segments |   window | selected_day_in_band   |   roc_auc |   ci_low |   ci_high |   published_auc |   features |
|:------------------|--------:|--------:|----------------:|---------:|:-----------------------|----------:|---------:|----------:|----------------:|-----------:|
| elbow_starter     |   26.88 |   29.39 |               1 |       28 | True                   |     0.925 |    0.905 |     0.944 |           0.868 |         18 |
| elbow_reliever    |   23.69 |   26.33 |               1 |       25 | True                   |     0.91  |    0.891 |     0.926 |           0.815 |         18 |
| shoulder_starter  |   12.69 |   32.93 |               4 |       23 | False                  |     0.855 |    0.824 |     0.883 |           0.815 |         18 |
| shoulder_reliever |   27.26 |   34.46 |               1 |       31 | True                   |     0.896 |    0.883 |     0.909 |           0.797 |         18 |

Bands come from the published GAM derivative structure, not maximum AUC. They are not confidence intervals. The candidate increment, smoother settings and band-to-integer conversion are undisclosed in the article and explicitly approximated here.

Multiple qualifying segments mean the earliest/latest interval is an envelope containing gaps; its rounded midpoint can fall outside every qualifying segment. Such a result is not a well-identified unique optimal window, even when its midpoint resembles the published window.

- **elbow / starter:** The band overlaps the published interval; this is descriptive agreement, not an equivalence test.
- **elbow / reliever:** The qualifying band is later/longer than the published band.
- **shoulder / starter:** 4 disconnected segments; the envelope does not identify a unique window.
- **shoulder / reliever:** The qualifying band is later/longer than the published band.

## Feature importance

| task              | feature                | domain   |   mean_gain |
|:------------------|:-----------------------|:---------|------------:|
| elbow_starter     | age_at_appearance      | context  |      0.1251 |
| elbow_starter     | release_z_avg          | release  |      0.0999 |
| elbow_starter     | kang_release_abs_x_avg | release  |      0.0909 |
| elbow_starter     | extension_avg          | release  |      0.0778 |
| elbow_starter     | kang_velocity_sd       | velocity |      0.0672 |
| elbow_reliever    | age_at_appearance      | context  |      0.2289 |
| elbow_reliever    | release_z_avg          | release  |      0.1039 |
| elbow_reliever    | kang_release_abs_x_avg | release  |      0.0986 |
| elbow_reliever    | extension_avg          | release  |      0.0871 |
| elbow_reliever    | kang_spin_max          | spin     |      0.0635 |
| shoulder_starter  | age_at_appearance      | context  |      0.1406 |
| shoulder_starter  | kang_release_abs_x_avg | release  |      0.0864 |
| shoulder_starter  | extension_avg          | release  |      0.0844 |
| shoulder_starter  | release_z_avg          | release  |      0.0756 |
| shoulder_starter  | spin_avg               | spin     |      0.0703 |
| shoulder_reliever | age_at_appearance      | context  |      0.2221 |
| shoulder_reliever | release_z_avg          | release  |      0.1051 |
| shoulder_reliever | kang_release_abs_x_avg | release  |      0.1037 |
| shoulder_reliever | extension_avg          | release  |      0.0949 |
| shoulder_reliever | max_velocity           | velocity |      0.0702 |

Gain is normalized total split gain averaged over training bootstrap refits; it does not identify physiological mechanisms.

## Fidelity and unresolved details

Matched: years, game grain, broad injury framing, role strata, random 67/33 games with pitcher overlap, XGBoost, inverse-frequency weighting, 50 repeated five-fold selection fits, domain Pearson threshold 0.7, ROC-AUC and 500 pitcher-season confidence resamples. The same GAM slope-band structure and prior-region injury sensitivity were implemented.

Approximate/unavailable: author cohort records, conflicting control wording, exact feature list (Y versus extension), outside-window case handling, exact random seed/grid, selection cutoffs/subset size, tuning procedure, hyperparameters, software versions, smoother settings and final window conversion. The published feature-specific rolling-risk attribution lacks enough detail for numerical replication and was not replaced with a new method.

This is a reproducible **qualified structural benchmark**, not an exact numerical replication or evidence of deployment performance. Do not change settings to force agreement. Author code/data would be needed to remove the documented ambiguities.

See [protocol](../../docs/kang_replication_methods.md), [config](../../config/kang_replication.yaml), and [direct comparison](tables/published_comparison.csv). Existing analyses and the central database remain unchanged.
