# Feature consolidation: completed development analysis

## What was resumed

The interruption left 19 readable CSV tables, seven fully readable Parquets, five JSON artifacts and 94 loadable fitted models. Inventory, correlations, 27 family comparisons, 56 nested representation models, four supported-cohort models, and all six Elastic Net tuning fits were complete. The last completed numerical step was the Elastic Net grid. Its selection metadata was missing; this was reconstructed from the saved fits. Sixteen whole-pitcher sparse refits, reduced candidates, conditional checks, coverage analyses, freezing, transfer evaluation, figures and notebooks were then completed.

All original numerical artifacts were reused. Thirty source-variable descriptions and eight timing descriptions were finalized in a **separate reconciled inventory**. A final provenance addendum explicitly identifies pitch-type filters and distinguishes Statcast pitch-type counts from official appearance pitch counts; `feature_inventory_review.csv` is the complete annotated inventory. The frozen statistical inventory remains unchanged. An unordered family-pair summary was also corrected in a separate file; the underlying correlations were unchanged. The original database, previous PCA/PLS results and 125 audited checkpoint artifacts remain protected by fingerprints. See `resume/` and `validation_checks.csv` for verification.

## Cohort and endpoint

| partition | rows | positive_rows | represented_events | initial_explicit_ucl_events | later_supplemental_ucl_confirmation | first_appearance | last_appearance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 27665 | 708 | 110 | 5 | 24 | 2024-03-20 | 2025-09-24 |
| train | 81145 | 1441 | 223 | 15 | 46 | 2015-04-14 | 2021-09-24 |
| validation | 26796 | 600 | 89 | 3 | 17 | 2022-04-07 | 2023-09-29 |

The registered **Elbow / UCL** target includes non-UCL elbow conditions. Only 15 training, three validation and five test represented events explicitly mention UCL in their initial text. Later surgical confirmation is outcome information, not a predictor and not proof that initial diagnoses were complete. This stage therefore selects features for a **broad elbow proxy**, pending UCL-specific adjudication. The primary prediction is after a completed appearance, for the next 30 days.

The original chronological boundaries and 60-day embargo remain intact: train through 2021; select on 2022–2023; inspect 2024–2025 only after freezing. Pitchers may recur chronologically, as intended for predicting future outings of known pitchers; there are no pitcher ID predictors. Whole-pitcher uncertainty and refits preserve within-pitcher dependence. This is not an unseen-pitcher generalization experiment. 2024–2025 has already been examined extensively and is **not independent confirmation**.

## 1. How much redundancy exists?

The inventory contains 538 fields: 536 registered predictors and two existing context fields. Training-only screening leaves 353 measured candidates, of which 333 are substantive and 20 describe support. 63/63 tested algebraic relationships hold on their co-observed training rows. Examples include outs/innings, usage/count identities, current-minus-prior changes, z-score definitions and current-inclusive versus prior-only workload totals. Missingness can still differ between algebraically related fields.

Among the 353 feasible fields, 981 pairs exceed |Pearson| or |Spearman| ≥0.90: 888 within families and 93 across families. Complete linkage on training Spearman correlations produces 189 within-family clusters; protecting distinct temporal roles gives 198 representatives, 189 substantive. This is substantial duplication, not evidence that all members are interchangeable.

## 2. Which families provide unique information?

Extension is the only family whose removal has an unadjusted paired AP-loss interval narrowly above zero. Its point loss is 0.00172 (95% interval 0.00004 to 0.00544). This exploratory result does not survive an assumed guarantee about multiple comparisons; no multiplicity-adjusted claim is made. Velocity, background and movement show positive point losses with wider uncertainty. The full model relies on several families under marginal permutation, including extension, spin, movement, background and release, but reliance is not the same as unique contribution.

| family | ap_loss | low | high |
| --- | --- | --- | --- |
| Velocity | 0.0025826 | -0.0038865 | 0.010521 |
| Release position and variability | -0.00096794 | -0.0070756 | 0.0070311 |
| Extension | 0.0017155 | 4.2582e-05 | 0.0054354 |
| Spin | -0.0009801 | -0.0040818 | 0.0010769 |
| Pitch movement | 0.0011822 | -0.002093 | 0.0054002 |
| Location and strike fractions | -0.0007022 | -0.0026052 | 0.00064358 |
| Pitch mix and pitch-type counts | -0.0030915 | -0.0092809 | 0.0007657 |
| Workload and rest | -0.0013278 | -0.0097194 | 0.0015335 |
| Role and recent starts | -0.00010694 | -0.0032181 | 0.0023527 |
| Appearance performance | 0.00028045 | -0.00012205 | 0.0011728 |
| Pitcher background and experience | 0.0013024 | -0.0017424 | 0.0060288 |

The background-only model has high validation AP relative to several larger representations. This supports examining persistent/contextual risk, not interpreting every Statcast coefficient as imminent injury. AP and advance-warning counts sometimes move in opposite directions.

## 3. Which representations duplicate each other, and which differ temporally?

Current levels, adjacent rolling windows and baseline means often cluster; pitch-type counts duplicate usage plus total volume, and innings/outs and overlapping workload windows add exact or near-exact dependencies. Departures and slopes explicitly encode a different time relationship, so they were protected during correlation-only reduction. Nevertheless, none of the positive nested representation increments has a 95% paired interval wholly above zero in this validation cohort. The one clearly negative AP increment is additional seven-day workload after longer workload context; its warning count moves differently. These are order-dependent comparisons, not a declaration that departures, slopes or workload have no value.

Full contrasts and alternative choices are in `representation_comparisons.csv`, `concept_representation_decisions.csv` and `conditional_feature_checks.csv`. The latter drops each retained feature and, where available, replaces/adds a correlated alternative; these are diagnostic checks and do not initiate further test-driven pruning.

## 4. Stability and sparse selection

Validation chose Elastic Net C=0.03, l1_ratio=0.25, with 213 nonzero measured fields (222 when selected missing flags count). Sixteen whole-pitcher training refits hold those settings fixed. 47 fields have both selection frequency ≥90% and conditional sign agreement ≥90%; 113 selected-at-least-once fields have sign agreement below 80%. Even stable selection is not proof of incremental value. Rank SD, measured-versus-flag selection, marginal permutation mean/SD and cluster selection frequency are saved for all 333 fields.

Top reproducibly selected fields:

| feature | selection_frequency | conditional_sign_stability | mean_rank | rank_sd | permutation_mean |
| --- | --- | --- | --- | --- | --- |
| ivb_inches | 1 | 1 | 10.938 | 10.188 | 0.0029715 |
| ff_avg_velocity_prev_5 | 1 | 1 | 19.25 | 27.212 | 0.0035855 |
| season_starts | 1 | 1 | 20.938 | 27.287 | 8.2742e-05 |
| ch_vertical_movement_avg | 1 | 1 | 17.625 | 18.323 | 0.0011082 |
| calendar_month | 1 | 1 | 24 | 24.86 | 0.0032987 |
| velo_z | 1 | 1 | 23.062 | 27.953 | 0.00073642 |
| ch_release_x_avg_prev_10 | 1 | 1 | 47.812 | 57.384 | 0.00082945 |
| is_starter | 1 | 1 | 35.375 | 39.164 | 7.3289e-05 |
| starter_share_prev_5 | 1 | 1 | 63.625 | 70.389 | 0.00017921 |
| release_x_avg_baseline_sd | 1 | 1 | 44.875 | 43.393 | 0.00090828 |
| high_velocity_pitch_count | 1 | 1 | 46.375 | 38.927 | 0.0012273 |
| si_avg_velocity_prev_5 | 1 | 1 | 54.875 | 37.829 | -0.00093684 |

These are 16 conditional refits, not a new hyperparameter search in every bootstrap or formal error-controlled stability selection. Sparse coefficients for binary flags use raw-unit scaling, while continuous fields are standardized. The final candidate comparisons all use the same L2 preprocessing and fitting rule.

## 5. Reduction and the recommended candidate

The declared validation rule selected **Departure only**, containing **8 input fields and 16 fitted design columns** including missingness indicators. It first permits an AP loss of at most one paired-bootstrap SE relative to the highest-AP candidate, then requires warning count within two events of the best AP-competitive model, and chooses the smallest set. This is a pragmatic development tolerance, not a noninferiority test.

Validation AP: full 333 fields **0.03061**; correlation-only 189 fields **0.02491**; selected **0.02673**. The selected-minus-full point difference is **-0.00388**. At a retrospective 5% alert budget, the candidate warns **20/120** events 7–30 days ahead, versus **11/120** for full features. The original denominator includes 120 qualifying events, of which 80 have an eligible opportunity in that warning window. Lead time is conditional on detection, not achievable for every event.

| variant | features | average_precision | ap_low | ap_high | alerted_events | injury_event_recall | selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full feasible | 333 | 0.030615 | 0.020762 | 0.048003 | 11 | 0.091667 | False |
| Previous 62 | 62 | 0.028366 | 0.021043 | 0.040727 | 13 | 0.10833 | False |
| Correlation representatives | 189 | 0.024911 | 0.018416 | 0.036432 | 10 | 0.083333 | False |
| Profile only | 13 | 0.029676 | 0.018459 | 0.050392 | 10 | 0.083333 | False |
| Departure only | 8 | 0.026734 | 0.020339 | 0.034641 | 20 | 0.16667 | True |
| Profile plus departure | 21 | 0.028448 | 0.0195 | 0.041081 | 17 | 0.14167 | False |
| Background context | 7 | 0.040475 | 0.023794 | 0.080392 | 9 | 0.075 | False |
| Elastic Net selected values | 213 | 0.032514 | 0.020694 | 0.055648 | 11 | 0.091667 | False |
| Stable representatives 90% | 62 | 0.034103 | 0.023889 | 0.048091 | 18 | 0.15 | False |
| Stable representatives 75% | 128 | 0.029966 | 0.021489 | 0.041583 | 16 | 0.13333 | False |
| Stable representatives 50% | 173 | 0.030571 | 0.020219 | 0.043796 | 12 | 0.1 | False |
| Stable representatives 25% | 183 | 0.030405 | 0.020158 | 0.043363 | 12 | 0.1 | False |

The retained set contains: 8 departure. These labels describe construction, not physiology. Background/history can include contextual seasonality, and a rolling mean is slowly changing, not an immutable pitcher trait.

**This is a warning-focused tradeoff, not a uniformly superior model.** The AP tolerance is broad because validation uncertainty is large: the selected AP is about one-third below the background-only point estimate. Correlation reduction by itself performs worse. The 62-field 90%-stability representative set offers a different development tradeoff (AP 0.03410, 18/120 events warned); it remains a documented alternative, not another test-selected winner. The eight-field set is retained as the predefined departure block, not because all eight individually show strong conditional evidence. Only `velo_z` has a positive unadjusted drop-one AP interval in that block; `release_x_sd_z` and `spin_z` have unstable signs in the full sparse refits. This prevents interpreting the final map as eight independently established biomarkers.

| feature | feature_group | representation | selection_frequency | conditional_sign_stability | ap_loss |
| --- | --- | --- | --- | --- | --- |
| velo_z | Velocity | pitcher-relative z-score | 1 | 1 | 0.0035095 |
| release_x_z | Release position and variability | pitcher-relative z-score | 1 | 1 | -0.00020836 |
| release_z_z | Release position and variability | pitcher-relative z-score | 1 | 0.875 | -9.7774e-06 |
| release_x_sd_z | Release position and variability | pitcher-relative z-score | 0.9375 | 0.53333 | 0.0001239 |
| release_z_sd_z | Release position and variability | pitcher-relative z-score | 0.875 | 0.92857 | 9.7938e-05 |
| extension_z | Extension | pitcher-relative z-score | 0.9375 | 0.86667 | 0.00082882 |
| spin_z | Spin | pitcher-relative z-score | 0.5625 | 0.66667 | -0.00019368 |
| zone_pct_z | Location and strike fractions | pitcher-relative z-score | 0.4375 | 0.85714 | 5.2346e-05 |

Every retained field has source, concept, representation, timing, correlated alternatives, conditional AP evidence and stability in `retained_features.csv`. All exclusions, including unavailable modern measurements, remain documented in `excluded_features.csv`; `feature_recommendation_details.csv` adds specific correlation, weak-selection and sign-instability reasons for each field. No master feature was deleted. The exact candidate is frozen in `config/ucl_candidate_features_v2.yaml` with fitted model and decision-table hashes in `candidate_freeze.json`.

## 6. Measurement support and low-value fields

| variant | features | average_precision | ap_difference_from_no_flags | ap_difference_low | ap_difference_high | alerted_events | brier_score | calibration_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Core, no flags | 8 | 0.026837 | 0 | 0 | 0 | 20 | 0.021898 | -0.0044426 |
| Core + missing flags | 8 | 0.026734 | -0.00010319 | -0.0004643 | 0.00024752 | 20 | 0.021898 | -0.0044118 |
| Core + support, no flags | 28 | 0.026663 | -0.00017464 | -0.0050211 | 0.0057626 | 16 | 0.021954 | -0.0087258 |
| Core + support + flags | 28 | 0.026583 | -0.00025476 | -0.005125 | 0.00557 | 16 | 0.021955 | -0.0087204 |
| Support only | 20 | 0.02434 | -0.002497 | -0.0089837 | 0.0058428 | 8 | 0.021971 | -0.0090696 |

Coverage and missingness may encode measurement availability, role, experience or historical era. Their performance cannot establish a physiological signal. The primary selection excluded explicit support fields; this sensitivity reports what changes when support and missingness flags are introduced separately. Season/role prediction and missingness summaries are in `coverage_role_season_audit.csv`. Brier score measures probability error, not calibration alone; mean bias and validation-defined calibration bins are also provided.

Arm angle and within-outing fastball decline are missing in about 82% and 80% of training rows respectively. They were not silently filled with zero or declared useless. Common-row modern and starter sensitivities show no clear AP gain, with limited event support. Raw circular spin-axis values require sine/cosine treatment before linear selection; prior UCL surgery fields require verified publication dates. `noise_and_stability_assessment.csv` reports constant/missing/rare-support flags alongside instability and conditional evidence rather than deleting individually weak fields automatically.

## 7. Temporal transfer and what to freeze

`frozen_performance.csv`, `frozen_paired_differences.csv` and `frozen_alert_budgets.csv` report already examined 2024–2025 transfer for fixed models. `validation_fixed_thresholds.csv` applies validation-derived numeric cutoffs to that period: ranking the top 5% retrospectively is different from maintaining 5% alerts prospectively. Calibration bins also use validation-defined edges. Test differences must not be used to revise this selected feature list under an untouched-test claim.

In 2024–2025, the departure candidate has AP **0.03098**, versus **0.02797** for full features and **0.04254** for the previous 62-field benchmark. Its AP difference from that benchmark is −0.01156 (paired interval −0.02687 to −0.00284). It warns **18/145** events, versus **26/145** for the previous benchmark and 15/145 for full features. The previous 62-field model must therefore remain a frozen comparator; the new candidate is not a validated replacement. The departure model's validation-fixed threshold produces **4.57%** test alerts. At the retrospective 5% budget its row precision is only **3.04%**, and median warning lead time of **23.5 days** applies to detected events only. This remains an exploratory warning system with substantial false-alert burden.

| partition | variant | average_precision | pr_lift | brier_score | calibration_bias |
| --- | --- | --- | --- | --- | --- |
| validation | Departure only | 0.026734 | 1.1939 | 0.021898 | -0.0044118 |
| validation | Full feasible | 0.030615 | 1.3672 | 0.022515 | -0.0018377 |
| validation | Previous 62 | 0.028366 | 1.2668 | 0.022004 | -0.00093139 |
| validation | Profile only | 0.029676 | 1.3253 | 0.02189 | -0.0016624 |
| validation | Profile plus departure | 0.028448 | 1.2705 | 0.021886 | -0.0016239 |
| test | Departure only | 0.030977 | 1.2104 | 0.024976 | -0.0077635 |
| test | Full feasible | 0.02797 | 1.0929 | 0.025757 | -0.0064862 |
| test | Previous 62 | 0.04254 | 1.6623 | 0.024857 | -0.0053659 |
| test | Profile only | 0.025964 | 1.0145 | 0.024994 | -0.005318 |
| test | Profile plus departure | 0.029009 | 1.1335 | 0.024977 | -0.0054698 |

Freeze the **feature list, exact derivations, training preprocessing, L2 fit, endpoint definition, 30-day horizon and 7–30-day/5%-budget evaluation** now. Retain profile/departure and the previous 62-field comparators. Before confirmatory UCL work, adjudicate UCL-specific events and knowledge dates, then register an untouched future cohort and a prospective threshold policy. Any endpoint revision requires a new development/freeze cycle; these elbow-proxy results cannot validate confirmed ligament injury warning.

## Deliverables and limitations

Three executed notebooks are in `notebooks/feature_consolidation/`. Six figures, each PNG and PDF, cover within-family correlations, cross-family redundancy, ablation/permutation, sparse stability, reduced-candidate performance and support sensitivity. Tables preserve all intermediate evidence and final decisions. `methods.md` documents the exact analysis; `validation_checks.csv` and `validation_summary.json` record integrity and execution checks.

Feature screening and repeated comparisons consume validation information. Unadjusted intervals are descriptive and conditional on chosen models; they do not remove tuning optimism. Linear models may miss nonlinear interactions. Correlated predictors can exchange weights. Only 16 training refits and limited events support stability conclusions. The selection rule balances AP, warning count and size, not all possible operational costs. No feature importance establishes causality, tissue damage or biological onset. The remaining scientific tasks are endpoint adjudication and genuinely future chronological validation, not unfinished numerical stages of this consolidation.
