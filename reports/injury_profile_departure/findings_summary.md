# Profile information versus short-term departure: findings

The fitted models use both types of information, but their incremental value differs by injury group and metric. Persistent profiles account for much of the Average Precision in several categories. Departures change which outings are alerted and offer the clearest additional evidence for shoulder injuries. This supports a mixture of profile association and short-term signal, not a demonstrated general injury-imminence detector.

## Direct comparison

All models use the original category-specific 30-day risk sets: train through 2021, validation 2022–2023, test 2024–2025, with the existing 60-day embargo. Both starters and relievers are included. The source cohort has 233,433 appearances; the supervised risk set is smaller and differs by injury category. No test outcomes choose feature blocks or model hyperparameters. These periods have already been examined in earlier analyses.

| Injury group | Profile AP | Departure AP | Combined AP | Original 62 AP | Profile → combined event recall |
|---|---:|---:|---:|---:|---:|
| Elbow / UCL | 0.0262 | 0.0306 | 0.0291 | 0.0426 | 2.8% → 12.4% |
| Shoulder | 0.0501 | 0.0431 | 0.0518 | 0.0559 | 9.3% → 15.5% |
| Forearm | 0.0174 | 0.0117 | 0.0175 | 0.0228 | 10.9% → 13.0% |
| Other arm | 0.0112 | 0.0099 | 0.0107 | 0.0101 | 8.9% → 11.1% |
| Trunk / neck | 0.0281 | 0.0182 | 0.0281 | 0.0273 | 15.3% → 17.6% |
| Lower body | 0.0461 | 0.0246 | 0.0467 | 0.0482 | 16.8% → 17.6% |
| Hand / wrist | 0.0083 | 0.0058 | 0.0078 | 0.0092 | 3.0% → 6.1% |

Event recall above requires an alert **7–30 days** before an event, at a budget of **5% of eligible outings**. Its denominator contains all qualifying events, including events with no usable warning opportunity.

Combined AP exceeds profile-only AP by point estimate for 5/7 test categories, but **0/7 paired 95% intervals exclude zero on the positive side**. Shoulder is the only category with a positive primary AP increment in both validation and test. Forearm and non-arm categories do not show a consistent, well-supported AP gain from departures.

## Event warnings and the workload adjustment

- **Elbow / UCL:** primary event-recall change is +6.7 percentage points in validation and +9.7 in test (test interval +4.8 to +15.0). With workload, role and calendar held common, it becomes +0.0 and +1.4 points; adjusted AP changes are +0.0011 and +0.0010.
- **Shoulder:** primary event-recall change is +8.1 percentage points in validation and +6.2 in test (test interval +2.8 to +10.1). With workload, role and calendar held common, it becomes +5.0 and +3.7 points; adjusted AP changes are +0.0045 and +0.0062.

The adjusted shoulder test AP difference is +0.0062 (95% interval +0.0010 to +0.0119); the validation interval includes zero. Elbow/UCL incremental gains are largely accounted for by shared workload/context. Adding eight strictly prior slopes to profile-plus-departure provides no clear additional AP benefit.

Profile scores repeatedly alert the same pitchers. For shoulder in test, the profile model alerts 67 distinct pitchers, versus 234 for combined, with the same 1,390 alerted outings. This helps explain why event recall can improve without a clear AP gain. It does not establish injury-specific timing: the unadjusted combined elbow/UCL and shoulder event-recall point estimates remain below the uniform-random alert expectation in both periods. The full budget curves retain that benchmark.

## Persistent scores versus approaching-event changes

An exact additive decomposition reconstructs the combined linear predictor as intercept + profile term + departure term, including source-specific missingness indicators. This is a decomposition of log-odds, not a fraction of causal risk or independent predictive importance. In test, 86–98% of profile-term variance lies between pitchers, compared with 4–9% for departure terms. Predictor blocks are correlated and are not biologically orthogonal.

Frozen combined-model scores are applied to the original **443 out-of-training matched pairs**, without rematching. Common complete measurement support leaves **68 shoulder pairs** with all five outings. The shoulder departure case-minus-control score increases 0.120 log-odds from the fifth-last to last outing (pitcher-bootstrap interval 0.007 to 0.260). Velocity departure has the strongest negative non-missingness standardized coefficient in the shoulder combined model: a lower-than-baseline velocity increases its score, conditional on the other predictors. Correlated coefficients are not causal effects.

The smaller balanced calendar cohort contains 38 shoulder pairs. Only the final **1–7-day** bin has a simultaneous difference band above zero; 8–14 days remains uncertain. The curves suggest late divergence, but do not establish a reliable earlier warning threshold. Appearance positions cannot be translated into fixed calendar lead time. Elbow/UCL, forearm and lower-body departure trajectories do not establish a comparable changing score pattern.

Lower-body profile scores show a persistent matched offset averaging 0.214 log-odds (0.061 to 0.371; 69 pairs). This is compatible with stable profile/history associations; it does not prove underlying susceptibility. The means used for “profile” are rolling summaries of at most 20 prior outings and can adapt to emerging injury.

## Robustness, probability quality and stability

- The primary arm event-recall improvements survive complete measurement rows, equal availability indicators, removal of final-six-day positive rows, and removal of the five history/context fields. However, **after workload adjustment and excluding final-six-day positives**, shoulder AP gains shrink to about +0.0006 in validation and +0.0005 in test, with intervals spanning zero. Its adjusted test event-recall increment is about +1.9 points, also uncertain. Thus the strongest adjusted discrimination result depends materially on the near-event window.
- Probability error changes are small. Shoulder test Brier loss improves by about 0.000091 in the primary comparison, but the corresponding validation interval includes zero. Reliability plots and mean bias do not support broad calibration improvement. For example, the combined lower-body test model predicts 4.32% mean risk against 2.30% observed. These are conditional risks in the selected analysis cohort, not population injury probabilities.
- Adding departures improves test rank/alert stability in elbow/UCL and modestly in shoulder, but slightly reduces it in forearm and lower body across 12 paired training-pitcher bootstrap refits. All transforms are refitted on each training resample. Twelve perturbations are diagnostics, not 12 independent generalization tests.
- There is no consistent evidence of **earlier** warnings for the same events. Jointly detected test shoulder events have a mean lead change of −1.8 days, with an interval spanning zero. Comparing conditional median lead across different detected events can be misleading; the paired common-event comparison is saved separately.
- Validation-derived thresholds produce different test alert rates; retrospective top-5% performance is not a deployable fixed-threshold promise.

## Interpretation and next experiment

The answer is **some combination of both**, with strong dependence on injury category. Profile/history information explains most of the observed AP for lower-body and several other categories. Shoulder provides the best evidence that departures add information beyond profile and workload, although much of its discrimination gain is very close to the recorded event. Elbow/UCL benefits in event coverage relative to a static-profile warning policy, but has weak evidence of additional information once workload/context is included. These results do not establish that the model detects biological onset or causal susceptibility.

A defensible next step is to freeze a shoulder-focused, explicitly ≥7-day-warning comparison before evaluating a new chronological period. Retain profile/workload and uniform-random alert benchmarks, inspect near-event reporting dates, and calibrate only within training/validation data if calibration is added. The 2024–2025 period here is exploratory, not a fresh confirmatory holdout.

## Scope and limitations

P is 8 prior measurement means + age, handedness, observed starts, days since debut and prior reported IL count. D is the corresponding 8 current pitcher z-scores. C is 15 workload fields + current role + month, held common rather than forced into either conceptual block. Missing flags are extra learned design columns; all raw NULLs remain untouched. Arm angle is omitted because long-history coverage is unavailable. Current-outing departures are available only after the outing. Pitcher z-scores include the pitcher-specific denominator and do not mathematically remove every stable characteristic.

All models retain the existing median/scaling/L2-logistic C=1 framework, with train-only preprocessing and tight numerical convergence. This tests these representations in the existing linear model; it is not a universal test that nonlinear or interaction-based departure signals cannot help. AP/Brier intervals condition on frozen fitted models; event intervals also condition on fixed budget allocations. Matched time bands adjust within each trajectory, not across every injury and outcome. Small groups, repeated use of controls, multiple comparisons and retrospectively refined injury records limit inference. IL/event dates are observable proxies, not verified biological onset.

Brier loss measures both discrimination and probability reliability; it is not a calibration-only score. See the [scikit-learn calibration guide](https://scikit-learn.org/1.8/modules/calibration.html).

## Figures and reproducibility

Figures are exported as PNG and PDF in [figures](figures/). Recommended for presentation: [incremental AP/event recall](figures/02_incremental_ap_and_event_recall.pdf), [alert-budget curves](figures/03_event_recall_alert_budgets.pdf), and [calendar score trajectories](figures/05_score_trajectories_calendar.pdf). The tables retain every model, category, partition, event denominator and uncertainty interval.

Run the two notebooks with `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-profiles`. Cached stages verify source/config/code/output fingerprints before reuse; set FORCE_RECOMPUTE=True in the notebooks to reproduce fitting. The DuckDB database, canonical injuries, original cohorts, matching and preceding results remain unchanged.
