# Locked temporal replication and operational warning

This stage applies the **existing `ucl_temporal` development freeze** unchanged to refreshed 2026 observations. It is a locked retrospective chronological replication sensitivity. The observation dates predate the final October 20, 2026 development freeze and appeared elsewhere in this project. Neither prospective nor fully independent confirmation is claimed.

## Reused design

- Original training: 2015–2021. Original development validation: 2022–2023. Previously examined 2024–2025 remains exploratory transfer.
- Four disjoint outcomes: **1–7, 8–14, 15–30 and 31–60 days** after a completed appearance.
- Primary comparisons share the same observed 60-day first-event risk set. Other-interval UCL events are negatives for the specified interval, not evidence of a healthy pitcher. Known competing injuries end follow-up; uncertain/ambiguous episodes are censored.
- The refined UCL endpoint, earliest defensible event-date proxy, regular-season pitcher eligibility, five prior appearances, chronological embargo and observed-negative follow-up requirements are reused. No source dates become predictors.
- The existing 113-variable union, 11 families, 28 model definitions per interval, training-fitted clipping/median imputation/missing indicators/scaling, and **C=1 L2 logistic models** are loaded directly. There are **zero new predictive model fits**.
- Velocity and Pitcher background and experience remain the only nominated follow-up families. All frozen models are retained for context, preventing selective reporting.

This is a replication of the original temporal freeze, separate from the revised-training study. It does not substitute a newly selected 2015–2023 fit for the original models.

## Sources, preservation and observation cutoff

The latest existing refresh supplies the read-only central DuckDB, appearances and source-derived histories, plus `data/injury/revised_2026/processed/` canonical injuries. Regular-season data extend through September 27, 2026; injury-source observation ends October 20, 2026. The same refined adjudication function runs into a new directory. Future-dated procedure evidence cannot establish a completed procedure before the observation cutoff. Surgery and activation dates remain outcomes, never predictors or onset substitutes.

The replication writes only under `reports/ucl_temporal_replication`, `data/injury/analysis/ucl_temporal_replication`, its new modules/config/docs/notebook. Original `ucl_temporal` report, notebook, model and cache files are hashed before scoring and verified afterward. The database is opened read-only and its source signature remains fixed.

Late-season landmarks require observed future MLB participation through 60 days within the existing 30-day grace to qualify as event-free negatives. Source coverage alone is insufficient. Unknown labels remain missing; alerting them does not resolve their outcomes. All-event warning denominators retain qualifying events without observable opportunities.

## Lock and thresholds

`replication_freeze.json` records original model signatures, byte hashes, candidate nomination, intervals, source/config signatures, numerical thresholds, reliability-bin edges and episode policy. `replication_access.json` records the first scoring time and freeze hash.

Each model/interval receives **1%, 2.5%, 5% and 10% development budgets**. Its fixed numeric cutoff is the corresponding upper score quantile of the original 2022–2023 observed-outcome validation predictions. All ties at the cutoff alert. No cutoff, model, family, lag, interval, regularization or feature is selected using 2026. Realized holdout alert rates may differ from the development budget. Original retrospective top-5% warnings are reported separately as ranking context, never used to reset a holdout cutoff.

## Ranking, temporal contrasts and shared events

Report AP, prevalence, AP/prevalence (PR lift), ROC-AUC, Brier score, mean risk, mean-risk bias and represented injury events for all 112 model-interval combinations. Whole-pitcher bootstrap draws are shared across models and intervals (300 draws, original seed 20261020).

For every family, reproduce each near interval's **normalized ablation loss minus the 31–60-day normalized ablation loss**: `(AP_full − AP_without_family) / interval_prevalence`. Retain the original joint centered max-t bands across all 33 contrasts. Also reproduce the five original reference-model 1–7 minus 31–60 PR-lift contrasts. These are the existing contrasts, not newly chosen post-result comparisons.

Report continuous effects and direction agreement relative to development and exploratory transfer. Same direction, magnitude similarity, statistical support and operational utility are separate judgments. There was no prespecified equivalence margin; do not infer comparable effects just because intervals overlap.

For the original shared-event sensitivity, intersect event IDs represented in all four intervals. Remove pre-UCL rows belonging to nonshared events while retaining other observed comparisons. Repeat metrics and contrasts without refitting. This is a retrospective composition diagnostic, not a new primary estimand.

## Event stability

For every frozen model and interval:

1. Allocate exact tie-aware AP contribution to each positive event, using precision at tied score thresholds. Contributions sum exactly to AP. Record the largest event's contribution and fraction of total AP.
2. Remove all evaluation appearances for each injured pitcher in turn, including that pitcher's negative rows; recompute AP, prevalence and PR lift. No model is refitted and no influential event is removed from the primary estimate.
3. Retain event and pitcher IDs, all leave-out results, and ranges in the headline summary.

An additive AP share is not causal attribution or identical to the leave-out change. Bootstrap intervals are conditional on frozen models, not nested model-selection uncertainty. Sparse positive counts, unequal interval widths and different event composition limit comparisons.

## Alert episodes and operational denominators

Apply frozen cutoffs to **all eligible 2026 landmarks**, including censored rows. Group consecutive alerted outings within pitcher-season. A non-alert outing, an ineligible outing, a season boundary, or more than 30 days between outings starts a new episode. Grouping does not inspect injury outcomes. Same-day ordering uses game IDs as the existing deterministic order.

Report alerted outings, alert episodes, episode persistence, episodes per eligible pitcher-season, alerts/episodes per detected injury, all-event recall, observable-event recall, earliest and median first-warning lead, and individual event warning histories. An event receives credit only from an alert whose next qualifying UCL event falls within that model's interval; intervening events cannot be skipped.

Non-actionable fraction uses **known-outcome alerted outings** as its denominator. Unknown alerts and mixed/unknown episodes are separate unresolved categories. A fully observed episode without an interval-positive alert is confirmed non-actionable for this defined target; it is not proof of general health. Each model/interval is evaluated independently, not combined into an ensemble or cumulative alert policy. The 1–7-day result is proximal; it cannot be generalized to a seven-day minimum-warning objective.

## Calibration

Brier score and mean-risk bias apply to every interval. Reliability plots use five coarse bins defined by development scores; each bin reports rows, positives and unique events. Event counts can repeat across bins because one injury can label multiple appearances.

A diagnostic unpenalized logistic calibration intercept and slope is estimated only with **at least 10 represented events and 20 positive rows**, using pitcher-cluster sandwich standard errors. Fits with unstable gradients or singular information are flagged. This fit describes calibration only; it never modifies scores, rankings or alert cutoffs. No holdout recalibration is allowed.

## Workload/rest interpretation diagnostic

Reuse the original 2022–2023 and 2024–2025 matched samples, original dependency clusters, final/2–3/4–5 outing alignment and four calendar bins. Require the same balanced events across points; explicitly report additional missing raw-measure support. No new 2026 matches or predictor search are introduced.

Decompose the existing frozen workload logit into original feature contributions, including their missing indicators, in training-score SD units. Verify that component changes sum to the previously reported workload-score change. Examine current pitch counts, existing rest/volume measures, appearance counts and role measures; save event changes and leave-event-out means. Dependency-cluster bootstrap intervals are descriptive and unadjusted. Changes in rest or appearances do not establish skipped scheduled starts, symptoms or clinical intent.

## Decision gate and reproduction

Consistent nominated-family contrasts, joint statistical support, shared-event robustness and limited single-event dependence would motivate a future heterogeneous architecture. Operational usefulness must also be shown separately. Without these, recommend dynamic multi-horizon UCL risk estimation without assigning acute/intermediate/chronic mechanisms. No new architecture is built in this stage.

```bash
.venv/bin/python -m src.injury_analysis.ucl_replication_pipeline
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-replication
.venv/bin/python -m pytest -q
```

Valid checkpoints are hash-verified and reused. Source/config changes trigger a failure rather than silently changing the lock. A later outcome-maturity analysis requires a separately versioned source and output location while carrying forward the same frozen models and policies. Source acquisition is deliberately outside the replication command.
