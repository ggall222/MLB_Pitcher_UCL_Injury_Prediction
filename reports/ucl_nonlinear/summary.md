# Nonlinear UCL model evaluation

## Design and models

This additive stage preserves the refined UCL endpoint, first-event competing-injury rules, chronological partitions, 60-day embargo, completed-appearance cutoff, source missingness and frozen feature representations. No previous analysis is replaced. The 30-day risk sets are unchanged: 87,332 training rows (59 represented events), 28,875 validation rows (25 represented events), and 29,349 exploratory-transfer rows (28 represented events). All-event warning denominators remain 34 validation and 43 transfer episodes; only 20 and 28 have a 7–30d opportunity.

Seven frozen representations were compared: background, profile, departure-8, profile+departure, trajectory-enhanced profile+departure, previous-62 and stable-62. Existing L2 logistic models were reused exactly. The nonlinear search fitted 35 candidates: cubic additive spline logistic models at C=0.1 or 1; histogram gradient boosting with 2, 4 or 8 leaves, depth 1/2/3, 150 iterations, learning rate .05, minimum leaf 200 and L2=10. Stumps are additive on log-odds; larger trees permit interactions. Training-only clipping, median imputation and missingness indicators were retained. There was no random validation split, synthetic injury, class weighting, neural network or new feature selection.

Settings were selected solely on 2022–2023 validation warnings at a 5% budget, with AP above prevalence and simple-model tie breaks. AP leaders were frozen separately as ranking hypotheses. Exact fits, settings and numeric validation thresholds were written before new 2024–2025 predictions. No model was refitted on validation. The already examined transfer period is exploratory; 2026 was not evaluated.

## Principal results

| Hypothesis | Validation AP | Transfer AP | Validation ≥7d warnings | Transfer ≥7d warnings |
|---|---:|---:|---:|---:|
| Linear departure-8 | 0.00727 | 0.00584 | 7/34 | 2/43 |
| Spline departure-8, C=.1 | 0.00679 | 0.00571 | 7/34 | 3/43 |
| Boosted departure-8, 8 leaves | 0.00643 | 0.00533 | 8/34 | 3/43 |
| Spline profile, C=.1 (AP leader) | 0.02466 | 0.04451 | 4/34 | 4/43 |
| Boosted profile, 4 leaves (AP leader) | 0.02943 | 0.00646 | 3/34 | 1/43 |
| Linear stable-62 | 0.01092 | 0.01846 | 3/34 | 9/43 |

The warning leaders are departure models, not the profile AP leaders. Nonlinear departure models do not improve AP; the one-event warning gain has wide paired uncertainty including zero. At ≥14 days, departure-tree warnings are 5/34 validation versus 2/43 transfer; at ≥21 days, 5/34 versus 0/43. Its conditional first-warning median is 26.5 days in validation and 17 days in transfer, among only 8 and 3 detected events. There are roughly 180 and 489 alerts per detected event at 5% budget. A longer median among a few detections is not generally earlier detection.

### A large AP gain is concentrated in very few pitchers

The profile spline is the clearest apparent nonlinear ranking improvement, but one event supplies 68.1% of validation AP and another 79.1% of transfer AP. Removing the corresponding pitcher changes validation AP from 0.02466 to 0.00598 and transfer AP from 0.04451 to 0.00731. Those pitchers are Humberto Castellanos (2022) and Justin Martinez (2025); identifiers and raw training-relative profile values are retained for audit. This diagnostic removes that pitcher's positive and negative rows; it is not a new tuned model or an excuse to discard the cases.

The paired whole-pitcher AP-increment intervals include zero in both periods. Across eight training refits, profile-spline validation AP ranges from 0.00581 to 0.03940. The profile-tree AP leader falls from training AP 0.12947 to validation 0.02943 and transfer 0.00646; its refit validation AP is 0.00446–0.01073. These are material composition sensitivity and overfitting warnings, despite impressive single-fit AP.

### Nonlinearity versus interaction

Additive splines can recover curved profile associations without interactions. The strongest profile-spline validation permutation effects involve historical release-Z variability, release-X position, extension, age and experience. These correlated, persistent profile variables do not establish short-term imminence. The spline-profile gain is too concentrated to claim a reproducible population-level susceptibility relationship.

Departure models retain a lower-velocity/higher-risk direction already present in logistic regression. Curved extension, release-variability and spin effects occur in flexible fits, but they do not consistently increase ranking or warning. Curves are training-supported accumulated local log-odds effects with eight-refit ranges, not medical thresholds. Marginal permutation importance is conditional on the fitted model and can be shared or distorted by correlated features.

The warning-selected profile+departure and previous-62 boosting models are stumps. Their learned log-odds interaction contrasts are zero by construction. Deeper alternatives did not improve their validation warning counts. Trees can fit velocity×release-variability and profile mechanical interactions, but local interaction maps and their refit variability do not establish useful incremental warning. No interaction is interpreted as a biomechanical mechanism.

The broader feature sets do not benefit uniformly: stable-62 nonlinear selections have lower validation and transfer AP than linear stable-62; previous-62 splines are approximately tied with the linear benchmark in transfer AP but lower in validation. Trajectory-enhanced models also fail to establish a consistent warning advantage. The comparison supports curved profile associations in a subset of cases, not a general need for complex injury interactions.

## Calibration, burden and uncertainty

Brier scores, calibration curves with validation-defined bins, warning budgets at 1/2.5/5/10%, ≥7/14/21d recall, first-warning event details, false/nonactionable alert fractions and alerts per detection are saved. Rare outcomes make small Brier differences hard to interpret; a high AP alone does not establish calibration. Fixed validation numeric thresholds are evaluated separately from retrospective top-N budgets. All-eligible-appearance burden includes unknown outcome alerts without assigning healthy labels.

Conditional uncertainty uses 300 paired whole-pitcher resamples; eight common whole-pitcher training refits assess fit variability without retuning. Leave-one-injured-pitcher-out diagnostics, event AP contributions, role and unseen-pitcher checks expose composition sensitivity. Eight-refit effect ranges are descriptive, not 95% confidence intervals. Selection uncertainty and repeated exploratory comparisons remain; an empirical zero-warning bootstrap interval is not a population certainty bound.

## What to retain

Nonlinear modeling does **not materially change the conclusion that reliable UCL early warning has not been demonstrated**. Model form may affect a few profile-based rankings, but it is not established as the main warning bottleneck. Endpoint uncertainty, heterogeneous events, limited positive-event counts and missing direct biomechanical measurements remain plausible constraints; this study cannot identify their causal contributions.

Keep the original linear departure-8 and broader linear benchmarks frozen. Retain spline departure-8 and the shallow departure-tree as prespecified nonlinear warning challengers, and the profile spline as a separate composition-sensitive ranking hypothesis. Do not promote it to a deployed detector based on AP. Exact model artifacts and thresholds are in `frozen_protocol.json`; the next useful test is a genuinely future, complete post-freeze cohort with prospectively archived injury evidence and no retuning.

## Artifacts and rerun

- `notebooks/ucl_nonlinear/`: design/comparison; effects/stability; warning/transfer synthesis.
- `reports/ucl_nonlinear/figures/`: six focused PNG/PDF figures.
- `reports/ucl_nonlinear/tables/`: tuning, paired uncertainty, warning/calibration, influence, local effects, interactions, refits and validation.
- `data/injury/analysis/ucl_nonlinear/`: fitted candidate/refit models, predictions and the training-only interpretation reference.
- Methods and source references: `docs/ucl_nonlinear_methods.md`.

Resume with `.venv/bin/python -m src.injury_analysis.ucl_nonlinear_pipeline`. Execute notebooks with `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-nonlinear`. Completed checkpoints are reused; changed inputs fail loudly.
