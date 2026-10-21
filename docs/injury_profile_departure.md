# Profile/departure methods

The [two notebooks](../notebooks/injury_profile_departure/README.md) and [computed findings](../reports/injury_profile_departure/findings_summary.md) document this experiment. The source database is connected read-only; existing cohorts, labels and matches are reused.

## Representations

- Profile: eight existing strictly prior baseline means (fastball velocity, release X/Z and their SDs, extension, spin, zone percentage), age, throwing hand, observed prior MLB starts, days since debut and prior publicly reported IL count.
- Departure: the corresponding eight current standardized pitcher deviations. Prior baseline SD is part of the representation; these features do not mathematically eliminate every stable pitcher attribute.
- Common adjustment: 15 existing workload variables, current starter role and month. Their conceptual ambiguity is retained rather than labelling them static traits or mechanical departures.
- Secondary history: eight pre-existing five-outing slopes excluding the current date.
- Comparators: original 62 predictors and workload/history.

Baseline means use at most 20 earlier outings, require five valid values and exclude the current date and ambiguous game timing. They adapt over time and can incorporate earlier subclinical changes. Age and prior injury history are contextual associations, not established causal susceptibility.

## Fitting and evaluation

All variants retain identical rows and targets within injury group/partition. Median imputation, missing flags and scaling fit on training only. Existing L2 logistic C=1 and chronological splits are retained, with tight convergence tolerance. No variable screening or hyperparameter selection uses test outcomes.

Primary paired AP and Brier uncertainty resamples whole evaluation pitchers, holding trained models fixed. Event uncertainty resamples each pitcher's events. The standard warning is 7–30 days at a 5% outing alert budget; other established budgets remain available. Every eligible injury event remains in the denominator. A separate frozen validation threshold measures actual test alert-rate drift.

Brier measures probability error; it is not calibration alone. Reliability bins use validation score quantiles, with raw counts and positive support saved. Probabilities are conditional on conservative observed-follow-up risk sets. No calibration map is fitted.

## Score timing

The combined logit is exactly decomposed into intercept, profile and departure terms, including each source field's missing indicator. Between-pitcher sum-of-squares fractions are descriptive; variance terms are correlated and not causal shares.

Use the exact existing matched controls and appearances. Primary windows are entirely outside training in 2022–2025. Both members must have complete P+D measurement support; balanced curves use the same pairs at every point. Report final-five-outing and calendar-bin trajectories with counts and within-curve simultaneous bootstrap bands. Multiple categories/components remain exploratory. Calendar injury dates are recorded proxies, not biological onset.

## Checks addressing interpretation

Complete-case refits, shared availability indicators, omission of final-six-day positive rows, removal of slow history fields, workload-adjusted arm checks, and 12 whole-pitcher training bootstrap refits test the main findings. Near-event rows are removed, never labelled healthy. Shared-availability fits disable automatic missing flags to avoid duplicate copies and their regularization effects.

The data already include previously examined validation/test periods. A subsequent confirmation needs a newly frozen chronological period. The source event construction and prior labels retain their existing timing/anatomy limitations.

