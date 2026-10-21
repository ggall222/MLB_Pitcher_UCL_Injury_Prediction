# Initial temporal signal decomposition

This stage asks whether **feature families have different incremental predictive
value at different lead times before a recorded UCL episode**. It extends the
refined endpoint and frozen representations. It does not change injury adjudication,
feature definitions, the database, or previous model artifacts.

## Inputs and time discipline

- Read `data/injury/analysis/ucl_dynamic/landmarks.parquet` and the adjudicated
  event checkpoint through the existing verified loaders.
- Train on 2015–2021; develop on 2022–2023. Preserve the existing 60-day boundary
  embargo and first-event rules. Repeated appearances from a pitcher are expected;
  this is prediction of future outings, not a new-pitcher-only study.
- 2024–2025 is previously examined **exploratory transfer**, never independent
  confirmation. It is scored after `development_freeze.json` records all model
  signatures, families, nominated follow-ups and development contrasts.
- 2026 is excluded from modeling and selection.
- Current-outing measurements are available after a completed outing. Existing
  historical terms use strictly earlier dates. No injury, activation, future
  surgery or later transaction is introduced into predictors.

## Interval estimand and common risk set

The primary intervals are **1–7, 8–14, 15–30 and 31–60 calendar days**, inclusive
and nonoverlapping. A binary model estimates the probability that the next injury
episode is a qualifying UCL episode in that interval, from the current appearance.
This is a landmark formulation, **not** a conditional hazard among pitchers known
to survive to the beginning of a future interval.

Every interval uses the **same evaluable 60-day cohort**, constructed by the
existing `risk_frame(..., horizon=60)`:

1. A qualifying first UCL event within 60 days establishes the first-event outcome.
2. A known non-UCL first injury is a competing outcome, not a healthy-control
   claim. Never skip it to label a subsequent UCL event.
3. An ambiguous/QA-excluded potential UCL or other uncertain first episode is
   censored.
4. Ordinary event-free comparisons require an observed MLB appearance at or after
   day 60, within the existing 30-day grace, and within injury observation coverage.

A first UCL event in another interval is a zero for this particular interval.
Unknown outcomes remain nullable. Summing the four observed binary labels exactly
recovers the observed 60-day UCL label.

Common rows make comparisons less sensitive to changing follow-up rules, but the
restriction preferentially retains pitchers with observable later MLB activity.
This is not a complete prospective deployment population. A sensitivity refits
Full union, Profile and Departure using each interval's upper-bound follow-up
requirement. The primary cohort is not changed after seeing that result.

Separate binary predictions are not constrained to sum to one. They are evaluated
as interval-specific rankings/probabilities and are **not** combined into a
multinomial survival distribution.

## Feature families and fixed model design

Use the **113-field union** of five frozen specifications: Background, Profile,
Departure-8, Previous-62 and Stable-62. Preserve each original specification
exactly. Group membership, timing, representation and source fields come directly
from `reports/feature_consolidation/tables/feature_inventory.csv`.

There are 11 usable families: velocity; release/variability; extension; spin;
movement; location/strike fractions; pitch mix/counts; workload/rest; role/recent
starts; appearance performance; background/experience.

The full union includes correlated alternatives intentionally: this first pass
tests family information, not a new feature-selection exercise. It is not the
entire 536-field registry. No new engineering or single-variable search is added.

Arm angle and within-appearance velocity change inherit earlier exclusions for
insufficient long-history measurement coverage. Prior UCL history inherits the
unverified publication-time exclusion. Prior reported IL count remains in the
background family. Coverage-only fields are not added. `omitted_groups.csv`
retains the original field-level reasons.

For each interval:

- 11 family-only logistic fits;
- 11 full-minus-family logistic fits;
- one full union fit;
- five frozen-representation reference fits.

Thus there are **112 primary interval models**. All use L2 logistic regression,
C=1, no class balancing, and the existing training-only 0.5%/99.5% tail bounds,
median imputation, missingness indicators and standardization. No random internal
split, hyperparameter search or interval-boundary tuning occurs. Original model
files stay unchanged; new fits have unique paths and exact-input signatures.

## Performance and grouped reliance

Report AP, PR lift (AP divided by interval prevalence), ROC-AUC, Brier score and
mean-risk bias. AP differences are paired on identical appearances. Interval
widths and represented event sets differ; even PR lift cannot make the tasks
fully exchangeable.

Ablation loss is `AP(full) − AP(full without family)`. Positive means that removing
the family worsens the refitted model. Negative values are retained: redundancy,
regularization and sample variability can make a reduced model better.

Grouped permutation jointly shuffles the full-model family's transformed fields
and its missingness indicators using the same row permutation within season.
This preserves within-family dependence and season, but breaks links to other
families, pitcher identity and outcomes. Three deterministic shuffles are averaged.
This measures **marginal model reliance**, not unique conditional information or
causality. Bootstrap uncertainty is different from variability across shuffles.

## Uncertainty and temporal contrasts

300 whole-pitcher validation resamples retain all of a pitcher's rows. Identical
draws pair every model and interval. Confidence intervals for AP differences,
ablation and mean permutation loss condition on the fitted models and fixed
permutations. They do not include every uncertainty in the adjudication or model
development process.

The primary temporal contrasts compare normalized ablation loss (`AP loss /
interval prevalence`) in each of 1–7, 8–14 and 15–30 against 31–60 days: **33
contrasts across 11 families**. Centered bootstrap maximum absolute standardized
deviations give a joint approximate 95% band across those contrasts. Pointwise
intervals are also saved. These bands address this declared contrast family,
not every table, trajectory or exploratory analysis in the project.

At most two follow-up families are nominated by mean validation normalized
ablation loss, with negative intervals and uncertainty retained. They are not
declared temporally specific. Six common whole-pitcher training resamples refit
Full union, Profile, Departure, and the nominated families' standalone/ablated
models. Preprocessing is refit each time. This is a limited development stability
check conditional on nomination, not nested selection validation.

## Event composition and practical warning

Different lead intervals can be represented by different events. A retrospective
sensitivity retains only UCL events with opportunities in **every** interval,
removing all 1–60-day rows for other UCL events while keeping the observed
comparison pool. It evaluates saved predictions without changing trained models.
Small common-event support is explicitly reported.

For every standalone family and reference, export tie-aware event contributions
to AP and leave-one-injured-pitcher-out evaluation. These diagnostics never remove
an inconvenient event from the main analysis and do not trigger refitting.

At the fixed retrospective 5% appearance budget, count events warned **within the
specific interval**, all-event and observable-event recall, alerts per detected
event and non-actionable alert fraction. All qualifying calendar-period UCL
events remain in the primary denominator, including those without opportunities.
Conditional recall intervals resample pitchers' event records with the realized
alert set fixed. They do not recalculate an operational threshold.

The 1–7-day interval includes warnings too late for the existing ≥7-day practical
objective. No new deployable threshold or prospective alarm policy is selected.
Unknown-outcome landmarks are not silently counted as negative alerts.

## Matched family trajectories

Reuse the refined dynamic stage's matched event/control rows and controls. Fit
one additional **60-day family-only** logistic model per family on training
data. Standardize its log-odds using its training mean/SD. The same score is used
at every lead time; choosing a horizon-specific model retrospectively for each
point would confound the trajectory with changing transformations.

Aggregate appearances within each event/control/bin first so frequently pitching
events do not receive more weight. Calendar bins match the four prediction
intervals. Outing bins are final, 2–3, 4–5, 6–10 and 11+. These are **not**
equivalent calendar durations.

Primary descriptive figures use balanced pairs across all four calendar bins,
and across final/2–3/4–5 outings. Available-pair tables preserve farther-out
observations and show changing support. Shared-case/control-pitcher connected
components define bootstrap clusters. Pointwise intervals and final-minus-distant
matched differences are descriptive and not multiplicity-adjusted. Validation
and transfer trajectories are reported separately.

These are retrospective matched analyses. A flat offset is not injury imminence;
the family score blends profile and departure fields and has no inherent
physiological meaning. Source reference dates are observable episode proxies,
not known tissue-damage onset.

## Reproducibility and outputs

```bash
.venv/bin/python -m src.injury_analysis.ucl_temporal_pipeline
.venv/bin/python scripts/create_ucl_temporal_notebooks.py
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-temporal
.venv/bin/python -m pytest -q
```

The notebook creation script preserves existing notebooks. Fits, tables and
Parquet caches are written atomically. Verified checkpoints prevent accidental
reuse after source/config changes. Completed original reports, notebooks, models
and source modules have preservation hashes. Nothing writes to DuckDB.

- Configuration: `config/ucl_temporal.yaml`
- Modules: `src/injury_analysis/ucl_temporal_*.py`
- Notebooks: `notebooks/ucl_temporal/`
- Reports/tables/figures: `reports/ucl_temporal/`
- Models/predictions/bootstrap draws: `data/injury/analysis/ucl_temporal/`
- Main synthesis: `reports/ucl_temporal/summary.md`

The 2024–2025 transfer period has already been examined extensively. A future
confirmation must use a genuinely post-freeze chronological cohort, unchanged
endpoint/families/intervals, and adequate event opportunity counts.
