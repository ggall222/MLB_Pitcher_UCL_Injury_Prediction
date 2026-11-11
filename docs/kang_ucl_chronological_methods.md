# Extension 2: chronological evaluation of the fixed UCL experiment

The single major change is the outer evaluation split. Fit all models on
2017–2023, evaluate on 2024–2025, and carry the same models into the locked 2026
test. The experiment is retrospective. The 2026 data existed in earlier project
analyses; this is a held-out layer within this experiment, not globally untouched
or prospective confirmation.

## Fixed methods and training-only quantities

The configuration inherits every experimental setting from the frozen Kang
replication through Extension 1. A new R driver sources the original functions
for XGBoost, inverse-frequency weights, stratified five-fold CV, 50 × five-fold
selection on 80% subsets, mean-Gain/frequency selection, within-domain Pearson
pruning, GAM slope-band selection, clustered AUC uncertainty and Gain refits.
No random outer train/test reservation remains; random CV folds and resampling
are confined to 2017–2023. XGBoost parameters and seed 20261030 remain unchanged.

Every window from 7 through 49 is evaluated using training-only CV. The original
mgcv smoother, derivative criterion, disconnected-band handling and rounded
envelope midpoint determine the integer window. A missing band yields no final
model. A disconnected envelope is reported as ambiguous, even if an integer
midpoint permits fitting under the existing procedure. It is not replaced with
the best-AUC window. Selection and class weights are recalculated from training.
Training-only pitcher-season bootstrap refits estimate mean total Gain.

All 22 original predictor definitions are preserved. There is no learned scaler,
imputer or global feature-missingness threshold in the parent: XGBoost uses native
NaNs. The fixed five-outing and 10% missing-game eligibility rules and season-role
tie-break are recomputed within each period, season by season. Later-season
eligibility cannot change training screening or predictor selection. Retrospective
season roles remain part of the inherited study design, rather than real-time
role assignments. The primary models use only the original 22 candidate fields;
the optional prior-injury sensitivity is not a second model-selection exercise
in this stage and is not fitted.

## Endpoint and cohort continuity

Reuse `endpoint_primary`, `prediction_reference_date`, and all existing
adjudication criteria from the same verified `ucl_temporal_replication` endpoint
artifact as Extension 1. Reuse `kang_ucl_endpoint_data.refine_cohorts` without
modification. Broad first shoulder/elbow events remain blockers; accepted
forearm-to-UCL episodes can become first cases. Generic elbow cases are excluded,
not added to controls. Only pre-reference case games enter; earlier case games
outside each cumulative window remain negative. Season role and identity joins
are unchanged. No landmark-risk, competing-hazard or additional-history rules
are introduced. The 2017–2025 reconstructed row keys, labels and all predictor
values are checked against Extension 1's archived cohorts.

Both the canonical episode file and adjudicated endpoint report an observation
cutoff of **2026-10-20**, verified from their actual `observation_end_date` fields.
The same adjudication already extends into 2026; it is reused, not redesigned.
New 2026 pitcher-game aggregates use exactly the frozen SQL field/QC definitions.
Canonical 2026 first-event framing uses the source available by this cutoff.
Post-cutoff procedures cannot add confirmation. Existing probable or ambiguous
UCL-adjacent episodes, timing flags, exclusions and unresolved status are audited.
Right-censored IL absence is not itself an uncertain injury diagnosis.

Adjudication is retrospective documentation under operational source rules,
not independent clinical review. Source retrieval/confirmation metadata can be
later than historical study events. No 2024–2026 event is used as a training
label; historical cases retain their already-fixed retrospective adjudication.
Source timestamps never enter predictors. This distinguishes period separation
from a claim that all training diagnoses were archived contemporaneously.

## Freeze and follow-up

The training subprocess receives only a 2017–2023 CSV. Its checkpoint signature
depends on that CSV, code, fixed settings and software, not later-period data.
Both roles must finish training (or record unavailability) before a development
freeze is written. Validation/test inputs are loaded and scored only after that
freeze. Access records and model hashes verify the order. Validation does not
select a new window, feature subset, learner or parameter. Neither period triggers
refitting or recalibration.

The only necessary label-observation adaptation is administrative follow-up:
a negative at window W requires game date + W ≤ source cutoff. A documented
positive observed by the cutoff remains known even if the full negative horizon
has not elapsed. Unknown late outcomes remain missing and are excluded from AUC,
not converted to negatives. The same rule uses the training cutoff for training
rows; regular-season training games have complete administrative horizons.
There is no new requirement for a later observed outing. Offseason incident
case exclusion remains inherited from the benchmark, so endpoint coverage is
still limited to its regular-season injury framing.

## Evaluation and descriptive diagnostics

Primary ROC-AUC and 500 pitcher-season bootstrap attempts use the original R
functions. Degenerate draws are counted and omitted as in the parent; undefined
AUC is reported when either class is absent. Very small event counts are flagged
regardless of apparently narrow conditional intervals. Cohort and represented
positive events are reported separately. Games, pitcher-seasons and positive
injury events cannot cross chronological periods; the same pitcher may return
in another season.

Seen/unseen status is relative to training pitcher IDs. Subgroup AUCs are purely
descriptive, with class and event counts; they never choose models. Score drift
uses all scored eligible games, including administratively censored games.
Predictor drift reports per-period missingness, mean/SD and quantiles relative to
the training mean/SD. Reliability bins are five quantile bins learned from
training scores only, with collapsed duplicate edges; rates and Brier/bias use
observed labels. No recalibration occurs. Because training uses class weighting,
model scores are not assumed to be calibrated absolute UCL risks.

The qualitative conclusion policy is frozen in the config before evaluation.
Later intervals overlapping chance, unavailable role/period estimates, or fewer
than five represented events prevent a strong overall persistence claim. This
is a conservative narrative policy, not a new inferential or selection method.
Overlap with the random comparator is not a statistical equivalence test.

## Run

```bash
.venv/bin/python -m src.injury_analysis.kang_ucl_chronological --train-only
.venv/bin/python -m src.injury_analysis.kang_ucl_chronological
.venv/bin/python -m pytest tests/test_kang_ucl_chronological.py -q
```

The full command verifies completed training checkpoints before later-period
evaluation. Source refreshes require a separately versioned experiment; do not
delete a freeze to silently accept changed inputs. Original benchmark and
Extension 1 files are content-hashed and never overwritten.
