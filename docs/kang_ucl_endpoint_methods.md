# Extension 1: endpoint specificity within the frozen Kang framework

Question: Does the broad-elbow ranking signal persist for documented UCL injury?
The comparator is the completed `kang_replication` implementation, including its
documented approximations to Kang et al. (DOI 10.1177/23259671261477532). No claim
of exact author-code replication is added here.

## Frozen experimental design

`config/kang_ucl_endpoint.yaml` inherits every model/cohort-screen setting from
`config/kang_replication.yaml`; only output/cache paths differ in the effective
model configuration. The runner calls the existing `execute_task` and the exact
unchanged `scripts/kang_replication.R`. It never invokes the benchmark runner's
data preparation or reporting, so archived benchmark outputs are read-only.

Held fixed: 2017–2025 regular-season pitcher-games, at least five outings per
season, the 10% missingness gate, retrospective majority season role with innings
tie-break, 22 Appendix A1 predictors, 67/33 stratified random games, overlapping
pitchers across sets, seed 20261030, inverse-frequency class weights, untuned
fixed XGBoost parameters, 50 × five-fold feature selection, 80% subsets,
frequency threshold 0.5, within-domain Pearson pruning at |r| > 0.7, integer
windows 7–49, five-fold window CV, mgcv GAM slope-band selection and rounded
envelope midpoint, ROC-AUC, 500 pitcher-season AUC resamples and 500 training
bootstrap Gain refits. The GAM is the original AUC-curve smoother, not a new
injury learner. Earlier case games outside the current window remain negative.

Random partitions are regenerated using the identical algorithm and seed,
stratified on the 49-day label. Changing labels and cohort membership changes
some assignments; this is not a paired-prediction comparison and is not a
chronological evaluation. No outcome-driven seed or partition search is allowed.
The automatic prior-region-history sensitivity still uses prior **broad elbow**
history, exactly as in the parent; it is supplemental, not a new UCL predictor.

## Authoritative endpoint reused without redesign

Reuse `endpoint_primary` and `prediction_reference_date` from the verified saved
adjudication at `data/injury/analysis/ucl_temporal_replication/adjudicated_events.parquet`.
Its authoritative implementation is `src/injury_analysis/ucl_dynamic_endpoint.py`
(`run_endpoint`, `classify_evidence`, `validate_procedure_link`), with the
180-day procedure bound in `config/ucl_dynamic.yaml`. The source checkpoint and
all reused inputs are hashed before fitting. The dynamic/temporal risk-set,
censoring, splits, predictors and models are **not** imported.

Primary events require official episode-linked UCL diagnosis text or an existing
identifier-linked MLB UCL procedure satisfying compatible anatomy, the 180-day
bound, activation/episode boundaries and absence of an intervening observed MLB
return. Procedure/recovery-only text does not independently confirm a new injury.
Existing exclusions for ambiguous timing, overlapping/continuation episodes,
postoperative entries, incompatible laterality/anatomy, identity problems and
incomplete sources remain part of endpoint adjudication. Compatible forearm-to-
UCL progressions are allowed. Generic elbow diagnoses and probable-only cases
do not qualify. “Adjudicated” means documented under the project's operational
rules, not independent clinical chart review.

The endpoint source observation period ends 2026-10-20; only 2017–2025 study
events and games enter this experiment. Confirmation fields may contain later
source-retrieval timestamps. These remain retrospective outcome evidence, not
predictors or a claim of contemporaneously archived confirmation. No 2026
appearance, performance score or chronological split is used.

## Endpoint-to-benchmark cohort adaptation

1. Reuse the archived appearances, broad episodes and season eligibility/roles.
2. Preserve all broad shoulder/elbow first-event blockers. Add accepted UCL
   episodes (including confirmed forearm presentations) to that event universe.
   Require the original placement identifier and regular-season placement date.
3. Select the first combined target episode per pitcher-season using the same
   placement-date/event-ID ordering. Only accepted UCL first episodes become
   cases. A later UCL event cannot replace an earlier generic elbow or shoulder
   episode. This preserves the benchmark's first-event framing.
4. Preserve the original no-shoulder/elbow control population except that a
   newly recognized first UCL episode promotes that pitcher-season to a case.
   Removed non-UCL elbow case-seasons are excluded, not added as healthy controls.
   Known UCL forearm cases are not silently left among controls.
5. For UCL cases, count cumulative 1..W days before the established
   `prediction_reference_date`, keeping only strictly earlier games. This may
   remove appearances or shift labels relative to the original placement date.
   It is a required endpoint-timing change, recorded separately; surgery date
   never substitutes for the reference date. Controls retain their original games.

Thus the primary contrast is documented UCL cases versus the preserved broad-
target-free controls. It is not a new UCL-versus-all-other-elbow classifier.
Source UCL events outside the eligible first-event frame are reported but not
promoted into the analysis. Event identity joins use canonical ID plus MLBAM
validation, never player names. Roles are never pooled.

## Equivalence, support and interpretation

Before fitting, automated checks require identical inherited settings, identical
feature crosswalk and R implementation, unchanged source hashes, unique game
keys, identical feature values and original season roles. A technical support
check applies the actual R split function and requires both outcome classes in
holdout and at least five training positives/negatives at every candidate window
so the unchanged five-fold design is defined. Failure is reported without pooling
roles or choosing a different seed. Passing this gate does not establish adequate
statistical power or many independent injury events.

No qualifying GAM slope band yields an unavailable final model, without a
maximum-AUC fallback. Disconnected bands and midpoint-outside-band results remain
explicitly flagged. AUC intervals are conditional on the original random-game
framework; they do not eliminate shared-pitcher dependence or identify a causal
effect of endpoint specificity. Different case composition, reference dates,
class weights induced by prevalence, selected features/windows and random
assignments all follow from changing the endpoint. They limit attribution of an
AUC difference to anatomy alone. No new inferential comparison is introduced.

## Run

```bash
.venv/bin/python -m src.injury_analysis.kang_ucl_endpoint --prepare-only
.venv/bin/python -m src.injury_analysis.kang_ucl_endpoint
.venv/bin/python -m pytest tests/test_kang_ucl_endpoint.py tests/test_kang_replication.py -q
```

Outputs live under `reports/kang_ucl_endpoint` and
`data/injury/analysis/kang_ucl_endpoint`; the synthesis notebook lives under
`notebooks/kang_ucl_endpoint`. Model checkpoints reuse the parent's fingerprinted
resume behavior. No parent artifact or database object is modified.
