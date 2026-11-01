# Kang-style benchmark: methodological specification

Status: **approximate reference implementation, foundation v1**. Specification
and implementation choices were established before inspecting benchmark model
scores. No final chronological UCL model is selected here.

Parent: Kang LS, Mavrommatis S, Ina JG, et al. *Game-Level Temporal Patterns
Identify Optimal Preinjury Windows for Shoulder and Elbow Injuries in Major
League Baseball Pitchers.* OJSM. 2026;14(9).
[Full publication, including appendix](https://journals.sagepub.com/doi/full/10.1177/23259671261477532).
Reviewed 2026-10-29. Source locators below refer to that publication; unknown
details remain unknown rather than being inferred from reported AUCs.

## Published benchmark and implementation decisions

| Component | Kang et al. | Our benchmark | Planned extension |
|---|---|---|---|
| Period | 2017–2025 regular seasons | Same years and game type R | Expanded historical chronological training |
| Observation | Pitcher-game; season-based eligibility | One MLBAM pitcher × game ID; doubleheaders remain separate | Preserve completed-outing landmark |
| Endpoint | First shoulder **or** elbow IL event per pitcher-season; separate endpoint models | Initial official-transaction anatomy; first combined shoulder/elbow placement; forearm remains separate | Adjudicated UCL-specific endpoint |
| Reference date | IL placement | `transaction_date`, not retroactive effective date, surgery or activation | Existing adjudicated earliest defensible date rules |
| Sources | Savant/Baseball Reference; Spotrac/Pro Sports Transactions | Existing Savant + official boxscore/people + MLB transaction episode pipeline | Retain provenance and UCL adjudication evidence |
| Exposure/missingness | At least 5 season appearances; at most 10% games missing a required field | Same thresholds on 23 declared inputs; residual NaNs retained | Landmark eligibility and historical support rules |
| Role | Majority role in season; innings break ties | Official appearance starts; count majority, then outs by role; unresolved ties excluded | Role known at prediction time |
| Controls | Methods: no shoulder/elbow IL; Results: no IL | Primary no IL; target-specific controls counted separately | Full prospective-style risk set and competing-event handling |
| Window | Cumulative 7–49 days pre-IL | Seven fixed weekly windows; earlier pre-IL games negative | Prespecified lead-time/family contrasts |
| Predictors | Age, outing workload, release/velocity/spin, pitch proportions | 23 mapped inputs; 18 existing plus 5 necessary aggregates | Limited original families, not all engineered derivatives |
| Preprocessing | Absolute horizontal release; domain Pearson pruning | Same conceptual operations; existing source QC; native XGBoost NaN handling | Training-only preprocessing retained |
| Learner | Class-weighted XGBoost | Same family; one fixed parameter configuration | Hold learner fixed for attributable comparisons |
| Selection | Repeated training-only gain/stability selection; domain correlation >0.7 | Three training subsample gain fits and greedy domain pruning, nested inside CV | Frozen selection within chronological development |
| Split | Approximately 67/33 stratified random pitcher-games | Same ratio; fixed reservation stratified at 49 days, reused across windows | Train 2015–2023 / validate 2024–2025 / reserve 2026 |
| Dependence | Same pitcher can occur in train/test; pitcher-season bootstrap | Overlap explicitly reported; 500 pitcher-season AUC bootstraps | Event-aware chronological separation and existing embargo |
| Window selection | GAM-smoothed CV AUC; slope band at 25–35% maximum slope | Highest mean CV AUC; shorter exact tie; **not a GAM replication** | Prespecified temporal comparisons, not test-selected windows |
| Metrics | Development and holdout ROC-AUC with bootstrap CI | AUC primary; AP, prevalence, PR lift, Brier and mean risk descriptive | Event recall, lead time, calibration and alert burden |

Machine-readable comparison: [kang_benchmark_comparison.csv](kang_benchmark_comparison.csv).
“Same” means the documented design element matches; it does **not** mean
individual source records, final features, models or estimated results match.

## What is not specified sufficiently for exact replication

The Methods, Tables 1–2, Appendix A1–A3 and feature-importance results support
the feature concepts and season rules. They do not provide the complete
training configuration, exact split seed, sampled-subset size, selection
frequency/gain cutoff, tuning grid, remaining-NaN strategy or final feature
lists. The temporal methods provide slope thresholds but not the GAM basis,
penalty, grid spacing or exact conversion of an interval to a final integer
window. The software version string does not pin executable R/package versions.

Three ambiguities materially affect comparability:

1. **Control definition.** Methods allow other injury IL seasons; Results describe
   controls without IL placement. Primary controls here have no recorded IL
   episode anchored in that calendar year. The alternative removes shoulder/elbow
   and unknown-anatomy seasons, permitting known other injuries. Both counts are
   saved; only the predeclared primary is modeled. This does not resolve the
   publication's discrepancy. Carry-in episodes anchored in an earlier year and
   undocumented injuries can still be missed; this is not a longitudinal risk set.
2. **Earlier case games.** The binary indicator and inclusion of preinjury games
   support treating games earlier than W days as negative, which is our explicit
   choice. Whether the original implementation instead omitted those games is
   not resolved by the text. Counts for that alternative are included in
   `cohort_support.csv`; no alternate models are mined for better agreement.
3. **Release Y versus extension.** Tables define longitudinal field-coordinate
   release Y, while the appendix/importance discussion lists extension. These
   are distinct measurements. Both are included and documented separately;
   within-domain pruning may remove one. Exact author naming/feature list is needed.

The missingness denominator and season role also use the full season, and
controls require future season injury ascertainment. These are retrospective
benchmark definitions, **not real-time eligibility variables**. They cannot
be transferred unchanged into chronological early-warning deployment.

Transaction descriptions were retrieved retrospectively. Their recorded dates
are not independently verified historical publication timestamps. Rebuilding
episodes at the 2025 cutoff prevents later transaction rows from revising QA;
it cannot establish that a source never revised an older description.

## Cohort construction and exclusions

1. Open `database/mlb_ucl.duckdb` **read-only**. Select `appearances` with game date
   2017-01-01 through 2025-12-31 and game type R. Join raw pitches by
   `(game_id, pitcher_id)` only for the five missing aggregates, with the same
   date bounds. No full pitch table is loaded into Python.
2. Keep canonical position P or TWP; two-way players are included when pitching.
   Emergency position-player outings are excluded. Primary-position metadata
   is the existing canonical lookup, not a newly reconstructed historical roster.
   Zero-physical-pitch appearances absent from the Statcast aggregate are not added.
3. Count all season outings. The missing fraction is games with any missing
   input divided by games, **not** missing cells divided by cells. Retain at
   least 5 outings and at most 10% missing games. No blanket imputation or
   substitution of fastball-only measurements occurs. Single measured pitches
   have undefined sample SD. Native NaN handling applies to residual missingness.
4. Infer season role from official boxscore start indicators, never pitch count.
   Ties use innings in each role (outs are equivalent and avoid baseball-decimal
   confusion). If still tied or role is unknown, exclude and count. Keep each
   outing within its pitcher-season role stratum, even if that outing's role differs.
5. Read the existing normalized pitcher IL transactions with SQL-filtered
   `transaction_date` in 2015–2025. Reuse `src/injury/episodes.py:build_episodes`
   with existing settings, observation end 2025-12-31, and observed appearances
   through that date only. The 2015–2016 records provide lifecycle history, not
   benchmark cases. This avoids importing episode QA flags revised by **2026**
   transactions from the current canonical event file. From this as-of-2025 view,
   select 2017–2025 initial anatomy and placement metadata, not revised UCL
   diagnosis, return/surgery predictors, or 2026 event records.
   Only Elbow / UCL and Shoulder are cases; forearm/flexor diagnoses are not
   silently relabeled. A case must be within that year's observed MLB regular-season
   dates. All calendar-year IL anchors can disqualify controls.
6. Choose the first shoulder/elbow placement **before** applying case QA. Exclude
   an absent placement ID, orphan transfer, left censoring, overlapping episode,
   uncertain timing, or multiple injury regions. Do not promote a later event
   when the first is ambiguous. This conservative QA is an explicit approximation
   beyond the paper's described criteria. Case seasons must have pre-IL appearances
   to contribute rows; zero-opportunity events remain in the audit.
7. Include only games dated strictly before case placement. Include all qualifying
   control-season games. Same-day games are excluded because within-day IL
   announcement timing is unknown. A positive satisfies `1 <= placement_date -
   game_date <= W`. All earlier case games are negative in this reference.
   Cases of the other target anatomy are not controls. Multiple seasons and
   games from the same pitcher remain linked by MLBAM ID.

This reproduces a retrospective **pre-IL discrimination target**, not an
unselected W-day cumulative incidence estimand. Negative control labels describe
the complete observed season, not fully observed W-day follow-up extending into
2026. Do not reuse these season-boundary negatives in a future-risk pipeline.

## Features and existing component reuse

Read-only schema inspection confirmed the existing `pitches`, `appearances`,
`model_features`, `workload_features`, `mechanical_features`, `pitcher_baselines`,
`injuries`, `injury_events`, `arm_injury_episodes` and `injury_labels` objects.
The broader multi-region injury episodes/transactions are separate Parquet
products. The benchmark therefore needs an adapter, not a replacement warehouse.
`config/ucl_modeling.yaml`, the frozen temporal/replication configurations,
`config/ucl_candidate_features_v2.yaml`, and the consolidation inventory were
reviewed; none is edited or used to redefine the parent benchmark's inputs.

The [23-row crosswalk](../reports/kang_benchmark/tables/feature_crosswalk.csv)
checks each source against DuckDB columns and `docs/data_dictionary.csv`; group
names reuse the existing consolidation inventory. No previous outcome-driven
retained-feature list is imported.

Existing `appearances` supplies age, physical pitch count, official innings and
batters faced, all-pitch mean/max velocity, spin mean, release Z, extension and
nine pitch fractions. Additional SQL summaries are all-pitch sample velocity SD,
spin maximum/SD, mean absolute X and mean longitudinal Y. They live only in a
derived cache; the master engineering tables and dictionary remain unchanged.
Published SD denominator and order of absolute-value aggregation are unspecified;
we use sample SD and average absolute pitch X. Raw source QC follows
`src/build_appearances.py`: velocity 30–110 mph, spin 0–4500 rpm, X -7–7 feet.
Y has finite-value screening only. Pitch fractions use all physical pitches,
including unclassified/other types in their denominator; the nine fractions
need not sum to one. Absent types are structural zero; tracking absence is NaN.

Pitching inputs describe a **completed outing**. Age is available before it.
No rolling feature, career mean, injury field, role label, season total or
outcome enters the 23-column design matrix. Source measurement changes and
historical pitch-type reclassification remain limitations.

| Existing component | Reuse in foundation |
|---|---|
| `src/ingest_statcast.py`, official boxscores/people | Existing retrieved data; no re-download or replacement |
| `src/build_appearances.py`, `appearances`, `pitches` | Grain, pitch masks, official role, aggregates |
| `src/injury/parser.py`, episode construction, normalized transaction Parquet | MLBAM identity, initial anatomy; same episode builder capped at 2025 |
| Data dictionary; consolidation feature inventory | Field semantics, units and family names only |
| `src/evaluation.py:rank_metrics` | Tie-safe AP and ROC-AUC |
| `ucl_dynamic_endpoint`, `ucl_revised_*` | Inspected; reserved for UCL/chronological extensions |
| `ucl_nonlinear_*`, `ucl_temporal_*`, PCA/PLS | Inspected; conventions retained, models/results not overwritten or reused as benchmark scores |

Existing nonlinear candidates use different learner formulations; the minimal
new XGBoost adapter is necessary to match the parent model family. Existing
dynamic risk-set labels and disjoint temporal intervals are deliberately not
called: they answer a different question from cumulative pre-IL windows.

## Model and selection implementation

`config/kang_benchmark.yaml` is the complete deterministic configuration.
One shared 67/33 random-game reservation per endpoint/role is stratified on the
maximum 49-day label, preventing games switching between development and
holdout when a window changes. That allocation detail is our approximation;
the paper does not document how partitions were shared across windows.

For each weekly window, stratified five-fold CV runs entirely inside development.
Within **each fitting fold**, three 80% stratified refits estimate mean gain.
Features are ordered by gain with lexical tie-breaking. Greedy within-domain
Pearson pruning removes a candidate only if it is correlated above 0.7 with an
already retained feature. Constants/all-null columns are removed using that
fitting fold. No additional minimum-gain threshold is invented. All selection
is rerun on full development before the final model fit. This bounded procedure
is not the paper's 50-repetition stability-selection algorithm.

Fixed XGBoost: 150 trees, depth 3, learning rate .05, minimum child weight 5,
L2 1, L1 0, histogram trees, no feature/row subsampling in the final learner.
Each fit receives inverse class-frequency weights normalized to mean one.
There is no hyperparameter search, holdout early stopping, scaler, imputer,
test-selected feature, recalibration, or operational threshold selection.

The reference window maximizes mean development CV AUC on [7,14,21,28,35,42,49].
This keeps a usable model implementation without claiming the undocumented
GAM plateau procedure has been reproduced. Fold SD is shown descriptively,
not as a confidence interval. Different windows change target prevalence and
case composition; a longer best-scoring window is not evidence of a mechanism.

Each task saves its window, feature names, parameters, split and model hashes
in `freeze.json` **before holdout scoring**. Only that model is scored on its
random holdout. AUC intervals use 500 whole-pitcher-season resamples of holdout
rows. They are conditional on the fitted model, not uncertainty from the full
selection process; clustering by season also does not remove cross-year
dependence or pitcher/event overlap across partitions. AP/PR lift give rare-event
context. Weighted-XGBoost scores are not calibrated population risks.

## Reproduce, inspect or resume

```bash
.venv/bin/python -m pip install -r requirements-benchmark.txt
.venv/bin/python -m src.injury_analysis.kang_benchmark         # cohort/crosswalk
.venv/bin/python -m src.injury_analysis.kang_benchmark --fit   # four bounded tasks
.venv/bin/python -m pytest tests/test_kang_benchmark.py -q
```

On macOS, XGBoost requires OpenMP; the standard installation is `brew install
libomp` ([official installation guide](https://xgboost.readthedocs.io/en/stable/install.html)).
This workstation has no Homebrew but already has the runtime distributed with
scikit-learn. The successful local execution used:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
  .venv/bin/python -m src.injury_analysis.kang_benchmark --fit
```

The fallback is environment-specific, not a replacement for installing the
runtime on another machine. No global library or interpreter was modified.

Inputs and completed model artifacts are hash-verified before checkpoint reuse;
incomplete/corrupt tasks are rebuilt. Cached input fingerprints include database
size/mtime, transaction-file SHA256, episode settings/builder, config and adapter code. The 19-GB database is
not hashed in full and must be treated as an immutable source during a run.
Outputs reside only under `reports/kang_benchmark` and
`data/injury/analysis/kang_benchmark`. `--config` supports a separate run folder;
the adapter explicitly rejects dates beyond 2025. The cache is regenerable,
while methods, config, notebook and small selected summary tables are public.

## Interpretation and next gate

The [computed summary](../reports/kang_benchmark/summary.md) supplies actual
cohort counts and scores. Differences from the publication are not corrected by
selectively adding cases, changing controls, or tuning against its reported AUCs.
Author code or clarification is needed for closer replication, especially control
construction, earlier case games, Y/extension, selected feature lists and window
smoothing. No contact has been made on the user's behalf.

Freeze this explicitly approximate reference after reviewing those issues. The
next controlled comparison should change only the elbow endpoint to adjudicated
UCL, with features and learner held fixed. Then introduce the chronological
evaluation and limited lead-time/family extension, with appropriate censoring
and embargo. Prior outputs remain exploratory support. 2026 stays reserved
from this stage and cannot be used to select any benchmark setting.
