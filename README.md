<<<<<<< HEAD
# MLB_Pitcher_UCL_Injury_Prediction
=======
# MLB starting-pitcher UCL early-warning research database

## Dimensionality reduction

Three follow-on notebooks examine PCA redundancy and loadings, separate prior pitcher profiles from within-pitcher departures, and compare reduced versus original models under the existing temporal evaluation. Start with the [notebook guide](notebooks/injury_dimensionality_reduction/README.md), [computed results](reports/injury_dimensionality_reduction/SUMMARY.md), and [methods](docs/injury_dimensionality_reduction.md). All transforms are fitted on training observations; prior analyses and the central database are preserved.

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-reduction
```

## Feature families and pre-injury timing

Three additional executed notebooks compare 18 predictor specifications across seven injury categories, then examine matched calendar and appearance trajectories, event-level shape, and sensitivity to injury timing and cohort restrictions. Start with the [notebook guide](notebooks/injury_signal_analysis/README.md), [computed findings](reports/injury_signal_analysis/SUMMARY.md), and [methods](docs/injury_signal_analysis.md). The extension saves 28 CSV tables and 12 figures in PNG/PDF, preserves the existing sources, and treats the previously examined 2024–2025 test period as exploratory.

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-signals
```

## Injury-type exploratory notebooks

Eight executed notebooks compare pre-injury trajectories, chronological predictability, severity proxies, recurrence, warning time and feature families across injury categories. They reuse the existing features and canonical injury episodes, with a read-only DuckDB connection. Start with the [notebook order and run instructions](notebooks/injury_type_analysis/README.md), [computed findings](reports/injury_type_analysis/SUMMARY.md), and [analysis methods](docs/injury_type_analysis.md).

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/run_injury_analysis_notebooks.py
```

The executed analysis includes 3,552 eligible episodes, 1,225 matched event/control pairs and 62 common historical predictors. Training ends in 2021, validation uses 2022–2023, testing uses 2024–2025, and 2026 remains descriptive. These are exploratory comparisons; the final capstone model has not been selected. Saved results include 37 CSV tables and 11 figures in PNG/PDF. See the [execution manifest](reports/injury_type_analysis/notebook_execution.json) and [verification](reports/injury_type_analysis/verification.json).

## General pitcher injury dataset (2015–2026)

The project now also builds documented MLB pitcher IL episodes across all body regions and illnesses, using official MLB transactions plus independent RosterResource and existing UCL surgery evidence. This is a separate dataset; it reads the central DuckDB without altering the existing UCL features or labels.

```bash
.venv/bin/python -m src.injury_pipeline          # Full build using resumable caches
.venv/bin/python -m src.injury_pipeline --update # Refresh recent source periods
.venv/bin/python -m src.injury_pipeline --offline
```

Canonical CSV/Parquet datasets are in [`data/injury/processed/`](data/injury/processed/). See the [executed injury build summary](reports/general_injury/build_summary.md), [validation metrics](reports/general_injury/mlb_pitcher_injury_validation_summary.csv), and [methods, schema, limitations and rerun instructions](docs/general_injury_pipeline.md). Uncertain anatomy, conflicting sources, missing activations and retrospective publication timing remain explicit; IL dates are observable proxies rather than automatically biological onset.

## Original UCL appearance study

This university Data Science capstone studies changes in pitch mechanics, workload and command before a publicly reported arm problem subsequently associated with a UCL procedure. Predictions are made **after a pitcher completes an appearance**. The original UCL modeling grain is **one MLB starting pitcher × one regular-season appearance**. The original source features retain nullable measurements. The separate broader-injury notebooks above fit exploratory models with training-only preprocessing; no final capstone model has been selected.

The historical configuration requests **2015-03-01 through `latest`**, which resolves to yesterday when each run begins. Actual successful coverage, database size, event counts and unresolved periods are recorded in [full_build_summary.md](reports/full_build_summary.md), [season_coverage.csv](reports/season_coverage.csv) and [modeling_readiness.md](reports/modeling_readiness.md). A season being represented does not guarantee every game was retrieved. During a running build these reports describe the last completed derivation.

The original 2024 proof of concept is preserved in `database/snapshots/poc_2024.duckdb`, `reports/poc_2024/`, and `config/config_poc_2024.yaml`. Raw cached responses are retained across updates.

<!-- EXECUTED-BUILD-START -->
Executed coverage: **2015-04-05–2026-09-27**. The **19.08 GiB** database contains **8,150,993 raw records**, **236,628 appearances**, and **55,244 official starts**. The 30-day starter target has **355 positives (0.704%)** representing **98 distinct events**. Unresolved source failures: **0**; unmatched injury identities: **21**.
<!-- EXECUTED-BUILD-END -->

## Where the data lives

The central database is **`database/mlb_ucl.duckdb`**. [data/SOURCE_TABLES.md](data/SOURCE_TABLES.md) maps each source to its tables. [sql/schema.sql](sql/schema.sql) defines source tables; [sql/views.sql](sql/views.sql) defines model views. [docs/data_dictionary.csv](docs/data_dictionary.csv) describes persisted columns.

| Source | DuckDB tables |
|---|---|
| Baseball Savant / Statcast | `pitches` |
| MLB schedule and official boxscores | `games`, `boxscore_appearances` |
| MLB native pitch telemetry | `pitch_tracking`, joined through `aligned_pitch_tracking` |
| MLB identities and team history | `pitchers`, `teams`, `team_seasons` |
| MLB transactions, rosters, prior-year totals | `transactions`, `roster_snapshots`, `season_pitching` |
| Roegele reconstruction and internal-brace lists | `injuries` |
| Linked injury episodes and procedure mapping | `arm_injury_episodes`, `injury_events`, `injury_event_procedures` |
| Engineered master and labels | `appearances`, `appearance_pitch_metrics`, `workload_features`, `mechanical_features`, `pitcher_baselines`, `injury_labels`, `model_features` |

`data/cache/` contains immutable compressed source responses with URL indexes and hashes. `data/exports/` contains Parquet exports and the master feature allowlist. `reports/` contains audits rather than complete raw source dumps. A DuckDB writer locks the main file; use the preserved snapshot for independent browsing during ingestion.

## Install and run

```bash
cd "/Users/garrettgall/Desktop/MLB pitch stats/mlb_ucl_project"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Resume all configured source periods and rebuild engineered data and reports.
.venv/bin/python -m src.pipeline --full-refresh

# Resume interrupted periods and ingest new dates through configured latest.
.venv/bin/python -m src.pipeline --update

# Validate the stored database without retrieving sources.
.venv/bin/python -m src.pipeline --validate-only
.venv/bin/python -m pytest -q
```

Python 3.9.6 is the executed environment; dependencies are pinned. No database server or API credentials are required. Baseball Savant's public CSV endpoint is called directly; pybaseball is not required.

Configuration in [config/config.yaml](config/config.yaml):

```yaml
statcast_start_date: '2015-03-01'
statcast_end_date: latest
historical_build: true
statcast_game_dates_only: true
network_workers: 4
```

CLI `--start YYYY-MM-DD` / `--end YYYY-MM-DD` override the run range. Existing raw records remain in the central database, even when a subsequent invocation requests a narrower range. Use a separate configured database/export/report path for an independent smaller study.

For staged historical operation:

```bash
# Daily pitch acquisition within season checkpoints.
.venv/bin/python -m src.pipeline --historical --ingest-only --phase statcast

# Official boxscores, transactions, identities, prior-season totals and surgeries.
.venv/bin/python -m src.pipeline --historical --ingest-only --phase support

# Native MLB telemetry for every collected game.
.venv/bin/python -m src.pipeline --historical --ingest-only --phase tracking

# Recompute histories, labels, exports and audits from stored sources.
.venv/bin/python -m src.pipeline --derive-only

# Reproduce from retained source cache; missing cached items remain explicit failures.
.venv/bin/python -m src.pipeline --full-refresh --offline

# Deliberately retrieve source corrections, including completed pitch dates.
.venv/bin/python -m src.pipeline --update --refresh-cache
```

`--full-refresh` now preserves raw tables and refreshes engineered data. It does not delete the database. Derived schema changes are additive; removing an existing master column raises an error. The original 459-feature contract remains available, alongside additional history/context fields. The initial database snapshot preserves the original engineered results.

## Historical acquisition and recovery

MLB's regular-season schedule identifies game dates, avoiding spring training and winter empty requests. Savant is downloaded one day at a time with four bounded workers, retry/backoff for temporary failures and natural-key upserts. Completed days and games are recorded in `ingestion_state`. Season phase summaries are recorded in `historical_checkpoints`, and each season writes a source manifest. Cancelled games are retained as `Cancelled`, even though MLB places them in its broad `Final` category; associated downloads are `not_applicable`. The per-source status and missing-game audit are authoritative when a phase checkpoint exists but some requests failed.

HTTP response bodies are immutable gzip files named by SHA-256; URL indexes select the latest cached response. Completed pitch/boxscore/tracking items are skipped on routine updates. Schedules, transactions, people and surgery lists use a 24-hour mutable cache policy. Interrupted builds retain completed periods; rerunning retries failed or missing items. Source revisions require `--refresh-cache` when immediate correction is needed. Only one process should write the database at a time.

Pitch aggregation runs by season to bound memory. Prior mechanical histories, baselines and official workloads use vectorized calculations. Engineered tables refresh atomically, so histories reflect newly collected earlier seasons. `src/database.py` remains the persistence boundary; feature builders operate on pandas.

Only regular-season MLB games are currently requested. Spring training, postseason and minor-league pitches are not silently mixed into the population. All starting and relief appearances in the retrieved MLB games are preserved. Future postseason ingestion must retain its game type and use a separate cohort.

## Starting appearance definition

`is_starter` uses official boxscore `gamesStarted`, with the first pitcher in official pitching order as fallback. **Pitch count never defines starter status.** `starter_source` records the method. Each game/team must have exactly one starter, and the starter is cross-checked against the first observed Statcast pitcher for the corresponding fielding side. Conflicts remain ambiguous for review.

Openers count as official starters. This definition does not guarantee a conventional rotation start; opener/role sensitivity analyses should follow. The core `model_ready_starting_pitcher_appearances` view requires:

- An unambiguous official MLB regular-season start.
- An official boxscore and reliable game timing; resumed games with uncertain timing are excluded.
- At least 80% measured pitch velocity and a physical-pitch versus official-count difference of at most two.

`model_ready_pitcher_appearances` retains the full master population. `training_starting_pitcher_appearances` additionally requires outcome eligibility and an observed 30-day target. Other target horizons require their own nonnull-label filter. Raw relief data and excluded quality rows remain available.

## Injury events and labels

The [Roegele workbook](https://docs.google.com/spreadsheets/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/edit) supplies reconstruction and internal-brace records. Credit Jon Roegele and MLB in research outputs. Source notes indicate incomplete public reporting, first-of-month placeholders, January 1 placeholders and non-throwing-arm markers. Raw dates are retained; uncertain dates become intervals, and out-of-scope/unconfirmed cases remain auditable.

MLBAM identifiers are preferred. Exact-name legacy joins are flagged for review. Additional reconciliation uses the [Chadwick register](https://github.com/chadwickbureau/register), FanGraphs identifiers, MLB biographies and career compatibility. Ambiguous candidates remain unmatched. `unmatched_injury_audit.csv` traces every original unmatched record and its candidate evidence.

An arm-IL episode begins with the first retrieved public elbow/UCL/forearm announcement. A confirmed UCL procedure is linked to an active compatible episode. The public announcement is the label reference; administrative/retroactive IL dates are separate. Clinical diagnosis, first symptom and last healthy appearance remain nullable. Last observed pre-reference appearance does not establish that the pitcher was healthy. Generic arm-IL-to-UCL linkage requires case-level review and may lag the actual problem.

Multiple procedures linked to one arm episode count as **one injury event**. Row-level positives, event counts and injured-pitcher counts are reported separately. Surgery-only cases have unknown onset and do not create fabricated precise positives. Their possible preoperative periods censor outcomes. An official IL activation followed by an observed MLB appearance permits risk-set reentry; source return dates never enter predictors.

Targets are `ucl_injury_next_14d`, `30d`, `60d`, `90d`, and `ucl_injury_next_3_appearances` / `5_appearances`. Appearance horizons count all MLB pitching outings, including relief, with a maximum 90-day follow-up. Positives must strictly precede the reference date. Calendar negatives require later observed MLB participation through the horizon, complete required sources, and no intervening arm episode or uncertain surgery interval. Disappearance is censored rather than called a negative. This conservative observation rule itself requires selection-bias assessment.

## Features, historical availability and leakage

An appearance is a prediction point after the outing. Current velocity, release, workload and command are available then. All historical windows exclude the current date, future dates, and appearances from games with uncertain suspension/resumption timing. Affected calendar and season workload totals remain NULL rather than being presented as complete. Year-to-date workload ends at the current appearance; prior-season statistics join only to year t−1. No future roster, source seasonStats snapshot, surgery, recovery, transaction narrative or label field enters the predictor allowlist.

FF/SI/FC form the fastball family. Within-game speed decline compares disjoint first/last 20 measured fastballs and requires 40. Prior 3/5/10 histories require the full number of outings and at least `max(2,floor(N/2))` valid measurements. Personal baselines use at most 20 prior appearances and require five valid measurements. Standardized deviations require nonzero prior variance. Observed start experience is left-truncated at study start, not mislabeled as full career experience.

Automatic ball/strike events remain raw but are excluded from physical pitches. Historical thrown intentional balls count as physical pitches. Official outs divided by three give decimal innings. Native MLB IVB is kept separate from Savant `pfx_z`; physical-pitch ordinal joins handle source numbering differences and require matching counts and compatible velocities. Unvalidated alignments remain NULL. Pitch usage zero means the pitch type was absent; missing measurements never become zero measurements.

[MLB's Statcast overview](https://www.mlb.com/glossary/statcast) supports 2015 as the full-park study start. [MLB's 2020 tracking update](https://www.mlb.com/news/2020-statcast-update-includes-pose-tracking-capabilities) and [Savant's CSV definitions](https://baseballsavant.mlb.com/csv-docs) motivate sensitivity comparisons across tracking eras. Velocity definitions differ across 2015–2016 versus 2017 onward; 2020 introduced Hawk-Eye; 2026 location and strike-zone definitions require separate checks. Arm-angle availability is determined empirically by season rather than assuming coverage for the entire history.

The 65-feature [model_features_v1.yaml](config/model_features_v1.yaml) includes a rationale for every selection: 61 long-history candidates and four optional modern arm-angle features. Structural availability must still be checked before using any candidate across an era. The full master table is retained. No imputation is performed. Future preprocessing must be learned within chronological training folds.

## Reports and evaluation preparation

- [Full build summary](reports/full_build_summary.md), [season coverage](reports/season_coverage.csv), [modeling readiness](reports/modeling_readiness.md).
- [Injury population](reports/injury_population_audit.csv), [label audit](reports/injury_label_audit.csv), [event audit](reports/injury_event_audit.csv), [unmatched identities](reports/unmatched_injury_audit.csv).
- [Starter distributions](reports/starter_population_audit.csv), [starter exceptions](reports/starter_identification_exceptions.csv), [source gaps](reports/failed_source_ranges.csv), [missing games](reports/missing_statcast_games.csv).
- [Feature missingness](reports/feature_missingness.csv), [availability by season](reports/feature_availability_by_season.csv), [raw measurement coverage](reports/raw_feature_availability_by_season.csv), [history sufficiency](reports/rolling_history_audit.csv), [curated feature audit](reports/model_v1_feature_audit.csv).
- [Temporal split audit](reports/temporal_split_audit.csv), [leakage and integrity checks](reports/leakage_and_integrity_audit.csv), [data dictionary](docs/data_dictionary.csv).

`src/evaluation.py` evaluates supplied held-out scores; it never trains a model. It supports Average Precision/PR-AUC, ROC-AUC, precision, recall, F2, alert budgets of 1%, 2.5%, 5% and 10%, injury-event recall, warning lead-time distributions, and false alerts per pitcher-season. A useful primary practical measure is **event recall for alerts 7–30 days before the reference while limiting alerts to 5% of eligible appearances**. Report all-event and observable-event denominators separately. Retrospective top-score budgets do not select a deployable threshold; that threshold belongs in validation data.

Temporal split recommendations use actual event support in each candidate partition. Never randomly split appearance rows. Apply a 90-day embargo for cross-horizon comparisons, purge any injury event shared across retained partitions, and consider an additional unseen-pitcher sensitivity analysis. Baseline modeling should begin only after the event audit, source gaps and structural feature missingness are reviewed.

The curated view/export `model_ready_starting_pitcher_features_v1` includes the 65 selected candidates, identifiers and labels while preserving nullable measurements. Predictor selection remains explicit:

```python
from src.build_model_dataset import load_model_data
X, y, keys = load_model_data(
    target='ucl_injury_next_30d', starters_only=True,
    feature_set='v1', include_modern=False,
)
# X still contains NULL/NaN where measurements or histories are unavailable.
# Split keys chronologically before fitting any preprocessing or model.
```

## Profile versus departure analysis

The [profile/departure notebooks](notebooks/injury_profile_departure/README.md) compare prior pitcher profiles, current departures and their combination on the existing injury-specific temporal risk sets. They include common workload adjustment, event-warning budgets, calibration diagnostics, original matched score trajectories and training-refit stability. See the [computed findings](reports/injury_profile_departure/findings_summary.md) and [methods](docs/injury_profile_departure.md). All prior outputs and source data are preserved.

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-profiles
.venv/bin/python scripts/verify_injury_profiles.py
```

## Supervised dimensionality reduction

The [supervised-reduction notebooks](notebooks/injury_supervised_reduction/README.md) compare training-fitted PLS and PCA against original features, including separate profile, departure and combined inputs. Component counts are chosen on validation data; earlier analyses remain intact. See the [computed findings](reports/injury_supervised_reduction/SUMMARY.md) and [methods](docs/injury_supervised_reduction.md).

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-supervised
.venv/bin/python scripts/verify_injury_supervised.py
```

## Focused elbow/UCL latent visuals

The [focused notebook](notebooks/ucl_latent_visuals/README.md) adds four figures connecting the frozen PCA/PLS analysis to advance-warning questions: matched component trajectories, lead-time separation, compression curves and individual-event heterogeneity. The established broad **Elbow / UCL** endpoint is explicitly distinguished from confirmed UCL injury. See the [computed findings](reports/injury_supervised_reduction/ucl_focus/SUMMARY.md), [publication captions](reports/injury_supervised_reduction/ucl_focus/FIGURE_CAPTIONS.md), and [methods](docs/ucl_latent_visuals.md). Previous analyses, fitted models and source data remain intact.

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-visuals
.venv/bin/python scripts/verify_ucl_latent_visuals.py
```

## Feature consolidation and redundancy

The [consolidation notebooks](notebooks/feature_consolidation/README.md) inventory the full engineered space, map training-only redundancy, compare family contributions, assess sparse-selection stability, and freeze a compact warning candidate. The interrupted run's valid outputs were reused. The database and prior analyses remain unchanged. See the [computed findings](reports/feature_consolidation/summary.md), [methods](reports/feature_consolidation/methods.md), and [candidate feature specification](config/ucl_candidate_features_v2.yaml).

The selected eight-feature departure block trades some validation AP for more events warned at a controlled alert budget. It is a development candidate for the broad elbow proxy, not a confirmed-UCL model or a validated replacement for the previous 62-feature benchmark.

```bash
.venv/bin/python -m src.injury_analysis.consolidation_workflow
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite feature-consolidation
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_feature_consolidation.py
```
>>>>>>> 4ed2890 (Initial UCL modeling pipeline and tests)
