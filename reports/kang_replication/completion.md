# Replication stage completion

Closed on 2026-11-01 (America/New_York). This is a completion record for the
existing experiment, not a new analysis or a revised protocol.

## Interruption and recovery

The last successful step before interruption was execution of the synthesis
notebook: all seven code cells completed without errors. All four R model
runs, comparison tables, figures, independent AUC checks and the full test suite
had already finished. The remaining work was to verify preservation and archive
the validation evidence. No model, window, feature-selection or bootstrap fit
was repeated during this resume.

The completed experiment includes all 43 candidate windows with five training
folds per task; 50 repetitions of five-fold feature selection; domain-wise
correlation pruning; the GAM slope-band procedure; four final XGBoost models;
500 pitcher-season confidence resamples per model; 500 training bootstrap
importance refits per model; and the prior-region-injury sensitivity.

## Verified results

The source screening retains 172,044 games and 5,943 pitcher-seasons. The latter
matches the paper's count; the former is 22 games higher. The primary Methods-
definition cohorts contain 356 elbow and 494 shoulder event-seasons with
pre-placement games, versus 430 and 545 in the publication. Elbow and shoulder
model cohorts contain 154,067 and 156,088 games, respectively; these overlapping
cohorts must not be summed as unique observations.

| Outcome / role | Replication slope band, days | Published band, days | Replication ROC-AUC | Published ROC-AUC |
|---|---:|---:|---:|---:|
| Elbow / starter | 26.88–29.39 | 27.5–29.5 | 0.925 | 0.868 |
| Elbow / reliever | 23.69–26.33 | 19.7–21.3 | 0.910 | 0.815 |
| Shoulder / starter | 12.69–32.93, disconnected | 22.8–25.7 | 0.855 | 0.815 |
| Shoulder / reliever | 27.26–34.46 | 18.6–20.3 | 0.896 | 0.797 |

Only the elbow-starter band closely reproduces the reported interval. The
shoulder-starter envelope covers four disconnected qualifying segments; its
predeclared rounded midpoint is outside the segments and does not establish a
unique optimal window. The larger AUC estimates do not establish improvement:
source cohorts, control definitions and undisclosed implementation details differ.

Release position and extension appear prominently, with velocity/spin measures
also contributing. Age has the largest mean Gain in all four replication
models. Prior-injury sensitivity improves AUC for relievers but reduces it for
starters. These findings describe this experiment, not physiological mechanisms.

## Integrity and validation

- All 56 final artifacts recorded by the four completion manifests pass their
  content hashes; current cohort, R code, config and software fingerprints match.
- All 2,372 intermediate R checkpoints are readable and have matching run
  signatures. No partial `.tmp` files remain in the four active task directories.
- All 20 existing report JSON/CSV/PNG artifacts parse or decode successfully;
  the notebook is valid and has no error outputs.
- Independent Python ROC-AUC calculations agree with the saved R results.
- The completed full test run reports **225 passed**. Its original output is
  archived in [test_results.txt](test_results.txt); it was not rerun solely for
  closeout because implementation and methodology were unchanged.
- All 1,213 pre-existing paths in the pre-stage preservation inventory retain
  their original size and modification time, including the central database
  and prior analysis artifacts. This is a metadata preservation check, distinct
  from the final-model artifact content hashes. See
  [preservation_check.json](preservation_check.json).

The earlier seed-API attempt remains archived under the regenerable
`data/injury/analysis/kang_replication/seed_api_correction` directory. The final
results use the successfully completed R `set.seed()` correction; archived
attempts are not inputs to the final tables or notebook.

## Remaining limits and recommended next stage

This completes the feasible **structural replication**, with the approximations
recorded in the [protocol](../../docs/kang_replication_methods.md). Exact
numerical replication remains limited by the paper's conflicting control wording,
author data availability, release-Y versus extension naming, outside-window
case handling, unknown candidate increment, hyperparameters/tuning procedure,
selection settings, GAM settings and conversion to a final integer window.
Feature-specific rolling-risk attribution remains unavailable because the
publication does not define it sufficiently; no substitute method was invented.

The recommended next stage is a benchmark acceptance and methods-clarification
review, especially the control definition and disconnected shoulder-starter
window. Preserve this run as the qualified reference. Seek author code or
clarification where practical before claiming exact replication. Any subsequent
UCL or chronological extension should be a separately specified, one-change
comparison; none was started during this resume.

Main outputs: [computed summary](summary.md),
[direct comparison](tables/published_comparison.csv),
[feature crosswalk](tables/feature_crosswalk.csv),
[synthesis notebook](../../notebooks/kang_replication/01_parent_paper_replication.ipynb),
and [frozen choices](../../config/kang_replication.yaml).
