# Focused elbow/UCL PCA–PLS visuals

This extension adds four figures without refitting models or changing prior analyses. Open the [executed notebook](../notebooks/ucl_latent_visuals/01_ucl_latent_visuals.ipynb), [computed summary](../reports/injury_supervised_reduction/ucl_focus/SUMMARY.md), or [suggested publication captions](../reports/injury_supervised_reduction/ucl_focus/FIGURE_CAPTIONS.md).

## Endpoint and inputs

The existing supervised models target the broad **Elbow / UCL** category, including unspecified elbow injuries. This is not equivalent to confirmed UCL injury or Tommy John surgery. `initial_explicit_ucl` and later `supplemental_ucl_confirmed` are retained as separate audit metadata. They do not change targets, component selection, predictors, or event dates. The small initially explicit UCL subset does not support a separate matched trajectory estimate.

The extension uses original cached appearances/events, the original `matched_controls.csv` and `aligned_trajectories.parquet`, and frozen supervised-reduction joblib files. The established cohort contains starters and relievers. Matching, role alignment, event exclusions and pre-event outing keys are inherited, with no new rematching. Existing profile/departure helpers project needed baseline columns from DuckDB through a **read-only connection**.

Both case and control windows and anchors must lie in 2022–2025, outside model training. This period has already been examined. Chronological model training ends in 2021; component counts come from the prior 2022–2023 validation comparison, preserving the original 60-day embargo and risk-set definitions. These figures are exploratory, not independent confirmation.

## Scores and selection

Keep every PLS component retained by the frozen validation-selected departure and combined models. Rank axes for the primary four-panel plot by `abs(training logistic coefficient × training component SD)` within each block. Use the highest-ranked departure axis for individual trajectories. Held-out labels, plot appearance, test AP, and event shapes never select a component.

The saved model uses orthonormalized nested PLS spans followed by L2 logistic regression. Scores are those actual classifier coordinates, divided by frozen training component SD and oriented by the sign of the frozen coefficient. No transform is fitted on matched data. There is no per-event centering, refitting or outcome-directed rescaling. An oriented score is a conditional logit direction, **not** a probability or an entire model's prediction.

Loadings describe weights on training-standardized input fields. Squared weight shares summarize coordinate geometry, not causal importance or explained injury risk. Combined components can contain both profile and departure fields. The original profile baselines summarize up to 20 prior appearances; they are slowly changing histories, not immutable traits. Current departures become available after completion of the outing.

## Matched trajectories and uncertainty

Require all existing profile-plus-departure input values for a descriptive plotted appearance. Incomplete values remain unplotted; imputation does not manufacture observed trajectories. Keep the same event pairs at every point across all retained components within an alignment:

- Five appearance positions: −5, −4, −3, −2, −1, inherited from original outing ranks. Removing a missing measurement never promotes an earlier outing into a later rank.
- Calendar bins: 31–60, 15–30, 8–14, 1–7 days. Average available observations per event/group/bin, then take the matched difference. No interpolation or smoothing; each event has equal weight.

Outing and calendar panels have different balanced cohorts. Every point reports or inherits an explicit common sample size. Complete measurement/history requirements can select events and pitchers; descriptive samples differ from predictive risk sets.

Use 300 deterministic bootstrap draws. Events sharing a pitcher in **either** case or control role belong to one connected resampling cluster, accounting for recurrent cases and cross-role dependence. Retain the original equal-event point estimate. Joint max-t 95% bands span all retained components and positions within each alignment, separately for case means, control means and paired differences. Shape contrasts use descriptive percentile intervals and are not multiplicity-adjusted across all contrasts. Uncertainty conditions on existing models and matches; it does not include model training or dimension-selection variability.

## Individual-event heatmap

Display the same rows in case, control, and paired-difference panels. Sort by final case score minus the average of the first two case scores, with deterministic event-ID order for ties. This descriptive sorting necessarily shows a gradient and must not be presented as discovered injury phenotypes. No score threshold or direction of change is a validated warning rule.

Use one training-derived origin and scale for all events. Clip only display colors at ±3 SD; report clipping count, and retain exact scores in CSV/Parquet. `individual_event_order.csv` maps plotted row labels to event IDs, pitchers, dates and initial explicit-UCL metadata.

## Compression and warning context

Plot existing validation AP versus the tested component count for PCA and PLS, with the same unreduced-input model as a reference in each panel. Keep the original component selection and candidate grids. Binary context and missingness indicators bypass projection, so report **total model columns as well as latent axes**. Selection-period peaks are optimistic; a finite grid does not establish a precise optimum or statistically certain plateau.

The summary carries forward paired AP intervals and the prior 5% alert-budget / 7–30-day event-warning comparison. These metrics preserve the existing evaluable risk set, all-event denominator and retrospective top-score selection. No new event-warning method or deployment threshold is fitted here. Outing-relative separation, particularly at an outing one day before a recorded event, is not evidence of a useful seven-day warning.

No additional stability plot is produced: existing bootstrap/refit results are not for this elbow/UCL target. No horizon-specific relevance heatmap is inferred from calendar means. That requires explicit horizon-specific outcome/evaluation work. Feature interpretation is supplied as tables instead of an additional figure.

## Rerun and verification

From the project root, using the existing analysis environment:

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-visuals
.venv/bin/python scripts/verify_ucl_latent_visuals.py
.venv/bin/python -m pytest -q
```

The focused notebook recomputes only lightweight scores, summaries and plots from frozen inputs. It does not rerun the prior suites. To regenerate this notebook's source, run `scripts/build_ucl_latent_notebook.py` first; this clears only its own outputs, so execute it afterward.

- Configuration: `config/ucl_latent_visuals.yaml`. Alignment settings must match inherited matched-bin conventions; disagreement fails explicitly.
- Helpers: `src/injury_analysis/ucl_latent_visuals.py`, `ucl_latent_plots.py`, `ucl_latent_report.py`.
- Reports, four PNG/PDF pairs, CSVs, captions, execution/verification manifests: `reports/injury_supervised_reduction/ucl_focus/`.
- Derived score and event-cell Parquet: `data/injury/analysis/supervised_reduction/ucl_focus/`.

The preservation snapshot fingerprints all prior notebooks, reports, and fitted models. Verification checks those hashes, source database size/mtime and source-event/config hashes, source matches, frozen model scores/scales, temporal ordering, complete balanced support, cross-role bootstrap grouping, reproducibility, unchanged predictive curves, and executed notebook/code fingerprints. No new packages or source-data writes are required.
