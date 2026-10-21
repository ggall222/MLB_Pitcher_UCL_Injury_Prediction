# Feature-family decomposition and trajectory shape

Three executed notebooks extend the existing injury-type workflow. The original cohorts, labels, matched pairs and historical features are reused; the central DuckDB remains read-only. Both starters and relievers are included. No final model is selected.

1. [Feature-family decomposition](01_feature_family_decomposition.ipynb): eight individual families, ten combinations/full specifications, paired AP uncertainty, event warning versus random alerts, variable inspection, and a separate modern arm-angle comparison.
2. [Pre-injury trajectories](02_preinjury_trajectories.ipynb): existing matched pairs, balanced calendar/appearance support, case/control and paired-difference plots, simultaneous time bands, event-level shape contrasts and heterogeneity.
3. [Timing and sensitivity](03_trajectory_timing_and_sensitivity.ipynb): supported timing versus apparent onset, final-week exclusion, strict QA, baseline strength, role/prior-injury/period comparisons, restricted predictive refits, and the computed synthesis.

Run from `mlb_ucl_project`:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-signals
.venv/bin/python -m pytest -q

# Resume at a particular notebook after resolving a failure:
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-signals --from-notebook 2
```

The earlier analytical cache and matched controls must exist. If its source/config changes, first rerun the original injury-type workflow; then execute this suite in order. Notebook sources are maintained in `scripts/build_injury_signal_notebooks.py`; regeneration clears this suite's notebook outputs, so execute afterward. The shared runner preserves successful/error outputs and writes an execution manifest.

Outputs:

- Follow-on analysis: [dimensionality reduction, profiles and departures](../injury_dimensionality_reduction/README.md).
- [Computed synthesis](../../reports/injury_signal_analysis/SUMMARY.md)
- [CSV tables](../../reports/injury_signal_analysis/tables/) and [PNG/PDF figures](../../reports/injury_signal_analysis/figures/)
- [Methods and interpretation](../../docs/injury_signal_analysis.md)
- [Configuration](../../config/injury_signals.yaml)
- [Execution manifest](../../reports/injury_signal_analysis/notebook_execution.json)
- Separate analysis Parquet: `data/injury/analysis/signal_analysis/`.

The primary temporal split remains training through 2021, validation 2022–2023, testing 2024–2025. Because this test period was previously examined, these are **exploratory comparisons**, not a fresh confirmatory evaluation. Trajectories also include descriptive 2026 observations, with a through-2025 sensitivity.
