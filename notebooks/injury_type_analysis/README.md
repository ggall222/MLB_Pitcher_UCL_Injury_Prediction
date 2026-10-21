# Injury-type analysis notebooks

Run notebooks **01 → 08**. They reuse the existing DuckDB features and canonical general-injury episodes. They do not ingest Statcast again, modify the source database, or replace the UCL pipeline. This is an executed exploratory comparison, including fixed baseline models, rather than a final selected model.

For paper or presentation use, read the [figure interpretation and caption guide](../../reports/injury_type_analysis/FIGURE_INTERPRETATION.md). It explains the 11 distinct figures across these eight notebooks and adds a separately computed equal-budget random-alert reference. That reference is important for interpreting event-level warning performance and is not yet drawn in the original figures.

| Notebook | Purpose |
|---|---|
| [01 — Overview](01_overview_and_categories.ipynb) | Source coverage, detailed classifications, category pooling, exclusions, missingness, time checks |
| [02 — Trajectories](02_preinjury_trajectories.ipynb) | Calendar/appearance alignment, matched controls, pitcher-relative changes, explicit throwing-side versus non-arm comparisons |
| [03 — Predictability](03_comparative_predictability.ipynb) | Fixed chronological baselines across seven categories and three horizons; AP/lift and pitcher-bootstrap uncertainty |
| [04 — Severity](04_severity_and_censoring.ipynb) | Observed IL duration, censored outcomes, long IL designations, documented surgery |
| [05 — Recurrence](05_recurrence_and_transitions.ipynb) | Repeated episodes, same-region transitions, intervals, first/repeat patterns and as-of prior-history rates |
| [06 — Warning timing](06_warning_timing_and_alerts.ipynb) | Event recall at alert budgets, 7-day minimum warning, lead-time distributions, false alerts, validation-fixed thresholds |
| [07 — Feature families](07_feature_families_and_interactions.ipynb) | Validation-only ablations, two workload interactions, modern arm-angle availability, conditional coefficients |
| [08 — Summary](08_capstone_summary.ipynb) | Computed findings, selected capstone figures and methodological limits |

## Run

From the `mlb_ucl_project` directory:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/run_injury_analysis_notebooks.py
.venv/bin/python -m pytest -q
```

The runner registers a kernel inside `.venv`, executes and saves all notebook outputs, and writes an execution manifest. VS Code can also open the notebooks using that interpreter. Re-execute in order after changing source data/configuration. To resume a failed execution after correcting it:

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --from-notebook 3
```

Notebook cell sources are maintained in `scripts/build_injury_analysis_notebooks.py`. Running that generator rewrites **these eight notebooks** and clears their outputs; execute them afterward. If editing notebooks directly, update the generator as well before using it again.

## Outputs and configuration

- Follow-on analysis: [feature-family decomposition and pre-injury timing](../injury_signal_analysis/README.md), with three additional executed notebooks and separate outputs.
- [Computed findings](../../reports/injury_type_analysis/SUMMARY.md)
- [CSV tables](../../reports/injury_type_analysis/tables/), [PNG/PDF figures](../../reports/injury_type_analysis/figures/)
- [Execution manifest](../../reports/injury_type_analysis/notebook_execution.json), [analytical source manifest](../../data/injury/analysis/manifest.json)
- [Configuration](../../config/injury_analysis.yaml), [helper modules](../../src/injury_analysis/)
- [Detailed methodology](../../docs/injury_type_analysis.md)
- `data/injury/analysis/`: narrow appearance projection, event cohort, event alignments, matched pairs and held-out predictions in Parquet. Source NULLs are preserved.

The 18 GB database is opened read-only and projected to a much smaller appearance-level table; raw pitches are not loaded into memory. Figures and tables are deterministic under the configured seed and source snapshots. Numerical comparisons are exploratory: injury-date proxies, selection of active pitchers, source revisions, censoring and era-dependent measurements constrain interpretation.
