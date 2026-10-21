# Focused elbow/UCL latent visuals

[Open the executed notebook](01_ucl_latent_visuals.ipynb). It reuses frozen PCA/PLS models and original matched controls to produce four figures:

1. Departure and combined PLS trajectories before recorded injury.
2. Component × outing/calendar lead-time separation heatmap.
3. UCL-proxy validation compression curves for PCA versus PLS.
4. Individual-event departure trajectories with matched controls.

The established **Elbow / UCL** endpoint includes nonspecific elbow diagnoses; it is not confirmed UCL injury only. These are exploratory results from periods already examined. Neither a group offset nor an individual positive component score is a validated early warning.

- [Computed findings and per-figure interpretation](../../reports/injury_supervised_reduction/ucl_focus/SUMMARY.md)
- [Suggested publication captions](../../reports/injury_supervised_reduction/ucl_focus/FIGURE_CAPTIONS.md)
- [PNG/PDF figures](../../reports/injury_supervised_reduction/ucl_focus/figures/)
- [CSV summaries and auditable event cells](../../reports/injury_supervised_reduction/ucl_focus/tables/)
- [Methods and limitations](../../docs/ucl_latent_visuals.md)
- [Execution manifest](../../reports/injury_supervised_reduction/ucl_focus/notebook_execution.json)
- [Verification](../../reports/injury_supervised_reduction/ucl_focus/verification.json)

From the project root:

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-visuals
.venv/bin/python scripts/verify_ucl_latent_visuals.py
```

All preprocessing, axes and classifier weights are frozen from training; prior validation selections remain fixed. New outputs live in a separate child directory of the existing supervised-reduction report/cache structure. Previous suites and the source DuckDB are preserved.
