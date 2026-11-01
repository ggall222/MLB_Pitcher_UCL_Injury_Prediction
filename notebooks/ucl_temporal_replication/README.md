# Locked temporal replication

Open [01_locked_replication.ipynb](01_locked_replication.ipynb) for the executed synthesis of cohort support, nominated-family replication, shared-event results, event influence, alert episodes, calibration and the workload diagnostic.

The original `ucl_temporal` freeze is unchanged. 2026 is a locked retrospective replication sensitivity, not prospective confirmation. No models are fitted here; no holdout thresholds are optimized.

- [Computed summary](../../reports/ucl_temporal_replication/summary.md)
- [Methods and rerun commands](../../docs/ucl_temporal_replication_methods.md)
- [Tables](../../reports/ucl_temporal_replication/tables)
- [Figures, PNG and PDF](../../reports/ucl_temporal_replication/figures)

Run `.venv/bin/python -m src.injury_analysis.ucl_replication_pipeline` from the project root, then `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-replication`. Existing valid checkpoints are verified and reused.
