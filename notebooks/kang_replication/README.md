# Kang parent-paper replication

This separate stage attempts the paper's broad shoulder/elbow experiment with
R XGBoost, random pitcher-games, ROC-AUC, repeated selection, and GAM window
selection. It does not run the project's later UCL or chronological extensions.

1. Read [protocol and approximation ledger](../../docs/kang_replication_methods.md).
2. Run `.venv/bin/python -m src.injury_analysis.kang_replication` from the project root.
3. Open `01_parent_paper_replication.ipynb` and the
   [computed report](../../reports/kang_replication/summary.md).

The notebook reads saved results, not the database, and never launches hidden
model fits. The earlier `notebooks/kang_benchmark` foundation is preserved.

Dependencies: existing Python analysis environment; R with `xgboost`, `mgcv`,
`jsonlite`. Actual R/package versions appear in
`reports/kang_replication/software.txt`. Configure the `rscript` path if it is
not on PATH. Run with `--prepare-only` for cohort/crosswalk construction alone.

Completed CV windows, selection repetitions and importance refits are resumable.
Model runs are keyed by cohort, configuration, source code and R version.
Change `cache` and `output` in a copied config to preserve an alternative run.

Stage closeout: [completion and integrity record](../../reports/kang_replication/completion.md).
The saved notebook and all four model runs are complete; the record distinguishes
the reproduced framework from unresolved publication details.
