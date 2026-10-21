# Nonlinear UCL model evaluation

Read the executed notebooks in order:

1. `01_model_form_and_validation.ipynb` — fixed cohort and features; linear/GAM/boosted-tree comparisons; validation-only selection; training fit and calibration.
2. `02_nonlinear_effects_and_stability.ipynb` — ALE, local interaction contrasts, refits, feature importance and influential-event diagnostics.
3. `03_warning_transfer_and_conclusions.ipynb` — ≥7/14/21-day warnings, alert burden, exploratory temporal transfer and what to freeze.

The stage reuses the complete refined UCL endpoint and existing logistic fits. It does not rerun feature selection, change injury definitions or modify the DuckDB database. Seven feature representations, two spline penalties and three tree sizes form a fixed 35-candidate nonlinear search.

Key result: large profile-spline AP gains are concentrated in very few pitchers, while nonlinear departure models provide only small, uncertain warning gains. No reliable prospective detector has been established.

```bash
.venv/bin/python -m src.injury_analysis.ucl_nonlinear_pipeline
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-nonlinear
.venv/bin/python -m pytest -q
```

Checkpoints reuse completed outputs and verify input/configuration hashes. Do not bypass mismatched-input checks; create a versioned stage for changed sources. Analysis caches and model artifacts are under `data/injury/analysis/ucl_nonlinear/`. Six figures, tables, selection freeze, summary and validation records are under `reports/ucl_nonlinear/`. Detailed methodology and primary references are in `docs/ucl_nonlinear_methods.md`.

The already examined 2024–2025 period is exploratory transfer. Future evaluation requires a genuinely post-freeze cohort with mature follow-up and contemporaneous archived injury evidence. ALE, permutation and interaction diagnostics describe models; they do not establish physiological thresholds, tissue damage or causality.
