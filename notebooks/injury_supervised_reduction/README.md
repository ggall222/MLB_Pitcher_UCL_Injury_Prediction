# Supervised dimensionality reduction

Three executed notebooks extend the PCA and profile/departure studies. They use the same cohort, labels, matched controls, temporal splits and evaluation utilities. The source DuckDB database is read-only.

1. [Compression and prediction](01_supervised_compression_and_prediction.ipynb): training-only PLS/PCA, common component grids, validation-only selection, total design dimensions, original-feature comparison and practical warnings.
2. [Components and injury patterns](02_components_and_injury_patterns.ipynb): feature/family weights, overlap with earlier PCA, persistent versus within-pitcher variation and matched pre-injury trajectories.
3. [Stability and sensitivity](03_stability_sensitivity_and_summary.ipynb): paired training-pitcher refits, shoulder availability/timing checks and computed synthesis.

From the project root:

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-supervised
.venv/bin/python scripts/verify_injury_supervised.py
.venv/bin/python -m pytest -q
```

The notebooks verify source/configuration/code/output fingerprints before reusing completed computations. Set `FORCE_RECOMPUTE=True` to refit. Run them in order after configuration changes. The generator is `scripts/build_injury_supervised_notebooks.py`; generating again clears this suite's notebook outputs, so execute afterward.

- [Findings and research interpretation](../../reports/injury_supervised_reduction/SUMMARY.md)
- [Methods](../../docs/injury_supervised_reduction.md)
- [Tables](../../reports/injury_supervised_reduction/tables/) and [PNG/PDF figures](../../reports/injury_supervised_reduction/figures/)
- [Configuration](../../config/injury_supervised_reduction.yaml)
- [Execution manifest](../../reports/injury_supervised_reduction/notebook_execution.json)
- [Artifact verification](../../reports/injury_supervised_reduction/verification.json)
- Fitted transforms/models and predictions: `data/injury/analysis/supervised_reduction/`.

Use the existing `requirements-analysis.txt`; no new dependencies are needed. Original injury-type caches and the previous PCA/profile results must exist. If source data changes, rebuild the preceding cohorts/results before this experiment; frozen comparisons must share the same source.

PLS learns directions from training labels, followed by the existing logistic classifier. It does not replace PCA. Orthonormalizing the nested PLS spans preserves their information while making L2 geometry comparable. Binary context and missingness indicators pass through, so latent dimensions must be reported alongside total model columns.

The already examined 2024–2025 test period remains exploratory. Compact outcome-directed prediction does not establish a compact physiological injury mechanism.

