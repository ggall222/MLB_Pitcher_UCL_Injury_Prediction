# Dimensionality reduction in the existing injury analysis

Run these three notebooks in order. They reuse the original injury cohorts, chronological partitions, engineered features and matched pairs. DuckDB stays read-only; prior notebooks/results are preserved.

1. [Feature space and components](01_feature_space_and_components.ipynb): training-only PCA, variance retained, loadings, correlated predictors, between/within-pitcher variation, and sensitivity to training era, repeated pitchers and missingness.
2. [Latent profiles and pre-injury change](02_latent_profiles_and_preinjury_change.ipynb): frozen PCA maps applied to original matches, primary 2022–2025 comparisons, balanced unsmoothed trajectories, event-level shape, and role/hand/timing sensitivities.
3. [Reduced models and stability](03_reduced_models_and_stability.ipynb): common-row PCA/raw comparisons, validation-only dimension selection, full-rotation numerical control, alert-budget evaluation, whole-pitcher training refits, and computed synthesis.

From the project root:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-reduction
.venv/bin/python scripts/verify_injury_reduction.py
.venv/bin/python -m pytest -q
```

The original injury-type cache and feature-family predictions must exist. If original data/configuration changes, rerun the preceding suites first. After changing reduction configuration/code, rerun all three reduction notebooks so transforms and downstream results remain consistent. To resume an execution failure without changing its inputs:

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-reduction --from-notebook 2
```

Notebook sources are maintained in `scripts/build_injury_reduction_notebooks.py`. Running that generator clears only this suite's notebook outputs; execute afterward. No new dependencies beyond the existing analysis requirements are needed.

- [Computed results](../../reports/injury_dimensionality_reduction/SUMMARY.md)
- [Tables](../../reports/injury_dimensionality_reduction/tables/) and [PNG/PDF figures](../../reports/injury_dimensionality_reduction/figures/)
- [Methods and interpretation](../../docs/injury_dimensionality_reduction.md)
- [Configuration](../../config/injury_reduction.yaml)
- [Execution manifest](../../reports/injury_dimensionality_reduction/notebook_execution.json)
- [Artifact verification](../../reports/injury_dimensionality_reduction/verification.json)
- Fitted transforms and narrow derived Parquet: `data/injury/analysis/dimensionality_reduction/`.

The previously examined 2024–2025 test period remains exploratory. Components represent combinations of measurements, not established physiological mechanisms. The number of principal components is smaller than the actual predictive design, which also retains binary context and missingness flags.
