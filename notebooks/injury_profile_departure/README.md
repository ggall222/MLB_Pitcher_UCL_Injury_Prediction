# Profile versus departure prediction

Two executed notebooks extend the prior PCA analysis without changing cohorts, matching, database data or earlier results.

1. [Predictive increment](01_predictive_increment.ipynb): common-row profile/departure/combined models, workload-adjusted contrasts, AP, 7–30-day event warnings, alert concentration and probability quality.
2. [Score timing and robustness](02_score_timing_and_robustness.ipynb): exact model-score decomposition, original matched trajectories, missingness/timing/history checks and paired training refits.

From the project root:

```bash
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-profiles
.venv/bin/python scripts/verify_injury_profiles.py
.venv/bin/python -m pytest -q
```

The notebooks verify source, configuration, code and artifact fingerprints before reusing a completed stage. Set `FORCE_RECOMPUTE=True` to refit; run both notebooks in order after changing model definitions. The generator is `scripts/build_injury_profile_notebooks.py`; regeneration clears only these notebook outputs.

- [Findings](../../reports/injury_profile_departure/findings_summary.md)
- [Figures, PNG/PDF](../../reports/injury_profile_departure/figures/)
- [All tables](../../reports/injury_profile_departure/tables/)
- [Feature definitions](../../reports/injury_profile_departure/tables/feature_definitions.csv)
- [Configuration](../../config/injury_profile_departure.yaml)
- [Verification](../../reports/injury_profile_departure/verification.json)
- Derived fitted models and predictions: `data/injury/analysis/profile_departure/`.

P contains eight prior baseline means and five slow context fields; D contains eight corresponding current z-scores. C holds workload/role/calendar common. A profile is a recent rolling summary, not an immutable trait. Current departures are observable after the outing. The seven injury-specific risk sets and train≤2021 / validation2022–23 / test2024–25 splits are unchanged; test results remain exploratory.

No new dependency is required. Shared resampling uses the previous feature-family analysis's seed and 300-replicate scheme. Model imputation exists only within training-fitted analytical pipelines; source NULLs are unchanged. Logistic models are linear and cannot establish that every possible nonlinear departure signal has been exhausted.

