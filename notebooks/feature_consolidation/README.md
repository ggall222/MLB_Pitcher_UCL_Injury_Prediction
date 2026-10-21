# Feature consolidation and redundancy

Three focused notebooks extend the existing injury analysis without changing the database or prior PCA/PLS work:

1. [Inventory and redundancy](01_inventory_and_redundancy.ipynb): schema/builders, exact identities, temporal representations, training correlations and cross-family overlap.
2. [Predictive value and stability](02_predictive_value_and_stability.ipynb): family ablation/permutation, nested representation comparisons, Elastic Net and whole-pitcher refits.
3. [Candidate set and freeze](03_candidate_set_and_freeze.ipynb): reduced candidates, conditional contributions, support sensitivity, frozen transfer and interpretation.

The inherited endpoint is **broad Elbow / UCL**, not confirmed ligament injury. Selection uses training through 2021 and validation in 2022–2023. Previously examined 2024–2025 results are exploratory temporal transfer.

The validation rule selects eight departure z-scores as a **warning-focused development candidate**, with lower AP but more validation events warned than the full 333-field model. The previous 62-feature benchmark remains important and performs better on AP in the already examined transfer period. Do not describe the compact set as a validated replacement.

- [Computed synthesis](../../reports/feature_consolidation/summary.md)
- [Exact methods and limitations](../../reports/feature_consolidation/methods.md)
- [Retained-feature evidence](../../reports/feature_consolidation/tables/retained_features.csv)
- [All concept decisions](../../reports/feature_consolidation/tables/concept_representation_decisions.csv)
- [Candidate configuration](../../config/ucl_candidate_features_v2.yaml)
- [Figure guide](../../reports/feature_consolidation/tables/figure_summary.csv)

```bash
.venv/bin/python -m src.injury_analysis.consolidation_workflow
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite feature-consolidation
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_feature_consolidation.py
```

The first command resumes only missing stages. Notebook execution reads completed artifacts and does not refit models. Six figures are exported in PNG/PDF under `reports/feature_consolidation/figures/`. Full matrices and all feature evidence remain available even when a figure displays a smaller representative subset.
