# Initial UCL temporal signal decomposition

1. **01_temporal_family_decomposition.ipynb** — disjoint intervals, common cohort,
   family-only and ablation comparisons, grouped reliance, frozen references.
2. **02_temporal_uncertainty_and_composition.ipynb** — whole-pitcher uncertainty,
   joint timing contrasts, training refits, event influence and shared-event support.
3. **03_matched_trajectories_warning_and_summary.ipynb** — retrospective matched
   family scores, calendar versus outing alignment, interval warning and synthesis.

Read [the methods](../../docs/ucl_temporal_methods.md) and
[the result summary](../../reports/ucl_temporal/summary.md).
Figures are exported as PNG and PDF under `reports/ucl_temporal/figures/`.
All null results and full family/interval tables are retained.

```bash
.venv/bin/python -m src.injury_analysis.ucl_temporal_pipeline
.venv/bin/python scripts/create_ucl_temporal_notebooks.py
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-temporal
```

2022–2023 is development validation; 2024–2025 is previously examined exploratory
transfer. Neither retrospective matched separation nor a high single-cell PR lift
establishes a physiological mechanism or a generalizable early-warning policy.
