# UCL endpoint refinement and dynamic warning

This additive stage preserves feature consolidation, PCA/PLS and every prior analysis. Read the executed notebooks in order:

1. **01_endpoint_and_landmarks.ipynb** — evidence-specific UCL definition, timing, exclusions, competing injuries and observation opportunities.
2. **02_dynamic_models_and_warning.ipynb** — frozen representations, 7/14/21/30-day landmark risks, practical warning and validation-only protocol selection.
3. **03_trajectories_robustness_and_freeze.ipynb** — matched longitudinal risks, individual heterogeneity, uncertainty, sensitivities and future evaluation.

The numerical results are in `reports/ucl_dynamic/summary.md`; figures have PNG and PDF versions. The database is read-only. Caches are under `data/injury/analysis/ucl_dynamic/` and verify source/configuration hashes before reuse. Do not delete checkpoints to reuse incompatible inputs: create a versioned new analysis when sources change.

```bash
.venv/bin/python -m src.injury_analysis.ucl_dynamic_pipeline
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-dynamic
.venv/bin/python -m pytest -q
```

Existing numerical checkpoints are reused. The interrupted original run had completed the schema review but had saved no new dynamic models or notebooks; feature consolidation was the last completed numerical stage. `resume_audit.json` and `prior_artifacts_snapshot.json` record that boundary. An initial provisional endpoint classification was tightened to prevent procedure-only text from confirming a new injury; its superseded artifacts are preserved in `reports/ucl_dynamic/revisions/`.

2024–2025 has been examined repeatedly and is exploratory transfer, not an independent test. 2026 is incomplete and descriptive. The final candidate is a research hypothesis, not a clinical injury detector.
