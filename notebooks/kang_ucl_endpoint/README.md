# Extension 1 — endpoint specificity

The separate UCL experiment reuses the frozen Kang replication's R model and
experimental settings. It changes the documented endpoint and resulting
case membership/reference dates. No chronological extension is included.

- [Protocol and endpoint provenance](../../docs/kang_ucl_endpoint_methods.md)
- [Inherited configuration](../../config/kang_ucl_endpoint.yaml)
- [Computed results](../../reports/kang_ucl_endpoint/summary.md)
- `01_ucl_endpoint_specificity.ipynb`: executed synthesis, no hidden fitting

Run from the project root:

```bash
.venv/bin/python -m src.injury_analysis.kang_ucl_endpoint --prepare-only
.venv/bin/python -m src.injury_analysis.kang_ucl_endpoint
.venv/bin/python -m pytest tests/test_kang_ucl_endpoint.py tests/test_kang_replication.py -q
```

The existing Python analysis environment and identical benchmark R environment
are required. Checkpoints resume without refitting completed tasks. The source
adjudication and benchmark files must match the pre-fit hashes in
`reports/kang_ucl_endpoint/experiment_freeze.json`. Do not delete that record to
silently accept changed inputs. Regenerable models, rows and bootstrap checkpoints
are under `data/injury/analysis/kang_ucl_endpoint`; public tables and figures are
under `reports/kang_ucl_endpoint`.
