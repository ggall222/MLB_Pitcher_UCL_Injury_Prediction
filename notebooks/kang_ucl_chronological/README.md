# Extension 2: chronological evaluation

Training uses 2017–2023. The same models evaluate 2024–2025 and 2026. Only the
outer evaluation chronology changes; all fitted choices are recomputed from
training using the frozen parent methods.

- [Protocol and limits](../../docs/kang_ucl_chronological_methods.md)
- [Configuration](../../config/kang_ucl_chronological.yaml)
- [Results](../../reports/kang_ucl_chronological/summary.md)
- `01_temporal_evaluation.ipynb`: saved-result synthesis; no fitting

Run from the project root with the existing Python and R environments:

```bash
.venv/bin/python -m src.injury_analysis.kang_ucl_chronological --train-only
.venv/bin/python -m src.injury_analysis.kang_ucl_chronological
.venv/bin/python -m pytest tests/test_kang_ucl_chronological.py -q
```

Completed training and scores are fingerprinted and resumable. The protocol
freeze protects both prior stages and source files. The development freeze
precedes later-period access and protects learned windows, selected features
and models. Never delete the freeze to silently accept a source refresh.

The 2026 source observation cutoff is verified from existing authoritative data,
not inferred from today's date. Incomplete negative horizons remain censored.
This is a retrospective chronological experiment, not prospective validation.
