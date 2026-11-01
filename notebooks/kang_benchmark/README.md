# Kang benchmark foundation

Open `01_benchmark_foundation.ipynb` after running:

```bash
.venv/bin/python -m src.injury_analysis.kang_benchmark --fit
```

See [methods](../../docs/kang_benchmark_spec.md) for the optional dependencies,
macOS OpenMP requirement and documented local fallback. The notebook reads
saved tables and figures; it does not silently fit models or access 2026.

It presents cohort flow, both control definitions, the feature crosswalk,
missingness, bounded development curves, random holdout metrics and the
planned three focused extensions. The holdout is random games from 2017–2025,
with repeated pitchers across partitions; it is not the chronological 2026 test.

Prior notebooks and their execution runner are unchanged. To execute this
standalone synthesis from the project root:

```bash
.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient
from pathlib import Path
p = Path('notebooks/kang_benchmark/01_benchmark_foundation.ipynb')
n = nbformat.read(p, as_version=4)
NotebookClient(n, timeout=120, kernel_name='python3',
               resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(n, p)
PY
```
