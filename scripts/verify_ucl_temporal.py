"""Verify completed temporal outputs and write a reproducibility manifest."""
from pathlib import Path
import json
import re
import sys
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.injury_analysis.common import source_signature,sha
from src.injury_analysis.ucl_temporal_common import OUT,CACHE,write_json,snapshot
from src.injury_analysis.ucl_temporal_reporting import validate

def main():
    checks=validate()
    tests=(OUT/'pytest.log').read_text()
    result=re.search(r'(\d+) passed',tests)
    assert result and not re.search(r'\d+ (failed|error)',tests),tests
    execution=json.loads((OUT/'notebook_execution.json').read_text())
    assert len(execution['notebooks'])==3
    cells=0
    for record in execution['notebooks']:
        path=ROOT/'notebooks/ucl_temporal'/record['name']
        assert record['success'] and record['errors']==0
        assert record['sha256']==sha(path)
        notebook=nbformat.read(path,as_version=4)
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        cells+=len(code)
    for path in (OUT/'tables').glob('*.csv'):
        pd.read_csv(path)  # parse every table, not just its header
    for path in CACHE.glob('*.parquet'):
        pq.read_metadata(path)
    for path in (OUT/'figures').glob('*.png'):
        with Image.open(path) as image:image.verify()
    for path in (OUT/'figures').glob('*.pdf'):
        data=path.read_bytes()
        assert data.startswith(b'%PDF-') and b'%%EOF' in data[-50:]
    resume=(OUT/'resume_verification.log').read_text()
    assert 'Fit ' not in resume and 'Verified checkpoint: reporting' in resume
    paths=list((OUT/'tables').glob('*.csv'))+list((OUT/'figures').glob('*'))
    paths+=list((ROOT/'notebooks/ucl_temporal').glob('*'))
    paths+=list((ROOT/'src/injury_analysis').glob('ucl_temporal_*.py'))
    paths += [ROOT/'config/ucl_temporal.yaml',ROOT/'docs/ucl_temporal_methods.md',
              ROOT/'scripts/create_ucl_temporal_notebooks.py',Path(__file__),
              ROOT/'tests/test_ucl_temporal.py',OUT/'summary.md',OUT/'development_freeze.json',
              OUT/'predeclared_analysis.json',OUT/'candidate_families.json',OUT/'notebook_execution.json']
    paths+=list(CACHE.glob('*.json'))+list(CACHE.glob('*.parquet'))
    write_json(dict(completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),source=source_signature(),
        notebooks=3,executed_code_cells=cells,figures=len(list((OUT/'figures').glob('*.png'))),
        csv_tables=len(list((OUT/'tables').glob('*.csv'))),tests_passed=int(result.group(1)),
        validation_checks_passed=len(checks),prior_artifacts_verified=len(snapshot()['artifacts']),
        primary_interval_models=112,trajectory_models=11,followup_sensitivity_models=12,
        training_refit_models=168,total_saved_models=len(list((CACHE/'models').glob('*.joblib'))),
        resume_verified_without_refitting=True,
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.is_file()}),
        OUT/'completion_manifest.json')
    print(json.dumps({k:v for k,v in json.loads((OUT/'completion_manifest.json').read_text()).items()
                      if k not in ['artifacts','source']},indent=2))

if __name__=='__main__':
    main()
