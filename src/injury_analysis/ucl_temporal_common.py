"""Isolated, atomic artifacts for the first temporal signal decomposition."""
import json
from pathlib import Path
import pandas as pd
import yaml
from .common import ROOT, source_signature, sha
from .ucl_dynamic_common import write_json

CONFIG = ROOT / 'config/ucl_temporal.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
OUT, CACHE = ROOT / CFG['output_path'], ROOT / CFG['cache_path']
for p in [OUT, OUT/'tables', OUT/'figures', CACHE, CACHE/'models']:
    p.mkdir(parents=True, exist_ok=True)

def table(frame, name):
    p = OUT/'tables'/(name+'.csv')
    tmp = p.with_suffix('.csv.partial')
    frame.to_csv(tmp, index=False); tmp.replace(p)
    return frame

def read_table(name):
    return pd.read_csv(OUT/'tables'/(name+'.csv'))

def parquet(frame, name):
    p = CACHE/(name+'.parquet'); tmp = p.with_suffix('.parquet.partial')
    frame.to_parquet(tmp, index=False); tmp.replace(p)
    return frame

def figure(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT/'figures'/(name+'.'+ext), dpi=240, bbox_inches='tight')

def snapshot():
    p = OUT/'prior_artifacts_snapshot.json'
    if p.exists():
        value = json.loads(p.read_text())
        if value['source'] != source_signature():
            raise ValueError('Upstream sources changed')
        return value
    paths = []
    for directory in ['reports', 'notebooks']:
        paths.extend(p for p in (ROOT/directory).rglob('*')
                     if p.is_file() and 'ucl_temporal' not in p.parts
                     and '.ipynb_checkpoints' not in p.parts and p.name != '.DS_Store')
    paths.extend(p for p in (ROOT/'data/injury/analysis').rglob('*.joblib')
                 if CACHE not in p.parents)
    paths.extend(p for p in (ROOT/'src/injury_analysis').glob('*.py')
                 if not p.name.startswith('ucl_temporal'))
    paths += [ROOT/'config/ucl_dynamic.yaml', ROOT/'config/ucl_nonlinear.yaml',
              ROOT/'config/ucl_candidate_features_v2.yaml',
              ROOT/'reports/feature_consolidation/tables/feature_inventory.csv',
              ROOT/'data/injury/analysis/ucl_dynamic/landmarks.parquet',
              ROOT/'data/injury/analysis/ucl_dynamic/adjudicated_events.parquet']
    value = dict(source=source_signature(),
                 artifacts={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))})
    write_json(value, OUT/'prior_artifacts_snapshot.json')
    return value

def checkpoint(name, files):
    write_json(dict(source=source_signature(), config=sha(CONFIG),
                    files={str(Path(p).relative_to(ROOT)): sha(p) for p in files}),
               CACHE/(name+'_checkpoint.json'))

def valid_checkpoint(name):
    p = CACHE/(name+'_checkpoint.json')
    if not p.exists(): return False
    value = json.loads(p.read_text())
    if value['source'] != source_signature() or value['config'] != sha(CONFIG):
        raise ValueError('Changed inputs/config: '+name)
    for f, h in value['files'].items():
        if not (ROOT/f).exists() or sha(ROOT/f) != h:
            raise ValueError('Missing/modified checkpoint artifact: '+f)
    print('Verified checkpoint: '+name, flush=True)
    return True

def interval_name(lo, hi):
    return f'{lo}-{hi}'

def freeze(value, path):
    if path.exists():
        old = json.loads(path.read_text())
        if {k:v for k,v in old.items() if k != 'frozen_at_utc'} != value:
            raise ValueError('Cannot revise frozen development decisions')
        return old
    value = dict(value, frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(value, path)
    return value
