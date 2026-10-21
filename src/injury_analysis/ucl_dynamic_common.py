"""Isolated, restartable artifacts; frozen upstream sources are always read-only."""
import json
from pathlib import Path
import pandas as pd
import yaml
from .common import ROOT,source_signature,sha

CONFIG=ROOT/'config/ucl_dynamic.yaml'
CFG=yaml.safe_load(CONFIG.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE,CACHE/'models']:p.mkdir(parents=True,exist_ok=True)

def table(frame,name):
    p=OUT/'tables'/(name+'.csv');q=p.with_suffix('.csv.partial')
    frame.to_csv(q,index=False);q.replace(p);return frame

def read_table(name):return pd.read_csv(OUT/'tables'/(name+'.csv'))

def parquet(frame,name):
    p=CACHE/(name+'.parquet');q=p.with_suffix('.parquet.partial')
    frame.to_parquet(q,index=False);q.replace(p);return frame

def write_json(value,path):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,default=str));temp.replace(path)

def snapshot():
    p=OUT/'prior_artifacts_snapshot.json'
    if p.exists():
        result=json.loads(p.read_text())
        if result['source']!=source_signature():raise ValueError('Upstream source changed; do not mix with this frozen development stage')
        return result
    paths=[]
    for directory in ['reports','notebooks']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and OUT not in p.parents and 'ucl_dynamic' not in p.parts and '.ipynb_checkpoints' not in p.parts and p.name!='.DS_Store')
    paths.extend(p for p in (ROOT/'data/injury/analysis').rglob('*.joblib') if CACHE not in p.parents)
    for p in [ROOT/'config/ucl_candidate_features_v2.yaml',ROOT/'data/injury/analysis/feature_consolidation/candidate_specifications.json',ROOT/'data/injury/processed/injury_event_transaction_links.csv']:
        paths.append(p)
    result=dict(source=source_signature(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))})
    write_json(result,OUT/'prior_artifacts_snapshot.json')
    write_json(dict(completed_new_stage_artifacts=0,last_completed_checkpoint='Feature consolidation complete; new-stage schema/evidence feasibility review only',
        incomplete_or_corrupt_new_stage_artifacts=[],action='Continue from reviewed sources; do not rerun feature selection'),OUT/'resume_audit.json')
    return result

def figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=240,bbox_inches='tight')

def partition_dates(dates):
    import numpy as np
    s=pd.to_datetime(dates)
    return pd.Series(np.select([s.le(CFG['train_end']),s.le(CFG['validation_end']),s.le(CFG['test_end'])],
        ['train','validation','test'],default='descriptive_2026'),index=s.index)

def checkpoint(name,files):
    write_json(dict(source=source_signature(),config_sha256=sha(CONFIG),
        files={str(Path(p).relative_to(ROOT)):sha(p) for p in files}),CACHE/(name+'_checkpoint.json'))

def valid_checkpoint(name):
    p=CACHE/(name+'_checkpoint.json')
    if not p.exists():return False
    x=json.loads(p.read_text())
    if x['source']!=source_signature() or x['config_sha256']!=sha(CONFIG):raise ValueError('Changed inputs/config for checkpoint '+name)
    for f,d in x['files'].items():
        if not (ROOT/f).exists() or sha(ROOT/f)!=d:raise ValueError('Missing or modified stage artifact '+f)
    return True
