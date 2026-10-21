"""Isolated artifacts and exact-input checkpoints for the nonlinear extension."""
import json
import pandas as pd
import yaml
from .common import ROOT,source_signature,sha
from .ucl_dynamic_common import write_json

CONFIG=ROOT/'config/ucl_nonlinear.yaml';CFG=yaml.safe_load(CONFIG.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE,CACHE/'models']:p.mkdir(parents=True,exist_ok=True)

def table(f,name):
    p=OUT/'tables'/(name+'.csv');tmp=p.with_suffix('.csv.partial');f.to_csv(tmp,index=False);tmp.replace(p);return f
def read_table(name):return pd.read_csv(OUT/'tables'/(name+'.csv'))
def parquet(f,name):
    p=CACHE/(name+'.parquet');tmp=p.with_suffix('.parquet.partial');f.to_parquet(tmp,index=False);tmp.replace(p);return f
def figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=240,bbox_inches='tight')
def snapshot():
    p=OUT/'prior_artifacts_snapshot.json'
    if p.exists():
        x=json.loads(p.read_text())
        if x['source']!=source_signature():raise ValueError('Upstream sources changed')
        return x
    paths=[]
    for directory in ['reports','notebooks']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and OUT not in p.parents and 'ucl_nonlinear' not in p.parts and '.ipynb_checkpoints' not in p.parts and p.name!='.DS_Store')
    paths.extend(p for p in (ROOT/'data/injury/analysis').rglob('*.joblib') if CACHE not in p.parents)
    paths.extend(p for p in (ROOT/'src/injury_analysis').glob('*.py') if not p.name.startswith('ucl_nonlinear'))
    paths.extend([ROOT/'config/ucl_dynamic.yaml',ROOT/'config/ucl_candidate_features_v2.yaml'])
    x=dict(source=source_signature(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in paths})
    write_json(x,OUT/'prior_artifacts_snapshot.json');return x
def checkpoint(name,files):
    write_json(dict(source=source_signature(),config=sha(CONFIG),files={str(p.relative_to(ROOT)):sha(p) for p in files}),CACHE/(name+'_checkpoint.json'))
def valid_checkpoint(name):
    p=CACHE/(name+'_checkpoint.json')
    if not p.exists():return False
    x=json.loads(p.read_text())
    if x['source']!=source_signature() or x['config']!=sha(CONFIG):raise ValueError('Changed source/config for '+name)
    for f,h in x['files'].items():
        if not (ROOT/f).exists() or sha(ROOT/f)!=h:raise ValueError('Missing/corrupt artifact: '+f)
    return True

def inputs():
    from .ucl_dynamic_data import prepare_data,risk_frame
    a,e,s=prepare_data()
    return risk_frame(a,e,CFG['horizon']),e,{k:s[k] for k in CFG['representations']}
