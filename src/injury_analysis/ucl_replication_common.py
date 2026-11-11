"""Isolated locked replication artifacts; original temporal files are read-only."""
import json
import pandas as pd
import yaml
from .common import ROOT,sha
from .ucl_dynamic_common import write_json
CONFIG=ROOT/'config/ucl_temporal_replication.yaml';CFG=yaml.safe_load(CONFIG.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path'];OLD=ROOT/'reports/ucl_temporal';OLD_CACHE=ROOT/'data/injury/analysis/ucl_temporal'
DB=ROOT/'database/mlb_ucl.duckdb';EVENTS=ROOT/'data/injury/revised_2026/processed/mlb_pitcher_injury_events_2015_2026.parquet'
for p in [OUT,OUT/'tables',OUT/'figures',CACHE]:p.mkdir(parents=True,exist_ok=True)
def table(f,name):
    p=OUT/'tables'/(name+'.csv');tmp=p.with_suffix('.csv.partial');f.to_csv(tmp,index=False);tmp.replace(p);return f
def read(name):return pd.read_csv(OUT/'tables'/(name+'.csv'))
def parquet(f,name):
    p=CACHE/(name+'.parquet');tmp=p.with_suffix('.parquet.partial');f.to_parquet(tmp,index=False);tmp.replace(p);return f
def figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=220,bbox_inches='tight')
def signature():
    s=DB.stat();return dict(db_size=s.st_size,db_mtime_ns=s.st_mtime_ns,events_sha=sha(EVENTS),config_sha=sha(CONFIG))
def checkpoint(name,paths):write_json(dict(source=signature(),files={str(p.relative_to(ROOT)):sha(p) for p in paths}),CACHE/(name+'_checkpoint.json'))
def valid(name):
    p=CACHE/(name+'_checkpoint.json')
    if not p.exists():return False
    v=json.loads(p.read_text())
    if v['source']!=signature():raise ValueError('Replication source/config changed: '+name)
    for f,h in v['files'].items():
        if not (ROOT/f).exists() or sha(ROOT/f)!=h:raise ValueError('Missing/corrupt replication artifact: '+f)
    print('Verified replication checkpoint: '+name,flush=True);return True

def locked():
    p=OUT/'replication_freeze.json'
    if not p.exists():raise ValueError('Freeze operational policy before 2026 scoring')
    obj=json.loads(p.read_text())
    if obj['source']!=signature():raise ValueError('Source changed since replication freeze')
    for f,h in obj['prior_artifacts'].items():
        if sha(ROOT/f)!=h:raise ValueError('Original temporal artifact changed: '+f)
    return obj
