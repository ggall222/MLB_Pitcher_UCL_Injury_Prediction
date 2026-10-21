"""Versioned chronological revision; old analysis caches are never overwritten."""
import json
import numpy as np
import pandas as pd
import yaml
from .common import ROOT,sha
from .ucl_dynamic_common import write_json

CONFIG=ROOT/'config/ucl_modeling.yaml';CFG=yaml.safe_load(CONFIG.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE,CACHE/'models']:p.mkdir(parents=True,exist_ok=True)

def table(f,name):
    p=OUT/'tables'/(name+'.csv');q=p.with_suffix('.csv.partial')
    f.to_csv(q,index=False);q.replace(p);return f
def read_table(name):return pd.read_csv(OUT/'tables'/(name+'.csv'))
def parquet(f,name):
    p=CACHE/(name+'.parquet');q=p.with_suffix('.parquet.partial')
    f.to_parquet(q,index=False);q.replace(p);return f
def figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=240,bbox_inches='tight')
def signature():
    p=ROOT/CFG['database_path'];s=p.stat()
    return dict(database_bytes=s.st_size,database_mtime_ns=s.st_mtime_ns,
        events_sha256=sha(ROOT/CFG['events_path']),config_sha256=sha(CONFIG))
def checkpoint(name,files):
    write_json(dict(source=signature(),files={str(p.relative_to(ROOT)):sha(p) for p in files}),CACHE/(name+'_checkpoint.json'))
def valid_checkpoint(name):
    p=CACHE/(name+'_checkpoint.json')
    if not p.exists():return False
    x=json.loads(p.read_text())
    if x['source']!=signature():raise ValueError('Changed refreshed sources/config: '+name)
    for path,d in x['files'].items():
        if not (ROOT/path).exists() or sha(ROOT/path)!=d:raise ValueError('Missing/corrupt artifact: '+path)
    print('Verified revised checkpoint: '+name,flush=True);return True
def partition_dates(dates):
    d=pd.to_datetime(dates)
    return pd.Series(np.select([d.le(CFG['train_end']),d.le(CFG['validation_end']),d.le(CFG['test_end'])],
        ['train','validation','test'],default='outside_study'),index=d.index)
def freeze(payload):
    payload=json.loads(json.dumps(payload,default=str))
    path=OUT/'frozen_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if {k:v for k,v in old.items() if k!='frozen_at_utc'}!=payload:
            raise ValueError('Frozen protocol differs; never alter it after test access')
        return old
    payload=dict(payload,frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(payload,path);return payload
