"""Isolated artifacts and read-only inputs for full-space feature consolidation."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import yaml
from .common import ROOT,CFG as ORIGINAL_CFG,CACHE as OLD_CACHE,source_signature,sha
from .cohort import load,label_frame

CONFIG_PATH=ROOT/'config/feature_consolidation.yaml'
CFG=yaml.safe_load(CONFIG_PATH.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE,CACHE/'models']:p.mkdir(parents=True,exist_ok=True)
CATEGORY=CFG['category']


def table(frame,name):
    destination=OUT/'tables'/f'{name}.csv'
    temporary=destination.with_suffix('.csv.partial')
    frame.to_csv(temporary,index=False)
    temporary.replace(destination)
    return frame


def read_table(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')


def figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    return fig


def snapshot():
    path=OUT/'prior_artifacts_snapshot.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['source']!=source_signature():raise ValueError('Source changed since consolidation began; review before replacing the frozen analysis.')
        return old
    paths=[]
    for root in [ROOT/'reports',ROOT/'notebooks']:
        paths.extend(p for p in root.rglob('*') if p.is_file() and OUT not in p.parents
                     and 'feature_consolidation' not in p.parts and '.ipynb_checkpoints' not in p.parts and p.name!='.DS_Store')
    paths.extend((ROOT/'data/injury/analysis').rglob('*.joblib'))
    result=dict(source=source_signature(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))})
    path.write_text(json.dumps(result,indent=2))
    return result


def input_signature():
    return dict(source=source_signature(),allowlist=sha(ROOT/'data/exports/feature_columns.json'),
        dictionary=sha(ROOT/'docs/data_dictionary.csv'),cohort=sha(OLD_CACHE/'manifest.json'))


def inputs():
    """Read only appearance-level registered fields; never scan raw 18 GB pitches."""
    signature=input_signature();record=CACHE/'input_manifest.json';path=CACHE/'risk_rows.parquet'
    if path.exists() and record.exists() and json.loads(record.read_text())==signature:
        return pd.read_parquet(path),pd.read_parquet(OLD_CACHE/'events.parquet'),pd.read_parquet(OLD_CACHE/'feature_spec.parquet')
    a,events,spec=load()
    f=label_frame(a,CATEGORY,CFG['horizon_days'])
    f=f.loc[f.partition.isin(['train','validation','test'])].copy()
    registered=json.loads((ROOT/'data/exports/feature_columns.json').read_text())
    keys=f[['game_id','pitcher_id']]
    with duckdb.connect(str(ROOT/ORIGINAL_CFG['database_path']),read_only=True) as con:
        schema=con.execute('describe model_features').fetchdf();table(schema,'source_schema')
        missing=set(registered)-set(schema.column_name)
        if missing:raise ValueError(f'Registered source fields missing: {missing}')
        con.register('_cohort_keys',keys)
        projection=','.join('m."'+c+'"' for c in registered)
        data=con.execute(f'SELECT m.game_id,m.pitcher_id,{projection} FROM model_features m JOIN _cohort_keys k USING(game_id,pitcher_id)').fetchdf()
        layers={name:con.execute(f'DESCRIBE {name}').fetchdf().column_name.tolist()
                for name in ['appearances','workload_features','mechanical_features','pitcher_baselines']}
    (CACHE/'source_layers.json').write_text(json.dumps(layers,indent=2))
    f=f.drop(columns=[c for c in registered if c in f]).merge(data,on=['game_id','pitcher_id'],validate='one_to_one')
    assert not f[['game_id','pitcher_id']].duplicated().any()
    for c in ['history_max_date','baseline_max_date','workload_max_date']:
        assert (pd.to_datetime(f[c]).isna()|pd.to_datetime(f[c]).lt(f.game_date)).all(),c
    f.to_parquet(path,index=False);record.write_text(json.dumps(signature,indent=2))
    assert source_signature()==signature['source']
    table(f.groupby('partition').agg(rows=('target','size'),positive_rows=('target','sum'),pitchers=('pitcher_id','nunique'),
        prevalence=('target','mean')).reset_index(),'cohort_counts')
    return f,events,spec


def save_manifest(extra=None):
    m=dict(signature=input_signature(),config=CFG,config_sha256=sha(CONFIG_PATH),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/injury_analysis').glob('consolidation_*.py')},
        endpoint='Broad Elbow / UCL initial-anatomy outcome, not confirmed-UCL-only',
        evaluation='Previously examined 2024–2025; exploratory temporal transfer only')
    if extra:m.update(extra)
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2))
    return m
