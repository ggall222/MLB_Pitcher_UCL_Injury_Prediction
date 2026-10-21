"""Read refreshed master features; reuse the refined endpoint and risk rules."""
import json
import duckdb
import numpy as np
import pandas as pd
from . import cohort
from .ucl_dynamic_data import frozen_representations,departure_trajectories,risk_frame,event_cohort
from .ucl_dynamic_endpoint import run_endpoint
from .ucl_revised_common import ROOT,OUT,CACHE,CFG,table,parquet,write_json,checkpoint,valid_checkpoint,partition_dates

def prepare():
    if valid_checkpoint('data'):
        return pd.read_parquet(CACHE/'landmarks.parquet'),pd.read_parquet(CACHE/'adjudicated_events.parquet'),json.loads((CACHE/'representations.json').read_text())
    events=cohort.classify_events(pd.read_parquet(ROOT/CFG['events_path']))
    if events.observation_end_date.max()!=pd.Timestamp(CFG['observation_end']):
        raise ValueError('Canonical injury observation end disagrees with the frozen as-of date')
    specs=frozen_representations()
    required=sorted({c for s in specs.values() for c in s['features'] if not c.startswith('dyn_')})
    context=['pitcher_id','game_id','game_date','season','pitcher_name','quality_eligible','timing_uncertain',
        'valid_starting_appearance','is_starter','prior_appearance_count','history_max_date','workload_max_date',
        'baseline_max_date','game_type']
    derived={'calendar_month','prior_reported_il_count'}
    names=list(dict.fromkeys(context+[c for c in required if c not in derived]))
    with duckdb.connect(str(ROOT/CFG['database_path']),read_only=True) as con:
        schema=con.execute('DESCRIBE model_features').fetchdf();table(schema,'source_model_schema')
        if not set(names)<=set(schema.column_name):raise ValueError('Missing required registered features')
        select=','.join('m."'+n+'"' for n in names)
        a=con.execute(f'''SELECT {select} FROM model_features m JOIN pitchers p USING(pitcher_id)
            WHERE m.game_type='R' AND p.position IN ('P','TWP') AND m.game_date<=?
            ORDER BY m.pitcher_id,m.game_date,m.game_id''',[CFG['observation_end']]).fetchdf()
    for c in ['game_date','history_max_date','workload_max_date','baseline_max_date']:a[c]=pd.to_datetime(a[c])
    a=cohort.align_risk_rows(a,events,settings=CFG)
    comparison=pd.read_csv(OUT/'injury_refresh/ucl_pipeline_comparison.csv')
    # A future scheduled procedure cannot confirm a completed procedure as of the
    # cutoff. Keep such evidence in a separate audit; never fabricate its outcome.
    after=pd.to_datetime(comparison.surgery_date_upper,errors='coerce').gt(CFG['observation_end'])
    table(comparison.loc[after],'procedure_evidence_after_cutoff')
    comparison=comparison.loc[~after].copy()
    ctx=dict(cache=CACHE,out=OUT,table=table,parquet=parquet,checkpoint=checkpoint,
             valid_checkpoint=valid_checkpoint,partition_dates=partition_dates)
    events=run_endpoint(dict(events=events,appearances=a[['pitcher_id','game_date','game_id']],comparison=comparison,
        transactions=pd.read_parquet(ROOT/'data/injury/revised_2026/processed/mlb_pitcher_injury_transactions_2015_2026.parquet'),
        transaction_links=pd.read_csv(ROOT/'data/injury/revised_2026/processed/injury_event_transaction_links.csv')),ctx)
    a,names=departure_trajectories(a)
    for c in ['history_max_date','workload_max_date','baseline_max_date','dynamic_history_max_date']:
        assert (a[c].isna()|a[c].lt(a.game_date)).all(),c
    assert not a[['pitcher_id','game_id']].duplicated().any()
    parquet(a,'landmarks');write_json(specs,CACHE/'representations.json')
    # Predictive decisions consume train/validation only. Test reporting is deferred.
    development=a.loc[a.partition.isin(['train','validation'])]
    table(development.groupby(['partition','season']).agg(rows=('row_id','size'),eligible=('analysis_eligible','sum'),
        pitchers=('pitcher_id','nunique'),first_date=('game_date','min'),last_date=('game_date','max')).reset_index(),'development_coverage')
    prior=pd.read_parquet(ROOT/'data/injury/analysis/ucl_dynamic/adjudicated_events.parquet')
    comparison=prior[['injury_event_id','endpoint_primary']].merge(events[['injury_event_id','endpoint_primary']],
        on='injury_event_id',how='outer',suffixes=('_before','_after'),indicator=True)
    table(comparison,'endpoint_refresh_changes')
    checkpoint('data',[CACHE/'landmarks.parquet',CACHE/'adjudicated_events.parquet',CACHE/'representations.json',
                       OUT/'tables/development_coverage.csv',OUT/'tables/source_model_schema.csv'])
    return a,events,specs

def frame(a,events,horizon=30,competing='negative'):
    return risk_frame(a,events,horizon=horizon,competing=competing)

def event_set(events,partition):
    return event_cohort(events,partition,settings=CFG)
