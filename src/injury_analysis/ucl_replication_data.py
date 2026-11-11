"""Freeze original fits and development cutoffs, then assemble refreshed 2026 landmarks."""
import json
import duckdb
import numpy as np
import pandas as pd
from .cohort import classify_events,align_risk_rows
from .ucl_dynamic_endpoint import run_endpoint
from .ucl_dynamic_data import risk_frame
from .ucl_temporal_models import load
from .ucl_temporal_data import interval_target
from .ucl_replication_common import ROOT,OUT,CACHE,OLD,OLD_CACHE,DB,EVENTS,CFG,CONFIG,sha,write_json,signature,table,parquet,checkpoint,valid,locked

def freeze_replication():
    if (OUT/'replication_freeze.json').exists():return locked()
    old=json.loads((OLD/'development_freeze.json').read_text())
    assert old['candidate_families']==CFG['candidate_families']
    assert old['config']['intervals']==CFG['intervals'] and old['config']['logistic_C']==1.
    paths=list(OLD.rglob('*'))+list(OLD_CACHE.rglob('*'))+list((ROOT/'notebooks/ucl_temporal').rglob('*'))
    paths=[p for p in paths if p.is_file() and p.name!='.DS_Store']
    thresholds={};edges={}
    for lo,hi in CFG['intervals']:
        interval=f'{lo}-{hi}';pred=pd.read_parquet(OLD_CACHE/f'predictions_validation_{interval}.parquet')
        assert pred.game_date.between('2022-01-01','2023-12-31').all()
        for name in old['models']:
            obj=load(interval,name)
            assert obj['signature']==old['fit_signatures'][interval+'|'+name]
            assert pd.Timestamp(obj['training_max_date'])<=pd.Timestamp('2021-12-31')
            assert obj['model'].C==1 and obj['model'].penalty=='l2'
            key=interval+'|'+name
            thresholds[key]={str(rate):float(pred[name].quantile(1-rate)) for rate in CFG['alert_rates']}
            edges[key]=np.unique(pred[name].quantile(np.arange(1,CFG['reliability_bins'])/CFG['reliability_bins']).to_numpy()).tolist()
    value=dict(source=signature(),config=CFG,original_development_freeze_sha=sha(OLD/'development_freeze.json'),
        original_development_frozen_at=old['frozen_at_utc'],models=old['models'],families=old['families'],
        candidate_families=old['candidate_families'],fit_signatures=old['fit_signatures'],thresholds=thresholds,reliability_edges=edges,
        prior_artifacts={str(p.relative_to(ROOT)):sha(p) for p in paths},frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        interpretation='Locked retrospective chronological replication sensitivity; 2026 predates final freeze and was present elsewhere; not prospective or fully independent',
        episode_rule='Consecutive above-cutoff eligible outings per pitcher-season; break at non-alert, ineligible outing, or >30-day gap',
        calibration_rule='Diagnostic intercept/slope only with >=10 events and >=20 positive rows; never applied to predictions',
        temporal_contrasts='Original 33 family ablation-lift near-minus-31–60 contrasts, joint max-t bands; original five reference lift contrasts',
        decision_gate='Temporal heterogeneity requires consistent nominated-family contrasts, multiplicity support, shared-event robustness and limited single-event dependence; operational value assessed separately',
        model_fitting_allowed=False)
    write_json(value,OUT/'replication_freeze.json');return value

def prepare():
    spec=locked()
    if valid('data'):return pd.read_parquet(CACHE/'risk_60.parquet'),pd.read_parquet(CACHE/'adjudicated_events.parquet'),spec
    e=classify_events(pd.read_parquet(EVENTS));assert e.observation_end_date.max()==pd.Timestamp(CFG['observation_end'])
    context=['pitcher_id','game_id','game_date','season','pitcher_name','quality_eligible','timing_uncertain',
        'valid_starting_appearance','is_starter','prior_appearance_count','history_max_date','workload_max_date','baseline_max_date','game_type']
    features=sorted(set(spec['models']['Full union'])-{'calendar_month','prior_reported_il_count'})
    names=list(dict.fromkeys(context+features))
    with duckdb.connect(str(DB),read_only=True) as con:
        schema=con.sql('DESCRIBE model_features').df();table(schema,'source_schema')
        assert set(names)<=set(schema.column_name)
        cols=','.join('m."'+c+'"' for c in names)
        a=con.execute(f"SELECT {cols} FROM model_features m JOIN pitchers p USING(pitcher_id) WHERE m.game_type='R' AND p.position IN ('P','TWP') AND m.game_date<=? ORDER BY pitcher_id,game_date,game_id",[CFG['observation_end']]).df()
    for c in ['game_date','history_max_date','workload_max_date','baseline_max_date']:a[c]=pd.to_datetime(a[c])
    a=align_risk_rows(a,e) # Original chronological/eligibility rules; no redefinition.
    comparison=pd.read_csv(ROOT/'reports/ucl_revised_2026/injury_refresh/ucl_pipeline_comparison.csv')
    after=pd.to_datetime(comparison.surgery_date_upper,errors='coerce').gt(CFG['observation_end'])
    table(comparison.loc[after],'future_procedure_evidence_excluded')
    context=dict(cache=CACHE,out=OUT,table=table,parquet=parquet,checkpoint=checkpoint,valid_checkpoint=valid,
        partition_dates=lambda d:pd.Series(np.where(pd.to_datetime(d).dt.year.eq(2026),'replication','historical'),index=d.index))
    e=run_endpoint(dict(events=e,appearances=a[['pitcher_id','game_date','game_id']],comparison=comparison.loc[~after],
        transactions=pd.read_parquet(EVENTS.parent/'mlb_pitcher_injury_transactions_2015_2026.parquet'),
        transaction_links=pd.read_csv(EVENTS.parent/'injury_event_transaction_links.csv')),context)
    f=risk_frame(a,e,horizon=60);f=f.loc[f.game_date.dt.year.eq(2026)].copy();f['partition']='replication'
    parquet(f,'risk_60');ev=event_set(e);rows=[]
    for lo,hi in CFG['intervals']:
        g=interval_target(f,lo,hi);known=g.loc[g.target.notna()];pos=known.loc[known.target.eq(1)]
        rows.append(dict(interval=f'{lo}-{hi}',appearance_rows=len(f),eligible_landmarks=int(f.analysis_eligible.sum()),rows=len(known),
            unknown_rows=int((f.analysis_eligible&f.target.isna()).sum()),pitchers=known.pitcher_id.nunique(),positive_rows=len(pos),
            prevalence=float(known.target.mean()),represented_events=pos.dynamic_next_event_id.nunique(),qualifying_events=len(ev),
            observable_events=pos.loc[pos.dynamic_next_event_id.isin(ev.injury_event_id),'dynamic_next_event_id'].nunique(),
            latest_appearance=str(f.game_date.max().date()),latest_observed_eventfree=str(f.loc[f.label_status.eq('observed_event_free_horizon'),'game_date'].max().date())))
    table(pd.DataFrame(rows),'cohort_support')
    table(f.loc[f.analysis_eligible].assign(month=lambda z:z.game_date.dt.strftime('%Y-%m')).groupby(['month','label_status']).size().rename('rows').reset_index(),'censoring')
    coverage=pd.DataFrame([dict(feature=c,missing_fraction=float(f.loc[f.analysis_eligible,c].isna().mean())) for c in features]);table(coverage,'feature_coverage')
    checkpoint('data',[CACHE/'risk_60.parquet',CACHE/'adjudicated_events.parquet',OUT/'tables/cohort_support.csv',OUT/'tables/censoring.csv',OUT/'tables/feature_coverage.csv'])
    return f,e,spec

def event_set(e):
    ev=e.loc[e.endpoint_primary&e.prediction_reference_date.dt.year.eq(2026)].copy()
    ev['pitcher_id']=ev.player_id;ev['label_reference_date']=ev.prediction_reference_date
    return ev

def score_replication():
    if valid('scores'):return
    f,e,spec=prepare()
    access=OUT/'replication_access.json'
    if not access.exists():write_json(dict(first_scored_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),freeze_sha256=sha(OUT/'replication_freeze.json')),access)
    files=[access]
    from .ucl_temporal_models import scores
    for lo,hi in CFG['intervals']:
        interval=f'{lo}-{hi}';g=interval_target(f,lo,hi);eligible=g.loc[g.analysis_eligible]
        pred=eligible[['row_id','pitcher_id','game_id','game_date','target']].copy()
        for name in spec['models']:
            obj=load(interval,name)
            assert obj['signature']==spec['fit_signatures'][interval+'|'+name]
            pred[name]=scores(obj,eligible)
        parquet(pred,'predictions_'+interval);files.append(CACHE/f'predictions_{interval}.parquet')
    checkpoint('scores',files)
