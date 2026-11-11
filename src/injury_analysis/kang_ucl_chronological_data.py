"""Chronological adapter of frozen Extension 1; no learned future quantities."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from .kang_spec import ROOT, EXTRA_SQL, DOMAINS, atomic_json, digest
from .kang_replication_data import PREDICTORS, season_screen
from .kang_ucl_endpoint_data import settings as extension1_settings, refine_cohorts, check_hashes

CONFIG=ROOT/'config/kang_ucl_chronological.yaml'
GRAIN=['pitcher_id','game_id']
EXPORT=['game_id','pitcher_id','season','game_date','injury_event_id','lead_days','case_season']+PREDICTORS


def settings():
    spec=yaml.safe_load(CONFIG.read_text())
    endpoint,base,ext1=extension1_settings(ROOT/spec['extension1_config'])
    assert spec['training_start']=='2017-01-01' and spec['training_end']=='2023-12-31'
    assert spec['validation_start']=='2024-01-01' and spec['validation_end']=='2025-12-31'
    assert spec['test_start']=='2026-01-01' and spec['test_end']=='2026-12-31'
    assert spec['negative_followup']=='calendar_horizon_within_source_cutoff'
    cfg=dict(base,output=spec['output'],cache=spec['cache'])
    assert cfg['output']=='reports/kang_ucl_chronological'
    assert cfg['cache']=='data/injury/analysis/kang_ucl_chronological'
    return spec,endpoint,base,ext1,cfg


def period_for(dates,spec):
    d=pd.to_datetime(dates)
    return pd.Series(np.select([d.between(spec[p+'_start'],spec[p+'_end']) for p in ['training','validation','test']],
                              ['training','validation','test'],default='excluded'),index=d.index)


def observed_labels(frame,window,cutoff):
    """Calendar administrative follow-up only; no new active-player risk set."""
    positive=frame.case_season & frame.lead_days.between(1,window)
    date=pd.to_datetime(frame.game_date)
    known_positive=positive & (date+pd.to_timedelta(frame.lead_days,unit='D')).le(pd.Timestamp(cutoff))
    observed=known_positive | (date+pd.to_timedelta(window,unit='D')).le(pd.Timestamp(cutoff))
    return positive.astype(float).where(observed)


def assert_separation(frames):
    """Pitchers can recur; games, pitcher-seasons and positive events cannot."""
    for i,(name,f) in enumerate(frames.items()):
        assert not f.duplicated(GRAIN).any()
        for other,g in list(frames.items())[i+1:]:
            assert not (set(map(tuple,f[GRAIN].to_numpy())) & set(map(tuple,g[GRAIN].to_numpy()))),(name,other,'game overlap')
            assert not (set(f.game_id) & set(g.game_id)),(name,other,'game_id overlap across pitchers')
            assert not (set(map(tuple,f[['pitcher_id','season']].to_numpy())) & set(map(tuple,g[['pitcher_id','season']].to_numpy()))),(name,other,'pitcher-season overlap')
            for w in range(7,50):
                fi=set(f.loc[f.case_season & f.lead_days.between(1,w),'injury_event_id'])
                gi=set(g.loc[g.case_season & g.lead_days.between(1,w),'injury_event_id'])
                assert not (fi & gi),(name,other,w,'positive injury-event overlap')


def freeze_protocol(spec,endpoint,base,ext1,cfg):
    out=ROOT/cfg['output'];(out/'tables').mkdir(parents=True,exist_ok=True)
    (ROOT/cfg['cache']).mkdir(parents=True,exist_ok=True)
    p=out/'protocol_freeze.json'
    if p.exists():
        saved=json.loads(p.read_text())
        assert saved['spec']==spec and saved['model_config']==cfg
        check_hashes(saved['prior_hashes']);check_hashes(saved['source_hashes'])
        return saved
    previous=json.loads((ROOT/ext1['output']/'experiment_freeze.json').read_text())
    check_hashes(previous['benchmark_hashes']);check_hashes(previous['endpoint_source_hashes'])
    paths=[p for d in [ext1['output'],ext1['cache'],'notebooks/kang_ucl_endpoint'] for p in (ROOT/d).rglob('*') if p.is_file() and p.name!='.DS_Store']
    paths += [ROOT/p for p in ['config/kang_ucl_endpoint.yaml','docs/kang_ucl_endpoint_methods.md',
        'src/injury_analysis/kang_ucl_endpoint.py','src/injury_analysis/kang_ucl_endpoint_data.py',
        'src/injury_analysis/kang_ucl_endpoint_reporting.py','scripts/make_kang_ucl_endpoint_notebook.py','tests/test_kang_ucl_endpoint.py']]
    hashes=dict(previous['benchmark_hashes']);hashes.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
    sources=dict(previous['endpoint_source_hashes']);sources[spec['canonical_episodes']]=digest(ROOT/spec['canonical_episodes'])
    with duckdb.connect(str(ROOT/base['database']),read_only=True) as con:
        cutoffs=[]
        for source in [endpoint['adjudicated_events'],spec['canonical_episodes']]:
            found=con.execute('SELECT DISTINCT observation_end_date FROM read_parquet(?)',[str(ROOT/source)]).fetchall()
            assert len(found)==1 and found[0][0] is not None
            cutoffs.append(str(pd.Timestamp(found[0][0]).date()))
        assert len(set(cutoffs))==1 and cutoffs[0]==spec['source_observation_end_expected']
    stat=(ROOT/base['database']).stat()
    saved=dict(spec=spec,model_config=cfg,prior_hashes=hashes,source_hashes=sources,
        source_cutoff=cutoffs[0],database_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),
        predictors=PREDICTORS,model_functions='scripts/kang_replication.R',
        intentional_change='Date-defined training/validation/test; all learned choices refitted on 2017–2023',
        followup_change='Administrative censoring of unobserved negative horizons at source cutoff',
        no_retuning=True,interpretation_policy_fixed_before_evaluation=True)
    atomic_json(p,saved)
    return saved


def load_period(period,spec,endpoint,base,ext1,cfg):
    """One period at a time; called for 2026 only after development freeze."""
    if period not in ['training','validation','test']:raise ValueError(period)
    start,end=spec[period+'_start'],spec[period+'_end']
    old=ROOT/base['cache'];out=ROOT/cfg['output'];cache=ROOT/cfg['cache']
    with duckdb.connect(str(ROOT/base['database']),read_only=True) as con:
        if period!='test':
            a=con.execute('SELECT * FROM read_parquet(?) WHERE game_date BETWEEN ?::DATE AND ?::DATE',
                [str(old/'appearances.parquet'),start,end]).df()
            e=con.execute('SELECT * FROM read_parquet(?) WHERE transaction_date BETWEEN ?::DATE AND ?::DATE',
                [str(old/'replication_episodes.parquet'),start,end]).df()
        else:
            if not (out/'development_freeze.json').exists():raise ValueError('2026 access requires completed training freeze')
            # Identical source fields and QC expressions to the frozen Kang adapter.
            names=list(dict.fromkeys(['game_id','pitcher_id','game_date','season','game_type','is_starter','outs_recorded']+
                                    [f for f in PREDICTORS if f not in EXTRA_SQL]))
            extra={k:v for k,v in EXTRA_SQL.items() if k in PREDICTORS}
            sql='''WITH a AS (SELECT {cols} FROM appearances WHERE game_type='R'
                AND game_date BETWEEN ?::DATE AND ?::DATE), extra AS (
                SELECT p.game_id,p.pitcher_id,{extra} FROM pitches p JOIN a
                ON p.game_id=a.game_id AND p.pitcher_id=a.pitcher_id
                WHERE p.is_actual_pitch AND p.game_date BETWEEN ?::DATE AND ?::DATE
                GROUP BY p.game_id,p.pitcher_id)
                SELECT a.*,{names} FROM a LEFT JOIN extra USING(game_id,pitcher_id)'''.format(
                    cols=','.join(names),extra=','.join(v+' AS '+k for k,v in extra.items()),
                    names=','.join('extra.'+k for k in extra))
            a=con.execute(sql,[start,end,start,end]).df()
            e=con.execute('SELECT * FROM read_parquet(?) WHERE transaction_date BETWEEN ?::DATE AND ?::DATE',
                [str(ROOT/spec['canonical_episodes']),start,end]).df()
        # No diagnosis or event belonging to a later modeled period enters training.
        u=con.execute('SELECT * FROM read_parquet(?) WHERE transaction_date BETWEEN ?::DATE AND ?::DATE',
            [str(ROOT/endpoint['adjudicated_events']),start,end]).df()
    a['game_date']=pd.to_datetime(a.game_date)
    assert period_for(a.game_date,spec).eq(period).all()
    s=season_screen(a,cfg)  # Fixed per-season rule, never learned from later seasons.
    frames,mapped,first,merged=refine_cohorts(a,e,u,s,cfg)
    rows=[]
    for role,f in frames.items():
        assert period_for(f.game_date,spec).eq(period).all()
        assert f.loc[f.case_season,'prediction_reference_date'].between(start,end).all()
        # The only historic adaptation is chronology; verify case rows/labels/features.
        if period!='test':
            with duckdb.connect() as con:
                archived=con.execute('SELECT * FROM read_csv_auto(?) WHERE game_date BETWEEN ?::DATE AND ?::DATE ORDER BY game_date,game_id,pitcher_id',
                    [str(ROOT/ext1['cache']/('ucl_'+role)/'cohort.csv'),start,end]).df()
            assert np.array_equal(f[GRAIN].to_numpy(),archived[GRAIN].to_numpy())
            assert f.case_season.reset_index(drop=True).equals(archived.case_season)
            assert np.allclose(f.lead_days,archived.lead_days,equal_nan=True)
            for feature in PREDICTORS:assert np.allclose(f[feature],archived[feature],equal_nan=True),feature
        d=cache/role;d.mkdir(exist_ok=True)
        f[EXPORT].to_csv(d/(period+'.csv'),index=False)
        rows.append(dict(period=period,role=role,games=len(f),pitcher_seasons=len(f[['pitcher_id','season']].drop_duplicates()),
            unique_pitchers=f.pitcher_id.nunique(),ucl_events=f.injury_event_id.nunique(),
            first_game=str(f.game_date.min().date()),last_game=str(f.game_date.max().date())))
    pd.DataFrame(rows).to_csv(out/'tables'/('cohort_'+period+'.csv'),index=False)
    s.to_csv(out/'tables'/('eligibility_'+period+'.csv'),index=False)
    feature_rows=[dict(period=period,feature=c,eligible_games=len(pd.concat(frames.values())),
        missing_fraction=float(pd.concat(frames.values())[c].isna().mean())) for c in PREDICTORS]
    pd.DataFrame(feature_rows).to_csv(out/'tables'/('coverage_'+period+'.csv'),index=False)
    if period=='test':
        # Outcome uncertainty is audited, not redefined or silently promoted.
        fields=['injury_event_id','player_id','initial_injury_region','evidence_category','endpoint_primary',
                'endpoint_exclusions','prediction_reference_date','ucl_confirmation_reported_date','right_censored',
                'procedure_link_valid','linked_procedure_ids']
        possible=u[u.uncertain_ucl_competitor | u.endpoint_primary].copy()
        selected_ids=set(pd.concat(frames.values()).injury_event_id.dropna())
        possible['in_model_cohort']=possible.injury_event_id.isin(selected_ids)
        possible[fields+['in_model_cohort']].to_csv(out/'tables/test_endpoint_uncertainty.csv',index=False)
        mapped.to_parquet(cache/'test_episode_mapping.parquet',index=False)
    return frames
