"""Historical derivation and audit, preserving the original master feature tables."""
import gc
import json
import logging
import numpy as np
import pandas as pd
import yaml
from .config import ROOT
from . import build_appearances, build_workload_features, build_mechanical_features, build_baselines, build_model_dataset
from . import research_labels, reconcile_injuries
from .dictionary import write_dictionary

LOG=logging.getLogger(__name__)
KEYS=['game_id','pitcher_id']


def add_context(a,box,injuries):
    """Immutable demographics and observations strictly before the prediction day."""
    pieces=[]
    prior={pid:g.surgery_date_upper.dropna().sort_values().to_numpy(dtype='datetime64[ns]')
           for pid,g in injuries[injuries.confirmed_ucl_surgery.fillna(False)].dropna(subset=['pitcher_id']).groupby('pitcher_id')}
    boxes={pid:g.sort_values(['game_date','game_id']) for pid,g in box.groupby('pitcher_id')}
    for pid,g in a.groupby('pitcher_id',sort=False):
        g=g.copy();b=boxes.get(pid,box.iloc[:0]);days=g.game_date.to_numpy(dtype='datetime64[ns]')
        bd=b.game_date.to_numpy(dtype='datetime64[ns]');ends=np.searchsorted(bd,days,side='left')
        starts=np.r_[0,np.cumsum(b.is_starter.fillna(False).to_numpy(dtype=int))]
        g['observed_prior_mlb_appearances']=ends;g['observed_prior_mlb_starts']=starts[ends]
        surgeries=prior.get(pid,np.array([],dtype='datetime64[ns]'))
        g['documented_prior_ucl_count']=np.searchsorted(surgeries,days,side='left')
        g['documented_prior_ucl']=g.documented_prior_ucl_count.gt(0)
        pieces.append(g)
    return pd.concat(pieces,ignore_index=True)


def build(db,client,cfg):
    # A narrower invocation must never remove previously collected observations.
    first=db.scalar('SELECT min(game_date) FROM pitches');last=db.scalar('SELECT max(game_date) FROM pitches')
    if first is None:raise RuntimeError('No pitches collected')
    cfg['start_date']=min(cfg['start_date'],first);cfg['end_date']=max(cfg['end_date'],last)
    reconcile_injuries.apply_reconciliation(db,client,cfg)
    pieces=[];types=[]
    years=db.query('SELECT DISTINCT year(game_date) AS year FROM pitches ORDER BY year').year.tolist()
    for year in years:
        LOG.info('Aggregating appearance metrics for %s',year)
        a,t=build_appearances.build(db,cfg,year);pieces.append(a);types.append(t);gc.collect()
    a=pd.concat(pieces,ignore_index=True);pt=pd.concat(types,ignore_index=True);del pieces,types
    box=db.query('SELECT b.*,g.game_date,g.scheduled_start,g.game_type,g.timing_uncertain FROM boxscore_appearances b JOIN games g USING(game_id)')
    starter_counts=box.groupby(['game_id','team_id']).is_starter.sum().rename('official_starters_on_team').reset_index()
    a=a.merge(starter_counts,on=['game_id','team_id'],how='left',validate='many_to_one')
    observed_first=db.query('''SELECT game_id,pitcher_id,CASE WHEN inning_topbot='Top' THEN home_team_id ELSE away_team_id END AS team_id
        FROM pitches JOIN games USING(game_id)
        QUALIFY row_number() OVER(PARTITION BY game_id,inning_topbot ORDER BY inning,at_bat_number,pitch_number)=1''')
    observed_first=observed_first.rename(columns={'pitcher_id':'first_observed_pitcher_id'})
    a=a.merge(observed_first,on=['game_id','team_id'],how='left',validate='many_to_one')
    a['starter_crosscheck_disagrees']=(a.is_starter.fillna(False)&a.first_observed_pitcher_id.notna()&a.pitcher_id.ne(a.first_observed_pitcher_id))
    ambiguous=a.starter_crosscheck_disagrees|a.is_starter.isna()|a.official_starters_on_team.ne(1)|a.starter_source.isna()
    a['starter_status']=np.where(ambiguous,'ambiguous',np.where(a.is_starter.fillna(False),'starter','relief'))
    a['valid_starting_appearance']=(a.starter_status.eq('starter')&a.quality_eligible&a.game_type.eq('R')).astype('boolean')
    injuries=db.query('SELECT * FROM injuries');transactions=db.query('SELECT * FROM transactions')
    reliable_box=box[~box.timing_uncertain.fillna(True)&box.game_type.eq('R')]
    a=add_context(a,reliable_box,injuries)
    a=a.sort_values(['pitcher_id','game_date','scheduled_start','game_id']).reset_index(drop=True)
    reliable_box=box[~box.timing_uncertain.fillna(True)&box.game_type.eq('R')]
    LOG.info('Building histories for %s appearances',len(a))
    w=build_workload_features.build(a,reliable_box,db.query('SELECT * FROM season_pitching'),cfg)
    w=mask_uncertain_workloads(a,w,box)
    m=build_mechanical_features.build(a,cfg);b=build_baselines.build(a,cfg)
    episodes,events,procedures=research_labels.build_events(injuries,transactions,reliable_box,cfg)
    procedure_updates=[]
    observed_dates={pid:g.game_date for pid,g in reliable_box.groupby('pitcher_id')}
    for r in procedures.itertuples():
        dates=observed_dates.get(r.pitcher_id,pd.Series(dtype='datetime64[ns]'))
        before=dates[dates<r.label_reference_date]
        procedure_updates.append(dict(injury_id=r.source_injury_id,known_onset_date=r.label_reference_date,
            il_placement_date=r.il_date,onset_basis=r.onset_confidence,onset_transaction_id=r.onset_transaction_id,
            first_reported_elbow_date=r.label_reference_date if r.onset_confidence=='elbow_public_report' else pd.NaT,
            last_pre_onset_appearance=before.max(),last_healthy_mlb_appearance=pd.NaT))
    db.upsert('injuries',pd.DataFrame(procedure_updates),['injury_id'])
    incomplete=db.scalar("SELECT count(*) FROM ingestion_state WHERE source IN ('schedule','boxscore','transactions','roegele') AND status='failed'")
    labels=research_labels.build(a,events,episodes,reliable_box,cfg,not incomplete)
    model,features=build_model_dataset.build(a,w,m,b,labels)
    LOG.info('Persisting %s master features; %s model rows',len(features),len(model))
    with db.transaction():
        for table,frame,keys in [('appearances',a,KEYS),('appearance_pitch_metrics',pt,KEYS+['pitch_type']),
            ('workload_features',w,KEYS),('mechanical_features',m,KEYS),('pitcher_baselines',b,KEYS),
            ('injury_labels',labels,KEYS),('model_features',model,KEYS),
            ('arm_injury_episodes',episodes,['episode_id']),('injury_events',events,['injury_event_id']),
            ('injury_event_procedures',procedures,['source_injury_id'])]:
            db.replace_derived(table,frame,keys)
        db.con.execute((ROOT/'sql/views.sql').read_text())
    del w,m,b,labels;gc.collect()
    (cfg['export_path']/'feature_columns.json').write_text(json.dumps(features,indent=2))
    model.to_parquet(cfg['export_path']/'model_ready_pitcher_appearances.parquet',index=False)
    model[model.valid_starting_appearance].to_parquet(cfg['export_path']/'model_ready_starting_pitcher_appearances.parquet',index=False)
    selected=yaml.safe_load((ROOT/'config/model_features_v1.yaml').read_text())['features']
    curated=['game_id','pitcher_id','pitcher_name','game_date','season','training_eligible','next_injury_event_id','next_known_ucl_onset']+[r['name'] for r in selected]+research_labels.LABEL_COLUMNS
    from .database import ident
    db.con.execute('CREATE OR REPLACE VIEW model_ready_starting_pitcher_features_v1 AS SELECT '+','.join(ident(c) for c in curated)+' FROM model_ready_starting_pitcher_appearances')
    model.loc[model.valid_starting_appearance,curated].to_parquet(cfg['export_path']/'model_ready_starting_pitcher_features_v1.parquet',index=False)
    write_dictionary(db,cfg,features)
    ddls=db.query("SELECT sql FROM duckdb_tables() WHERE schema_name='main'")
    (cfg['report_path']/'derived_schema.sql').write_text('\n\n'.join(ddls.sql.dropna()))
    db.con.execute('CHECKPOINT')
    from .research_reports import generate
    report=generate(db,cfg,features,model,events)
    return report


def mask_uncertain_workloads(a,w,box):
    """Do not call partial workload totals complete around suspended/resumed games."""
    out=w.merge(a[['game_id','pitcher_id','game_date']],on=KEYS,validate='one_to_one')
    bad=box[box.timing_uncertain.fillna(True)]
    groups={pid:g for pid,g in bad.groupby('pitcher_id')}
    for pid,index in out.groupby('pitcher_id').groups.items():
        g=groups.get(pid)
        if g is None:continue
        days=out.loc[index,'game_date']
        season_bad=np.zeros(len(index),dtype=bool)
        rest_bad=np.zeros(len(index),dtype=bool)
        for n in (7,14,28):
            affected=np.zeros(len(index),dtype=bool)
            for r in g.itertuples():
                start=r.game_date
                finish=max(start,r.scheduled_start.normalize()) if pd.notna(r.scheduled_start) else pd.Timestamp(start.year,12,31)
                affected |= ((days>start)&(days-pd.Timedelta(days=n)<=finish)).to_numpy()
                season_bad |= ((days>start)&days.dt.year.eq(start.year)).to_numpy()
                rest_bad |= ((days>start)&(out.loc[index,'workload_max_date']<=finish)).to_numpy()
            columns=[f'pitches_prev_{n}d',f'pitches_{n}d',f'innings_prev_{n}d',f'innings_{n}d',f'appearances_{n}d',f'appearances_prev_{n}d']
            out.loc[np.asarray(index)[affected],columns]=np.nan
            out.loc[np.asarray(index)[affected],f'workload_{n}d_complete']=False
        affected30=np.zeros(len(index),dtype=bool)
        for r in g.itertuples():
            finish=max(r.game_date,r.scheduled_start.normalize()) if pd.notna(r.scheduled_start) else pd.Timestamp(r.game_date.year,12,31)
            affected30 |= ((days>r.game_date)&(days-pd.Timedelta(days=30)<=finish)).to_numpy()
        out.loc[np.asarray(index)[affected30],'starts_prev_30d']=np.nan
        out.loc[np.asarray(index)[season_bad],['season_pitch_count','season_innings','season_appearances','season_starts','current_vs_previous_season_workload']]=np.nan
        out.loc[np.asarray(index)[rest_bad],'rest_days']=np.nan
    out['workload_acceleration_7d_vs_21d']=out.pitches_prev_7d/7-(out.pitches_prev_28d-out.pitches_prev_7d)/21
    out['acute_chronic_pitch_ratio']=(out.pitches_7d/7)/(out.pitches_28d/28).replace(0,np.nan)
    return out.drop(columns='game_date')
