"""Frozen representations, strictly dated departure histories, landmark outcomes."""
import json
import duckdb
import numpy as np
import pandas as pd
from .ucl_dynamic_common import ROOT,CACHE,OUT,CFG,table,parquet,write_json,checkpoint,valid_checkpoint
from .ucl_dynamic_endpoint import run_endpoint

INTERACTIONS=[('avg_fb_velo_baseline_mean','velo_z'),
    ('release_z_sd_baseline_mean','release_z_sd_z'),('extension_avg_baseline_mean','extension_z')]

def frozen_representations():
    old=json.loads((ROOT/'data/injury/analysis/feature_consolidation/candidate_specifications.json').read_text())
    d=old['Departure only']['features'];p=old['Profile only']['features']
    trajectories=[f'dyn_{c}_{kind}' for c in CFG['trajectory_fields'] for kind in ['slope4','absmean4','persist4']]
    return {'Background':dict(features=old['Background context']['features'],interactions=[]),
        'Profile':dict(features=p,interactions=[]),'Departure':dict(features=d,interactions=[]),
        'Profile + departure':dict(features=p+d,interactions=[]),
        'Profile + departure + trajectory':dict(features=p+d+trajectories,interactions=[]),
        'Profile + departure + trajectory + interactions':dict(features=p+d+trajectories,interactions=INTERACTIONS),
        'Previous 62':dict(features=old['Previous 62']['features'],interactions=[]),
        'Stable 62':dict(features=old['Stable representatives 90%']['features'],interactions=[])}

def departure_trajectories(frame):
    """Last three strictly earlier-day outings plus current, within 30 days.

    Full four-outing support is required for each field. No injury/event dates
    select the history. Current inclusion is intentional: prediction is post-outing.
    """
    out=frame.copy();names=[]
    for c in CFG['trajectory_fields']:
        for kind in ['slope4','absmean4','persist4']:
            name=f'dyn_{c}_{kind}';out[name]=np.nan;names.append(name)
    out['dynamic_history_max_date']=pd.NaT
    for _,group in out.groupby('pitcher_id',sort=False):
        g=group.sort_values(['game_date','game_id']);dates=g.game_date.to_numpy(dtype='datetime64[ns]')
        values=g[CFG['trajectory_fields']].to_numpy(dtype=float)
        quality=(g.quality_eligible.fillna(False)&~g.timing_uncertain.fillna(True)).to_numpy()
        for i in range(len(g)):
            stop=np.searchsorted(dates,dates[i],side='left')
            candidates=np.flatnonzero(quality[:stop]&(dates[:stop]>=dates[i]-np.timedelta64(CFG['trajectory_max_days'],'D')))
            if len(candidates)<CFG['trajectory_history_outings']:continue
            ix=np.r_[candidates[-CFG['trajectory_history_outings']:],i]
            v=values[ix];valid=np.isfinite(v).all(axis=0)
            # Slope per observed outing, not per day; irregular timing stays explicit.
            x=np.arange(len(ix),dtype=float);x-=x.mean()
            slopes=x@v/np.dot(x,x)
            for j,c in enumerate(CFG['trajectory_fields']):
                if valid[j]:
                    out.loc[g.index[i],f'dyn_{c}_slope4']=slopes[j]
                    out.loc[g.index[i],f'dyn_{c}_absmean4']=np.abs(v[:,j]).mean()
                    out.loc[g.index[i],f'dyn_{c}_persist4']=(np.abs(v[:,j])>=CFG['absolute_departure_threshold']).mean()
            out.loc[g.index[i],'dynamic_history_max_date']=pd.Timestamp(dates[ix[-2]])
    return out,names

def prepare_data():
    if valid_checkpoint('data'):
        return pd.read_parquet(CACHE/'landmarks.parquet'),run_endpoint(),json.loads((CACHE/'representations.json').read_text())
    events=run_endpoint();a=pd.read_parquet(ROOT/'data/injury/analysis/appearances.parquet')
    specs=frozen_representations();needed=sorted(set(c for s in specs.values() for c in s['features'] if not c.startswith('dyn_')))
    missing=[c for c in needed if c not in a]
    if missing:
        with duckdb.connect(str(ROOT/'database/mlb_ucl.duckdb'),read_only=True) as con:
            columns=set(con.execute('DESCRIBE model_features').fetchdf().column_name)
            assert set(missing)<=columns
            con.register('_keys',a[['pitcher_id','game_id']])
            projection=','.join('m."'+c+'"' for c in missing)
            extra=con.execute(f'SELECT m.pitcher_id,m.game_id,{projection} FROM model_features m JOIN _keys k USING(pitcher_id,game_id)').fetchdf()
        a=a.merge(extra,on=['pitcher_id','game_id'],validate='one_to_one')
    a,names=departure_trajectories(a)
    records=[]
    for model,spec in specs.items():
        for c in spec['features']:
            records.append(dict(representation=model,feature=c,derivation='current + three prior valid-day outings within 30 days' if c.startswith('dyn_') else 'unchanged frozen existing field',
                timing='after completed outing; historical terms strictly prior'))
        for left,right in spec['interactions']:
            records.append(dict(representation=model,feature=left+' × '+right,
                derivation='product of training-standardized main effects; product mean/SD fit on training only',timing='after completed outing'))
    table(pd.DataFrame(records),'feature_representations')
    table(a.groupby(['partition','season'])[names].agg(lambda x:x.notna().mean()).reset_index(),'trajectory_feature_support')
    parquet(a,'landmarks');write_json(specs,CACHE/'representations.json')
    checkpoint('data',[CACHE/'landmarks.parquet',CACHE/'representations.json',OUT/'tables/feature_representations.csv'])
    return a,events,specs

def risk_frame(a,events,horizon=30,endpoint='primary',competing='negative',timing='earliest'):
    """First-event landmark risk with ambiguous elbow outcomes censored.

    A known non-UCL injury is a competing first event (Y=0), never a healthy
    control claim. The sensitivity policy excludes these rows. Other negatives
    require an observed MLB outing at/after the horizon within the legacy grace.
    """
    f=a.copy().reset_index(drop=True);f['target']=pd.Series(pd.NA,index=f.index,dtype='Int8')
    f['dynamic_next_event_id']=None;f['dynamic_next_event_date']=pd.NaT
    f['dynamic_next_target']=False;f['dynamic_competing_known']=False
    f['label_status']='not_eligible';f['dynamic_followup']=False
    event_col='endpoint_'+endpoint
    by_event={pid:g for pid,g in events.groupby('player_id')};end=events.observation_end_date.max()
    for pid,g in f.groupby('pitcher_id',sort=False):
        g=g.sort_values(['game_date','game_id']);ix=g.index.to_numpy();dates=g.game_date.to_numpy(dtype='datetime64[ns]')
        thresholds=dates+np.timedelta64(horizon,'D');after=np.searchsorted(dates,thresholds)
        followed=(after<len(dates))&(thresholds<=np.datetime64(end))
        last=dates[np.minimum(after,len(dates)-1)]
        followed&=last<=thresholds+np.timedelta64(CFG['negative_followup_grace_days'],'D')
        f.loc[ix,'dynamic_followup']=followed
        es=by_event.get(pid)
        if es is None:continue
        es=es.copy();es['_ref']=es.public_known_date if timing=='public' else es.blocking_date
        es=es.dropna(subset=['_ref']).sort_values(['_ref','injury_event_id'])
        if es.empty:continue
        k=np.searchsorted(es._ref.to_numpy(dtype='datetime64[ns]'),dates,side='right');valid=k<len(es)
        matched=es.iloc[k[valid]]
        f.loc[ix[valid],'dynamic_next_event_id']=matched.injury_event_id.to_numpy()
        f.loc[ix[valid],'dynamic_next_event_date']=matched._ref.to_numpy()
        f.loc[ix[valid],'dynamic_next_target']=matched[event_col].to_numpy()
        known=matched.target_eligible&~matched.uncertain_ucl_competitor
        f.loc[ix[valid],'dynamic_competing_known']=known.to_numpy()
    f['dynamic_lead_days']=(f.dynamic_next_event_date-f.game_date).dt.days
    near=f.dynamic_lead_days.between(1,horizon);base=f.analysis_eligible
    positive=base&near&f.dynamic_next_target
    rival=base&near&~f.dynamic_next_target&f.dynamic_competing_known
    negative=base&~near&f.dynamic_followup
    f.loc[base,'label_status']='censored_no_observed_followup'
    f.loc[base&near,'label_status']='censored_ambiguous_or_excluded_episode'
    f.loc[negative,'target']=0;f.loc[negative,'label_status']='observed_event_free_horizon'
    f.loc[positive,'target']=1;f.loc[positive,'label_status']='ucl_first_event'
    if competing=='negative':f.loc[rival,'target']=0
    f.loc[rival,'label_status']='non_ucl_competing_first_event' if competing=='negative' else 'censored_competing_event'
    f['next_event_id']=f.dynamic_next_event_id
    f['horizon']=horizon;f['endpoint']=endpoint;f['competing_policy']=competing;f['timing_policy']=timing
    return f

def event_cohort(events,partition,endpoint='primary',timing='earliest',settings=None):
    cfg=CFG if settings is None else settings
    starts={'train':'2014-12-31','validation':cfg['train_end'],'test':cfg['validation_end'],'descriptive_2026':cfg['test_end']}
    ends={'train':cfg['train_end'],'validation':cfg['validation_end'],'test':cfg['test_end'],'descriptive_2026':str(events.observation_end_date.max().date())}
    ref=events.public_known_date if timing=='public' else events.prediction_reference_date
    e=events.loc[events['endpoint_'+endpoint]&ref.gt(starts[partition])&ref.le(ends[partition])].copy()
    e['label_reference_date']=ref.loc[e.index];e['pitcher_id']=e.player_id
    return e
