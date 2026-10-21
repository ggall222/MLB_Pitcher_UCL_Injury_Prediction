"""Reuse matched controls; extend only newly eligible UCL episodes, without smoothing."""
import json
import joblib
import numpy as np
import pandas as pd
from .ucl_dynamic_common import ROOT,OUT,CACHE,CFG,table,parquet,checkpoint,valid_checkpoint
from .ucl_dynamic_data import prepare_data,risk_frame
from .ucl_dynamic_models import model_path,scores
from .ucl_latent_visuals import paired_dependency_clusters

def event_window(rows,anchor,index_date=None):
    f=rows.loc[rows.analysis_eligible&rows.game_date.lt(anchor)&rows.game_date.ge(anchor-pd.Timedelta(days=60))].copy()
    if index_date is not None:f=f.loc[f.game_date.le(index_date)]
    f=f.sort_values(['game_date','game_id']);f['lead_days']=(anchor-f.game_date).dt.days
    f['outing']=np.arange(-len(f),0);return f

def match_events(a,events):
    old=pd.read_csv(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv',parse_dates=['anchor_date','control_anchor_date'])
    old=old.set_index('event_id');by={p:g for p,g in a.groupby('pitcher_id')}
    ev=events.loc[events.endpoint_primary&events.prediction_reference_date.between('2022-01-01','2025-12-31')].sort_values('prediction_reference_date')
    used=set(old.loc[old.index.isin(ev.injury_event_id),'control_pitcher_id']);records=[];matches=[];exclusions=[]
    pool=a.loc[a.analysis_eligible&a.followup_60d&(a.next_event_lead_days.isna()|a.next_event_lead_days.gt(90))
        &(a.days_since_any_event.isna()|a.days_since_any_event.gt(60))].copy()
    pool['month']=pool.game_date.dt.month;pool['age_band']=(pool.age_at_appearance//5).fillna(-1)
    pool['experience_band']=np.select([pool.prior_appearance_count.le(20),pool.prior_appearance_count.le(100)],[0,1],default=2)
    rng=np.random.default_rng(CFG['seed'])
    for e in ev.itertuples():
        g=by.get(e.player_id)
        if g is None:continue
        case=event_window(g,e.prediction_reference_date)
        case=case.loc[case.dynamic_next_event_id.eq(e.injury_event_id)].copy()
        case['outing']=np.arange(-len(case),0)
        if len(case)<3 or case.lead_days.min()>14:
            exclusions.append(dict(event_id=e.injury_event_id,reason='Fewer than 3 eligible prior outings or last outing >14d'));continue
        last=case.iloc[-1];ctrl=None;origin='new same-strata match'
        if e.injury_event_id in old.index:
            m=old.loc[e.injury_event_id];pid=m.control_pitcher_id;pseudo=m.control_anchor_date
            index=pseudo-pd.Timedelta(days=int(m.last_appearance_lead))
            ctrl=event_window(by[pid],pseudo,index);origin='reused original matched control'
        else:
            band=0 if last.prior_appearance_count<=20 else (1 if last.prior_appearance_count<=100 else 2)
            age=int(last.age_at_appearance//5) if pd.notna(last.age_at_appearance) else -1
            candidates=pool.loc[pool.season.eq(last.season)&pool.month.eq(last.game_date.month)&pool.is_starter.eq(last.is_starter)
                &pool.age_band.eq(age)&pool.experience_band.eq(band)&pool.pitcher_id.ne(e.player_id)&~pool.pitcher_id.isin(used)]
            for j in rng.permutation(len(candidates)):
                c=candidates.iloc[j];pid=c.pitcher_id;pseudo=c.game_date+pd.Timedelta(days=int(case.lead_days.min()))
                candidate=event_window(by[pid],pseudo,c.game_date)
                if len(candidate)>=3:ctrl=candidate;break
        if ctrl is None or len(ctrl)<3:
            exclusions.append(dict(event_id=e.injury_event_id,reason='No eligible unused control'));continue
        used.add(pid)
        for kind,window in [('Injury',case),('Control',ctrl)]:
            window=window.copy();window['kind']=kind;window['event_id']=e.injury_event_id;window['event_partition']=e.endpoint_partition
            records.append(window)
        matches.append(dict(event_id=e.injury_event_id,pitcher_id=e.player_id,player_name=e.player_name,control_pitcher_id=pid,
            case_anchor=e.prediction_reference_date,control_anchor=pseudo,case_outings=len(case),control_outings=len(ctrl),origin=origin))
    return pd.concat(records,ignore_index=True),pd.DataFrame(matches),pd.DataFrame(exclusions)

def balanced_cells(aligned,model,mode):
    f=aligned.copy()
    if mode=='outings':f=f.loc[f.outing.ge(-5)].copy();f['point']=f.outing+5;points=5
    else:
        f['point']=pd.cut(f.lead_days,[1,8,15,31,61],labels=[3,2,1,0],right=False).astype(int);points=4
    cells=f.groupby(['event_id','dependency_cluster','event_partition','kind','point'])[model].mean().unstack('kind').dropna().reset_index()
    complete=cells.groupby('event_id').point.nunique();cells=cells.loc[cells.event_id.isin(complete.loc[complete.eq(points)].index)].copy()
    cells['difference']=cells.Injury-cells.Control
    return cells,points

def bootstrap_means(values,clusters):
    unique,inv=np.unique(clusters,return_inverse=True);rng=np.random.default_rng(CFG['seed'])
    if len(unique)<2:return np.full((CFG['bootstrap_replicates'],values.shape[1]),np.nan)
    counts=np.array([np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique)) for _ in range(CFG['bootstrap_replicates'])])
    weights=counts[:,inv];return weights@values/weights.sum(axis=1)[:,None]

def run_trajectories():
    if valid_checkpoint('trajectories'):return
    a,events,specs=prepare_data();a=risk_frame(a,events)
    aligned,matches,exclusions=match_events(a,events)
    selected=json.loads((OUT/'frozen_protocol.json').read_text())['selected_model']
    names=list(dict.fromkeys(['Profile','Departure','Profile + departure','Profile + departure + trajectory',selected]))
    clusters=paired_dependency_clusters(aligned);aligned['dependency_cluster']=aligned.event_id.map(clusters)
    for n in names:
        obj=joblib.load(CACHE/'models'/(model_path(30,n)+'.joblib'));aligned[n]=scores(obj,aligned)
    parquet(aligned,'matched_risk_trajectories');table(matches,'matched_controls');table(exclusions,'trajectory_exclusions')
    summaries=[];shapes=[];allcells=[]
    for n in names:
        for mode in ['outings','calendar']:
            cells,points=balanced_cells(aligned,n,mode);cells['model']=n;cells['mode']=mode;allcells.append(cells)
            for part in ['combined','validation','test']:
                cf=cells if part=='combined' else cells.loc[cells.event_partition.eq(part)]
                if cf.empty:continue
                meta=cf.drop_duplicates('event_id').set_index('event_id')
                for kind in ['Injury','Control','difference']:
                    values=cf.pivot(index='event_id',columns='point',values=kind).reindex(columns=range(points))
                    cluster=meta.loc[values.index,'dependency_cluster'].to_numpy();v=values.to_numpy()
                    boot=bootstrap_means(v,cluster);mean=v.mean(axis=0);se=np.nanstd(boot,axis=0,ddof=1)
                    standardized=np.divide(boot-mean,se,out=np.zeros_like(boot),where=se>0)
                    critical=np.nanquantile(np.max(np.abs(standardized),axis=1),.95)
                    for point in range(points):
                        summaries.append(dict(model=n,mode=mode,partition=part,kind=kind,point=point,
                            mean=mean[point],ci_low=np.nanquantile(boot[:,point],.025),ci_high=np.nanquantile(boot[:,point],.975),
                            simultaneous_low=mean[point]-critical*se[point],simultaneous_high=mean[point]+critical*se[point],
                            events=len(v),dependency_clusters=len(np.unique(cluster))))
                    change=boot[:,-1]-boot[:,0]
                    shapes.append(dict(model=n,mode=mode,partition=part,kind=kind,events=len(v),mean_offset=float(v.mean()),
                        last_minus_first=float((v[:,-1]-v[:,0]).mean()),change_ci_low=float(np.nanquantile(change,.025)),change_ci_high=float(np.nanquantile(change,.975)),
                        fraction_events_rising=float(np.mean(v[:,-1]>v[:,0]))))
    table(pd.DataFrame(summaries),'risk_trajectory_summary');table(pd.DataFrame(shapes),'risk_trajectory_shapes');table(pd.concat(allcells,ignore_index=True),'balanced_risk_cells')
    checkpoint('trajectories',[CACHE/'matched_risk_trajectories.parquet',OUT/'tables/risk_trajectory_summary.csv',OUT/'tables/risk_trajectory_shapes.csv'])
