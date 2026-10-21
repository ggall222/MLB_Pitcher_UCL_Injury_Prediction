"""Retrospective matched family-score trajectories; fixed score across lead times."""
import numpy as np
import pandas as pd
from .ucl_dynamic_trajectories import bootstrap_means
from .ucl_temporal_common import ROOT,OUT,CACHE,CFG,table,read_table,parquet,checkpoint,valid_checkpoint
from .ucl_temporal_models import load,scores
from .ucl_temporal_data import prepare

CALENDAR=['31–60','15–30','8–14','1–7']
OUTINGS=['11+','6–10','4–5','2–3','Final']

def paired_cells(aligned, family, mode):
    f=aligned.copy()
    if mode=='calendar':
        f['point']=pd.cut(f.lead_days,[0,7,14,30,60],labels=[3,2,1,0]).astype(float)
    else:
        distance=-f.outing
        f['point']=np.select([distance.eq(1),distance.between(2,3),distance.between(4,5),distance.between(6,10)],
                             [4,3,2,1],default=0)
    cells=f.dropna(subset=['point']).groupby(
        ['event_id','dependency_cluster','event_partition','kind','point'])[family].mean().unstack('kind').dropna().reset_index()
    cells['difference']=cells.Injury-cells.Control
    return cells

def summarize_cells(cells, family, mode, part, support):
    cf=cells.loc[cells.event_partition.eq(part)].copy()
    required=list(range(4)) if mode=='calendar' else [2,3,4]
    if support=='balanced':
        counts=cf.loc[cf.point.isin(required)].groupby('event_id').point.nunique()
        cf=cf.loc[cf.event_id.isin(counts[counts.eq(len(required))].index)&cf.point.isin(required)]
    rows=[]
    for point,g in cf.groupby('point'):
        for kind in ['Injury','Control','difference']:
            v=g[kind].to_numpy()[:,None]
            b=bootstrap_means(v,g.dependency_cluster.to_numpy())[:,0]
            rows.append(dict(partition=part,family=family,mode=mode,support=support,point=int(point),
                label=(CALENDAR if mode=='calendar' else OUTINGS)[int(point)],kind=kind,
                mean=float(v.mean()),ci_low=float(np.nanquantile(b,.025)),ci_high=float(np.nanquantile(b,.975)),
                events=len(g),dependency_clusters=g.dependency_cluster.nunique()))
    # Paired trajectory endpoint contrast on the SAME events; never compare
    # an early point in one event set with a late point in a different event set.
    shape=None
    if support=='balanced' and len(cf):
        v=cf.pivot(index='event_id',columns='point',values='difference').reindex(columns=required)
        clusters=cf.drop_duplicates('event_id').set_index('event_id').loc[v.index,'dependency_cluster'].to_numpy()
        delta=(v.iloc[:,-1]-v.iloc[:,0]).to_numpy()[:,None]
        b=bootstrap_means(delta,clusters)[:,0]
        shape=dict(partition=part,family=family,mode=mode,events=len(v),dependency_clusters=len(np.unique(clusters)),
            final_minus_distant_difference=float(delta.mean()),ci_low=float(np.nanquantile(b,.025)),
            ci_high=float(np.nanquantile(b,.975)),fraction_events_increasing=float((delta>0).mean()),
            interpretation='Pointwise exploratory paired contrast; no physiological or prospective claim')
    return rows,shape

def run_trajectories():
    if valid_checkpoint('trajectories'):return
    if not (OUT/'development_freeze.json').exists():
        raise ValueError('Finalize predictive development before retrospective plots')
    _,_,groups,_=prepare()
    aligned=pd.read_parquet(ROOT/'data/injury/analysis/ucl_dynamic/matched_risk_trajectories.parquet')
    scales=read_table('trajectory_score_scales').set_index('family')
    for family in groups:
        obj=load('1-60','Trajectory | '+family)
        aligned[family]=(scores(obj,aligned,logit=True)-scales.loc[family,'mean'])/scales.loc[family,'sd']
    summaries=[];shapes=[];allcells=[]
    for family in groups:
        for mode in ['calendar','outings']:
            cells=paired_cells(aligned,family,mode)
            allcells.append(cells.assign(family=family,mode=mode))
            for part in ['validation','test']:
                for support in ['available_pairs','balanced']:
                    rows,shape=summarize_cells(cells,family,mode,part,support)
                    summaries+=rows
                    if shape is not None:shapes.append(shape)
    table(pd.DataFrame(summaries),'trajectory_summary')
    table(pd.DataFrame(shapes),'trajectory_shape')
    table(pd.concat(allcells,ignore_index=True),'trajectory_cells')
    # Both role-specific appearance schedules and calendar distances are retained.
    metadata=aligned[['event_id','event_partition','kind','outing','lead_days','is_starter','pitcher_id']].copy()
    metadata['outing_bin']=np.select([metadata.outing.eq(-1),metadata.outing.between(-3,-2),
        metadata.outing.between(-5,-4),metadata.outing.between(-10,-6)],['Final','2–3','4–5','6–10'],default='11+')
    schedule=metadata.groupby(['event_partition','kind','is_starter','outing_bin']).agg(
        rows=('lead_days','size'),events=('event_id','nunique'),median_lead_days=('lead_days','median'),
        minimum_lead_days=('lead_days','min'),maximum_lead_days=('lead_days','max')).reset_index()
    table(schedule,'appearance_calendar_support')
    parquet(aligned,'matched_family_scores')
    checkpoint('trajectories',[OUT/'tables/trajectory_summary.csv',OUT/'tables/trajectory_shape.csv',
        OUT/'tables/trajectory_cells.csv',OUT/'tables/appearance_calendar_support.csv',CACHE/'matched_family_scores.parquet'])
