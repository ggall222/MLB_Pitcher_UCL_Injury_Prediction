"""Narrow interpretation diagnostic on the original matched sample; no feature search."""
import numpy as np
import pandas as pd
from .ucl_temporal_models import load
from .ucl_temporal_trajectories import paired_cells
from .ucl_dynamic_trajectories import bootstrap_means
from .ucl_replication_common import ROOT,OLD,OLD_CACHE,CFG,OUT,table,checkpoint,valid

RAW=['total_pitches','rest_days','pitches_prev_7d','pitches_prev_14d','pitches_prev_28d',
     'appearances_prev_14d','appearances_prev_28d','is_starter','starter_share_prev_5']

def run_workload():
    if valid('workload'):return
    a=pd.read_parquet(OLD_CACHE/'matched_family_scores.parquet');family='Workload and rest'
    obj=load('1-60','Trajectory | '+family);encoder=obj['encoder'];X=encoder.transform(a);beta=obj['model'].coef_[0]
    scales=pd.read_csv(OLD/'tables/trajectory_score_scales.csv').set_index('family');sd=scales.loc[family,'sd']
    contribution=X*beta/sd;features=encoder.features
    for feature in features:
        indices=[j for j,name in enumerate(encoder.names) if name==feature or name=='missingindicator_'+feature]
        a['contribution__'+feature]=contribution[:,indices].sum(axis=1)
    fields=[family]+[c for c in RAW if c in a]+['contribution__'+c for c in features]
    rows=[];events=[];loo=[];cells=[]
    for mode in ['calendar','outings']:
        required=[0,1,2,3] if mode=='calendar' else [2,3,4]
        baseline=paired_cells(a,family,mode)
        for part in ['validation','test']:
            base=baseline.loc[baseline.event_partition.eq(part)&baseline.point.isin(required)]
            counts=base.groupby('event_id').point.nunique();support=set(counts[counts.eq(len(required))].index)
            for field in fields:
                cc=paired_cells(a,field,mode);cc=cc.loc[cc.event_partition.eq(part)&cc.event_id.isin(support)&cc.point.isin(required)]
                counts=cc.groupby('event_id').point.nunique();cc=cc.loc[cc.event_id.isin(counts[counts.eq(len(required))].index)]
                pv=cc.pivot(index='event_id',columns='point',values='difference').reindex(columns=required)
                cl=cc.drop_duplicates('event_id').set_index('event_id').loc[pv.index,'dependency_cluster'].to_numpy()
                delta=(pv.iloc[:,-1]-pv.iloc[:,0]).to_numpy();draw=bootstrap_means(delta[:,None],cl)[:,0]
                kind='score' if field==family else 'coefficient_contribution' if field.startswith('contribution__') else 'raw_measure'
                rows.append(dict(partition=part,mode=mode,field=field,kind=kind,events=len(delta),dependency_clusters=len(np.unique(cl)),
                    near_minus_far_case_control_difference=float(delta.mean()),ci_low=float(np.nanquantile(draw,.025)),ci_high=float(np.nanquantile(draw,.975)),
                    fraction_events_declining=float((delta<0).mean())))
                for event,d in zip(pv.index,delta):
                    events.append(dict(partition=part,mode=mode,field=field,event_id=event,near_minus_far_difference=d))
                    keep=pv.index!=event
                    loo.append(dict(partition=part,mode=mode,field=field,excluded_event=event,remaining_events=int(keep.sum()),
                        mean_without_event=float(delta[keep].mean()) if keep.sum() else np.nan))
                cells.append(cc.assign(field=field,mode=mode))
    table(pd.DataFrame(rows),'workload_diagnostic');table(pd.DataFrame(events),'workload_event_changes')
    table(pd.DataFrame(loo),'workload_leave_event_out');table(pd.concat(cells,ignore_index=True),'workload_matched_cells')
    note='''# Workload/rest diagnostic scope

This is an interpretation check on the **original 2022–2023 development and 2024–2025 exploratory-transfer matched pairs**, not a new 2026 matched analysis. It decomposes the frozen 60-day workload score into its existing standardized linear contributions (including each field's missingness indicator). No predictors, models, windows or matches are added.

Calendar balance retains the same pairs in all four original bins; outing balance retains final, 2–3 and 4–5 outings. Each raw measure reports its available balanced support. Change means final-minus-distant change in the matched case-minus-control difference. Intervals bootstrap dependency clusters linking shared case/control pitchers. They are descriptive, pointwise and unadjusted.

Lower pitch counts, fewer recent appearances, longer rest and changing starter share are usage patterns. They cannot establish skipped scheduled starts, medical advice, symptoms, intent, or biological deterioration without independent dated documentation. A score decrease is a weighted combination, not a direct workload measurement. Event-level changes and leave-event-out means expose dependence on individual episodes. None of these diagnostics is used for nomination, fitting or threshold selection.
'''
    (OUT/'workload_diagnostic_notes.md').write_text(note)
    checkpoint('workload',[OUT/'tables'/(n+'.csv') for n in ['workload_diagnostic','workload_event_changes','workload_leave_event_out','workload_matched_cells']]+[OUT/'workload_diagnostic_notes.md'])
