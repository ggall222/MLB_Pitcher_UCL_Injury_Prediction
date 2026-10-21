"""Whole-pitcher sparse refits; value, availability and rank stability kept separate."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .consolidation_common import CFG,CACHE,table,read_table,inputs
from .consolidation_models import fit_named,scores,source_coefficients,feature_sets,summary_metrics,grouped_permutation


def pitcher_bootstrap_indices(frame,seed):
    groups=list(frame.groupby('pitcher_id',sort=True).indices.values())
    rng=np.random.default_rng(seed)
    selected=rng.integers(0,len(groups),len(groups))
    return np.concatenate([groups[i] for i in selected])


def run_stability():
    f,events,_=inputs();train=f.loc[f.partition.eq('train')].reset_index(drop=True);valid=f.loc[f.partition.eq('validation')]
    selected=json.loads((CACHE/'elastic_selection.json').read_text());core,_,_=feature_sets()
    if selected.get('input_features')!=len(core):raise ValueError('Elastic Net must be tuned on the full feasible substantive pool before stability.')
    params={k:selected[k] for k in ['C','l1_ratio']};records=[];importance=[];performance=[]
    for rep in range(CFG['training_refits']):
        indices=pitcher_bootstrap_indices(train,CFG['seed']+rep)
        draw=train.iloc[indices].copy()
        obj,score=fit_named(draw,valid,core,f'bootstrap_{rep:02d}',elastic=params)
        coefs=source_coefficients(obj);coefs['repeat']=rep
        coefs['rank']=coefs.coefficient.abs().rank(method='average',ascending=False)
        coefs['top_ranked']=coefs['rank'].le(max(1,int(np.ceil(len(core)*CFG['top_rank_fraction']))))
        coefs['positive_sign']=coefs.coefficient.gt(CFG['coefficient_nonzero_tolerance'])
        coefs['negative_sign']=coefs.coefficient.lt(-CFG['coefficient_nonzero_tolerance'])
        records.append(coefs)
        performance.append(dict(repeat=rep,unique_training_pitchers=draw.pitcher_id.nunique(),
            distinct_positive_training_events=draw.loc[draw.target.eq(1),'next_event_id'].nunique(),
            train_rows=len(draw),**summary_metrics(valid,score,events)))
        with threadpool_limits(limits=1):
            imp=grouped_permutation(obj,valid,{c:[c] for c in core},repetitions=1).rename(columns={'group':'feature','repeat':'permutation_repeat'})
        imp['training_repeat']=rep;importance.append(imp)
        table(pd.concat(records,ignore_index=True),'elastic_net_refit_coefficients')
        table(pd.concat(importance,ignore_index=True),'feature_refit_permutation')
        table(pd.DataFrame(performance),'training_refit_performance')
        print(f'Whole-pitcher stability {rep+1}/{CFG["training_refits"]}: {int(coefs.selected.sum())} measured fields selected.',flush=True)
    raw=pd.concat(records,ignore_index=True)
    summary=raw.groupby('feature').agg(selection_frequency=('selected','mean'),
        selection_with_indicator_frequency=('selected_including_indicator','mean'),top_rank_frequency=('top_ranked','mean'),
        positive_frequency=('positive_sign','mean'),negative_frequency=('negative_sign','mean'),
        mean_coefficient=('coefficient','mean'),coefficient_sd=('coefficient','std'),
        mean_rank=('rank','mean'),rank_sd=('rank','std'),median_absolute_coefficient=('coefficient',lambda s:s.abs().median()),
        mean_absolute_indicator_coefficient=('indicator_coefficient',lambda s:s.abs().mean())).reset_index()
    summary['conditional_sign_stability']=summary[['positive_frequency','negative_frequency']].max(axis=1)/summary.selection_frequency.replace(0,np.nan)
    imp=pd.concat(importance,ignore_index=True).groupby('feature').ap_drop.agg(
        permutation_mean='mean',permutation_sd='std',permutation_positive_frequency=lambda x:x.gt(0).mean()).reset_index()
    summary=summary.merge(imp,on='feature',validate='one_to_one')
    table(summary.sort_values(['selection_frequency','median_absolute_coefficient'],ascending=False),'feature_stability')
    clusters=read_table('correlation_clusters')
    joined=raw.merge(clusters[['feature','correlation_cluster','signal_interpretation']],on='feature',validate='many_to_one')
    per=joined.groupby(['correlation_cluster','signal_interpretation','repeat']).selected.max().reset_index()
    frequency=per.groupby(['correlation_cluster','signal_interpretation']).selected.mean().rename('cluster_selection_frequency').reset_index()
    table(frequency,'cluster_selection_frequency')
    return summary
