"""Train-only pairwise dependence, semantic guarded clusters, and readable summaries."""
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster,leaves_list
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits
from .consolidation_common import CFG,CACHE,table,read_table,inputs


def correlation_sample(train):
    """Cap each pitcher's contribution without consulting outcomes; preserve keys."""
    rng=np.random.default_rng(CFG['seed'])
    parts=[]
    for _,g in train.sort_values(['pitcher_id','game_date','game_id']).groupby('pitcher_id',sort=True):
        idx=rng.choice(len(g),min(len(g),CFG['correlation_appearances_per_pitcher']),replace=False)
        parts.append(g.iloc[np.sort(idx)])
    return pd.concat(parts).sort_values('row_id')


def cluster_coordinates(correlation,minimum=.90):
    """Complete linkage constrains every within-cluster |Spearman| to the cut."""
    if len(correlation)==1:return np.array([1]),np.array([0]),np.empty((0,4))
    distance=1-correlation.abs().fillna(0).clip(0,1).to_numpy()
    distance=(distance+distance.T)/2;np.fill_diagonal(distance,0)
    tree=linkage(squareform(distance,checks=True),method='complete',optimal_ordering=True)
    return fcluster(tree,t=1-minimum,criterion='distance'),leaves_list(tree),tree


def representative_order(row):
    # Baseline and five-outing summaries are interpretable defaults. This ordering
    # is independent of any validation/test outcome, coefficient or AP.
    preference={'baseline mean':0,'pitcher-relative z-score':0,'current level':0,
                'rolling historical level':1,'change from recent history':1,'baseline sd':1}
    return (bool(row.pitch_type_specific),preference.get(row.representation,2),
            0 if '_prev_5' in row.feature else 1,float(row.training_missing_fraction),row.feature)


def run_redundancy():
    frame,_,_=inputs();train=frame.loc[frame.partition.eq('train')]
    inventory=read_table('feature_inventory');allowed=inventory.loc[inventory.eligible_predictor].copy()
    sample=correlation_sample(train)
    table(sample[['row_id','pitcher_id','game_id','game_date']],'correlation_sample_keys')
    names=allowed.feature.tolist();x=sample[names].astype(float).replace([np.inf,-np.inf],np.nan)
    print(f'Pairwise Pearson/Spearman: {len(names)} columns; {len(x):,} train-only pitcher-capped rows.',flush=True)
    with threadpool_limits(limits=1):
        observed=x.notna().to_numpy(dtype=float)
        count=pd.DataFrame(observed.T@observed,index=names,columns=names)
        pearson=x.corr(method='pearson',min_periods=CFG['minimum_correlation_pairs'])
        # pandas computes pairwise ranks on co-observed values, not globally
        # ranked columns followed by a Pearson shortcut with missing data.
        spearman=x.corr(method='spearman',min_periods=CFG['minimum_correlation_pairs'])
    for obj,name in [(pearson,'pearson'),(spearman,'spearman'),(count,'coobserved_counts')]:
        obj.to_parquet(CACHE/f'{name}.parquet')
    family=allowed.set_index('feature').feature_group.to_dict();pairs=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            p=pearson.loc[a,b];s=spearman.loc[a,b]
            if np.isfinite(p) or np.isfinite(s):
                pairs.append(dict(feature_a=a,feature_b=b,group_a=family[a],group_b=family[b],
                    pearson=p,spearman=s,coobserved_rows=int(count.loc[a,b]),same_group=family[a]==family[b],
                    high_correlation=max(abs(p) if np.isfinite(p) else 0,abs(s) if np.isfinite(s) else 0)>=CFG['high_pair_cut']))
    pairs=pd.DataFrame(pairs);pairs.to_parquet(CACHE/'all_correlation_pairs.parquet',index=False)
    high=pairs.loc[pairs.high_correlation].copy()
    table(high.sort_values('spearman',key=lambda s:s.abs(),ascending=False),'high_correlation_pairs')
    table(high.loc[~high.same_group],'cross_group_high_pairs')
    cross=[]
    canonical=pairs.copy()
    canonical['group_a']=[min(a,b) for a,b in zip(pairs.group_a,pairs.group_b)]
    canonical['group_b']=[max(a,b) for a,b in zip(pairs.group_a,pairs.group_b)]
    for (a,b),g in canonical.groupby(['group_a','group_b']):
        cross.append(dict(group_a=a,group_b=b,pairs=len(g),high_pairs=int(g.high_correlation.sum()),
            high_pair_fraction=g.high_correlation.mean(),max_abs_spearman=g.spearman.abs().max(),
            median_abs_spearman=g.spearman.abs().median()))
    table(pd.DataFrame(cross),'cross_group_summary')
    memberships=[];trees={};orders={}
    for group,g in allowed.groupby('feature_group',sort=True):
        columns=g.feature.tolist();labels,order,tree=cluster_coordinates(spearman.loc[columns,columns],CFG['correlation_cut'])
        orders[group]=[columns[i] for i in order];trees[group]=tree.tolist()
        for feature,label in zip(columns,labels):
            memberships.append(dict(feature=feature,feature_group=group,correlation_cluster=group+f'__{label:03d}'))
    clusters=pd.DataFrame(memberships).merge(allowed,on=['feature','feature_group'],validate='one_to_one')
    # Preserve distinct temporal information even if current and historical levels
    # correlate. Cluster membership is evidence, not an automatic deletion rule.
    clusters['representative']=False
    clusters['representative_feature']=''
    for _,g in clusters.groupby(['correlation_cluster','signal_interpretation']):
        rows=sorted(g.itertuples(),key=representative_order);chosen=rows[0].feature
        clusters.loc[g.index,'representative_feature']=chosen
        clusters.loc[g.index[g.feature.eq(chosen)],'representative']=True
    table(clusters[['feature','feature_group','underlying_concept','representation','signal_interpretation',
        'correlation_cluster','representative','representative_feature','training_missing_fraction']],'correlation_clusters')
    table(clusters.groupby(['feature_group','correlation_cluster']).agg(features=('feature','size'),
        representatives=('representative','sum'),members=('feature',lambda x:';'.join(x))).reset_index(),'cluster_summary')
    (CACHE/'hierarchies.json').write_text(json.dumps(dict(orders=orders,trees=trees),indent=2))
    print(f'{len(high):,} high-correlation pairs; {clusters.correlation_cluster.nunique()} within-group clusters; '
          f'{clusters.representative.sum()} representatives after protecting temporal roles.',flush=True)
    return clusters
