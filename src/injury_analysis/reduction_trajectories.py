"""Frozen training PCA maps applied to the existing matched injury histories."""
import numpy as np
import pandas as pd
from .common import CACHE as ORIGINAL_CACHE, CFG as ORIGINAL_CFG
from .signals_common import mean_ci
from .trajectory_shapes import paired_cells, simultaneous_summary, shape_metrics
from .reduction import CFG, CACHE, table


def latent_alignment(events, matches):
    scores = pd.read_parquet(CACHE/'appearance_scores.parquet')
    old = pd.read_parquet(ORIGINAL_CACHE/'aligned_trajectories.parquet')
    keys = ['event_id','analysis_group','cluster_pitcher_id','pitcher_id','game_id','game_date',
            'lead_days','appearance_before','kind']
    f = old.loc[old.event_id.isin(matches.event_id),keys].copy()
    cols = [c for c in scores if c.startswith(('profile_','departure_'))]+['throws_left']
    f = f.merge(scores[['pitcher_id','game_id']+cols],on=['pitcher_id','game_id'],validate='many_to_one')
    f = f.merge(matches[['event_id','anchor_date','control_anchor_date','is_starter']],on='event_id',validate='many_to_one')
    f = f.merge(events[['injury_event_id','strict_qa_eligible']],left_on='event_id',right_on='injury_event_id',validate='many_to_one')
    assert set(f.event_id)==set(matches.event_id)
    assert not f[['event_id','kind','pitcher_id','game_id']].duplicated().any()
    assert f.game_date.lt(f.anchor_date.where(f.kind.eq('Injury'),f.control_anchor_date)).all()
    # Group comparisons never turn median-imputed missing measurements into a
    # purported physiological trajectory. Require all eight inputs per block.
    features=[]
    for prefix in ['profile','departure']:
        for j in range(1,CFG['plotted_components']+1):
            feature=f'{prefix}_PC{j}'
            f.loc[~f[prefix+'_complete'],feature]=np.nan
            features.append(feature)
    last=f.sort_values('appearance_before').drop_duplicates(['event_id','kind'])
    hands=last.pivot(index='event_id',columns='kind',values='throws_left')
    same=hands.Injury.notna()&hands.Control.notna()&hands.Injury.eq(hands.Control)
    f['same_throwing_hand']=f.event_id.map(same)
    start=pd.Timestamp(ORIGINAL_CFG['train_end'])+pd.Timedelta(days=1)
    end=pd.Timestamp(ORIGINAL_CFG['test_end'])
    # Keep complete pairs/windows whose observations are all out of PCA training.
    eligible=f.assign(in_period=f.game_date.between(start,end)).groupby('event_id').in_period.all()
    anchors=matches.set_index('event_id').anchor_date.between(start,end)
    eligible &= anchors.reindex(eligible.index)
    f['heldout_2022_2025']=f.event_id.map(eligible)
    f.to_parquet(CACHE/'latent_alignment.parquet',index=False)
    table(f.groupby(['analysis_group','kind']).agg(events=('event_id','nunique'),appearances=('game_id','size'),
        profile_complete_fraction=('profile_complete','mean'),departure_complete_fraction=('departure_complete','mean')).reset_index(),
        'latent_alignment_coverage')
    return f,features


def run_latent_trajectories(events,matches):
    f,features=latent_alignment(events,matches)
    summaries,unbalanced,memberships,shapes,centroids=[],[],[],[],[]
    for cohort,selected in [('Heldout 2022–2025',f.loc[f.heldout_2022_2025]),('All years descriptive',f)]:
        print(f'Latent trajectories: {cohort}, {selected.event_id.nunique()} original pairs',flush=True)
        for feature in features:
            for mode in ['calendar','appearance']:
                points=range(4 if mode=='calendar' else 5)
                cells=paired_cells(selected,feature,mode)
                for category,g in cells.groupby('analysis_group'):
                    summary,ids=simultaneous_summary(g,list(points))
                    if summary.empty:
                        continue
                    summaries.append(summary.assign(cohort=cohort,feature=feature,mode=mode,category=category))
                    memberships.extend(dict(cohort=cohort,feature=feature,mode=mode,category=category,event_id=i) for i in sorted(ids))
                    balanced=g.loc[g.event_id.isin(ids)]
                    centroids.append(balanced.assign(cohort=cohort,feature=feature,mode=mode))
                    for point,h in g.groupby('point'):
                        unbalanced.append(dict(cohort=cohort,feature=feature,mode=mode,category=category,point=point,
                            **mean_ci(h,'difference',cluster='cluster_pitcher_id')))
                    # Fixed paired event summaries expose persistent separation vs
                    # changing separation without assuming individual independence.
                    event_stats=[]
                    for event,h in balanced.groupby('event_id'):
                        h=h.sort_values('point')
                        difference=h.difference.to_numpy()
                        values={'average_separation':difference.mean(),
                                'last_minus_first_separation':difference[-1]-difference[0]}
                        if mode=='appearance':
                            case,control=shape_metrics(h.Injury),shape_metrics(h.Control)
                            for metric in ['slope_per_appearance','late_minus_early_mean','final_step_vs_previous_two']:
                                values[metric]=case[metric]-control[metric]
                        for metric,value in values.items():
                            event_stats.append(dict(event_id=event,pitcher_id=h.cluster_pitcher_id.iloc[0],metric=metric,value=value))
                    stats=pd.DataFrame(event_stats)
                    for metric,h in stats.groupby('metric'):
                        shapes.append(dict(cohort=cohort,feature=feature,mode=mode,category=category,metric=metric,
                            **mean_ci(h,'value')))
    table(pd.concat(summaries,ignore_index=True),'latent_balanced_trajectories')
    table(pd.DataFrame(unbalanced),'latent_available_trajectories')
    table(pd.DataFrame(shapes),'latent_shape_contrasts')
    pd.DataFrame(memberships).to_parquet(CACHE/'latent_balanced_membership.parquet',index=False)
    pd.concat(centroids,ignore_index=True).to_parquet(CACHE/'latent_event_cells.parquet',index=False)
    # The primary map is fixed; these restrictions do not refit it or reindex time.
    rows=[]
    held=f.loc[f.heldout_2022_2025]
    conditions={'Primary':pd.Series(True,index=held.index),
                'Same throwing hand':held.same_throwing_hand,
                'Strict QA':held.strict_qa_eligible,
                'Starter index':held.is_starter,
                'Reliever index':~held.is_starter,
                'Exclude final seven days':held.lead_days.ge(8)}
    for condition,mask in conditions.items():
        subset=held.loc[mask&held.lead_days.le(14)]
        for feature in features:
            means=subset.groupby(['event_id','analysis_group','cluster_pitcher_id','kind'])[feature].mean().unstack('kind')
            means=means.dropna(subset=['Injury','Control']).reset_index()
            means['difference']=means.Injury-means.Control
            for category,g in means.groupby('analysis_group'):
                rows.append(dict(condition=condition,feature=feature,category=category,
                    **mean_ci(g,'difference',cluster='cluster_pitcher_id')))
    table(pd.DataFrame(rows),'latent_sensitivity')
    # Direct comparison is paired with the same events for far and near bins.
    # Scatterplot data remains one event per point, never thousands of correlated outings.
    return pd.concat(summaries,ignore_index=True)
