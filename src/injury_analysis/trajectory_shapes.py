"""Paired, balanced trajectories and event-level shape summaries without smoothing."""
import numpy as np
import pandas as pd
from .common import ROOT, CACHE as OLD_CACHE
from .signals_common import CFG,CACHE,OUT,TRAJECTORY_FEATURES,BASELINE_COUNTS,table,cluster_draws,mean_ci


def expand_alignment(a,events,matches):
    """Reuse precisely the existing matched pairs and outing keys; never rematch controls."""
    features=list(TRAJECTORY_FEATURES)
    selection=[dict(feature=f,basis='predeclared existing baseball/trajectory measure') for f in features]
    path=OUT/'tables/variable_permutation_summary.csv'
    if path.exists():
        importance=pd.read_csv(path)
        for category,g in importance.groupby('category'):
            for feature in g.loc[~g.family.eq('pitcher_history')].nlargest(3,'mean').feature:
                if feature not in features and feature in a:
                    features.append(feature)
                    selection.append(dict(feature=feature,basis='exploratory addition from validation variable inspection: '+category))
    old=pd.read_parquet(OLD_CACHE/'aligned_trajectories.parquet')
    keys=['event_id','analysis_group','cluster_pitcher_id','pitcher_id','game_id','game_date','is_starter',
          'lead_days','appearance_before','kind']
    f=old.loc[old.event_id.isin(matches.event_id),keys].copy()
    cols=features+list(BASELINE_COUNTS.values())+['prior_appearance_count','prior_reported_il_count']
    f=f.merge(a[['pitcher_id','game_id']+cols],on=['pitcher_id','game_id'],how='left',validate='many_to_one')
    meta=matches[['event_id','anchor_date','control_anchor_date','season','is_starter']].rename(columns={'is_starter':'index_is_starter'})
    f=f.merge(meta,on='event_id',validate='many_to_one')
    em=events[['injury_event_id','strict_qa_eligible','throwing_arm_injury','body_group']]
    f=f.merge(em,left_on='event_id',right_on='injury_event_id',validate='many_to_one')
    anchor=f.anchor_date.where(f.kind.eq('Injury'),f.control_anchor_date)
    assert f.game_date.lt(anchor).all() and f.lead_days.between(1,60).all()
    assert not f[['event_id','kind','game_id','pitcher_id']].duplicated().any()
    last=f.loc[f.kind.eq('Injury')].sort_values('appearance_before').drop_duplicates('event_id')
    f['prior_reported_episodes_at_index']=f.event_id.map(last.set_index('event_id').prior_reported_il_count)
    f.to_parquet(CACHE/'expanded_alignment.parquet',index=False)
    table(pd.DataFrame(selection),'trajectory_feature_selection')
    table(f.groupby(['analysis_group','kind']).agg(appearances=('game_id','size'),events=('event_id','nunique'),
        pitchers=('pitcher_id','nunique')).reset_index(),'trajectory_cohort_counts')
    return f,features


def paired_cells(aligned,feature,mode='calendar',minimum_days=1):
    f=aligned.loc[aligned.lead_days.ge(minimum_days)].copy()
    if mode=='calendar':
        f['point']=pd.cut(f.lead_days,CFG['calendar_bins'],labels=[3,2,1,0],right=False).astype(int)
        # Chronological order: 0=31–60d, 1=15–30d, 2=8–14d, 3=1–7d.
    else:
        f=f.loc[f.appearance_before.le(CFG['last_appearances'])]
        f['point']=CFG['last_appearances']-f.appearance_before
    cells=f.groupby(['event_id','analysis_group','cluster_pitcher_id','kind','point'])[feature].mean().unstack('kind')
    if not {'Injury','Control'}<=set(cells.columns):return pd.DataFrame()
    cells=cells.dropna(subset=['Injury','Control']).reset_index()
    cells['difference']=cells.Injury-cells.Control
    return cells


def simultaneous_summary(cells,points):
    """Balanced-event max-t bootstrap bands across time points for this feature/group."""
    matrices={name:cells.pivot(index='event_id',columns='point',values=name).reindex(columns=points)
              for name in ['Injury','Control','difference']}
    valid=matrices['difference'].notna().all(axis=1)
    matrices={k:v.loc[valid] for k,v in matrices.items()}
    if not valid.sum():return pd.DataFrame(),set()
    ids=cells.drop_duplicates('event_id').set_index('event_id').cluster_pitcher_id.reindex(matrices['difference'].index)
    draws,inv=cluster_draws(ids.to_numpy())
    weights=draws[:,inv]; denominator=weights.sum(axis=1)[:,None]
    rows=[]
    for name,matrix in matrices.items():
        vals=matrix.to_numpy(dtype=float);mean=vals.mean(axis=0)
        boot=(weights@vals)/denominator
        sd=boot.std(axis=0,ddof=1)
        standardized=np.divide(boot-mean,sd,out=np.zeros_like(boot),where=sd>1e-12)
        q=np.quantile(np.abs(standardized).max(axis=1),.95)
        for j,point in enumerate(points):
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            rows.append(dict(kind=name,point=point,mean=mean[j],low=lo,high=hi,
                simultaneous_low=mean[j]-q*sd[j],simultaneous_high=mean[j]+q*sd[j],
                events=len(vals),pitchers=ids.nunique()))
    return pd.DataFrame(rows),set(matrices['difference'].index)


def sustained_support(summary,minimum_events=30):
    """First of two consecutive same-direction supported points; never biological onset."""
    s=summary.sort_values('point')
    sign=np.where(s.simultaneous_low.gt(0),1,np.where(s.simultaneous_high.lt(0),-1,0))
    if s.empty or s.events.min()<minimum_events:return None,0
    for i in range(len(sign)-1):
        if sign[i]!=0 and sign[i]==sign[i+1] and s.point.iloc[i+1]==s.point.iloc[i]+1:
            return int(s.point.iloc[i]),int(sign[i])
    return None,0


def run_balanced_trajectories(aligned,features):
    records=[];unbalanced=[];support=[];timing=[]
    for feature in features:
        for mode in ['calendar','appearance']:
            points=list(range(4 if mode=='calendar' else CFG['last_appearances']))
            cells=paired_cells(aligned,feature,mode)
            for category,g in cells.groupby('analysis_group'):
                summary,ids=simultaneous_summary(g,points)
                if len(summary):
                    summary=summary.assign(feature=feature,mode=mode,category=category)
                    records.append(summary)
                    first,sign=sustained_support(summary.loc[summary.kind.eq('difference')],CFG['minimum_events_for_timing'])
                    diff=summary.loc[summary.kind.eq('difference')]
                    supported=diff.loc[(diff.simultaneous_low.gt(0)|diff.simultaneous_high.lt(0))
                        &diff.events.ge(CFG['minimum_events_for_timing']),'point'].astype(int).tolist()
                    timing.append(dict(feature=feature,mode=mode,category=category,balanced_events=len(ids),
                        first_sustained_point=first,direction=sign,
                        individually_supported_points=';'.join(map(str,supported)),
                        earliest_single_supported_point=min(supported) if supported else None,
                        already_present_at_earliest_observed_point=first==0 if first is not None else False,
                        criterion='two adjacent same-sign simultaneous bands exclude zero; >=30 events; not a clinical onset estimate'))
                    support.extend(dict(feature=feature,mode=mode,category=category,event_id=i) for i in sorted(ids))
                for point,h in g.groupby('point'):
                    for kind in ['Injury','Control','difference']:
                        unbalanced.append(dict(feature=feature,mode=mode,category=category,point=point,kind=kind,
                            **mean_ci(h,kind,cluster='cluster_pitcher_id')))
    table(pd.concat(records,ignore_index=True),'balanced_trajectory_summary')
    table(pd.DataFrame(unbalanced),'available_pair_trajectory_summary')
    table(pd.DataFrame(timing),'trajectory_supported_timing')
    pd.DataFrame(support).to_parquet(CACHE/'balanced_event_membership.parquet',index=False)
    support_frame=pd.DataFrame(support)
    members=support_frame.loc[support_frame.feature.eq('velo_z')&support_frame['mode'].eq('appearance'),'event_id']
    leads=aligned.loc[aligned.kind.eq('Injury')&aligned.event_id.isin(members)&aligned.appearance_before.le(5)]
    table(leads.groupby(['analysis_group','appearance_before']).lead_days.agg(
        events='count',median='median',minimum='min',maximum='max',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index(),
        'velocity_appearance_calendar_support')
    print('Balanced calendar/appearance trajectories and simultaneous bands complete.',flush=True)
    return pd.concat(records,ignore_index=True)


def slope(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
    if len(x)<3 or np.ptp(x)==0:return np.nan
    return float(np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2))


def shape_metrics(values):
    """Five ordered outings, oldest to nearest; fixed contrasts, not fitted breakpoints."""
    v=np.asarray(values,dtype=float)
    if len(v)!=5 or not np.isfinite(v).all():raise ValueError('Need five finite ordered outings')
    return {'slope_per_appearance':slope(np.arange(5),v),
        'late_minus_early_mean':v[-2:].mean()-v[:2].mean(),
        'final_step_vs_previous_two':v[-1]-v[-3:-1].mean(),
        'late_minus_early_slope':((v[4]-v[2])-(v[2]-v[0]))/2,
        'absolute_z_ge_1_fraction':np.mean(np.abs(v)>=CFG['absolute_z_reference']),
        'maximum_absolute_departure':np.abs(v).max()}


def run_shapes(aligned,features):
    records=[]
    for feature in features:
        cells=paired_cells(aligned,feature,'appearance')
        for (event_id,category,pid),g in cells.groupby(['event_id','analysis_group','cluster_pitcher_id']):
            g=g.set_index('point').reindex(range(5))
            if g[['Injury','Control']].isna().any().any():continue
            case=shape_metrics(g.Injury);control=shape_metrics(g.Control)
            for metric in case:
                if metric in ['absolute_z_ge_1_fraction','maximum_absolute_departure'] and not feature.endswith('_z'):continue
                records.append(dict(event_id=event_id,category=category,pitcher_id=pid,feature=feature,metric=metric,
                    case=case[metric],control=control[metric],difference=case[metric]-control[metric]))
        # Actual calendar dates; no interpolation or equal-spacing assumption.
        for (event_id,category,pid),g in aligned.groupby(['event_id','analysis_group','cluster_pitcher_id']):
            vals={}
            for kind,h in g.groupby('kind'):
                h=h.dropna(subset=[feature])
                if len(h)<3 or h.lead_days.max()<30 or h.lead_days.min()>14 or h.lead_days.nunique()<3:continue
                vals[kind]=slope(-h.lead_days.to_numpy()/10,h[feature].to_numpy())
            if {'Injury','Control'}<=set(vals):
                records.append(dict(event_id=event_id,category=category,pitcher_id=pid,feature=feature,metric='slope_per_10_days',
                    case=vals['Injury'],control=vals['Control'],difference=vals['Injury']-vals['Control']))
    raw=pd.DataFrame(records);raw.to_parquet(CACHE/'event_shape_metrics.parquet',index=False)
    summaries=[]
    for (category,feature,metric),g in raw.groupby(['category','feature','metric']):
        summaries.append(dict(category=category,feature=feature,metric=metric,**mean_ci(g,'difference'),
            median=g.difference.median(),q10=g.difference.quantile(.1),q90=g.difference.quantile(.9),
            positive_fraction=g.difference.gt(0).mean()))
    return table(pd.DataFrame(summaries),'trajectory_shape_metrics')


def sensitivity_filter(aligned,condition,feature):
    f=aligned
    if condition=='Strict QA':f=f.loc[f.strict_qa_eligible]
    elif condition=='Exclude final 3 days':f=f.loc[f.lead_days.ge(4)]
    elif condition=='Exclude final 7 days':f=f.loc[f.lead_days.ge(8)]
    elif condition=='Starter index':f=f.loc[f.index_is_starter]
    elif condition=='Reliever index':f=f.loc[~f.index_is_starter]
    elif condition=='First reported episode':f=f.loc[f.prior_reported_episodes_at_index.eq(0)]
    elif condition=='Prior reported episode':f=f.loc[f.prior_reported_episodes_at_index.gt(0)]
    elif condition=='Stronger baseline':
        count=BASELINE_COUNTS.get(feature,'prior_appearance_count')
        f=f.loc[f[count].ge(CFG['strong_baseline_minimum'])]
    elif condition=='Explicit throwing side':
        f=f.loc[~f.body_group.eq('Arm') | f.throwing_arm_injury.fillna(False)]
    elif condition=='Complete years through 2025':f=f.loc[f.season.le(2025)]
    elif condition=='Validation years 2022–2023':f=f.loc[f.season.between(2022,2023)]
    elif condition=='Test years 2024–2025':f=f.loc[f.season.between(2024,2025)]
    elif condition!='Primary':raise ValueError(condition)
    return f


def run_trajectory_sensitivity(aligned):
    conditions=['Primary','Strict QA','Exclude final 3 days','Exclude final 7 days','Starter index','Reliever index',
        'First reported episode','Prior reported episode','Stronger baseline','Explicit throwing side','Complete years through 2025',
        'Validation years 2022–2023','Test years 2024–2025']
    features=['velo_z','fb_velo_sd_change_vs_prev_5','release_z_sd_z','extension_z','workload_acceleration_7d_vs_21d']
    rows=[];paired=[]
    for feature in features:
        for condition in conditions:
            f=sensitivity_filter(aligned,condition,feature)
            f=f.loc[f.lead_days.le(14)]
            means=f.groupby(['event_id','analysis_group','cluster_pitcher_id','kind'])[feature].mean().unstack('kind')
            means=means.dropna(subset=['Injury','Control']).reset_index()
            means['difference']=means.Injury-means.Control
            paired.append(means.assign(feature=feature,condition=condition))
            for category,g in means.groupby('analysis_group'):
                rows.append(dict(feature=feature,condition=condition,category=category,
                    **mean_ci(g,'difference',cluster='cluster_pitcher_id')))
    table(pd.DataFrame(rows),'trajectory_sensitivity')
    allpairs=pd.concat(paired,ignore_index=True)
    # Removing final days changes the eligible pairs. Show the primary effect on that
    # SAME subset separately, rather than attributing attrition to a temporal change.
    contrasts=[]
    for feature in features:
        primary=allpairs.loc[allpairs.feature.eq(feature)&allpairs.condition.eq('Primary')]
        for condition in ['Exclude final 3 days','Exclude final 7 days']:
            restricted=allpairs.loc[allpairs.feature.eq(feature)&allpairs.condition.eq(condition)]
            joined=restricted.merge(primary,on=['event_id','analysis_group','cluster_pitcher_id'],suffixes=('_restricted','_primary'))
            joined['change']=joined.difference_restricted-joined.difference_primary
            for category,g in joined.groupby('analysis_group'):
                contrasts.append(dict(feature=feature,condition=condition,category=category,
                    primary_on_same_pairs=g.difference_primary.mean(),restricted_mean=g.difference_restricted.mean(),
                    **mean_ci(g,'change',cluster='cluster_pitcher_id')))
    table(pd.DataFrame(contrasts),'trajectory_sensitivity_same_pair_contrasts')
    # Direct body-group comparison; compare changes, not whether separate intervals overlap.
    body=aligned.drop_duplicates('event_id')[['event_id','body_group','throwing_arm_injury']]
    p=allpairs.loc[allpairs.condition.eq('Primary')].merge(body,on='event_id')
    p=p.loc[p.body_group.eq('Non-arm') | p.throwing_arm_injury.fillna(False)]
    contrasts=[]
    for feature,g in p.groupby('feature'):
        draws,inv=cluster_draws(g.cluster_pitcher_id.to_numpy());w=draws[:,inv]
        arm=g.body_group.eq('Arm').to_numpy();non=g.body_group.eq('Non-arm').to_numpy();x=g.difference.to_numpy()
        vals=(w[:,arm]@x[arm])/w[:,arm].sum(axis=1)-(w[:,non]@x[non])/w[:,non].sum(axis=1)
        lo,hi=np.quantile(vals,[.025,.975])
        contrasts.append(dict(feature=feature,arm_events=int(arm.sum()),nonarm_events=int(non.sum()),
            arm_mean=x[arm].mean(),nonarm_mean=x[non].mean(),arm_minus_nonarm=x[arm].mean()-x[non].mean(),low=lo,high=hi))
    table(pd.DataFrame(contrasts),'arm_nonarm_direct_contrasts')
    return pd.DataFrame(rows)
