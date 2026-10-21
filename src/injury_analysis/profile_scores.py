"""Interpret the combined predictor as additive profile/departure log-odds terms."""
import joblib
import numpy as np
import pandas as pd
from .profile_departure import (CFG,CACHE,PROFILE,DEPARTURE,ORIGINAL_CACHE,ORIGINAL_CFG,
    table,clean,label_frame)
from .reduction import between_pitcher_fraction
from .trajectory_shapes import paired_cells,simultaneous_summary
from .signals_common import mean_ci


def score_contributions(encoder,model,frame):
    """Exact linear terms, with each missing flag assigned to its source block."""
    if encoder.pca is not None or encoder.rotate_index:
        raise ValueError('Exact block interpretation requires the unreduced combined model')
    names=encoder.names
    source=[n.removeprefix('missingindicator_') for n in names]
    if set(source)-set(PROFILE+DEPARTURE):raise ValueError('Unknown predictor in block decomposition')
    z=encoder.standardized(frame);coef=encoder.effective_coefficients(model)
    values={}
    for name,cols in [('profile_score',PROFILE),('departure_score',DEPARTURE)]:
        ix=[j for j,c in enumerate(source) if c in cols]
        values[name]=z[:,ix]@coef[ix]
    result=pd.DataFrame(values,index=frame.index)
    result['intercept']=float(model.intercept_[0])
    result['logit']=result.profile_score+result.departure_score+result.intercept
    np.testing.assert_allclose(result.logit,model.decision_function(encoder.transform(frame)),atol=1e-10)
    return result


def run_score_analysis(a,events,matches):
    old=pd.read_parquet(ORIGINAL_CACHE/'aligned_trajectories.parquet')
    keys=['event_id','analysis_group','cluster_pitcher_id','pitcher_id','game_id','game_date',
          'lead_days','appearance_before','kind']
    aligned=old.loc[old.event_id.isin(matches.event_id),keys].copy()
    aligned=aligned.merge(a[['pitcher_id','game_id']+PROFILE+DEPARTURE],on=['pitcher_id','game_id'],validate='many_to_one')
    aligned=aligned.merge(matches[['event_id','anchor_date','control_anchor_date']],on='event_id',validate='many_to_one')
    start=pd.Timestamp(ORIGINAL_CFG['train_end'])+pd.Timedelta(days=1)
    end=pd.Timestamp(ORIGINAL_CFG['test_end'])
    valid=aligned.assign(in_period=aligned.game_date.between(start,end)).groupby('event_id').in_period.all()
    valid &= matches.set_index('event_id').anchor_date.between(start,end).reindex(valid.index)
    aligned=aligned.loc[aligned.event_id.map(valid)].copy()
    assert aligned.game_date.lt(aligned.anchor_date.where(aligned.kind.eq('Injury'),aligned.control_anchor_date)).all()
    assert not aligned[['event_id','kind','pitcher_id','game_id']].duplicated().any()
    aligned['complete']=clean(aligned,PROFILE+DEPARTURE).notna().all(axis=1)
    persistence=[];scored=[];coverage=[];max_error=0.
    for category,g in aligned.groupby('analysis_group'):
        path=CACHE/f"models_{category.replace('/','-').replace(' ','_')}.joblib"
        if not path.exists():continue
        encoder,model=joblib.load(path)['Profile + Departure']
        contributions=score_contributions(encoder,model,g)
        max_error=max(max_error,float(np.max(np.abs(contributions.logit-model.decision_function(encoder.transform(g))))))
        for col in contributions:g[col]=contributions[col]
        # Common complete rows keep missingness and changing pair composition out
        # of the principal matched curves; predictor evaluation retains all rows.
        g.loc[~g.complete,['profile_score','departure_score']]=np.nan
        scored.append(g)
        coverage.append(dict(category=category,pairs=g.event_id.nunique(),appearance_rows=len(g),
                             complete_rows=int(g.complete.sum())))
        risk=label_frame(a,category,30)
        for part in ['train','validation','test']:
            f=risk.loc[risk.partition.eq(part)]
            terms=score_contributions(encoder,model,f)
            covariance=np.cov(terms.profile_score,terms.departure_score,ddof=0)[0,1]
            for block in ['profile_score','departure_score']:
                persistence.append(dict(category=category,partition=part,block=block,rows=len(f),pitchers=f.pitcher_id.nunique(),
                    between_pitcher_fraction=between_pitcher_fraction(terms[block],f.pitcher_id),
                    variance=terms[block].var(ddof=0),profile_departure_covariance=covariance,
                    total_logit_variance=terms.logit.var(ddof=0)))
    f=pd.concat(scored,ignore_index=True)
    table(pd.DataFrame(coverage),'score_alignment_coverage')
    table(pd.DataFrame(persistence),'score_variation')
    table(pd.DataFrame([dict(check='Exact combined linear predictor reconstruction',max_absolute_error=max_error)]),
          'contribution_identity')
    summaries=[];shapes=[];members=[];point_support=[]
    for feature in ['profile_score','departure_score']:
        for mode in ['calendar','appearance']:
            points=list(range(4 if mode=='calendar' else 5))
            cells=paired_cells(f,feature,mode)
            for category,g in cells.groupby('analysis_group'):
                summary,ids=simultaneous_summary(g,points)
                if summary.empty:continue
                summaries.append(summary.assign(feature=feature,mode=mode,category=category))
                members.extend(dict(feature=feature,mode=mode,category=category,event_id=i) for i in ids)
                records=[]
                for event,h in g.loc[g.event_id.isin(ids)].groupby('event_id'):
                    d=h.sort_values('point').difference.to_numpy()
                    values={'average_separation':d.mean(),'last_minus_first_separation':d[-1]-d[0],
                            'last_minus_previous_two':d[-1]-d[-3:-1].mean()}
                    for metric,value in values.items():
                        records.append(dict(event_id=event,pitcher_id=h.cluster_pitcher_id.iloc[0],metric=metric,value=value))
                for metric,h in pd.DataFrame(records).groupby('metric'):
                    shapes.append(dict(feature=feature,mode=mode,category=category,metric=metric,**mean_ci(h,'value')))
                if mode=='appearance' and feature=='profile_score':
                    source=f.loc[f.event_id.isin(ids)&f.kind.eq('Injury')&f.appearance_before.le(5)]
                    for point,h in source.groupby('appearance_before'):
                        point_support.append(dict(category=category,appearance_before=point,events=h.event_id.nunique(),
                            median_lead_days=h.lead_days.median(),q25=h.lead_days.quantile(.25),q75=h.lead_days.quantile(.75)))
    table(pd.concat(summaries,ignore_index=True),'score_trajectories')
    table(pd.DataFrame(shapes),'score_shape_contrasts')
    table(pd.DataFrame(point_support),'score_appearance_lead_days')
    pd.DataFrame(members).to_parquet(CACHE/'score_balanced_membership.parquet',index=False)
    f.to_parquet(CACHE/'matched_score_alignment.parquet',index=False)
    print(f'Supervised score trajectories: {f.event_id.nunique()} original out-of-training pairs.',flush=True)
    return pd.concat(summaries,ignore_index=True)


def run_alert_diagnostics():
    """Show repeated-alert concentration; this is descriptive, not a new policy."""
    from src.evaluation import alert_mask
    f=pd.read_parquet(CACHE/'predictions.parquet')
    records=[]
    for (category,part),g in f.groupby(['category','partition']):
        for name in ['Profile only','Departure only','Profile + Departure','Context + Profile',
                     'Context + Profile + Departure','Workload + history','Original 62']:
            selected=g.loc[alert_mask(g,'score__'+name,.05)]
            counts=selected.groupby('pitcher_id').size().sort_values(ascending=False)
            records.append(dict(category=category,partition=part,variant=name,alerts=len(selected),
                eligible_pitchers=g.pitcher_id.nunique(),alerted_pitchers=len(counts),
                alerts_per_alerted_pitcher=len(selected)/len(counts),
                fraction_alerts_top_ten_pitchers=counts.head(10).sum()/len(selected),
                median_alerted_outings_per_pitcher=counts.median()))
    return table(pd.DataFrame(records),'alert_concentration')

