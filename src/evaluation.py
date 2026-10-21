"""Evaluation of supplied predictions only. No fitting, threshold tuning or imputation."""
import numpy as np
import pandas as pd


def rank_metrics(y, score):
    y=np.asarray(y);score=np.asarray(score,dtype=float)
    if y.ndim!=1 or score.ndim!=1:
        raise ValueError('Targets and scores must be one-dimensional')
    if len(y)!=len(score) or not len(y) or not np.isfinite(score).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Need aligned, finite scores and observed binary targets')
    y=y.astype(int)
    order=np.argsort(-score,kind='stable');ys=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(ys)[ends];fp=(ends+1)-tp;positive=y.sum();negative=len(y)-positive
    ap=float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/positive) if positive else np.nan
    auc=float(np.trapezoid(np.r_[0,tp]/positive,np.r_[0,fp]/negative)) if positive and negative else np.nan
    return {'average_precision':ap,'roc_auc':auc}


def alert_mask(frame, score_column, rate):
    """Top floor(rate*N) alerts. Resolve score ties using immutable game/pitcher IDs."""
    if not frame.index.is_unique:raise ValueError('Reset the row index before selecting an alert budget')
    if not 0<=rate<=1:raise ValueError('Alert rate must be between 0 and 1')
    if frame[['game_id','pitcher_id']].duplicated().any():raise ValueError('Duplicate appearance keys')
    if not np.isfinite(frame[score_column].to_numpy(dtype=float)).all():raise ValueError('Nonfinite scores')
    n=int(np.floor(rate*len(frame)))
    index=frame.sort_values([score_column,'game_id','pitcher_id'],ascending=[False,True,True]).index[:n]
    return pd.Series(frame.index.isin(index),index=frame.index)


def evaluate_predictions(frame, events, target='ucl_injury_next_30d', score='score',
                         rates=(.01,.025,.05,.10), warning_window=(7,30)):
    """Events must be the predeclared test-period event cohort, including missed events.

    Input rows must have observed targets and be from one held-out partition. This
    is a retrospective alert-budget curve: deployable thresholds must be chosen
    on validation data. Repeated alerts are counted separately; no silent cooldown.
    """
    if frame[target].isna().any():raise ValueError('Remove censored/unknown target rows explicitly')
    f=frame.copy().reset_index(drop=True);f['game_date']=pd.to_datetime(f.game_date)
    base=rank_metrics(f[target].to_numpy(),f[score]);y=f[target].astype(int).to_numpy()
    ev=events.drop_duplicates('injury_event_id').copy()
    ev['label_reference_date']=pd.to_datetime(ev.label_reference_date)
    if ev.label_reference_date.isna().any():raise ValueError('Event evaluation needs defensible reference dates')
    opportunities=f[['pitcher_id','game_id','game_date']].merge(ev[['pitcher_id','injury_event_id','label_reference_date']],on='pitcher_id')
    opportunities['lead_days']=(opportunities.label_reference_date-opportunities.game_date).dt.days
    opportunities=opportunities[opportunities.lead_days.between(*warning_window)]
    eligible_ids=set(opportunities.injury_event_id);summaries=[];lead_records=[]
    pitcher_seasons=f.assign(season=f.game_date.dt.year)[['pitcher_id','season']].drop_duplicates()
    for rate in rates:
        mask=alert_mask(f,score,rate);alerts=f[mask];tp=int(y[mask].sum());fp=int(mask.sum()-tp);fn=int(y.sum()-tp)
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else np.nan
        f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else np.nan
        hits=opportunities.merge(alerts[['pitcher_id','game_id']],on=['pitcher_id','game_id'])
        earliest=hits.groupby('injury_event_id').lead_days.max()
        for eid,lead in earliest.items():lead_records.append({'alert_rate':rate,'injury_event_id':eid,'warning_lead_days':int(lead)})
        summaries.append(dict(base,alert_rate=rate,actual_alert_rate=float(mask.mean()),rows=len(f),alerts=int(mask.sum()),
            recall=recall,precision=precision,f2=f2,qualifying_events=len(ev),observable_events=len(eligible_ids),
            alerted_events=len(earliest),injury_event_recall=len(earliest)/len(ev) if len(ev) else np.nan,
            observable_event_recall=len(earliest)/len(eligible_ids) if eligible_ids else np.nan,
            median_warning_lead_days=float(earliest.median()) if len(earliest) else np.nan,
            false_alerts_per_pitcher_season=fp/len(pitcher_seasons) if len(pitcher_seasons) else np.nan,
            non_actionable_alerts_per_pitcher_season=(len(alerts)-len(hits[['pitcher_id','game_id']].drop_duplicates()))/len(pitcher_seasons) if len(pitcher_seasons) else np.nan))
    return pd.DataFrame(summaries),pd.DataFrame(lead_records,columns=['alert_rate','injury_event_id','warning_lead_days'])


def temporal_partition(frame, train_end, validation_end, horizon_days=90):
    """Chronological partitions with boundary embargo and cross-partition event purge."""
    f=frame.copy();days=pd.to_datetime(f.game_date);train=pd.Timestamp(train_end);valid=pd.Timestamp(validation_end)
    if train>=valid:raise ValueError('Partition boundaries must be chronological')
    result=pd.Series(np.where(days<=train,'train',np.where(days<=valid,'validation','test')),index=f.index)
    for boundary in (train,valid):
        result[(days<=boundary)&(days>boundary-pd.Timedelta(days=horizon_days))]='purged'
    # An event must belong to a single retained partition, even for unusual horizons.
    if 'next_injury_event_id' in f:
        targets=[c for c in f if c.startswith('ucl_injury_next_')]
        if targets:
            related=f[targets].fillna(False).eq(True).any(axis=1)
        elif 'next_known_ucl_onset' in f:
            related=(pd.to_datetime(f.next_known_ucl_onset)-days).dt.days.between(1,horizon_days)
        else:
            related=pd.Series(True,index=f.index)
        event_partitions=f.loc[related].assign(partition=result[related]).query("partition != 'purged'").dropna(subset=['next_injury_event_id'])
        counts=event_partitions.groupby('next_injury_event_id').partition.nunique()
        spanning=set(counts[counts>1].index)
        result[related & f.next_injury_event_id.isin(spanning)]='purged'
    return result
