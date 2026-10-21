"""First-event warning evaluation; uncertain follow-up is never a healthy label."""
import numpy as np
import pandas as pd
from src.evaluation import rank_metrics,alert_mask
from .ucl_dynamic_common import CFG

def metrics(frame,score):
    y=frame.target.to_numpy(dtype=int);score=np.asarray(score)
    m=dict(rows=len(frame),pitchers=frame.pitcher_id.nunique(),positive_rows=int(y.sum()),
        represented_events=frame.loc[frame.target.eq(1),'dynamic_next_event_id'].nunique(),
        prevalence=float(y.mean()),**rank_metrics(y,score),brier_score=float(np.mean((y-score)**2)),
        mean_predicted_risk=float(score.mean()),calibration_bias=float(score.mean()-y.mean()))
    m['pr_lift']=m['average_precision']/m['prevalence'] if m['prevalence'] else np.nan
    return m

def warning(frame,score,events,rate=.05,minimum=7,maximum=30,threshold=None):
    """Credit only the next event: never jump over a competing injury.

    Denominator includes eligible events without an observed prediction opportunity.
    Top-N budgets are retrospective; a fixed validation threshold is separate.
    """
    f=frame.reset_index(drop=True).assign(score=np.asarray(score))
    f['alert']=alert_mask(f,'score',rate) if threshold is None else f.score.ge(threshold)
    eligible=(f.dynamic_next_target & f.dynamic_lead_days.between(minimum,maximum)
              & f.dynamic_next_event_id.isin(events.injury_event_id))
    hit=f.loc[eligible&f.alert]
    leads=hit.groupby('dynamic_next_event_id').dynamic_lead_days.max()
    e=events[['injury_event_id','pitcher_id','label_reference_date']].copy()
    e['opportunity_count']=e.injury_event_id.map(f.loc[eligible].dynamic_next_event_id.value_counts()).fillna(0).astype(int)
    e['warning_count']=e.injury_event_id.map(hit.dynamic_next_event_id.value_counts()).fillna(0).astype(int)
    e['first_warning_lead_days']=e.injury_event_id.map(leads);e['detected']=e.warning_count.gt(0)
    # Persistence measured only across consecutive evaluated outings, with <=30d gap.
    ordered=f.sort_values(['pitcher_id','game_date','game_id'])
    prior=ordered.groupby('pitcher_id').alert.shift(fill_value=False).astype(bool)
    gap=ordered.groupby('pitcher_id').game_date.diff().dt.days
    repeat=ordered.alert&prior&gap.le(30)
    ps=f[['pitcher_id','season']].drop_duplicates().shape[0]
    alerts=int(f.alert.sum());tp=int(f.loc[f.alert,'target'].sum());positive=int(f.target.sum())
    detected=int(e.detected.sum());fp=alerts-tp
    m=dict(alert_rate=rate,policy='retrospective_budget' if threshold is None else 'frozen_validation_threshold',
        threshold=threshold,minimum_lead=minimum,maximum_lead=maximum,rows=len(f),alerts=alerts,
        actual_alert_rate=float(f.alert.mean()),qualifying_events=len(e),observable_events=int(e.opportunity_count.gt(0).sum()),
        alerted_events=detected,injury_event_recall=detected/len(e) if len(e) else np.nan,
        observable_event_recall=detected/e.opportunity_count.gt(0).sum() if e.opportunity_count.gt(0).any() else np.nan,
        median_warning_lead_days=float(leads.median()),alerts_per_detected_event=alerts/detected if detected else np.nan,
        false_alert_fraction=fp/alerts if alerts else np.nan,
        non_actionable_alert_fraction=(alerts-len(hit))/alerts if alerts else np.nan,
        false_alerts_per_pitcher_season=fp/ps if ps else np.nan,
        non_actionable_alerts_per_pitcher_season=(alerts-len(hit))/ps if ps else np.nan,
        repeat_alert_fraction=float(repeat.sum()/alerts) if alerts else np.nan,
        precision=tp/alerts if alerts else 0.,recall=tp/positive if positive else np.nan,
        f2=5*tp/(5*tp+4*(positive-tp)+fp) if positive+alerts else np.nan)
    return m,e

def event_bootstrap(e):
    """Whole-pitcher uncertainty conditional on scores and alert cutoff."""
    g=e.groupby('pitcher_id').agg(hits=('detected','sum'),n=('detected','size'))
    if not len(g):return np.full(CFG['bootstrap_replicates'],np.nan)
    rng=np.random.default_rng(CFG['seed']);draws=rng.integers(0,len(g),(CFG['bootstrap_replicates'],len(g)))
    return g.hits.to_numpy()[draws].sum(axis=1)/g.n.to_numpy()[draws].sum(axis=1)

def selection(metrics_table,warning_table):
    """Predeclared: validation 7–30d recall at 5%, then fewer fields, then AP.

    AP must exceed validation prevalence. Test rows are explicitly rejected.
    This chooses an exploratory candidate, not a demonstrated superior model.
    """
    if not metrics_table.partition.eq('validation').all() or not warning_table.partition.eq('validation').all():
        raise ValueError('Selection accepts validation results only')
    m=metrics_table.loc[metrics_table.horizon.eq(30)].copy()
    w=warning_table.loc[warning_table.horizon.eq(30)&warning_table.minimum_lead.eq(7)&warning_table.alert_rate.eq(.05)]
    m=m.merge(w[['model','injury_event_recall']],on='model',validate='one_to_one')
    eligible=m.loc[m.average_precision>m.prevalence]
    if eligible.empty:return 'Departure'
    return eligible.sort_values(['injury_event_recall','raw_features','average_precision'],ascending=[False,True,False]).iloc[0].model
