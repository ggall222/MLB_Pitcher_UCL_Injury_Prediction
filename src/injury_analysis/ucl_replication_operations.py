"""Development-fixed cutoffs, alert episodes and diagnostic calibration only."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from .ucl_temporal_data import interval_target
from .ucl_dynamic_evaluation import warning,event_bootstrap
from .ucl_temporal_analysis import uncertainty
from .ucl_replication_common import CFG,OUT,CACHE,table,read,parquet,checkpoint,valid
from .ucl_replication_data import prepare,event_set


def alert_episodes(frame,score,threshold,gap_days=30):
    """Use all appearances, including ineligible breaks, with no outcome-based grouping."""
    g=frame.copy().assign(score=np.asarray(score))
    g=g.sort_values(['pitcher_id','game_date','game_id'])
    g['alert']=g.analysis_eligible & g.score.ge(threshold)
    prior=g.groupby('pitcher_id').alert.shift(fill_value=False)
    gap=g.groupby('pitcher_id').game_date.diff().dt.days
    season_prior=g.groupby('pitcher_id').season.shift()
    start=g.alert & (~prior|gap.gt(gap_days)|g.season.ne(season_prior))
    g['episode_id']=start.cumsum().where(g.alert).astype('Int64')
    return g


def episode_metrics(g,events,lo,hi):
    alerted=g.loc[g.alert].copy();known=alerted.target.notna()
    actionable=alerted.target.eq(1).fillna(False)&alerted.dynamic_next_event_id.isin(events.injury_event_id)
    hit=alerted.loc[actionable]
    ep=alerted.groupby('episode_id').agg(pitcher_id=('pitcher_id','first'),season=('season','first'),
        start=('game_date','min'),end=('game_date','max'),outings=('game_id','size'),known_outings=('target','count')).reset_index()
    ep['unknown_outings']=ep.outings-ep.known_outings
    ep['actionable']=ep.episode_id.isin(hit.episode_id)
    ep['fully_observed']=ep.unknown_outings.eq(0)
    leads=hit.groupby('dynamic_next_event_id').dynamic_lead_days.max()
    allop=g.loc[g.analysis_eligible&g.target.eq(1).fillna(False)&g.dynamic_next_event_id.isin(events.injury_event_id)]
    ev=events[['injury_event_id','pitcher_id','label_reference_date']].copy()
    ev['opportunity_count']=ev.injury_event_id.map(allop.dynamic_next_event_id.value_counts()).fillna(0).astype(int)
    ev['alert_outings']=ev.injury_event_id.map(hit.dynamic_next_event_id.value_counts()).fillna(0).astype(int)
    ev['alert_episodes']=ev.injury_event_id.map(hit.groupby('dynamic_next_event_id').episode_id.nunique()).fillna(0).astype(int)
    ev['earliest_warning_lead_days']=ev.injury_event_id.map(leads);ev['detected']=ev.alert_outings.gt(0)
    n=int(ev.detected.sum());observable=int(ev.opportunity_count.gt(0).sum());ps=g.loc[g.analysis_eligible,['pitcher_id','season']].drop_duplicates().shape[0]
    metrics=dict(eligible_landmarks=int(g.analysis_eligible.sum()),qualifying_events=len(ev),observable_events=observable,
        detected_events=n,event_recall=n/len(ev) if len(ev) else np.nan,observable_event_recall=n/observable if observable else np.nan,
        alerted_outings=len(alerted),alert_episodes=len(ep),pitcher_seasons=ps,alert_episodes_per_pitcher_season=len(ep)/ps if ps else np.nan,
        realized_alert_rate=len(alerted)/g.analysis_eligible.sum(),alerts_per_detected_injury=len(alerted)/n if n else np.nan,
        episodes_per_detected_injury=len(ep)/n if n else np.nan,actionable_alerts=len(hit),
        unknown_outcome_alerts=int((~known).sum()),known_outcome_alerts=int(known.sum()),
        nonactionable_known_alerts=int(known.sum())-len(hit),
        non_actionable_alert_fraction=(int(known.sum())-len(hit))/known.sum() if known.sum() else np.nan,
        unknown_alert_fraction=float((~known).mean()) if len(alerted) else np.nan,
        confirmed_nonactionable_episodes=int((ep.fully_observed&~ep.actionable).sum()),
        unresolved_episodes=int((~ep.fully_observed&~ep.actionable).sum()),
        earliest_warning_days=float(leads.max()) if len(leads) else np.nan,median_warning_lead_days=float(leads.median()) if len(leads) else np.nan,
        repeated_episode_fraction=float(ep.outings.gt(1).mean()) if len(ep) else np.nan)
    return metrics,ep,ev


def calibration_diagnostic(g,score):
    """Evaluate calibration, never transform held-out predictions using these fits."""
    y=g.target.to_numpy(dtype=int);x=logit(np.clip(score,1e-8,1-1e-8));n_events=g.loc[g.target.eq(1),'dynamic_next_event_id'].nunique()
    out=dict(positive_rows=int(y.sum()),represented_events=n_events,intercept=np.nan,slope=np.nan,
        intercept_se=np.nan,slope_se=np.nan,status='insufficient_event_support')
    if n_events<CFG['minimum_calibration_events'] or y.sum()<CFG['minimum_calibration_positive_rows']:return out
    X=np.column_stack([np.ones(len(x)),x])
    def objective(beta):
        z=X@beta;p=expit(z)
        return np.sum(np.logaddexp(0,z)-y*z),X.T@(p-y)
    result=minimize(objective,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':1000})
    if not np.isfinite(result.x).all() or np.linalg.norm(result.jac)>1e-3:
        out['status']='fit_unstable';return out
    beta=result.x;p=expit(X@beta);hess=X.T@((p*(1-p))[:,None]*X)
    if np.linalg.cond(hess)>1e10:out['status']='singular_information';return out
    residual=X*(y-p)[:,None];cluster=pd.DataFrame(residual).groupby(g.pitcher_id.to_numpy()).sum().to_numpy()
    bread=np.linalg.inv(hess);cov=bread@(cluster.T@cluster)@bread;se=np.sqrt(np.diag(cov))
    out.update(intercept=float(beta[0]),slope=float(beta[1]),intercept_se=float(se[0]),slope_se=float(se[1]),status='diagnostic_only_no_recalibration')
    return out


def run_operations():
    if valid('operations'):return
    f,e,spec=prepare();ev=event_set(e);rows=[];episodes=[];details=[];cal=[];reliability=[];retrospective=[]
    for lo,hi in CFG['intervals']:
        interval=f'{lo}-{hi}';g=interval_target(f,lo,hi);p=pd.read_parquet(CACHE/f'predictions_{interval}.parquet').set_index('row_id')
        for name in spec['models']:
            score=g.row_id.map(p[name]).to_numpy();known=g.loc[g.target.notna()];ks=known.row_id.map(p[name]).to_numpy();key=interval+'|'+name
            for rate in CFG['alert_rates']:
                threshold=spec['thresholds'][key][str(rate)]
                grouped=alert_episodes(g,score,threshold,CFG['max_episode_gap_days'])
                m,ep,ed=episode_metrics(grouped,ev,lo,hi)
                ci=uncertainty(event_bootstrap(ed))
                rows.append(dict(interval=interval,model=name,development_budget=rate,threshold=threshold,**m,
                    event_recall_ci_low=ci['ci_low'],event_recall_ci_high=ci['ci_high']))
                episodes.append(ep.assign(interval=interval,model=name,development_budget=rate))
                details.append(ed.assign(interval=interval,model=name,development_budget=rate))
            # Existing top-5% ranking warning is retained explicitly as retrospective context.
            wm,_=warning(known,ks,ev,.05,lo,hi);retrospective.append(dict(interval=interval,model=name,**wm))
            cal.append(dict(interval=interval,model=name,**calibration_diagnostic(known,ks)))
            edges=np.r_[-np.inf,spec['reliability_edges'][key],np.inf]
            bins=pd.cut(ks,edges,labels=False);q=known.assign(score=ks,bin=bins)
            for b,group in q.groupby('bin'):
                reliability.append(dict(interval=interval,model=name,bin=b,rows=len(group),positive_rows=int(group.target.sum()),
                    represented_events=group.loc[group.target.eq(1),'dynamic_next_event_id'].nunique(),
                    mean_risk=float(group.score.mean()),observed=float(group.target.mean())))
    table(pd.DataFrame(rows),'alert_policy_results');table(pd.concat(episodes,ignore_index=True),'alert_episodes')
    table(pd.concat(details,ignore_index=True),'event_warning_results');table(pd.DataFrame(cal),'calibration_diagnostics')
    table(pd.DataFrame(reliability),'reliability_bins');table(pd.DataFrame(retrospective),'retrospective_warning_context')
    checkpoint('operations',[OUT/'tables'/(n+'.csv') for n in ['alert_policy_results','alert_episodes','event_warning_results','calibration_diagnostics','reliability_bins','retrospective_warning_context']])
