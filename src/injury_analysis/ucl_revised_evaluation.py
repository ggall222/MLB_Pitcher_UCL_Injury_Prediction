"""Frozen-policy evaluation; unknown late-season outcomes remain explicitly unknown."""
import json
import numpy as np
import pandas as pd
from .common import sha
from .ucl_dynamic_evaluation import metrics,warning,event_bootstrap
from .ucl_temporal_analysis import bootstrap_ap,uncertainty
from .ucl_revised_common import ROOT,OUT,CACHE,CFG,table,parquet,write_json,checkpoint,valid_checkpoint,signature
from .ucl_revised_data import prepare,frame,event_set
from .ucl_revised_models import load,scores,metric_row

def frozen_protocol():
    p=OUT/'frozen_protocol.json'
    if not p.exists():raise ValueError('Freeze validation decisions before evaluating the 2026 test layer')
    x=json.loads(p.read_text())
    if x['source']!=signature():raise ValueError('Refreshed sources changed after freeze')
    if x['test_adaptation_allowed']:raise ValueError('Test adaptation is forbidden')
    for name,digest in x['model_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen model/preprocessing changed: '+name)
    return x

def score_frozen(obj,g,protocol,h,mid):
    if obj['signature']!=protocol['fit_signatures'][f'{h}|{mid}']:
        raise ValueError('Model not registered in the freeze')
    if pd.Timestamp(obj['training_max_date'])>pd.Timestamp(CFG['train_end']):
        raise ValueError('Nontraining observations entered fitted preprocessing/model')
    return scores(obj,g)

def operational(g,score,events,threshold,h,lead=7):
    """All eligible landmarks; never equate unknown-outcome alerts with false alerts."""
    g=g.copy();g['alert']=np.asarray(score)>=threshold
    known=g.target.notna();positive=g.target.eq(1).fillna(False)
    useful=g.dynamic_next_target & g.dynamic_lead_days.between(lead,h) & g.dynamic_next_event_id.isin(events.injury_event_id)
    hits=g.loc[useful&g.alert]
    alerts=int(g.alert.sum())
    return dict(eligible_landmarks=len(g),alerts=alerts,actual_alert_rate=float(g.alert.mean()),
        known_outcome_alerts=int((g.alert&known).sum()),unknown_outcome_alerts=int((g.alert&~known).sum()),
        observed_negative_alerts=int((g.alert&known&~positive).sum()),
        actionable_alerts=len(hits),alerted_events=hits.dynamic_next_event_id.nunique(),
        all_qualifying_events=len(events),unknown_outcome_rows=int((~known).sum()),
        observed_nonactionable_alerts=int((g.alert&known&~useful).sum()),
        unresolved_alert_fraction=float((g.alert&~known).sum()/alerts) if alerts else np.nan)

def run_evaluation():
    if valid_checkpoint('evaluation'):return
    protocol=frozen_protocol();a,e,specs=prepare()
    access=OUT/'test_access.json'
    record=dict(first_scored_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),freeze_sha256=sha(OUT/'frozen_protocol.json'),
        freeze_at_utc=protocol['frozen_at_utc'],test_year=2026,purpose='Evaluate only pre-frozen models and thresholds')
    if access.exists():
        if json.loads(access.read_text())['freeze_sha256']!=record['freeze_sha256']:
            raise ValueError('Freeze changed after first test access')
    else:write_json(record,access)
    rows=[];warnings=[];eventrows=[];burden=[];cohort=[];statuses=[];calibration=[];predictions=[]
    for h in sorted({h for h,mid in protocol['evaluation_pairs']}):
        f=frame(a,e,h)
        for part in ['train','validation','test']:
            g=f.loc[f.partition.eq(part)&f.analysis_eligible];known=g.loc[g.target.notna()];ev=event_set(e,part)
            opportunities=g.dynamic_next_target & g.dynamic_lead_days.between(7,h) & g.dynamic_next_event_id.isin(ev.injury_event_id)
            cohort.append(dict(partition=part,horizon=h,appearance_rows=int(f.partition.eq(part).sum()),eligible_landmarks=len(g),
                rows=len(known),pitchers=known.pitcher_id.nunique(),positive_rows=int(known.target.sum()),
                represented_events=known.loc[known.target.eq(1),'dynamic_next_event_id'].nunique(),
                qualifying_events=len(ev),events_with_7d_warning_opportunity=g.loc[opportunities,'dynamic_next_event_id'].nunique(),
                warning_opportunity_rows=int(opportunities.sum()),unknown_outcome_rows=int(g.target.isna().sum()),
                prevalence=float(known.target.mean()),first_date=g.game_date.min(),last_date=g.game_date.max(),
                last_eventfree_negative_date=g.loc[g.label_status.eq('observed_event_free_horizon'),'game_date'].max(),
                last_known_outcome_date=known.game_date.max()))
            ss=g.assign(month=g.game_date.dt.to_period('M').astype(str)).groupby(['month','label_status']).size().rename('rows').reset_index()
            statuses.append(ss.assign(partition=part,horizon=h))
        for part in ['validation','test']:
            prospective=f.loc[f.partition.eq(part)&f.analysis_eligible].copy()
            g=prospective.loc[prospective.target.notna()];ev=event_set(e,part)
            p=g[['row_id','pitcher_id','game_id','game_date','target']].copy()
            for hh,mid in protocol['evaluation_pairs']:
                if hh!=h:continue
                obj=load(h,mid);s=score_frozen(obj,g,protocol,h,mid);p[mid]=s
                rows.append(metric_row(g,s,e,protocol['catalog'][mid],h,part))
                for lead in [l for l in CFG['minimum_leads'] if l<=h]:
                    for rate in CFG['alert_rates']:
                        t=protocol['thresholds'][f'{h}|{mid}'][str(rate)]
                        for cutoff in [None,t]:
                            w,ed=warning(g,s,ev,rate,lead,h,cutoff)
                            warnings.append(dict(partition=part,horizon=h,model_id=mid,**w))
                            if rate==.05:
                                ed=ed.assign(partition=part,horizon=h,model_id=mid,minimum_lead=lead,
                                    policy='retrospective_budget' if cutoff is None else 'frozen_validation_threshold')
                                eventrows.append(ed)
                threshold=protocol['thresholds'][f'{h}|{mid}']['0.05']
                allscore=score_frozen(obj,prospective,protocol,h,mid)
                burden.append(dict(partition=part,horizon=h,model_id=mid,threshold=threshold,
                    **operational(prospective,allscore,ev,threshold,h)))
                edges=np.r_[-np.inf,protocol['calibration_edges'][f'{h}|{mid}'],np.inf]
                bins=pd.cut(s,edges,labels=False)
                c=pd.DataFrame(dict(bin=bins,score=s,target=g.target.to_numpy(dtype=int))).groupby('bin').agg(
                    rows=('score','size'),positives=('target','sum'),predicted=('score','mean'),observed=('target','mean')).reset_index()
                calibration.append(c.assign(partition=part,horizon=h,model_id=mid))
            parquet(p,f'predictions_{part}_h{h}')
    table(pd.DataFrame(rows),'model_metrics');table(pd.DataFrame(warnings),'warning_metrics')
    table(pd.concat(eventrows,ignore_index=True),'event_warnings');table(pd.DataFrame(burden),'operational_alert_burden')
    table(pd.DataFrame(cohort),'cohort_support');table(pd.concat(statuses,ignore_index=True),'followup_by_month')
    table(pd.concat(calibration,ignore_index=True),'calibration_bins')
    # Feature coverage is described after freeze and never changes the selected model.
    features=sorted({c for mid in {m for h,m in protocol['evaluation_pairs']} for c in protocol['catalog'][mid]['features']})
    coverage=[];distribution=[]
    tr=a.loc[a.partition.eq('train')&a.analysis_eligible]
    for c in features:
        x=pd.to_numeric(tr[c],errors='coerce');q=x.quantile([.005,.995]);median=float(x.median());sd=float(x.std())
        for part in ['train','validation','test']:
            g=a.loc[a.partition.eq(part)&a.analysis_eligible];v=pd.to_numeric(g[c],errors='coerce')
            coverage.append(dict(feature=c,partition=part,rows=len(g),missing_fraction=float(v.isna().mean()),
                observed_pitchers=g.loc[v.notna(),'pitcher_id'].nunique()))
            distribution.append(dict(feature=c,partition=part,median=float(v.median()),training_median=median,
                median_shift_training_sd=(float(v.median())-median)/sd if sd else np.nan,
                fraction_outside_training_99pct=float((v.dropna().lt(q.iloc[0])|v.dropna().gt(q.iloc[1])).mean())))
    table(pd.DataFrame(coverage),'feature_coverage');table(pd.DataFrame(distribution),'feature_distribution_shift')
    checkpoint('evaluation',[OUT/'test_access.json']+[OUT/'tables'/(n+'.csv') for n in ['model_metrics','warning_metrics',
        'event_warnings','operational_alert_burden','cohort_support','followup_by_month','calibration_bins','feature_coverage','feature_distribution_shift']]+
        list(CACHE.glob('predictions_*.parquet')))

def run_uncertainty():
    if valid_checkpoint('uncertainty'):return
    protocol=frozen_protocol();a,e,_=prepare();f=frame(a,e);rows=[];draws=[];warningrows=[];window=[]
    reference='Departure__Linear__1p0'
    details=pd.read_csv(OUT/'tables/event_warnings.csv')
    for part in ['validation','test']:
        g=f.loc[f.partition.eq(part)&f.target.notna()]
        p=pd.read_parquet(CACHE/f'predictions_{part}_h30.parquet');assert p.row_id.tolist()==g.row_id.tolist()
        names=[mid for h,mid in protocol['evaluation_pairs'] if h==30]
        b,prev=bootstrap_ap(g,{mid:p[mid].to_numpy() for mid in names})
        draws.append(pd.DataFrame(b).assign(prevalence=prev,partition=part,replicate=np.arange(len(prev))))
        for mid in names:
            r=dict(partition=part,model_id=mid)
            for prefix,value in [('ap',b[mid]),('lift',b[mid]/prev),('ap_delta_from_departure',b[mid]-b[reference])]:
                r.update({prefix+'_'+k:v for k,v in uncertainty(value).items()})
            rows.append(r)
            for policy in ['retrospective_budget','frozen_validation_threshold']:
                ed=details.loc[details.partition.eq(part)&details.horizon.eq(30)&details.model_id.eq(mid)&details.minimum_lead.eq(7)&details.policy.eq(policy)]
                w=event_bootstrap(ed)
                warningrows.append(dict(partition=part,model_id=mid,policy=policy,**uncertainty(w)))
        if part=='validation':
            old=pd.read_parquet(CACHE/'validation_selected_h30.parquet')
            bb,_=bootstrap_ap(g,{name:old[name].to_numpy() for name in old if name.startswith('Old window | ')})
            for mid in names:
                delta=b[mid]-bb['Old window | '+mid]
                window.append(dict(model_id=mid,comparison='2015–2023 minus 2015–2021 training; identical 2024–2025 validation',**uncertainty(delta)))
    table(pd.DataFrame(rows),'performance_uncertainty');table(pd.DataFrame(warningrows),'warning_uncertainty')
    table(pd.DataFrame(window),'training_window_uncertainty');parquet(pd.concat(draws,ignore_index=True),'performance_bootstrap')
    checkpoint('uncertainty',[OUT/'tables/performance_uncertainty.csv',OUT/'tables/warning_uncertainty.csv',
        OUT/'tables/training_window_uncertainty.csv',CACHE/'performance_bootstrap.parquet'])
