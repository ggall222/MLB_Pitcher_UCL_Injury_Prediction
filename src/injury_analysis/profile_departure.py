"""Direct profile versus departure experiment, reusing existing injury risk sets."""
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
import yaml
from .common import ROOT,CFG as ORIGINAL_CFG,CACHE as ORIGINAL_CACHE,CATEGORIES,source_signature,sha
from .reduction import inputs as reduction_inputs,BASELINE_PAIRS,fit_model,clean
from .family_decomposition import (feature_variants,paired_bootstrap_ap,target_events,
                                  opportunity_table,random_detection_probability)
from .signals_common import cluster_draws,mean_ci
from .cohort import label_frame
from src.evaluation import rank_metrics,evaluate_predictions,alert_mask

CONFIG_PATH=ROOT/'config/injury_profile_departure.yaml'
CFG=yaml.safe_load(CONFIG_PATH.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE]:p.mkdir(parents=True,exist_ok=True)
PROFILE=list(BASELINE_PAIRS)+CFG['profile_context']
DEPARTURE=list(BASELINE_PAIRS.values())
PRIMARY=['Profile only','Departure only','Profile + Departure']
CONTRASTS=[('Profile + Departure','Profile only'),('Profile + Departure','Departure only'),
           ('Context + Profile + Departure','Context + Profile'),
           ('Context + Profile + Departure','Context + Departure'),
           ('Profile + Departure + slopes','Profile + Departure')]


def table(frame,name):
    frame.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    return frame


def read_table(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')


def figure(fig,name):
    for suffix in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{suffix}',dpi=220,bbox_inches='tight')
    return fig


def inputs():
    a,events,spec,matches=reduction_inputs()
    extra=[c for c in CFG['prior_slopes']+['history_max_date','baseline_max_date'] if c not in a]
    before=source_signature()
    if extra:
        with duckdb.connect(str(ROOT/ORIGINAL_CFG['database_path']),read_only=True) as con:
            columns=set(con.execute('describe model_features').fetchdf().column_name)
            if not set(extra)<=columns:raise ValueError(f'Existing feature fields absent: {set(extra)-columns}')
            projection=','.join('"'+c+'"' for c in extra)
            supplement=con.execute(f'SELECT pitcher_id,game_id,{projection} FROM model_features').fetchdf()
        a=a.merge(supplement,on=['pitcher_id','game_id'],how='left',validate='one_to_one')
    assert before==source_signature()
    for date in ['history_max_date','baseline_max_date']:
        a[date]=pd.to_datetime(a[date])
        assert (a[date].isna() | a[date].lt(a.game_date)).all(),f'Nonprior history in {date}'
    return a,events,spec,matches


def model_specs(spec):
    previous,groups=feature_variants(spec)
    context=groups['workload']+['is_starter','calendar_month']
    assert not set(PROFILE)&set(DEPARTURE)
    assert not set(context)&set(PROFILE+DEPARTURE)
    return {'Profile only':PROFILE,'Departure only':DEPARTURE,'Profile + Departure':PROFILE+DEPARTURE,
            'Context + Profile':context+PROFILE,'Context + Departure':context+DEPARTURE,
            'Context + Profile + Departure':context+PROFILE+DEPARTURE,
            'Profile + Departure + slopes':PROFILE+DEPARTURE+CFG['prior_slopes'],
            'Workload + history':previous['Workload + history'],'Original 62':previous['Full']}


def feature_inventory(a,spec):
    settings=model_specs(spec)
    rows=[]
    for name in sorted(set(c for cols in settings.values() for c in cols)):
        if name in BASELINE_PAIRS:
            block='Profile: prior measurement mean';why='Up to 20 strictly prior outings, at least five valid measurements; slowly evolving, not immutable.'
        elif name in CFG['profile_context']:
            block='Profile: slowly changing context';why='Age, handedness, observed experience or strictly prior reported IL history; not proof of causal susceptibility.'
        elif name in DEPARTURE:
            block='Departure: current relative to prior baseline';why='Current value minus prior mean divided by prior SD; available after current outing.'
        elif name in CFG['prior_slopes']:
            block='Secondary: prior trend';why='Slope over up to five earlier outings, excluding current and future days.'
        elif name in settings['Context + Profile']:
            block='Shared workload/role/calendar adjustment';why='Mixed persistent and situational meaning; held common in the adjusted comparison.'
        else:
            block='Original-model comparator only';why='Absolute current measurements, raw pitch mix or transformations not forced into either primary block.'
        rows.append(dict(feature=name,block=block,rationale=why,
                         train_missing_fraction=clean(a.loc[a.partition.eq('train')&a.analysis_eligible],[name])[name].isna().mean()))
    table(pd.DataFrame(rows),'feature_definitions')
    table(pd.DataFrame([dict(variant=n,features=len(c),columns=';'.join(c)) for n,c in settings.items()]),'model_specifications')
    return settings


def probability_metrics(y,score):
    """Brier assesses probability error, not calibration alone; also show mean bias."""
    y=np.asarray(y,dtype=float);p=np.asarray(score,dtype=float)
    if len(y)!=len(p) or not len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Need aligned nonempty binary outcomes and probabilities in [0,1]')
    if not np.isin(y,[0,1]).all():raise ValueError('Targets must be observed binary outcomes')
    return dict(brier_score=np.mean((p-y)**2),mean_prediction=p.mean(),observed_rate=y.mean(),
                calibration_bias=p.mean()-y.mean())


def brier_bootstrap(frame,scores):
    draws,inv=cluster_draws(frame.pitcher_id.to_numpy(),CFG['bootstrap_replicates'])
    totals=np.bincount(inv);denom=draws@totals
    y=frame.target.to_numpy(dtype=float)
    return {name:(draws@np.bincount(inv,weights=(score-y)**2))/denom for name,score in scores.items()}


def reliability_bins(frame,score,edges):
    f=frame[['pitcher_id','target']].copy();f['probability']=score
    f['bin']=pd.cut(f.probability,edges,include_lowest=True,labels=False)
    if f.bin.isna().any():raise ValueError('Reliability edges must cover all probabilities')
    return f.groupby('bin').agg(rows=('target','size'),positive_rows=('target','sum'),
        pitchers=('pitcher_id','nunique'),mean_prediction=('probability','mean'),observed_rate=('target','mean')).reset_index()


def event_detection(frame,score,events,rate=.05):
    f=frame.reset_index(drop=True).copy();f['score']=score
    opp=opportunity_table(f,events)
    hits=opp.loc[opp.eval_row.isin(np.flatnonzero(alert_mask(f,'score',rate)))]
    lead=hits.groupby('injury_event_id').lead_days.max()
    result=events[['injury_event_id','pitcher_id']].copy()
    result['warning_lead_days']=result.injury_event_id.map(lead)
    result['detected']=result.warning_lead_days.notna().astype(int)
    return result


def event_difference(first,second):
    f=first.merge(second,on=['injury_event_id','pitcher_id'],suffixes=('_first','_second'),validate='one_to_one')
    f['difference']=f.detected_first-f.detected_second
    result=mean_ci(f,'difference')
    result.update(gained_events=int(((f.detected_first==1)&(f.detected_second==0)).sum()),
        lost_events=int(((f.detected_first==0)&(f.detected_second==1)).sum()),
        shared_detected_events=int(((f.detected_first==1)&(f.detected_second==1)).sum()))
    common=f.dropna(subset=['warning_lead_days_first','warning_lead_days_second']).copy()
    common['lead_difference']=common.warning_lead_days_first-common.warning_lead_days_second
    lead=mean_ci(common,'lead_difference')
    result.update({f'common_lead_{k}':v for k,v in lead.items()})
    return result


def evaluate_scores(frame,scores,events,category,partition,analysis='Primary',contrasts=None):
    contrasts=CONTRASTS if contrasts is None else contrasts
    bs,prev=paired_bootstrap_ap(frame,scores);brier_bs=brier_bootstrap(frame,scores)
    rows,budgets,leads,paired=[],[],[],[];detections={}
    metadata=dict(category=category,partition=partition,analysis=analysis)
    opp=opportunity_table(frame,events)
    counts=opp.groupby('injury_event_id').size().reindex(events.injury_event_id,fill_value=0)
    for name,score in scores.items():
        metric=dict(metadata,variant=name,rows=len(frame),positive_rows=int(frame.target.sum()),
            represented_events=frame.loc[frame.target.eq(1),'next_event_id'].nunique(),pitchers=frame.pitcher_id.nunique(),
            prevalence=frame.target.mean(),**rank_metrics(frame.target,score),**probability_metrics(frame.target,score))
        metric['pr_lift']=metric['average_precision']/metric['prevalence']
        metric['ap_low'],metric['ap_high']=np.nanquantile(bs[name],[.025,.975])
        metric['brier_low'],metric['brier_high']=np.nanquantile(brier_bs[name],[.025,.975])
        rows.append(metric)
        f=frame.copy();f['score']=score
        b,l=evaluate_predictions(f,events,target='target',rates=ORIGINAL_CFG['alert_rates'],warning_window=(7,30))
        b['random_expected_event_recall']=[np.mean([random_detection_probability(len(f),int(rate*len(f)),int(m))
                                                  for m in counts]) for rate in b.alert_rate]
        budgets.append(b.assign(variant=name,**metadata));leads.append(l.assign(variant=name,**metadata))
        detections[name]=event_detection(frame,score,events)
    metrics=pd.DataFrame(rows).set_index('variant')
    for first,second in contrasts:
        if not {first,second}<=set(scores):continue
        lo,hi=np.nanquantile(bs[first]-bs[second],[.025,.975])
        blo,bhi=np.nanquantile(brier_bs[first]-brier_bs[second],[.025,.975])
        paired.append(dict(metadata,variant=first,reference=second,
            ap_difference=metrics.loc[first,'average_precision']-metrics.loc[second,'average_precision'],ap_low=lo,ap_high=hi,
            brier_difference=metrics.loc[first,'brier_score']-metrics.loc[second,'brier_score'],brier_low=blo,brier_high=bhi,
            **{'event_'+k:v for k,v in event_difference(detections[first],detections[second]).items()}))
    return metrics.reset_index(),pd.concat(budgets),pd.concat(leads),pd.DataFrame(paired)


def run_models(a,events,spec):
    settings=feature_inventory(a,spec)
    results,alerts,leads,differences,counts,wide,coefficients,calibration,fixed,dimensions=[],[],[],[],[],[],[],[],[],[]
    for category in CATEGORIES:
        f=label_frame(a,category,CFG['horizon_days'])
        parts={p:f.loc[f.partition.eq(p)].copy() for p in ['train','validation','test']}
        numbers={p:x.loc[x.target.eq(1),'next_event_id'].nunique() for p,x in parts.items()}
        enough=numbers['train']>=30 and min(numbers['validation'],numbers['test'])>=10
        for p,x in parts.items():counts.append(dict(category=category,partition=p,rows=len(x),positive_rows=int(x.target.sum()),events=numbers[p],run=enough))
        if not enough:continue
        print(f'Profile/departure models: {category}; represented events {numbers}',flush=True)
        train,valid,test=[parts[p] for p in ['train','validation','test']]
        scores={p:{} for p in ['validation','test']};fitted={}
        for name,features in settings.items():
            enc,model,(vs,ts)=fit_model(train,[valid,test],features,tolerance=CFG['logistic_tolerance'])
            fitted[name]=(enc,model);scores['validation'][name]=vs;scores['test'][name]=ts
            dimensions.append(dict(category=category,variant=name,source_features=len(features),design_columns=enc.output_dimensions,
                                   missing_indicators=len(enc.names)-len(features),training_rows=enc.fit_rows))
            coefficients.extend(dict(category=category,variant=name,feature=n,standardized_log_odds=b)
                                for n,b in zip(enc.names,enc.effective_coefficients(model)))
        for part in ['validation','test']:
            ev=target_events(events,category,part)
            met,b,l,d=evaluate_scores(parts[part],scores[part],ev,category,part)
            results.append(met);alerts.append(b);leads.append(l);differences.append(d)
            w=parts[part][['row_id','pitcher_id','game_id','game_date','season','is_starter','target','next_event_id',
                           'next_event_lead_days','next_event_strict_qa']].copy().assign(category=category,partition=part)
            w['seen_training_pitcher']=w.pitcher_id.isin(train.pitcher_id)
            for name,score in scores[part].items():
                w['score__'+name]=score
                edges=np.unique(np.r_[0.,np.quantile(scores['validation'][name],np.linspace(0,1,CFG['calibration_bins']+1)[1:-1]),1.])
                calibration.append(reliability_bins(parts[part],score,edges).assign(category=category,partition=part,variant=name))
            wide.append(w)
        test_events=target_events(events,category,'test');opp=opportunity_table(test,test_events)
        for name,score in scores['test'].items():
            threshold=np.quantile(scores['validation'][name],.95,method='higher');mask=score>=threshold
            hits=opp.loc[opp.eval_row.isin(np.flatnonzero(mask))]
            lead=hits.groupby('injury_event_id').lead_days.max()
            fixed.append(dict(category=category,variant=name,validation_threshold=threshold,
                validation_actual_rate=np.mean(scores['validation'][name]>=threshold),test_alert_rate=mask.mean(),
                test_precision=test.target.to_numpy()[mask].mean() if mask.any() else np.nan,
                test_recall=test.target.to_numpy()[mask].sum()/test.target.sum(),
                test_event_recall=len(lead)/len(test_events),median_warning_lead_days=lead.median()))
        joblib.dump({n:fitted[n] for n in PRIMARY+['Context + Profile','Context + Profile + Departure']},
                    CACHE/f"models_{category.replace('/','-').replace(' ','_')}.joblib")
        table(pd.concat(results,ignore_index=True),'model_performance')
        table(pd.concat(differences,ignore_index=True),'incremental_comparisons')
    for rows,name in [(alerts,'alert_budget_performance'),(leads,'warning_leads'),(calibration,'reliability_bins')]:table(pd.concat(rows,ignore_index=True),name)
    for rows,name in [(counts,'cohort_counts'),(coefficients,'model_coefficients'),(fixed,'validation_fixed_thresholds'),(dimensions,'model_dimensions')]:table(pd.DataFrame(rows),name)
    pd.concat(wide,ignore_index=True).to_parquet(CACHE/'predictions.parquet',index=False)
    save_manifest()
    return pd.concat(results,ignore_index=True)


def save_manifest():
    result=dict(source_signature=source_signature(),config=CFG,config_sha256=sha(CONFIG_PATH),
        original_cohort_sha256=sha(ORIGINAL_CACHE/'manifest.json'),
        matching_sha256=sha(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/injury_analysis').glob('profile*.py')},
        test_status='Previously examined 2024–2025; exploratory temporal comparisons, not new confirmation.')
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2))
    return result
