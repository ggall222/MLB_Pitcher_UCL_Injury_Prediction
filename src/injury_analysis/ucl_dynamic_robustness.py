"""Paired uncertainty, training refits and prespecified endpoint sensitivities."""
import json
import joblib
import numpy as np
import pandas as pd
from .ucl_dynamic_common import ROOT,OUT,CACHE,CFG,table,checkpoint,valid_checkpoint
from .ucl_dynamic_data import prepare_data,risk_frame,event_cohort
from .ucl_dynamic_models import fit_model,scores,model_path
from .ucl_dynamic_evaluation import metrics,warning,event_bootstrap
from .family_decomposition import paired_bootstrap_ap

def run_robustness():
    if valid_checkpoint('robustness'):return
    a,events,specs=prepare_data();f=risk_frame(a,events)
    freeze=json.loads((OUT/'frozen_protocol.json').read_text());selected=freeze['selected_model']
    objects={n:joblib.load(CACHE/'models'/(model_path(30,n)+'.joblib')) for n in specs}
    comparisons=[('Departure','Profile'),('Profile + departure','Profile'),
        ('Profile + departure + trajectory','Profile + departure'),
        ('Profile + departure + trajectory + interactions','Profile + departure + trajectory'),(selected,'Departure')]
    uncertainty=[];calibration=[];bootrows=[]
    validation=f.loc[f.partition.eq('validation')&f.target.notna()]
    edges={n:np.unique(np.r_[-np.inf,np.quantile(scores(o,validation),np.arange(.1,1,.1)),np.inf]) for n,o in objects.items()}
    for part in ['validation','test']:
        rows=f.loc[f.partition.eq(part)&f.target.notna()];ev=event_cohort(events,part)
        ss={n:scores(o,rows) for n,o in objects.items()};ap,_=paired_bootstrap_ap(rows,ss)
        eb={}
        for n,s in ss.items():
            _,detail=warning(rows,s,ev);eb[n]=event_bootstrap(detail)
            for rep,value in enumerate(ap[n]):bootrows.append(dict(partition=part,model=n,replicate=rep,average_precision=value,event_recall=eb[n][rep]))
            bins=pd.cut(s,edges[n],labels=False)
            t=pd.DataFrame(dict(bin=bins,score=s,target=rows.target.to_numpy(dtype=int)))
            c=t.groupby('bin').agg(rows=('score','size'),mean_score=('score','mean'),observed_risk=('target','mean'),positives=('target','sum')).reset_index()
            c['model']=n;c['partition']=part;calibration.append(c)
        for new,old in comparisons:
            for metric,arr in [('average_precision',ap),('event_recall_7_30d',eb)]:
                diff=arr[new]-arr[old]
                uncertainty.append(dict(partition=part,comparison=new+' minus '+old,metric=metric,
                    mean_bootstrap_difference=float(np.nanmean(diff)),ci_low=float(np.nanquantile(diff,.025)),ci_high=float(np.nanquantile(diff,.975)),
                    fraction_positive=float(np.nanmean(diff>0)),interpretation='conditional on trained models; whole-pitcher resampling'))
    table(pd.DataFrame(uncertainty),'paired_increment_uncertainty');table(pd.concat(calibration,ignore_index=True),'calibration_bins')
    table(pd.DataFrame(bootrows),'performance_bootstrap')
    train=f.loc[f.partition.eq('train')&f.target.notna()];groups={p:g for p,g in train.groupby('pitcher_id')};ids=np.array(list(groups))
    rng=np.random.default_rng(CFG['seed']);refitmetrics=[];coefficients=[]
    refitnames=list(dict.fromkeys(['Profile','Departure','Profile + departure','Profile + departure + trajectory','Profile + departure + trajectory + interactions',selected]))
    for rep in range(CFG['training_refits']):
        draw=rng.choice(ids,len(ids),replace=True);sample=pd.concat([groups[p] for p in draw],ignore_index=True)
        for n in refitnames:
            obj=fit_model(sample,specs[n],f'refit_{rep:02d}_'+model_path(30,n));s=scores(obj,validation)
            m,_=warning(validation,s,event_cohort(events,'validation'))
            refitmetrics.append(dict(replicate=rep,model=n,**metrics(validation,s),injury_event_recall=m['injury_event_recall']))
            coefficients.extend(dict(replicate=rep,model=n,feature=c,coefficient=float(b)) for c,b in zip(obj['encoder'].names,obj['model'].coef_[0]))
    table(pd.DataFrame(refitmetrics),'training_refit_metrics');c=pd.DataFrame(coefficients);table(c,'training_refit_coefficients')
    stability=c.groupby(['model','feature']).coefficient.agg(mean='mean',sd='std',minimum='min',maximum='max',positive_fraction=lambda x:(x>0).mean()).reset_index()
    table(stability,'coefficient_stability')
    sensitivity=[];skips=[]
    policies=[('competing injuries censored','primary','censor','earliest',None),
        ('probable included','expanded','negative','earliest',None),
        ('public placement date','primary','negative','public',None),
        ('strict official endpoint, frozen scores','strict_official','negative','earliest','strict'),
        ('exclude final six days','primary','negative','earliest','early'),
        ('complete trajectory support','primary','negative','earliest','supported'),
        ('starters only','primary','negative','earliest','starters'),
        ('relievers only','primary','negative','earliest','relievers'),
        ('pitchers unseen in training','primary','negative','earliest','unseen')]
    for label,endpoint,competing,timing,subset in policies:
        sf=risk_frame(a,events,30,endpoint,competing,timing) if subset in [None,'strict'] else f
        tr=sf.loc[sf.partition.eq('train')&sf.target.notna()]
        for n in list(dict.fromkeys([selected,'Departure','Profile + departure'])):
            if subset is None:
                if tr.loc[tr.target.eq(1),'dynamic_next_event_id'].nunique()<CFG['minimum_sensitivity_training_events']:
                    skips.append(dict(sensitivity=label,model=n,reason='Too few training events'));continue
                obj=fit_model(tr,specs[n],'sensitivity_'+label.replace(' ','_')+'_'+model_path(30,n))
            else:obj=objects[n]
            for part in ['validation','test']:
                rows=sf.loc[sf.partition.eq(part)&sf.target.notna()].copy();ev=event_cohort(events,part,endpoint,timing)
                if subset=='early':rows=rows.loc[~rows.dynamic_lead_days.between(1,6)]
                if subset=='supported':rows=rows.loc[rows[[c for c in a if c.startswith('dyn_')]].notna().all(axis=1)]
                if subset=='starters':rows=rows.loc[rows.is_starter]
                if subset=='relievers':rows=rows.loc[~rows.is_starter]
                if subset=='unseen':rows=rows.loc[~rows.pitcher_id.isin(ids)]
                # Subgroup denominators are events with a subgroup opportunity; explicitly report that distinction.
                if subset in ['supported','starters','relievers','unseen']:
                    ev=ev.loc[ev.injury_event_id.isin(rows.loc[rows.dynamic_next_target&rows.dynamic_lead_days.between(7,30),'dynamic_next_event_id'])]
                if len(rows)==0 or rows.target.nunique()<2:
                    skips.append(dict(sensitivity=label,model=n,reason=part+': no evaluable classes'));continue
                s=scores(obj,rows);w,_=warning(rows,s,ev)
                sensitivity.append(dict(sensitivity=label,partition=part,model=n,refit=subset is None,
                    event_denominator='subgroup observable events' if subset in ['supported','starters','relievers','unseen'] else 'all eligible endpoint events',
                    **metrics(rows,s),**{k:w[k] for k in ['qualifying_events','observable_events','alerted_events','injury_event_recall','median_warning_lead_days','false_alert_fraction']}))
    table(pd.DataFrame(sensitivity),'sensitivity_results');table(pd.DataFrame(skips,columns=['sensitivity','model','reason']),'sensitivity_skips')
    checkpoint('robustness',[OUT/'tables/paired_increment_uncertainty.csv',OUT/'tables/coefficient_stability.csv',OUT/'tables/sensitivity_results.csv'])
