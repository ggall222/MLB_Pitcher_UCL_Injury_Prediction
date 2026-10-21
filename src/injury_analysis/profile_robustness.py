"""Sensitivity and training perturbations for the frozen profile/departure contrast."""
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .profile_departure import (CFG,CACHE,PROFILE,DEPARTURE,BASELINE_PAIRS,table,read_table,
    clean,fit_model,label_frame,target_events,evaluate_scores,probability_metrics)
from .common import CFG as ORIGINAL_CFG
from src.evaluation import rank_metrics,alert_mask

CONTRAST=[('Profile + Departure','Profile only')]


def shared_availability_fit(train,targets,features):
    """Same explicit P+D missing flags in both models, without duplicate auto-flags."""
    all_fields=PROFILE+DEPARTURE
    def design(f):
        x=clean(f,features).copy()
        for name in all_fields:x['available__'+name]=clean(f,[name])[name].notna().astype(float)
        return x
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=False),StandardScaler(),
            LogisticRegression(C=ORIGINAL_CFG['logistic_C'],max_iter=3000,tol=CFG['logistic_tolerance'],
                               random_state=CFG['seed'],solver='lbfgs'))
        pipe.fit(design(train),train.target)
        scores=[pipe.predict_proba(design(t))[:,1] for t in targets]
    return scores


def sensitivity_frames(frame,condition):
    """Filter once, then give exactly the same rows/targets to both representations."""
    f=frame.copy()
    if condition=='Complete P+D measurements':
        f=f.loc[clean(f,PROFILE+DEPARTURE).notna().all(axis=1)]
    elif condition=='Exclude final six days':
        f=f.loc[~(f.target.eq(1)&f.next_event_lead_days.lt(7))]
    elif condition not in ['Shared availability flags','Measurement profiles only']:
        raise ValueError(condition)
    return {p:f.loc[f.partition.eq(p)].copy() for p in ['train','validation','test']}


def run_sensitivity(a,events):
    metrics=[];comparisons=[];coverage=[]
    for category in CFG['focus_categories']:
        original=label_frame(a,category,30)
        for condition in ['Complete P+D measurements','Shared availability flags',
                          'Exclude final six days','Measurement profiles only']:
            parts=sensitivity_frames(original,condition)
            profile=list(BASELINE_PAIRS) if condition=='Measurement profiles only' else PROFILE
            features={'Profile only':profile,'Profile + Departure':profile+DEPARTURE}
            scores={p:{} for p in ['validation','test']}
            for name,cols in features.items():
                target=[parts[p] for p in ['validation','test']]
                if condition=='Shared availability flags':ss=shared_availability_fit(parts['train'],target,cols)
                else:_,_,ss=fit_model(parts['train'],target,cols,tolerance=CFG['logistic_tolerance'])
                for p,score in zip(['validation','test'],ss):scores[p][name]=score
            for p in ['validation','test']:
                met,_,_,delta=evaluate_scores(parts[p],scores[p],target_events(events,category,p),
                    category,p,analysis=condition,contrasts=CONTRAST)
                metrics.append(met);comparisons.append(delta)
            for p,f in parts.items():
                coverage.append(dict(category=category,condition=condition,partition=p,rows=len(f),
                    positive_rows=int(f.target.sum()),represented_events=f.loc[f.target.eq(1),'next_event_id'].nunique(),
                    original_rows=int(original.partition.eq(p).sum())))
            print(f'Sensitivity complete: {category} / {condition}',flush=True)
    table(pd.concat(metrics,ignore_index=True),'sensitivity_performance')
    table(pd.concat(comparisons,ignore_index=True),'sensitivity_increment')
    table(pd.DataFrame(coverage),'sensitivity_coverage')
    return pd.concat(comparisons,ignore_index=True)


def run_stability(a):
    """Paired whole-pitcher training bootstrap, with every preprocessing fit repeated."""
    frozen=pd.read_parquet(CACHE/'predictions.parquet')
    records=[]
    for category in CFG['focus_categories']:
        f=label_frame(a,category,30);train=f.loc[f.partition.eq('train')].reset_index(drop=True)
        groups=train.groupby('pitcher_id').indices
        people=np.array(sorted(groups));rng=np.random.default_rng(CFG['seed'])
        parts={p:f.loc[f.partition.eq(p)] for p in ['validation','test']}
        baseline={}
        for p,g in parts.items():
            old=frozen.loc[frozen.category.eq(category)&frozen.partition.eq(p)].set_index('row_id').reindex(g.row_id)
            assert old.target.to_numpy().tolist()==g.target.to_numpy().tolist()
            baseline[p]={name:old['score__'+name].to_numpy() for name in ['Profile only','Profile + Departure']}
        for repetition in range(CFG['training_refits']):
            sampled=rng.choice(people,len(people),replace=True)
            boot=train.iloc[np.concatenate([groups[pid] for pid in sampled])]
            for name,columns in [('Profile only',PROFILE),('Profile + Departure',PROFILE+DEPARTURE)]:
                enc,model,scores=fit_model(boot,list(parts.values()),columns,tolerance=CFG['logistic_tolerance'])
                for (part,g),score in zip(parts.items(),scores):
                    reference=baseline[part][name]
                    new_alert=alert_mask(g.assign(score=score),'score',.05)
                    old_alert=alert_mask(g.assign(score=reference),'score',.05)
                    records.append(dict(category=category,partition=part,variant=name,refit=repetition,
                        rank_correlation=spearmanr(reference,score).statistic,
                        alert_jaccard=np.sum(new_alert&old_alert)/np.sum(new_alert|old_alert),
                        mean_abs_probability_shift=np.mean(np.abs(score-reference)),
                        **rank_metrics(g.target,score),**probability_metrics(g.target,score)))
            print(f'Training perturbation {category}: {repetition+1}/{CFG["training_refits"]}',flush=True)
    raw=table(pd.DataFrame(records),'training_refits')
    summary=raw.groupby(['category','partition','variant']).agg(
        refits=('refit','size'),mean_ap=('average_precision','mean'),sd_ap=('average_precision','std'),
        median_rank_correlation=('rank_correlation','median'),median_alert_jaccard=('alert_jaccard','median'),
        mean_probability_shift=('mean_abs_probability_shift','mean')).reset_index()
    table(summary,'stability_summary')
    deltas=raw.pivot(index=['category','partition','refit'],columns='variant',values='average_precision')
    deltas['ap_difference']=deltas['Profile + Departure']-deltas['Profile only']
    table(deltas.reset_index(),'training_refit_ap_differences')
    return summary


def run_adjusted_sensitivity(a,events,spec):
    """Probe the two arm findings after workload/role/calendar adjustment."""
    from .profile_departure import model_specs
    settings=model_specs(spec);metrics=[];comparisons=[]
    for category in ['Elbow / UCL','Shoulder']:
        original=label_frame(a,category,30)
        for condition in ['Complete P+D measurements','Shared availability flags','Exclude final six days']:
            parts=sensitivity_frames(original,condition)
            scores={p:{} for p in ['validation','test']}
            for name in ['Context + Profile','Context + Profile + Departure']:
                cols=settings[name];target=[parts[p] for p in ['validation','test']]
                if condition=='Shared availability flags':ss=shared_availability_fit(parts['train'],target,cols)
                else:_,_,ss=fit_model(parts['train'],target,cols,tolerance=CFG['logistic_tolerance'])
                for p,score in zip(['validation','test'],ss):scores[p][name]=score
            for p in ['validation','test']:
                met,_,_,delta=evaluate_scores(parts[p],scores[p],target_events(events,category,p),
                    category,p,analysis=condition,
                    contrasts=[('Context + Profile + Departure','Context + Profile')])
                metrics.append(met);comparisons.append(delta)
            print(f'Adjusted sensitivity: {category} / {condition}',flush=True)
    table(pd.concat(metrics,ignore_index=True),'adjusted_sensitivity_performance')
    return table(pd.concat(comparisons,ignore_index=True),'adjusted_sensitivity_increment')

