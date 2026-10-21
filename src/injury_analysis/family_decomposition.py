"""Family-only and incremental models with identical rows and paired uncertainty."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import CFG as ORIGINAL_CFG
from .cohort import label_frame
from .prediction import fit_scores
from .signals_common import CFG, CACHE, OUT, table, cluster_draws, mean_ci, save_manifest
from src.evaluation import rank_metrics, evaluate_predictions, alert_mask


def feature_variants(spec):
    selected=spec.loc[spec.selected_for_prediction]
    groups={g:f.name.tolist() for g,f in selected.groupby('family')}
    def union(*families): return list(dict.fromkeys(c for g in families for c in groups[g]))
    variants=dict(groups)
    context=['workload','pitcher_history']
    statcast=['velocity','release_mechanics','extension','spin_movement','command','pitch_mix']
    variants.update({
        'Workload + history':union(*context),
        'Velocity + mechanics':union('velocity','release_mechanics','extension'),
        'Statcast only':union(*statcast),
        'WH + velocity':union(*context,'velocity'),
        'WH + mechanics':union(*context,'release_mechanics','extension'),
        'WH + velocity + mechanics':union(*context,'velocity','release_mechanics','extension'),
        'WH + spin/movement':union(*context,'spin_movement'),
        'WH + command':union(*context,'command'),
        'WH + pitch mix':union(*context,'pitch_mix'),
        'Full':selected.name.tolist(),
    })
    assert set(variants['Full'])==set(union(*(context+statcast)))
    return variants,groups


def weighted_ap(y, scores, weights):
    """Weighted AP retaining tie groups; weights may represent cluster resampling."""
    order=np.argsort(-np.asarray(scores),kind='stable')
    y=np.asarray(y)[order]; s=np.asarray(scores)[order]; w=np.asarray(weights)[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(w*y)[ends]; n=np.cumsum(w)[ends]
    if tp[-1]==0: return np.nan
    precision=np.divide(tp,n,out=np.zeros(len(tp)),where=n>0)
    return np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1]


def paired_bootstrap_ap(frame, scores):
    """One set of pitcher draws for every model; conditional on trained models."""
    draws,inv=cluster_draws(frame.pitcher_id.to_numpy())
    y=frame.target.to_numpy(dtype=int)
    prevalence=np.array([np.dot(y,w[inv])/w[inv].sum() for w in draws])
    result={}
    for name,score in scores.items():
        order=np.argsort(-score,kind='stable'); yy=y[order]; ss=np.asarray(score)[order]; ii=inv[order]
        ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
        vals=[]
        for draw in draws:
            weights=draw[ii]
            tp=np.cumsum(weights*yy)[ends]; nn=np.cumsum(weights)[ends]
            prec=np.divide(tp,nn,out=np.zeros(len(tp)),where=nn>0)
            vals.append(np.sum(np.diff(np.r_[0,tp])*prec)/tp[-1] if tp[-1] else np.nan)
        result[name]=np.asarray(vals)
    return result,prevalence


def target_events(events, category, partition, strict=False):
    start,end=(ORIGINAL_CFG['train_end'],ORIGINAL_CFG['validation_end']) if partition=='validation' else (ORIGINAL_CFG['validation_end'],ORIGINAL_CFG['test_end'])
    mask=events.target_eligible & events.analysis_group.eq(category) & events.anchor_date.gt(start) & events.anchor_date.le(end)
    if strict: mask &= events.strict_qa_eligible
    return events.loc[mask].rename(columns={'player_id':'pitcher_id','anchor_date':'label_reference_date'}).copy()


def opportunity_table(frame, events, minimum=7, maximum=30):
    f=frame.reset_index(drop=True).reset_index(names='eval_row')
    opp=f[['eval_row','pitcher_id','game_id','game_date']].merge(
        events[['injury_event_id','pitcher_id','label_reference_date']],on='pitcher_id')
    opp['lead_days']=(opp.label_reference_date-opp.game_date).dt.days
    return opp.loc[opp.lead_days.between(minimum,maximum)]


def random_detection_probability(n,k,m):
    if m==0 or k==0:return 0.
    if m>n-k:return 1.
    return float(1-np.prod([(n-k-j)/(n-j) for j in range(m)]))


def evaluate_variants(frame, scores, events, metadata):
    metrics=[];budgets=[];event_differences=[];lead_tables=[]
    boot,prev_boot=paired_bootstrap_ap(frame,scores)
    reference='Workload + history' if 'Workload + history' in scores else 'Full without arm angle'
    opp=opportunity_table(frame,events)
    counts=opp.groupby('injury_event_id').size().reindex(events.injury_event_id,fill_value=0)
    n=len(frame)
    detection={}
    for name,score in scores.items():
        row=dict(metadata,variant=name,rows=n,pitchers=frame.pitcher_id.nunique(),positive_rows=int(frame.target.sum()),
            represented_events=frame.loc[frame.target.eq(1),'next_event_id'].nunique(),prevalence=frame.target.mean(),
            **rank_metrics(frame.target,score))
        row['pr_lift']=row['average_precision']/row['prevalence']
        row['ap_low'],row['ap_high']=np.nanquantile(boot[name],[.025,.975])
        row['lift_low'],row['lift_high']=np.nanquantile(boot[name]/prev_boot,[.025,.975])
        row['reference']=reference
        row['ap_delta']=row['average_precision']-rank_metrics(frame.target,scores[reference])['average_precision']
        row['delta_low'],row['delta_high']=np.nanquantile(boot[name]-boot[reference],[.025,.975])
        metrics.append(row)
        f=frame.copy().reset_index(drop=True);f['score']=score
        b,l=evaluate_predictions(f,events,target='target',rates=ORIGINAL_CFG['alert_rates'],warning_window=(7,30))
        random=[]
        for rate in b.alert_rate:
            k=int(np.floor(rate*n))
            random.append(np.mean([random_detection_probability(n,k,int(m)) for m in counts]))
        b['random_expected_event_recall']=random
        b['event_recall_minus_random']=b.injury_event_recall-b.random_expected_event_recall
        budgets.append(b.assign(variant=name,**metadata))
        lead_tables.append(l.assign(variant=name,**metadata))
        # Conditional fixed top-5% alerts. Event bootstrap preserves a pitcher's events.
        alerted_rows=np.flatnonzero(alert_mask(f,'score',.05))
        hit=set(opp.loc[opp.eval_row.isin(alerted_rows),'injury_event_id'])
        detection[name]=events.injury_event_id.isin(hit).to_numpy(dtype=float)
    for name in scores:
        ev=events[['pitcher_id','injury_event_id']].copy()
        ev['difference']=detection[name]-detection[reference]
        event_differences.append(dict(metadata,variant=name,reference=reference,
            **mean_ci(ev,'difference')))
        ev['difference']=detection[name]-np.array([random_detection_probability(n,int(.05*n),int(m)) for m in counts])
        event_differences.append(dict(metadata,variant=name,reference='Uniform random expectation',
            **mean_ci(ev,'difference')))
    # Keep paired cross-category draws available for direct differences in incremental AP.
    np.savez_compressed(CACHE/f"bootstrap_{metadata['analysis']}_{metadata['category'].replace('/','-').replace(' ','_')}_{metadata['partition']}.npz",
        **{name:vals for name,vals in boot.items()})
    return pd.DataFrame(metrics),pd.concat(budgets),pd.DataFrame(event_differences),pd.concat(lead_tables)


def inspection_importance(pipe, valid, features, families, family_metrics, category):
    """Validation-only stratified permutation; shuffle variation is not a sampling CI."""
    names=pipe.named_steps['simpleimputer'].get_feature_names_out(features)
    coef=pd.DataFrame({'category':category,'feature':names,'standardized_log_odds':pipe.named_steps['logisticregression'].coef_[0]})
    pure=family_metrics.loc[family_metrics.variant.isin(families)].nlargest(2,'average_precision')
    inspect_families=pure.variant.tolist()
    selected=list(dict.fromkeys(c for family in inspect_families for c in families[family]))
    X=valid[features].replace([np.inf,-np.inf],np.nan).reset_index(drop=True)
    strata=list(valid.reset_index(drop=True).groupby(['season','is_starter']).indices.values())
    rng=np.random.default_rng(CFG['seed']);rows=[]
    with threadpool_limits(limits=1):
        base=rank_metrics(valid.target,pipe.predict_proba(X)[:,1])['average_precision']
        for feature in selected:
            if feature=='is_starter':continue # Fixed by permutation strata; coefficient still exported.
            for rep in range(CFG['permutation_repeats']):
                altered=X.copy();vals=X[feature].to_numpy(copy=True)
                for index in strata:vals[index]=vals[rng.permutation(index)]
                altered[feature]=vals
                ap=rank_metrics(valid.target,pipe.predict_proba(altered)[:,1])['average_precision']
                rows.append(dict(category=category,feature=feature,family=next(g for g,v in families.items() if feature in v),
                    repeat=rep,baseline_ap=base,permuted_ap=ap,ap_drop=base-ap,
                    inspection_basis='top two individual families by validation AP; exploratory selection'))
    return coef,pd.DataFrame(rows)


def run_decomposition(a,events,spec):
    variants,families=feature_variants(spec)
    table(pd.DataFrame([dict(variant=k,features=len(v),columns=';'.join(v)) for k,v in variants.items()]),'family_definitions')
    all_metrics=[];all_budgets=[];all_diffs=[];all_leads=[];inventories=[];wide=[];coefs=[];importance=[];fixed=[]
    for category in CFG['categories']:
        f=label_frame(a,category,30)
        parts={part:f.loc[f.partition.eq(part)].copy() for part in ['train','validation','test']}
        counts={p:x.loc[x.target.eq(1),'next_event_id'].nunique() for p,x in parts.items()}
        enough=counts['train']>=30 and counts['validation']>=10 and counts['test']>=10
        for p,x in parts.items():inventories.append(dict(category=category,partition=p,rows=len(x),events=counts[p],
            positives=int(x.target.sum()),predictors=len(variants['Full']),run=enough))
        if not enough:continue
        print(f'Family decomposition: {category}; events={counts}',flush=True)
        scores={p:{} for p in ['validation','test']};full_pipe=None
        for name,features in variants.items():
            pipe,(vs,ts)=fit_scores(parts['train'],[parts['validation'],parts['test']],features)
            scores['validation'][name]=vs;scores['test'][name]=ts
            if name=='Full':full_pipe=pipe
        for part in ['validation','test']:
            metadata=dict(analysis='long_history',category=category,partition=part)
            m,b,d,l=evaluate_variants(parts[part],scores[part],target_events(events,category,part),metadata)
            all_metrics.append(m);all_budgets.append(b);all_diffs.append(d);all_leads.append(l)
            w=parts[part][['row_id','pitcher_id','game_id','game_date','season','is_starter','target','next_event_id',
                'next_event_lead_days','next_event_strict_qa']].copy().assign(category=category,partition=part)
            for name,score in scores[part].items():w['score__'+name]=score
            wide.append(w)
            if part=='validation' and category in CFG['focus_categories']:
                c,i=inspection_importance(full_pipe,parts[part],variants['Full'],families,m,category)
                coefs.append(c);importance.append(i)
        for name in variants:
            threshold=np.quantile(scores['validation'][name],.95,method='higher')
            test=parts['test'];mask=scores['test'][name]>=threshold
            fixed.append(dict(category=category,variant=name,validation_threshold=threshold,test_alert_rate=mask.mean(),
                precision=test.target.to_numpy()[mask].mean() if mask.any() else np.nan,
                recall=test.target.to_numpy()[mask].sum()/test.target.sum()))
        # Checkpoint only completed categories; rerun remains deterministic.
        table(pd.concat(all_metrics,ignore_index=True),'family_performance')
        table(pd.concat(all_budgets,ignore_index=True),'family_alert_budgets')
        table(pd.concat(all_diffs,ignore_index=True),'family_event_recall_differences')
    table(pd.DataFrame(inventories),'family_cohort_counts')
    table(pd.concat(all_leads,ignore_index=True),'family_warning_leads')
    table(pd.concat(coefs,ignore_index=True),'family_full_coefficients')
    table(pd.concat(importance,ignore_index=True),'variable_permutation_repeats')
    table(pd.DataFrame(fixed),'family_validation_fixed_thresholds')
    perm=pd.concat(importance).groupby(['category','family','feature']).ap_drop.agg(['mean','std','min','max']).reset_index()
    table(perm,'variable_permutation_summary')
    pd.concat(wide,ignore_index=True).to_parquet(CACHE/'family_predictions.parquet',index=False)
    save_manifest(dict(family_variants=variants))
    return pd.concat(all_metrics,ignore_index=True)


def run_modern(a,events,spec):
    base,_=feature_variants(spec)
    angle=['arm_angle_avg','arm_angle_sd','arm_angle_change_vs_prev_5','arm_angle_z']
    common=a.loc[a.game_date.ge(CFG['modern_start']) & a[angle].notna().all(axis=1)].copy()
    variants={'arm_angle':angle,'Full without arm angle':base['Full'],'Full + arm angle':base['Full']+angle}
    metrics=[];budgets=[];diffs=[];counts=[]
    for category in CFG['categories']:
        f=label_frame(common,category,30)
        parts={p:f.loc[f.partition.eq(p)].copy() for p in ['train','validation','test']}
        nums={p:s.loc[s.target.eq(1),'next_event_id'].nunique() for p,s in parts.items()}
        enough=nums['train']>=30 and min(nums['validation'],nums['test'])>=10
        for p,s in parts.items():counts.append(dict(category=category,partition=p,rows=len(s),events=nums[p],run=enough))
        if not enough:continue
        print(f'Modern common-row arm-angle comparison: {category}; events={nums}',flush=True)
        scores={p:{} for p in ['validation','test']}
        for name,features in variants.items():
            _,(vs,ts)=fit_scores(parts['train'],[parts['validation'],parts['test']],features)
            scores['validation'][name]=vs;scores['test'][name]=ts
        for part in ['validation','test']:
            m,b,d,_=evaluate_variants(parts[part],scores[part],target_events(events,category,part),
                dict(analysis='modern_complete_angle',category=category,partition=part))
            metrics.append(m);budgets.append(b);diffs.append(d)
    table(pd.DataFrame(counts),'modern_cohort_counts')
    table(pd.concat(budgets,ignore_index=True),'modern_alert_budgets')
    table(pd.concat(diffs,ignore_index=True),'modern_event_recall_differences')
    return table(pd.concat(metrics,ignore_index=True),'modern_family_performance')


def run_predictive_sensitivity(a,events,spec):
    variants,_=feature_variants(spec)
    rows=[]
    for category in CFG['focus_categories']:
        for condition in ['Strict QA','Exclude final 6 days']:
            f=label_frame(a,category,30,strict=condition=='Strict QA')
            if condition=='Exclude final 6 days':f=f.loc[f.target.eq(0) | f.next_event_lead_days.ge(7)]
            train=f.loc[f.partition.eq('train')];test=f.loc[f.partition.eq('test')]
            if min(train.target.sum(),test.target.sum())<10:continue
            scores={}
            for name in ['Workload + history','WH + velocity + mechanics','Full']:
                _,(scores[name],)=fit_scores(train,[test],variants[name])
            m,_,_,_=evaluate_variants(test,scores,target_events(events,category,'test',strict=condition=='Strict QA'),
                dict(analysis=condition,category=category,partition='test'))
            rows.append(m)
    result=table(pd.concat(rows,ignore_index=True),'family_sensitivity_refits')
    original=pd.read_parquet(CACHE/'family_predictions.parquet')
    subgroup=[]
    for (category,part),f in original.groupby(['category','partition']):
        if part!='test' or category not in CFG['focus_categories']:continue
        for role in [True,False]:
            s=f.loc[f.is_starter.eq(role)]
            if s.target.sum()<10:continue
            scores={name:s['score__'+name].to_numpy() for name in ['Workload + history','WH + velocity + mechanics','Full']}
            bs,_=paired_bootstrap_ap(s,scores)
            ref=rank_metrics(s.target,scores['Workload + history'])['average_precision']
            for name,score in scores.items():
                met=rank_metrics(s.target,score)
                lo,hi=np.nanquantile(bs[name]-bs['Workload + history'],[.025,.975])
                subgroup.append(dict(category=category,role='Starter' if role else 'Reliever',variant=name,
                    rows=len(s),positive_rows=int(s.target.sum()),events=s.loc[s.target.eq(1),'next_event_id'].nunique(),
                    ap_delta=met['average_precision']-ref,delta_low=lo,delta_high=hi,**met))
    table(pd.DataFrame(subgroup),'family_role_sensitivity')
    return result
