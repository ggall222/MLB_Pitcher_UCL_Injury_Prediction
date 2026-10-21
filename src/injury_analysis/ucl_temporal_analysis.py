"""Paired pitcher uncertainty, explicit temporal contrasts and limited refits."""
import json
import numpy as np
import pandas as pd
from .ucl_temporal_common import OUT,CACHE,CFG,table,read_table,parquet,checkpoint,valid_checkpoint,interval_name,freeze,write_json
from .ucl_temporal_data import prepare,interval_target
from .ucl_temporal_models import fit,load,scores,evaluate
from .ucl_dynamic_evaluation import event_bootstrap
from .family_decomposition import weighted_ap

def bootstrap_ap(frame, predictions):
    """All models AND intervals share the same whole-pitcher bootstrap draws."""
    ids, inv=np.unique(frame.pitcher_id,return_inverse=True)
    rng=np.random.default_rng(CFG['seed'])
    draws=np.array([np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))
                    for _ in range(CFG['bootstrap_replicates'])])
    y=frame.target.to_numpy(dtype=int)
    prev=np.array([np.dot(w[inv],y)/w[inv].sum() for w in draws])
    result={}
    for name,score in predictions.items():
        order=np.argsort(-np.asarray(score),kind='stable')
        yy=y[order];ss=np.asarray(score)[order];ii=inv[order]
        ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
        values=[]
        for draw in draws:
            w=draw[ii];tp=np.cumsum(w*yy)[ends];nn=np.cumsum(w)[ends]
            precision=np.divide(tp,nn,out=np.zeros(len(tp),dtype=float),where=nn>0)
            values.append(np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1] if tp[-1] else np.nan)
        result[name]=np.array(values)
    return result,prev

def uncertainty(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    if not len(x):return dict(ci_low=np.nan,ci_high=np.nan,bootstrap_positive_fraction=np.nan,valid_replicates=0)
    lo,hi=np.quantile(x,[.025,.975])
    return dict(ci_low=float(lo),ci_high=float(hi),bootstrap_positive_fraction=float((x>0).mean()),valid_replicates=len(x))

def event_draws(details, model):
    g=details.loc[details.model.eq(model)].sort_values('injury_event_id')
    return event_bootstrap(g)

def analyze_partition(part='validation'):
    stage='uncertainty_'+part
    if valid_checkpoint(stage): return
    f,events,groups,models=prepare()
    metric=read_table('development_metrics' if part=='validation' else 'transfer_metrics')
    metric=metric.loc[metric.partition.eq(part)]
    details=read_table('development_event_warnings' if part=='validation' else 'transfer_event_warnings')
    matrix=[];reference=[];bootrows=[];warningrows=[];boot_by_h={};key_reference=None
    for lo,hi in CFG['intervals']:
        interval=interval_name(lo,hi);g=interval_target(f,lo,hi)
        g=g.loc[g.partition.eq(part)&g.target.notna()]
        p=pd.read_parquet(CACHE/f'predictions_{part}_{interval}.parquet')
        assert g.row_id.tolist()==p.row_id.tolist()
        if key_reference is None:key_reference=g.row_id.tolist()
        else:assert key_reference==g.row_id.tolist(),'Common risk set changed between intervals'
        columns=list(models)+[c for c in p if c.startswith('Permute ')]
        boot,prev=bootstrap_ap(g,{c:p[c].to_numpy() for c in columns})
        boot_by_h[interval]=(boot,prev)
        b=pd.DataFrame(boot).assign(prevalence=prev,replicate=np.arange(len(prev)),interval=interval,partition=part)
        bootrows.append(b)
        m=metric.loc[metric.interval.eq(interval)].set_index('model')
        ev=details.loc[details.interval.eq(interval)]
        warning_boot={name:event_draws(ev,name) for name in models}
        for name in models:
            warningrows.append(dict(partition=part,interval=interval,model=name,
                recall=m.loc[name,'injury_event_recall'],**uncertainty(warning_boot[name])))
        for family in groups:
            only='Only | '+family;without='Without | '+family
            pointprev=float(g.target.mean())
            loss=float(m.loc['Full union','average_precision']-m.loc[without,'average_precision'])
            ab=boot['Full union']-boot[without]
            perm_names=[f'Permute {r} | {family}' for r in range(CFG['permutation_repeats'])]
            perm_point=np.mean([weighted_ap(g.target,p[c],np.ones(len(g))) for c in perm_names])
            perm_draw=boot['Full union']-np.mean([boot[c] for c in perm_names],axis=0)
            warn_delta=warning_boot['Full union']-warning_boot[without]
            row=dict(partition=part,interval=interval,family=family,fields=len(groups[family]),
                     rows=len(g),positive_rows=int(g.target.sum()),represented_events=m.loc[only,'represented_events'],
                     prevalence=pointprev,standalone_ap=m.loc[only,'average_precision'],
                     standalone_lift=m.loc[only,'pr_lift'],ablation_ap_loss=loss,ablation_lift_loss=loss/pointprev,
                     permutation_ap_loss=m.loc['Full union','average_precision']-perm_point,
                     standalone_warned_events=m.loc[only,'alerted_events'],
                     full_warning_recall=m.loc['Full union','injury_event_recall'],
                     warning_recall_loss=m.loc['Full union','injury_event_recall']-m.loc[without,'injury_event_recall'])
            for prefix,values in [('standalone_lift',boot[only]/prev),('ablation',ab),
                                  ('ablation_lift',ab/prev),('permutation',perm_draw),('warning',warn_delta)]:
                row.update({prefix+'_'+k:v for k,v in uncertainty(values).items()})
            matrix.append(row)
        for name in ['Full union']+CFG['representations']:
            row=dict(partition=part,interval=interval,model=name,
                average_precision=m.loc[name,'average_precision'],pr_lift=m.loc[name,'pr_lift'],
                warned_events=m.loc[name,'alerted_events'],all_events=m.loc[name,'qualifying_events'])
            row.update({f'lift_{k}':v for k,v in uncertainty(boot[name]/prev).items()})
            ref=boot['Departure']/prev-boot['Profile']/prev
            row.update({f'departure_minus_profile_{k}':v for k,v in uncertainty(ref).items()})
            reference.append(row)
        print('Pitcher bootstrap complete: '+part+' / '+interval,flush=True)
    table(pd.DataFrame(matrix),'family_matrix_'+part)
    table(pd.DataFrame(reference),'reference_uncertainty_'+part)
    table(pd.DataFrame(warningrows),'warning_uncertainty_'+part)
    parquet(pd.concat(bootrows,ignore_index=True),'bootstrap_'+part)
    # Joint centered max-t bands across all 33 predeclared family-time contrasts.
    # Controls exploratory multiplicity in these contrasts, not every analysis.
    mat=pd.DataFrame(matrix);contrasts=[];draws=[]
    far='31-60';bf,pf=boot_by_h[far]
    for family in groups:
        without='Without | '+family
        for near in ['1-7','8-14','15-30']:
            bn,pn=boot_by_h[near]
            d=(bn['Full union']-bn[without])/pn-(bf['Full union']-bf[without])/pf
            point=float(mat.loc[mat.family.eq(family)&mat.interval.eq(near),'ablation_lift_loss'].iloc[0]-
                        mat.loc[mat.family.eq(family)&mat.interval.eq(far),'ablation_lift_loss'].iloc[0])
            contrasts.append(dict(partition=part,family=family,contrast=near+' minus '+far,
                                  ablation_lift_difference=point,**uncertainty(d)))
            draws.append(d)
    draws=np.array(draws);points=np.array([r['ablation_lift_difference'] for r in contrasts])
    se=np.nanstd(draws,axis=1,ddof=1)
    standardized=np.divide(draws-points[:,None],se[:,None],out=np.zeros_like(draws),where=se[:,None]>0)
    crit=float(np.nanquantile(np.nanmax(np.abs(standardized),axis=0),.95))
    for i,row in enumerate(contrasts):
        row.update(simultaneous_low=points[i]-crit*se[i],simultaneous_high=points[i]+crit*se[i],
                   simultaneous_critical_value=crit,comparisons=len(contrasts))
    table(pd.DataFrame(contrasts),'temporal_contrasts_'+part)
    # The two frozen representation hypotheses are retained even when negative.
    rr=[]
    for name in ['Departure','Profile','Background','Previous 62','Stable 62']:
        bn,pn=boot_by_h['1-7']
        d=bn[name]/pn-bf[name]/pf
        point=float(metric.loc[metric.interval.eq('1-7')&metric.model.eq(name),'pr_lift'].iloc[0]-
                    metric.loc[metric.interval.eq(far)&metric.model.eq(name),'pr_lift'].iloc[0])
        rr.append(dict(partition=part,model=name,contrast='1-7 minus 31-60 PR lift',
                       lift_difference=point,**uncertainty(d)))
    table(pd.DataFrame(rr),'reference_temporal_contrasts_'+part)
    checkpoint(stage,[OUT/'tables'/f'{n}_{part}.csv' for n in ['family_matrix','reference_uncertainty',
        'warning_uncertainty','temporal_contrasts','reference_temporal_contrasts']]+[CACHE/f'bootstrap_{part}.parquet'])

def nominate():
    """Limited development-only follow-up; never called with transfer metrics."""
    p=OUT/'candidate_families.json'
    if p.exists():return json.loads(p.read_text())
    m=read_table('family_matrix_validation')
    ranking=m.groupby('family').agg(mean_ablation_lift=('ablation_lift_loss','mean'),
        positive_intervals=('ablation_ap_loss',lambda s:int((s>0).sum())),
        lower_bound_positive_intervals=('ablation_ci_low',lambda s:int((s>0).sum())),
        mean_permutation_ap_loss=('permutation_ap_loss','mean')).reset_index()
    ranking=ranking.sort_values(['mean_ablation_lift','family'],ascending=[False,True])
    chosen=ranking.head(CFG['candidate_families']).family.tolist()
    value=dict(families=chosen,rule='Highest mean validation ablation loss divided by interval prevalence; exploratory nominees, not established temporal mechanisms',
               selected_from='2022–2023 only',selected_at_utc=pd.Timestamp.now(tz='UTC').isoformat())
    table(ranking,'candidate_ranking');write_json(value,p)
    return value

def run_refits():
    if valid_checkpoint('training_refits'):return
    f,events,groups,models=prepare();chosen=nominate()['families']
    names=['Full union','Profile','Departure']+[prefix+c for c in chosen for prefix in ['Only | ','Without | ']]
    tr0=f.loc[f.partition.eq('train')&f.target.notna()]
    ids=np.sort(tr0.pitcher_id.unique())
    rng=np.random.default_rng(CFG['seed']+1000)
    draws=[rng.choice(ids,len(ids),replace=True) for _ in range(CFG['training_refits'])]
    rows=[]
    for lo,hi in CFG['intervals']:
        interval=interval_name(lo,hi);g=interval_target(f,lo,hi)
        tr=g.loc[g.partition.eq('train')&g.target.notna()]
        va=g.loc[g.partition.eq('validation')&g.target.notna()]
        by={pid:x for pid,x in tr.groupby('pitcher_id',sort=False)}
        for r,draw in enumerate(draws):
            sample=pd.concat([by[pid] for pid in draw],ignore_index=True)
            for name in names:
                obj=fit(sample,models[name],interval,name,r)
                row,_=evaluate(va,scores(obj,va),events,'validation',lo,hi,name,len(models[name]))
                rows.append(dict(refit=r,**row))
    results=table(pd.DataFrame(rows),'training_refit_metrics')
    losses=[]
    for (interval,r),g in results.groupby(['interval','refit']):
        g=g.set_index('model');full=g.loc['Full union','average_precision']
        for family in chosen:
            losses.append(dict(interval=interval,refit=r,family=family,
                ablation_ap_loss=full-g.loc['Without | '+family,'average_precision'],
                ablation_lift_loss=(full-g.loc['Without | '+family,'average_precision'])/g.loc['Full union','prevalence'],
                standalone_lift=g.loc['Only | '+family,'pr_lift']))
    table(pd.DataFrame(losses),'training_refit_family_losses')
    checkpoint('training_refits',[OUT/'tables/training_refit_metrics.csv',
        OUT/'tables/training_refit_family_losses.csv',OUT/'candidate_families.json']+
        list((CACHE/'models').glob('*__refit*.joblib')))

def finalize_development():
    f,events,groups,models=prepare()
    for stage in ['development_models','uncertainty_validation','followup_sensitivity','training_refits']:
        if not valid_checkpoint(stage):raise ValueError('Incomplete development stage: '+stage)
    chosen=nominate()
    matrix=read_table('family_matrix_validation');contrasts=read_table('temporal_contrasts_validation')
    supported=contrasts.loc[contrasts.simultaneous_low.gt(0)|contrasts.simultaneous_high.lt(0)]
    signatures={interval_name(lo,hi)+'|'+name:load(interval_name(lo,hi),name)['signature']
                for lo,hi in CFG['intervals'] for name in models}
    value=dict(config=CFG,models=models,families=groups,candidate_families=chosen['families'],
        candidate_basis=chosen['rule'],fit_signatures=signatures,
        multiplicity_adjusted_temporal_contrasts=supported.to_dict('records'),
        development_summary=dict(families=len(groups),intervals=len(CFG['intervals']),
            positive_ablation_cells=int(matrix.ablation_ap_loss.gt(0).sum()),
            ablation_pointwise_lower_above_zero=int(matrix.ablation_ci_low.gt(0).sum()),
            simultaneous_temporal_contrasts_excluding_zero=len(supported)),
        transfer_selection_allowed=False,refit_interpretation='Six development pitcher refits conditional on two nominated families; not nested selection validation')
    freeze(value,OUT/'development_freeze.json')
