"""Paired uncertainty, whole-pitcher refits, local effects and warning burden."""
import json
import numpy as np
import pandas as pd
from .ucl_nonlinear_common import ROOT,OUT,CACHE,CFG,inputs,table,parquet,write_json,checkpoint,valid_checkpoint
from .ucl_nonlinear_models import fit,scores,load_model
from .ucl_dynamic_data import event_cohort
from .ucl_dynamic_evaluation import metrics,warning,event_bootstrap
from .family_decomposition import paired_bootstrap_ap
from src.evaluation import rank_metrics
from .ucl_nonlinear_effects import reference_sample,effect_tables

def interpretation_models(freeze):
    """Leaders on either development objective plus predeclared departure probes."""
    ids=list(dict.fromkeys(list(freeze['warning_leaders'].values())+list(freeze['ranking_leaders'].values())+
        [freeze['per_representation']['Departure|'+alg] for alg in ['Spline','Boosting']]))
    reps={freeze['catalog'][mid]['representation'] for mid in ids}
    return list(dict.fromkeys(ids+['Linear|'+r for r in sorted(reps)]))

def permute_features(obj,frame,features,repeats,seed):
    """Marginal validation permutation, descriptive under correlated predictors.

    Do not claim conditional/causal importance. Keep every derived field for a
    source together when grouped, including regenerated missingness indicators.
    """
    baseline=rank_metrics(frame.target,scores(obj,frame))['average_precision'];rng=np.random.default_rng(seed);rows=[]
    for feature in features:
        for rep in range(repeats):
            f=frame.copy();f[feature]=f[feature].to_numpy()[rng.permutation(len(f))]
            drop=baseline-rank_metrics(f.target,scores(obj,f))['average_precision']
            rows.append(dict(feature=feature,repeat=rep,ap_drop=drop))
    return pd.DataFrame(rows)

def run_analysis():
    if valid_checkpoint('analysis'):return
    f,events,specs=inputs();freeze=json.loads((OUT/'frozen_protocol.json').read_text());catalog=freeze['catalog']
    selected=freeze['transfer_models'];objects={mid:load_model(mid) for mid in selected}
    train=f.loc[f.partition.eq('train')&f.target.notna()];valid=f.loc[f.partition.eq('validation')&f.target.notna()]
    rows=[];bootrows=[];eventrows=[]
    for part in ['validation','test']:
        g=f.loc[f.partition.eq(part)&f.target.notna()];ev=event_cohort(events,part);ss={mid:scores(objects[mid],g) for mid in selected}
        boot,_=paired_bootstrap_ap(g,ss);eb={}
        for mid,s in ss.items():
            _,detail=warning(g,s,ev);eb[mid]=event_bootstrap(detail)
            for j,b in enumerate(boot[mid]):bootrows.append(dict(partition=part,model_id=mid,replicate=j,average_precision=b,event_recall=eb[mid][j]))
        for mid in selected:
            if catalog[mid]['algorithm']=='Linear':continue
            base='Linear|'+catalog[mid]['representation']
            for metric,collection in [('average_precision',boot),('event_recall_7_30d',eb)]:
                diff=collection[mid]-collection[base];rows.append(dict(partition=part,model_id=mid,reference=base,metric=metric,
                    mean_difference=float(np.nanmean(diff)),ci_low=float(np.nanquantile(diff,.025)),ci_high=float(np.nanquantile(diff,.975)),fraction_positive=float(np.nanmean(diff>0))))
    table(pd.DataFrame(rows),'paired_uncertainty');table(pd.DataFrame(bootrows),'performance_bootstrap')
    ids=interpretation_models(freeze);ref=reference_sample(train);parquet(ref,'training_effect_reference')
    curves=[];interactions=[];permutations=[];refitmetrics=[];plan={}
    for mid in ids:
        spec=catalog[mid];available=spec['features']
        features=[c for c in CFG['interpretation_features'] if c in available]
        # Feature set includes explicit baseball probes, not just permutation winners.
        pairs=[p for p in CFG['interaction_pairs'] if set(p)<=set(available)]
        plan[mid]=dict(features=features,pairs=pairs)
        obj=objects[mid]
        c,i=effect_tables(obj,ref,features,pairs)
        if not c.empty:c['model_id']=mid;c['refit']=-1;curves.append(c)
        if not i.empty:i['model_id']=mid;i['refit']=-1;interactions.append(i)
        if spec['algorithm']!='Linear':
            p=permute_features(obj,valid,available,CFG['permutation_repeats'],CFG['seed']);p['model_id']=mid;p['refit']=-1;permutations.append(p)
    # Same whole-pitcher draws for every model; settings never retuned per draw.
    groups={p:g for p,g in train.groupby('pitcher_id')};people=np.array(list(groups));rng=np.random.default_rng(CFG['seed'])
    write_json(dict(models=ids,effect_features=plan,refit_settings='fixed before resampling; no transfer inspection',reference_rows=len(ref)),OUT/'interpretation_plan.json')
    for rep in range(CFG['training_refits']):
        draw=rng.choice(people,len(people),replace=True);sample=pd.concat([groups[p] for p in draw],ignore_index=True)
        for mid in ids:
            spec=catalog[mid];obj=fit(sample,spec,f'refit_{rep:02d}__'+mid.replace('|','__'))
            s=scores(obj,valid);w,_=warning(valid,s,event_cohort(events,'validation'))
            refitmetrics.append(dict(refit=rep,model_id=mid,**metrics(valid,s),injury_event_recall=w['injury_event_recall'],alerted_events=w['alerted_events']))
            c,i=effect_tables(obj,ref,plan[mid]['features'],plan[mid]['pairs'])
            if not c.empty:c['model_id']=mid;c['refit']=rep;curves.append(c)
            if not i.empty:i['model_id']=mid;i['refit']=rep;interactions.append(i)
            if spec['algorithm']!='Linear':
                p=permute_features(obj,valid,plan[mid]['features'],CFG['refit_permutation_repeats'],CFG['seed']+rep);p['model_id']=mid;p['refit']=rep;permutations.append(p)
    table(pd.DataFrame(refitmetrics),'training_refit_metrics')
    allcurves=pd.concat(curves,ignore_index=True);allinter=pd.concat(interactions,ignore_index=True);allperm=pd.concat(permutations,ignore_index=True)
    table(allcurves,'local_effect_curves');table(allinter,'local_interaction_cells');table(allperm,'permutation_importance')
    baseperm=allperm.loc[allperm.refit.eq(-1)].groupby(['model_id','feature']).ap_drop.agg(mean='mean',sd='std').reset_index()
    table(baseperm.sort_values(['model_id','mean'],ascending=[True,False]),'permutation_summary')
    stability=allperm.loc[allperm.refit.ge(0)].groupby(['model_id','feature']).ap_drop.agg(mean='mean',sd='std',positive_fraction=lambda x:(x>0).mean(),minimum='min',maximum='max').reset_index()
    table(stability,'permutation_stability')
    interaction_summary=[]
    for (mid,rep,left,right),g in allinter.groupby(['model_id','refit','left','right']):
        h=g.loc[g.supported];interaction_summary.append(dict(model_id=mid,refit=rep,left=left,right=right,
            supported_cells=len(h),total_cells=len(g),supported_rows=int(h.rows.sum()),
            weighted_rms_local_contrast=float(np.sqrt(np.average(h.mixed_log_odds**2,weights=h.rows))) if len(h) else np.nan))
    table(pd.DataFrame(interaction_summary),'interaction_strength')
    # Shape summary distinguishes negative and positive departures where +/-1 are supported.
    asymmetry=[]
    for (mid,rep,c),g in allcurves.groupby(['model_id','refit','feature']):
        if not c.endswith('_z'):continue
        g=g.sort_values('x')
        if g.x.min()>-1 or g.x.max()<1:continue
        lo,zero,hi=np.interp([-1,0,1],g.x,g.ale_log_odds)
        asymmetry.append(dict(model_id=mid,refit=rep,feature=c,negative_minus_zero=lo-zero,positive_minus_zero=hi-zero,
            positive_minus_negative=(hi-zero)-(lo-zero),curvature=hi+lo-2*zero))
    table(pd.DataFrame(asymmetry),'departure_shape_summary')
    operational=[]
    for part in ['validation','test']:
        g=f.loc[f.partition.eq(part)&f.analysis_eligible];ev=event_cohort(events,part)
        for mid in selected:
            s=scores(objects[mid],g);alert=s>=freeze['thresholds'][mid]
            op=g.dynamic_next_target&g.dynamic_lead_days.between(7,30)&g.dynamic_next_event_id.isin(ev.injury_event_id)
            operational.append(dict(partition=part,model_id=mid,eligible_rows=len(g),alerts=int(alert.sum()),actual_alert_rate=float(alert.mean()),
                unknown_outcome_alerts=int((alert&g.target.isna()).sum()),known_negative_alerts=int((alert&g.target.eq(0)).sum()),
                known_positive_alerts=int((alert&g.target.eq(1)).sum()),warned_events=g.loc[alert&op,'dynamic_next_event_id'].nunique(),qualifying_events=len(ev)))
    table(pd.DataFrame(operational),'operational_burden')
    checkpoint('analysis',[OUT/'tables'/(n+'.csv') for n in ['paired_uncertainty','training_refit_metrics','local_effect_curves','local_interaction_cells','permutation_importance','operational_burden']]+list((CACHE/'models').glob('refit*.joblib')))
