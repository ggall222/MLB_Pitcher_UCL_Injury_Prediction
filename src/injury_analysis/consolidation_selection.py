"""Evidence-linked candidate sets, validation selection, coverage and frozen transfer."""
import datetime as dt
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from .consolidation_common import ROOT,CFG,CACHE,OUT,table,read_table,inputs,sha,save_manifest
from .consolidation_models import feature_sets,fit_named,scores,source_coefficients,summary_metrics
from .family_decomposition import paired_bootstrap_ap,target_events
from .profile_departure import PROFILE,DEPARTURE,evaluate_scores,event_detection
from .signals_common import mean_ci
from .consolidation_inventory import GROUPS
from src.evaluation import evaluate_predictions


def candidate_sets(spec):
    core,reps,_=feature_sets();stability=read_table('feature_stability').set_index('feature')
    inv=read_table('feature_inventory').set_index('feature');clusters=read_table('correlation_clusters')
    candidates={'Full feasible':core,'Previous 62':spec.loc[spec.selected_for_prediction,'name'].tolist(),
        'Correlation representatives':reps,'Profile only':[c for c in PROFILE if c in core],
        'Departure only':[c for c in DEPARTURE if c in core],'Profile plus departure':[c for c in PROFILE+DEPARTURE if c in core],
        'Background context':[c for c in core if inv.loc[c,'feature_group']==GROUPS[12]]}
    coefs=read_table('elastic_net_selected_coefficients')
    candidates['Elastic Net selected values']=coefs.loc[coefs.selected,'feature'].tolist()
    for threshold in CFG['selection_frequencies']:
        eligible=set(stability.index[stability.selection_frequency.ge(threshold)])
        chosen=[]
        for _,g in clusters.loc[clusters.feature.isin(eligible)].groupby(['correlation_cluster','signal_interpretation']):
            ranked=stability.reindex(g.feature).sort_values(['selection_frequency','median_absolute_coefficient'],ascending=False,kind='stable')
            chosen.append(ranked.index[0])
        if chosen:candidates[f'Stable representatives {threshold:.0%}']=sorted(chosen)
    # Identical feature sets do not get multiple opportunities in model selection.
    seen=set();unique={}
    for name,cols in candidates.items():
        key=tuple(sorted(cols))
        if key and key not in seen:unique[name]=list(cols);seen.add(key)
    return unique


def choose_candidate(performance,bootstraps):
    """Predeclared AP-tolerance then warning-budget gate, then smaller set.

    A paired bootstrap SE is a pragmatic development tolerance, NOT evidence of
    equivalence. Among AP-competitive candidates, stay within two detected events
    of the best 5% budget warning count. Never inspect test metrics here.
    """
    valid=performance.loc[performance.partition.eq('validation')].copy()
    if valid.empty:raise ValueError('Need validation performance for candidate selection')
    best=valid.sort_values(['average_precision','features'],ascending=[False,True]).iloc[0]
    valid['ap_loss_from_best']=best.average_precision-valid.average_precision
    valid['paired_ap_se']=[np.nanstd(bootstraps[best.variant]-bootstraps[n],ddof=1) for n in valid.variant]
    valid['ap_competitive']=valid.ap_loss_from_best.le(valid.paired_ap_se+1e-12)
    warning_best=valid.loc[valid.ap_competitive,'alerted_events'].max()
    valid['warning_competitive']=valid.alerted_events.ge(warning_best-CFG['maximum_validation_warning_events_lost'])
    eligible=valid.loc[valid.ap_competitive&valid.warning_competitive]
    chosen=eligible.sort_values(['features','average_precision','variant'],ascending=[True,False,True]).iloc[0].variant
    valid['selected']=valid.variant.eq(chosen)
    return chosen,valid


def run_candidate_selection():
    f,events,spec=inputs();train=f.loc[f.partition.eq('train')];valid=f.loc[f.partition.eq('validation')]
    variants=candidate_sets(spec);metrics=[];predictions={};models={}
    for i,(name,features) in enumerate(variants.items()):
        existing={'Full feasible':'family_Full_feasible',
            'Previous 62':'family_Previous_62_(same_preprocessing)',
            'Correlation representatives':'family_Correlation_representatives',
            'Background context':'family_Only_family_12'}
        model_name=existing.get(name,f'candidate_{i:02d}')
        obj,score=fit_named(train,valid,features,model_name)
        models[name]=model_name;predictions[name]=score
        event_interval=mean_ci(event_detection(valid,score,target_events(events,'Elbow / UCL','validation')),'detected')
        metrics.append(dict(variant=name,partition='validation',features=len(features),design_columns=len(obj['encoder'].names),
                            event_low=event_interval['low'],event_high=event_interval['high'],
                            **summary_metrics(valid,score,events)))
    bs,_=paired_bootstrap_ap(valid,predictions)
    for r in metrics:r['ap_low'],r['ap_high']=np.nanquantile(bs[r['variant']],[.025,.975])
    chosen,performance=choose_candidate(pd.DataFrame(metrics),bs)
    table(performance,'candidate_reduction_curve')
    raw=valid[['row_id','pitcher_id','game_id','game_date','target']].copy()
    for name,score in predictions.items():raw['score__'+name]=score
    raw.to_parquet(CACHE/'candidate_validation_predictions.parquet',index=False)
    definitions={k:dict(features=v,model_name=models[k]) for k,v in variants.items()}
    (CACHE/'candidate_specifications.json').write_text(json.dumps(definitions,indent=2))
    result=dict(selected=chosen,features=variants[chosen],model_name=models[chosen],
        selection_rule='Within one paired-bootstrap AP SE of highest validation AP; within two detected events of best warning among AP-competitive models; smallest input set',
        parameters_chosen_without_test=True)
    (CACHE/'candidate_selection.json').write_text(json.dumps(result,indent=2))
    print(f'Candidate: {chosen}; {len(variants[chosen])} fields. Selected on validation only.',flush=True)
    return performance


def run_conditional_checks():
    f,events,_=inputs();train=f.loc[f.partition.eq('train')];valid=f.loc[f.partition.eq('validation')]
    selection=json.loads((CACHE/'candidate_selection.json').read_text());features=selection['features']
    obj=joblib.load(CACHE/'models'/(selection['model_name']+'.joblib'));base=scores(obj,valid)
    main=summary_metrics(valid,base,events);rows=[];scoremap={'Selected':base}
    clusters=read_table('correlation_clusters');stability=read_table('feature_stability').set_index('feature')
    for i,feature in enumerate(features):
        if len(features)>1:
            _,score=fit_named(train,valid,[c for c in features if c!=feature],f'conditional_drop_{i:03d}')
            m=summary_metrics(valid,score,events);name='drop__'+feature;scoremap[name]=score
            rows.append(dict(feature=feature,comparison='drop',alternative='',model=name,
                ap_loss=main['average_precision']-m['average_precision'],event_recall_loss=main['injury_event_recall']-m['injury_event_recall']))
        # A closest same-group correlation alternative tests exchangeability.
        cluster=clusters.loc[clusters.feature.eq(feature)]
        if cluster.empty:continue
        alt=clusters.loc[clusters.correlation_cluster.eq(cluster.correlation_cluster.iloc[0])&~clusters.feature.isin(features),'feature'].tolist()
        alt=[c for c in alt if c in stability.index]
        if not alt:continue
        alternative=stability.reindex(alt).sort_values(['selection_frequency','median_absolute_coefficient'],ascending=False).index[0]
        for kind,cols in [('replace',[alternative if c==feature else c for c in features]),('add',features+[alternative])]:
            _,score=fit_named(train,valid,cols,f'conditional_{kind}_{i:03d}')
            m=summary_metrics(valid,score,events);name=kind+'__'+feature;scoremap[name]=score
            rows.append(dict(feature=feature,comparison=kind,alternative=alternative,model=name,
                ap_loss=main['average_precision']-m['average_precision'],event_recall_loss=main['injury_event_recall']-m['injury_event_recall']))
    bs,_=paired_bootstrap_ap(valid,scoremap)
    for row in rows:row['low'],row['high']=np.nanquantile(bs['Selected']-bs[row['model']],[.025,.975])
    return table(pd.DataFrame(rows),'conditional_feature_checks')


def run_coverage_checks():
    f,events,_=inputs();train=f.loc[f.partition.eq('train')];valid=f.loc[f.partition.eq('validation')]
    selection=json.loads((CACHE/'candidate_selection.json').read_text());core=selection['features'];_,_,support=feature_sets()
    rows=[];scoremap={};designs={};strata=[]
    for name,features,flags in [('Core, no flags',core,False),('Core + missing flags',core,True),
        ('Core + support, no flags',core+support,False),('Core + support + flags',core+support,True),('Support only',support,False)]:
        model=selection['model_name'] if features==core and flags else 'coverage_'+str(len(rows))
        obj,score=fit_named(train,valid,features,model,indicators=flags)
        designs[name]=dict(model_name=model,features=features,indicators=flags);scoremap[name]=score
        rows.append(dict(variant=name,features=len(features),design_columns=len(obj['encoder'].names),**summary_metrics(valid,score,events)))
        v=valid[['season','is_starter','target']].copy();v['score']=score
        v['core_missing_fraction']=valid[core].isna().mean(axis=1)
        for (season,role),g in v.groupby(['season','is_starter']):
            strata.append(dict(variant=name,season=season,is_starter=bool(role),rows=len(g),positive_rows=int(g.target.sum()),
                mean_prediction=g.score.mean(),prevalence=g.target.mean(),mean_missing_fraction=g.core_missing_fraction.mean()))
    bs,_=paired_bootstrap_ap(valid,scoremap)
    for row in rows:
        delta=bs[row['variant']]-bs['Core, no flags']
        row['ap_difference_from_no_flags']=row['average_precision']-rows[0]['average_precision']
        row['ap_difference_low'],row['ap_difference_high']=np.nanquantile(delta,[.025,.975])
    table(pd.DataFrame(rows),'coverage_comparison');table(pd.DataFrame(strata),'coverage_role_season_audit')
    (CACHE/'coverage_specifications.json').write_text(json.dumps(designs,indent=2))
    return pd.DataFrame(rows)


def freeze_candidate():
    selection=json.loads((CACHE/'candidate_selection.json').read_text());names=selection['features']
    inv=read_table('reconciled_feature_inventory');stability=read_table('feature_stability');clusters=read_table('correlation_clusters')
    checks=read_table('conditional_feature_checks')
    retained=inv.loc[inv.feature.isin(names)].merge(stability,on='feature',how='left',validate='one_to_one').merge(
        clusters[['feature','correlation_cluster']],on='feature',how='left',validate='one_to_one')
    retained=retained.merge(checks.loc[checks.comparison.eq('drop'),['feature','ap_loss','low','high','event_recall_loss']],on='feature',how='left',validate='one_to_one')
    retained['reason_retained']=[f'{r.representation.capitalize()} of {r.underlying_concept}; represents {r.signal_interpretation.lower()} in the validation-selected joint set. Individual incremental evidence is reported separately.' for r in retained.itertuples()]
    retained['closest_redundant_alternatives']=retained.feature.map({r.feature:';'.join(clusters.loc[
        clusters.correlation_cluster.eq(r.correlation_cluster)&~clusters.feature.eq(r.feature),'feature']) for r in retained.itertuples()})
    retained['incremental_evidence_status']=np.where(retained.low.gt(0),'Positive conditional AP evidence under fixed comparison',
        'Conditional AP uncertain; retained as part of the selected joint representation')
    table(retained,'retained_features')
    exclusions=inv.loc[~inv.feature.isin(names)].merge(clusters[['feature','correlation_cluster','representative_feature']],on='feature',how='left')
    exclusions=exclusions.merge(stability,on='feature',how='left')
    exclusions['recommendation_reason']=exclusions.screen_exclusion.fillna('')
    m=exclusions.recommendation_reason.eq('')&exclusions.support_only_sensitivity
    exclusions.loc[m,'recommendation_reason']='Availability/support predictor kept in separate sensitivity; not in primary substantive pool'
    m=exclusions.recommendation_reason.eq('')
    exclusions.loc[m,'recommendation_reason']='Not in validation-selected compact set; see cluster, sparse stability and nested representation comparisons. Not proof of no value.'
    table(exclusions,'excluded_features')
    mapping=[];rep=read_table('representation_comparisons')
    for group,g in inv.groupby('feature_group'):
        keep=retained.loc[retained.feature_group.eq(group)]
        mapping.append(dict(concept_group=group,candidate_representations=';'.join(sorted(g.representation.unique())),
            candidate_features=len(g),feasible_features=int(g.eligible_predictor.sum()),
            correlation_clusters=clusters.loc[clusters.feature_group.eq(group),'correlation_cluster'].nunique(),
            retained_features=';'.join(keep.feature),mean_selection_frequency=keep.selection_frequency.mean(),
            timing_roles=';'.join(sorted(keep.signal_interpretation.unique())),
            exclusion_rationale='; '.join(sorted(exclusions.loc[exclusions.feature_group.eq(group),'recommendation_reason'].unique()))))
    table(pd.DataFrame(mapping),'final_feature_map')
    fitted=joblib.load(CACHE/'models'/(selection['model_name']+'.joblib'))
    config=dict(status='Development candidate frozen before this stage scores 2024–2025; future untouched cohort required',
        endpoint='Broad initial Elbow / UCL proxy; adjudicate confirmed UCL before confirmatory modeling',
        prediction_time='after completed appearance',horizon_days=30,warning_window_days=[7,30],alert_budget=.05,
        selection=selection,preprocessing=dict(median_imputation='training only',winsor_quantiles=CFG['winsor_quantiles'],
            standardization='training only',missing_indicators=True,total_design_columns=len(fitted['encoder'].names)),
        features=[dict(name=r.feature,group=r.feature_group,concept=r.underlying_concept,representation=r.representation,
            rationale=r.reason_retained,timing=r.timing_availability,signal_type=r.signal_interpretation) for r in retained.itertuples()])
    output=ROOT/'config/ucl_candidate_features_v2.yaml';output.write_text(yaml.safe_dump(config,sort_keys=False))
    freeze=dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),candidate_file=str(output.relative_to(ROOT)),
        candidate_sha256=sha(output),selection=selection,model_sha256=sha(CACHE/'models'/(selection['model_name']+'.joblib')),
        decision_tables={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'tables').glob('*.csv')})
    (OUT/'candidate_freeze.json').write_text(json.dumps(freeze,indent=2))
    print(f'Frozen {len(names)} features before test transfer evaluation.',flush=True)
    return config


def run_frozen_transfer():
    freeze=json.loads((OUT/'candidate_freeze.json').read_text())
    if sha(ROOT/freeze['candidate_file'])!=freeze['candidate_sha256']:raise ValueError('Frozen candidate changed')
    f,events,_=inputs();specs=json.loads((CACHE/'candidate_specifications.json').read_text())
    chosen=freeze['selection']['selected'];names=list(dict.fromkeys([chosen,'Full feasible','Previous 62','Profile only','Departure only','Profile plus departure']))
    all_metrics=[];all_diffs=[];all_budgets=[];all_leads=[];thresholds=[];pred=[];calibration=[];edges={}
    for part in ['validation','test']:
        frame=f.loc[f.partition.eq(part)].copy();scoremap={}
        for name in names:
            obj=joblib.load(CACHE/'models'/(specs[name]['model_name']+'.joblib'))
            scoremap[name]=scores(obj,frame)
        contrasts=[(chosen,n) for n in names if n!=chosen]
        m,b,l,d=evaluate_scores(frame,scoremap,target_events(events,'Elbow / UCL',part),'Elbow / UCL',part,
            analysis='Frozen consolidation transfer',contrasts=contrasts)
        all_metrics.append(m);all_budgets.append(b);all_leads.append(l);all_diffs.append(d)
        wide=frame[['row_id','pitcher_id','game_id','game_date','season','target','next_event_id']].copy()
        for name,score in scoremap.items():wide['score__'+name]=score
        for name,score in scoremap.items():
            if part=='validation':edges[name]=np.unique(np.r_[0,np.quantile(score,np.arange(.1,1,.1)),1])
            bins=pd.cut(score,edges[name],include_lowest=True,labels=False)
            cf=pd.DataFrame({'bin':bins,'score':score,'target':frame.target.to_numpy()})
            for b,g in cf.groupby('bin'):
                calibration.append(dict(partition=part,variant=name,bin=int(b),rows=len(g),positive_rows=int(g.target.sum()),
                    mean_prediction=g.score.mean(),observed_rate=g.target.mean(),
                    lower=float(edges[name][int(b)]),upper=float(edges[name][int(b)+1])))
        wide['partition']=part;pred.append(wide)
        if part=='validation':validation_scores=scoremap
        else:
            for name,score in scoremap.items():
                cutoff=np.quantile(validation_scores[name],.95,method='higher');alerts=score>=cutoff
                thresholds.append(dict(variant=name,validation_95th_percentile=cutoff,test_alert_rate=alerts.mean(),
                    test_precision=frame.target.to_numpy()[alerts].mean() if alerts.any() else np.nan))
    table(pd.concat(all_metrics,ignore_index=True),'frozen_performance')
    table(pd.concat(all_budgets,ignore_index=True),'frozen_alert_budgets')
    table(pd.concat(all_diffs,ignore_index=True),'frozen_paired_differences')
    table(pd.concat(all_leads,ignore_index=True),'frozen_warning_leads')
    table(pd.DataFrame(thresholds),'validation_fixed_thresholds')
    table(pd.DataFrame(calibration),'calibration_bins')
    pd.concat(pred,ignore_index=True).to_parquet(CACHE/'frozen_predictions.parquet',index=False)
    assert sha(ROOT/freeze['candidate_file'])==freeze['candidate_sha256']
    save_manifest(dict(candidate_sha256=freeze['candidate_sha256']))
