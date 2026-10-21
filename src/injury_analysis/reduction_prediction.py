"""Common-row temporal PCA comparisons, paired evaluation and training refits."""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import CFG as ORIGINAL_CFG, CATEGORIES
from .cohort import label_frame
from .family_decomposition import (feature_variants, paired_bootstrap_ap, target_events,
                                   opportunity_table, random_detection_probability)
from .signals_common import mean_ci, CACHE as SIGNAL_CACHE
from .reduction import CFG, CACHE, table, read_table, fit_model, BINARY, save_manifest
from src.evaluation import rank_metrics, evaluate_predictions, alert_mask


def variants(spec):
    families, groups = feature_variants(spec)
    full = families['Full']
    continuous = [c for c in full if c not in BINARY]
    statcast = [c for c in full if c not in families['Workload + history']]
    result = {'Original 62': (full, [], None),
              'Workload + history': (families['Workload + history'], [], None)}
    for k in CFG['fixed_components']:
        result[f'PCA {k}'] = (full, continuous, k)
    result['PCA 90%'] = (full, continuous, CFG['variance_fraction'])
    result['Full-rank PCA control'] = (full, continuous, 'all')
    result['WH + Statcast PCA 10'] = (full, statcast, 10)
    result['WH + Statcast PCA 90%'] = (full, statcast, CFG['variance_fraction'])
    result['Role/hand + missingness'] = (full, continuous, 0)
    return result


def choose_dimension(validation_metrics):
    """Validation AP only; prefer the smaller design for exact ties."""
    candidates = validation_metrics.loc[validation_metrics.variant.isin(
        [f'PCA {k}' for k in CFG['fixed_components']]+['PCA 90%'])]
    return candidates.sort_values(['average_precision','design_dimensions'],ascending=[False,True]).iloc[0].variant


def evaluate(frame, scores, events, category, partition, dimensions):
    bs, prevalence_bs = paired_bootstrap_ap(frame, scores)
    metrics, budget_rows, event_rows, leads = [], [], [], []
    opp = opportunity_table(frame, events)
    counts = opp.groupby('injury_event_id').size().reindex(events.injury_event_id,fill_value=0)
    detections = {}
    for name, score in scores.items():
        r = dict(category=category,partition=partition,variant=name,rows=len(frame),positive_rows=int(frame.target.sum()),
                 represented_events=frame.loc[frame.target.eq(1),'next_event_id'].nunique(),
                 prevalence=frame.target.mean(),design_dimensions=dimensions[name],**rank_metrics(frame.target,score))
        r['pr_lift'] = r['average_precision']/r['prevalence']
        r['ap_low'],r['ap_high'] = np.nanquantile(bs[name],[.025,.975])
        for reference, suffix in [('Original 62','original'),('Workload + history','wh')]:
            r['ap_delta_'+suffix] = r['average_precision']-rank_metrics(frame.target,scores[reference])['average_precision']
            r['delta_low_'+suffix],r['delta_high_'+suffix] = np.nanquantile(bs[name]-bs[reference],[.025,.975])
        metrics.append(r)
        f = frame.copy().reset_index(drop=True)
        f['score'] = score
        b,l = evaluate_predictions(f,events,target='target',rates=ORIGINAL_CFG['alert_rates'],warning_window=(7,30))
        b['random_expected_event_recall'] = [np.mean([random_detection_probability(len(f),int(len(f)*rate),int(m))
                                             for m in counts]) for rate in b.alert_rate]
        budget_rows.append(b.assign(category=category,partition=partition,variant=name))
        leads.append(l.assign(category=category,partition=partition,variant=name))
        hit = set(opp.loc[opp.eval_row.isin(np.flatnonzero(alert_mask(f,'score',.05))),'injury_event_id'])
        detections[name] = events.injury_event_id.isin(hit).to_numpy(dtype=float)
    for name in scores:
        ev = events[['pitcher_id','injury_event_id']].copy()
        ev['difference'] = detections[name]-detections['Original 62']
        event_rows.append(dict(category=category,partition=partition,variant=name,reference='Original 62',
                               **mean_ci(ev,'difference')))
    return pd.DataFrame(metrics),pd.concat(budget_rows),pd.DataFrame(event_rows),pd.concat(leads)


def run_prediction(a, events, spec):
    settings = variants(spec)
    table(pd.DataFrame([dict(variant=n,source_features=len(f),rotated_features=len(r),requested_components=k,
        feature_names=';'.join(f),rotated_names=';'.join(r)) for n,(f,r,k) in settings.items()]),'predictive_designs')
    results, budgets, diffs, leads, inventory, selection, wide, dimensions, fixed, sanity, coefs = ([] for _ in range(11))
    legacy=pd.read_parquet(SIGNAL_CACHE/'family_predictions.parquet')
    for category in CATEGORIES:
        f = label_frame(a,category,30)
        parts = {p:f.loc[f.partition.eq(p)].copy() for p in ['train','validation','test']}
        counts = {p:x.loc[x.target.eq(1),'next_event_id'].nunique() for p,x in parts.items()}
        enough = counts['train']>=30 and counts['validation']>=10 and counts['test']>=10
        for p,x in parts.items():
            inventory.append(dict(category=category,partition=p,rows=len(x),events=counts[p],positive_rows=int(x.target.sum()),run=enough))
        if not enough:
            continue
        print(f'PCA predictive comparison: {category}; events={counts}',flush=True)
        train, valid, test = [parts[p] for p in ['train','validation','test']]
        scores = {'validation':{},'test':{}}
        designs, fitted = {}, {}
        for name, (features,reduced,k) in settings.items():
            encoder,model,(vs,ts) = fit_model(train,[valid,test],features,reduced,k)
            scores['validation'][name],scores['test'][name] = vs,ts
            fitted[name] = (encoder,model)
            designs[name] = encoder.output_dimensions
            dimensions.append(dict(category=category,variant=name,source_features=len(features),
                missingness_indicators=len(encoder.names)-len(features),retained_components=encoder.pca.n_components_ if encoder.pca else 0,
                unrotated_columns=len(encoder.keep_index),design_dimensions=encoder.output_dimensions,
                retained_variance=encoder.pca.explained_variance_ratio_.sum() if encoder.pca else (0. if reduced else 1.)))
            beta = encoder.effective_coefficients(model)
            coefs.extend(dict(category=category,variant=name,feature=n,standardized_log_odds=b)
                         for n,b in zip(encoder.names,beta))
        for p in ['validation','test']:
            m,b,d,l = evaluate(parts[p],scores[p],target_events(events,category,p),category,p,designs)
            results.append(m);budgets.append(b);diffs.append(d);leads.append(l)
            if p=='validation':
                chosen = choose_dimension(m)
                selection.append(dict(category=category,selected_variant=chosen,
                    validation_ap=m.loc[m.variant.eq(chosen),'average_precision'].iloc[0],
                    basis='Highest validation AP among predeclared global PCA dimensions; no test selection'))
            w = parts[p][['row_id','pitcher_id','game_id','game_date','season','is_starter','target',
                         'next_event_id','next_event_lead_days','next_event_strict_qa']].copy()
            w['category'],w['partition'] = category,p
            w['seen_training_pitcher'] = w.pitcher_id.isin(train.pitcher_id)
            for name, score in scores[p].items():
                w['score__'+name] = score
            wide.append(w)
            raw,rotation = scores[p]['Original 62'],scores[p]['Full-rank PCA control']
            previous=legacy.loc[legacy.category.eq(category)&legacy.partition.eq(p)].set_index('row_id')
            prior_scores=previous.loc[parts[p].row_id,'score__Full'].to_numpy()
            sanity.append(dict(category=category,partition=p,max_score_difference=np.max(np.abs(raw-rotation)),
                score_correlation=np.corrcoef(raw,rotation)[0,1],
                ap_difference=rank_metrics(parts[p].target,rotation)['average_precision']-rank_metrics(parts[p].target,raw)['average_precision'],
                tight_original_ap=rank_metrics(parts[p].target,raw)['average_precision'],
                legacy_default_tolerance_ap=rank_metrics(parts[p].target,prior_scores)['average_precision'],
                legacy_max_score_difference=np.max(np.abs(raw-prior_scores))))
        for name in settings:
            threshold = np.quantile(scores['validation'][name],.95,method='higher')
            mask = scores['test'][name]>=threshold
            fixed.append(dict(category=category,variant=name,validation_threshold=threshold,
                validation_actual_rate=np.mean(scores['validation'][name]>=threshold),test_alert_rate=mask.mean(),
                test_precision=test.target.to_numpy()[mask].mean() if mask.any() else np.nan,
                test_recall=test.target.to_numpy()[mask].sum()/test.target.sum()))
        joblib.dump({n:fitted[n] for n in ['Original 62',chosen]},CACHE/f"models_{category.replace('/','-').replace(' ','_')}.joblib")
        # Explicitly checkpoint completed categories, keeping previous suites untouched.
        table(pd.concat(results,ignore_index=True),'predictive_performance')
    for rows,name in [(inventory,'predictive_cohort'),(selection,'validation_dimension_selection'),
                      (dimensions,'predictive_dimensions'),(fixed,'validation_fixed_thresholds'),
                      (sanity,'full_rotation_sanity'),(coefs,'effective_coefficients')]:
        table(pd.DataFrame(rows),name)
    for rows,name in [(budgets,'alert_budget_performance'),(diffs,'event_recall_differences'),(leads,'warning_leads')]:
        table(pd.concat(rows,ignore_index=True),name)
    pd.concat(wide,ignore_index=True).to_parquet(CACHE/'predictions.parquet',index=False)
    save_manifest()
    return pd.concat(results,ignore_index=True)


def run_stability(a,spec):
    """Small paired whole-pitcher training bootstrap; REFIT every learned step.

    This diagnoses fitted-score stability, not a confidence interval or new test.
    The dimension specification is frozen at its original validation choice.
    """
    settings=variants(spec)
    selected=read_table('validation_dimension_selection').set_index('category').selected_variant
    predictions=pd.read_parquet(CACHE/'predictions.parquet')
    records,subgroups,coefficient_stability=[],[],[]
    for category in CFG['focus_categories']:
        f=label_frame(a,category,30)
        train=f.loc[f.partition.eq('train')].reset_index(drop=True)
        test=f.loc[f.partition.eq('test')].reset_index(drop=True)
        original=predictions.loc[predictions.category.eq(category)&predictions.partition.eq('test')].set_index('row_id').loc[test.row_id]
        models=joblib.load(CACHE/f"models_{category.replace('/','-').replace(' ','_')}.joblib")
        groups=list(train.groupby('pitcher_id').indices.values())
        rng=np.random.default_rng(CFG['seed'])
        for rep in range(CFG['pitcher_bootstrap_refits']):
            chosen=rng.integers(0,len(groups),len(groups))
            boot=train.iloc[np.concatenate([groups[i] for i in chosen])]
            for name in ['Original 62',selected[category]]:
                features,reduced,k=settings[name]
                enc,model,(score,)=fit_model(boot,[test],features,reduced,k)
                reference=original['score__'+name].to_numpy()
                top=int(len(test)*.05)
                index=np.argsort(-score,kind='stable')[:top];baseline=np.argsort(-reference,kind='stable')[:top]
                overlap=len(set(index)&set(baseline))/len(set(index)|set(baseline))
                records.append(dict(category=category,variant=name,repeat=rep,training_rows=len(boot),
                    design_dimensions=enc.output_dimensions,spearman_score=spearmanr(score,reference).statistic,
                    top_5pct_jaccard=overlap,mean_absolute_score_change=np.mean(np.abs(score-reference)),
                    average_precision=rank_metrics(test.target,score)['average_precision']))
                enc0,model0=models[name]
                b0=pd.Series(enc0.effective_coefficients(model0),index=enc0.names)
                b=pd.Series(enc.effective_coefficients(model),index=enc.names).reindex(b0.index,fill_value=0.)
                coefficient_stability.append(dict(category=category,variant=name,repeat=rep,
                    coefficient_cosine=np.dot(b,b0)/(np.linalg.norm(b)*np.linalg.norm(b0))))
            print(f'Paired training refit {category}: {rep+1}/{CFG["pitcher_bootstrap_refits"]}',flush=True)
        # Frozen scores: compare roles, unseen pitchers, and actionable lead time.
        for condition,mask in [('Starter',original.is_starter),('Reliever',~original.is_starter),
                               ('Unseen training pitcher',~original.seen_training_pitcher),
                               ('Seen training pitcher',original.seen_training_pitcher),
                               ('Exclude final six days',original.target.eq(0)|original.next_event_lead_days.ge(7))]:
            sub=original.loc[mask]
            if sub.target.sum()<1 or sub.target.nunique()<2:
                continue
            ss={name:sub['score__'+name].to_numpy() for name in ['Original 62',selected[category]]}
            bs,_=paired_bootstrap_ap(sub,ss)
            for name,score in ss.items():
                met=rank_metrics(sub.target,score)
                delta=met['average_precision']-rank_metrics(sub.target,ss['Original 62'])['average_precision']
                lo,hi=np.nanquantile(bs[name]-bs['Original 62'],[.025,.975])
                subgroups.append(dict(category=category,condition=condition,variant=name,rows=len(sub),
                    events=sub.loc[sub.target.eq(1),'next_event_id'].nunique(),prevalence=sub.target.mean(),
                    ap_delta=delta,delta_low=lo,delta_high=hi,**met))
        table(pd.DataFrame(records),'training_refit_repeats')
    table(pd.DataFrame(coefficient_stability),'coefficient_stability')
    table(pd.DataFrame(subgroups),'predictive_subgroups')
    summary=pd.DataFrame(records).groupby(['category','variant']).agg(
        repeats=('repeat','count'),ap_mean=('average_precision','mean'),ap_sd=('average_precision','std'),
        score_rank_correlation_mean=('spearman_score','mean'),alert_jaccard_mean=('top_5pct_jaccard','mean'),
        mean_absolute_score_change=('mean_absolute_score_change','mean')).reset_index()
    return table(summary,'training_refit_summary')


def event_concentration_audit():
    """Frozen-score leave-one-positive-pitcher-out diagnostic; no model reselection."""
    f=pd.read_parquet(CACHE/'predictions.parquet')
    selected=read_table('validation_dimension_selection').set_index('category').selected_variant
    rows=[]
    for category,g in f.loc[f.partition.eq('test')].groupby('category'):
        name=selected[category]
        for pid in [None]+sorted(g.loc[g.target.eq(1),'pitcher_id'].unique().tolist()):
            h=g if pid is None else g.loc[g.pitcher_id.ne(pid)]
            raw=rank_metrics(h.target,h['score__Original 62'])['average_precision']
            reduced=rank_metrics(h.target,h['score__'+name])['average_precision']
            rows.append(dict(category=category,variant=name,removed_positive_pitcher=pid,
                             original_ap=raw,reduced_ap=reduced,ap_delta=reduced-raw,
                             remaining_positive_rows=int(h.target.sum())))
    return table(pd.DataFrame(rows),'positive_pitcher_influence')
