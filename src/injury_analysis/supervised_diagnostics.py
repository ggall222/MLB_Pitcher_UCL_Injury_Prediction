"""Interpretation, paired training refits and focused timing checks for PLS."""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .supervised_reduction import (CFG,CACHE,OUT,ROOT,PROFILE,DEPARTURE,table,read_table,
    feature_blocks,SupervisedBasis,fit_classifier,fit_block,evaluate_scores,rank_metrics,
    probability_metrics,label_frame,target_events,clean)
from .common import CACHE as OLD_CACHE
from .reduction import between_pitcher_fraction,subspace_overlap
from .trajectory_shapes import paired_cells,simultaneous_summary
from .signals_common import mean_ci
from src.evaluation import alert_mask


def model_path(category,block):
    return CACHE/('model_'+category.replace('/','-').replace(' ','_')+'__'+block.replace(' ','_')+'.joblib')


def feature_family(feature,spec,block):
    if block=='Combined':return 'Prior profile' if feature in PROFILE else 'Current departure'
    if block=='Profile':return 'Prior measurement mean' if feature.endswith('_baseline_mean') else 'History/context'
    existing=spec.set_index('name').family.to_dict()
    existing['zone_pct_z']='command'
    return existing.get(feature,'Other')


def run_interpretation(a,events,spec,matches):
    loads=read_table('basis_loadings')
    loads['family']=[feature_family(r.feature,spec,r.block) for r in loads.itertuples()]
    table(loads,'component_loadings')
    family=loads.groupby(['category','block','method','component','family']).squared_axis_weight.sum().reset_index()
    table(family,'component_family_weights')
    old_pca=joblib.load(ROOT/'data/injury/analysis/dimensionality_reduction/descriptive_pca.joblib')
    pc=old_pca['Within-pitcher departure']
    records=[];persistence=[];alignments=[]
    old=pd.read_parquet(OLD_CACHE/'aligned_trajectories.parquet')
    keys=['event_id','analysis_group','cluster_pitcher_id','pitcher_id','game_id','game_date','lead_days','appearance_before','kind']
    aligned=old[keys].merge(a[['pitcher_id','game_id']+PROFILE+DEPARTURE],on=['pitcher_id','game_id'],validate='many_to_one')
    aligned=aligned.merge(matches[['event_id','anchor_date','control_anchor_date']],on='event_id',validate='many_to_one')
    ok=aligned.assign(in_period=aligned.game_date.between('2022-01-01','2025-12-31')).groupby('event_id').in_period.all()
    ok &= matches.set_index('event_id').anchor_date.between('2022-01-01','2025-12-31').reindex(ok.index)
    aligned=aligned.loc[aligned.event_id.map(ok)].copy()
    aligned['complete']=clean(aligned,PROFILE+DEPARTURE).notna().all(axis=1)
    for category in a_next_categories():
        risk=label_frame(a,category,30);train=risk.loc[risk.partition.eq('train')]
        g=aligned.loc[aligned.analysis_group.eq(category)].copy()
        for block in ['Original 62','Profile','Departure','Combined']:
            fitted=joblib.load(model_path(category,block));basis=fitted['basis']
            for method in ['PCA','PLS']:
                for j in range(3):
                    # Sign fixed using TRAINING covariance with y; not case curves.
                    train_score=basis.transform(train,method,j+1)[:,j]
                    sign=1. if np.cov(train_score,train.target)[0,1]>=0 else -1.
                    sd=basis.training_score_sd[method][j]
                    column=f'{block}__{method}{j+1}'
                    if block!='Original 62':
                        g[column]=sign*basis.transform(g,method,j+1)[:,j]/sd
                        g.loc[~g.complete,column]=np.nan
                    for part in ['train','validation','test']:
                        f=risk.loc[risk.partition.eq(part)]
                        scores=basis.transform(f,method,j+1)[:,j]
                        persistence.append(dict(category=category,block=block,method=method,component=j+1,partition=part,
                            between_pitcher_fraction=between_pitcher_fraction(scores,f.pitcher_id),
                            training_orientation=sign,training_score_sd=sd,rows=len(f)))
            if block=='Departure':
                z=pc.standardized(train)[:,pc.rotate_index];old_scores=pc.pca.transform(z)
                for j in range(3):
                    score=basis.transform(train,'PLS',j+1)[:,j]
                    for k in range(3):
                        records.append(dict(category=category,pls_component=j+1,prior_pca_component=k+1,
                            training_score_correlation=np.corrcoef(score,old_scores[:,k])[0,1],
                            cohort='Same category training risk rows; both axes fitted on training only'))
        alignments.append(g)
    f=pd.concat(alignments,ignore_index=True)
    table(pd.DataFrame(records),'prior_pca_score_correlations')
    table(pd.DataFrame(persistence),'component_persistence')
    orientations=pd.DataFrame(persistence).query("partition=='train'")[['category','block','method','component','training_orientation']]
    loads=loads.drop(columns=['training_orientation','injury_oriented_axis_weight','injury_oriented_feature_score_correlation'],errors='ignore')
    loads=loads.merge(orientations,on=['category','block','method','component'],validate='many_to_one')
    loads['injury_oriented_axis_weight']=loads.axis_weight*loads.training_orientation
    loads['injury_oriented_feature_score_correlation']=loads.feature_score_correlation*loads.training_orientation
    table(loads,'component_loadings')
    f.to_parquet(CACHE/'matched_component_alignment.parquet',index=False)
    summaries=[];shapes=[];members=[]
    for block in ['Profile','Departure','Combined']:
        feature=block+'__PLS1'
        for mode in ['appearance','calendar']:
            points=list(range(5 if mode=='appearance' else 4))
            cells=paired_cells(f,feature,mode)
            for category,g in cells.groupby('analysis_group'):
                summary,ids=simultaneous_summary(g,points)
                if summary.empty:continue
                summaries.append(summary.assign(block=block,feature=feature,mode=mode,category=category))
                members.extend(dict(block=block,mode=mode,category=category,event_id=i) for i in ids)
                stats=[]
                for event,h in g.loc[g.event_id.isin(ids)].groupby('event_id'):
                    d=h.sort_values('point').difference.to_numpy()
                    for metric,value in [('average_separation',d.mean()),('last_minus_first',d[-1]-d[0])]:
                        stats.append(dict(event_id=event,pitcher_id=h.cluster_pitcher_id.iloc[0],metric=metric,value=value))
                for metric,h in pd.DataFrame(stats).groupby('metric'):
                    shapes.append(dict(category=category,block=block,mode=mode,metric=metric,**mean_ci(h,'value')))
    table(pd.concat(summaries,ignore_index=True),'component_trajectories')
    table(pd.DataFrame(shapes),'component_shape_contrasts')
    pd.DataFrame(members).to_parquet(CACHE/'balanced_component_membership.parquet',index=False)
    return family


def a_next_categories():
    from .common import CATEGORIES
    return CATEGORIES


def run_stability(a,spec):
    blocks=feature_blocks(spec);records=[];families=[]
    for category in CFG['stability_categories']:
        frame=label_frame(a,category,30)
        train=frame.loc[frame.partition.eq('train')].reset_index(drop=True)
        targets={p:frame.loc[frame.partition.eq(p)] for p in ['validation','test']}
        groups=train.groupby('pitcher_id').indices;people=np.array(sorted(groups))
        for block in CFG['stability_blocks']:
            reference=joblib.load(model_path(category,block));basis0=reference['basis']
            rng=np.random.default_rng(CFG['seed'])
            for repeat in range(CFG['training_refits']):
                draw=rng.choice(people,len(people),replace=True)
                boot=train.iloc[np.concatenate([groups[pid] for pid in draw])]
                basis=SupervisedBasis(blocks[block],CFG['component_grids'][block]+list(reference['selected'].values())).fit(boot)
                ratio=basis0.encoder.scaler.scale_[basis0.index]/basis.encoder.scaler.scale_[basis.index]
                for method in ['PCA','PLS']:
                    k=reference['selected'][method]
                    converted=ratio[:,None]*basis.bases[method][:,:k]
                    q=np.linalg.qr(converted,mode='reduced')[0]
                    overlap=subspace_overlap(basis0.bases[method][:,:k].T,q.T,k)
                    first_cos=abs(np.dot(converted[:,0]/np.linalg.norm(converted[:,0]),basis0.bases[method][:,0]))
                    _,scores=fit_classifier(basis.transform(boot,method,k),boot.target,
                                           [basis.transform(g,method,k) for g in targets.values()])
                    for (part,g),score in zip(targets.items(),scores):
                        ref=reference['models'][method+' selected'].predict_proba(basis0.transform(g,method,k))[:,1]
                        original_alert=alert_mask(g.assign(score=ref),'score',.05)
                        new_alert=alert_mask(g.assign(score=score),'score',.05)
                        records.append(dict(category=category,block=block,method=method,partition=part,repeat=repeat,
                            selected_components=k,train_positive_rows=int(boot.target.sum()),
                            distinct_train_positive_events=boot.loc[boot.target.eq(1),'next_event_id'].nunique(),
                            first_axis_abs_cosine=first_cos,selected_subspace_overlap=overlap,
                            rank_correlation=spearmanr(ref,score).statistic,
                            alert_jaccard=(original_alert&new_alert).sum()/(original_alert|new_alert).sum(),
                            **rank_metrics(g.target,score)))
                    weights=(converted[:,0]/np.linalg.norm(converted[:,0]))**2
                    for name in set(feature_family(c,spec,block) for c in basis.continuous):
                        energy=sum(w for w,c in zip(weights,basis.continuous) if feature_family(c,spec,block)==name)
                        families.append(dict(category=category,block=block,method=method,repeat=repeat,family=name,first_axis_squared_weight=energy))
                print(f'PLS/PCA stability {category} / {block}: {repeat+1}/{CFG["training_refits"]}',flush=True)
    raw=table(pd.DataFrame(records),'training_refits')
    table(pd.DataFrame(families),'refit_family_weights')
    summary=raw.groupby(['category','block','method','partition']).agg(
        refits=('repeat','size'),mean_ap=('average_precision','mean'),sd_ap=('average_precision','std'),
        median_rank_correlation=('rank_correlation','median'),median_alert_jaccard=('alert_jaccard','median'),
        median_first_axis_cosine=('first_axis_abs_cosine','median'),min_first_axis_cosine=('first_axis_abs_cosine','min'),
        median_subspace_overlap=('selected_subspace_overlap','median')).reset_index()
    return table(summary,'stability_summary')


def run_sensitivity(a,events,spec):
    """Shoulder timing/availability concerns specified by the preceding analysis."""
    blocks=feature_blocks(spec);results=[];differences=[];choices=[];curves=[]
    category='Shoulder';base=label_frame(a,category,30)
    for condition in ['Complete P+D measurements','Exclude final six days']:
        f=base.loc[clean(base,PROFILE+DEPARTURE).notna().all(axis=1)] if condition.startswith('Complete') else base.loc[
            ~(base.target.eq(1)&base.next_event_lead_days.lt(7))]
        parts={p:f.loc[f.partition.eq(p)] for p in ['train','validation','test']}
        for block in ['Original 62','Departure','Combined']:
            basis,curve,selection,scores,fitted,check=fit_block(*parts.values(),blocks[block],block,category,condition)
            choices.append(selection);curves.append(curve)
            for p in ['validation','test']:
                m,b,l,d=evaluate_scores(parts[p],scores[p],target_events(events,category,p),category,p,analysis=condition,
                    contrasts=[('PLS selected','Unreduced'),('PLS selected','PCA selected')])
                results.append(m.assign(block=block));differences.append(d.assign(block=block))
            print(f'Supervised sensitivity: {condition} / {block}',flush=True)
    table(pd.concat(results,ignore_index=True),'sensitivity_performance')
    table(pd.concat(differences,ignore_index=True),'sensitivity_differences')
    table(pd.concat(choices,ignore_index=True),'sensitivity_selection')
    table(pd.concat(curves,ignore_index=True),'sensitivity_component_curves')
    return pd.concat(differences,ignore_index=True)
