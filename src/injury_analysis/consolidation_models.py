"""Train-fitted preprocessing, sparse selection and validation-only comparisons."""
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .consolidation_common import CFG,CACHE,CATEGORY,table,read_table,inputs
from .consolidation_inventory import GROUPS
from .family_decomposition import target_events,paired_bootstrap_ap
from src.evaluation import rank_metrics,evaluate_predictions


class ConsolidationEncoder:
    """Median, tail bounds, missing flags and scale fit strictly on training rows.

    Original nullable database values remain unchanged. Quantile capping is a
    numerical robustness measure for heavy-tailed z-scores/ratios. Binary/rare
    columns whose two bounds coincide are not flattened to a constant.
    """
    def __init__(self,features,indicators=True):
        self.features=list(features);self.indicators=indicators

    def fit(self,frame):
        x=frame[self.features].astype(float).replace([np.inf,-np.inf],np.nan)
        if x.isna().all().any():raise ValueError('All-null training feature')
        self.bounds=x.quantile(CFG['winsor_quantiles'])
        degenerate=self.bounds.iloc[0].ge(self.bounds.iloc[1])
        if degenerate.any():self.bounds.loc[:,degenerate]=np.array([[-np.inf],[np.inf]])
        self.imputer=SimpleImputer(strategy='median',add_indicator=self.indicators)
        self.scaler=StandardScaler()
        self.scaler.fit(self.imputer.fit_transform(self.clip(x)))
        self.names=self.imputer.get_feature_names_out(self.features).tolist()
        self.fit_rows=len(frame)
        self.fit_max_date=str(frame.game_date.max()) if 'game_date' in frame else None
        return self

    def clip(self,x):return x.clip(lower=self.bounds.iloc[0],upper=self.bounds.iloc[1],axis=1)
    def transform(self,frame):
        x=frame[self.features].astype(float).replace([np.inf,-np.inf],np.nan)
        return self.scaler.transform(self.imputer.transform(self.clip(x)))


def fit(train,features,elastic=None,indicators=True):
    if 'partition' in train and not train.partition.eq('train').all():
        raise ValueError('Feature-selection models may only fit training rows')
    start=time.monotonic()
    encoder=ConsolidationEncoder(features,indicators).fit(train)
    options=dict(C=CFG['logistic_C'],solver='lbfgs',max_iter=5000,tol=CFG['logistic_tolerance'])
    if elastic is not None:
        # Rare flags scaled to unit variance can dominate SAGA's step size.
        # Keep binary values/flags on unit raw scale for the sparse screening
        # model; continuous variables remain training-standardized. This is an
        # explicit penalty geometry choice, not a validation-dependent change.
        binary=[c for c in features if set(train[c].dropna().unique())<=set([0,1])]
        for j,name in enumerate(encoder.names):
            if name in binary or name.startswith('missingindicator_'):encoder.scaler.scale_[j]=1.
        options.update(penalty='elasticnet',solver='saga',C=float(elastic['C']),l1_ratio=float(elastic['l1_ratio']),
            max_iter=CFG['elastic_net_max_iter'],tol=CFG['elastic_net_tolerance'])
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(random_state=CFG['seed'],**options)
        model.fit(encoder.transform(train),train.target)
    return dict(encoder=encoder,model=model,seconds=time.monotonic()-start,elastic=elastic,
                train_rows=len(train),train_positive_rows=int(train.target.sum()))


def scores(fitted,frame):
    with threadpool_limits(limits=1):return fitted['model'].predict_proba(fitted['encoder'].transform(frame))[:,1]


def source_coefficients(fitted):
    enc=fitted['encoder'];beta=pd.Series(fitted['model'].coef_[0],index=enc.names)
    rows=[]
    for feature in enc.features:
        b=beta[feature];flag=beta.get('missingindicator_'+feature,0.)
        rows.append(dict(feature=feature,coefficient=b,indicator_coefficient=flag,
            source_weight=np.hypot(b,flag),selected=abs(b)>CFG['coefficient_nonzero_tolerance'],
            selected_including_indicator=max(abs(b),abs(flag))>CFG['coefficient_nonzero_tolerance']))
    return pd.DataFrame(rows)


def summary_metrics(frame,score,events,partition='validation'):
    result=dict(rows=len(frame),positive_rows=int(frame.target.sum()),represented_events=frame.loc[frame.target.eq(1),'next_event_id'].nunique(),
        prevalence=frame.target.mean(),**rank_metrics(frame.target,score))
    result['pr_lift']=result['average_precision']/result['prevalence']
    result['brier_score']=np.mean((score-frame.target.to_numpy())**2)
    result['calibration_bias']=score.mean()-frame.target.mean()
    ev=target_events(events,CATEGORY,partition)
    b,_=evaluate_predictions(frame.assign(score=score),ev,target='target',rates=[.05],warning_window=(7,30))
    for k in ['qualifying_events','observable_events','alerted_events','injury_event_recall','median_warning_lead_days',
              'false_alerts_per_pitcher_season','non_actionable_alerts_per_pitcher_season','precision','recall']:
        result[k]=b[k].iloc[0]
    return result


def feature_sets(inventory=None):
    inv=read_table('feature_inventory') if inventory is None else inventory
    core=inv.loc[inv.eligible_predictor&~inv.support_only_sensitivity,'feature'].tolist()
    support=inv.loc[inv.eligible_predictor&inv.support_only_sensitivity,'feature'].tolist()
    cluster=read_table('correlation_clusters')
    reps=cluster.loc[cluster.representative&cluster.feature.isin(core),'feature'].tolist()
    return core,reps,support


def fit_named(train,valid,features,name,elastic=None,indicators=True):
    path=CACHE/'models'/(name+'.joblib')
    # Paths are unique within this stage. Reuse only exact feature/config/input
    # signatures; this is a resumable cache, not permission to reuse stale fits.
    from .consolidation_common import input_signature
    signature=dict(config=CFG,inputs=input_signature(),features=list(features),elastic=elastic,indicators=indicators,
                   train_keys=train.row_id.tolist(),validation_keys=valid.row_id.tolist())
    if elastic is not None:signature['sparse_binary_scale']='unit raw scale v1'
    import hashlib
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    if path.exists():
        obj=joblib.load(path)
        if obj.get('signature')==digest:return obj,scores(obj,valid)
    obj=fit(train,features,elastic,indicators);obj['signature']=digest
    temporary=path.with_suffix('.joblib.partial')
    joblib.dump(obj,temporary)
    temporary.replace(path)
    print(f'Fit {name}: {len(features)} fields, {len(obj["encoder"].names)} columns, {obj["seconds"]:.1f}s',flush=True)
    return obj,scores(obj,valid)


def run_family_models():
    f,events,spec=inputs();train=f.loc[f.partition.eq('train')];valid=f.loc[f.partition.eq('validation')]
    inv=read_table('feature_inventory');core,reps,support=feature_sets(inv)
    family=inv.set_index('feature').feature_group.to_dict();groups={g:[c for c in core if family[c]==g] for g in GROUPS}
    original=spec.loc[spec.selected_for_prediction,'name'].tolist()
    variants={'Full feasible':core,'Correlation representatives':reps,'Previous 62 (same preprocessing)':original,
              'Full plus support':core+support,'Support only':support}
    for j,(group,features) in enumerate(groups.items()):
        if features:
            variants[f'Without family {j:02d}']=[c for c in core if c not in features]
            variants[f'Only family {j:02d}']=features
    rows=[];all_scores=valid[['row_id','pitcher_id','game_id','game_date','target']].copy();specs=[]
    for j,(name,features) in enumerate(variants.items()):
        fitname='family_'+name.replace(' ','_').replace('/','_')
        obj,score=fit_named(train,valid,features,fitname)
        metadata=dict(variant=name,features=len(features),design_columns=len(obj['encoder'].names),
            family=GROUPS[int(name[-2:])] if name.startswith(('Without family','Only family')) else '',partition='validation')
        rows.append(dict(**metadata,**summary_metrics(valid,score,events)))
        all_scores['score__'+name]=score
        specs.append(dict(variant=name,features=features,path=str((CACHE/'models'/(fitname+'.joblib')).relative_to(CACHE))))
        table(pd.DataFrame(rows),'family_performance')
    all_scores.to_parquet(CACHE/'family_validation_predictions.parquet',index=False)
    (CACHE/'family_model_specifications.json').write_text(json.dumps(specs,indent=2))
    # Paired uncertainty for ablations on identical validation pitchers/rows.
    scoremap={r['variant']:all_scores['score__'+r['variant']].to_numpy() for r in specs}
    bs,_=paired_bootstrap_ap(valid,scoremap);base=bs['Full feasible'];perf=pd.DataFrame(rows).set_index('variant')
    diff=[]
    for name,b in bs.items():
        delta=base-b
        diff.append(dict(variant=name,family=perf.loc[name,'family'],ap_loss=perf.loc['Full feasible','average_precision']-perf.loc[name,'average_precision'],
            low=np.nanquantile(delta,.025),high=np.nanquantile(delta,.975)))
    table(pd.DataFrame(diff),'family_paired_differences')
    return pd.DataFrame(rows)


def grouped_permutation(obj,valid,groups,repetitions=None):
    """Jointly permute a family's original values AND associated missing flags.

    Permutations stay within season/role where role is not the tested family.
    Row permutation breaks serial dependence; repeat SD is a perturbation
    diagnostic, not a sampling confidence interval or conditional causal effect.
    """
    reps=CFG['permutation_repeats'] if repetitions is None else repetitions
    enc=obj['encoder'];z=enc.transform(valid);beta=obj['model'].coef_[0];intercept=obj['model'].intercept_[0]
    from scipy.special import expit
    logits=z@beta+intercept;base=rank_metrics(valid.target,expit(logits))['average_precision'];rows=[]
    for group,features in groups.items():
        idx=[i for i,n in enumerate(enc.names) if n in features or n.removeprefix('missingindicator_') in features]
        if not idx:continue
        columns=['season'] if 'is_starter' in features else ['season','is_starter']
        strata=list(valid.reset_index(drop=True).groupby(columns).indices.values())
        contribution=z[:,idx]@beta[idx];rng=np.random.default_rng(CFG['seed'])
        for rep in range(reps):
            order=np.arange(len(valid))
            for ix in strata:order[ix]=rng.permutation(ix)
            ap=rank_metrics(valid.target,expit(logits-contribution+contribution[order]))['average_precision']
            rows.append(dict(group=group,repeat=rep,features=len(features),baseline_ap=base,permuted_ap=ap,ap_drop=base-ap,
                strata=';'.join(columns),interpretation='Marginal grouped permutation conditional only on season/role strata'))
    return pd.DataFrame(rows)


def run_family_permutation():
    f,_,_=inputs();valid=f.loc[f.partition.eq('validation')]
    inv=read_table('feature_inventory');core,_,_=feature_sets(inv)
    groups={g:x.loc[x.feature.isin(core),'feature'].tolist() for g,x in inv.groupby('feature_group')}
    obj=joblib.load(CACHE/'models/family_Full_feasible.joblib')
    with threadpool_limits(limits=1):raw=grouped_permutation(obj,valid,groups)
    table(raw,'grouped_permutation_repeats')
    return table(raw.groupby('group').ap_drop.agg(['mean','std','min','max']).reset_index(),'grouped_permutation_summary')


def run_elastic_tuning():
    f,events,_=inputs();train=f.loc[f.partition.eq('train')];valid=f.loc[f.partition.eq('validation')]
    # Sparse selection sees ALL feasible substantive fields, so correlation
    # alternatives can exchange weight across refits instead of being predeleted.
    reps,_,_=feature_sets();rows=[]
    for c in CFG['elastic_net_C']:
        for mix in CFG['elastic_net_l1_ratio']:
            name=f'elastic_C{c}_mix{mix}'
            obj,score=fit_named(train,valid,reps,name,elastic={'C':c,'l1_ratio':mix})
            coef=source_coefficients(obj)
            row=dict(model=name,C=c,l1_ratio=mix,input_features=len(reps),nonzero_features=int(coef.selected.sum()),
                     nonzero_with_flags=int(coef.selected_including_indicator.sum()),**summary_metrics(valid,score,events))
            rows.append(row);table(pd.DataFrame(rows),'elastic_net_tuning')
    selected=sorted(rows,key=lambda r:(-r['average_precision'],r['nonzero_features'],r['C'],-r['l1_ratio']))[0]
    (CACHE/'elastic_selection.json').write_text(json.dumps(selected,indent=2))
    obj=joblib.load(CACHE/'models'/(selected['model']+'.joblib'))
    table(source_coefficients(obj),'elastic_net_selected_coefficients')
    print(f'Validation-selected Elastic Net: {selected["model"]}; {selected["nonzero_features"]} nonzero measured features.',flush=True)
    return pd.DataFrame(rows)
