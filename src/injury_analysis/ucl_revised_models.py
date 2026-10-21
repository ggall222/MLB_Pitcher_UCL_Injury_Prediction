"""Train/validation development, immutable freeze, then a single 2026 scoring layer."""
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .consolidation_models import ConsolidationEncoder
from .ucl_dynamic_models import DynamicEncoder
from .ucl_nonlinear_models import AdditiveSplineEncoder
from .ucl_dynamic_evaluation import metrics,warning
from .ucl_revised_common import ROOT,OUT,CACHE,CFG,table,parquet,write_json,checkpoint,valid_checkpoint,freeze,signature
from .ucl_revised_data import prepare,frame,event_set

def model_id(rep,algorithm,setting):
    return rep.replace(' + ','_').replace(' ','_')+'__'+algorithm+'__'+str(setting).replace('.','p')

def candidates(specs):
    rows=[]
    for rep in CFG['representations']+['Profile + departure + trajectory + interactions']:
        spec=specs[rep]
        for j,c in enumerate(CFG['logistic_C']):
            rows.append(dict(model_id=model_id(rep,'Linear',c),representation=rep,algorithm='Linear',
                setting=c,complexity=j,algorithm_complexity=0,**spec))
        if spec['interactions']:continue
        for j,c in enumerate(CFG['spline_C']):
            rows.append(dict(model_id=model_id(rep,'Spline',c),representation=rep,algorithm='Spline',
                setting=c,complexity=j,algorithm_complexity=1,**spec))
        for j,(leaves,depth) in enumerate(zip(CFG['tree_leaf_limits'],CFG['tree_depth_limits'])):
            rows.append(dict(model_id=model_id(rep,'Boosting',leaves),representation=rep,algorithm='Boosting',
                setting=leaves,depth=depth,complexity=j,algorithm_complexity=2,**spec))
    return {r['model_id']:r for r in rows}

def path(h,mid,legacy=False):return CACHE/'models'/(f'h{h:02d}__'+mid+('__train2015_2021' if legacy else '')+'.joblib')

def fit(train,spec,horizon=30,legacy=False):
    if not train.partition.eq('train').all() or train.game_date.gt(CFG['train_end']).any():
        raise ValueError('Fit requires expanded training dates only; validation/test forbidden')
    if train.target.isna().any():raise ValueError('Unknown targets cannot be fitted')
    p=path(horizon,spec['model_id'],legacy)
    key=hashlib.sha256(json.dumps(dict(source=signature(),spec=spec,horizon=horizon,legacy=legacy,
        keys=train.row_id.tolist(),target=train.target.astype(int).tolist()),sort_keys=True).encode()).hexdigest()
    if p.exists():
        obj=joblib.load(p)
        if obj['signature']!=key:raise ValueError('Changed model training inputs: '+str(p))
        return obj
    if (OUT/'test_access.json').exists():
        raise ValueError('Test layer has been opened; new fits require a separately versioned study')
    if spec['algorithm']=='Spline':encoder=AdditiveSplineEncoder(spec['features'])
    elif spec['interactions']:encoder=DynamicEncoder(spec)
    else:encoder=ConsolidationEncoder(spec['features'])
    start=time.monotonic();encoder.fit(train)
    if spec['algorithm']=='Boosting':
        model=HistGradientBoostingClassifier(max_leaf_nodes=int(spec['setting']),max_depth=spec['depth'],
            max_iter=CFG['tree_iterations'],learning_rate=CFG['tree_learning_rate'],min_samples_leaf=CFG['tree_min_leaf'],
            l2_regularization=CFG['tree_l2'],early_stopping=False,random_state=CFG['seed'])
    else:model=LogisticRegression(C=spec['setting'],max_iter=5000,tol=1e-7,solver='lbfgs',random_state=CFG['seed'])
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(encoder.transform(train),train.target.to_numpy(dtype=int))
    obj=dict(encoder=encoder,model=model,spec=spec,signature=key,horizon=horizon,
        training_max_date=str(train.game_date.max()),training_min_date=str(train.game_date.min()),
        training_rows=len(train),positive_rows=int(train.target.sum()),
        represented_events=train.loc[train.target.eq(1),'dynamic_next_event_id'].nunique(),seconds=time.monotonic()-start)
    tmp=p.with_suffix('.joblib.partial');joblib.dump(obj,tmp);tmp.replace(p)
    print(f'Fit H{horizon} {spec["model_id"]} old-window={legacy}: {len(train)} rows / {obj["represented_events"]} events, {obj["seconds"]:.1f}s',flush=True)
    return obj

def load(h,mid,legacy=False):return joblib.load(path(h,mid,legacy))
def scores(obj,f):
    with threadpool_limits(limits=1):return obj['model'].predict_proba(obj['encoder'].transform(f))[:,1]

def select(validation):
    if not validation.partition.eq('validation').all():raise ValueError('Validation only: 2026 cannot select models')
    q=validation.loc[validation.average_precision>validation.prevalence]
    if q.empty:raise ValueError('No candidate meets predeclared AP floor; do not select from test')
    return q.sort_values(['injury_event_recall','raw_features','algorithm_complexity','complexity','average_precision','model_id'],
        ascending=[False,True,True,True,False,True]).iloc[0].model_id

def metric_row(g,s,events,spec,horizon,part):
    w,_=warning(g,s,event_set(events,part),.05,7,horizon)
    return dict(partition=part,horizon=horizon,model_id=spec['model_id'],representation=spec['representation'],
        algorithm=spec['algorithm'],setting=spec['setting'],complexity=spec['complexity'],algorithm_complexity=spec['algorithm_complexity'],
        raw_features=len(spec['features'])+len(spec['interactions']),**metrics(g,s),
        **{k:w[k] for k in ['injury_event_recall','alerted_events','qualifying_events','observable_events','median_warning_lead_days','non_actionable_alert_fraction']})

def run_development():
    if valid_checkpoint('development'):return
    a,e,specs=prepare();catalog=candidates(specs)
    write_json(dict(config=CFG,candidates=catalog,test_use='2026 excluded from every development/selection input',
        prior_exposure='2026 descriptive injury counts and source quality were inspected in earlier stages; never previously scored by these candidate models',
        selection_scope='Select among established feature representations and model settings; no new individual-feature search',
        warning_policy='Primary 30-day outcome, event warnings 7–30d, 5% observed-validation appearance budget',
        comparison='Selected specifications also refit on 2015–2021 for an equal-validation-cohort training-window diagnostic',
        outcome_cutoff=CFG['observation_end']),OUT/'predeclared_analysis.json')
    f=frame(a,e);tr=f.loc[f.partition.eq('train')&f.target.notna()];va=f.loc[f.partition.eq('validation')&f.target.notna()]
    assert va.game_date.between('2024-01-01','2025-12-31').all()
    rows=[];pred=va[['row_id','pitcher_id','game_id','game_date','target']].copy()
    for mid,spec in catalog.items():
        obj=fit(tr,spec);pred[mid]=scores(obj,va)
        rows.append(metric_row(va,pred[mid],e,spec,30,'validation'))
        rows.append(dict(partition='train',horizon=30,model_id=mid,representation=spec['representation'],
            algorithm=spec['algorithm'],setting=spec['setting'],**metrics(tr,scores(obj,tr))))
    metric=table(pd.DataFrame(rows),'development_search')
    parquet(pred,'validation_predictions_h30')
    v=metric.loc[metric.partition.eq('validation')]
    primary=select(v);warning_leaders={alg:select(g) for alg,g in v.groupby('algorithm')}
    ranking={alg:g.sort_values(['average_precision','raw_features','complexity','model_id'],ascending=[False,True,True,True]).iloc[0].model_id for alg,g in v.groupby('algorithm')}
    ranking_primary=v.sort_values(['average_precision','raw_features','complexity','model_id'],ascending=[False,True,True,True]).iloc[0].model_id
    references=[model_id(rep,'Linear',1.) for rep in CFG['representations']]
    kept=list(dict.fromkeys([primary,ranking_primary]+list(warning_leaders.values())+list(ranking.values())+references))
    # Selection ends here. Auxiliary horizon models use the 30-day selected
    # specification, plus the fixed Profile/Departure references, without tuning.
    auxiliary=list(dict.fromkeys([primary,model_id('Departure','Linear',1.),model_id('Profile','Linear',1.)]))
    all_pairs=[(30,mid) for mid in kept]+[(h,mid) for h in CFG['secondary_horizons'] for mid in auxiliary]
    thresholds={};signatures={};model_files={};calibration={};extra=[];window_rows=[]
    for h in [30]+CFG['secondary_horizons']:
        g=frame(a,e,h);trh=g.loc[g.partition.eq('train')&g.target.notna()];vah=g.loc[g.partition.eq('validation')&g.target.notna()]
        names=kept if h==30 else auxiliary
        ph=vah[['row_id','pitcher_id','game_id','game_date','target']].copy()
        for mid in names:
            spec=catalog[mid];obj=load(30,mid) if h==30 else fit(trh,spec,h)
            score=pred[mid].to_numpy() if h==30 else scores(obj,vah);ph[mid]=score
            key=f'{h}|{mid}'
            thresholds[key]={str(r):float(np.quantile(score,1-r)) for r in CFG['alert_rates']}
            calibration[key]=np.unique(np.quantile(score,np.arange(.1,1,.1))).tolist()
            signatures[key]=obj['signature']
            from .common import sha
            model_files[str(path(h,mid).relative_to(ROOT))]=sha(path(h,mid))
            if h!=30:extra.append(metric_row(vah,score,e,spec,h,'validation'))
            if h==30:
                # Preserve the old boundary embargo at the end of 2021 for this
                # diagnostic. No new hyperparameters are chosen in the shorter fit.
                old=trh.loc[trh.game_date.le(pd.Timestamp('2021-12-31')-pd.Timedelta(days=CFG['embargo_days']))].copy()
                objold=fit(old,spec,h,legacy=True);oldscore=scores(objold,vah)
                ph['Old window | '+mid]=oldscore
                r=metric_row(vah,oldscore,e,spec,h,'validation')
                r.update(training_window='2015-2021',training_rows=len(old));window_rows.append(r)
                r=metric_row(vah,score,e,spec,h,'validation')
                r.update(training_window='2015-2023',training_rows=len(trh));window_rows.append(r)
        parquet(ph,f'validation_selected_h{h}')
    table(pd.DataFrame(extra),'auxiliary_validation_metrics')
    table(pd.DataFrame(window_rows),'training_window_comparison')
    payload=dict(train_period='2015–2023',validation_period='2024–2025',test_period='2026',
        selection_rule=CFG['selection_rule'],primary_horizon=30,selected_primary_model=primary,selected_ap_model=ranking_primary,
        warning_leaders=warning_leaders,ranking_leaders=ranking,reference_models=references,
        evaluation_pairs=all_pairs,thresholds=thresholds,calibration_edges=calibration,
        catalog=catalog,fit_signatures=signatures,model_sha256=model_files,source=signature(),
        alert_policy='Frozen numeric validation quantiles applied unchanged; retrospective budget curves reported separately',
        followup_policy='First-event target; ambiguous episodes censored; event-free negatives require observed MLB follow-up within 30-day grace and source cutoff',
        endpoint_policy='Unchanged refined source-documented UCL; earliest defensible canonical proxy; surgery not onset',
        prior_2026_exposure='Descriptive source/event counts only in prior stages; not fully unseen clinical ascertainment',
        test_adaptation_allowed=False)
    freeze(payload)
    checkpoint('development',[OUT/'frozen_protocol.json',OUT/'predeclared_analysis.json',OUT/'tables/development_search.csv',
        OUT/'tables/auxiliary_validation_metrics.csv',OUT/'tables/training_window_comparison.csv']+
        list(CACHE.glob('validation_*.parquet'))+list((CACHE/'models').glob('*.joblib')))
    print('FROZEN primary:',primary,'; AP:',ranking_primary,flush=True)
