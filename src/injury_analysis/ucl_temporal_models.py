"""Fixed L2 logistic models: family-only, family ablation, frozen references."""
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .consolidation_models import ConsolidationEncoder
from .ucl_dynamic_evaluation import metrics, warning
from .ucl_dynamic_data import event_cohort, prepare_data, risk_frame
from .ucl_temporal_common import OUT, CACHE, CFG, table, parquet, checkpoint, valid_checkpoint, interval_name, write_json
from .ucl_temporal_data import prepare, interval_target

def model_path(interval, name, refit=None):
    slug = name.replace(' | ', '__').replace(' ', '_').replace('/', '-')
    return CACHE/'models'/(interval+'__'+slug+('' if refit is None else f'__refit{refit}')+'.joblib')

def fit(train, features, interval, name, refit=None):
    if not train.partition.eq('train').all():
        raise ValueError('Only the chronological training partition may be fitted')
    if train.target.isna().any():
        raise ValueError('Cannot train on unknown outcomes')
    path = model_path(interval, name, refit)
    digest = hashlib.sha256(json.dumps(dict(config=CFG, features=features,
        keys=train.row_id.tolist(), target=train.target.astype(int).tolist()), sort_keys=True).encode()).hexdigest()
    if path.exists():
        obj = joblib.load(path)
        if obj['signature'] != digest: raise ValueError('Stale model: '+str(path))
        return obj
    start = time.monotonic()
    enc = ConsolidationEncoder(features).fit(train)
    model = LogisticRegression(C=CFG['logistic_C'], solver='lbfgs', max_iter=5000,
                               tol=1e-7, random_state=CFG['seed'])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(enc.transform(train), train.target.to_numpy(dtype=int))
    obj = dict(encoder=enc, model=model, signature=digest, interval=interval,
               name=name, refit=refit, rows=len(train), positive_rows=int(train.target.sum()),
               training_max_date=str(train.game_date.max()), seconds=time.monotonic()-start)
    tmp = path.with_suffix('.joblib.partial'); joblib.dump(obj, tmp); tmp.replace(path)
    print(f'Fit {interval} / {name} / refit={refit}: {len(features)} fields, {obj["seconds"]:.1f}s', flush=True)
    return obj

def load(interval, name, refit=None):
    return joblib.load(model_path(interval, name, refit))

def scores(obj, frame, logit=False):
    with threadpool_limits(limits=1):
        x = obj['encoder'].transform(frame)
        return obj['model'].decision_function(x) if logit else obj['model'].predict_proba(x)[:,1]

def permutation_scores(obj, frame, groups):
    """Permute all standardized fields and associated missing flags together.

    Preserve within-family dependence and season; break links to other families
    and outcomes. This is marginal grouped reliance, NOT conditional importance.
    Pitcher sampling uncertainty is assessed separately from shuffle variability.
    """
    with threadpool_limits(limits=1):
        x = obj['encoder'].transform(frame)
        beta = obj['model'].coef_[0]
        logit = obj['model'].decision_function(x)
        strata = list(frame.reset_index(drop=True).groupby('season').indices.values())
        result = {}
        for family, fields in groups.items():
            ix = [j for j,n in enumerate(obj['encoder'].names)
                  if n in fields or n.removeprefix('missingindicator_') in fields]
            contribution = x[:,ix] @ beta[ix]
            for repeat in range(CFG['permutation_repeats']):
                # Same permutations across families/horizons on the identical row set.
                rng = np.random.default_rng(CFG['seed']+repeat)
                order = np.arange(len(frame))
                for index in strata: order[index] = rng.permutation(index)
                from scipy.special import expit
                result[f'Permute {repeat} | {family}'] = expit(logit-contribution+contribution[order])
        return result

def evaluate(frame, score, events, part, lo, hi, name, features, policy='common_60d'):
    m = metrics(frame, score)
    w, details = warning(frame, score, event_cohort(events, part), CFG['alert_rate'], lo, hi)
    keys = ['alerts', 'qualifying_events', 'observable_events', 'alerted_events',
            'injury_event_recall', 'observable_event_recall', 'median_warning_lead_days',
            'alerts_per_detected_event', 'non_actionable_alert_fraction']
    row = dict(partition=part, interval=interval_name(lo,hi), model=name,
               fields=features, risk_policy=policy, **m, **{k:w[k] for k in keys})
    details = details.assign(partition=part, interval=interval_name(lo,hi), model=name, risk_policy=policy)
    return row, details

def run_development():
    if valid_checkpoint('development_models'): return
    f, events, groups, models = prepare()
    rows, event_rows = [], []
    for lo, hi in CFG['intervals']:
        interval = interval_name(lo,hi); g = interval_target(f,lo,hi)
        tr = g.loc[g.partition.eq('train') & g.target.notna()]
        va = g.loc[g.partition.eq('validation') & g.target.notna()]
        pred = va[['row_id','pitcher_id','game_id','game_date','target']].copy()
        for name, fields in models.items():
            obj = fit(tr, fields, interval, name)
            pred[name] = scores(obj,va)
            row, det = evaluate(va,pred[name],events,'validation',lo,hi,name,len(fields))
            rows.append(row); event_rows.append(det)
            mt = metrics(tr,scores(obj,tr))
            rows.append(dict(partition='train', interval=interval, model=name, fields=len(fields),
                             risk_policy='common_60d', **mt))
        for key,s in permutation_scores(load(interval,'Full union'),va,groups).items():
            pred[key] = s
        parquet(pred, 'predictions_validation_'+interval)
    table(pd.DataFrame(rows), 'development_metrics')
    table(pd.concat(event_rows,ignore_index=True), 'development_event_warnings')
    # A single fixed 60-day family score for descriptive trajectories. This does
    # not choose different scores depending on the event's retrospective lead.
    tr = f.loc[f.partition.eq('train')&f.target.notna()]
    scales = []
    for family, fields in groups.items():
        obj = fit(tr,fields,'1-60','Trajectory | '+family)
        z = scores(obj,tr,logit=True)
        scales.append(dict(family=family,mean=float(z.mean()),sd=float(z.std(ddof=0)),
                           training_rows=len(tr),training_max_date=str(tr.game_date.max())))
    table(pd.DataFrame(scales),'trajectory_score_scales')
    checkpoint('development_models', [OUT/'tables/development_metrics.csv',
        OUT/'tables/development_event_warnings.csv',OUT/'tables/trajectory_score_scales.csv']+
        list(CACHE.glob('predictions_validation_*.parquet'))+list((CACHE/'models').glob('*.joblib')))

def run_followup_sensitivity():
    if valid_checkpoint('followup_sensitivity'): return
    a, events, _ = prepare_data()
    _,_,_,models = prepare(); rows=[]
    for lo,hi in CFG['intervals']:
        f = interval_target(risk_frame(a,events,horizon=hi),lo,hi)
        tr = f.loc[f.partition.eq('train')&f.target.notna()]
        va = f.loc[f.partition.eq('validation')&f.target.notna()]
        for name in ['Full union','Profile','Departure']:
            fields=models[name];obj=fit(tr,fields,'sensitivity_'+interval_name(lo,hi),name)
            row,_=evaluate(va,scores(obj,va),events,'validation',lo,hi,name,len(fields),'interval_upper_followup')
            row['training_rows']=len(tr)
            rows.append(row)
    table(pd.DataFrame(rows),'followup_sensitivity')
    checkpoint('followup_sensitivity',[OUT/'tables/followup_sensitivity.csv']+
               list((CACHE/'models').glob('sensitivity_*.joblib')))

def run_transfer():
    if valid_checkpoint('transfer_models'): return
    frozen = OUT/'development_freeze.json'
    if not frozen.exists():
        raise ValueError('Finalize development and freeze before transfer scoring')
    protocol=json.loads(frozen.read_text())
    f,events,groups,models=prepare();rows=[];event_rows=[]
    for lo,hi in CFG['intervals']:
        interval=interval_name(lo,hi);g=interval_target(f,lo,hi)
        te=g.loc[g.partition.eq('test')&g.target.notna()]
        pred=te[['row_id','pitcher_id','game_id','game_date','target']].copy()
        for name,fields in models.items():
            obj=load(interval,name)
            if obj['signature'] != protocol['fit_signatures'][interval+'|'+name]:
                raise ValueError('Model changed after development freeze')
            pred[name]=scores(obj,te)
            row,det=evaluate(te,pred[name],events,'test',lo,hi,name,len(fields))
            rows.append(row);event_rows.append(det)
        for key,s in permutation_scores(load(interval,'Full union'),te,groups).items():
            pred[key]=s
        parquet(pred,'predictions_test_'+interval)
    table(pd.DataFrame(rows),'transfer_metrics')
    table(pd.concat(event_rows,ignore_index=True),'transfer_event_warnings')
    write_json(dict(started_after_development_freeze=protocol['frozen_at_utc'],
                    completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    selection_allowed=False), OUT/'transfer_manifest.json')
    checkpoint('transfer_models',[OUT/'tables/transfer_metrics.csv',
               OUT/'tables/transfer_event_warnings.csv',OUT/'transfer_manifest.json']+
               list(CACHE.glob('predictions_test_*.parquet')))
