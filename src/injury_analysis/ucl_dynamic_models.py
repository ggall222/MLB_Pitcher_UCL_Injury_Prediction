"""Frozen-feature landmark logistic comparisons and pre-transfer model freeze."""
import hashlib,json,time,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .consolidation_models import ConsolidationEncoder
from .ucl_dynamic_common import ROOT,OUT,CACHE,CFG,table,parquet,write_json,checkpoint,valid_checkpoint
from .ucl_dynamic_data import prepare_data,risk_frame,event_cohort
from .ucl_dynamic_evaluation import metrics,warning,selection,event_bootstrap

class DynamicEncoder:
    """Training-only preprocessing; interpretable standardized main-effect products."""
    def __init__(self,spec):self.spec=spec
    def fit(self,f):
        self.base=ConsolidationEncoder(self.spec['features']).fit(f)
        z=self.base.transform(f);self.names=self.base.names.copy()
        self.indices=[(self.names.index(l),self.names.index(r)) for l,r in self.spec['interactions']]
        p=self.products(z)
        self.product_mean=p.mean(axis=0);self.product_sd=p.std(axis=0)
        self.product_sd[self.product_sd==0]=1
        self.names += [l+' × '+r for l,r in self.spec['interactions']]
        return self
    def products(self,z):
        return np.column_stack([z[:,i]*z[:,j] for i,j in self.indices]) if self.indices else np.empty((len(z),0))
    def transform(self,f):
        z=self.base.transform(f)
        return np.column_stack([z,(self.products(z)-self.product_mean)/self.product_sd])

def fit_model(train,spec,name):
    if not train.partition.eq('train').all():raise ValueError('Fit must contain training rows only')
    key=hashlib.sha256(json.dumps(dict(spec=spec,config=CFG,keys=train.row_id.tolist(),
        target=train.target.astype(int).tolist()),sort_keys=True).encode()).hexdigest()
    path=CACHE/'models'/(name+'.joblib')
    if path.exists():
        obj=joblib.load(path)
        if obj['signature']!=key:raise ValueError('Changed fit input for '+name)
        return obj
    enc=DynamicEncoder(spec).fit(train);start=time.monotonic()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=CFG['logistic_C'],solver='lbfgs',max_iter=5000,tol=1e-7,random_state=CFG['seed'])
        model.fit(enc.transform(train),train.target.to_numpy(dtype=int))
    obj=dict(encoder=enc,model=model,signature=key,training_max_date=str(train.game_date.max()),
        rows=len(train),events=train.loc[train.target.eq(1),'dynamic_next_event_id'].nunique(),seconds=time.monotonic()-start)
    temp=path.with_suffix('.joblib.partial');joblib.dump(obj,temp);temp.replace(path)
    print('Fit',name,'rows',len(train),'events',obj['events'],flush=True)
    return obj

def scores(obj,f):
    with threadpool_limits(limits=1):return obj['model'].predict_proba(obj['encoder'].transform(f))[:,1]

def model_path(horizon,name):return f'h{horizon:02d}_'+name.replace(' + ','_').replace(' ','_')

def run_models():
    if valid_checkpoint('models'):return
    a,events,specs=prepare_data();frames={h:risk_frame(a,events,h) for h in CFG['horizons']}
    counts=[]
    for h,f in frames.items():
        for (part,status),g in f.groupby(['partition','label_status']):
            counts.append(dict(horizon=h,partition=part,status=status,rows=len(g),pitchers=g.pitcher_id.nunique(),
                positive_rows=int(g.target.fillna(0).sum()),represented_events=g.loc[g.target.eq(1),'dynamic_next_event_id'].nunique()))
    table(pd.DataFrame(counts),'landmark_cohort_counts')
    write_json(dict(rule='Validation 30-day event recall for 7–30-day warnings at 5% alert budget; AP above prevalence; ties prefer fewer raw features then AP',
        candidate_hypotheses=specs,hyperparameters='Fixed L2 logistic C=1, no new feature selection',
        endpoint='Retrospectively source-documented UCL; earliest canonical event date; known non-UCL first events as competing negatives'),OUT/'predeclared_analysis.json')
    rows=[];warn=[];pred=[];coef=[];objects={};thresholds={};skips=[]
    for h,f in frames.items():
        tr=f.loc[f.partition.eq('train')&f.target.notna()];va=f.loc[f.partition.eq('validation')&f.target.notna()]
        if tr.loc[tr.target.eq(1),'dynamic_next_event_id'].nunique()<CFG['minimum_training_events'] or va.loc[va.target.eq(1),'dynamic_next_event_id'].nunique()<CFG['minimum_validation_events']:
            skips.append(dict(horizon=h,reason='Insufficient distinct training/validation events'));continue
        ev=event_cohort(events,'validation')
        for name,spec in specs.items():
            obj=fit_model(tr,spec,model_path(h,name));objects[h,name]=obj;score=scores(obj,va)
            key=f'{h}|{name}';thresholds[key]=float(np.quantile(score,.95))
            rows.append(dict(partition='validation',horizon=h,model=name,raw_features=len(spec['features'])+len(spec['interactions']),
                encoded_features=len(obj['encoder'].names),**metrics(va,score)))
            for lead in [x for x in CFG['minimum_leads'] if x<=h]:
                for rate in CFG['alert_rates']:
                    m,_=warning(va,score,ev,rate,lead,h)
                    warn.append(dict(partition='validation',horizon=h,model=name,**m))
            p=va[['row_id','pitcher_id','game_id','game_date','target']].copy();p['score']=score;p['model']=name;p['horizon']=h;p['partition']='validation';pred.append(p)
            coef.extend(dict(model=name,horizon=h,feature=n,coefficient=float(b)) for n,b in zip(obj['encoder'].names,obj['model'].coef_[0]))
    mt=pd.DataFrame(rows);wt=pd.DataFrame(warn)
    selected=selection(mt,wt)
    freeze=dict(selected_model=selected,primary_horizon=30,comparator_models=['Departure','Profile','Previous 62'],
        threshold_5_percent_validation=thresholds['30|'+selected],thresholds=thresholds,
        selection_period='2022–2023 validation only',future_evaluation='Prospectively acquired post-freeze complete cohort; 2024–2025 exploratory, 2026 not confirmatory',
        selected_spec=specs[selected],source_checkpoint=json.loads((CACHE/'endpoint_checkpoint.json').read_text()),
        fit_signatures={f'{h}|{n}':o['signature'] for (h,n),o in objects.items()},
        frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat())
    freeze_path=OUT/'frozen_protocol.json'
    if freeze_path.exists():
        old=json.loads(freeze_path.read_text())
        if {k:v for k,v in old.items() if k!='frozen_at_utc'}!={k:v for k,v in freeze.items() if k!='frozen_at_utc'}:
            raise ValueError('Protocol changed after freeze; requires explicit versioned revision')
    else:write_json(freeze,freeze_path)
    table(mt,'validation_metrics');table(wt,'validation_warning');table(pd.DataFrame(coef),'training_coefficients')
    table(pd.DataFrame(skips,columns=['horizon','reason']),'unsupported_horizons')
    # Only after protocol and thresholds are on disk do we calculate transfer scores.
    for h,f in frames.items():
        test=f.loc[f.partition.eq('test')&f.target.notna()];ev=event_cohort(events,'test')
        for name,spec in specs.items():
            if (h,name) not in objects:continue
            obj=objects[h,name];score=scores(obj,test)
            rows.append(dict(partition='test',horizon=h,model=name,raw_features=len(spec['features'])+len(spec['interactions']),encoded_features=len(obj['encoder'].names),**metrics(test,score)))
            for lead in [x for x in CFG['minimum_leads'] if x<=h]:
                for rate in CFG['alert_rates']:
                    m,_=warning(test,score,ev,rate,lead,h);warn.append(dict(partition='test',horizon=h,model=name,**m))
            p=test[['row_id','pitcher_id','game_id','game_date','target']].copy();p['score']=score;p['model']=name;p['horizon']=h;p['partition']='test';pred.append(p)
    table(pd.DataFrame(rows),'model_metrics');table(pd.DataFrame(warn),'warning_metrics');parquet(pd.concat(pred,ignore_index=True),'predictions')
    event_rows=[];fixed=[]
    for part in ['validation','test']:
        f=frames[30].loc[frames[30].partition.eq(part)&frames[30].target.notna()];ev=event_cohort(events,part)
        for name in specs:
            score=scores(objects[30,name],f)
            for lead in CFG['minimum_leads']:
                m,e=warning(f,score,ev,.05,lead,30)
                e['partition']=part;e['model']=name;e['minimum_lead']=lead;event_rows.append(e)
                boot=event_bootstrap(e);m['recall_ci_low'],m['recall_ci_high']=np.quantile(boot,[.025,.975])
                fixed.append(dict(partition=part,model=name,**m))
                m,_=warning(f,score,ev,.05,lead,30,thresholds['30|'+name]);fixed.append(dict(partition=part,model=name,**m))
    table(pd.concat(event_rows,ignore_index=True),'event_warning_details');table(pd.DataFrame(fixed),'warning_policies')
    checkpoint('models',[OUT/'frozen_protocol.json',OUT/'tables/model_metrics.csv',OUT/'tables/warning_metrics.csv',CACHE/'predictions.parquet'])
    print('Frozen validation-selected model:',selected,flush=True)
