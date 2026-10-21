"""Small train/validation search: additive splines, boosted stumps and shallow trees."""
import hashlib,json,time,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .consolidation_models import ConsolidationEncoder
from .ucl_dynamic_models import model_path as linear_path,scores as linear_scores
from .ucl_dynamic_data import event_cohort
from .ucl_dynamic_evaluation import metrics,warning
from .ucl_nonlinear_common import ROOT,OUT,CACHE,CFG,inputs,snapshot,table,parquet,write_json,checkpoint,valid_checkpoint

class AdditiveSplineEncoder:
    """Separate cubic bases; no cross-products or interactions, including missingness.

    Low-support or discrete variables remain linear. Every knot and transformation
    is training-derived. A missing-value indicator stays an additive main effect.
    """
    def __init__(self,features):self.features=list(features)
    def fit(self,f):
        self.base=ConsolidationEncoder(self.features).fit(f);z=self.base.transform(f)
        self.splines={};self.names=[];self.sources=[];self.rules=[]
        for j,c in enumerate(self.base.names):
            q=np.unique(np.quantile(z[:,j],np.linspace(0,1,CFG['spline_knots'])))
            if j<len(self.features) and len(np.unique(z[:,j]))>=8 and len(q)>=3:
                tr=SplineTransformer(knots=q[:,None],degree=CFG['spline_degree'],include_bias=False,extrapolation='constant')
                tr.fit(z[:,[j]]);self.splines[j]=tr;n=tr.n_features_out_
                self.names.extend([c+f'__spline_{k}' for k in range(n)]);self.sources.extend([c]*n)
                rule='additive cubic spline, '+str(len(q))+' distinct training knots'
            else:self.names.append(c);self.sources.append(c);rule='linear: indicator, discrete or insufficient distinct knot support'
            self.rules.append(dict(feature=c,rule=rule))
        self.scaler=StandardScaler().fit(self.expand(z));return self
    def expand(self,z):
        return np.column_stack([self.splines[j].transform(z[:,[j]]) if j in self.splines else z[:,[j]] for j in range(z.shape[1])])
    def transform(self,f):return self.scaler.transform(self.expand(self.base.transform(f)))

def model_id(representation,algorithm,setting):
    return representation.replace(' + ','_').replace(' ','_')+'__'+algorithm+'__'+str(setting).replace('.','p')
def candidates(specs):
    rows=[]
    for rep,spec in specs.items():
        for j,c in enumerate(CFG['spline_C']):
            rows.append(dict(model_id=model_id(rep,'Spline',c),representation=rep,algorithm='Spline',setting=c,complexity=j,features=spec['features']))
        for j,(leaves,depth) in enumerate(zip(CFG['tree_leaf_limits'],CFG['tree_depth_limits'])):
            rows.append(dict(model_id=model_id(rep,'Boosting',leaves),representation=rep,algorithm='Boosting',setting=leaves,depth=depth,complexity=j,features=spec['features']))
    return rows
def fit(train,spec,name=None):
    if not train.partition.eq('train').all():raise ValueError('Training partition required')
    name=spec['model_id'] if name is None else name
    signature=hashlib.sha256(json.dumps(dict(config=CFG,spec=spec,keys=train.row_id.tolist(),target=train.target.astype(int).tolist()),sort_keys=True).encode()).hexdigest()
    path=CACHE/'models'/(name+'.joblib')
    if path.exists():
        obj=joblib.load(path)
        if obj['signature']!=signature:raise ValueError('Changed cached fit '+name)
        return obj
    start=time.monotonic()
    encoder=(AdditiveSplineEncoder(spec['features']) if spec['algorithm']=='Spline' else ConsolidationEncoder(spec['features'])).fit(train)
    if spec['algorithm']=='Spline':model=LogisticRegression(C=spec['setting'],max_iter=5000,tol=1e-7,solver='lbfgs')
    elif spec['algorithm']=='Linear':model=LogisticRegression(C=1.,max_iter=5000,tol=1e-7,solver='lbfgs')
    else:
        model=HistGradientBoostingClassifier(learning_rate=CFG['tree_learning_rate'],max_iter=CFG['tree_iterations'],max_leaf_nodes=int(spec['setting']),
            max_depth=spec['depth'],min_samples_leaf=CFG['tree_min_leaf'],l2_regularization=CFG['tree_l2'],
            early_stopping=False,random_state=CFG['seed'],class_weight=None)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);model.fit(encoder.transform(train),train.target.to_numpy(dtype=int))
    obj=dict(encoder=encoder,model=model,spec=spec,signature=signature,training_max_date=str(train.game_date.max()),
        rows=len(train),positive_rows=int(train.target.sum()),seconds=time.monotonic()-start)
    tmp=path.with_suffix('.joblib.partial');joblib.dump(obj,tmp);tmp.replace(path)
    print(f'Fit {name}: {len(spec["features"])} raw fields, {obj["seconds"]:.1f}s',flush=True)
    return obj
def scores(obj,f,logit=False):
    with threadpool_limits(limits=1):
        z=obj['encoder'].transform(f)
        return obj['model'].decision_function(z) if logit else obj['model'].predict_proba(z)[:,1]
def load_model(mid):
    if mid.startswith('Linear|'):
        return joblib.load(ROOT/'data/injury/analysis/ucl_dynamic/models'/(linear_path(30,mid.split('|',1)[1])+'.joblib'))
    return joblib.load(CACHE/'models'/(mid+'.joblib'))
def select(validation):
    """Selection explicitly rejects transfer rows; no model is guaranteed superior."""
    if not validation.partition.eq('validation').all():raise ValueError('Validation-only selection')
    eligible=validation.loc[validation.average_precision>validation.prevalence]
    if eligible.empty:return validation.sort_values(['raw_features','complexity','model_id']).iloc[0].model_id
    return eligible.sort_values(['injury_event_recall','raw_features','complexity','average_precision','model_id'],ascending=[False,True,True,False,True]).iloc[0].model_id

def run_models():
    if valid_checkpoint('models'):return
    snapshot();f,events,specs=inputs();tr=f.loc[f.partition.eq('train')&f.target.notna()];va=f.loc[f.partition.eq('validation')&f.target.notna()]
    grid=candidates(specs);write_json(dict(config=CFG,candidates=grid,selection_period='2022–2023 only',
        rationale='30-day first-event landmark; all existing timing/competing/embargo/follow-up rules unchanged; no class balancing or synthetic positives'),OUT/'predeclared_analysis.json')
    rows=[];objects={};catalog={};validation_scores={};cohort=[]
    for part in ['train','validation','test']:
        g=f.loc[f.partition.eq(part)&f.target.notna()];cohort.append(dict(partition=part,rows=len(g),pitchers=g.pitcher_id.nunique(),positive_rows=int(g.target.sum()),represented_events=g.loc[g.target.eq(1),'dynamic_next_event_id'].nunique()))
    table(pd.DataFrame(cohort),'cohort_counts')
    for rep,spec in specs.items():
        mid='Linear|'+rep;obj=load_model(mid);objects[mid]=obj;catalog[mid]=dict(model_id=mid,representation=rep,algorithm='Linear',setting=1.,complexity=0,features=spec['features'])
    for spec in grid:objects[spec['model_id']]=fit(tr,spec);catalog[spec['model_id']]=spec
    for mid,obj in objects.items():
        spec=catalog[mid]
        for part,g in [('train',tr),('validation',va)]:
            s=scores(obj,g);w,_=warning(g,s,event_cohort(events,part));m=metrics(g,s)
            rows.append(dict(partition=part,model_id=mid,representation=spec['representation'],algorithm=spec['algorithm'],setting=spec['setting'],
                complexity=spec['complexity'],raw_features=len(spec['features']),**m,**{k:w[k] for k in ['injury_event_recall','alerted_events','qualifying_events','median_warning_lead_days','non_actionable_alert_fraction']}))
            if part=='validation':validation_scores[mid]=s
    dev=pd.DataFrame(rows);v=dev.loc[dev.partition.eq('validation')]
    selected={}
    for (rep,algorithm),g in v.groupby(['representation','algorithm']):selected[rep+'|'+algorithm]=select(g)
    global_warning={alg:select(v.loc[v.algorithm.eq(alg)]) for alg in ['Linear','Spline','Boosting']}
    global_ap={alg:v.loc[v.algorithm.eq(alg)].sort_values(['average_precision','complexity'],ascending=[False,True]).iloc[0].model_id for alg in ['Linear','Spline','Boosting']}
    kept=list(dict.fromkeys(list(selected.values())+list(global_ap.values())))
    freeze=dict(frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),per_representation=selected,warning_leaders=global_warning,
        ranking_leaders=global_ap,transfer_models=kept,thresholds={mid:float(np.quantile(validation_scores[mid],.95)) for mid in kept},
        catalog=catalog,fit_signatures={mid:objects[mid]['signature'] for mid in kept},selection_rule=CFG['selection_rule'],
        upstream_protocol_sha256=__import__('hashlib').sha256((ROOT/'reports/ucl_dynamic/frozen_protocol.json').read_bytes()).hexdigest())
    p=OUT/'frozen_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        if {k:x for k,x in old.items() if k!='frozen_at_utc'}!={k:x for k,x in freeze.items() if k!='frozen_at_utc'}:raise ValueError('Changed protocol after freeze')
    else:write_json(freeze,p)
    table(dev,'development_search');write_json(catalog,CACHE/'model_catalog.json')
    warnings_rows=[];event_rows=[];predictions=[];calibration=[];test=f.loc[f.partition.eq('test')&f.target.notna()]
    for part,g in [('validation',va),('test',test)]:
        ev=event_cohort(events,part)
        for mid in kept:
            obj=objects[mid];spec=catalog[mid];s=validation_scores[mid] if part=='validation' else scores(obj,g)
            if part=='test':
                w,_=warning(g,s,ev);rows.append(dict(partition=part,model_id=mid,representation=spec['representation'],algorithm=spec['algorithm'],setting=spec['setting'],complexity=spec['complexity'],raw_features=len(spec['features']),**metrics(g,s),**{k:w[k] for k in ['injury_event_recall','alerted_events','qualifying_events','median_warning_lead_days','non_actionable_alert_fraction']}))
            pframe=g[['row_id','pitcher_id','game_id','game_date','target']].copy();pframe['score']=s;pframe['model_id']=mid;pframe['partition']=part;predictions.append(pframe)
            for lead in [7,14,21]:
                for rate in [.01,.025,.05,.10]:
                    w,e=warning(g,s,ev,rate,lead,30);warnings_rows.append(dict(partition=part,model_id=mid,**w))
                    if rate==.05:e['model_id']=mid;e['partition']=part;e['minimum_lead']=lead;event_rows.append(e)
                w,_=warning(g,s,ev,.05,lead,30,freeze['thresholds'][mid]);warnings_rows.append(dict(partition=part,model_id=mid,**w))
            edges=np.unique(np.r_[-np.inf,np.quantile(validation_scores[mid],np.arange(.1,1,.1)),np.inf]);b=pd.cut(s,edges,labels=False)
            c=pd.DataFrame(dict(bin=b,score=s,target=g.target.to_numpy(dtype=int))).groupby('bin').agg(rows=('score','size'),mean_score=('score','mean'),observed_risk=('target','mean'),positives=('target','sum')).reset_index()
            c['model_id']=mid;c['partition']=part;calibration.append(c)
    table(pd.DataFrame(rows),'model_metrics');table(pd.DataFrame(warnings_rows),'warning_metrics');table(pd.concat(event_rows,ignore_index=True),'event_warnings')
    table(pd.concat(calibration,ignore_index=True),'calibration_bins');parquet(pd.concat(predictions,ignore_index=True),'predictions')
    checkpoint('models',[OUT/'frozen_protocol.json',OUT/'tables/model_metrics.csv',OUT/'tables/warning_metrics.csv',CACHE/'predictions.parquet']+list((CACHE/'models').glob('*.joblib')))
    print('Validation warning leaders:',global_warning,'; AP leaders:',global_ap,flush=True)
