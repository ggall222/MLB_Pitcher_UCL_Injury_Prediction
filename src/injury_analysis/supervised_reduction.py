"""Training-only PLS/PCA subspaces with comparable L2-logistic geometry."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .common import ROOT,CFG as ORIGINAL_CFG,CATEGORIES,source_signature,sha
from .reduction import PCAEncoder,clean
from .profile_departure import PROFILE,DEPARTURE,inputs,evaluate_scores,probability_metrics
from .family_decomposition import feature_variants,target_events
from .cohort import label_frame
from src.evaluation import rank_metrics

CONFIG_PATH=ROOT/'config/injury_supervised_reduction.yaml'
CFG=yaml.safe_load(CONFIG_PATH.read_text())
from .signals_common import CFG as SHARED_RESAMPLING_CFG
if any(CFG[k]!=SHARED_RESAMPLING_CFG[k] for k in ['seed','bootstrap_replicates']):
    raise ValueError('Paired evaluation reuses injury_signals.yaml; seed and bootstrap_replicates must agree.')
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE]:p.mkdir(parents=True,exist_ok=True)
BINARY=['throws_left','is_starter']


def table(f,name):f.to_csv(OUT/'tables'/f'{name}.csv',index=False);return f
def read_table(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')
def figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    return fig


def feature_blocks(spec):
    variants,_=feature_variants(spec)
    return {'Original 62':variants['Full'],'Profile':PROFILE,'Departure':DEPARTURE,'Combined':PROFILE+DEPARTURE}


def choose_components(curve):
    """Only validation rows can select complexity; higher AP, then smaller k."""
    valid=curve.loc[curve.partition.eq('validation')]
    if valid.empty:raise ValueError('Component selection requires validation results')
    return int(valid.sort_values(['average_precision','components'],ascending=[False,True]).iloc[0].components)


def orthogonal_span(rotations):
    """Nested QR preserves each PLS span while removing arbitrary L2 anisotropy."""
    q,r=np.linalg.qr(rotations,mode='reduced')
    if np.any(np.abs(np.diag(r))<1e-9):raise ValueError('PLS directions lost rank; reduce component grid')
    signs=np.where(np.diag(r)<0,-1.,1.)
    return q*signs


class SupervisedBasis:
    """Learn both bases on one category/block training sample, never heldout labels.

    Continuous fields are projected. Binary context and training-observed missing
    indicators bypass reduction identically for both methods. QR orthogonalizes the
    PLS rotations but keeps every nested supervised subspace. No whitening or
    latent rescaling changes the isotropic L2 penalty.
    """
    def __init__(self,features,grid):
        self.features=list(features);self.requested_grid=list(grid)

    def fit(self,train):
        self.encoder=PCAEncoder(self.features).fit(train)
        self.continuous=[c for c in self.features if c not in BINARY]
        self.index=[self.encoder.names.index(c) for c in self.continuous]
        self.keep=[j for j in range(len(self.encoder.names)) if j not in self.index]
        z=self.encoder.standardized(train);x=z[:,self.index]
        self.center=x.mean(axis=0);x=x-self.center
        with threadpool_limits(limits=1),warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.pca=PCA(svd_solver='full').fit(x)
            self.k90=int(np.searchsorted(np.cumsum(self.pca.explained_variance_ratio_),.9)+1)
            self.grid=sorted(set(k for k in self.requested_grid+[self.k90] if 1<=k<=len(self.index)))
            self.pls=PLSRegression(n_components=max(self.grid),scale=False,max_iter=1000,tol=CFG['pls_tolerance'])
            self.pls.fit(x,train.target.to_numpy(dtype=float))
        self.bases={'PCA':self.pca.components_.T,'PLS':orthogonal_span(self.pls.x_rotations_)}
        self.fit_rows=len(train);self.training_prevalence=float(train.target.mean())
        self.training_components={name:x@q for name,q in self.bases.items()}
        self.training_total_variance=np.var(x,axis=0).sum()
        self.training_score_sd={name:np.std(value,axis=0,ddof=0) for name,value in self.training_components.items()}
        return self

    def transform(self,frame,method,components):
        z=self.encoder.standardized(frame)
        if method=='Unreduced':return z
        if method=='Context/availability only':
            return z[:,self.keep] if self.keep else np.ones((len(frame),1))
        if components not in range(1,self.bases[method].shape[1]+1):raise ValueError('Invalid retained dimension')
        scores=(z[:,self.index]-self.center)@self.bases[method][:,:components]
        return np.column_stack([scores,z[:,self.keep]])

    def score_variance_fraction(self,method,k):
        return np.var(self.training_components[method][:,:k],axis=0).sum()/self.training_total_variance

    def interpret(self,train,method,components):
        z=self.encoder.standardized(train)[:,self.index]-self.center
        scores=z@self.bases[method][:,:components]
        corr=np.corrcoef(np.column_stack([z,scores]),rowvar=False)[:len(self.index),len(self.index):]
        return pd.DataFrame([dict(component=j+1,feature=c,axis_weight=self.bases[method][i,j],
            squared_axis_weight=self.bases[method][i,j]**2,feature_score_correlation=corr[i,j])
            for j in range(components) for i,c in enumerate(self.continuous)])


def fit_classifier(x,y,targets):
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=ORIGINAL_CFG['logistic_C'],max_iter=3000,solver='lbfgs',
                                 tol=CFG['logistic_tolerance'],random_state=CFG['seed'])
        model.fit(x,y)
        scores=[model.predict_proba(t)[:,1] for t in targets]
    return model,scores


def fit_block(train,valid,test,features,block,category,condition='Primary'):
    basis=SupervisedBasis(features,CFG['component_grids'][block]).fit(train)
    parts={'validation':valid,'test':test};curves=[];candidates={};models={}
    for method in ['Unreduced','Context/availability only','PCA','PLS']:
        grid=basis.grid if method in ['PCA','PLS'] else [0]
        for k in grid:
            model,scores=fit_classifier(basis.transform(train,method,k),train.target,
                                       [basis.transform(g,method,k) for g in parts.values()])
            candidates[method,k]=dict(zip(parts,scores));models[method,k]=model
            for part,score in zip(parts,scores):
                g=parts[part]
                curves.append(dict(category=category,block=block,condition=condition,method=method,components=k,
                    design_columns=basis.transform(g.iloc[:1],method,k).shape[1],passthrough_columns=len(basis.keep),
                    partition=part,rows=len(g),positive_rows=int(g.target.sum()),prevalence=g.target.mean(),
                    **rank_metrics(g.target,score),**probability_metrics(g.target,score),
                    training_variance_retained=basis.score_variance_fraction(method,k) if method in ['PCA','PLS'] else np.nan))
    curve=pd.DataFrame(curves);selected={};selection=[]
    for method in ['PCA','PLS']:
        grid=curve.loc[curve.method.eq(method)]
        k=choose_components(grid);selected[method]=k
        val=grid.loc[grid.partition.eq('validation')].copy()
        best=float(val.average_precision.max());baseline=float(val.prevalence.iloc[0])
        cutoff=baseline+CFG['plateau_fraction_of_validation_ap_gain']*(best-baseline)
        plateau=int(val.loc[val.average_precision.ge(cutoff),'components'].min()) if best>baseline else np.nan
        selection.append(dict(category=category,block=block,condition=condition,method=method,selected_components=k,
            continuous_features=len(basis.index),passthrough_columns=len(basis.keep),design_columns=k+len(basis.keep),
            validation_ap=best,plateau_components=plateau,plateau_cutoff_ap=cutoff,
            training_variance_retained=basis.score_variance_fraction(method,k),pca_variance_90_components=basis.k90,
            train_rows=len(train),train_positive_rows=int(train.target.sum()),
            train_positive_events=train.loc[train.target.eq(1),'next_event_id'].nunique()))
    frozen={p:{'Unreduced':candidates['Unreduced',0][p],
               'Context/availability only':candidates['Context/availability only',0][p],
               'PCA selected':candidates['PCA',selected['PCA']][p],
               'PLS selected':candidates['PLS',selected['PLS']][p]} for p in parts}
    fitted={'basis':basis,'selected':selected,'models':{
        'Unreduced':models['Unreduced',0],'PCA selected':models['PCA',selected['PCA']],
        'PLS selected':models['PLS',selected['PLS']]}}
    # A complete orthonormal extension of the supervised span retains all fields.
    # This is a numerical L2-geometry control, not a candidate compressed model.
    z=basis.encoder.standardized(train);q=np.linalg.qr(basis.bases['PLS'],mode='complete')[0]
    def rotate(g):
        zz=basis.encoder.standardized(g)
        return np.column_stack([(zz[:,basis.index]-basis.center)@q,zz[:,basis.keep]])
    _,control=fit_classifier(rotate(train),train.target,[rotate(g) for g in parts.values()])
    checks=[dict(category=category,block=block,condition=condition,partition=p,
                 max_score_difference=np.max(np.abs(score-frozen[p]['Unreduced'])),
                 ap_difference=rank_metrics(parts[p].target,score)['average_precision']-
                               rank_metrics(parts[p].target,frozen[p]['Unreduced'])['average_precision'])
            for p,score in zip(parts,control)]
    return basis,curve,pd.DataFrame(selection),frozen,fitted,pd.DataFrame(checks)


def run_prediction(a,events,spec):
    before=source_signature();performance=[];budgets=[];differences=[];leads=[];curves=[];selections=[];loadings=[];sanity=[];wide=[]
    blocks=feature_blocks(spec)
    for category in CATEGORIES:
        frame=label_frame(a,category,CFG['horizon_days'])
        parts={p:frame.loc[frame.partition.eq(p)].copy() for p in ['train','validation','test']}
        for block,features in blocks.items():
            print(f'Supervised reduction: {category} / {block}',flush=True)
            basis,curve,selection,scores,fitted,check=fit_block(*parts.values(),features,block,category)
            curves.append(curve);selections.append(selection);sanity.append(check)
            for method in ['PCA','PLS']:
                loading=basis.interpret(parts['train'],method,min(3,len(basis.index)))
                loadings.append(loading.assign(category=category,block=block,method=method))
            for p in ['validation','test']:
                met,b,l,d=evaluate_scores(parts[p],scores[p],target_events(events,category,p),category,p,analysis=block,
                    contrasts=[('PLS selected','Unreduced'),('PLS selected','PCA selected'),('PCA selected','Unreduced')])
                performance.append(met.assign(block=block));budgets.append(b.assign(block=block))
                leads.append(l.assign(block=block));differences.append(d.assign(block=block))
                w=parts[p][['row_id','pitcher_id','game_id','game_date','season','is_starter','target','next_event_id',
                            'next_event_lead_days']].copy().assign(category=category,partition=p,block=block)
                for n,s in scores[p].items():w['score__'+n]=s
                wide.append(w)
            name=category.replace('/','-').replace(' ','_')+'__'+block.replace(' ','_')
            # Remove large in-sample component matrices after summary generation.
            basis.training_components=None
            basis.pls.x_scores_=None
            basis.pls.y_scores_=None
            joblib.dump(fitted,CACHE/f'model_{name}.joblib')
            table(pd.concat(performance,ignore_index=True),'selected_performance')
            table(pd.concat(selections,ignore_index=True),'validation_selection')
    for data,name in [(curves,'component_curves'),(budgets,'alert_budgets'),(differences,'selected_differences'),
                      (leads,'warning_leads'),(loadings,'basis_loadings'),(sanity,'full_span_control')]:
        table(pd.concat(data,ignore_index=True),name)
    pd.concat(wide,ignore_index=True).to_parquet(CACHE/'predictions.parquet',index=False)
    assert source_signature()==before
    save_manifest()
    return pd.concat(performance,ignore_index=True)


def save_manifest():
    result={'source':source_signature(),'configuration':CFG,'config_sha256':sha(CONFIG_PATH),
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/injury_analysis').glob('supervised*.py')},
        'previous_profile_predictions_sha256':sha(ROOT/'data/injury/analysis/profile_departure/predictions.parquet'),
        'test_status':'2024–2025 was previously examined; exploratory, not independent confirmation.'}
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2))
    return result
