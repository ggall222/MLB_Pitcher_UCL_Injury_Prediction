"""Focused elbow/UCL visuals from frozen supervised models; no model refitting."""
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from .common import ROOT,CACHE as ORIGINAL_CACHE,source_signature,sha
from .profile_departure import inputs,PROFILE,DEPARTURE,clean
from .supervised_diagnostics import model_path
from .signals_common import CFG as MATCHED_CFG
from .trajectory_shapes import paired_cells

CONFIG_PATH=ROOT/'config/ucl_latent_visuals.yaml'
CFG=yaml.safe_load(CONFIG_PATH.read_text())
OUT=ROOT/CFG['output_path'];CACHE=ROOT/CFG['cache_path']
for p in [OUT,OUT/'tables',OUT/'figures',CACHE]:p.mkdir(parents=True,exist_ok=True)
UPSTREAM=ROOT/'reports/injury_supervised_reduction'
CATEGORY=CFG['category']


def validate_alignment_config():
    """The inherited matched-cell helper must use the documented time windows."""
    for key in ['calendar_bins','last_appearances']:
        if CFG[key]!=MATCHED_CFG[key]:
            raise ValueError(f'{key} must match the existing matched-alignment configuration')


def paired_dependency_clusters(aligned):
    """Keep events sharing either a case OR control pitcher in one bootstrap unit.

    A pitcher can be a healthy control for one episode and a case later. Connected
    matched-pair sets account for that dependence without changing any matches.
    """
    parent={}
    def root(value):
        parent.setdefault(value,value)
        while parent[value]!=value:
            parent[value]=parent[parent[value]]
            value=parent[value]
        return value
    memberships=aligned[['event_id','pitcher_id']].drop_duplicates()
    for _,g in memberships.groupby('event_id',sort=True):
        people=sorted(g.pitcher_id.unique())
        for person in people[1:]:
            first,second=root(people[0]),root(person)
            parent[max(first,second)]=min(first,second)
    return memberships.groupby('event_id').pitcher_id.first().map(root)


def bootstrap_weights(ids):
    """Deterministic whole-dependency-cluster draws; events retain equal weight."""
    unique,inverse=np.unique(ids,return_inverse=True)
    if len(unique)<2:raise ValueError('Need at least two independent matched-pair clusters')
    rng=np.random.default_rng(CFG['seed'])
    counts=np.array([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
                     for _ in range(CFG['bootstrap_replicates'])])
    return counts[:,inverse]


def table(frame,name):frame.to_csv(OUT/'tables'/f'{name}.csv',index=False);return frame
def read_table(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')
def figure(fig,name):
    for suffix in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{suffix}',dpi=240,bbox_inches='tight')
    return fig


def select_components(fitted,block):
    """Selection is based only on frozen validation k and training coefficients/SD."""
    basis=fitted['basis'];k=fitted['selected']['PLS']
    beta=fitted['models']['PLS selected'].coef_[0][:k]
    sd=basis.training_score_sd['PLS'][:k]
    if not np.isfinite(sd).all() or np.any(sd<=0):
        raise ValueError('Frozen training component scales must be finite and positive')
    importance=np.abs(beta*sd)
    rank=np.argsort(-importance,kind='stable')
    prefix='D' if block=='Departure' else 'C'
    return pd.DataFrame([dict(component=f'{prefix}{j+1}',block=block,axis=j+1,
        retained_axes=k,design_columns=k+len(basis.keep),native_coefficient=beta[j],
        training_score_sd=sd[j],orientation=1 if beta[j]>=0 else -1,
        training_absolute_standardized_logit_weight=importance[j],
        training_rank=int(np.flatnonzero(rank==j)[0]+1),
        primary_trajectory=int(j in rank[:CFG['trajectory_components_per_block']]),
        individual_heatmap=bool(block==CFG['individual_representation'] and j==rank[0]),
        selection_basis='Existing validation-selected count; ordering from training coefficient × training score SD')
        for j in range(k)])


def build_alignment(a,events,matches):
    before=source_signature()
    old=pd.read_parquet(ORIGINAL_CACHE/'aligned_trajectories.parquet')
    keys=['event_id','analysis_group','cluster_pitcher_id','pitcher_id','game_id','game_date',
          'lead_days','appearance_before','kind']
    f=old.loc[old.analysis_group.eq(CATEGORY),keys].copy()
    f=f.merge(a[['pitcher_id','game_id']+PROFILE+DEPARTURE],on=['pitcher_id','game_id'],
              how='left',validate='many_to_one',indicator=True)
    assert f._merge.eq('both').all(),'Original matched appearance absent from frozen cohort'
    f=f.drop(columns='_merge')
    f=f.merge(matches[['event_id','anchor_date','control_anchor_date']],on='event_id',validate='many_to_one')
    valid=f.assign(in_period=f.game_date.between('2022-01-01','2025-12-31')).groupby('event_id').in_period.all()
    valid &= matches.set_index('event_id').anchor_date.between('2022-01-01','2025-12-31').reindex(valid.index)
    valid &= matches.set_index('event_id').control_anchor_date.between('2022-01-01','2025-12-31').reindex(valid.index)
    f=f.loc[f.event_id.map(valid)].copy()
    f['complete']=clean(f,PROFILE+DEPARTURE).notna().all(axis=1)
    assert f.game_date.lt(f.anchor_date.where(f.kind.eq('Injury'),f.control_anchor_date)).all()
    assert not f[['event_id','kind','pitcher_id','game_id']].duplicated().any()
    definitions=[];loadings=[];model_sources={}
    for block in CFG['blocks']:
        path=model_path(CATEGORY,block);model_sources[str(path.relative_to(ROOT))]=sha(path)
        fitted=joblib.load(path);basis=fitted['basis'];selection=select_components(fitted,block)
        k=fitted['selected']['PLS'];scores=basis.transform(f,'PLS',k)[:,:k]
        for r in selection.itertuples():
            f[r.component]=scores[:,r.axis-1]*r.orientation/r.training_score_sd
            f.loc[~f.complete,r.component]=np.nan
            for feature,weight in zip(basis.continuous,basis.bases['PLS'][:,r.axis-1]):
                loadings.append(dict(component=r.component,block=block,feature=feature,
                    injury_oriented_weight=weight*r.orientation,squared_axis_weight=weight**2,
                    representation='Profile' if feature in PROFILE else 'Departure'))
        definitions.append(selection)
    definition=pd.concat(definitions,ignore_index=True)
    f=f.merge(events[['injury_event_id','player_name','initial_injury_structure','initial_explicit_ucl',
                    'supplemental_ucl_confirmed','strict_qa_eligible']],left_on='event_id',
              right_on='injury_event_id',validate='many_to_one')
    f.to_parquet(CACHE/'aligned_components.parquet',index=False)
    table(definition,'component_selection')
    table(pd.DataFrame(loadings),'component_loadings')
    assert source_signature()==before and all(sha(ROOT/p)==h for p,h in model_sources.items())
    return f,definition,model_sources


def balanced_summary(aligned,components,mode):
    """One shared set of pairs; max-t bands across ALL components and time points."""
    validate_alignment_config()
    if mode not in ['appearance','calendar']:raise ValueError(f'Unknown alignment: {mode}')
    points=list(range(5 if mode=='appearance' else 4))
    cells=pd.concat([paired_cells(aligned,c,mode).assign(component=c) for c in components],ignore_index=True)
    valid=cells.groupby(['event_id','component']).point.nunique().unstack('component').reindex(columns=components)
    ids=sorted(valid.index[valid.eq(len(points)).all(axis=1)])
    if len(ids)<2:raise ValueError(f'Insufficient balanced matched support for {mode}')
    cells=cells.loc[cells.event_id.isin(ids)].copy()
    pid=cells.drop_duplicates('event_id').set_index('event_id').cluster_pitcher_id.reindex(ids)
    clusters=paired_dependency_clusters(aligned.loc[aligned.event_id.isin(ids)]).reindex(ids)
    cells['resample_cluster_id']=cells.event_id.map(clusters)
    weights=bootstrap_weights(clusters.to_numpy());denominator=weights.sum(axis=1)[:,None]
    grid=pd.MultiIndex.from_product([components,points],names=['component','point'])
    records=[]
    for kind in ['Injury','Control','difference']:
        values=cells.pivot(index='event_id',columns=['component','point'],values=kind).reindex(index=ids,columns=grid).to_numpy()
        if not np.isfinite(values).all():raise AssertionError('Balanced matrix has missing cells')
        mean=values.mean(axis=0);boot=(weights@values)/denominator
        sd=boot.std(axis=0,ddof=1)
        normalized=np.divide(boot-mean,sd,out=np.zeros_like(boot),where=sd>1e-12)
        critical=np.quantile(np.abs(normalized).max(axis=1),.95)
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        for j,(component,point) in enumerate(grid):
            records.append(dict(mode=mode,component=component,point=point,kind=kind,mean=mean[j],
                low=lo[j],high=hi[j],simultaneous_low=mean[j]-critical*sd[j],
                simultaneous_high=mean[j]+critical*sd[j],events=len(ids),pitchers=pid.nunique(),
                bootstrap_clusters=clusters.nunique(),
                simultaneous_family=f'All {len(components)} retained components × {len(points)} positions; {kind}'))
    cells['mode']=mode
    return pd.DataFrame(records),cells


def shape_summaries(cells):
    event_rows=[]
    for (component,mode,event),g in cells.groupby(['component','mode','event_id']):
        g=g.sort_values('point');d=g.difference.to_numpy()
        for metric,value in [('average_separation',d.mean()),('last_minus_first_separation',d[-1]-d[0]),
                             ('final_step_separation',d[-1]-d[-3:-1].mean()),
                             ('case_last_minus_first',g.Injury.iloc[-1]-g.Injury.iloc[0])]:
            event_rows.append(dict(component=component,mode=mode,event_id=event,pitcher_id=g.cluster_pitcher_id.iloc[0],
                                   resample_cluster_id=g.resample_cluster_id.iloc[0],
                                   metric=metric,value=value))
    event=pd.DataFrame(event_rows);event.to_parquet(CACHE/'event_shape_values.parquet',index=False)
    summary=[]
    for (c,m,k),g in event.groupby(['component','mode','metric']):
        weights=bootstrap_weights(g.resample_cluster_id.to_numpy())
        boot=weights@g.value.to_numpy()/weights.sum(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        summary.append(dict(component=c,mode=m,metric=k,mean=g.value.mean(),low=lo,high=hi,
            events=len(g),pitchers=g.pitcher_id.nunique(),bootstrap_clusters=g.resample_cluster_id.nunique()))
    return table(pd.DataFrame(summary),'trajectory_shape_summary')


def run():
    validate_alignment_config()
    a,events,spec,matches=inputs()
    aligned,definitions,models=build_alignment(a,events,matches)
    summaries=[];cells=[]
    for mode in ['appearance','calendar']:
        s,c=balanced_summary(aligned,definitions.component.tolist(),mode)
        summaries.append(s);cells.append(c)
    summary=table(pd.concat(summaries,ignore_index=True),'balanced_trajectories')
    cell=pd.concat(cells,ignore_index=True);cell.to_parquet(CACHE/'balanced_event_cells.parquet',index=False)
    shape_summaries(cell)
    table(events.loc[events.target_eligible&events.analysis_group.eq(CATEGORY)].groupby('season').agg(
        elbow_ucl_proxy_events=('injury_event_id','size'),initial_explicit_ucl_events=('initial_explicit_ucl','sum'),
        later_supplemental_ucl_confirmation=('supplemental_ucl_confirmed','sum')).reset_index(),'endpoint_definition_audit')
    coverage=[]
    for cohort,ids in [('All eligible proxy events',set(events.loc[events.target_eligible&events.analysis_group.eq(CATEGORY),'injury_event_id'])),
                       ('Out-of-training matched pairs',set(aligned.event_id)),
                       ('Balanced five outings',set(cell.loc[cell['mode'].eq('appearance'),'event_id'])),
                       ('Balanced calendar bins',set(cell.loc[cell['mode'].eq('calendar'),'event_id']))]:
        e=events.loc[events.injury_event_id.isin(ids)]
        coverage.append(dict(cohort=cohort,events=len(e),pitchers=e.player_id.nunique(),
            initial_explicit_ucl_events=int(e.initial_explicit_ucl.fillna(False).sum()),
            later_supplemental_ucl_confirmation=int(e.supplemental_ucl_confirmed.fillna(False).sum())))
    table(pd.DataFrame(coverage),'cohort_support')
    # Initial explicit UCL is shown as a count/audit, never silently substituted
    # for the much broader outcome on which the saved PLS was trained.
    table(aligned.drop_duplicates('event_id')[['event_id','player_name','anchor_date','initial_injury_structure',
        'initial_explicit_ucl','supplemental_ucl_confirmed','strict_qa_eligible']],'matched_event_audit')
    individual=definitions.loc[definitions.individual_heatmap,'component'].iloc[0]
    g=cell.loc[cell['mode'].eq('appearance')&cell.component.eq(individual)].copy()
    case=g.pivot(index='event_id',columns='point',values='Injury')
    order=(case[4]-case[[0,1]].mean(axis=1)).sort_values(kind='stable')
    ids=order.index.tolist()
    keys=events.set_index('injury_event_id').reindex(ids)[['player_id','player_name','anchor_date','initial_explicit_ucl']]
    keys['row_number']=np.arange(1,len(keys)+1);keys['event_label']=['E%02d'%i for i in keys.row_number]
    keys['case_late_minus_early']=order
    keys['case_mean_score']=case.mean(axis=1).reindex(ids);keys['case_final_score']=case[4].reindex(ids)
    table(keys.reset_index(),'individual_event_order')
    table(g.merge(keys[['row_number','event_label']],left_on='event_id',right_index=True,validate='many_to_one'),
          'individual_event_cells')
    leads=aligned.loc[aligned.event_id.isin(ids)&aligned.kind.eq('Injury')&aligned.appearance_before.le(5)]
    table(leads.groupby('appearance_before').lead_days.agg(n='size',median='median',q25=lambda v:v.quantile(.25),
        q75=lambda v:v.quantile(.75),minimum='min',maximum='max').reset_index(),'outing_calendar_support')
    # Curves and uncertainty are copied as derived narrow views, never recomputed
    # or selected using the matched trajectories or held-out test performance.
    for file in ['component_curves','validation_selection','selected_performance','selected_differences','alert_budgets']:
        source=pd.read_csv(UPSTREAM/'tables'/f'{file}.csv')
        table(source.loc[source.category.eq(CATEGORY)].copy(),'ucl_'+file)
    manifest={'source':source_signature(),'config':CFG,'config_sha256':sha(CONFIG_PATH),'frozen_models':models,
        'original_matching_sha256':sha(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv'),
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [
            ORIGINAL_CACHE/'aligned_trajectories.parquet',ORIGINAL_CACHE/'appearances.parquet',
            ORIGINAL_CACHE/'events.parquet']},
        'upstream_curves_sha256':sha(UPSTREAM/'tables/component_curves.csv'),
        'endpoint':'Original broad Elbow / UCL category, not exclusively confirmed ligament injury',
        'component_selection':'Frozen validation counts; training-only coefficient magnitude selects emphasized axes',
        'heldout_period':'2022–2025, previously examined; exploratory',
        'uncertainty':f'{CFG["bootstrap_replicates"]} connected matched-pair pitcher-cluster bootstraps; joint bands across all retained axes and positions within alignment/kind',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/injury_analysis').glob('ucl_latent*.py')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Elbow/UCL: {aligned.event_id.nunique()} matched pairs; {len(ids)} balanced five-outing pairs. No model refitting.',flush=True)
    return summary
