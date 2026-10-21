"""Event-composition sensitivity without changing intervals, families or models."""
import numpy as np
import pandas as pd
from src.evaluation import rank_metrics
from .ucl_temporal_common import OUT,CACHE,CFG,table,checkpoint,valid_checkpoint,interval_name
from .ucl_temporal_data import prepare,interval_target

def run_composition(part='validation'):
    stage='composition_'+part
    if valid_checkpoint(stage):return
    f,events,groups,models=prepare()
    base=f.loc[f.partition.eq(part)&f.target.notna()].copy()
    sets=[]
    for lo,hi in CFG['intervals']:
        g=interval_target(base,lo,hi)
        sets.append(set(g.loc[g.target.eq(1),'dynamic_next_event_id']))
    shared=set.intersection(*sets)
    first_ucl=base.dynamic_next_target & base.dynamic_lead_days.between(1,60)
    # Restrict event COMPOSITION, not controls' future risk based on predictors.
    # Retrospective sensitivity only; keep all other observed comparison rows.
    keep=~first_ucl|base.dynamic_next_event_id.isin(shared)
    rows=[];composition=[];contributions=[];influence=[]
    for lo,hi in CFG['intervals']:
        interval=interval_name(lo,hi);g=interval_target(base,lo,hi).reset_index(drop=True)
        p=pd.read_parquet(CACHE/f'predictions_{part}_{interval}.parquet')
        assert g.row_id.tolist()==p.row_id.tolist()
        mask=keep.to_numpy()
        for name in models:
            y=g.loc[mask,'target'].to_numpy(dtype=int);s=p.loc[mask,name].to_numpy()
            m=rank_metrics(y,s)
            rows.append(dict(partition=part,interval=interval,model=name,rows=len(y),
                shared_events=len(shared),positive_rows=int(y.sum()),prevalence=float(y.mean()),
                pr_lift=m['average_precision']/y.mean() if y.mean() else np.nan,**m))
        for outcome,gg in [('positive',g.loc[g.target.eq(1)]),('comparison',g.loc[g.target.eq(0)])]:
            composition.append(dict(partition=part,interval=interval,outcome=outcome,rows=len(gg),
                pitchers=gg.pitcher_id.nunique(),starter_fraction=float(gg.is_starter.mean()),
                median_prior_appearances=float(gg.prior_appearance_count.median()),
                median_age=float(gg.age_at_appearance.median())))
        # Every standalone family and reference retained; no mining only favorable cases.
        inspect=[n for n in models if not n.startswith('Without | ')]
        for name in inspect:
            s=p[name].to_numpy();ap=rank_metrics(g.target,s)['average_precision']
            order=np.argsort(-s,kind='stable');y=g.target.to_numpy(dtype=int)[order]
            ss=s[order];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
            precision=np.cumsum(y)[ends]/(ends+1)
            threshold=np.repeat(precision,np.diff(np.r_[-1,ends]))
            ordered=g.iloc[order].copy()
            ordered['ap_contribution']=y*threshold/y.sum()
            cc=ordered.loc[ordered.target.eq(1)].groupby(['pitcher_id','dynamic_next_event_id']).agg(
                positive_rows=('target','size'),ap_contribution=('ap_contribution','sum')).reset_index()
            cc['fraction_total_ap']=cc.ap_contribution/ap
            contributions.append(cc.assign(partition=part,interval=interval,model=name))
            for pid in g.loc[g.target.eq(1),'pitcher_id'].unique():
                retained=g.pitcher_id.ne(pid)
                after=rank_metrics(g.loc[retained,'target'],s[retained])['average_precision']
                influence.append(dict(partition=part,interval=interval,model=name,pitcher_id=pid,
                    original_ap=ap,ap_without_pitcher=after,ap_drop=ap-after))
    table(pd.DataFrame(rows),'shared_event_sensitivity_'+part)
    table(pd.DataFrame(composition),'role_composition_'+part)
    table(pd.concat(contributions,ignore_index=True),'event_ap_contributions_'+part)
    table(pd.DataFrame(influence),'pitcher_influence_'+part)
    checkpoint(stage,[OUT/'tables'/f'{n}_{part}.csv' for n in [
        'shared_event_sensitivity','role_composition','event_ap_contributions','pitcher_influence']])
