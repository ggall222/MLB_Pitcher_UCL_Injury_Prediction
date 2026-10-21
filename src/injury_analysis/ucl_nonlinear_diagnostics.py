"""Influence and subgroup diagnostics; no changes to the frozen model choices."""
import json
import numpy as np
import pandas as pd
from .ucl_nonlinear_common import OUT,CFG,inputs,table,read_table,checkpoint,valid_checkpoint
from .ucl_nonlinear_models import load_model,scores
from .ucl_dynamic_data import event_cohort
from .ucl_dynamic_evaluation import metrics,warning
from src.evaluation import alert_mask,rank_metrics

def run_diagnostics():
    if valid_checkpoint('diagnostics'):return
    f,e,s=inputs();protocol=json.loads((OUT/'frozen_protocol.json').read_text())
    ids=list(dict.fromkeys(list(protocol['warning_leaders'].values())+list(protocol['ranking_leaders'].values())))
    models={mid:load_model(mid) for mid in ids};tr=f.loc[f.partition.eq('train')&f.target.notna()]
    influence=[];contribution=[];subgroups=[]
    for part in ['validation','test']:
        g=f.loc[f.partition.eq(part)&f.target.notna()].reset_index(drop=True);ev=event_cohort(e,part)
        injured=g.loc[g.target.eq(1),'pitcher_id'].unique()
        for mid,obj in models.items():
            score=scores(obj,g);base=rank_metrics(g.target,score)['average_precision']
            for pid in injured:
                keep=g.pitcher_id.ne(pid);remaining=g.loc[keep];ap=rank_metrics(remaining.target,score[keep])['average_precision']
                influence.append(dict(partition=part,model_id=mid,excluded_pitcher_id=pid,removed_positive_rows=int(g.loc[~keep,'target'].sum()),
                    excluded_positive_events=g.loc[~keep&g.target.eq(1),'dynamic_next_event_id'].nunique(),original_ap=base,ap_without_pitcher=ap,ap_drop_without_pitcher=base-ap))
            ordered=g.assign(score=score).sort_values('score',ascending=False,kind='stable');y=ordered.target.to_numpy(dtype=int);score_ordered=ordered.score.to_numpy()
            # Tie-aware AP contribution assigns the threshold precision to each positive at that score.
            ends=np.r_[np.flatnonzero(score_ordered[:-1]!=score_ordered[1:]),len(ordered)-1]
            precision=np.cumsum(y)[ends]/(ends+1);at_end=np.repeat(precision,np.diff(np.r_[-1,ends]))
            ordered['ap_contribution']=y*at_end/y.sum()
            c=ordered.loc[ordered.target.eq(1)].groupby(['pitcher_id','dynamic_next_event_id']).agg(positive_rows=('target','size'),ap_contribution=('ap_contribution','sum'),maximum_score=('score','max')).reset_index()
            c['fraction_total_ap']=c.ap_contribution/base;c['model_id']=mid;c['partition']=part;contribution.append(c)
            subsets={'seen in training':g.pitcher_id.isin(tr.pitcher_id),'unseen in training':~g.pitcher_id.isin(tr.pitcher_id),
                'starter':g.is_starter,'reliever':~g.is_starter}
            for name,mask in subsets.items():
                h=g.loc[mask]
                if h.target.nunique()<2:continue
                hev=ev.loc[ev.injury_event_id.isin(h.loc[h.dynamic_next_target&h.dynamic_lead_days.between(7,30),'dynamic_next_event_id'])]
                w,_=warning(h,score[mask],hev)
                subgroups.append(dict(partition=part,model_id=mid,subset=name,event_denominator='subgroup observable events',**metrics(h,score[mask]),
                    **{k:w[k] for k in ['qualifying_events','alerted_events','injury_event_recall']}))
    table(pd.DataFrame(influence),'leave_one_injured_pitcher_out');table(pd.concat(contribution,ignore_index=True),'event_ap_contributions');table(pd.DataFrame(subgroups),'subgroup_diagnostics')
    search=read_table('development_search')
    main=search.loc[search.partition.eq('validation')].copy()
    main['model_form']=np.where(main.algorithm.eq('Linear'),'linear additive',np.where(main.algorithm.eq('Spline'),'nonlinear additive spline',np.where(main.setting.eq(2),'nonlinear additive boosted stumps','nonlinear trees with interactions')))
    table(main,'additivity_comparison')
    overfit=search.pivot(index='model_id',columns='partition',values='average_precision').reset_index();overfit['training_minus_validation_ap']=overfit.train-overfit.validation
    table(overfit,'training_validation_gap')
    checkpoint('diagnostics',[OUT/'tables'/p for p in ['leave_one_injured_pitcher_out.csv','event_ap_contributions.csv','subgroup_diagnostics.csv','additivity_comparison.csv','training_validation_gap.csv']])
