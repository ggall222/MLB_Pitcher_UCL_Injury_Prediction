"""Disjoint outcomes, dependence-aware uncertainty and family reliance invariants."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from src.injury_analysis.ucl_temporal_data import interval_target,definitions
from src.injury_analysis.ucl_temporal_analysis import bootstrap_ap
from src.injury_analysis.ucl_temporal_models import fit,permutation_scores
from src.injury_analysis.ucl_temporal_trajectories import paired_cells,summarize_cells
from src.injury_analysis.consolidation_models import ConsolidationEncoder
from src.injury_analysis.family_decomposition import weighted_ap
from sklearn.linear_model import LogisticRegression

def outcomes():
    lead=[1,7,8,14,15,30,31,60,61,5,10,20]
    return pd.DataFrame(dict(horizon=60,dynamic_next_target=[True]*9+[False,True,False],
        dynamic_lead_days=lead,target=pd.array([1]*8+[0,0,None,None],dtype='Int8')))

def test_disjoint_endpoints_and_unknowns_preserved():
    f=outcomes()
    parts=[interval_target(f,*h).target for h in [(1,7),(8,14),(15,30),(31,60)]]
    s=pd.concat(parts,axis=1)
    np.testing.assert_equal(s.iloc[:8].astype(int).sum(axis=1),np.ones(8))
    assert s.iloc[8:10].eq(0).all().all()
    assert s.iloc[10:].isna().all().all()
    assert f.target.iloc[0]==1  # never mutate source

def test_intervening_injury_is_not_skipped_and_wrong_interval_is_negative():
    f=outcomes()
    g=interval_target(f,31,60)
    assert g.target.iloc[0]==0 and g.target.iloc[6]==1
    assert g.target.iloc[9]==0  # known first event is non-UCL
    with pytest.raises(ValueError):interval_target(f,30,61)

def test_family_union_unchanged_and_disjoint_inventory_groups():
    from src.injury_analysis.ucl_dynamic_data import frozen_representations
    inv,groups,models=definitions(frozen_representations())
    assert len(inv)==113 and len(groups)==11
    assert len([c for v in groups.values() for c in v])==len(set(models['Full union']))
    for name,fields in groups.items():
        assert not set(fields)&set(models['Without | '+name])
        assert set(fields)|set(models['Without | '+name])==set(models['Full union'])

def test_weighted_ap_ties_match_replication():
    y=np.array([1,0,1,0,0]);s=np.array([.5,.5,.2,.3,.1]);w=np.array([3,2,0,1,2])
    assert weighted_ap(y,s,w)==pytest.approx(average_precision_score(np.repeat(y,w),np.repeat(s,w)))

def test_whole_pitcher_bootstrap_pairs_models_and_is_deterministic():
    f=pd.DataFrame(dict(pitcher_id=np.repeat(np.arange(10),3),target=np.tile([0,0,1],10)))
    s=np.linspace(0,1,len(f))
    a,p=bootstrap_ap(f,{'first':s,'same':s})
    b,q=bootstrap_ap(f,{'first':s})
    np.testing.assert_allclose(a['first'],a['same'])
    np.testing.assert_allclose(a['first'],b['first'])
    # Each sampled pitcher has exactly one positive of three rows.
    np.testing.assert_allclose(p,1/3)
    np.testing.assert_allclose(p,q)

def test_temporal_fit_rejects_nontraining_rows():
    with pytest.raises(ValueError,match='chronological training'):
        fit(pd.DataFrame({'partition':['validation']}),['x'],'1-7','invalid')

def test_joint_family_permutation_preserves_redundant_pair_and_missing_flags():
    rng=np.random.default_rng(22);n=180
    x=rng.normal(size=n)
    f=pd.DataFrame(dict(x=x,y=2*x,season=np.repeat([2019,2020],n//2),
                        game_date=pd.date_range('2019-01-01',periods=n)))
    f.loc[::9,['x','y']]=np.nan
    enc=ConsolidationEncoder(['x','y']).fit(f)
    # Exact cancellation of two standardized copies; shuffling separately would
    # destroy this identity and create spurious grouped reliance.
    model=LogisticRegression().fit(enc.transform(f),rng.integers(0,2,n))
    model.coef_[:]=0;model.coef_[0,:2]=[1,-1]
    obj=dict(encoder=enc,model=model)
    expected=model.predict_proba(enc.transform(f))[:,1]
    for s in permutation_scores(obj,f,{'group':['x','y']}).values():
        np.testing.assert_allclose(s,expected)

def test_matched_bins_require_both_case_control_and_keep_event_weight():
    f=pd.DataFrame([
        dict(event_id='a',dependency_cluster=1,event_partition='validation',kind=k,
             lead_days=d,outing=-i,family=v)
        for k,v in [('Injury',2.),('Control',1.)] for i,d in enumerate([3,10,20,40],1)])
    c=paired_cells(f,'family','calendar')
    assert len(c)==4 and c.difference.eq(1).all()
    missing=f.loc[~(f.kind.eq('Control')&f.lead_days.eq(40))]
    assert len(paired_cells(missing,'family','calendar'))==3
