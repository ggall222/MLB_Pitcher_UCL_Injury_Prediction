"""Locked replication: alert episodes, unknowns, exact AP and event dependence."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src.injury_analysis.ucl_replication_operations import alert_episodes,episode_metrics,calibration_diagnostic
from src.injury_analysis.ucl_replication_analysis import event_contributions,leave_injured_pitcher_out
from src.injury_analysis.ucl_temporal_data import interval_target

def example():
    return pd.DataFrame(dict(pitcher_id=1,game_id=range(8),game_date=pd.to_datetime(['2025-12-30','2026-01-01','2026-01-03','2026-01-05','2026-01-07','2026-01-09','2026-01-11','2026-03-20']),
        season=[2025,2026,2026,2026,2026,2026,2026,2026],analysis_eligible=[True,True,True,True,True,False,True,True]))

def test_episode_breaks_use_season_nonalert_ineligible_and_gap():
    g=alert_episodes(example(),[.9,.9,.9,.1,.9,.9,.9,.9],.5)
    assert g.episode_id.iloc[1]==g.episode_id.iloc[2]
    assert g.episode_id.dropna().nunique()==5
    assert pd.isna(g.episode_id.iloc[3]) and pd.isna(g.episode_id.iloc[5])

def test_episode_grouping_is_independent_of_outcomes():
    x=example();a=alert_episodes(x.assign(target=0),np.ones(8),.5)
    b=alert_episodes(x.assign(target=1),np.ones(8),.5)
    pd.testing.assert_series_equal(a.episode_id,b.episode_id)

def test_censored_alerts_are_unresolved_not_false():
    g=example().iloc[:3].copy();g['target']=pd.array([1,0,None],dtype='Int8')
    g['dynamic_next_event_id']=['e',None,None];g['dynamic_lead_days']=[10,np.nan,np.nan]
    g=alert_episodes(g,[.9,.9,.9],.5)
    ev=pd.DataFrame(dict(injury_event_id=['e'],pitcher_id=[1],label_reference_date=pd.to_datetime(['2026-01-09'])))
    m,ep,_=episode_metrics(g,ev,8,14)
    assert m['unknown_outcome_alerts']==1 and m['nonactionable_known_alerts']==1
    assert m['non_actionable_alert_fraction']==.5 and m['unresolved_episodes']==1

def test_ap_event_decomposition_handles_ties_and_sums_exactly():
    g=pd.DataFrame(dict(target=[1,0,1,1,0],pitcher_id=[1,2,3,3,4],dynamic_next_event_id=['a',None,'b','b',None]))
    s=np.array([.8,.8,.8,.2,.1]);cc=event_contributions(g,s)
    assert np.isclose(cc.ap_contribution.sum(),average_precision_score(g.target,s))
    assert np.isclose(cc.fraction_total_ap.sum(),1)

def test_leave_injured_pitcher_out_removes_their_negative_rows_too():
    g=pd.DataFrame(dict(target=[1,0,1,0,0],pitcher_id=[1,1,2,3,4],dynamic_next_event_id=['a',None,'b',None,None]))
    result=leave_injured_pitcher_out(g,[.9,.8,.7,.6,.5]).set_index('pitcher_id')
    assert result.loc[1,'rows']==3 and result.loc[1,'positive_rows']==1

def test_sparse_calibration_is_not_fitted():
    g=pd.DataFrame(dict(target=[1,0,0],pitcher_id=[1,2,3],dynamic_next_event_id=['a',None,None]))
    r=calibration_diagnostic(g,[.5,.2,.1]);assert r['status']=='insufficient_event_support' and np.isnan(r['slope'])

def test_original_interval_outcome_keeps_censoring_and_other_intervals():
    g=pd.DataFrame(dict(target=pd.array([1,1,None,0],dtype='Int8'),dynamic_next_target=[True,True,True,False],dynamic_lead_days=[3,12,4,20],horizon=60))
    r=interval_target(g,1,7);assert r.target.iloc[0]==1 and r.target.iloc[1]==0 and pd.isna(r.target.iloc[2])
