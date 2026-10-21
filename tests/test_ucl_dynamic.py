"""Temporal and episode invariants for the additive dynamic UCL stage."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.ucl_dynamic_endpoint import validate_procedure_link,classify_evidence
from src.injury_analysis.ucl_dynamic_data import departure_trajectories,risk_frame
from src.injury_analysis.ucl_dynamic_evaluation import warning,selection
from src.injury_analysis.ucl_dynamic_models import DynamicEncoder

def appearances():
    return pd.DataFrame(dict(pitcher_id=[1]*7,game_id=range(7),game_date=pd.to_datetime(['2023-04-01','2023-04-05','2023-04-09','2023-04-13','2023-04-13','2023-04-20','2023-06-01']),
        quality_eligible=True,timing_uncertain=False,analysis_eligible=True,season=2023,
        velo_z=[0.,1,2,3,30,4,5],release_z_sd_z=1.,extension_z=0.,spin_z=0.))

def test_trajectory_excludes_future_and_same_day_other_outing():
    a=appearances();x,_=departure_trajectories(a)
    assert x.loc[3,'dyn_velo_z_slope4']==pytest.approx(1.)
    assert x.loc[3,'dyn_velo_z_absmean4']==pytest.approx(1.5)
    assert x.loc[3,'dyn_velo_z_persist4']==pytest.approx(.75)
    a.loc[4:,'velo_z']=999
    y,_=departure_trajectories(a)
    assert x.loc[3,'dyn_velo_z_slope4']==y.loc[3,'dyn_velo_z_slope4']
    assert pd.isna(x.loc[6,'dyn_velo_z_slope4'])
    assert (x.dynamic_history_max_date.dropna()<x.loc[x.dynamic_history_max_date.notna(),'game_date']).all()

def test_trajectory_current_outing_is_intentionally_included():
    a=appearances();x,_=departure_trajectories(a);a.loc[3,'velo_z']+=1;y,_=departure_trajectories(a)
    assert x.loc[3,'dyn_velo_z_slope4']!=y.loc[3,'dyn_velo_z_slope4']

def test_procedure_link_has_bounded_episode_and_return_rules():
    e=SimpleNamespace(anchor_date=pd.Timestamp('2023-04-01'),player_id=1,activation_match_valid=False,
        activation_date=pd.NaT,episode_boundary_date=pd.NaT,public_known_date=pd.Timestamp('2023-04-02'),initial_injury_region='Forearm')
    p=SimpleNamespace(player_id=1,surgery_date_lower='2023-04-15',surgery_date_upper='2023-04-15',surgery_source_level='MLB')
    assert validate_procedure_link(e,p,pd.DataFrame())[0]
    assert not validate_procedure_link(e,p,pd.DataFrame({'game_date':pd.to_datetime(['2023-04-10'])}))[0]
    p.surgery_date_upper='2024-04-15';assert not validate_procedure_link(e,p,pd.DataFrame())[0]

def test_procedure_only_text_does_not_confirm_new_episode():
    e=SimpleNamespace(explicit_ucl=True,injury_region='Elbow / UCL')
    assert classify_evidence(e,False,True,False)=='probable_supplemental_documentation'
    assert classify_evidence(e,True,True,False)=='confirmed_procedure_supported'

def event_data():
    return pd.DataFrame(dict(player_id=[1,1],injury_event_id=['shoulder','ucl'],
        observation_end_date=pd.to_datetime(['2023-12-31']*2),public_known_date=pd.to_datetime(['2023-04-15','2023-05-01']),
        blocking_date=pd.to_datetime(['2023-04-15','2023-05-01']),endpoint_primary=[False,True],
        target_eligible=[True,True],uncertain_ucl_competitor=[False,True]))

def test_first_competing_event_prevents_labeling_later_ucl():
    f=risk_frame(appearances(),event_data())
    assert f.loc[0,'target']==0
    assert f.loc[0,'label_status']=='non_ucl_competing_first_event'
    assert f.loc[5,'target']==1
    censored=risk_frame(appearances(),event_data(),competing='censor')
    assert pd.isna(censored.loc[0,'target'])

def test_ambiguous_elbow_is_not_a_negative():
    ev=event_data();ev.loc[0,'uncertain_ucl_competitor']=True
    f=risk_frame(appearances(),ev)
    assert pd.isna(f.loc[0,'target'])

def test_warning_never_jumps_over_competing_event():
    f=risk_frame(appearances(),event_data());f=f.loc[f.target.notna()]
    ev=pd.DataFrame(dict(injury_event_id=['ucl'],pitcher_id=[1],label_reference_date=pd.to_datetime(['2023-05-01'])))
    score=np.where(f.game_id.eq(0),1.,0.)
    m,e=warning(f,score,ev,threshold=.5)
    assert m['alerted_events']==0
    assert e.opportunity_count.iloc[0]==1
    assert m['false_alert_fraction']==1

def test_no_observed_followup_is_not_healthy():
    f=risk_frame(appearances(),event_data())
    assert pd.isna(f.iloc[-1].target)

def test_encoder_is_fixed_when_future_values_change():
    tr=pd.DataFrame(dict(a=[1.,2,3,np.nan,4],b=[2.,3,4,5,6],game_date=pd.date_range('2021-01-01',periods=5)))
    enc=DynamicEncoder(dict(features=['a','b'],interactions=[('a','b')])).fit(tr)
    before=enc.transform(tr);bounds=enc.base.bounds.copy()
    enc.transform(tr.assign(a=10000,b=-9999))
    np.testing.assert_allclose(before,enc.transform(tr));pd.testing.assert_frame_equal(bounds,enc.base.bounds)
    assert 'a × b' in enc.names

def test_selection_rejects_test_metrics():
    with pytest.raises(ValueError,match='validation'):
        selection(pd.DataFrame({'partition':['test']}),pd.DataFrame({'partition':['validation']}))
