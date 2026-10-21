"""Small synthetic cases for the new analytical time boundaries and preprocessing."""
import numpy as np
import pandas as pd
from src.injury_analysis.cohort import classify_events, align_risk_rows, label_frame
from src.injury_analysis.trajectories import window
from src.injury_analysis.prediction import fit_scores


def event(**updates):
    base = dict(injury_event_id='e1',player_id=1,initial_injury_region='Shoulder',
        initial_injury_diagnosis='Strain',initial_laterality='Left',throws='L',
        injury_event_date='2024-06-10',public_known_date='2024-06-12',transaction_date='2024-06-12',
        activation_date='2024-06-25',episode_boundary_date='2024-06-25',
        surgery_date=None,official_surgery_date=None,observation_end_date='2024-12-31',
        placement_description='Left rotator cuff strain.',qa_flags='[]',
        left_censored=False,timing_uncertain=False,overlapping_episode=False,
        multiple_injury_regions=False,inferred_continuation_count=0,
        transaction_sources_complete=True,qa_review_required=False,activation_match_valid=True)
    base.update(updates)
    return base


def appearances(dates):
    return pd.DataFrame(dict(pitcher_id=1,game_id=np.arange(len(dates)),game_date=pd.to_datetime(dates),
        quality_eligible=True,timing_uncertain=False,prior_appearance_count=10))


def test_initial_anatomy_used_without_future_ucl_upgrade():
    e=classify_events(pd.DataFrame([event(initial_injury_region='Forearm',injury_region='Elbow / UCL',surgery_date='2024-07-15')]))
    assert e.iloc[0].analysis_group=='Forearm'
    assert e.iloc[0].target_eligible


def test_postoperative_and_unknown_cases_never_positive_targets():
    e=classify_events(pd.DataFrame([event(initial_injury_diagnosis='Surgery / Postoperative Recovery'),
        event(injury_event_id='e2',initial_injury_region='Unknown / Unspecified')]))
    assert not e.target_eligible.any()
    assert e.blocking_date.notna().all()


def test_censoring_does_not_select_only_fast_recoveries():
    e=classify_events(pd.DataFrame([event(activation_date=None,qa_review_required=True,qa_flags='["unresolved_duration"]')]))
    assert e.target_eligible.all()
    assert not e.strict_qa_eligible.any()


def test_throwing_side_is_explicit_not_inferred_from_arm_location():
    e=classify_events(pd.DataFrame([event(initial_laterality='Unknown'),event(injury_event_id='e2',initial_laterality='Right')]))
    assert pd.isna(e.iloc[0].throwing_arm_injury)
    assert not e.iloc[1].throwing_arm_injury


def test_positive_only_before_anchor_and_active_outings_excluded():
    a=appearances(['2024-06-01','2024-06-10','2024-06-11','2024-06-25','2024-07-01','2024-07-15'])
    e=classify_events(pd.DataFrame([event()]))
    f=align_risk_rows(a,e)
    assert f.iloc[0].any_injury_14d==1
    assert f.iloc[1:3].any_injury_14d.isna().all()
    assert f.iloc[1:3].active_episode.all()
    assert f.iloc[0].prior_reported_il_count==0
    assert f.iloc[3].prior_reported_il_count==1


def test_unknown_event_censors_negatives_instead_of_becoming_healthy():
    a=appearances(['2024-06-01','2024-06-20','2024-07-01','2024-07-20','2024-08-05'])
    e=classify_events(pd.DataFrame([event(initial_injury_region='Unknown / Unspecified')]))
    f=align_risk_rows(a,e)
    assert pd.isna(f.iloc[0].any_injury_30d)


def test_negative_requires_future_observed_followup():
    e=classify_events(pd.DataFrame([event(player_id=2)]))
    a=appearances(['2024-06-01','2024-07-01','2024-07-08'])
    f=align_risk_rows(a,e)
    assert f.iloc[0].any_injury_30d==0
    assert pd.isna(f.iloc[-1].any_injury_30d)
    assert pd.isna(f.iloc[0].any_injury_60d)


def test_competing_injury_excluded_from_category_versus_healthy():
    e=classify_events(pd.DataFrame([event()]))
    f=align_risk_rows(appearances(['2024-06-01','2024-07-01','2024-08-05']),e)
    shoulder=label_frame(f,'Shoulder',30)
    elbow=label_frame(f,'Elbow / UCL',30)
    assert shoulder.target.sum()==1
    assert elbow.target.sum()==0
    assert not elbow.game_date.eq('2024-06-01').any()


def test_partition_embargo_removes_cross_year_event_rows():
    e=classify_events(pd.DataFrame([event(injury_event_date='2022-01-10',public_known_date='2022-01-10')]))
    f=align_risk_rows(appearances(['2021-12-20','2022-01-05','2022-02-01']),e)
    assert f.iloc[0].partition=='purged'
    assert f.iloc[1].partition=='validation'


def test_trajectory_excludes_current_and_future_outings():
    a=appearances(['2024-06-01','2024-06-09','2024-06-10','2024-06-11'])
    a['analysis_eligible']=True
    a['is_starter']=True
    a['velo_z']=[1.,2.,999.,999.]
    f=window(a,pd.Timestamp('2024-06-10'),'e','Shoulder',1,'Injury',['velo_z'])
    assert f.velo_z.tolist()==[1.,2.]
    assert f.appearance_before.tolist()==[2,1]


def test_preprocessing_fits_train_values_only():
    train=pd.DataFrame({'x':[1.,2.,np.nan,4.,5.,6.],'target':[0,0,0,1,1,1]})
    held=pd.DataFrame({'x':[1000.,np.nan],'target':[0,1]})
    pipe,(scores,)=fit_scores(train,[held],['x'])
    assert pipe.named_steps['simpleimputer'].statistics_[0]==4.
    assert pipe.named_steps['standardscaler'].mean_[0]<10
    assert len(scores)==2 and np.isfinite(scores).all()
