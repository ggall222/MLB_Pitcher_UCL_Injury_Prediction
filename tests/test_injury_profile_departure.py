"""Behavioral safeguards for the profile/departure experiment."""
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.profile_departure import (
    PROFILE,DEPARTURE,probability_metrics,reliability_bins,event_detection,event_difference)
from src.injury_analysis.profile_robustness import sensitivity_frames,shared_availability_fit
from src.injury_analysis.profile_scores import score_contributions
from src.injury_analysis.reduction import fit_model


def sample(n=240,seed=8):
    rng=np.random.default_rng(seed)
    f=pd.DataFrame(rng.normal(size=(n,len(PROFILE+DEPARTURE))),columns=PROFILE+DEPARTURE)
    f['target']=(f.velo_z+.4*f.avg_fb_velo_baseline_mean+rng.normal(size=n)>0).astype(int)
    f.loc[::13,'velo_z']=np.nan;f.loc[::17,'spin_avg_baseline_mean']=np.nan
    return f


def test_probability_loss_and_bias_with_invalid_outcomes_rejected():
    result=probability_metrics([0,1],[.2,.8])
    assert result['brier_score']==pytest.approx(.04)
    assert result['calibration_bias']==pytest.approx(0)
    for y,p in [([0,None],[.1,.2]),([0,1],[.1,1.2]),([0,1],[.2]),([0,1],[.2,np.nan])]:
        with pytest.raises(ValueError):probability_metrics(y,p)


def test_reliability_bins_cover_every_row_including_zero_and_one():
    f=pd.DataFrame({'pitcher_id':[1,1,2,3],'target':[0,0,1,1]})
    b=reliability_bins(f,np.array([0,.2,.8,1]),[0,.5,1])
    assert b.rows.sum()==4 and b.positive_rows.sum()==2
    assert b.observed_rate.tolist()==[0,1]


def test_profile_departure_terms_reconstruct_missing_aware_logit():
    f=sample();enc,model,_=fit_model(f,[],PROFILE+DEPARTURE)
    terms=score_contributions(enc,model,f)
    np.testing.assert_allclose(terms.logit,model.decision_function(enc.transform(f)),atol=1e-10)
    g=f.copy();g['velo_z']=20.
    updated=score_contributions(enc,model,g)
    np.testing.assert_array_equal(terms.profile_score,updated.profile_score)
    assert not np.allclose(terms.departure_score,updated.departure_score)


def test_representation_blocks_are_disjoint_and_no_outcome_fields():
    assert len(PROFILE)==13 and len(DEPARTURE)==8
    assert not set(PROFILE)&set(DEPARTURE)
    assert not any(x in c for c in PROFILE+DEPARTURE for x in ['next_','activation','surgery','target','event_date'])


def test_shared_availability_fit_cannot_learn_from_evaluation_outcomes():
    train=sample();valid=sample(70,2)
    score=shared_availability_fit(train,[valid],PROFILE)[0]
    changed=valid.copy();changed.target=1-changed.target
    again=shared_availability_fit(train,[changed],PROFILE)[0]
    np.testing.assert_allclose(score,again,atol=1e-12)


def test_current_relative_features_do_not_change_profile_prediction():
    train=sample();v=sample(70,2)
    enc,model,(score,)=fit_model(train,[v],PROFILE)
    v[DEPARTURE]=99999.
    np.testing.assert_array_equal(score,model.predict_proba(enc.transform(v))[:,1])


def test_excluding_near_event_rows_never_relabels_them_healthy():
    f=sample(30);f['partition']=np.repeat(['train','validation','test'],10)
    f['next_event_lead_days']=np.tile(np.arange(1,11),3)
    result=sensitivity_frames(f,'Exclude final six days')
    kept=pd.concat(result.values())
    assert kept.target.equals(f.loc[kept.index,'target'])
    assert not ((kept.target==1)&(kept.next_event_lead_days<7)).any()
    assert set(f.index[f.target.eq(0)])<=set(kept.index)


def test_complete_case_uses_same_joint_support_for_both_models():
    f=sample(30);f['partition']='train'
    result=sensitivity_frames(f,'Complete P+D measurements')['train']
    assert set(result.index)==set(f.index[f[PROFILE+DEPARTURE].notna().all(axis=1)])


def test_event_warning_excludes_final_six_days_and_keeps_unobservable_events():
    f=pd.DataFrame({'pitcher_id':[1,1,2,2],'game_id':[1,2,3,4],
        'game_date':pd.to_datetime(['2024-05-02','2024-05-15','2024-05-20','2024-05-25'])})
    e=pd.DataFrame({'injury_event_id':['a','b','unobservable'],'pitcher_id':[1,2,3],
        'label_reference_date':pd.to_datetime(['2024-06-01']*3)})
    r=event_detection(f,np.array([.5,.2,.8,.9]),e,rate=.5).set_index('injury_event_id')
    assert r.detected.to_dict()=={'a':0,'b':1,'unobservable':0}
    assert r.loc['b','warning_lead_days']==12  # Six-day high score cannot create a warning.
    assert pd.isna(r.loc['unobservable','warning_lead_days'])


def test_warning_lead_comparison_uses_jointly_detected_events():
    a=pd.DataFrame({'injury_event_id':['a','b','c'],'pitcher_id':[1,2,3],
                    'detected':[1,1,0],'warning_lead_days':[20,7,np.nan]})
    b=a.copy();b.detected=[1,0,1];b.warning_lead_days=[15,np.nan,29]
    result=event_difference(a,b)
    assert result['mean']==0
    assert result['gained_events']==1 and result['lost_events']==1
    assert result['common_lead_mean']==5 and result['common_lead_events']==1

