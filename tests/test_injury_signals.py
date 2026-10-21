"""Temporal, paired-design and numerical checks for the signal-analysis extension."""
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.family_decomposition import weighted_ap,random_detection_probability,paired_bootstrap_ap
from src.injury_analysis.trajectory_shapes import paired_cells,simultaneous_summary,shape_metrics,sustained_support,sensitivity_filter
from src.evaluation import rank_metrics


def test_weighted_ap_respects_score_ties_and_pitcher_multiplicity():
    y=np.array([0,1,1,0,1]);score=np.array([.2,.8,.8,.2,.1]);w=np.array([2,1,3,2,1])
    expected=rank_metrics(np.repeat(y,w),np.repeat(score,w))['average_precision']
    assert weighted_ap(y,score,w)==pytest.approx(expected)


def test_shared_bootstrap_produces_zero_delta_for_identical_models():
    f=pd.DataFrame({'pitcher_id':[1,1,2,2,3,3],'target':[1,0,0,1,0,1]})
    s=np.array([.6,.4,.2,.8,.1,.9]);bs,_=paired_bootstrap_ap(f,{'a':s,'b':s.copy()})
    np.testing.assert_array_equal(bs['a'],bs['b'])


@pytest.mark.parametrize('n,k,m',[(5,2,2),(6,3,0),(6,0,4),(5,5,1),(7,2,5)])
def test_event_random_reference_matches_enumeration(n,k,m):
    sets=list(combinations(range(n),k))
    expected=np.mean([bool(set(s)&set(range(m))) for s in sets])
    assert random_detection_probability(n,k,m)==pytest.approx(expected)


def test_shape_distinguishes_gradual_decline_from_final_step():
    gradual=shape_metrics([0,-1,-2,-3,-4]);step=shape_metrics([0,0,0,0,-4])
    assert gradual['slope_per_appearance']==pytest.approx(-1)
    assert gradual['late_minus_early_slope']==pytest.approx(0)
    assert step['final_step_vs_previous_two']==-4
    assert step['late_minus_early_slope']<0


def test_balanced_trajectory_drops_partial_events_instead_of_changing_composition():
    rows=[]
    for event,pid,points,value in [('complete',1,[0,1,2,3],2.),('late_only',2,[3],100.)]:
        for point in points:rows.append(dict(event_id=event,cluster_pitcher_id=pid,point=point,
            Injury=value,Control=1.,difference=value-1))
    result,ids=simultaneous_summary(pd.DataFrame(rows),[0,1,2,3])
    assert ids=={'complete'}
    assert result[result.kind.eq('difference')]['mean'].eq(1).all()


def test_paired_missingness_does_not_compare_different_case_and_control_events():
    f=pd.DataFrame(dict(event_id=['a','a','b','b'],analysis_group='Shoulder',cluster_pitcher_id=[1,1,2,2],
        kind=['Injury','Control','Injury','Control'],lead_days=3,appearance_before=1,
        feature=[2.,1.,100.,np.nan]))
    cells=paired_cells(f,'feature')
    assert cells.event_id.tolist()==['a'] and cells.difference.tolist()==[1.]


def test_timing_requires_adjacent_same_direction_support_and_sufficient_events():
    s=pd.DataFrame(dict(point=[0,1,2,3],events=40,simultaneous_low=[-1,-1,-1,-1],simultaneous_high=[-.1,.2,.2,-.1]))
    assert sustained_support(s)==(None,0)
    s.loc[2,'simultaneous_high']=-.1
    assert sustained_support(s)==(2,-1)
    s['events']=10
    assert sustained_support(s)==(None,0)


def test_excluding_final_days_never_relabels_or_reindexes_remaining_outings():
    f=pd.DataFrame(dict(lead_days=[1,4,8,15],appearance_before=[1,2,3,4]))
    filtered=sensitivity_filter(f,'Exclude final 7 days','velo_z')
    assert filtered.lead_days.tolist()==[8,15]
    assert filtered.appearance_before.tolist()==[3,4]


def test_strong_baseline_uses_measured_count_not_total_pitching_experience():
    f=pd.DataFrame({'avg_fb_velo_baseline_count':[3,17],'prior_appearance_count':[100,100]})
    kept=sensitivity_filter(f,'Stronger baseline','velo_z')
    assert kept.index.tolist()==[1]
