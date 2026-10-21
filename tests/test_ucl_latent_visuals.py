"""Selection, balanced support and dependence safeguards for frozen UCL plots."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis import ucl_latent_visuals as u


def test_emphasized_axes_use_training_standardized_weights_and_frozen_count():
    basis=SimpleNamespace(training_score_sd={'PLS':np.array([2.,.1,1.])},keep=[0,1])
    fitted={'basis':basis,'selected':{'PLS':2},'models':{
        'PLS selected':SimpleNamespace(coef_=np.array([[-.3,1.,999.,999.]]))}}
    selected=u.select_components(fitted,'Departure')
    assert selected.component.tolist()==['D1','D2']
    assert selected.orientation.tolist()==[-1,1]
    assert selected.training_rank.tolist()==[1,2]
    assert selected.individual_heatmap.tolist()==[True,False]
    assert selected.design_columns.tolist()==[4,4]
    np.testing.assert_allclose(selected.training_absolute_standardized_logit_weight,[.6,.1])


def test_invalid_training_scale_fails_instead_of_rescaling_from_heldout_data():
    fitted={'basis':SimpleNamespace(training_score_sd={'PLS':np.array([0.])},keep=[]),
            'selected':{'PLS':1},'models':{'PLS selected':SimpleNamespace(coef_=np.array([[.1]]))}}
    with pytest.raises(ValueError,match='training component scales'):u.select_components(fitted,'Departure')


def test_cross_role_pitchers_and_recurrent_cases_stay_in_one_cluster():
    f=pd.DataFrame({'event_id':['a','a','b','b','c','c','d','d'],
                    'pitcher_id':[1,2,2,3,4,5,1,6]})
    clusters=u.paired_dependency_clusters(f)
    assert clusters.a==clusters.b==clusters.d
    assert clusters.c!=clusters.a
    pd.testing.assert_series_equal(clusters,u.paired_dependency_clusters(f.sample(frac=1,random_state=7)))
    weights=u.bootstrap_weights(clusters.to_numpy())
    np.testing.assert_array_equal(weights[:,0],weights[:,1])
    np.testing.assert_array_equal(weights[:,0],weights[:,3])
    np.testing.assert_array_equal(weights,u.bootstrap_weights(clusters.to_numpy()))


def alignment():
    rows=[]
    for event in range(5):
        for kind,pid in [('Injury',event+1),('Control',event+101)]:
            for outing in range(1,6):
                rows.append(dict(event_id=str(event),analysis_group='Elbow / UCL',cluster_pitcher_id=event+1,
                    pitcher_id=pid,kind=kind,lead_days=outing*2,appearance_before=outing,
                    D1=float(event+outing+(kind=='Injury')),C1=float(event-outing)))
    return pd.DataFrame(rows)


def test_one_missing_cell_removes_event_from_every_axis_and_position():
    f=alignment()
    f.loc[f.event_id.eq('4')&f.kind.eq('Control')&f.appearance_before.eq(3),'C1']=np.nan
    summary,cells=u.balanced_summary(f,['D1','C1'],'appearance')
    assert set(cells.event_id)=={'0','1','2','3'}
    assert summary.events.eq(4).all()
    assert cells.groupby(['event_id','component']).point.nunique().eq(5).all()
    np.testing.assert_allclose(summary.query("kind=='difference' and component=='D1'")['mean'],1.)
    np.testing.assert_allclose(summary.query("kind=='difference' and component=='C1'")['mean'],0.)


def test_calendar_cells_weight_events_equally_despite_different_outing_counts():
    rows=[]
    for e in range(3):
        for kind,pid in [('Injury',e),('Control',100+e)]:
            for day in [1,8,15,31]:
                for _ in range(5 if e==0 else 1):
                    rows.append(dict(event_id=str(e),analysis_group='Elbow / UCL',cluster_pitcher_id=e,
                        pitcher_id=pid,kind=kind,lead_days=day,appearance_before=1,
                        D1=float(e if kind=='Injury' else 0)))
    s,c=u.balanced_summary(pd.DataFrame(rows),['D1'],'calendar')
    np.testing.assert_allclose(s.query("kind=='difference'")['mean'],1.)
    assert len(c)==12


def test_joint_bands_are_centered_on_means_and_reproducible():
    a,c=u.balanced_summary(alignment(),['D1','C1'],'appearance')
    b,_=u.balanced_summary(alignment(),['D1','C1'],'appearance')
    pd.testing.assert_frame_equal(a,b)
    assert a.simultaneous_low.le(a['mean']).all() and a.simultaneous_high.ge(a['mean']).all()
    assert a.simultaneous_family.str.contains('All 2 retained components').all()


def test_inherited_alignment_configuration_cannot_silently_disagree(monkeypatch):
    monkeypatch.setitem(u.CFG,'last_appearances',4)
    with pytest.raises(ValueError,match='last_appearances'):u.validate_alignment_config()


def test_insufficient_independent_support_fails_instead_of_false_precision():
    with pytest.raises(ValueError,match='at least two'):u.bootstrap_weights([1,1,1])
