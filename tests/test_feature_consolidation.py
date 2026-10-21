"""Leakage boundaries, temporal roles and grouped selection behavior."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.consolidation_inventory import classify
from src.injury_analysis.consolidation_redundancy import cluster_coordinates, correlation_sample
from src.injury_analysis.consolidation_models import ConsolidationEncoder, fit, source_coefficients
from src.injury_analysis.consolidation_stability import pitcher_bootstrap_indices
from src.injury_analysis.consolidation_selection import choose_candidate


@pytest.mark.parametrize('name,representation,current', [
    ('fb_velo_slope_prev_5','slope/trend',False),
    ('fb_velo_change_vs_prev_3','change from recent history',True),
    ('avg_fb_velo_prev_5','rolling historical level',False),
    ('avg_fb_velo_baseline_mean','baseline mean',False),
    ('velo_z','pitcher-relative z-score',True)])
def test_builder_suffix_precedence_preserves_timing(name,representation,current):
    r=classify(name,'mechanical_features')
    assert r['representation']==representation
    assert r['current_appearance_included']==current
    assert r['feature_group']=='Velocity'


def test_workload_current_and_strict_prior_have_different_availability():
    assert classify('pitches_7d','workload_features')['current_appearance_included']
    assert not classify('pitches_prev_7d','workload_features')['current_appearance_included']
    assert not classify('workload_acceleration_7d_vs_21d','workload_features')['current_appearance_included']
    assert classify('season_pitch_count','workload_features')['current_appearance_included']


def test_unknown_ucl_publication_timing_is_not_declared_known_before_outing():
    r=classify('documented_prior_ucl_count','pitcher_baselines')
    assert 'unknown' in r['timing_availability'] and 'excluded' in r['timing_availability']


def test_absolute_correlation_and_complete_linkage_prevent_chaining():
    corr=pd.DataFrame([[1,-.96,.86],[-.96,1,-.95],[.86,-.95,1]],index=list('abc'),columns=list('abc'))
    labels,order,tree=cluster_coordinates(corr,.9)
    assert len(set(labels))==2
    assert labels[0]!=labels[2]
    assert set(order)=={0,1,2} and tree.shape==(2,4)


def test_missing_correlations_do_not_create_redundancy():
    corr=pd.DataFrame([[1,np.nan],[np.nan,1]])
    labels,_,_=cluster_coordinates(corr,.9)
    assert len(set(labels))==2


def test_correlation_sampling_caps_pitchers_and_never_uses_outcome():
    f=pd.DataFrame({'pitcher_id':np.repeat([1,2,3],30),'game_id':np.arange(90),'row_id':np.arange(90),
        'game_date':pd.date_range('2020-01-01',periods=90),'target':np.arange(90)%2})
    original=correlation_sample(f)
    changed=f.copy();changed.target=1-changed.target
    assert original.groupby('pitcher_id').size().eq(12).all()
    np.testing.assert_array_equal(original.row_id,correlation_sample(changed).row_id)


def test_encoder_learns_only_training_distribution():
    f=pd.DataFrame({'a':[1.,2.,3.,4.,np.nan], 'b':[2.,3.,4.,5.,6.],
                    'game_date':pd.date_range('2020-01-01',periods=5)})
    enc=ConsolidationEncoder(['a','b']).fit(f)
    med=enc.imputer.statistics_.copy();scale=enc.scaler.scale_.copy();bounds=enc.bounds.copy()
    result=enc.transform(pd.DataFrame({'a':[1e10,np.nan],'b':[-1e10,1e10]}))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(med,enc.imputer.statistics_)
    np.testing.assert_array_equal(scale,enc.scaler.scale_)
    pd.testing.assert_frame_equal(bounds,enc.bounds)
    assert enc.fit_max_date=='2020-01-05 00:00:00'


def test_rare_binary_flag_is_not_erased_by_quantile_capping():
    f=pd.DataFrame({'rare':[0.]*999+[1.]})
    z=ConsolidationEncoder(['rare']).fit(f).transform(f)
    assert z[-1,0]>z[0,0]


def test_all_null_training_feature_fails_explicitly():
    with pytest.raises(ValueError,match='All-null'):
        ConsolidationEncoder(['a']).fit(pd.DataFrame({'a':[np.nan,np.nan]}))


def test_fit_rejects_validation_and_test_rows():
    for part in ['validation','test']:
        with pytest.raises(ValueError,match='only fit training'):
            fit(pd.DataFrame({'partition':[part,part],'a':[1,2],'target':[0,1]}),['a'])


def test_bootstrap_preserves_every_appearance_of_each_selected_pitcher():
    f=pd.DataFrame({'pitcher_id':[10]*3+[20]*2+[30]*4},index=np.arange(9)+40)
    idx=pitcher_bootstrap_indices(f,15)
    counts=pd.Series(idx).value_counts().reindex(range(len(f)),fill_value=0)
    for positions in f.groupby('pitcher_id').indices.values():
        assert counts.iloc[positions].nunique()==1
    assert sum(counts.iloc[g[0]] for g in f.groupby('pitcher_id').indices.values())==3
    np.testing.assert_array_equal(idx,pitcher_bootstrap_indices(f,15))


def test_sparse_measured_value_selection_is_distinct_from_missingness_selection():
    obj={'encoder':SimpleNamespace(features=['a','b'],names=['a','b','missingindicator_a']),
         'model':SimpleNamespace(coef_=np.array([[0.,-.2,.5]]))}
    c=source_coefficients(obj).set_index('feature')
    assert not c.loc['a','selected'] and c.loc['a','selected_including_indicator']
    assert c.loc['b','selected']


def selection_example():
    perf=pd.DataFrame({'variant':['best','compact','tiny','test_only'],
        'partition':['validation']*3+['test'],'features':[90,20,3,1],
        'average_precision':[.04,.0395,.0394,.99],'alerted_events':[10,9,7,100]})
    noise=np.linspace(-.002,.002,300)
    bs={'best':np.full(300,.04),'compact':.0395+noise,'tiny':.0394+noise}
    return perf,bs


def test_selection_uses_paired_tolerance_and_warning_gate_not_test_performance():
    perf,bs=selection_example()
    chosen,summary=choose_candidate(perf,bs)
    assert chosen=='compact'
    assert 'test_only' not in set(summary.variant)
    assert not summary.set_index('variant').loc['tiny','warning_competitive']


def test_candidate_selection_requires_validation_rows():
    perf,bs=selection_example()
    with pytest.raises(ValueError,match='validation'):
        choose_candidate(perf.loc[perf.partition.eq('test')],bs)
