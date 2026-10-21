"""Behavioral checks for training-only projections and paired reduced models."""
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.reduction import PCAEncoder, fit_model, subspace_overlap, between_pitcher_fraction
from src.injury_analysis.reduction_prediction import choose_dimension


def sample(seed=1,n=150):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,4))
    x[:,1]=.9*x[:,0]+.2*x[:,1]
    f=pd.DataFrame(x,columns=['a','b','c','d'])
    f['role']=rng.integers(0,2,n)
    f['target']=(x[:,0]-.4*x[:,2]+rng.normal(size=n)>0).astype(int)
    f.loc[::11,'d']=np.nan
    return f


def test_validation_extremes_cannot_change_training_transform():
    train=sample();enc=PCAEncoder(['a','b','c','d','role'],['a','b','c','d'],2).fit(train)
    med=enc.imputer.statistics_.copy();scale=enc.scaler.scale_.copy();basis=enc.pca.components_.copy()
    out=sample(2);out['a']=1e8;out['d']=np.nan
    before=enc.transform(train)
    enc.transform(out)
    np.testing.assert_array_equal(enc.imputer.statistics_,med)
    np.testing.assert_array_equal(enc.scaler.scale_,scale)
    np.testing.assert_array_equal(enc.pca.components_,basis)
    np.testing.assert_array_equal(enc.transform(train),before)
    assert enc.imputer.statistics_[3]==pytest.approx(train.d.median())


def test_binary_and_missingness_flags_bypass_reduction():
    enc=PCAEncoder(['a','b','c','d','role'],['a','b','c','d'],2).fit(sample())
    assert [enc.names[i] for i in enc.keep_index]==['role','missingindicator_d']
    assert enc.output_dimensions==4
    z=enc.standardized(sample())
    np.testing.assert_allclose(enc.transform(sample())[:,2:],z[:,enc.keep_index])


def test_zero_component_diagnostic_retains_only_role_and_missingness():
    enc=PCAEncoder(['a','b','c','d','role'],['a','b','c','d'],0).fit(sample())
    assert enc.output_dimensions==2
    assert enc.transform(sample()).shape==(150,2)
    assert [enc.names[i] for i in enc.keep_index]==['role','missingindicator_d']


def test_full_unwhitened_rotation_preserves_linear_model_and_penalty():
    train=sample(n=400);test=sample(4)
    features=['a','b','c','d','role']
    enc0,m0,(s0,)=fit_model(train,[test],features,tolerance=1e-9)
    enc1,m1,(s1,)=fit_model(train,[test],features,features[:-1],'all',tolerance=1e-9)
    np.testing.assert_allclose(s0,s1,atol=2e-6)
    np.testing.assert_allclose(enc0.effective_coefficients(m0),enc1.effective_coefficients(m1),atol=2e-5)
    assert np.linalg.norm(m0.coef_)==pytest.approx(np.linalg.norm(m1.coef_),rel=1e-5)


def test_all_missing_training_field_has_actionable_error():
    f=sample();f['d']=np.nan
    with pytest.raises(ValueError,match='All-missing training columns'):
        PCAEncoder(['a','d'],['a','d'],1).fit(f)


def test_variance_cutoff_is_learned_from_training_only():
    enc=PCAEncoder(['a','b','c','d'],['a','b','c','d'],.9).fit(sample())
    assert enc.pca.explained_variance_ratio_.sum()>=.9
    assert enc.pca.explained_variance_ratio_[:-1].sum()<.9
    k=enc.pca.n_components_
    enc.transform(sample(9))
    assert enc.pca.n_components_==k


def test_subspace_measure_ignores_sign_and_within_subspace_rotation():
    basis=np.eye(4)
    other=np.array([[1.,1.,0,0],[1.,-1.,0,0],[0,0,1.,0],[0,0,0,1.]])
    other[:2]/=np.sqrt(2)
    assert subspace_overlap(basis,other,2)==pytest.approx(1)
    assert subspace_overlap(basis,np.roll(basis,2,axis=0),2)==pytest.approx(0)


def test_profile_vs_change_variance_decomposition():
    ids=np.repeat([1,2,3],3)
    assert between_pitcher_fraction(np.repeat([-2,0,2],3),ids)==pytest.approx(1)
    assert between_pitcher_fraction(np.tile([-1,0,1],3),ids)==pytest.approx(0)


def test_dimension_choice_uses_only_validation_metric_and_excludes_controls():
    f=pd.DataFrame({'variant':['PCA 5','PCA 10','PCA 20','PCA 90%','Original 62','Full-rank PCA control'],
        'average_precision':[.08,.09,.07,.09,.99,.99], 'design_dimensions':[7,12,22,30,62,62]})
    assert choose_dimension(f)=='PCA 10'
