"""Leakage, subspace and selection tests for supervised compression."""
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.supervised_reduction import (
    SupervisedBasis,orthogonal_span,choose_components,fit_classifier,fit_block)


def sample(n=220,seed=4):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,5));x[:,1]=x[:,0]+.4*x[:,1]
    f=pd.DataFrame(x,columns=['a','b','c','d','e'])
    f['throws_left']=rng.integers(0,2,n)
    f['target']=(x[:,2]-x[:,4]+rng.normal(size=n)>0).astype(int)
    f['next_event_id']=np.where(f.target.eq(1),np.arange(n).astype(str),'')
    f.loc[::11,'e']=np.nan
    return f


FEATURES=['a','b','c','d','e','throws_left']


def test_pls_uses_training_outcomes_but_pca_does_not():
    f=sample();g=f.copy();g.target=np.random.default_rng(8).permutation(g.target)
    a=SupervisedBasis(FEATURES,[1,2,3]).fit(f)
    b=SupervisedBasis(FEATURES,[1,2,3]).fit(g)
    np.testing.assert_allclose(a.bases['PCA'],b.bases['PCA'])
    assert abs(a.bases['PLS'][:,0]@b.bases['PLS'][:,0])<.95


def test_heldout_values_and_outcomes_cannot_modify_learned_basis():
    f=sample();basis=SupervisedBasis(FEATURES,[1,2,3]).fit(f)
    before=basis.bases['PLS'].copy();med=basis.encoder.imputer.statistics_.copy()
    new=sample(45,8);new['a']=1e7;new.target=1-new.target
    basis.transform(new,'PLS',2)
    np.testing.assert_array_equal(basis.bases['PLS'],before)
    np.testing.assert_array_equal(basis.encoder.imputer.statistics_,med)


def test_qr_preserves_each_nested_supervised_span():
    rng=np.random.default_rng(7);rotation=rng.normal(size=(7,4));q=orthogonal_span(rotation)
    np.testing.assert_allclose(q.T@q,np.eye(4),atol=1e-12)
    for k in [1,2,3,4]:
        original=rotation[:,:k]
        np.testing.assert_allclose(q[:,:k]@(q[:,:k].T@original),original,atol=1e-12)


def test_missing_and_binary_columns_pass_through_both_projections():
    f=sample();basis=SupervisedBasis(FEATURES,[1,2,3]).fit(f)
    assert [basis.encoder.names[j] for j in basis.keep]==['throws_left','missingindicator_e']
    for method in ['PCA','PLS']:
        assert basis.transform(f,method,2).shape[1]==4
        np.testing.assert_allclose(basis.transform(f,method,2)[:,2:],basis.encoder.standardized(f)[:,basis.keep])


def test_full_orthogonal_completion_preserves_logistic_predictions():
    f=sample();v=sample(70,12);basis=SupervisedBasis(FEATURES,[1,2,3]).fit(f)
    q=np.linalg.qr(basis.bases['PLS'],mode='complete')[0]
    def design(g):
        z=basis.encoder.standardized(g)
        return np.column_stack([(z[:,basis.index]-basis.center)@q,z[:,basis.keep]])
    _,(first,)=fit_classifier(basis.encoder.standardized(f),f.target,[basis.encoder.standardized(v)])
    _,(second,)=fit_classifier(design(f),f.target,[design(v)])
    np.testing.assert_allclose(first,second,atol=2e-6)


def test_selection_ignores_test_metrics_and_breaks_ties_toward_fewer_axes():
    f=pd.DataFrame({'partition':['validation','validation','test'],'components':[1,3,8],
                    'average_precision':[.08,.08,.99]})
    assert choose_components(f)==1
    with pytest.raises(ValueError,match='validation'):choose_components(f.loc[f.partition.eq('test')])


def test_test_outcome_changes_do_not_change_selected_k_or_fitted_probabilities():
    train=sample();val=sample(75,17);test=sample(65,29);changed=test.copy();changed.target=1-changed.target
    first=fit_block(train,val,test,FEATURES,'Departure','synthetic')
    second=fit_block(train,val,changed,FEATURES,'Departure','synthetic')
    assert first[4]['selected']==second[4]['selected']
    for name in first[3]['test']:
        np.testing.assert_allclose(first[3]['test'][name],second[3]['test'][name],atol=1e-12)

