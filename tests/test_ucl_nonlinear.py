"""Model-form, training-only transforms and local-effect invariants."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from src.injury_analysis.ucl_nonlinear_models import AdditiveSplineEncoder,select,fit,scores
from src.injury_analysis.ucl_nonlinear_effects import ale_1d,local_interactions,reference_sample

def sample(n=800):
    rng=np.random.default_rng(42)
    return pd.DataFrame(dict(x=rng.normal(size=n),y=rng.normal(size=n),binary=rng.integers(0,2,n),
        pitcher_id=np.arange(n)%80,game_date=pd.date_range('2018-01-01',periods=n),partition='train',row_id=np.arange(n)))

def test_spline_encoder_training_only_and_discrete_fallback():
    tr=sample();tr.loc[0,'x']=np.nan;enc=AdditiveSplineEncoder(['x','y','binary']).fit(tr)
    original=enc.transform(tr);bounds=enc.base.bounds.copy();enc.transform(tr.assign(x=1e6,y=-1e6))
    np.testing.assert_allclose(original,enc.transform(tr));pd.testing.assert_frame_equal(bounds,enc.base.bounds)
    assert any(n.startswith('x__spline_') for n in enc.names)
    assert 'binary' in enc.names and not any(n.startswith('binary__spline') for n in enc.names)
    assert 'missingindicator_x' in enc.names

def test_spline_log_odds_are_additive_even_with_nonlinear_response():
    f=sample();enc=AdditiveSplineEncoder(['x','y']).fit(f);rng=np.random.default_rng(23)
    beta=rng.normal(size=len(enc.names))
    fn=lambda z:enc.transform(z)@beta
    c=local_interactions(f,'x','y',fn,[-2,0,2],[-2,0,2])
    assert c.supported.all()
    np.testing.assert_allclose(c.mixed_log_odds,0,atol=1e-12)

def test_local_interactions_detect_known_product_and_mask_sparse_cells():
    f=sample();fn=lambda z:z.x.to_numpy()*z.y.to_numpy()
    c=local_interactions(f,'x','y',fn,[-2,0,2],[-2,0,2])
    np.testing.assert_allclose(c.loc[c.supported,'mixed_log_odds'],4.)
    sparse=local_interactions(f.head(10),'x','y',fn,[-2,0,2],[-2,0,2])
    assert not sparse.supported.any() and sparse.mixed_log_odds.isna().all()

def test_ale_recovers_linear_local_differences_and_centers():
    f=sample();c=ale_1d(f,'x',lambda z:2*z.x.to_numpy()+z.y.to_numpy(),[-2,-1,0,1,2])
    np.testing.assert_allclose(c.local_difference,2.)
    assert np.average(c.ale_log_odds,weights=c.rows)==pytest.approx(0.,abs=1e-12)
    np.testing.assert_allclose(np.diff(c.ale_log_odds),2.)

def test_reference_selection_ignores_outcomes_and_caps_pitcher_dominance():
    f=sample();a=reference_sample(f.assign(target=0));b=reference_sample(f.assign(target=1))
    assert a.row_id.tolist()==b.row_id.tolist()
    assert a.groupby('pitcher_id').size().max()<=5

def test_selection_rejects_transfer_and_prefers_warning_over_ap():
    t=pd.DataFrame(dict(partition=['validation']*2,model_id=['a','b'],average_precision=[.02,.01],
        prevalence=[.005]*2,raw_features=[8]*2,complexity=[0,1],injury_event_recall=[.1,.2]))
    assert select(t)=='b'
    with pytest.raises(ValueError,match='Validation'):select(t.assign(partition='test'))

def test_model_fit_rejects_validation_before_cache_access():
    with pytest.raises(ValueError,match='Training'):fit(sample().assign(partition='validation'),{})

def test_boosted_stumps_have_no_logit_interactions():
    f=sample();y=(f.x**2+f.y>.5).astype(int)
    m=HistGradientBoostingClassifier(max_leaf_nodes=2,max_depth=1,max_iter=8,early_stopping=False).fit(f[['x','y']],y)
    c=local_interactions(f,'x','y',lambda z:m.decision_function(z[['x','y']]),[-2,0,2],[-2,0,2])
    np.testing.assert_allclose(c.mixed_log_odds,0,atol=1e-12)
