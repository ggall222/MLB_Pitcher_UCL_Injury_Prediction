"""Endpoint-only adaptation and frozen parent experiment invariants."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.injury_analysis.kang_spec import ROOT, digest
from src.injury_analysis.kang_replication_data import PREDICTORS, season_screen
from src.injury_analysis.kang_replication import execute_task
from src.injury_analysis import kang_ucl_endpoint as pipeline
from src.injury_analysis.kang_ucl_endpoint_data import settings, refine_cohorts, labels, check_hashes


def example():
    cfg=settings()[2]; rows=[]
    for pid in range(1,7):
        for i,day in enumerate([1,5,10,15,20,25]):
            rows.append(dict({c:float(pid) for c in PREDICTORS},pitcher_id=pid,game_id=pid*100+i,
                season=2020,game_date=pd.Timestamp(2020,4,day),is_starter=pid%2==1,outs_recorded=15))
    a=pd.DataFrame(rows)
    e=pd.DataFrame([
        dict(injury_event_id='ucl',player_id=1,transaction_date='2020-04-20',placement_transaction_id=1,injury_region='Elbow / UCL'),
        dict(injury_event_id='generic',player_id=2,transaction_date='2020-04-20',placement_transaction_id=2,injury_region='Elbow / UCL'),
        dict(injury_event_id='forearm',player_id=3,transaction_date='2020-04-20',placement_transaction_id=3,injury_region='Forearm'),
        dict(injury_event_id='shoulder',player_id=4,transaction_date='2020-04-10',placement_transaction_id=4,injury_region='Shoulder'),
        dict(injury_event_id='later_ucl',player_id=4,transaction_date='2020-04-20',placement_transaction_id=5,injury_region='Elbow / UCL'),
        dict(injury_event_id='excluded',player_id=6,transaction_date='2020-04-20',placement_transaction_id=6,injury_region='Elbow / UCL')])
    u=e.rename(columns={'player_id':'player_id'}).copy()
    u['endpoint_primary']=u.injury_event_id.isin(['ucl','forearm','later_ucl'])
    u['prediction_reference_date']=pd.to_datetime(u.transaction_date)-pd.Timedelta(days=2)
    u['evidence_category']=np.where(u.endpoint_primary,'confirmed_official_documentation','broader_elbow')
    u.loc[u.injury_event_id.eq('excluded'),'evidence_category']='confirmed_procedure_supported'
    u['endpoint_exclusions']=np.where(u.injury_event_id.eq('excluded'),'overlapping_episode','')
    u['initial_injury_region']=u.injury_region
    u['procedure_link_valid']=False;u['linked_procedure_ids']=''
    u['ucl_confirmation_reported_date']=pd.to_datetime(u.transaction_date)
    u['endpoint_partition']='test'  # deliberately ignored chronological metadata
    s=season_screen(a,cfg)
    return a,e,u,s,cfg


def test_inherits_every_model_setting_and_identical_implementation():
    spec,base,cfg=settings()
    assert {k:v for k,v in cfg.items() if k not in ['cache','output']}=={k:v for k,v in base.items() if k not in ['cache','output']}
    assert pipeline.execute_task is execute_task
    assert len(PREDICTORS)==22
    assert list(range(cfg['window_min'],cfg['window_max']+1,cfg['window_step']))==list(range(7,50))
    assert cfg['test_fraction']==.33 and cfg['seed']==20261030
    assert cfg['xgboost']==base['xgboost'] and cfg['gam']==base['gam']


def test_non_ucl_and_endpoint_qa_failures_never_become_cases_or_controls():
    a,e,u,s,cfg=example(); frames,*_=refine_cohorts(a,e,u,s,cfg)
    f=pd.concat(frames.values())
    assert not f.pitcher_id.isin([2,6]).any()
    assert set(f.loc[f.case_season,'injury_event_id'])=={'ucl','forearm'}
    assert set(f.loc[~f.case_season,'pitcher_id'])=={5}


def test_existing_reference_date_and_cumulative_labels_not_surgery():
    a,e,u,s,cfg=example();u['surgery_date']=pd.Timestamp('2020-06-01')
    frames,*_=refine_cohorts(a,e,u,s,cfg);f=frames['starter'].query('pitcher_id==1')
    assert f.game_date.max()==pd.Timestamp('2020-04-15')
    assert f.lead_days.tolist()==[17,13,8,3]
    assert labels(f,7).tolist()==[0,0,0,1]
    assert labels(f,14).tolist()==[0,1,1,1]
    assert labels(f,49).tolist()==[1,1,1,1]


def test_no_later_ucl_promotion_after_first_shoulder_or_generic_elbow():
    a,e,u,s,cfg=example();frames,_,first,_=refine_cohorts(a,e,u,s,cfg)
    assert not pd.concat(frames.values()).pitcher_id.eq(4).any()
    assert first.loc[first.pitcher_id.eq(4),'injury_event_id'].tolist()==['shoulder']


def test_grain_role_and_source_features_unchanged():
    a,e,u,s,cfg=example();frames,*_=refine_cohorts(a,e,u,s,cfg)
    for role,f in frames.items():
        assert not f.duplicated(['pitcher_id','game_id']).any()
        assert f.role.eq(role).all()
        z=f.merge(a,on=['pitcher_id','game_id'],suffixes=('','_source'),validate='one_to_one')
        for c in PREDICTORS:assert z[c].equals(z[c+'_source'])
        assert f.pitcher_id.isin(s.loc[s.role.eq(role),'pitcher_id']).all()
    with pytest.raises(AssertionError):refine_cohorts(pd.concat([a,a.iloc[:1]]),e,u,s,cfg)


def test_adjudication_partition_metadata_does_not_enter_cohort_or_features():
    a,e,u,s,cfg=example();f,*_=refine_cohorts(a,e,u,s,cfg)
    u['endpoint_partition']='future_untouched_holdout';u['analysis_eligible']=False
    g,*_=refine_cohorts(a,e,u,s,cfg)
    for role in f:pd.testing.assert_frame_equal(f[role],g[role])
    assert 'risk_frame' not in inspect.getsource(pipeline)


def test_identifier_mismatch_fails_instead_of_name_matching():
    a,e,u,s,cfg=example();u.loc[0,'player_id']=999
    with pytest.raises(AssertionError):refine_cohorts(a,e,u,s,cfg)


def test_hash_guard_detects_modified_frozen_file(tmp_path):
    p=tmp_path/'frozen';p.write_text('original');old={'frozen':digest(p)}
    check_hashes(old,tmp_path);p.write_text('changed')
    with pytest.raises(ValueError,match='Frozen'):check_hashes(old,tmp_path)


def test_saved_benchmark_and_authoritative_endpoint_are_unchanged():
    _,_,cfg=settings();p=ROOT/cfg['output']/'experiment_freeze.json'
    if not p.exists():pytest.skip('Prepare the experiment to create a pre-fit preservation snapshot')
    saved=json.loads(p.read_text());check_hashes(saved['benchmark_hashes']);check_hashes(saved['endpoint_source_hashes'])


def test_completed_extension_auc_and_labels_independently():
    _,_,cfg=settings();found=0
    for role in ['starter','reliever']:
        d=ROOT/cfg['cache']/('ucl_'+role)
        if not (d/'metrics.csv').exists():continue
        found+=1;p=pd.read_csv(d/'predictions.csv');m=pd.read_csv(d/'metrics.csv').iloc[0]
        assert abs(roc_auc_score(p.target,p.prediction)-m.roc_auc)<1e-10
        c=pd.read_csv(d/'cohort.csv',low_memory=False);z=p.merge(c,on=['pitcher_id','game_id'],validate='one_to_one')
        assert z.target.equals(labels(z,int(m.window)))
        assert c.season.between(2017,2025).all()
    if not found:pytest.skip('No completed final models yet; pre-fit invariants are tested separately')
