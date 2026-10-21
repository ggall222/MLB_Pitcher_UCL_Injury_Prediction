"""New chronological boundaries, censored follow-up and immutable test decisions."""
import json
import numpy as np
import pandas as pd
import pytest
from src.injury_analysis.cohort import align_risk_rows
from src.injury_analysis.ucl_dynamic_data import risk_frame,event_cohort
from src.injury_analysis.ucl_revised_common import CFG
from src.injury_analysis.ucl_revised_models import fit,select,candidates
from src.injury_analysis.ucl_revised_evaluation import operational,score_frozen

def test_expanded_partitions_keep_embargo_and_hold_out_2026():
    dates=pd.to_datetime(['2021-07-01','2023-10-31','2023-12-01','2024-04-01','2025-12-01','2026-04-01'])
    a=pd.DataFrame(dict(pitcher_id=1,game_id=range(len(dates)),game_date=dates,
        quality_eligible=True,timing_uncertain=False,prior_appearance_count=10))
    e=pd.DataFrame(dict(player_id=[99],observation_end_date=pd.to_datetime(['2026-10-20'])))
    result=align_risk_rows(a,e,settings=CFG)
    assert result.partition.tolist()==['train','train','purged','validation','purged','test']

def test_event_denominators_follow_revised_not_legacy_split():
    e=pd.DataFrame(dict(injury_event_id=['a','b','c'],player_id=[1,2,3],endpoint_primary=True,
        prediction_reference_date=pd.to_datetime(['2023-07-01','2025-07-01','2026-07-01']),
        observation_end_date=pd.to_datetime(['2026-10-20']*3)))
    assert event_cohort(e,'train',settings=CFG).injury_event_id.tolist()==['a']
    assert event_cohort(e,'validation',settings=CFG).injury_event_id.tolist()==['b']
    assert event_cohort(e,'test',settings=CFG).injury_event_id.tolist()==['c']

def test_late_2026_absence_of_followup_stays_unknown():
    a=pd.DataFrame(dict(pitcher_id=1,game_id=[1,2,3],game_date=pd.to_datetime(['2026-08-15','2026-09-15','2026-09-27']),analysis_eligible=True))
    e=pd.DataFrame(dict(player_id=[99],observation_end_date=pd.to_datetime(['2026-10-20'])))
    f=risk_frame(a,e,horizon=30)
    assert f.target.iloc[0]==0 and f.target.iloc[1:].isna().all()
    assert f.label_status.iloc[-1]=='censored_no_observed_followup'

def test_model_fit_refuses_validation_and_test_dates():
    for partition,day in [('validation','2024-07-01'),('test','2026-07-01'),('train','2026-07-01')]:
        f=pd.DataFrame(dict(partition=[partition],game_date=pd.to_datetime([day])))
        with pytest.raises(ValueError,match='training'):fit(f,{})

def test_model_choice_refuses_test_and_uses_warning_rule():
    x=pd.DataFrame(dict(partition='validation',model_id=['a','b'],average_precision=[.02,.01],prevalence=.005,
        injury_event_recall=[.1,.2],raw_features=8,algorithm_complexity=0,complexity=0))
    assert select(x)=='b'
    with pytest.raises(ValueError,match='Validation only'):select(x.assign(partition='test'))

def test_test_scores_require_registered_training_signature():
    obj=dict(signature='bad',training_max_date='2023-09-30')
    with pytest.raises(ValueError,match='registered'):score_frozen(obj,pd.DataFrame(),{'fit_signatures':{'30|x':'good'}},30,'x')
    obj.update(signature='good',training_max_date='2024-01-01')
    with pytest.raises(ValueError,match='Nontraining'):score_frozen(obj,pd.DataFrame(),{'fit_signatures':{'30|x':'good'}},30,'x')

def test_operational_unknown_alerts_not_false_negatives():
    g=pd.DataFrame(dict(target=pd.array([1,0,None],dtype='Int8'),dynamic_next_target=[True,False,False],
        dynamic_lead_days=[15,np.nan,np.nan],dynamic_next_event_id=['e',None,None]))
    ev=pd.DataFrame({'injury_event_id':['e']})
    m=operational(g,[.9,.8,.7],ev,.5,30)
    assert m['alerts']==3 and m['unknown_outcome_alerts']==1
    assert m['observed_negative_alerts']==1 and m['actionable_alerts']==1

def test_no_test_information_in_predeclared_candidate_grid():
    from src.injury_analysis.ucl_dynamic_data import frozen_representations
    grid=candidates(frozen_representations())
    assert len(grid)==59
    assert set(s['algorithm'] for s in grid.values())=={'Linear','Spline','Boosting'}
    assert all(not any(c.startswith('ucl_injury_next') for c in s['features']) for s in grid.values())

def test_failed_boxscore_refetches_stale_postponed_game_cache():
    from datetime import date
    from src.ingest_mlb import ingest_boxscores
    class DB:
        def __init__(self):self.writes=[]
        def query(self,sql,params=None):
            if 'SELECT pitcher_id FROM pitchers' in sql:return pd.DataFrame({'pitcher_id':[11]})
            if 'FROM games ' in sql:return pd.DataFrame({'game_id':[7]})
            if 'starter_source' in sql:return pd.DataFrame({'game_id':[]})
            return pd.DataFrame({'id':[7] if "status='failed'" in sql else []})
        def upsert(self,table,rows,keys):self.writes.append((table,rows))
    class Client:
        refresh=False
        def __init__(self):self.forced=[]
        def many(self,jobs,fn):
            for j in jobs:yield j,fn(j),None
        def json(self,*args):raise AssertionError('Stale cache must be bypassed')
        def get(self,url,force_refresh=False):
            self.forced.append(force_refresh)
            return json.dumps({'teams':{'home':{'team':{'id':1},'pitchers':[11],
                'players':{'ID11':{'person':{'id':11,'fullName':'Pitcher'},'stats':{'pitching':{'gamesPlayed':1,'gamesStarted':1,'outs':3}}}}},
                'away':{'team':{'id':2},'pitchers':[],'players':{}}}})
    db=DB();client=Client();ingest_boxscores(db,client,date(2026,9,22),date(2026,9,22))
    assert client.forced==[True]
    state=[rows for name,rows in db.writes if name=='ingestion_state'][0][0]
    assert state['status']=='ok' and state['row_count']==1
