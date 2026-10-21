"""Historical counterexamples: event recurrence, chronology, numerical history, alert budgets."""
from datetime import date
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.research_labels import build_events,build
from src.evaluation import rank_metrics,evaluate_predictions,alert_mask,temporal_partition
from src.history_math import prior_ends,moments
from src.database import Database


def test_vector_history_matches_bruteforce_with_gaps_and_same_date():
    rng=np.random.default_rng(14);v=rng.normal(size=(80,3));v[rng.random(v.shape)<.25]=np.nan
    dates=pd.Series(pd.to_datetime('2020-01-01')+pd.to_timedelta(np.arange(80)//2,unit='D'))
    ends=prior_ends(dates)
    for window in (3,5,10,20):
        n,mean,sd=moments(v,ends,window)
        for i,end in enumerate(ends):
            expected=pd.DataFrame(v[max(0,end-window):end])
            np.testing.assert_array_equal(n[i],expected.count())
            np.testing.assert_allclose(mean[i],expected.mean(),atol=1e-10,equal_nan=True)
            np.testing.assert_allclose(sd[i],expected.std(),atol=1e-10,equal_nan=True)


def fixture_events():
    cfg=load_config();cfg.update(start_date=date(2024,1,1),observation_end=date(2025,12,1))
    procedures=pd.DataFrame([dict(injury_id='first',pitcher_id=1,player_name='Pitcher',surgery_date_lower=pd.Timestamp('2024-04-12'),
        surgery_date_upper=pd.Timestamp('2024-04-12'),surgery_date=pd.Timestamp('2024-04-12'),surgery_date_precision='day',
        confirmed_ucl_surgery=True,procedure_category='ucl_reconstruction',repeat_procedure=False,position='P',
        identity_method='mlbam_source',diagnosis_date=pd.NaT)])
    transactions=pd.DataFrame([
        dict(transaction_id=1,player_id=1,transaction_date=pd.Timestamp('2024-04-06'),effective_date=pd.Timestamp('2024-04-03'),
             description='Placed on injured list. Right elbow inflammation.',is_il_placement=True,is_il_activation=False,is_elbow_related=True),
        dict(transaction_id=2,player_id=1,transaction_date=pd.Timestamp('2025-06-01'),effective_date=pd.Timestamp('2025-06-01'),
             description='Activated from injured list.',is_il_placement=False,is_il_activation=True,is_elbow_related=False)])
    box=pd.DataFrame({'pitcher_id':1,'game_date':pd.to_datetime(['2024-04-01','2024-04-05','2025-06-03','2025-08-05','2025-11-01'])})
    ep,ev,proc=build_events(procedures,transactions,box,cfg)
    return cfg,ep,ev,proc,box,procedures,transactions


def test_public_reference_separate_from_symptom_and_retroactive_il_date():
    cfg,ep,ev,*_=fixture_events()
    assert ev.loc[0,'label_reference_date']==pd.Timestamp('2024-04-06')
    assert ev.loc[0,'il_date']==pd.Timestamp('2024-04-03')
    assert ev.first_reported_symptom_date.isna().all()
    assert ev.last_healthy_appearance.isna().all()


def test_reentry_after_activation_and_observed_mlb_return():
    cfg,ep,ev,proc,box,*_=fixture_events()
    a=pd.DataFrame({'game_id':range(4),'pitcher_id':1,'game_date':pd.to_datetime(['2024-04-01','2024-04-06','2025-06-02','2025-06-03'])})
    result=build(a,ev,ep,box,cfg)
    assert result.loc[0,'ucl_injury_next_30d']
    assert result.loc[1:2,'post_known_injury'].all()
    assert result.loc[1:2,'ucl_injury_next_30d'].isna().all()
    assert result.loc[3,'label_eligible'] and not result.loc[3,'ucl_injury_next_30d']


def test_multiple_procedures_one_episode_not_multiple_events():
    cfg,ep,ev,proc,box,injuries,t=fixture_events()
    more=injuries.copy();more['injury_id']='second';more['procedure_category']='ucl_internal_brace_repair'
    episodes,events,procedures=build_events(pd.concat([injuries,more]),t,box,cfg)
    assert len(procedures)==2 and len(events)==1 and events.loc[0,'procedure_count']==2


def test_surgery_only_and_missing_followup_are_unknown():
    cfg,ep,ev,proc,box,injuries,t=fixture_events()
    episodes,events,_=build_events(injuries,t.iloc[:0],box,cfg)
    a=pd.DataFrame({'game_id':[1],'pitcher_id':[1],'game_date':[pd.Timestamp('2024-04-01')]})
    labels=build(a,events,episodes,box,cfg)
    assert events.label_reference_date.isna().all()
    assert labels.ucl_injury_next_30d.isna().all()


def test_event_metrics_count_events_and_use_warning_window():
    f=pd.DataFrame({'game_id':[1,2,3,4],'pitcher_id':[1,1,2,2],
        'game_date':pd.to_datetime(['2024-04-01','2024-04-08','2024-04-01','2024-04-08']),
        'ucl_injury_next_30d':[1,1,0,0],'score':[.9,.8,.7,.1]})
    ev=pd.DataFrame({'pitcher_id':[1,3],'injury_event_id':['a','unobservable'],'label_reference_date':pd.to_datetime(['2024-04-15','2024-04-15'])})
    summary,lead=evaluate_predictions(f,ev,rates=(.5,))
    assert summary.loc[0,'alerted_events']==1 and summary.loc[0,'injury_event_recall']==.5
    assert summary.loc[0,'observable_event_recall']==1
    assert summary.loc[0,'median_warning_lead_days']==14
    assert summary.loc[0,'average_precision']==1 and summary.loc[0,'roc_auc']==1
    assert summary.loc[0,'precision']==1 and summary.loc[0,'f2']==1
    assert len(lead)==1


def test_alert_budget_ties_and_degenerate_classes():
    f=pd.DataFrame({'game_id':[4,2,1,3],'pitcher_id':1,'score':.5})
    assert list(f.loc[alert_mask(f,'score',.5),'game_id'])==[2,1]
    assert rank_metrics([1,0],[.5,.5])=={'average_precision':.5,'roc_auc':.5}
    assert np.isnan(rank_metrics([0,0],[.1,.2])['average_precision'])
    with pytest.raises(ValueError):rank_metrics([0,1],[.2,np.nan])


def test_temporal_partition_purges_cross_boundary_event():
    f=pd.DataFrame({'game_date':pd.to_datetime(['2021-09-01','2022-02-01','2022-12-15','2023-02-01']),
        'next_injury_event_id':['shared','shared',None,None]})
    result=temporal_partition(f,'2021-12-31','2022-12-31',90)
    assert result.tolist()==['purged','purged','purged','test']


def test_derived_schema_addition_preserves_original_columns():
    d=Database(':memory:');d.initialize()
    d.replace_derived('example',pd.DataFrame({'row_id':[1],'old':[2.]}),['row_id'])
    d.replace_derived('example',pd.DataFrame({'row_id':[1,2],'old':[2.,3.],'new':[4.,5.]}),['row_id'])
    assert d.columns('example')==['row_id','old','new']
    assert d.scalar('select count(*) from example')==2
    with pytest.raises(RuntimeError):d.replace_derived('example',pd.DataFrame({'row_id':[1],'new':[4.]}),['row_id'])
    d.close()


def test_history_variance_stable_for_constant_spin():
    x=np.repeat(2345.7,1000)[:,None]
    ends=np.arange(1000)
    count,mean,sd=moments(x,ends,20)
    assert np.all(sd[20:]==0)
    np.testing.assert_allclose(mean[20:],2345.7)


def test_failure_censors_historical_outcomes():
    cfg,ep,ev,proc,box,*_=fixture_events()
    a=pd.DataFrame({'game_id':[1],'pitcher_id':[1],'game_date':[pd.Timestamp('2024-04-01')]})
    result=build(a,ev,ep,box,cfg,sources_complete=False)
    assert not result.label_eligible.any()
    assert result.filter(like='ucl_injury_next_').isna().all().all()


def test_forearm_episode_can_link_ucl_but_does_not_invent_symptoms():
    cfg,ep,ev,proc,box,injuries,t=fixture_events()
    t.loc[0,'description']='Placed on disabled list. Forearm strain.';t.loc[0,'is_elbow_related']=False
    _,events,_=build_events(injuries,t,box,cfg)
    assert events.loc[0,'onset_confidence']=='forearm_public_report'
    assert events.first_reported_symptom_date.isna().all()


def test_evaluation_rejects_fractional_targets_and_duplicate_indices():
    with pytest.raises(ValueError):rank_metrics([.5,1],[.1,.9])
    with pytest.raises(ValueError):alert_mask(pd.DataFrame({'game_id':[1,2],'pitcher_id':1,'score':.5},index=[0,0]),'score',.5)


def test_distant_negative_with_same_future_event_is_not_cross_partition_leakage():
    f=pd.DataFrame({'game_date':pd.to_datetime(['2021-06-01','2023-05-01']),
        'next_injury_event_id':['same','same'],'ucl_injury_next_30d':[False,True]})
    assert temporal_partition(f,'2021-12-31','2022-12-31').tolist()==['train','test']


def test_duplicate_roster_entries_are_preserved_without_stopping_ingestion():
    from src.ingest_mlb import ingest_rosters
    class Client:
        def many(self,jobs,fn):
            yield 1,{'roster':[
                {'person':{'id':7,'fullName':'Player'},'position':{'abbreviation':'P'},'status':{'description':'Active'}},
                {'person':{'id':7,'fullName':'Player'},'position':{'abbreviation':'DH'},'status':{'description':'Active'}}]},None
    d=Database(':memory:');d.initialize();d.upsert('teams',[{'team_id':1}],['team_id'])
    ingest_rosters(d,Client(),date(2021,4,1))
    assert d.scalar('select count(*) from roster_snapshots')==1
    assert d.scalar('select position from roster_snapshots') is None
    assert d.scalar("select json_array_length(raw_json,'$.duplicate_entries') from roster_snapshots")==2
    assert d.scalar("select status from ingestion_state where source='rosters'")=='ok'
    assert d.scalar('select count(*) from ingestion_issues')==1
    d.close()


def test_failed_statcast_day_bypasses_bad_cached_response(monkeypatch):
    from src import ingest_statcast
    from src.utils import state
    d=Database(':memory:');d.initialize()
    d.upsert('games',[dict(game_id=1,game_date='2024-04-01',game_type='R',status='Final')],['game_id'])
    state(d,'statcast','2024-04-01','failed',0,'Previously cached empty response')
    calls=[]
    def fetch(client,day,force_refresh=False):
        calls.append(force_refresh)
        return pd.DataFrame()
    class Client:
        refresh=False
        def many(self,jobs,fn):
            for job in jobs:yield job,fn(job),None
    monkeypatch.setattr(ingest_statcast,'fetch_day',fetch)
    cfg=load_config();cfg.update(start_date=date(2024,4,1),end_date=date(2024,4,1))
    ingest_statcast.ingest(d,Client(),cfg)
    assert calls==[True]
    assert d.scalar("select status from ingestion_state where source='statcast'")=='failed'
    d.close()


def test_cancelled_schedule_is_not_requested_as_played_baseball():
    from src.ingest_mlb import ingest_schedule,ingest_boxscores
    from src.ingest_statcast import ingest
    class Client:
        refresh=False
        def json(self,*args,**kwargs):
            return {'dates':[{'games':[dict(gamePk=1,officialDate='2016-10-03',gameDate='2016-10-03T18:00:00Z',
                season='2016',gameType='R',status={'abstractGameState':'Final','codedGameState':'C','detailedState':'Cancelled'},
                teams={'home':{'team':{'id':1}},'away':{'team':{'id':1}}})]}]}
        def many(self,jobs,fn):
            assert not jobs,'A cancelled game must not trigger pitch/boxscore downloads'
            return iter(())
    d=Database(':memory:');d.initialize();d.upsert('teams',[{'team_id':1}],['team_id'])
    cfg=load_config();cfg.update(start_date=date(2016,10,3),end_date=date(2016,10,3))
    ingest_schedule(d,Client(),cfg['start_date'],cfg['end_date'])
    ingest_boxscores(d,Client(),cfg['start_date'],cfg['end_date']);ingest(d,Client(),cfg)
    assert d.scalar('select status from games')=='Cancelled'
    assert d.scalar('select count(*) from games')==1
    d.close()


@pytest.mark.parametrize('name',['mechanical','baselines'])
def test_resumed_games_cannot_enter_history_counts_or_values(name):
    from test_pipeline import outings
    from src.build_mechanical_features import build as mechanical
    from src.build_baselines import build as baselines
    cfg=load_config();a=outings(cfg);a['timing_uncertain']=False;a.loc[5,'timing_uncertain']=True
    builder=mechanical if name=='mechanical' else baselines
    expected=builder(a[~a.timing_uncertain],cfg).set_index('game_id')
    actual=builder(a,cfg).set_index('game_id').loc[expected.index]
    pd.testing.assert_frame_equal(expected,actual)


def test_suspended_game_partial_workload_remains_null():
    from test_pipeline import outings
    from src.research_pipeline import mask_uncertain_workloads
    from src.build_workload_features import build as workload
    cfg=load_config();cfg['workload_start_date']=date(2024,1,1)
    a=outings(cfg);a['timing_uncertain']=False;a.loc[5,'timing_uncertain']=True
    a.loc[5,'scheduled_start']=a.loc[5,'game_date']+pd.Timedelta(days=20)
    seasons=pd.DataFrame(columns=['pitcher_id','season','innings_pitched'])
    partial=workload(a,a[~a.timing_uncertain],seasons,cfg)
    masked=mask_uncertain_workloads(a,partial,a)
    assert pd.notna(masked.loc[4,'pitches_prev_7d'])
    assert pd.isna(masked.loc[6,'pitches_prev_7d'])
    assert pd.isna(masked.loc[6,'season_pitch_count'])
    assert pd.isna(masked.loc[6,'starts_prev_30d'])
    assert not masked.loc[6,'workload_7d_complete']
