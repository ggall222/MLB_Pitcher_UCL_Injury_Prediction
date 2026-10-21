"""Offline, synthetic counterexamples for leakage, censoring and relational integrity."""
from datetime import date
import json
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.database import Database
from src.build_mechanical_features import build as mechanics, METRICS
from src.build_baselines import build as baselines
from src.build_workload_features import build as workload
from src.build_labels import build as labels, LABEL_COLUMNS
from src.build_appearances import circular_mean, aggregate_pitches
from src.ingest_injuries import parse_surgery_date, normalized_name
from src.ingest_statcast import normalize
from src.utils import innings_to_outs


@pytest.fixture
def cfg():
    c = load_config()
    c.update(start_date=date(2024, 3, 20), end_date=date(2024, 4, 30),
             workload_start_date=date(2024, 1, 1), observation_end=date(2024, 7, 29))
    return c


def outings(cfg, n=12):
    days = pd.date_range('2024-03-20', periods=n, freq='3D')
    a = pd.DataFrame({'game_id': np.arange(1,n+1), 'pitcher_id': 1, 'game_date': days,
                      'scheduled_start': days + pd.Timedelta(hours=18), 'is_starter': True,
                      'pitch_count': np.arange(n) + 80, 'innings_pitched': 5.0})
    for c in METRICS:
        a[c] = np.arange(n) + 90.0
    for pt in cfg['major_pitch_types']:
        for c in ['avg_velocity','spin_avg','release_x_avg','release_z_avg','extension_avg']:
            a[pt.lower()+'_'+c] = np.arange(n) + 1.0
    return a


def empty_injuries():
    return pd.DataFrame({c: pd.Series(dtype=d) for c,d in {
        'pitcher_id':'Int64','confirmed_ucl_surgery':'bool','position':'object',
        'surgery_date_upper':'datetime64[ns]','surgery_date_lower':'datetime64[ns]',
        'known_onset_date':'datetime64[ns]','injury_id':'object','onset_basis':'object'}.items()})


def empty_transactions():
    return pd.DataFrame({c: pd.Series(dtype=d) for c,d in {
        'player_id':'Int64','is_elbow_related':'bool','is_il_placement':'bool',
        'is_il_activation':'bool','transaction_date':'datetime64[ns]'}.items()})


def injury(onset='2024-04-06', surgery='2024-04-12'):
    return pd.DataFrame([{'pitcher_id':1,'confirmed_ucl_surgery':True,'position':'P',
       'surgery_date_lower':pd.Timestamp(surgery),'surgery_date_upper':pd.Timestamp(surgery),
       'known_onset_date':pd.Timestamp(onset) if onset else pd.NaT,
       'injury_id':'one','onset_basis':'elbow_il_report_linked_by_mlbam_and_time'}])


def followup(days):
    return pd.DataFrame({'pitcher_id':1,'game_date':pd.to_datetime(days)})


@pytest.mark.parametrize('builder', [mechanics, baselines])
def test_future_perturbation_cannot_change_earlier_features(builder, cfg):
    a = outings(cfg)
    expected = builder(a, cfg)
    changed = a.copy()
    changed.loc[changed.game_id > 7, METRICS] = 10000.0
    actual = builder(changed, cfg)
    pd.testing.assert_frame_equal(expected[expected.game_id <= 7],actual[actual.game_id <= 7])


def test_current_not_in_own_historical_mean(cfg):
    a=outings(cfg); a.loc[5,'avg_fb_velo']=1000
    m=mechanics(a,cfg).set_index('game_id')
    assert m.loc[6,'avg_fb_velo_prev_5'] == 92
    assert m.loc[6,'fb_velo_change_vs_prev_5'] == 908
    assert pd.isna(m.loc[5,'avg_fb_velo_prev_5'])


def test_same_day_history_is_excluded(cfg):
    a=outings(cfg); a.loc[6,['game_date','scheduled_start']]=a.loc[5,['game_date','scheduled_start']].values
    result=mechanics(a,cfg).set_index('game_id')
    assert result.loc[7,'avg_fb_velo_prev_5']==92


def test_baseline_minimum_and_zero_variance(cfg):
    a=outings(cfg); a['avg_fb_velo']=95.0
    b=baselines(a,cfg)
    assert b.velo_z.isna().all()
    assert b.loc[4,'avg_fb_velo_baseline_count']==4
    assert pd.isna(b.loc[4,'avg_fb_velo_baseline_mean'])
    assert b.loc[5,'avg_fb_velo_baseline_mean']==95


def test_workload_uses_prior_dates_and_real_innings(cfg):
    a=outings(cfg)
    s=pd.DataFrame([{'pitcher_id':1,'season':2023,'innings_pitched':180}])
    expected=workload(a,a,s,cfg)
    changed=a.copy(); changed.loc[changed.game_id>7,['pitch_count','innings_pitched']]=10000
    actual=workload(a,changed,s,cfg)
    pd.testing.assert_frame_equal(expected[expected.game_id<=7],actual[actual.game_id<=7])
    assert expected.loc[2,'pitches_prev_7d']==161
    assert expected.loc[2,'pitches_7d']==243
    assert expected.loc[2,'rest_days']==2
    assert innings_to_outs('5.2')==17
    with pytest.raises(ValueError): innings_to_outs('5.3')


def test_labels_day_boundary_and_post_onset_exclusion(cfg):
    a=pd.DataFrame({'game_id':[1,2,3], 'pitcher_id':1,
                    'game_date':pd.to_datetime(['2024-03-20','2024-04-05','2024-04-06'])})
    l=labels(a,injury(),empty_transactions(),followup(['2024-03-20','2024-04-05','2024-07-20']),cfg)
    assert not l.loc[0,'ucl_injury_next_14d']
    assert l.loc[0,'ucl_injury_next_30d']
    assert l.loc[1,'ucl_injury_next_14d']
    assert not l.loc[2,'label_eligible']
    assert l.loc[2,LABEL_COLUMNS].isna().all()


def test_disappearing_pitcher_not_false_negative(cfg):
    a=pd.DataFrame({'game_id':[1], 'pitcher_id':[1], 'game_date':[pd.Timestamp('2024-04-01')]})
    l=labels(a,empty_injuries(),empty_transactions(),followup(['2024-04-01']),cfg)
    assert l[LABEL_COLUMNS].isna().all().all()


def test_surgery_only_does_not_fabricate_onset(cfg):
    a=pd.DataFrame({'game_id':[1], 'pitcher_id':[1], 'game_date':[pd.Timestamp('2024-04-01')]})
    l=labels(a,injury(onset=None),empty_transactions(),followup(['2024-04-01','2024-07-20']),cfg)
    assert l.loc[0,'uncertain_injury_timing']
    assert l[LABEL_COLUMNS].isna().all().all()


def test_appearance_horizon_counts_event_even_if_no_return(cfg):
    a=pd.DataFrame({'game_id':[1], 'pitcher_id':[1], 'game_date':[pd.Timestamp('2024-04-01')]})
    l=labels(a,injury(),empty_transactions(),followup(['2024-04-01','2024-04-05']),cfg)
    assert l.loc[0,'ucl_injury_next_3_appearances']
    assert l.loc[0,'ucl_injury_next_5_appearances']


def test_failed_injury_source_yields_unknown_targets(cfg):
    a=pd.DataFrame({'game_id':[1], 'pitcher_id':[1], 'game_date':[pd.Timestamp('2024-04-01')]})
    l=labels(a,injury(),empty_transactions(),followup(['2024-04-01','2024-07-20']),cfg,False)
    assert l[LABEL_COLUMNS].isna().all().all()


def test_source_date_placeholders_and_identity():
    exact,lower,upper,precision=parse_surgery_date('6/1/2024')
    assert exact is None and lower==date(2024,6,1) and upper==date(2024,6,30)
    assert parse_surgery_date('1/1/2024')[2]==date(2024,12,31)
    assert parse_surgery_date('4/12/2024')[0]==date(2024,4,12)
    assert parse_surgery_date('Pending')[0] is None
    assert normalized_name('Pérez, Eury')==normalized_name('Eury Perez')


def test_circular_spin_axis_wrap():
    v=circular_mean(pd.Series([359,1]))
    assert min(abs(v),abs(v-360))<1e-8
    assert np.isnan(circular_mean(pd.Series([0,180])))


def test_raw_fields_retained_and_automatic_events_excluded(cfg):
    raw=pd.DataFrame([{'game_pk':1,'pitcher':1,'batter':2,'at_bat_number':1,'pitch_number':i+1,
        'game_date':'2024-04-01','description':'automatic_ball' if i==0 else 'ball',
        'events':None,'type':'B','pitch_type':'FF','release_speed':100+i,
        'pitcher_days_until_next_game':6} for i in range(3)])
    p=normalize(raw)
    assert json.loads(p.loc[0,'raw_json'])['pitcher_days_until_next_game']==6
    p['ivb_inches']=[None,12,16]
    a,_=aggregate_pitches(p,cfg)
    assert a.loc[0,'ivb_inches']==14
    assert a.loc[0,'total_pitches']==2
    assert a.loc[0,'avg_fb_velo']==101.5
    assert pd.isna(a.loc[0,'velo_drop_game'])


def test_upsert_idempotency_and_referenced_parent_refresh():
    db=Database(':memory:');db.initialize()
    db.upsert('teams',[{'team_id':1,'team_name':'Team'}],['team_id'])
    game={'game_id':1,'game_date':'2024-04-01','home_team_id':1,'away_team_id':1,'status':'Final'}
    db.upsert('games',[game],['game_id'])
    db.upsert('pitchers',[{'pitcher_id':1,'pitcher_name':'Pitcher'}],['pitcher_id'])
    p={'game_id':1,'at_bat_number':1,'pitch_number':1,'pitcher_id':1,'game_date':'2024-04-01','raw_json':'{}'}
    db.upsert('pitches',[p],['game_id','at_bat_number','pitch_number'])
    db.upsert('games',[dict(game,status='Corrected')],['game_id'])
    db.upsert('pitches',[p],['game_id','at_bat_number','pitch_number'])
    assert db.scalar('SELECT count(*) FROM pitches')==1
    assert db.scalar('SELECT status FROM games')=='Corrected'
    with pytest.raises(ValueError): db.upsert('pitches',[p,p],['game_id','at_bat_number','pitch_number'])
    db.close()


def test_incremental_ingestion_repairs_hole_without_refetching_complete_day(cfg, monkeypatch):
    from src import ingest_statcast
    from src.utils import state
    db=Database(':memory:');db.initialize()
    db.upsert('pitchers',[{'pitcher_id':1,'pitcher_name':'Pitcher'}],['pitcher_id'])
    for gid,day in [(1,'2024-04-01'),(2,'2024-04-02')]:
        db.upsert('games',[{'game_id':gid,'game_date':day,'status':'Final','game_type':'R'}],['game_id'])
    db.upsert('pitches',[{'game_id':2,'at_bat_number':1,'pitch_number':1,'pitcher_id':1,
                         'game_date':'2024-04-02','raw_json':'{}'}],['game_id','at_bat_number','pitch_number'])
    state(db,'statcast','2024-04-02','ok',1)
    fetched=[]
    def fetch(client,day):
        fetched.append(day)
        return pd.DataFrame([{'game_pk':1,'pitcher':1,'batter':2,'at_bat_number':1,'pitch_number':1,
          'game_date':str(day),'description':'ball','events':None,'type':'B','pitch_type':'FF',
          'release_speed':95,'player_name':'Pitcher'}])
    class Client:
        refresh=False
        def many(self,jobs,fn):
            for job in jobs: yield job,fn(job),None
    monkeypatch.setattr(ingest_statcast,'fetch_day',fetch)
    cfg=dict(cfg,start_date=date(2024,4,1),end_date=date(2024,4,2))
    ingest_statcast.ingest(db,Client(),cfg)
    assert fetched==[date(2024,4,1)]
    assert db.scalar('SELECT count(*) FROM pitches')==2
    db.close()


def test_il_announcement_not_retroactive_effective_date(cfg):
    from src.ingest_injuries import link_onsets
    db=Database(':memory:');db.initialize()
    db.upsert('pitchers',[{'pitcher_id':1,'pitcher_name':'Pitcher'}],['pitcher_id'])
    db.upsert('injuries',[{'injury_id':'one','pitcher_id':1,'surgery_date_lower':'2024-04-12',
                          'surgery_date_upper':'2024-04-12'}],['injury_id'])
    db.upsert('transactions',[{'transaction_id':1,'player_id':1,'transaction_date':'2024-04-06',
        'effective_date':'2024-04-03','is_il_placement':True,'is_il_activation':False,
        'is_elbow_related':True}],['transaction_id'])
    link_onsets(db,cfg)
    assert db.scalar('SELECT known_onset_date FROM injuries')==date(2024,4,6)
    assert db.scalar('SELECT il_placement_date FROM injuries')==date(2024,4,3)
    assert db.scalar('SELECT last_healthy_mlb_appearance FROM injuries') is None
    db.close()


def test_tracking_alignment_accounts_for_automatic_strikes():
    db=Database(':memory:');db.initialize()
    db.upsert('games',[{'game_id':1,'game_date':'2024-04-01'}],['game_id'])
    db.upsert('pitchers',[{'pitcher_id':1}],['pitcher_id'])
    p=[{'game_id':1,'pitcher_id':1,'at_bat_number':1,'pitch_number':n,
         'game_date':'2024-04-01','is_actual_pitch':n>1,'release_speed':v,'raw_json':'{}'}
         for n,v in [(1,None),(2,78),(3,95)]]
    db.upsert('pitches',p,['game_id','at_bat_number','pitch_number'])
    t=[{'game_id':1,'at_bat_number':1,'pitch_number':n,'ivb_inches':ivb,
         'raw_json':json.dumps({'pitchData':{'startSpeed':v}})} for n,v,ivb in [(1,78,-5),(2,95,17)]]
    db.upsert('pitch_tracking',t,['game_id','at_bat_number','pitch_number'])
    result=db.query('SELECT * FROM aligned_pitch_tracking ORDER BY pitch_number')
    assert result.pitch_number.tolist()==[2,3]
    assert result.mlb_pitch_number.tolist()==[1,2]
    assert result.ivb_inches.tolist()==[-5,17]
    assert result.alignment_valid.all()
    db.close()
