"""Counterexamples for injury parsing, lifecycle boundaries and immutable acquisition."""
from datetime import date
import json

import pandas as pd
import pytest

from src.injury.acquire import months, flatten_transaction
from src.injury.parser import classify_transaction, parse_injury, parse_dates
from src.injury.episodes import build_episodes
from src.injury.taxonomy import UNKNOWN


@pytest.mark.parametrize('text,region,structure,diagnosis,side', [
    ('Placed LHP A.J. Murray on the 15-day disabled list with a left rotator cuff strain.', 'Shoulder', 'Rotator Cuff', 'Strain', 'Left'),
    ('Placed RHP Player on the 15-day injured list with right forearm tightness.', 'Forearm', None, 'Tightness', 'Right'),
    ('Placed RHP Player on the 60-day injured list following Tommy John surgery.', 'Elbow / UCL', 'UCL', 'Surgery / Postoperative Recovery', 'Unknown'),
    ('Club placed RHP Player on the 15-day injured list. Ulnar collateral ligament tear.', 'Elbow / UCL', 'UCL', 'Tear / Rupture', 'Unknown'),
    ('Club placed LHP Player on the 15-day injured list. Right olecranon stress fracture.', 'Elbow / UCL', 'Olecranon', 'Stress Fracture', 'Right'),
    ('Club placed RHP Player on the 15-day injured list. Left supraspinatus strain.', 'Shoulder', 'Supraspinatus', 'Strain', 'Left'),
    ('Club placed RHP Player on the 15-day injured list. Right hip labral tear.', 'Hip / Groin', 'Labrum', 'Tear / Rupture', 'Right'),
    ('Club placed RHP Player on the 15-day injured list. Left thumb UCL sprain.', 'Hand / Wrist / Finger', 'UCL', 'Sprain', 'Left'),
    ('Club placed RHP Player on the 15-day injured list. Left lat strain.', 'Lat', 'Latissimus', 'Strain', 'Left'),
    ('Club placed RHP Player on the 15-day injured list. Strained left latissimus dorsi.', 'Lat', 'Latissimus', 'Strain', 'Left'),
    ('Club placed RHP Player on the 15-day injured list. Right Achilles tendinitis.', 'Foot / Ankle', 'Achilles', 'Tendinopathy', 'Right'),
    ('Club placed RHP Player on the 15-day injured list. Right meniscus tear.', 'Knee', 'Meniscus', 'Tear / Rupture', 'Right'),
    ('Club placed RHP Player on the 15-day injured list. Left ACL tear.', 'Knee', 'ACL', 'Tear / Rupture', 'Left'),
    ('Club placed RHP Player on the 15-day injured list. Right flexor-pronator strain.', 'Forearm', 'Flexor-pronator mass', 'Strain', 'Right'),
    ('Club placed RHP Player on the 15-day injured list. Both knees are sore.', 'Knee', None, 'Soreness', 'Bilateral'),
    ('Club placed RHP Player on the injured list.', UNKNOWN, None, UNKNOWN, 'Unknown'),
    ('Club placed RHP Player on the injured list. Unrelated evaluation.', UNKNOWN, None, UNKNOWN, 'Unknown'),
    ('Club placed RHP Player on the injured list. Clavicle fracture.', 'Shoulder', 'Clavicle', 'Fracture', 'Unknown'),
    ('Club placed RHP Player on the injured list. Undisclosed illness.', 'Illness / non-musculoskeletal', None, UNKNOWN, 'Unknown'),
])
def test_anatomy_parser(text, region, structure, diagnosis, side):
    parsed = parse_injury(text)
    assert parsed['injury_region'] == region
    assert parsed['injury_structure'] == structure
    assert parsed['injury_diagnosis'] == diagnosis
    assert parsed['laterality'] == side


def test_surgery_is_separate_from_injury_and_handedness():
    parsed = parse_injury('Placed RHP Player on the 60-day injured list following Tommy John surgery.')
    assert parsed['surgery_flag'] and parsed['surgery_type'] == 'Tommy John / UCL reconstruction'
    assert parsed['laterality'] == 'Unknown'
    dates = parse_dates('Placed player on IL following surgery.', '2024-04-01', '2024-03-30')
    assert dates['surgery_date'] is None
    assert dates['injury_event_date'] == date(2024,3,30)


@pytest.mark.parametrize('description,action', [
    ('Placed Player on the 15-day disabled list.', 'placement'),
    ('Placed Player on the 10-day IL.', 'placement'),
    ('Transferred Player from the 15-day injured list to the 60-day injured list.', 'transfer'),
    ('Activated Player from the injured list.', 'activation'),
    ('Reinstated Player from the disabled list.', 'activation'),
    ('Player sent on a rehab assignment.', 'rehab_assignment'),
    ('Placed Player on paternity leave.', 'other'),
    ('Placed Player on the bereavement list.', 'other'),
    (None, 'other'),
])
def test_transaction_actions(description, action):
    assert classify_transaction(description)['il_action'] == action


def test_multiple_regions_and_players_flagged_and_negation_not_confirmed_ucl():
    p = parse_injury('Placed RHP Player on injured list. Right shoulder and left knee soreness.')
    assert p['multiple_injury_regions'] and p['parser_review_required'] and p['laterality'] == 'Unknown'
    p = parse_injury('Placed RHP A and LHP B on injured list. Right elbow strain.')
    assert p['multiple_players_in_description'] and p['injury_region'] == UNKNOWN
    p = parse_injury('Placed RHP Player on injured list. No UCL tear.')
    assert not p['explicit_ucl'] and p['injury_region'] == UNKNOWN
    p = parse_injury('Placed RHP Player on injured list. Possible UCL tear.')
    assert not p['explicit_ucl'] and p['diagnosis_uncertain']


def test_retroactive_dates_known_date_and_invalid_dates():
    p = parse_dates('Placed Player on IL retroactive to March 28, 2024. Shoulder strain.', '2024-04-01', '2024-03-30')
    assert p['injury_event_date'] == date(2024,3,28) and p['event_date_known_at'] == date(2024,4,1)
    assert p['injury_event_date_basis'] == 'retroactive_description_date'
    p = parse_dates('Placed Player on IL retroactive to December 29.', '2024-01-02', None)
    assert p['retroactive_date'] == date(2023,12,29)
    p = parse_dates('Placed Player on IL retroactive to April 12, 2024.', '2024-04-01', '2024-04-01')
    assert p['date_parse_uncertain'] and p['injury_event_date'] == date(2024,4,1)
    p = parse_dates('Placed Player on IL. Injury occurred on March 20, 2024.', '2024-04-01', '2024-03-28')
    assert p['injury_event_date'] == date(2024,3,20)


def tx(tid, date_, action='placement', injury='Right shoulder strain.', length=15, effective=None):
    descriptions = {
        'placement': f'Club placed RHP Player on the {length}-day injured list. {injury}',
        'transfer': f'Club transferred RHP Player from the 15-day injured list to the {length}-day injured list. {injury}',
        'activation': f'Club activated RHP Player from the {length}-day injured list.',
    }
    description = descriptions[action]
    row = dict(transaction_id=tid, player_id=123, player_name='Player', description=description,
               primary_position='P', throws='R', two_way_player=False, has_statcast_history=True,
               pitcher_identity_basis='mlb_primary_position', identity_source='fixture', team_id=1, team_name='Club',
               source_url=f'https://example.com/{tid}', source_retrieved_at='2026-09-18',
               identity_review_required=False, source_record_conflict=False,
               resolution_date=pd.Timestamp(effective or date_))
    row.update(classify_transaction(description))
    row.update(parse_injury(description))
    row.update(parse_dates(description, date_, effective or date_))
    for col in ('transaction_date', 'effective_date', 'injury_event_date', 'retroactive_date'):
        row[col] = pd.Timestamp(row[col]) if row[col] else pd.NaT
    return row


CFG = dict(end_date=date(2026,9,18), activation_max_gap_days=730, long_duration_review_days=180)


def test_transfer_stays_in_episode_resolution_ignored_and_activation_sets_duration():
    f = pd.DataFrame([tx(1,'2024-04-01'),tx(2,'2024-04-10','transfer',length=60),tx(3,'2024-07-01','activation',length=60)])
    e,links,_ = build_episodes(f,CFG)
    assert len(e)==1 and len(links)==3
    assert e.loc[0,'days_out_observed']==91 and e.loc[0,'transferred_to_60_day']
    assert e.loc[0,'initial_il_length']==15 and e.loc[0,'max_il_length']==60
    assert e.loc[0,'activation_match_valid'] and not e.loc[0,'right_censored']


def test_new_placement_prevents_distant_activation_from_closing_old_episode():
    f = pd.DataFrame([tx(1,'2024-04-01'),tx(2,'2024-06-01',injury='Left knee sprain.'),tx(3,'2024-07-01','activation')])
    e,_,_=build_episodes(f,CFG)
    assert len(e)==2 and e.overlapping_episode.all()
    assert e.loc[0,'right_censored'] and pd.isna(e.loc[0,'days_out_observed'])
    assert e.loc[1,'activation_transaction_id']==3


def test_missing_activation_censors_and_orphan_transfer_does_not_invent_placement():
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01','transfer',length=60)]),CFG)
    assert e.loc[0,'right_censored'] and e.loc[0,'left_censored']
    assert pd.isna(e.loc[0,'injury_event_date']) and pd.isna(e.loc[0,'placement_transaction_id'])


@pytest.mark.parametrize('activation,reason', [('2024-04-01','activation_not_strictly_after_placement'),('2026-08-01','activation_exceeds_maximum_matching_gap')])
def test_invalid_activation_dates_remain_censored(activation,reason):
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01'),tx(2,activation,'activation')]),CFG)
    assert e.loc[0,'right_censored'] and e.loc[0,'activation_rejection_reason']==reason
    assert pd.isna(e.loc[0,'days_out_observed'])


def test_observed_return_blocks_false_long_absence():
    observed=pd.DataFrame({'pitcher_id':[123],'game_date':pd.to_datetime(['2024-05-01'])})
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01'),tx(2,'2024-07-01','activation')]),CFG,observed)
    assert not e.loc[0,'activation_match_valid']
    assert e.loc[0,'activation_rejection_reason']=='observed_mlb_pitching_before_activation'


def test_diagnosis_from_later_transfer_has_its_own_known_date():
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01',injury=''),tx(2,'2024-05-01','transfer',injury='Right elbow UCL tear.',length=60)]),CFG)
    assert e.loc[0,'initial_injury_region']==UNKNOWN and e.loc[0,'injury_region']=='Elbow / UCL'
    assert e.loc[0,'diagnosis_known_at']==pd.Timestamp('2024-05-01')
    assert e.loc[0,'public_known_date']==pd.Timestamp('2024-04-01')


def test_duplicate_placement_is_flagged_without_duplicate_event():
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01'),tx(2,'2024-04-01')]),CFG)
    assert len(e)==1 and e.loc[0,'duplicate_placement_count']==1 and e.loc[0,'qa_review_required']


def test_month_windows_and_raw_unknown_fields_are_preserved():
    chunks=list(months(date(2020,1,15),date(2020,3,2)))
    assert chunks==[(date(2020,1,15),date(2020,1,31)),(date(2020,2,1),date(2020,2,29)),(date(2020,3,1),date(2020,3,2))]
    raw={'id':1,'person':{'id':123},'new_unrecognized_source_field':{'evidence':[1,2]}}
    record=flatten_transaction(raw,'sha','2026-01-01',date(2020,1,1),date(2020,1,31))
    assert json.loads(record['raw_json'])==raw


def test_implausibly_old_retroactive_year_is_preserved_but_not_used_for_duration():
    p=parse_dates('Placed player on IL retroactive to May 26, 2015.', '2016-05-27','2015-05-26')
    assert p['retroactive_date']==date(2015,5,26)
    assert p['effective_date']==date(2015,5,26)
    assert p['injury_event_date']==date(2016,5,27) and p['date_parse_uncertain']


def test_placement_on_longer_il_is_continuation_only_when_anatomy_and_history_support_it():
    f=pd.DataFrame([tx(1,'2024-04-01'),tx(2,'2024-05-01',length=60),tx(3,'2024-08-01','activation',length=60)])
    e,links,_=build_episodes(f,CFG)
    assert len(e)==1 and e.loc[0,'inferred_continuation_count']==1
    assert 'inferred_il_continuation' in set(links.relationship)
    assert e.loc[0,'days_out_observed']==122
    observed=pd.DataFrame({'pitcher_id':[123],'game_date':pd.to_datetime(['2024-04-20'])})
    e,_,_=build_episodes(f,CFG,observed)
    assert len(e)==2 and not e.overlapping_episode.any()
    assert e.loc[0,'censor_reason']=='observed_pitching_without_activation'
    assert pd.isna(e.loc[0,'days_out_observed'])


def test_specific_injury_location_prevents_false_multiple_region_flag():
    p=parse_injury('Placed pitcher on IL. Right middle finger blister.')
    assert p['injury_region']=='Hand / Wrist / Finger' and not p['multiple_injury_regions']
    p=parse_injury('Placed pitcher on IL. Right shoulder fatigue.')
    assert p['injury_region']=='Shoulder' and not p['multiple_injury_regions']
    p=parse_injury('Placed pitcher on IL. Right elbow flexor strain.')
    assert p['injury_region']=='Elbow / UCL' and p['injury_structure']=='Flexor (unspecified)'


def test_same_day_blank_and_informative_duplicate_are_one_episode():
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01',injury=''),tx(2,'2024-04-01')]),CFG)
    assert len(e)==1 and e.loc[0,'injury_region']=='Shoulder'


def test_retrieval_timestamp_is_separate_from_unverified_historical_publication():
    e,_,_=build_episodes(pd.DataFrame([tx(1,'2024-04-01')]),CFG)
    assert e.loc[0,'diagnosis_known_at']==pd.Timestamp('2024-04-01')
    assert e.loc[0,'evidence_observed_at']=='2026-09-18'
    assert not e.loc[0,'historical_publication_verified']


def test_failed_refresh_retains_previous_raw_rows_and_reports_failed_window(tmp_path,monkeypatch):
    from src.injury import acquire
    cfg=dict(start_date=date(2024,1,1),end_date=date(2024,2,29),output_path=tmp_path/'data',
             report_path=tmp_path/'reports',refresh_recent_days=62)
    class Client:
        fail=False
        def __init__(self,*a,**kw):self.records=[]
        def get(self,url,params,force_refresh=False):
            if self.fail and params['startDate']=='2024-02-01':raise RuntimeError('temporary outage')
            import hashlib
            body=json.dumps({'transactions':[{'id':int(params['startDate'][5:7]),'person':{'id':123},'date':params['startDate']}]}).encode()
            self.records.append(dict(sha256=hashlib.sha256(body).hexdigest(),retrieved_at='2024-03-01'))
            return body
        def many(self,jobs,fn):
            for j in jobs:
                try:yield j,fn(j),None
                except Exception as exc:yield j,None,str(exc)
    monkeypatch.setattr(acquire,'HttpClient',Client)
    first,_,status=acquire.acquire_transactions(cfg)
    assert len(first)==2 and all(s['status']=='ok' for s in status.values())
    snapshot=list((cfg['output_path']/'raw/snapshots').glob('*.parquet'))[0]
    before=snapshot.read_bytes()
    Client.fail=True
    second,_,status=acquire.acquire_transactions(cfg,refresh_recent=True)
    assert len(second)==2 and status['2024-02-01:2024-02-29']['status']=='failed'
    assert snapshot.read_bytes()==before
