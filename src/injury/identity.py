"""MLBAM identity and official MLB team membership; never a name-based join."""
import json
import logging
import re

import pandas as pd

from ..ingest_mlb import API
from .acquire import save_frame
from .parser import classify_transaction, parse_dates, parse_injury

LOG = logging.getLogger(__name__)


def normalize_transactions(raw, cfg):
    records = []
    for row in raw.to_dict('records'):
        d = dict(row)
        desc = d.get('description')
        if pd.isna(desc):
            desc = ''
        d['description'] = desc
        d.update(classify_transaction(desc, d.get('type_code'), d.get('type_description')))
        if d['il_related']:
            d.update(parse_injury(desc))
            d.update(parse_dates(desc, d.get('transaction_date'), d.get('effective_date'), cfg.get('maximum_retroactive_gap_days',31)))
        records.append(d)
    frame = pd.DataFrame(records)
    for col in ('transaction_date', 'effective_date', 'resolution_date', 'retroactive_date',
                'injury_event_date', 'explicitly_documented_onset_date', 'surgery_date',
                'event_date_known_at', 'diagnosis_known_at'):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], errors='coerce')
    frame['in_study_window'] = frame.transaction_date.between(pd.Timestamp(cfg['start_date']), pd.Timestamp(cfg['end_date']))
    return frame


def lookup_identities(frame, con, client, cfg):
    ids = sorted(int(n) for n in frame.loc[frame.il_related, 'player_id'].dropna().unique())
    people, failures, team_rows = {}, [], []
    # Existing MLB metadata is a documented fallback, never a name guess.
    known = con.execute('SELECT pitcher_id, pitcher_name, position, throws, mlb_debut, raw_json FROM pitchers').fetchdf()
    for r in known.itertuples():
        people[int(r.pitcher_id)] = dict(player_id=int(r.pitcher_id), primary_position=r.position,
                                         throws=r.throws, mlb_debut=r.mlb_debut,
                                         identity_source='existing_mlb_people_snapshot', player_name_lookup=r.pitcher_name)
    jobs = [ids[i:i+100] for i in range(0, len(ids), 100)]
    for batch, payload, error in client.many(jobs, lambda b: client.json(API + '/people', {'personIds': ','.join(map(str, b))})):
        if error:
            failures.append(dict(source='people', key=','.join(map(str, batch)), error=error))
            continue
        returned = set()
        for p in payload.get('people', []):
            returned.add(p['id'])
            people[p['id']] = dict(player_id=p['id'], primary_position=p.get('primaryPosition', {}).get('abbreviation'),
                                    throws=p.get('pitchHand', {}).get('code'), mlb_debut=p.get('mlbDebutDate'),
                                    identity_source='mlb_stats_api_people', player_name_lookup=p.get('fullName'))
        for pid in set(batch) - returned:
            failures.append(dict(source='people_missing_id', key=str(pid), error='Requested person omitted by API'))
    for year, payload, error in client.many(range(cfg['start_date'].year, cfg['end_date'].year+1),
                                           lambda y: client.json(API + '/teams', dict(sportId=1, season=y))):
        if error:
            failures.append(dict(source='teams', key=str(year), error=error))
            continue
        for t in payload.get('teams', []):
            if t.get('sport', {}).get('id', 1) == 1:
                team_rows.append(dict(season=year, team_id=t['id'], team_name=t['name']))
    if not team_rows:
        raise RuntimeError('MLB team membership could not be verified')
    people_frame = pd.DataFrame([people[n] for n in ids if n in people])
    people_frame['mlb_debut'] = pd.to_datetime(people_frame.mlb_debut, errors='coerce')
    save_frame(people_frame, cfg['output_path'] / 'interim/player_identities.parquet')
    save_frame(pd.DataFrame(team_rows), cfg['output_path'] / 'interim/mlb_team_seasons.parquet')
    save_frame(pd.DataFrame(failures, columns=['source', 'key', 'error']), cfg['report_path'] / 'identity_acquisition_failures.csv')
    history = con.execute('SELECT pitcher_id, count(*) AS appearance_count, min(game_date) AS first_game, max(game_date) AS last_game FROM appearances GROUP BY 1').fetchdf()
    history = {int(r.pitcher_id): r for r in history.itertuples()}
    teams = {(r['season'], r['team_id']): r['team_name'] for r in team_rows}
    candidates = frame[frame.il_related & frame.in_study_window].copy()
    identified = []
    for r in candidates.to_dict('records'):
        pid = int(r['player_id']) if pd.notna(r['player_id']) else None
        meta = people.get(pid, {})
        r.update({k: meta.get(k) for k in ['primary_position', 'throws', 'mlb_debut', 'identity_source', 'player_name_lookup']})
        season = r['transaction_date'].year
        # Placement/transfer/activation roster destination is the controlling team.
        # Only use source team when destination is absent, not when it's a MiLB club.
        team_id = r['to_team_id'] if pd.notna(r['to_team_id']) else r['from_team_id']
        team_id = int(team_id) if pd.notna(team_id) else None
        r['team_id'] = team_id
        r['team_name'] = teams.get((season, team_id), r.get('to_team_name'))
        r['mlb_team_verified'] = (season, team_id) in teams
        position = r['primary_position']
        historical_role = pid in history and bool(re.search(r'\b(?:RHP|LHP|TWP)\b', r['description']))
        r['pitcher_verified'] = position in ('P', 'TWP') or historical_role
        r['pitcher_identity_basis'] = ('mlb_primary_position' if position in ('P', 'TWP') else
                                       'observed_mlb_pitching_and_transaction_role' if historical_role else 'not_verified_pitcher')
        r['two_way_player'] = position == 'TWP'
        r['has_statcast_history'] = pid in history
        r['identity_review_required'] = not meta or (position not in ('P', 'TWP') and historical_role)
        r['included_primary_population'] = bool(r['mlb_team_verified'] and r['pitcher_verified'] and pid is not None)
        r['population_exclusion_reason'] = (None if r['included_primary_population'] else
            'missing_mlbam_id' if pid is None else 'non_mlb_team_or_team_unknown' if not r['mlb_team_verified'] else
            'position_player' if position else 'missing_position_metadata')
        identified.append(r)
    linked = pd.DataFrame(identified)
    for col in ('team_id', 'player_id', 'transaction_id', 'il_length', 'il_from_length'):
        linked[col] = pd.to_numeric(linked[col], errors='coerce').astype('Int64')
    save_frame(linked, cfg['output_path'] / 'interim/il_transactions_all_positions.parquet')
    save_frame(linked[~linked.included_primary_population], cfg['report_path'] / 'excluded_il_transactions.csv')
    LOG.info('IL identity: %s transactions, %s retained for MLB pitchers', len(linked), int(linked.included_primary_population.sum()))
    return linked, failures
