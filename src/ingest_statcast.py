"""Primary pitch source: daily Baseball Savant CSV requests, all raw fields retained."""
from datetime import timedelta
from io import BytesIO
import json
import logging
import numpy as np
import pandas as pd
from .ingest_mlb import ensure_pitchers
from .utils import state, issue

URL = 'https://baseballsavant.mlb.com/statcast_search/csv'
LOG = logging.getLogger(__name__)
KEYS = ['game_id', 'at_bat_number', 'pitch_number']
RENAME = {'game_pk': 'game_id', 'pitcher': 'pitcher_id', 'batter': 'batter_id', 'type': 'pitch_result'}
NUMERIC = ['game_id', 'pitcher_id', 'batter_id', 'at_bat_number', 'pitch_number', 'inning', 'balls',
           'strikes', 'release_speed', 'effective_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
           'release_extension', 'arm_angle', 'release_spin_rate', 'pfx_x', 'pfx_z', 'spin_axis',
           'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'zone', 'outs_when_up', 'api_break_z_with_gravity']


def fetch_day(client, day, force_refresh=False):
    body = client.get(URL, {'all': 'true', 'type': 'details', 'player_type': 'pitcher',
                           'game_date_gt': str(day), 'game_date_lt': str(day), 'hfGT': 'R|',
                           'group_by': 'name', 'min_pitches': 0}, force_refresh=force_refresh)
    if body.lstrip().startswith(b'<'):
        raise ValueError('Savant returned HTML instead of CSV')
    df = pd.read_csv(BytesIO(body), low_memory=False, encoding='utf-8-sig')
    if not {'game_pk', 'pitcher', 'at_bat_number', 'pitch_number', 'game_date'}.issubset(df.columns):
        raise ValueError('Statcast CSV schema missing required identifiers')
    if len(df) >= 25000:
        raise ValueError('Potential Statcast response truncation at row cap')
    return df


def normalize(df):
    raw = df.to_json(orient='records', date_format='iso')
    result = df.rename(columns=RENAME).copy()
    result['raw_json'] = [json.dumps(row, separators=(',', ':')) for row in json.loads(raw)]
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors='coerce') if col in result else np.nan
    result['game_date'] = pd.to_datetime(result['game_date'], errors='coerce').dt.date
    desc = result['description'].fillna('')
    result['is_actual_pitch'] = ~desc.isin(['automatic_ball', 'automatic_strike'])
    result['is_hbp'] = desc.eq('hit_by_pitch') | result['events'].eq('hit_by_pitch')
    flags = pd.Series('', index=result.index)
    for label, mask in {
        'velocity_out_of_range': result.release_speed.notna() & ~result.release_speed.between(30, 110),
        'release_x_out_of_range': result.release_pos_x.abs().gt(7),
        'release_z_out_of_range': result.release_pos_z.notna() & ~result.release_pos_z.between(0, 9),
        'extension_out_of_range': result.release_extension.notna() & ~result.release_extension.between(2, 9),
        'spin_out_of_range': result.release_spin_rate.notna() & ~result.release_spin_rate.between(0, 4500),
        'arm_angle_out_of_range': result.arm_angle.notna() & ~result.arm_angle.between(-90, 180),
        'missing_tracking': result.release_speed.isna() & result.is_actual_pitch,
    }.items():
        flags.loc[mask] += label + ';'
    result['quality_flags'] = flags.str.rstrip(';').replace('', None)
    return result


def ingest(db, client, cfg):
    start, end = cfg['start_date'], cfg['end_date']
    newest = db.scalar('SELECT max(game_date) FROM pitches')
    done = set(db.query("SELECT item_key FROM ingestion_state WHERE source='statcast' AND status='ok'").item_key)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if cfg.get('statcast_game_dates_only'):
        game_days = set(db.query("SELECT DISTINCT game_date FROM games WHERE status='Final' AND game_type='R'").game_date.dt.date)
        days = [d for d in days if d in game_days]
    # Include failed/missing historical days, so a later success never hides an earlier hole.
    jobs = days if client.refresh else [d for d in days if str(d) not in done]
    LOG.info('Statcast newest=%s; pending days=%s', newest, len(jobs))
    known_games = set(db.query('SELECT game_id FROM games').game_id)
    failed=set(db.query("SELECT item_key FROM ingestion_state WHERE source='statcast' AND status='failed'").item_key)
    # A cached empty/HTML response is not a successful download; retry it over the network.
    def download(day):
        return fetch_day(client,day,force_refresh=True) if str(day) in failed else fetch_day(client,day)
    for day, raw, error in client.many(jobs, download):
        if error:
            state(db, 'statcast', day, 'failed', detail=error)
            continue
        if raw.empty:
            games = db.scalar("SELECT count(*) FROM games WHERE game_date=? AND status='Final'", [day])
            state(db, 'statcast', day, 'failed' if games else 'ok', 0,
                  'Empty response for finalized games' if games else None)
            continue
        df = normalize(raw)
        invalid = (df[KEYS + ['pitcher_id', 'game_date']].isna().any(axis=1) |
                   df.pitcher_id.le(0) | df[KEYS + ['pitcher_id']].mod(1).ne(0).any(axis=1) | ~df.game_id.isin(known_games) |
                   df.game_date.ne(day))
        for idx, row in df[invalid].iterrows():
            issue(db, 'statcast', f'{day}:{idx}', 'Invalid key/date or missing schedule join', json.loads(row.raw_json))
        df = df[~invalid].copy()
        duplicate = df.duplicated(KEYS, keep=False)
        for _, row in df[duplicate].iterrows():
            issue(db, 'statcast', ':'.join(str(row[k]) for k in KEYS), 'Duplicate source pitch', json.loads(row.raw_json))
        # Conflict resolution is deterministic and explicitly audited above.
        df = df.drop_duplicates(KEYS, keep='last')
        people = [{'id': int(pid), 'fullName': group.player_name.iloc[0]}
                  for pid, group in df.groupby('pitcher_id')]
        ensure_pitchers(db, people, 'statcast')
        with db.transaction():
            db.upsert('pitches', df, KEYS)
            state(db, 'statcast', day, 'ok', len(df), f'{int(invalid.sum())} quarantined; {int(duplicate.sum())} duplicate source rows')
        LOG.info('Statcast %s: %s records', day, len(df))
