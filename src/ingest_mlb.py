"""Public MLB Stats API adapter; all seasonStats in boxscores stay out of features."""
from datetime import timedelta
import logging
import json
import re
import pandas as pd
from .utils import now, json_text, date_chunks, innings_to_outs, state, issue

API = 'https://statsapi.mlb.com/api/v1'
LOG = logging.getLogger(__name__)


def ensure_pitchers(db, people, source='mlb_stats_api'):
    rows = {}
    for p in people:
        pid = p.get('id')
        if not pid or int(pid) <= 0:
            continue
        row = {'pitcher_id': int(pid), 'pitcher_name': p.get('fullName'),
               'metadata_source': source, 'retrieved_at': now()}
        # Do not replace previously enriched metadata with thin boxscore references.
        if source == 'mlb_people':
            row.update(birth_date=p.get('birthDate'), mlb_debut=p.get('mlbDebutDate'),
                       throws=p.get('pitchHand', {}).get('code'),
                       position=p.get('primaryPosition', {}).get('abbreviation'), raw_json=json_text(p))
        rows[int(pid)] = row
    existing = set(db.query('SELECT pitcher_id FROM pitchers').pitcher_id)
    new = [r for pid, r in rows.items() if pid not in existing or source == 'mlb_people']
    db.upsert('pitchers', new, ['pitcher_id'])


def ingest_teams(db, client, season):
    payload = client.json(API + '/teams', {'sportId': 1, 'season': season})
    rows = [{'team_id': t['id'], 'team_name': t['name'], 'abbreviation': t.get('abbreviation'),
             'raw_json': json_text(t), 'retrieved_at': now()} for t in payload['teams']]
    db.upsert('teams', rows, ['team_id'])
    db.upsert('team_seasons', [dict(r, season=season) for r in rows], ['team_id','season'])
    state(db, 'teams', season, 'ok', len(rows))


def ingest_schedule(db, client, start, end):
    for left, right in date_chunks(start, end):
        key = str(left) + ':' + str(right)
        try:
            payload = client.json(API + '/schedule', {'sportId': 1, 'startDate': str(left),
                                  'endDate': str(right), 'gameType': 'R'}, mutable=True)
            rows = []
            for day in payload.get('dates', []):
                for g in day.get('games', []):
                    if g['gameType'] != 'R':
                        continue
                    rows.append({'game_id': g['gamePk'], 'game_date': g['officialDate'],
                                 'scheduled_start': g['gameDate'].replace('Z', '+00:00'),
                                 'season': int(g['season']), 'game_type': g['gameType'],
                                 'status': 'Cancelled' if g['status'].get('codedGameState')=='C' or g['status'].get('detailedState')=='Cancelled' else g['status']['abstractGameState'],
                                 'home_team_id': g['teams']['home']['team']['id'],
                                 'away_team_id': g['teams']['away']['team']['id'],
                                 'double_header': g.get('doubleHeader'), 'game_number': g.get('gameNumber'),
                                 'timing_uncertain': bool(g.get('resumeDate') or g.get('resumedFrom') or
                                                         g.get('resumedFromDate')),
                                 'raw_json': json_text(g)})
            # Resumed games can be returned twice under different dates.
            rows = list({r['game_id']: r for r in rows}.values())
            db.upsert('games', rows, ['game_id'])
            state(db, 'schedule', key, 'ok', len(rows))
        except Exception as exc:
            state(db, 'schedule', key, 'failed', detail=str(exc))
            LOG.warning('Schedule failed %s: %s', key, exc)


def ingest_boxscores(db, client, start, end):
    games = db.query("SELECT game_id FROM games WHERE game_date BETWEEN ? AND ? AND status='Final'",
                     [start, end]).game_id.tolist()
    done = set(db.query("SELECT CAST(item_key AS BIGINT) AS id FROM ingestion_state WHERE source='boxscore' AND status='ok'").id)
    legacy = set(db.query('SELECT DISTINCT game_id FROM boxscore_appearances WHERE starter_source IS NULL').game_id)
    jobs = games if client.refresh else [g for g in games if g not in done or g in legacy]
    LOG.info('MLB boxscores: %s pending (%s cached in database)', len(jobs), len(games) - len(jobs))
    failed = set(db.query("SELECT CAST(item_key AS BIGINT) AS id FROM ingestion_state WHERE source='boxscore' AND status='failed'").id)
    def download(gid):
        # A postponed game may have an immutable cached empty boxscore. Retrying
        # the failed item must request a new source version, as Statcast does.
        if gid in failed:
            return json.loads(client.get(API + f'/game/{gid}/boxscore', force_refresh=True))
        return client.json(API + f'/game/{gid}/boxscore')
    for i, (gid, payload, error) in enumerate(client.many(jobs, download)):
        if error:
            state(db, 'boxscore', gid, 'failed', detail=error)
            continue
        people, rows = [], []
        for side, opponent in (('home', 'away'), ('away', 'home')):
            team = payload['teams'][side]
            for index, pid in enumerate(team.get('pitchers', [])):
                player = team['players']['ID' + str(pid)]
                stats = player.get('stats', {}).get('pitching', {})
                if not stats or not stats.get('gamesPlayed', stats.get('gamesPitched', 0)):
                    continue
                people.append(player['person'])
                outs = stats.get('outs')
                if outs is None:
                    outs = innings_to_outs(stats.get('inningsPitched'))
                rows.append({'game_id': gid, 'pitcher_id': pid, 'team_id': team['team']['id'],
                             'opponent_id': payload['teams'][opponent]['team']['id'],
                             'is_starter': bool(stats['gamesStarted']) if 'gamesStarted' in stats else bool(index == 0),
                             'starter_source': 'boxscore_games_started' if 'gamesStarted' in stats else 'boxscore_pitching_order',
                             'pitching_order': index + 1,
                             'outs_recorded': outs, 'innings_pitched': outs / 3 if outs is not None else None,
                             'pitch_count': stats.get('numberOfPitches', stats.get('pitchesThrown')),
                             'batters_faced': stats.get('battersFaced'), 'walks': stats.get('baseOnBalls'),
                             'strikeouts': stats.get('strikeOuts'), 'hit_batters': stats.get('hitBatsmen'),
                             'raw_json': json_text({'person': player['person'], 'stats': stats})})
        ensure_pitchers(db, people)
        db.upsert('boxscore_appearances', rows, ['game_id', 'pitcher_id'])
        state(db, 'boxscore', gid, 'ok' if rows else 'failed', len(rows), None if rows else 'No pitching appearances')
        if (i + 1) % 100 == 0:
            LOG.info('MLB boxscores processed %s/%s', i + 1, len(jobs))


def ingest_people(db, client, missing_only=False):
    where = " WHERE metadata_source <> 'mlb_people' OR metadata_source IS NULL" if missing_only else ''
    ids = db.query('SELECT pitcher_id FROM pitchers' + where + ' ORDER BY pitcher_id').pitcher_id.tolist()
    for offset in range(0, len(ids), 100):
        batch = ids[offset:offset + 100]
        try:
            people = client.json(API + '/people', {'personIds': ','.join(map(str, batch))}, mutable=True)['people']
            ensure_pitchers(db, people, 'mlb_people')
            state(db, 'people', ','.join(map(str, batch)), 'ok', len(people))
        except Exception as exc:
            state(db, 'people', ','.join(map(str, batch)), 'failed', detail=str(exc))
            LOG.warning('Player metadata batch %s failed: %s', offset, exc)


def ingest_rosters(db, client, snapshot_date):
    ids = db.query('SELECT team_id FROM teams').team_id.tolist()
    for tid, payload, error in client.many(ids, lambda t: client.json(API + f'/teams/{t}/roster',
                                                    {'rosterType': 'active', 'date': str(snapshot_date)})):
        if error:
            state(db, 'rosters', f'{tid}:{snapshot_date}', 'failed', detail=error)
            continue
        rows = [{'team_id': tid, 'snapshot_date': snapshot_date, 'player_id': r['person']['id'],
                 'player_name': r['person']['fullName'], 'position': r.get('position', {}).get('abbreviation'),
                 'status': r.get('status', {}).get('description'), 'raw_json': json_text(r)}
                for r in payload.get('roster', [])]
        grouped = {}
        for row in rows:
            grouped.setdefault(row['player_id'], []).append(row)
        canonical = []
        for pid, entries in grouped.items():
            row = entries[0].copy()
            if len(entries) > 1:
                originals = [json.loads(r['raw_json']) for r in entries]
                issue(db,'rosters',f'{tid}:{snapshot_date}:{pid}','Duplicate roster entries retained; conflicting attributes set NULL',originals)
                for field in ('position','status','player_name'):
                    if len({r[field] for r in entries}) > 1:row[field] = None
                row['raw_json'] = json_text({'duplicate_entries': originals})
            canonical.append(row)
        db.upsert('roster_snapshots', canonical, ['team_id', 'snapshot_date', 'player_id'])
        state(db, 'rosters', f'{tid}:{snapshot_date}', 'ok', len(canonical))


def ingest_transactions(db, client, start, end):
    for left, right in date_chunks(start, end):
        key = str(left) + ':' + str(right)
        try:
            payload = client.json(API + '/transactions', {'sportId': 1, 'startDate': str(left),
                                  'endDate': str(right)}, mutable=True)
            rows = []
            for t in payload.get('transactions', []):
                desc = t.get('description', '')
                lower = desc.lower()
                rows.append({'transaction_id': t['id'], 'player_id': t.get('person', {}).get('id'),
                             'player_name': t.get('person', {}).get('fullName'),
                             'transaction_date': t.get('date'), 'effective_date': t.get('effectiveDate'),
                             'resolution_date': t.get('resolutionDate'),
                             'from_team_id': t.get('fromTeam', {}).get('id'),
                             'to_team_id': t.get('toTeam', {}).get('id'), 'type_code': t.get('typeCode'),
                             'type_description': t.get('typeDesc'), 'description': desc,
                             'is_il_placement': ('injured list' in lower or 'disabled list' in lower) and ('placed ' in lower or 'transferred ' in lower),
                             'is_il_activation': 'activated ' in lower and ('injured list' in lower or 'disabled list' in lower),
                             'is_elbow_related': bool(re.search(r'\belbow\b|\bucl\b|tommy john|ulnar collateral', lower)),
                             'raw_json': json_text(t), 'source_url': API + '/transactions?transactionIds=' + str(t['id'])})
            db.upsert('transactions', list({r['transaction_id']: r for r in rows}.values()), ['transaction_id'])
            state(db, 'transactions', key, 'ok', len(rows))
        except Exception as exc:
            state(db, 'transactions', key, 'failed', detail=str(exc))
            LOG.warning('Transactions failed %s: %s', key, exc)


def ingest_previous_seasons(db, client, first_year, last_year):
    for year in range(first_year - 1, last_year):
        try:
            payload = client.json(API + '/stats', {'stats': 'season', 'group': 'pitching', 'season': year,
                                  'sportIds': 1, 'playerPool': 'ALL', 'limit': 10000})
            splits = payload['stats'][0]['splits'] if payload.get('stats') else []
            ensure_pitchers(db, [s['player'] for s in splits])
            rows = []
            for s in splits:
                stats = s['stat']
                outs = innings_to_outs(stats.get('inningsPitched'))
                rows.append({'pitcher_id': s['player']['id'], 'season': year,
                             'innings_pitched': outs / 3 if outs is not None else None,
                             'pitch_count': stats.get('numberOfPitches', stats.get('pitchesThrown')),
                             'games_played': stats.get('gamesPlayed'), 'games_started': stats.get('gamesStarted'),
                             'raw_json': json_text(s)})
            # API can return team splits; aggregate only if no aggregate row is supplied.
            df = pd.DataFrame(rows)
            if not df.empty and df.duplicated(['pitcher_id', 'season']).any():
                raise ValueError('Unexpected multiple season totals per player; preserve source and flag for review')
            db.upsert('season_pitching', df, ['pitcher_id', 'season'])
            state(db, 'season_pitching', year, 'ok', len(rows))
        except Exception as exc:
            state(db, 'season_pitching', year, 'failed', detail=str(exc))
            LOG.warning('Previous-season stats failed: %s', exc)


def ingest_pitch_tracking(db, client, cfg):
    """Supplement Savant with MLB's explicitly named breakVerticalInduced field."""
    if not cfg.get('include_mlb_pitch_tracking', True):
        return
    games = db.query('SELECT DISTINCT game_id FROM pitches WHERE game_date BETWEEN ? AND ? ORDER BY game_id', [cfg['start_date'],cfg['end_date']]).game_id.tolist()
    done = set(db.query("SELECT CAST(item_key AS BIGINT) AS game_id FROM ingestion_state WHERE source='pitch_tracking' AND status='ok'").game_id)
    jobs = games if client.refresh else [g for g in games if g not in done]
    url = 'https://statsapi.mlb.com/api/v1.1/game/{}/feed/live'
    LOG.info('MLB supplemental pitch tracking: %s pending games', len(jobs))
    for index, (gid, payload, error) in enumerate(client.many(jobs, lambda g: client.json(url.format(g)))):
        if error:
            state(db, 'pitch_tracking', gid, 'failed', detail=error)
            continue
        rows = []
        for play in payload.get('liveData', {}).get('plays', {}).get('allPlays', []):
            ab = play.get('atBatIndex', play.get('about', {}).get('atBatIndex'))
            if ab is None:
                continue
            for event in play.get('playEvents', []):
                if not event.get('isPitch') or event.get('pitchNumber') is None:
                    continue
                breaks = event.get('pitchData', {}).get('breaks', {})
                rows.append({'game_id': gid, 'at_bat_number': int(ab) + 1,
                             'pitch_number': event['pitchNumber'],
                             'ivb_inches': breaks.get('breakVerticalInduced'),
                             'horizontal_break_inches': breaks.get('breakHorizontal'),
                             'vertical_break_inches': breaks.get('breakVertical'),
                             'pitch_start': event.get('startTime'), 'pitch_end': event.get('endTime'),
                             'raw_json': json_text(event)})
        frame = pd.DataFrame(rows)
        if not frame.empty:
            duplicates = frame.duplicated(['game_id','at_bat_number','pitch_number'], keep=False)
            for row in frame[duplicates].itertuples():
                issue(db, 'pitch_tracking', f'{gid}:{row.at_bat_number}:{row.pitch_number}',
                      'Duplicate supplemental event key', json.loads(row.raw_json))
            frame = frame.drop_duplicates(['game_id','at_bat_number','pitch_number'], keep='last')
            db.upsert('pitch_tracking', frame, ['game_id','at_bat_number','pitch_number'])
        state(db, 'pitch_tracking', gid, 'ok' if len(frame) else 'failed', len(frame))
        if (index + 1) % 100 == 0:
            LOG.info('MLB supplemental pitch tracking processed %s/%s', index + 1, len(jobs))
