"""Roegele workbook adapter; no fabricated dates, diagnoses or healthy appearances."""
from calendar import monthrange
from datetime import date, timedelta
from io import BytesIO
import csv
import json
import logging
import re
import unicodedata
import pandas as pd
from .ingest_mlb import ensure_pitchers
from .utils import stable_id, now, state, issue

LOG = logging.getLogger(__name__)


def normalized_name(value):
    text = str(value).strip()
    if ',' in text:
        pieces = text.split(',', 1)
        text = pieces[1] + ' ' + pieces[0]
    text = unicodedata.normalize('NFKD', text)
    return re.sub(r'[^a-z0-9]', '', ''.join(c for c in text if not unicodedata.combining(c)).lower())


def parse_surgery_date(value):
    """Author's notes: day 1 can encode unknown day; Jan 1 can encode unknown month."""
    try:
        value = str(value).strip()
        if not re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', value):
            return None, None, None, 'unknown'
        month, day, year = map(int, value.split('/'))
        parsed = date(year, month, day)
    except (ValueError, TypeError):
        return None, None, None, 'unknown'
    if day == 1 and month == 1:
        return None, date(year, 1, 1), date(year, 12, 31), 'year_or_exact_jan_1'
    if day == 1:
        return None, date(year, month, 1), date(year, month, monthrange(year, month)[1]), 'month_or_exact_day_1'
    return parsed, parsed, parsed, 'day'


def clean(value):
    return None if value is None or pd.isna(value) or str(value).strip() == '' else str(value).strip()


def number(value):
    try:
        n = float(value)
        return n if pd.notna(n) else None
    except (TypeError, ValueError):
        return None


def read_sheet(body):
    lines = body.decode('utf-8-sig').splitlines()
    # TJ List has a donations row before the header; tabs differ in layout.
    index = next((i for i, line in enumerate(lines[:10]) if next(csv.reader([line]), [''])[0] == 'Player'), None)
    if index is None:
        raise ValueError('Could not locate Roegele Player header')
    return pd.read_csv(BytesIO(('\n'.join(lines[index:])).encode()), dtype=str).where(lambda x: x.notna(), None)


def ingest(db, client, cfg):
    base = 'https://docs.google.com/spreadsheets/d/' + cfg['roegele_sheet_id']
    tabs = [('TJ List', base + '/export', {'format': 'csv', 'gid': 0}, 'ucl_reconstruction')]
    if cfg['include_internal_brace']:
        tabs.append(('UCL InternalBrace', base + '/gviz/tq', {'tqx': 'out:csv', 'sheet': 'UCL InternalBrace'}, 'ucl_internal_brace_repair'))
    # Cache author's interpretation rules alongside the datasets.
    try:
        client.get(base + '/gviz/tq', {'tqx': 'out:csv', 'sheet': 'How To Use'}, mutable=True)
    except Exception as exc:
        issue(db, 'injury_notes', 'How To Use', str(exc))
    all_frames = []
    successful = []
    for tab, url, params, procedure in tabs:
        try:
            frame = read_sheet(client.get(url, params, mutable=True))
            if 'Player' not in frame or len(frame) < 5:
                raise ValueError('Unexpected empty/small surgery sheet')
            all_frames.append((tab, frame, procedure))
            successful.append(tab)
            state(db, 'roegele', tab, 'ok', len(frame))
        except Exception as exc:
            state(db, 'roegele', tab, 'failed', detail=str(exc))
            LOG.warning('Roegele %s unavailable: %s', tab, exc)
    if not all_frames:
        return False
    # MLBAM IDs in the primary sheet are authoritative; no fuzzy joins.
    people = []
    for _, frame, _ in all_frames:
        if 'mlbamid' in frame:
            for _, row in frame.iterrows():
                pid = number(row.get('mlbamid'))
                if pid and pid > 0 and pid.is_integer():
                    people.append({'id': int(pid), 'fullName': row['Player']})
    ensure_pitchers(db, people, 'roegele_id')
    identities = db.query('SELECT pitcher_id, pitcher_name, position FROM pitchers')
    name_ids = {}
    for row in identities.itertuples():
        name_ids.setdefault(normalized_name(row.pitcher_name), set()).add(int(row.pitcher_id))
    positions = identities.set_index('pitcher_id').position.to_dict()
    rows = []
    for tab, frame, procedure in all_frames:
        for _, raw in frame.iterrows():
            name = clean(raw.get('Player'))
            if not name:
                continue
            pid = number(raw.get('mlbamid'))
            pid = int(pid) if pid and pid > 0 and pid.is_integer() else None
            method = 'mlbam_source' if pid else 'unmatched'
            flags = []
            if not pid:
                candidates = name_ids.get(normalized_name(name), set())
                if len(candidates) == 1:
                    pid = next(iter(candidates))
                    method = 'unique_exact_normalized_name_crosswalk'
                    flags.append('name_join_review')
                elif len(candidates) > 1:
                    flags.append('ambiguous_identity')
            raw_date = clean(raw.get('TJ Surgery Date') if tab == 'TJ List' else raw.get('UCL primary repair with internal brace construction Date'))
            surgery, lower, upper, precision = parse_surgery_date(raw_date)
            if precision != 'day':
                flags.append('uncertain_surgery_day')
            position = clean(raw.get('Position')) or positions.get(pid)
            throwing = clean(raw.get('Throws'))
            non_throwing = bool(throwing and '*' in throwing)
            if non_throwing:
                flags.append('non_throwing_arm')
            if not pid:
                flags.append('unmatched_mlbam')
            returned, _, _, _ = parse_surgery_date(raw.get('Return Date (same level)', raw.get('Return Date\n(same level)')))
            rows.append({'injury_id': stable_id(tab, pid or normalized_name(name), raw_date, procedure),
                         'pitcher_id': pid, 'player_name': name, 'fangraphs_id': clean(raw.get('fgid')),
                         'team': clean(raw.get('Team')), 'level': clean(raw.get('Level')), 'position': position,
                         'throws': throwing, 'age': number(raw.get('Age')), 'procedure_category': procedure,
                         'repeat_procedure': False, 'surgery_date': surgery, 'surgery_date_raw': raw_date,
                         'surgery_date_lower': lower, 'surgery_date_upper': upper,
                         'surgery_date_precision': precision, 'return_date': returned,
                         'recovery_months': number(raw.get('Recovery Time (months)')), 'surgeon': clean(raw.get('Surgeon(s)')),
                         'source_url': base + '/edit', 'source_tab': tab, 'source_reference': None,
                         'identity_method': method, 'confirmed_ucl_surgery': bool(lower and not non_throwing),
                         'quality_flags': ';'.join(flags) or None, 'raw_json': json.dumps(raw.to_dict(), ensure_ascii=False),
                         'retrieved_at': now()})
    rows = list({r['injury_id']: r for r in rows}.values())
    history = {}
    for row in sorted(rows, key=lambda r: r['surgery_date_lower'] or date.max):
        pid = row['pitcher_id']
        if pid:
            row['repeat_procedure'] = history.get(pid, 0) > 0
            history[pid] = history.get(pid, 0) + 1
    with db.transaction():
        for tab in successful:
            db.con.execute('DELETE FROM injuries WHERE source_tab=?', [tab])
        db.upsert('injuries', rows, ['injury_id'])
    LOG.info('Roegele: %s surgery records, %s with MLBAM IDs', len(rows), sum(r['pitcher_id'] is not None for r in rows))
    return len(successful) == len(tabs)


def link_onsets(db, cfg):
    injuries = db.query('SELECT * FROM injuries')
    transactions = db.query('SELECT * FROM transactions ORDER BY transaction_date, transaction_id')
    appearances = db.query('SELECT b.pitcher_id, g.game_date FROM boxscore_appearances b JOIN games g USING(game_id)')
    by_player = {pid: g for pid, g in transactions.groupby('player_id')}
    dates = {pid: g.game_date for pid, g in appearances.groupby('pitcher_id')}
    updates = []
    for row in injuries.itertuples():
        update = {'injury_id': row.injury_id, 'first_reported_elbow_date': None, 'il_placement_date': None,
                  'diagnosis_date': None, 'known_onset_date': None, 'onset_basis': 'surgery_only_unknown_onset',
                  'onset_transaction_id': None, 'last_pre_onset_appearance': None,
                  'last_healthy_mlb_appearance': None}
        if pd.notna(row.pitcher_id) and pd.notna(row.surgery_date_lower):
            t = by_player.get(row.pitcher_id, pd.DataFrame())
            if not t.empty:
                lower = row.surgery_date_lower - pd.Timedelta(days=cfg['injury_link_lookback_days'])
                candidates = t[t.is_il_placement & t.is_elbow_related &
                               t.transaction_date.between(lower, row.surgery_date_upper)].copy()
                # An activation between a candidate and surgery breaks the inferred episode.
                activations = t[t.is_il_activation & (t.transaction_date < row.surgery_date_lower)]
                if not activations.empty:
                    candidates = candidates[candidates.transaction_date > activations.transaction_date.max()]
                if not candidates.empty:
                    earliest = candidates.iloc[0]
                    update.update(first_reported_elbow_date=earliest.transaction_date,
                                  il_placement_date=earliest.effective_date,
                                  known_onset_date=earliest.transaction_date,
                                  onset_basis='elbow_il_report_linked_by_mlbam_and_time',
                                  onset_transaction_id=int(earliest.transaction_id))
                    previous = dates.get(row.pitcher_id, pd.Series(dtype='datetime64[ns]'))
                    previous = previous[previous < earliest.transaction_date]
                    if not previous.empty:
                        update['last_pre_onset_appearance'] = previous.max()
        updates.append(update)
    db.upsert('injuries', updates, ['injury_id'])
