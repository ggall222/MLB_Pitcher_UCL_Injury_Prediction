"""Read-only adapter for retrospective Kang-style pitcher-season cohorts.

This intentionally does not call the UCL landmark risk-set builder: cumulative
pre-IL labels and season-defined controls are different estimands.
"""
from pathlib import Path
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import yaml

from ..injury.episodes import build_episodes
from .kang_spec import ROOT, EXTRA_SQL, FEATURE_NAMES, atomic_json, crosswalk, digest

KEY = ['pitcher_id', 'season']
REGIONS = {'Elbow / UCL': 'elbow', 'Shoulder': 'shoulder'}


def bounded_episodes(transactions, observed, cfg):
    """Reuse the canonical lifecycle logic without subsequent-year QA updates."""
    cutoff = pd.Timestamp(cfg['end_date'])
    if pd.to_datetime(transactions.transaction_date).max() > cutoff or pd.to_datetime(observed.game_date).max() > cutoff:
        raise ValueError('Episode reconstruction must not use observations after the benchmark cutoff.')
    episode_cfg = yaml.safe_load((ROOT / cfg['injury_episode_config']).read_text())
    episode_cfg['end_date'] = cfg['end_date']
    events, _, _ = build_episodes(transactions, episode_cfg, observed)
    return events


def load_inputs(cfg):
    db, injury = ROOT / cfg['database'], ROOT / cfg['injury_transactions']
    if not db.is_file() or not injury.is_file():
        raise FileNotFoundError('Run the existing historical/injury pipeline first: missing database or canonical injury Parquet.')
    cache = ROOT / cfg['cache']
    cache.mkdir(parents=True, exist_ok=True)
    stat = db.stat()
    fingerprint = hashlib.sha256(json.dumps(dict(
        config=cfg, database_size=stat.st_size, database_mtime_ns=stat.st_mtime_ns,
        injury_sha256=digest(injury), builder_sha256=digest(__file__),
        episode_builder_sha256=digest(ROOT / 'src/injury/episodes.py'),
        episode_config_sha256=digest(ROOT / cfg['injury_episode_config']),
        spec_sha256=digest(Path(__file__).with_name('kang_spec.py'))), sort_keys=True).encode()).hexdigest()
    manifest_path = cache / 'input_manifest.json'
    files = [cache / 'appearances.parquet', cache / 'injury_placements.parquet']
    with duckdb.connect(str(db), read_only=True) as c:
        feature_map = crosswalk(c)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('fingerprint') == fingerprint and all(
                    p.exists() and digest(p) == manifest.get('files', {}).get(p.name) for p in files):
                print('Verified cached 2017–2025 appearance and placement inputs.', flush=True)
                return pd.read_parquet(files[0]), pd.read_parquet(files[1]), feature_map, manifest
        print('Reading existing appearances; aggregating five missing paper metrics in SQL.', flush=True)
        columns = list(dict.fromkeys(['game_id', 'pitcher_id', 'game_date', 'season', 'position',
            'pitcher_name', 'game_type', 'is_starter', 'outs_recorded', 'starter_status',
            'boxscore_missing', 'timing_uncertain', 'quality_eligible'] +
            [f for f in FEATURE_NAMES if f not in EXTRA_SQL]))
        query = '''WITH a AS (
            SELECT {cols} FROM appearances
            WHERE game_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) AND game_type = ?
        ), extra AS (
            SELECT p.game_id, p.pitcher_id, {extra}
            FROM pitches p JOIN a ON p.game_id=a.game_id AND p.pitcher_id=a.pitcher_id
            WHERE p.is_actual_pitch AND p.game_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            GROUP BY p.game_id, p.pitcher_id
        ) SELECT a.*, {names} FROM a LEFT JOIN extra USING (game_id,pitcher_id)
          ORDER BY a.game_date,a.game_id,a.pitcher_id'''.format(
              cols=','.join(columns), extra=','.join(v + ' AS ' + k for k, v in EXTRA_SQL.items()),
              names=','.join('extra.' + k for k in EXTRA_SQL))
        a = c.execute(query, [cfg['start_date'], cfg['end_date'], cfg['game_type'],
                              cfg['start_date'], cfg['end_date']]).df()
        # A pre-2026 episode row in the current canonical table can still carry
        # overlap/timing QA updated by a 2026 transaction. Refold the existing
        # normalized transactions with an explicit observation cutoff instead.
        transactions = c.execute('''SELECT * FROM read_parquet(?)
            WHERE transaction_date BETWEEN DATE '2015-01-01' AND CAST(? AS DATE)
            AND included_primary_population
            ORDER BY player_id,transaction_date,transaction_id''', [str(injury), cfg['end_date']]).df()
        observed = c.execute('''SELECT pitcher_id,game_date FROM appearances
            WHERE game_date BETWEEN DATE '2015-01-01' AND CAST(? AS DATE)
            AND NOT timing_uncertain AND game_type='R' ''', [cfg['end_date']]).df()
        rebuilt = bounded_episodes(transactions, observed, cfg)
        fields = ['injury_event_id', 'player_id', 'player_name', 'transaction_date',
            'placement_transaction_id', 'initial_injury_region', 'initial_injury_structure',
            'initial_injury_diagnosis', 'placement_description', 'orphan_transfer', 'left_censored',
            'overlapping_episode', 'timing_uncertain', 'multiple_injury_regions', 'injury_source_primary']
        e = rebuilt.loc[pd.to_datetime(rebuilt.transaction_date).between(cfg['start_date'], cfg['end_date']), fields].copy()
        e.rename(columns={'player_id': 'pitcher_id'}, inplace=True)
    for frame, path in zip([a, e], files):
        tmp = path.with_suffix('.parquet.tmp')
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    manifest = dict(fingerprint=fingerprint, config=cfg, database_read_only=True,
                    sql_end_date=cfg['end_date'], source_database_stat=[stat.st_size, stat.st_mtime_ns],
                    episode_reconstruction_end=cfg['end_date'],
                    episode_transactions=len(transactions), episode_observed_appearances=len(observed),
                    injury_sha256=digest(injury), files={p.name: digest(p) for p in files})
    atomic_json(manifest_path, manifest)
    return a, e, feature_map, manifest


def season_eligibility(appearances, cfg):
    a = appearances.copy()
    if a.duplicated(['game_id', 'pitcher_id']).any():
        raise ValueError('Duplicate pitcher-game keys in source appearances.')
    if pd.to_datetime(a.game_date).max() > pd.Timestamp('2025-12-31'):
        raise ValueError('2026 is protected.')
    a = a[a.position.isin(cfg['positions'])].copy()
    a[FEATURE_NAMES] = a[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    a['row_missing'] = a[FEATURE_NAMES].isna().any(axis=1)
    a['start_count'] = a.is_starter.eq(True).astype(int)
    a['relief_count'] = a.is_starter.eq(False).astype(int)
    a['start_outs'] = a.outs_recorded.where(a.is_starter.eq(True), 0)
    a['relief_outs'] = a.outs_recorded.where(a.is_starter.eq(False), 0)
    a['unknown_role'] = a.is_starter.isna()
    s = a.groupby(KEY).agg(appearances=('game_id', 'size'),
        missing_fraction=('row_missing', 'mean'), starts=('start_count', 'sum'),
        relief=('relief_count', 'sum'), start_outs=('start_outs', 'sum'),
        relief_outs=('relief_outs', 'sum'), unknown_role=('unknown_role', 'any')).reset_index()
    s['season_role'] = np.where(s.starts > s.relief, 'starter', 'reliever')
    tie = s.starts.eq(s.relief)
    s.loc[tie, 'season_role'] = np.where(s.loc[tie, 'start_outs'] > s.loc[tie, 'relief_outs'], 'starter', 'reliever')
    unresolved = tie & s.start_outs.eq(s.relief_outs) | s.unknown_role
    s.loc[unresolved, 'season_role'] = 'ambiguous'
    s['passes_appearances'] = s.appearances.ge(cfg['minimum_season_appearances'])
    s['passes_missingness'] = s.missing_fraction.le(cfg['maximum_season_missing_fraction'])
    s['eligible'] = s.passes_appearances & s.passes_missingness & ~unresolved
    return a.merge(s[KEY + ['season_role', 'eligible']], on=KEY, validate='many_to_one'), s


def construct_cohorts(appearances, events, cfg):
    a, seasons = season_eligibility(appearances, cfg)
    e = events.copy()
    e['transaction_date'] = pd.to_datetime(e.transaction_date)
    e['season'] = e.transaction_date.dt.year
    e['endpoint'] = e.initial_injury_region.map(REGIONS)
    # Dates of actual regular-season games; outside-season placements cannot
    # become incident cases. Calendar-year placements still disqualify controls.
    bounds = appearances.groupby('season').game_date.agg(['min', 'max'])
    e = e.join(bounds, on='season')
    e['in_regular_season'] = e.transaction_date.between(e['min'], e['max'])
    first = e[e.endpoint.notna() & e.in_regular_season].sort_values(
        ['pitcher_id', 'transaction_date', 'injury_event_id']).drop_duplicates(KEY).copy()
    flags = ['orphan_transfer', 'left_censored', 'overlapping_episode', 'timing_uncertain', 'multiple_injury_regions']
    first['eligible_case'] = first.placement_transaction_id.notna() & ~first[flags].fillna(True).any(axis=1)
    # Do not replace an ambiguous first event with a later clearer event.
    first['case_exclusion'] = np.where(first.eligible_case, '', 'ambiguous first placement/episode')
    first = first.merge(seasons[KEY + ['eligible']], on=KEY, how='left', validate='one_to_one')
    first['eligible_case'] &= first.eligible.eq(True)
    first.loc[~first.eligible.eq(True), 'case_exclusion'] = 'pitcher-season ineligible or absent'
    event_lookup = first.set_index(KEY)
    any_il = set(map(tuple, e[KEY].drop_duplicates().to_numpy()))
    target_il = set(map(tuple, e.loc[e.endpoint.notna(), KEY].drop_duplicates().to_numpy()))
    # Unknown anatomy cannot safely certify a healthy target-specific control.
    unknown_il = set(map(tuple, e.loc[e.initial_injury_region.isna() |
        e.initial_injury_region.eq('Unknown / Unspecified'), KEY].drop_duplicates().to_numpy()))
    cohorts, support, flow = {}, [], []
    for stage, f in [('all_regular_season_pitching', appearances), ('pitchers_and_two_way', a),
                     ('eligible_pitcher_seasons', a[a.eligible])]:
        flow.append(dict(stage=stage, rows=len(f), pitcher_seasons=len(f[KEY].drop_duplicates()),
                         unique_pitchers=f.pitcher_id.nunique()))
    for rule in cfg['control_rules']:
        blocked = any_il if rule == 'no_any_il' else target_il | unknown_il
        if rule not in ['no_any_il', 'no_shoulder_elbow_il']:
            raise ValueError('Unknown control rule: ' + rule)
        f = a[a.eligible].copy().set_index(KEY)
        f['is_control'] = ~f.index.isin(blocked)
        f = f.join(event_lookup[['injury_event_id', 'transaction_date', 'endpoint', 'eligible_case']])
        f = f.reset_index()
        f['lead_days'] = (f.transaction_date - pd.to_datetime(f.game_date)).dt.days
        for endpoint in ['elbow', 'shoulder']:
            keep_case = f.endpoint.eq(endpoint) & f.eligible_case.eq(True) & f.lead_days.gt(0)
            c = f[f.is_control | keep_case].copy()
            c['is_case_season'] = ~c.is_control
            for role in ['starter', 'reliever']:
                frame = c[c.season_role.eq(role)].sort_values(['game_date', 'game_id', 'pitcher_id']).reset_index(drop=True)
                cohorts[(rule, endpoint, role)] = frame
                for window in cfg['windows_days']:
                    y = window_labels(frame, window)
                    support.append(dict(control_rule=rule, endpoint=endpoint, role=role,
                        window_days=window, rows=len(frame), pitcher_seasons=len(frame[KEY].drop_duplicates()),
                        unique_pitchers=frame.pitcher_id.nunique(), control_seasons=len(frame.loc[frame.is_control, KEY].drop_duplicates()),
                        case_seasons=len(frame.loc[frame.is_case_season, KEY].drop_duplicates()),
                        represented_events=frame.loc[y.eq(1), 'injury_event_id'].nunique(),
                        positive_rows=int(y.sum()), prevalence=float(y.mean()),
                        earlier_case_negative_rows=int((frame.is_case_season & frame.lead_days.gt(window)).sum()),
                        rows_if_outside_window_cases_omitted=int((frame.is_control | y.eq(1)).sum()),
                        residual_missing_rows=int(frame[FEATURE_NAMES].isna().any(axis=1).sum())))
    return cohorts, seasons, first, pd.DataFrame(support), pd.DataFrame(flow)


def window_labels(frame, window):
    """Cumulative 1..W days before placement; same-day games excluded upstream.

    Earlier pre-IL games are negative under the explicit binary-indicator
    interpretation. They are not silently discarded or reused after injury.
    """
    return (frame.is_case_season & frame.lead_days.between(1, window)).astype(int)
