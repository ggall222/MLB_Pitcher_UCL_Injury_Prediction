"""Reuse appearance features; derive auditable event alignments and conservative risk sets.

No Statcast reaggregation, no UCL-label overwrite, and no writable database connection.
Injury dates are roster proxies. Reported publication dates are NOT verified archives.
"""
import json
import hashlib
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import yaml
from .common import ROOT, CFG, CACHE, OUT, GROUPS, CATEGORIES, ARM, NONARM, table, source_signature
from src.evaluation import temporal_partition


def feature_spec():
    spec = yaml.safe_load((ROOT / 'config/model_features_v1.yaml').read_text())['features']
    rows = []
    for x in spec:
        row = dict(x)
        name = row['name']
        if name.startswith('arm_angle'):
            row['family'] = 'arm_angle'
        elif name.startswith('extension'):
            row['family'] = 'extension'
        elif 'usage' in name:
            row['family'] = 'pitch_mix'
        else:
            row['family'] = {'pitch_shape': 'spin_movement', 'pitcher_context': 'pitcher_history'}.get(row['category'], row['category'])
        # The old UCL history uses surgery chronology rather than verified publication.
        row['candidate_predictor'] = row['era'] != 'modern_optional' and name != 'documented_prior_ucl'
        rows.append(row)
    for name in ['release_x_sd_z', 'release_z_sd_z']:
        rows.append(dict(name=name, category='release_mechanics', family='release_mechanics',
            era='long_history', candidate_predictor=False,
            rationale='Existing pitcher-relative release variability for descriptive trajectories; outside the fixed v1 predictor set.'))
    for name, rationale in [
        ('is_starter', 'Official role in the completed prediction outing.'),
        ('prior_reported_il_count', 'Count of prior placements reported strictly before this date; not career injuries.'),
        ('calendar_month', 'Calendar context available on the prediction date.')]:
        rows.append(dict(name=name, category='pitcher_context', family='pitcher_history',
            era='long_history', candidate_predictor=True, rationale=rationale))
    return pd.DataFrame(rows)


def classify_events(events):
    """Initial anatomy defines targets; later surgery/anatomy only inform outcome analyses."""
    e = events.copy()
    for c in ['injury_event_date', 'public_known_date', 'transaction_date', 'activation_date',
              'episode_boundary_date', 'surgery_date', 'official_surgery_date', 'observation_end_date']:
        e[c] = pd.to_datetime(e[c])
    e['analysis_group'] = e.initial_injury_region.map(GROUPS).fillna('Unknown')
    e['body_group'] = np.select([e.analysis_group.isin(ARM), e.analysis_group.isin(NONARM)],
                               ['Arm', 'Non-arm'], default='Other / unknown')
    hand = e.throws.map({'L':'Left', 'R':'Right', 'Left':'Left', 'Right':'Right'})
    e['throwing_arm_injury'] = pd.Series(pd.NA, index=e.index, dtype='boolean')
    known_side = e.body_group.eq('Arm') & e.initial_laterality.isin(['Left','Right']) & hand.notna()
    e.loc[known_side, 'throwing_arm_injury'] = e.loc[known_side, 'initial_laterality'].eq(hand.loc[known_side])
    # Conservative alignment: no appearance on/after either plausible onset or reported placement.
    e['anchor_date'] = e[['injury_event_date', 'public_known_date']].min(axis=1)
    # Undated/orphan events block negative labeling but never become positive targets.
    e['blocking_date'] = e.anchor_date.fillna(e.transaction_date)
    postoperative = e.initial_injury_diagnosis.eq('Surgery / Postoperative Recovery')
    postoperative |= e.placement_description.fillna('').str.contains(
        r'recover(?:ing|y) from|post[ -]?operat|rehabilitat.*surgery', case=False, regex=True)
    postoperative |= e.official_surgery_date.notna() & e.official_surgery_date.le(e.anchor_date)
    flags = {
        'missing_initial_date': e.injury_event_date.isna(), 'left_censored': e.left_censored,
        'uncertain_timing': e.timing_uncertain, 'overlapping_episode': e.overlapping_episode,
        'multiple_regions': e.multiple_injury_regions, 'postoperative_or_preexisting': postoperative,
        'inferred_continuation': e.inferred_continuation_count.gt(0),
        'anatomy_source_conflict': e.qa_flags.str.contains('anatomy_conflict', na=False),
        'identity_review': e.qa_flags.str.contains('identity', na=False),
        'unclassified_group': ~e.analysis_group.isin(CATEGORIES),
        'incomplete_sources': ~e.transaction_sources_complete,
    }
    for name, mask in flags.items():
        e['exclude_' + name] = mask.fillna(True).astype(bool)
    e['target_eligible'] = ~e[[f'exclude_{name}' for name in flags]].any(axis=1)
    e['strict_qa_eligible'] = e.target_eligible & ~e.qa_review_required
    e['exclusion_reasons'] = e.apply(lambda r: ';'.join(n for n in flags if r['exclude_' + n]), axis=1)
    e['season'] = e.anchor_date.dt.year.astype('Int64')
    return e.sort_values(['player_id', 'blocking_date', 'injury_event_id']).reset_index(drop=True)


def align_risk_rows(appearances, events, settings=None):
    """Join by MLBAM; censor ambiguous next episodes and require observed negative follow-up.

    Follow-up means an MLB outing on/after the horizon within the configured 30-day
    grace period, not proof of perfect health. All IL anatomy, even excluded cases,
    blocks healthy labels. Outcome ascertainment is never a predictor.
    """
    cfg = CFG if settings is None else settings
    a = appearances.sort_values(['pitcher_id', 'game_date', 'game_id']).copy().reset_index(drop=True)
    a['row_id'] = np.arange(len(a))
    a['active_episode'] = False
    a['days_since_any_event'] = np.nan
    a['prior_reported_il_count'] = 0
    a['next_event_id'] = None
    a['next_event_group'] = None
    a['next_event_date'] = pd.NaT
    a['next_event_eligible'] = False
    a['next_event_strict_qa'] = False
    for h in cfg['horizons']:
        a[f'followup_{h}d'] = False
    by_event = {pid: f for pid, f in events.groupby('player_id')}
    observation_end = events.observation_end_date.max()
    for pid, rows in a.groupby('pitcher_id', sort=False):
        idx = rows.index.to_numpy()
        dates = rows.game_date.to_numpy(dtype='datetime64[ns]')
        for h in cfg['horizons']:
            threshold = dates + np.timedelta64(h, 'D')
            k = np.searchsorted(dates, threshold, side='left')
            valid = k < len(dates)
            next_date = dates[np.minimum(k, len(dates) - 1)]
            valid &= next_date <= threshold + np.timedelta64(cfg['negative_followup_grace_days'], 'D')
            valid &= threshold <= np.datetime64(observation_end)
            a.loc[idx, f'followup_{h}d'] = valid
        es = by_event.get(pid)
        if es is None:
            continue
        es = es.dropna(subset=['blocking_date']).sort_values(['blocking_date', 'injury_event_id'])
        ds = es.blocking_date.to_numpy(dtype='datetime64[ns]')
        known = np.sort(es.public_known_date.dropna().to_numpy(dtype='datetime64[ns]'))
        a.loc[idx, 'prior_reported_il_count'] = np.searchsorted(known, dates, side='left')
        prev = np.searchsorted(ds, dates, side='right') - 1
        valid_prev = prev >= 0
        a.loc[idx[valid_prev], 'days_since_any_event'] = (
            dates[valid_prev] - ds[prev[valid_prev]]) / np.timedelta64(1, 'D')
        nxt = np.searchsorted(ds, dates, side='right')
        valid = nxt < len(es)
        matched = es.iloc[nxt[valid]]
        for target, source in [('next_event_id', 'injury_event_id'), ('next_event_group', 'analysis_group'),
                ('next_event_date', 'blocking_date'), ('next_event_eligible', 'target_eligible'),
                ('next_event_strict_qa', 'strict_qa_eligible')]:
            a.loc[idx[valid], target] = matched[source].to_numpy()
        # Do not assert medical recovery. A documented activation or observed subsequent
        # MLB outing ends an analytical exclusion interval only. Canonical dates stay intact.
        for event in es.itertuples():
            start = np.datetime64(event.blocking_date)
            reported = np.datetime64(event.public_known_date if pd.notna(event.public_known_date) else event.blocking_date)
            after = np.searchsorted(dates, max(start, reported), side='right')
            returns = [np.datetime64(event.activation_date)] if pd.notna(event.activation_date) and event.activation_match_valid else []
            if after < len(dates):
                returns.append(dates[after])
            end = min(returns) if returns else np.datetime64(observation_end) + np.timedelta64(1, 'D')
            a.loc[idx[(dates >= start) & (dates < end)], 'active_episode'] = True
    a['next_event_lead_days'] = (a.next_event_date - a.game_date).dt.days
    # Preserve original quality/role flags, and require strictly prior feature history.
    a['analysis_eligible'] = (a.quality_eligible.fillna(False) & ~a.timing_uncertain.fillna(True)
        & a.prior_appearance_count.ge(cfg['minimum_prior_appearances']) & ~a.active_episode)
    # Reuse the project's chronological embargo helper; general event IDs are audited below.
    a['partition'] = temporal_partition(a, cfg['train_end'], cfg['validation_end'], cfg['embargo_days'])
    a.loc[a.game_date.gt(cfg['test_end']), 'partition'] = cfg.get('post_test_partition', 'descriptive_2026')
    a['calendar_month'] = a.game_date.dt.month
    for h in cfg['horizons']:
        near = a.next_event_lead_days.between(1, h)
        positive = near & a.next_event_eligible
        negative = (~near) & a[f'followup_{h}d']
        label = pd.Series(pd.NA, index=a.index, dtype='Int8')
        label.loc[a.analysis_eligible & negative] = 0
        label.loc[a.analysis_eligible & positive] = 1
        a[f'any_injury_{h}d'] = label
    return a


def prepare(force=False):
    signature = source_signature()
    manifest_path = CACHE / 'manifest.json'
    if not force and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['source_signature'] == signature and old.get('cohort_code_sha256') == hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            return load()
    print('Reading existing appearance features and injury episodes (DuckDB read-only).', flush=True)
    e = classify_events(pd.read_parquet(ROOT / CFG['events_path']))
    spec = feature_spec()
    with duckdb.connect(str(ROOT / CFG['database_path']), read_only=True) as con:
        schema = con.execute('describe model_features').fetchdf()
        table(schema, 'source_model_schema')
        columns = set(schema.column_name)
        context = ['pitcher_id', 'game_id', 'game_date', 'season', 'pitcher_name', 'quality_eligible',
            'timing_uncertain', 'valid_starting_appearance', 'is_starter', 'prior_appearance_count',
            'history_max_date', 'workload_max_date', 'baseline_max_date', 'game_type']
        names = list(dict.fromkeys(context + [x for x in spec.name if x in columns]))
        missing = [x for x in context if x not in columns]
        if missing:
            raise ValueError(f'Required appearance columns missing: {missing}')
        select = ','.join('m."' + n + '"' for n in names)
        # Official primary pitcher / two-way metadata, rather than every position player
        # who threw an emergency inning. No ever-injured-only selection.
        a = con.execute(f'''SELECT {select} FROM model_features m JOIN pitchers p USING (pitcher_id)
            WHERE m.game_type='R' AND p.position IN ('P','TWP') ORDER BY pitcher_id, game_date, game_id''').fetchdf()
        positions = con.execute('SELECT position, count(*) AS pitchers FROM pitchers GROUP BY position').fetchdf()
        table(positions, 'source_pitcher_positions')
    if a.empty:
        raise ValueError('No official pitcher appearances; inspect position codes.')
    a['game_date'] = pd.to_datetime(a.game_date)
    if a[['pitcher_id', 'game_id']].duplicated().any():
        raise AssertionError('Duplicate pitcher/game keys in source')
    for c in ['history_max_date', 'workload_max_date', 'baseline_max_date']:
        if not (pd.to_datetime(a[c]).isna() | pd.to_datetime(a[c]).lt(a.game_date)).all():
            raise AssertionError(f'{c} includes current/future observations')
    a = align_risk_rows(a, e)
    # Don't include dates or outcome flags in X. Feature selection is train-only.
    candidates = spec.loc[spec.candidate_predictor & spec.name.isin(a), 'name'].tolist()
    training = a.loc[a.analysis_eligible & a.partition.eq('train'), candidates].replace([np.inf, -np.inf], np.nan)
    missing = training.isna().mean()
    allowed = missing[missing.le(CFG['maximum_training_missing_fraction']) & training.nunique().gt(1)].index.tolist()
    spec['train_missing_fraction'] = spec.name.map(missing)
    spec['selected_for_prediction'] = spec.name.isin(allowed)
    table(spec, 'feature_selection')
    safe = set(json.loads((ROOT / 'data/exports/feature_columns.json').read_text()))
    assert set(allowed) - {'is_starter', 'prior_reported_il_count', 'calendar_month'} <= safe
    a.to_parquet(CACHE / 'appearances.parquet', index=False)
    e.to_parquet(CACHE / 'events.parquet', index=False)
    spec.to_parquet(CACHE / 'feature_spec.parquet', index=False)
    coverage = a.groupby('season').agg(appearances=('row_id', 'size'), pitchers=('pitcher_id', 'nunique'),
        starts=('is_starter', 'sum'), analysis_eligible=('analysis_eligible', 'sum'),
        first_date=('game_date', 'min'), last_date=('game_date', 'max')).reset_index()
    table(coverage, 'appearance_coverage')
    exclusion = pd.DataFrame([{'reason': c.replace('exclude_', ''), 'events': int(e[c].sum())}
                              for c in e if c.startswith('exclude_')])
    table(exclusion, 'event_exclusions_overlapping_counts')
    counts = e.groupby('analysis_group').agg(all_episodes=('injury_event_id', 'size'),
        target_eligible=('target_eligible', 'sum'), strict_qa_eligible=('strict_qa_eligible', 'sum'),
        unique_pitchers=('player_id', 'nunique')).reset_index()
    table(counts, 'category_counts')
    for detail in ['initial_injury_region', 'initial_injury_structure', 'initial_injury_diagnosis']:
        table(e.groupby(detail, dropna=False).agg(episodes=('injury_event_id', 'size'),
            target_eligible=('target_eligible', 'sum')).reset_index(), detail + '_counts')
    features = spec.loc[spec.name.isin(a), 'name'].tolist()
    availability = a.groupby('season')[features].agg(lambda s: s.notna().mean()).reset_index().melt(
        id_vars='season', var_name='feature', value_name='available_fraction')
    table(availability, 'feature_availability_by_season')
    retained = a.loc[a.analysis_eligible & a.next_event_eligible & a.next_event_lead_days.between(1, max(CFG['horizons']))
        & a.partition.isin(['train', 'validation', 'test'])]
    spans = retained.groupby('next_event_id').partition.nunique()
    assert spans.le(1).all(), 'An injury event spans partitions despite the embargo'
    checks = pd.DataFrame([
        {'check': 'unique pitcher/game keys', 'passed': not a[['pitcher_id','game_id']].duplicated().any()},
        {'check': 'strictly prior rolling/workload dates', 'passed': True},
        {'check': 'positive events confined to one retained partition', 'passed': spans.le(1).all()},
        {'check': 'current/future outcome fields excluded from X', 'passed': True},
        {'check': 'database size and mtime unchanged', 'passed': source_signature() == signature},
        {'check': 'all negative labels have observed horizon follow-up', 'passed': all(
            a.loc[a[f'any_injury_{h}d'].eq(0).fillna(False), f'followup_{h}d'].all() for h in CFG['horizons'])},
    ])
    table(checks, 'analysis_validation_checks')
    assert checks.passed.all()
    manifest = {'source_signature': signature, 'cohort_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config': CFG, 'appearance_rows': len(a), 'events': len(e), 'predictors': allowed,
        'statcast_end': str(a.game_date.max().date()), 'injury_end': str(e.observation_end_date.max().date()),
        'prediction_time': 'after completed appearance; strictly before event/placement date',
        'publication_limitation': 'Historical source descriptions can be revised. Reported dates are publication proxies, not verified contemporaneous archives.'}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Prepared {len(a):,} appearances, {int(e.target_eligible.sum()):,} target-eligible episodes, {len(allowed)} predictors.', flush=True)
    return a, e, spec


def load():
    if not (CACHE / 'manifest.json').exists():
        return prepare()
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    if (manifest['source_signature'] != source_signature()
            or manifest.get('cohort_code_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Analytical cache is stale. Run notebook 01 (or cohort.prepare(force=True)) before continuing.')
    return (pd.read_parquet(CACHE / 'appearances.parquet'), pd.read_parquet(CACHE / 'events.parquet'),
            pd.read_parquet(CACHE / 'feature_spec.parquet'))


def label_frame(appearances, category, horizon, strict=False):
    """Category versus common IL-free comparison pool; competing injuries are excluded.

    Negative means no observed IL episode, not verified absence of symptoms.
    The cohort is explicit so PR-AUC is never presented without its prevalence.
    """
    a = appearances
    base = a[f'any_injury_{horizon}d']
    positive = base.eq(1).fillna(False) & a.next_event_group.eq(category)
    if strict:
        positive &= a.next_event_strict_qa
    negative = base.eq(0).fillna(False)
    f = a.loc[positive | negative].copy()
    f['target'] = positive.loc[f.index].astype(int)
    return f


if __name__ == '__main__':
    prepare(force=True)
