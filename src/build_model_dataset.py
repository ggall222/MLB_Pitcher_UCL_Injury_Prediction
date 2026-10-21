"""Explicit feature registration blocks raw Savant future fields and label metadata."""
import json
import pandas as pd
from .build_labels import LABEL_COLUMNS

UNSAFE = set(LABEL_COLUMNS + ['game_id', 'pitcher_id', 'game_date', 'scheduled_start', 'pitcher_name',
    'first_observed_pitcher_id', 'starter_crosscheck_disagrees', 'official_starters_on_team', 'next_injury_event_id', 'starter_source', 'pitching_order', 'starter_status', 'valid_starting_appearance',
    'game_type', 'appearance_throws', 'team', 'opponent', 'team_id', 'opponent_id', 'birth_date', 'mlb_debut', 'position', 'throws',
    'season', 'history_max_date', 'baseline_max_date', 'workload_max_date',
    'label_eligible', 'training_eligible', 'post_known_injury', 'uncertain_injury_timing',
    'risk_set_stop_date', 'next_known_ucl_onset', 'next_injury_id', 'last_observed_appearance',
    'label_observation_end', 'label_sources_complete', 'label_onset_basis',
    'timing_uncertain', 'boxscore_missing', 'quality_eligible'])


def build(appearances, workload, mechanical, baselines, labels):
    result = appearances.copy()
    feature_columns = [c for c in appearances if c not in UNSAFE and pd.api.types.is_numeric_dtype(appearances[c])]
    if 'is_starter' not in feature_columns:
        feature_columns.append('is_starter')
    for table in (workload, mechanical, baselines):
        feature_columns.extend(c for c in table if c not in UNSAFE)
        result = result.merge(table, on=['game_id', 'pitcher_id'], validate='one_to_one')
    result = result.merge(labels, on=['game_id', 'pitcher_id'], validate='one_to_one')
    result['training_eligible'] = (result.quality_eligible & result.label_eligible).astype('boolean')
    assert not (set(feature_columns) & UNSAFE)
    assert len(feature_columns) == len(set(feature_columns))
    return result, feature_columns


def load_model_data(database_path=None, target='ucl_injury_next_30d', starters_only=False, feature_set='master', include_modern=False):
    from .config import load_config
    from .database import Database
    cfg = load_config()
    if target not in LABEL_COLUMNS:
        raise ValueError('Unknown target')
    features = json.loads((cfg['export_path'] / 'feature_columns.json').read_text())
    if feature_set == 'v1':
        import yaml
        from .config import ROOT
        selected=yaml.safe_load((ROOT/'config/model_features_v1.yaml').read_text())['features']
        features=[r['name'] for r in selected if include_modern or r['era']!='modern_optional']
    elif feature_set != 'master':
        raise ValueError('feature_set must be master or v1')
    db = Database(database_path or cfg['database_path'], read_only=True)
    try:
        clause = ' AND valid_starting_appearance' if starters_only else ''
        df = db.query(f'SELECT * FROM model_ready_pitcher_appearances WHERE training_eligible AND {target} IS NOT NULL' + clause)
    finally:
        db.close()
    return df[features], df[target].astype(int), df[['pitcher_id', 'game_id', 'game_date']]
