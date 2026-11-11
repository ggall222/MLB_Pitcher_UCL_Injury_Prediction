"""Small, explicit paper-to-schema adapter; no changes to the master features."""
from pathlib import Path
import hashlib
import json

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = 'https://journals.sagepub.com/doi/full/10.1177/23259671261477532'
# (paper concept, output feature, existing source field, model correlation domain)
FEATURES = [
    ('Age', 'age_at_appearance', 'appearances.age_at_appearance', 'context'),
    ('Total pitches', 'total_pitches', 'appearances.total_pitches', 'workload'),
    ('Innings pitched', 'innings_pitched', 'appearances.innings_pitched', 'workload'),
    ('Batters faced', 'batters_faced', 'appearances.batters_faced', 'workload'),
    ('Mean velocity', 'avg_velocity', 'appearances.avg_velocity', 'velocity'),
    ('Maximum velocity', 'max_velocity', 'appearances.max_velocity', 'velocity'),
    ('Velocity SD (all pitch types)', 'kang_velocity_sd', 'pitches.release_speed', 'velocity'),
    ('Mean spin rate', 'spin_avg', 'appearances.spin_avg', 'spin'),
    ('Maximum spin rate', 'kang_spin_max', 'pitches.release_spin_rate', 'spin'),
    ('Spin rate SD', 'kang_spin_sd', 'pitches.release_spin_rate', 'spin'),
    ('Mean absolute horizontal release position', 'kang_release_abs_x_avg', 'pitches.release_pos_x', 'release'),
    ('Mean longitudinal release position', 'kang_release_y_avg', 'pitches.release_pos_y', 'release'),
    ('Mean vertical release position', 'release_z_avg', 'appearances.release_z_avg', 'release'),
    ('Mean release extension', 'extension_avg', 'appearances.extension_avg', 'release'),
] + [(name + ' fraction', code + '_usage_pct', 'appearances.' + code + '_usage_pct', 'pitch_mix')
     for name, code in [('Splitter', 'fs'), ('Slider', 'sl'), ('Four-seam fastball', 'ff'),
                        ('Sweeper', 'st'), ('Knuckle-curve', 'kc'), ('Cutter', 'fc'),
                        ('Sinker', 'si'), ('Curveball', 'cu'), ('Changeup', 'ch')]]
FEATURE_NAMES = [x[1] for x in FEATURES]
DOMAINS = {x[1]: x[3] for x in FEATURES}
EXTRA_SQL = {
    'kang_velocity_sd': 'stddev_samp(CASE WHEN release_speed BETWEEN 30 AND 110 THEN release_speed END)',
    'kang_spin_max': 'max(CASE WHEN release_spin_rate BETWEEN 0 AND 4500 THEN release_spin_rate END)',
    'kang_spin_sd': 'stddev_samp(CASE WHEN release_spin_rate BETWEEN 0 AND 4500 THEN release_spin_rate END)',
    'kang_release_abs_x_avg': 'avg(CASE WHEN release_pos_x BETWEEN -7 AND 7 THEN abs(release_pos_x) END)',
    'kang_release_y_avg': 'avg(CASE WHEN isfinite(release_pos_y) THEN release_pos_y END)',
}


def load_config(path=None):
    path = Path(path) if path else ROOT / 'config/kang_benchmark.yaml'
    cfg = yaml.safe_load(path.read_text())
    if not (pd.Timestamp('2017-01-01') <= pd.Timestamp(cfg['start_date']) <=
            pd.Timestamp(cfg['end_date']) <= pd.Timestamp('2025-12-31')):
        raise ValueError('Kang foundation is restricted to 2017–2025; 2026 is protected.')
    if cfg['primary_control_rule'] not in cfg['control_rules']:
        raise ValueError('Primary control rule must be explicitly audited.')
    if any(not isinstance(w, int) or not 7 <= w <= 49 for w in cfg['windows_days']):
        raise ValueError('Benchmark windows must be integer days within 7–49.')
    return cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def crosswalk(connection):
    dictionary = pd.read_csv(ROOT / 'docs/data_dictionary.csv').fillna('')
    inventory = pd.read_csv(ROOT / 'reports/feature_consolidation/tables/feature_inventory.csv').set_index('feature')
    schemas = {t: set(connection.execute('DESCRIBE ' + t).df().column_name)
               for t in ['appearances', 'pitches']}
    rows = []
    for concept, feature, source, domain in FEATURES:
        table, field = source.split('.')
        if field not in schemas[table]:
            raise ValueError('Missing required source field: ' + source)
        d = dictionary[(dictionary['table'] == table) & (dictionary.variable_name == field)]
        if d.empty:
            raise ValueError('Source has no dictionary entry: ' + source)
        is_new = feature in EXTRA_SQL
        group = (inventory.loc[feature, 'feature_group'] if feature in inventory.index else
                 {'velocity': 'Velocity', 'spin': 'Spin', 'release': 'Release position and variability'}[domain])
        note = 'Same concept; source cleaning and original author code may differ.'
        if feature == 'kang_velocity_sd':
            note += ' Do not substitute fastball-only fb_velo_sd; sample SD requires 2 measured pitches.'
        if feature == 'kang_spin_sd':
            note += ' Sample SD requires 2 measured pitches; paper SD denominator not specified.'
        if feature == 'kang_release_abs_x_avg':
            note += ' Absolute value before averaging; paper operation order unspecified.'
        if feature in ['kang_release_y_avg', 'extension_avg']:
            note += ' Paper tables name release Y but appendix/importance name extension; include both separately, identity unresolved.'
        if feature.endswith('_usage_pct'):
            note += ' Fraction 0–1 over all physical pitches; absent type is zero, not missing. Historical coding may change.'
        rows.append(dict(kang_concept=concept, feature=feature, source_variables=source,
                         project_feature_group=group, correlation_domain=domain,
                         availability='derived from existing raw field' if is_new else 'existing appearance feature',
                         transformation=EXTRA_SQL.get(feature, d.iloc[0].transformation),
                         dictionary_description=d.iloc[0].description, unit=d.iloc[0].unit,
                         timing_availability='before outing' if feature == 'age_at_appearance' else 'after completed outing',
                         current_appearance_included=feature != 'age_at_appearance',
                         match_status='approximate implementation; concept match',
                         notes=note, paper_locator='Methods; Tables 1–2; Appendix A1; Figure 4', paper_url=PAPER))
    return pd.DataFrame(rows)
