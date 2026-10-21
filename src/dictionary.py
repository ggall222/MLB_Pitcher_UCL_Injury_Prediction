"""Generate a column-complete dictionary directly from persisted table schemas."""
import json
import re
import pandas as pd

TABLES = {
 'team_seasons': ('MLB /teams?season=', 'Season-specific team identity; avoids future franchise names on historical games.'),
 'historical_checkpoints': ('Pipeline', 'Season phase checkpoints; ingestion_state is authoritative for unresolved source failures.'),
 'arm_injury_episodes': ('MLB transactions + official appearances', 'Public arm IL episodes; documented activation followed by MLB appearance permits risk-set reentry.'),
 'injury_events': ('Roegele + MLB public evidence', 'One linked arm episode with UCL procedure confirmation; surgery-only onset remains NULL.'),
 'injury_event_procedures': ('Roegele + episode linkage', 'One source procedure mapped to its episode; multiple procedures can belong to one injury event.'),
 'aligned_pitch_tracking': ('Statcast + MLB game feed', 'Physical-pitch ordinal alignment with count equality and speed agreement within 0.5 mph; invalid matches have NULL IVB.'),
 'pitch_tracking': ('MLB /api/v1.1/game/{id}/feed/live', 'Supplemental physical pitch telemetry keyed by game + zero-based play index plus 1 + pitch number.'),
 'teams': ('MLB Stats API /teams', 'MLB team dimension; historical payload retained.'),
 'pitchers': ('MLB /people; Statcast; Roegele MLBAM IDs', 'Player identity dimension; can include surgery-listed non-pitchers and placeholders.'),
 'games': ('MLB /schedule', 'One scheduled MLB regular-season game; final and unfinished games distinguished.'),
 'pitches': ('Baseball Savant CSV', 'One source pitch/event, keyed by game + plate appearance + pitch number; all raw columns retained as JSON.'),
 'boxscore_appearances': ('MLB /game/{game_id}/boxscore', 'One pitcher per official MLB appearance, including outcome follow-up games.'),
 'roster_snapshots': ('MLB /teams/{team_id}/roster', 'Historical active roster at study start; never current-roster cohort selection.'),
 'transactions': ('MLB /transactions', 'All transactions, including hitters; player_id is not restricted to the pitcher dimension.'),
 'season_pitching': ('MLB /stats?stats=season&group=pitching', 'Completed prior-season totals; joined only to season t-1.'),
 'injuries': ('Jon Roegele workbook + MLB transactions', 'One source procedure; uncertain surgery dates and unmatched identities remain explicit.'),
 'appearances': ('Statcast + MLB boxscore + static MLB biography', 'One pitcher by appearance; contemporaneous measurements and quality flags.'),
 'appearance_pitch_metrics': ('Statcast pitch aggregation', 'One pitcher by appearance by pitch type; velocity, mechanics, spin, movement and usage.'),
 'workload_features': ('Historical MLB boxscores and prior-year /stats', 'Current plus strictly prior-day official workloads; histories exclude same-day other games.'),
 'mechanical_features': ('Historical appearance metrics', 'Prior 3/5/10 appearances, differences and slopes; current day excluded.'),
 'pitcher_baselines': ('Historical appearance metrics', 'At most 20 prior outings and at least 5 valid measurements; sample standard deviations.'),
 'injury_labels': ('Roegele-confirmed procedures + MLB public elbow-IL evidence', 'Outcomes and retrospective eligibility only; never features.'),
 'model_features': ('Joined engineered tables', 'One pitcher by appearance; use exported feature_columns.json for X and eligibility filters.'),
 'ingestion_state': ('Pipeline', 'Per-source/item checkpoint, counts, retry status and update timestamp.'),
 'ingestion_issues': ('Pipeline', 'Retained invalid/duplicate source rows and ingestion problems, with stable issue IDs.')}

EXACT = {
 'injury_event_id': ('Distinct UCL-linked public arm episode; separate procedures map to this event', 'identifier'),
 'next_injury_event_id': ('Next linked event used only for outcome/evaluation audit, never X', 'identifier'),
 'next_injury_id': ('Legacy alias of next_injury_event_id in historical builds', 'identifier'),
 'label_reference_date': ('First retrieved public arm-IL announcement in linked episode; unknown for surgery-only cases', 'date'),
 'first_reported_symptom_date': ('NULL unless symptom-specific evidence is retrieved; IL date is not silently substituted', 'date'),
 'last_healthy_appearance': ('NULL unless health is independently established; appearance alone is insufficient', 'date'),
 'reentry_date': ('First observed MLB appearance on/after documented IL activation; outcome eligibility only', 'date'),
 'starter_source': ('Official gamesStarted or official pitching-order fallback; never pitch count', 'category'),
 'starter_status': ('starter, relief or ambiguous after official-team-count and first-observed-pitcher checks', 'category'),
 'valid_starting_appearance': ('Regular-season unambiguous official start passing current appearance quality rules', 'boolean'),
 'documented_prior_ucl': ('At least one documented confirmed procedure with upper date bound strictly before this appearance; missing ascertainment is possible', 'boolean'),
 'documented_prior_ucl_count': ('Count of documented prior procedures, using conservative upper date bounds; not count of unique clinical episodes', 'count'),
 'observed_prior_mlb_starts': ('MLB starts strictly before this date within observed history; left-truncated at study start', 'count'),
 'observed_prior_mlb_appearances': ('MLB pitching appearances strictly before this date within observed history', 'count'),
 'throws_left': ('Left hand observed in this appearance; NULL for unavailable or conflicting hands', 'boolean'),
 'game_id': ('MLB gamePk identifier', 'identifier'), 'pitcher_id': ('MLBAM person identifier', 'identifier'),
 'batter_id': ('MLBAM batter identifier', 'identifier'), 'player_id': ('MLBAM person identifier; may be a hitter', 'identifier'),
 'game_date': ('Official game date; resumed games flagged as timing uncertain', 'date'),
 'at_bat_number': ('Plate appearance sequence within game', 'count'), 'pitch_number': ('Pitch sequence within plate appearance', 'count'),
 'raw_json': ('Complete source record, including unused/unsafe fields; JSON null means missing', 'JSON'),
 'pitch_result': ('Savant B=ball, S=strike, X=in play', 'category'),
 'description': ('Source pitch call or transaction narrative', 'text'),
 'release_speed': ('Out-of-hand tracked pitch speed; suspect values retained raw', 'mph'),
 'pfx_x': ('Savant horizontal movement, catcher perspective', 'feet'),
 'pfx_z': ('Savant vertical movement; not renamed as a separately validated IVB measurement', 'feet'),
 'alignment_valid': ('Equal physical pitch counts and measured source velocities agreeing within 0.5 mph, or one speed unavailable', 'boolean'),
 'mlb_pitch_number': ('Native MLB pitch number; excludes administrative events and can differ from canonical Savant pitch number', 'count'),
 'physical_ordinal': ('Sequence of physical pitches within this plate appearance after excluding administrative events', 'count'),
 'is_actual_pitch': ('False for automatic ball/strike administrative events; historical thrown intentional balls are physical pitches', 'boolean'),
 'is_hbp': ('Hit-by-pitch description or terminal event', 'boolean'),
 'is_starter': ('Official boxscore gamesStarted; first pitcher listed is fallback. Openers count as starters.', 'boolean'),
 'rest_days': ('Full calendar days without a pitching appearance since previous prior-day MLB outing; first observed is NULL', 'days'),
 'innings_pitched': ('Official outs recorded divided by 3; 5.2 baseball notation becomes 5.6667', 'innings'),
 'ivb_inches': ('Native MLB game-feed pitchData.breaks.breakVerticalInduced, averaged over measured physical pitches; distinct from Savant pfx_z', 'inches'),
 'zone_pct': ('Fraction of measured-zone pitches with Savant zone 1 through 9; missing zones excluded', 'fraction 0–1'),
 'strike_pct': ('Fraction of classified pitches with S or X result, including contact in play', 'fraction 0–1'),
 'first_pitch_strike_pct': ('Fraction of pitch_number=1 classified pitches with S or X result', 'fraction 0–1'),
 'walk_rate': ('Official walks divided by official batters faced', 'fraction 0–1'),
 'strikeout_rate': ('Official strikeouts divided by official batters faced', 'fraction 0–1'),
 'velo_drop_game': ('Last 20 measured FF/SI/FC velocities minus first 20; NULL unless at least 40 measured fastballs', 'mph'),
 'location_variability': ('sqrt(plate_x_sd^2 + plate_z_sd^2)', 'feet'),
 'age_at_appearance': ('Appearance date minus birth date divided by 365.2425; avoids present-day age', 'years'),
 'previous_season_innings': ('Complete official innings in calendar season t-1; missing player total stays NULL', 'innings'),
 'current_vs_previous_season_workload': ('Partial season-to-date innings divided by complete prior-season innings; not pace-adjusted', 'ratio'),
 'workload_acceleration_7d_vs_21d': ('Prior 7-day pitches/7 minus prior days 8–28 pitches/21', 'pitches/day'),
 'acute_chronic_pitch_ratio': ('Current-inclusive 7-day daily pitch rate divided by current-inclusive 28-day daily pitch rate', 'ratio'),
 'starter_role_prev_5': ('At least 3 of previous 5 prior-day appearances were starts; NULL before 5 appearances', 'boolean'),
 'known_onset_date': ('Earliest linked public elbow-IL announcement; proxy for public evidence, not biological onset', 'date'),
 'first_reported_elbow_date': ('Earliest retrieved linked MLB elbow-IL announcement; earlier media discomfort can be unknown', 'date'),
 'il_placement_date': ('Administrative effective date of linked IL transaction; can be retroactive', 'date'),
 'transaction_date': ('MLB transaction announcement date used for knowledge timing', 'date'),
 'effective_date': ('MLB administrative effective date; never silently substituted for announcement date', 'date'),
 'diagnosis_date': ('NULL unless a dedicated reliable diagnosis source is implemented', 'date'),
 'last_healthy_mlb_appearance': ('NULL: merely appearing does not establish that a pitcher was healthy', 'date'),
 'last_pre_onset_appearance': ('Last observed MLB outing before public evidence; not certified healthy and not a feature', 'date'),
 'surgery_date': ('Exact source surgery day only; first-of-month placeholders set to NULL', 'date'),
 'surgery_date_lower': ('Lower date bound implied by source day/month/year precision', 'date'),
 'surgery_date_upper': ('Upper date bound implied by source day/month/year precision', 'date'),
 'surgery_date_precision': ('day, month_or_exact_day_1, year_or_exact_jan_1, or unknown', 'category'),
 'return_date': ('Source same-level return day if exact; never a model feature or risk-set reentry rule', 'date'),
 'training_eligible': ('Quality-eligible and at risk before known/uncertain injury; target still must be nonnull', 'boolean'),
 'label_eligible': ('Retrospective at-risk eligibility with sufficient source retrieval; not a feature', 'boolean'),
 'label_sources_complete': ('Both configured Roegele tabs and MLB schedule/transaction/boxscore requests succeeded', 'boolean'),
 'risk_set_stop_date': ('Current active arm-IL episode start or next arm-IL episode start; NULL if none observed', 'date'),
 'post_known_injury': ('Appearance during a known active arm-IL episode before documented return; never a healthy negative', 'boolean'),
 'quality_eligible': ('Boxscore present, unambiguous game timing, >=80% tracked velocity and pitch counts within 2', 'boolean'),
 'repeat_procedure': ('A documented prior confirmed procedure has an upper date bound before this procedure lower date bound; overlapping uncertain dates do not establish repeat order', 'boolean'),
 'identity_method': ('Authoritative MLBAM, unique exact normalized name crosswalk, or unmatched; no fuzzy joins', 'category'),
 'confirmed_ucl_surgery': ('Procedure reported in Roegele and dated to at least a year; non-throwing-arm asterisk excluded', 'boolean')}


def describe(name):
    if name in EXACT:
        return EXACT[name]
    if name.startswith('ucl_injury_next_'):
        if name.endswith('_appearances'):
            return ('Confirmed public-evidence onset before the Nth subsequent appearance, capped at 90 days; NULL if censored', 'nullable boolean target')
        return ('Confirmed public-evidence onset in (appearance date, appearance date + N days]; negatives require complete follow-up; otherwise NULL', 'nullable boolean target')
    unit = 'numeric'
    if 'ivb_inches' in name or 'break_inches' in name: unit = 'inches'
    elif any(x in name for x in ['velo', 'speed']): unit = 'mph'
    elif 'spin_axis' in name or 'arm_angle' in name: unit = 'degrees'
    elif 'spin' in name: unit = 'rpm'
    elif any(x in name for x in ['release_x', 'release_z', 'release_pos', 'extension', 'plate_x', 'plate_z', 'movement']): unit = 'feet'
    elif 'innings' in name: unit = 'innings'
    elif any(x in name for x in ['pct', 'rate', 'coverage', 'share']): unit = 'fraction 0–1'
    elif any(x in name for x in ['pitches', 'count', 'appearances', 'starts', 'outs', 'walks', 'strikeouts', 'batters']): unit = 'count'
    if name.endswith('_z'): return ('Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8', 'standard deviations')
    if '_baseline_count' in name: return ('Valid observations in last 20 strictly prior-day outings for the named metric', 'count')
    if '_baseline_' in name: return ('Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations', unit)
    if '_prev_' in name and re.search(r'_prev_(3|5|10)$', name):
        return ('Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values', unit)
    if '_change_vs_' in name: return ('Current named appearance metric minus its previous-5-appearance mean', unit)
    if '_slope_prev_' in name: return ('OLS slope over previous 5 prior-day outings, at least 3 valid measurements; x is appearance index', unit + '/appearance')
    if name.startswith('pitches_prev_'): return ('Official pitches in [date-N days, date), excluding all current-day appearances', 'pitches')
    if name.startswith('innings_prev_'): return ('Official innings in [date-N days, date), excluding all current-day appearances', 'innings')
    if re.fullmatch(r'(pitches|innings|appearances)_\d+d', name): return ('Named workload in [date-N days,date) plus the current appearance; other same-day games omitted', unit)
    if name.startswith('season_'): return ('Named workload from January 1 through prior day plus current appearance', unit)
    if name.endswith('_max_date'): return ('Latest contributing historical source date; must precede prediction date; audit-only', 'date')
    if name.endswith('_date') or name.endswith('_at'): unit = 'date/timestamp'
    if name.startswith(('is_', 'has_')) or name.endswith('_complete'): unit = 'boolean'
    if name.endswith('_id'): unit = 'identifier'
    return (name.replace('_', ' ').capitalize() + '; see source/table methodology and builder for exact aggregation', unit)


def write_dictionary(db, cfg, features):
    rows = []
    for table, (source, transformation) in TABLES.items():
        for col in db.query(f'DESCRIBE "{table}"').itertuples():
            desc, unit = describe(col.column_name)
            engineered = table in ['appearances','appearance_pitch_metrics','workload_features','mechanical_features','pitcher_baselines','injury_labels','model_features']
            rows.append({'table': table, 'variable_name': col.column_name, 'data_type': col.column_type,
                         'nullable': col.null == 'YES', 'description': desc, 'source': source,
                         'transformation': transformation, 'unit': unit, 'kind': 'engineered' if engineered else 'source/metadata',
                         'safe_for_modeling': col.column_name in features and table != 'injury_labels'})
    # Document every extra raw Savant field, including known future-looking fields.
    raw = json.loads(db.scalar('SELECT raw_json FROM pitches LIMIT 1'))
    typed = set(db.columns('pitches'))
    for col in raw:
        desc, unit = describe(col)
        rows.append({'table': 'pitches.raw_json', 'variable_name': col, 'data_type': 'JSON value',
                     'nullable': True, 'description': desc, 'source': 'Baseball Savant CSV',
                     'transformation': 'Retained without feature selection; consult Savant csv-docs. Not allowlisted.',
                     'unit': unit, 'kind': 'raw', 'safe_for_modeling': False})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'docs/data_dictionary.csv', index=False)
    model = frame[frame.table.eq('model_features')]
    lines = ['# Model data dictionary', '', 'Complete table-level dictionary: [data_dictionary.csv](data_dictionary.csv).',
             'A safe feature can still be NULL or constant in a short sample. Fit imputation/selection only on the training fold.', '',
             '| Variable | Type | Unit | Feature | Definition |', '|---|---|---|---|---|']
    lines += [f'| {r.variable_name} | {r.data_type} | {r.unit} | {r.safe_for_modeling} | {r.description} |' for r in model.itertuples()]
    (ROOT / 'docs/data_dictionary.md').write_text('\n'.join(lines) + '\n')

from .config import ROOT
