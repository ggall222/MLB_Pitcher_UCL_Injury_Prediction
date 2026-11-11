"""Published-design cohort adapter, separate from conservative UCL risk sets."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from .kang_cohort import load_inputs, bounded_episodes
from .kang_spec import ROOT, FEATURE_NAMES, DOMAINS

# Appendix A1 specifies extension; longitudinal Y appears only in other tables.
PREDICTORS = [f for f in FEATURE_NAMES if f != 'kang_release_y_avg']


def config(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / 'config/kang_replication.yaml').read_text())
    if cfg['start_date'] != '2017-01-01' or cfg['end_date'] != '2025-12-31':
        raise ValueError('Replication follows the paper\'s 2017–2025 study period.')
    if cfg['primary_control_rule'] != 'no_shoulder_elbow_il' or cfg['outside_window_case_games'] != 'negative' or cfg['features'] != 'appendix_a1':
        raise ValueError('This frozen replication uses Methods controls, earlier-case negatives and Appendix A1 inputs.')
    return cfg


def season_screen(a, cfg):
    """Publication thresholds; no current-position or project quality filter."""
    f = a.copy()
    if f.duplicated(['game_id', 'pitcher_id']).any():
        raise ValueError('Duplicate pitcher-game keys')
    f['missing_game'] = f[PREDICTORS].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    f['start'] = f.is_starter.eq(True).astype(int)
    f['relief'] = f.is_starter.eq(False).astype(int)
    f['start_outs'] = f.outs_recorded.where(f.is_starter.eq(True), 0)
    f['relief_outs'] = f.outs_recorded.where(f.is_starter.eq(False), 0)
    f['unknown_role'] = f.is_starter.isna()
    s = f.groupby(['pitcher_id', 'season']).agg(games=('game_id', 'size'),
        missing_fraction=('missing_game', 'mean'), starts=('start', 'sum'), relief=('relief', 'sum'),
        start_outs=('start_outs', 'sum'), relief_outs=('relief_outs', 'sum'), unknown_role=('unknown_role', 'any')).reset_index()
    s['role'] = np.where(s.starts > s.relief, 'starter', 'reliever')
    ties = s.starts.eq(s.relief)
    s.loc[ties, 'role'] = np.where(s.loc[ties, 'start_outs'] > s.loc[ties, 'relief_outs'], 'starter', 'reliever')
    s.loc[(ties & s.start_outs.eq(s.relief_outs)) | s.unknown_role, 'role'] = 'unavailable'
    s['fewer_than_five'] = s.games.lt(cfg['minimum_season_appearances'])
    s['excess_missingness'] = s.missing_fraction.gt(cfg['maximum_season_missing_fraction'])
    s['eligible'] = ~s.fewer_than_five & ~s.excess_missingness & s.role.ne('unavailable')
    return s


def build_cohorts(a, events, cfg):
    s = season_screen(a, cfg)
    e = events.rename(columns={'player_id': 'pitcher_id'}).copy()
    e['transaction_date'] = pd.to_datetime(e.transaction_date)
    e['season'] = e.transaction_date.dt.year
    e['endpoint'] = e[cfg['anatomy_field']].map({'Elbow / UCL': 'elbow', 'Shoulder': 'shoulder'})
    bounds = a.groupby('season').game_date.agg(['min', 'max'])
    e = e.join(bounds, on='season')
    e['during_season'] = e.transaction_date.between(e['min'], e['max'])
    placements = e[e.placement_transaction_id.notna() & e.during_season].copy()
    first = placements[placements.endpoint.notna()].sort_values(
        ['pitcher_id', 'transaction_date', 'injury_event_id']).drop_duplicates(['pitcher_id', 'season'])
    base = a.merge(s[['pitcher_id', 'season', 'eligible', 'role']], on=['pitcher_id', 'season'], validate='many_to_one')
    base = base[base.eligible].copy()
    base = base.merge(first[['pitcher_id', 'season', 'injury_event_id', 'endpoint', 'transaction_date']],
                      on=['pitcher_id', 'season'], how='left', validate='many_to_one')
    base['case_season'] = base.injury_event_id.notna()
    base['lead_days'] = (base.transaction_date - base.game_date).dt.days
    # Only publication-defined exclusions; return/overlap/timing QA is descriptive.
    base = base[~base.case_season | base.lead_days.gt(0)].copy()
    any_il = set(map(tuple, placements[['pitcher_id', 'season']].drop_duplicates().to_numpy()))
    unknown_il = set(map(tuple, placements.loc[placements.injury_region.eq('Unknown / Unspecified'),
                                               ['pitcher_id', 'season']].drop_duplicates().to_numpy()))
    base['any_il'] = [tuple(x) in any_il for x in base[['pitcher_id', 'season']].to_numpy()]
    base['unknown_il'] = [tuple(x) in unknown_il for x in base[['pitcher_id', 'season']].to_numpy()]
    cohorts, counts = {}, []
    for endpoint in ['elbow', 'shoulder']:
        f = base[~base.case_season | base.endpoint.eq(endpoint)].copy()
        # Paper's prior-region history sensitivity: previous seasons, study years only.
        history = placements[placements.endpoint.eq(endpoint)].groupby('pitcher_id').season.min()
        f['prior_region_il'] = f.pitcher_id.map(history).lt(f.season).astype(int)
        for role in ['starter', 'reliever']:
            c = f[f.role.eq(role)].sort_values(['game_date', 'game_id', 'pitcher_id']).reset_index(drop=True)
            cohorts[endpoint + '_' + role] = c
            for definition in ['no_shoulder_elbow_il', 'no_any_il']:
                z = c if definition == 'no_shoulder_elbow_il' else c[c.case_season | ~c.any_il]
                counts.append(dict(endpoint=endpoint, role=role, control_definition=definition, games=len(z),
                    unique_pitchers=z.pitcher_id.nunique(), pitcher_seasons=len(z[['pitcher_id','season']].drop_duplicates()),
                    events=z.injury_event_id.nunique(), control_seasons=len(z.loc[~z.case_season,['pitcher_id','season']].drop_duplicates()),
                    unknown_diagnosis_control_seasons=len(z.loc[~z.case_season & z.unknown_il,['pitcher_id','season']].drop_duplicates())))
    return cohorts, s, first, pd.DataFrame(counts)


def prepare(cfg):
    out, cache = ROOT / cfg['output'], ROOT / cfg['cache']
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    a, _, mapping, manifest = load_inputs(cfg)  # existing SQL summaries; no new predictors
    with duckdb.connect(str(ROOT / cfg['database']), read_only=True) as con:
        tx = con.execute('''SELECT * FROM read_parquet(?) WHERE included_primary_population
            AND transaction_date BETWEEN DATE '2015-01-01' AND DATE '2025-12-31' ''',
            [str(ROOT / cfg['injury_transactions'])]).df()
        observed = con.execute("SELECT pitcher_id,game_date FROM appearances WHERE game_type='R' AND NOT timing_uncertain AND game_date BETWEEN DATE '2015-01-01' AND DATE '2025-12-31'").df()
    e = bounded_episodes(tx, observed, cfg)
    e = e[pd.to_datetime(e.transaction_date).between(cfg['start_date'], cfg['end_date'])].copy()
    e.to_parquet(cache / 'replication_episodes.parquet', index=False)
    cohorts, seasons, first, counts = build_cohorts(a, e, cfg)
    seasons.to_csv(cache / 'season_eligibility.csv', index=False)
    first.to_parquet(cache / 'first_target_events.parquet', index=False)
    counts.to_csv(out / 'tables/cohort_counts.csv', index=False)
    columns = ['game_id','pitcher_id','season','game_date','injury_event_id','lead_days','case_season','prior_region_il'] + PREDICTORS
    for task, f in cohorts.items():
        (cache / task).mkdir(exist_ok=True)
        f[columns].to_csv(cache / task / 'cohort.csv', index=False)
    descriptive=[]
    for task,f in cohorts.items():
        for feature in PREDICTORS:
            descriptive.append(dict(task=task,feature=feature,observed_games=int(f[feature].notna().sum()),
                                    mean=float(f[feature].mean()),sd=float(f[feature].std())))
    pd.DataFrame(descriptive).to_csv(out/'tables/descriptive_game_metrics.csv',index=False)
    flow=[]
    for stage, f in [('raw_pitching',a),('five_appearances',a.merge(seasons.query('not fewer_than_five')[['pitcher_id','season']],on=['pitcher_id','season'])),
                     ('missingness_and_role',a.merge(seasons.query('eligible')[['pitcher_id','season']],on=['pitcher_id','season']))]:
        flow.append(dict(stage=stage,games=len(f),pitcher_seasons=len(f[['pitcher_id','season']].drop_duplicates()),unique_pitchers=f.pitcher_id.nunique()))
    pd.DataFrame(flow).to_csv(out/'tables/cohort_flow.csv',index=False)
    missing=[]
    for year,g in a.groupby('season'):
        for feature in PREDICTORS:
            missing.append(dict(season=year,feature=feature,games=len(g),missing_games=int(g[feature].isna().sum()),
                                missing_fraction=float(g[feature].isna().mean())))
    pd.DataFrame(missing).to_csv(out/'tables/feature_missingness.csv',index=False)
    seasons.groupby('season').agg(pitcher_seasons=('pitcher_id','size'),eligible=('eligible','sum'),
        excluded_fewer_five=('fewer_than_five','sum'),excluded_missing=('excess_missingness','sum')).to_csv(out/'tables/exclusions_by_season.csv')
    mapping['used_in_model'] = mapping.feature.isin(PREDICTORS)
    mapping['match_quality'] = np.where(mapping.feature.str.startswith('kang_') | mapping.feature.eq('extension_avg'), 'close proxy','exact')
    mapping[['kang_concept','feature','match_quality','used_in_model','notes','correlation_domain','source_variables']].to_csv(out/'tables/feature_crosswalk.csv',index=False)
    support=[]
    for task, f in cohorts.items():
        for w in range(cfg['window_min'], cfg['window_max']+1, cfg['window_step']):
            positive=f.case_season & f.lead_days.between(1,w)
            support.append(dict(task=task,window=w,games=len(f),positive_games=int(positive.sum()),represented_events=f.loc[positive,'injury_event_id'].nunique()))
    pd.DataFrame(support).to_csv(out/'tables/window_support.csv',index=False)
    qa=[]
    for endpoint,g in first.groupby('endpoint'):
        seen=set(pd.concat([f for task,f in cohorts.items() if task.startswith(endpoint)]).injury_event_id.dropna())
        qa.append(dict(endpoint=endpoint,first_placements=len(g),events_contributing_games=len(seen),
            overlap_flags=int(g.overlapping_episode.sum()),timing_flags=int(g.timing_uncertain.sum()),
            multiple_anatomy_flags=int(g.multiple_injury_regions.sum())))
    pd.DataFrame(qa).to_csv(out/'tables/event_qa_descriptive.csv',index=False)
    return cohorts, manifest
