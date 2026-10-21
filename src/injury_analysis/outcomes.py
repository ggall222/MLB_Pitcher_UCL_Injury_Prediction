"""Descriptive severity, recurrence and warning timing; outcomes never enter predictors."""
import numpy as np
import pandas as pd
from .common import CFG, CACHE, TRAJECTORY_FEATURES, table, cluster_mean_ci


def severity_analysis(events):
    e = events.loc[events.target_eligible].copy()
    valid_duration = e.activation_match_valid & e.duration_observed & ~e.offseason_activation
    e['usable_duration'] = e.days_out_observed.where(valid_duration).astype(float)
    e['duration_group'] = np.select([e.usable_duration.le(30), e.usable_duration.between(31,90),
        e.usable_duration.gt(90)], ['Observed ≤30d', 'Observed 31–90d', 'Observed >90d'], default='Unresolved / administrative return')
    counts = e.groupby('analysis_group').agg(events=('injury_event_id','size'),
        censored=('right_censored','mean'), usable_durations=('usable_duration','count'),
        observed_median_days=('usable_duration','median'), long_il_share=('transferred_to_60_day','mean'),
        documented_surgery_share=('surgery_flag','mean')).reset_index()
    table(counts, 'severity_counts')
    table(e.groupby(['analysis_group','duration_group']).size().rename('events').reset_index(), 'severity_duration_groups')
    features = pd.read_parquet(CACHE / 'event_preinjury_features.parquet')
    joined = features.merge(e, left_on='event_id', right_on='injury_event_id', suffixes=('', '_event'))
    joined['surgery_documentation'] = np.where(joined.surgery_flag, 'Documented surgery', 'No surgery documentation')
    rows = []
    for group_field in ['duration_group','surgery_documentation']:
        for (category, group), f in joined.groupby(['analysis_group',group_field]):
            for feature in ['velo_z','extension_z','release_z_sd_z','workload_acceleration_7d_vs_21d']:
                rows.append(dict(category=category, comparison=group_field, subgroup=group, feature=feature,
                    **cluster_mean_ci(f,feature,cluster='cluster_pitcher_id')))
    summaries = table(pd.DataFrame(rows), 'severity_preinjury_features')
    table(joined[['event_id','player_id','analysis_group','duration_group','surgery_documentation',
        'usable_duration','right_censored','max_il_length','offseason_activation']], 'severity_event_cohort')
    return counts, summaries


def recurrence_analysis(a, events):
    e = events.loc[events.target_eligible].sort_values(['player_id','anchor_date','injury_event_id']).copy()
    e['qualifying_episode_number'] = e.groupby('player_id').cumcount() + 1
    e['prior_qualifying_date'] = e.groupby('player_id').anchor_date.shift()
    e['prior_qualifying_group'] = e.groupby('player_id').analysis_group.shift()
    e['prior_qualifying_region'] = e.groupby('player_id').initial_injury_region.shift()
    e['interval_days'] = (e.anchor_date - e.prior_qualifying_date).dt.days
    by_pitcher = {p: f.game_date.to_numpy() for p, f in a.groupby('pitcher_id')}
    e['observed_pitching_between'] = False
    for i, row in e.loc[e.prior_qualifying_date.notna()].iterrows():
        ds = by_pitcher.get(row.player_id, np.array([], dtype='datetime64[ns]'))
        e.at[i,'observed_pitching_between'] = ((ds > np.datetime64(row.prior_qualifying_date)) & (ds < np.datetime64(row.anchor_date))).any()
    e['repeat_status'] = np.where(e.qualifying_episode_number.eq(1), 'First observed qualifying episode',
        np.where(e.observed_pitching_between, 'Repeat after observed MLB pitching', 'Repeat without observed MLB pitching'))
    e['same_broad_group'] = e.analysis_group.eq(e.prior_qualifying_group)
    e['same_detailed_region'] = e.initial_injury_region.eq(e.prior_qualifying_region)
    repeats = e.loc[e.observed_pitching_between & e.interval_days.gt(0)].copy()
    table(e[['injury_event_id','player_id','anchor_date','analysis_group','initial_injury_region',
        'qualifying_episode_number','prior_qualifying_date','prior_qualifying_group','interval_days',
        'repeat_status','same_broad_group','same_detailed_region']], 'recurrence_events')
    transitions = pd.crosstab(repeats.prior_qualifying_group, repeats.analysis_group)
    table(transitions.reset_index(), 'recurrence_transitions_counts')
    table(transitions.div(transitions.sum(axis=1),axis=0).reset_index(), 'recurrence_transitions_fraction')
    table(repeats.groupby('analysis_group').agg(repeats=('injury_event_id','size'), pitchers=('player_id','nunique'),
        median_interval_days=('interval_days','median'), same_region_fraction=('same_detailed_region','mean')).reset_index(), 'recurrence_summary')
    features = pd.read_parquet(CACHE / 'event_preinjury_features.parquet')
    joined = features.merge(e[['injury_event_id','repeat_status']],left_on='event_id',right_on='injury_event_id')
    rows = []
    for (category,status), f in joined.groupby(['analysis_group','repeat_status']):
        for feature in ['velo_z','extension_z','release_z_sd_z','workload_acceleration_7d_vs_21d']:
            rows.append(dict(category=category,repeat_status=status,feature=feature,
                **cluster_mean_ci(f,feature,cluster='cluster_pitcher_id')))
    table(pd.DataFrame(rows), 'recurrence_preinjury_features')
    # Crude appearance risk by AS-OF prior reported placement count. Future recurrence ranks
    # are forbidden as predictors and are never used here. Not causal risk or survival rates.
    observed = a.loc[a.any_injury_30d.notna() & a.partition.isin(['train','validation','test'])].copy()
    observed['prior_episode_band'] = pd.cut(observed.prior_reported_il_count, [-1,0,1,3,np.inf], labels=['0','1','2–3','4+'])
    crude = observed.groupby(['prior_episode_band','is_starter'],observed=True).agg(
        appearances=('row_id','size'), positive_rows=('any_injury_30d','sum'),
        rate=('any_injury_30d','mean'), pitchers=('pitcher_id','nunique')).reset_index()
    table(crude, 'recurrence_crude_appearance_rates')
    return e, repeats, transitions, crude


def warning_timing(events):
    p = pd.read_parquet(CACHE / 'test_predictions.parquet')
    p = p.loc[p.horizon.eq(30)].merge(events[['injury_event_id','analysis_group']],
        left_on='next_event_id', right_on='injury_event_id',how='left')
    p = p.loc[p.category.eq(p.analysis_group) & p.next_event_lead_days.between(1,60)].copy()
    p['lead_bin'] = pd.cut(p.next_event_lead_days, [0,7,14,30,60], labels=['1–7','8–14','15–30','31–60'])
    means = p.groupby(['category','next_event_id','pitcher_id','lead_bin'],observed=True).score.mean().reset_index()
    rows = []
    for (category,bin_name), f in means.groupby(['category','lead_bin'],observed=True):
        rows.append(dict(category=category, lead_bin=bin_name, **cluster_mean_ci(f,'score')))
    return table(pd.DataFrame(rows),'event_risk_trajectories')
