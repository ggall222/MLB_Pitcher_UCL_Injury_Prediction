"""Event-weighted pre-injury trajectories and deterministic matched controls."""
import numpy as np
import pandas as pd
from .common import CFG, CACHE, ARM, NONARM, TRAJECTORY_FEATURES, table, cluster_mean_ci

CALENDAR_BINS = [1, 8, 15, 31, 61]
CALENDAR_LABELS = ['1–7 days', '8–14 days', '15–30 days', '31–60 days']


def window(rows, anchor, event_id, group, cluster_id, kind, features):
    f = rows.loc[rows.game_date.lt(anchor) & rows.game_date.ge(anchor - pd.Timedelta(days=60))
                 & rows.analysis_eligible].sort_values(['game_date', 'game_id']).copy()
    f['lead_days'] = (anchor - f.game_date).dt.days
    f['appearance_before'] = np.arange(len(f), 0, -1)
    f['calendar_bin'] = pd.cut(f.lead_days, CALENDAR_BINS, right=False, labels=CALENDAR_LABELS).astype(str)
    f['event_id'] = event_id
    f['analysis_group'] = group
    f['cluster_pitcher_id'] = cluster_id
    f['kind'] = kind
    return f[['event_id', 'analysis_group', 'cluster_pitcher_id', 'pitcher_id', 'game_date', 'game_id',
              'is_starter', 'lead_days', 'appearance_before', 'calendar_bin', 'kind'] + features]


def build_trajectories(a, events):
    features = [f for f in TRAJECTORY_FEATURES if f in a]
    by_pitcher = {pid: f for pid, f in a.groupby('pitcher_id')}
    eligible = a.analysis_eligible & a.followup_60d
    eligible &= a.next_event_lead_days.isna() | a.next_event_lead_days.gt(90)
    eligible &= a.days_since_any_event.isna() | a.days_since_any_event.gt(60)
    pool = a.loc[eligible].copy()
    pool['month'] = pool.game_date.dt.month
    pool['age_band'] = (pool.age_at_appearance // 5).fillna(-1)
    pool['experience_band'] = pd.cut(pool.prior_appearance_count, [0, 20, 100, np.inf], labels=[0, 1, 2]).astype(int)
    rng = np.random.default_rng(CFG['seed'])
    # Sample without reusing a control pitcher: a more conservative effective control N,
    # and pitcher-cluster uncertainty remains meaningful for paired differences.
    used_control_pitchers = set()
    records, matches, exclusions = [], [], []
    ordered = events.loc[events.target_eligible].sort_values(['anchor_date', 'injury_event_id'])
    # Randomize event processing deterministically so early seasons do not monopolize controls.
    ordered = ordered.iloc[rng.permutation(len(ordered))]
    for event in ordered.itertuples():
        rows = by_pitcher.get(event.player_id)
        if rows is None:
            exclusions.append((event.injury_event_id, 'no MLB pitcher appearance history'))
            continue
        case = window(rows, event.anchor_date, event.injury_event_id, event.analysis_group, event.player_id, 'Injury', features)
        if len(case) < 3 or case.lead_days.min() > 14:
            exclusions.append((event.injury_event_id, 'fewer than 3 prior outings in 60d or no outing within 14d'))
            continue
        last = rows.loc[rows.game_id.eq(case.iloc[-1].game_id)].iloc[0]
        age_band = int(last.age_at_appearance // 5) if pd.notna(last.age_at_appearance) else -1
        experience_band = 0 if last.prior_appearance_count <= 20 else (1 if last.prior_appearance_count <= 100 else 2)
        candidates = pool.loc[pool.season.eq(last.season) & pool.month.eq(last.game_date.month)
            & pool.is_starter.eq(last.is_starter) & pool.age_band.eq(age_band)
            & pool.experience_band.eq(experience_band) & pool.pitcher_id.ne(event.player_id)
            & ~pool.pitcher_id.isin(used_control_pitchers)]
        control = None
        for pos in rng.permutation(len(candidates)):
            candidate = candidates.iloc[pos]
            pseudo = candidate.game_date + pd.Timedelta(days=int(case.lead_days.min()))
            ctrl = window(by_pitcher[candidate.pitcher_id], pseudo, event.injury_event_id,
                          event.analysis_group, event.player_id, 'Control', features)
            # Matching the index outing, not silently including subsequent control outings.
            ctrl = ctrl.loc[ctrl.game_date.le(candidate.game_date)].copy()
            ctrl['appearance_before'] = np.arange(len(ctrl), 0, -1)
            if len(ctrl) >= 3:
                control = (candidate, pseudo, ctrl)
                break
        records.append(case)
        if control is None:
            exclusions.append((event.injury_event_id, 'no unused matched control pitcher'))
            continue
        candidate, pseudo, ctrl = control
        records.append(ctrl)
        used_control_pitchers.add(candidate.pitcher_id)
        matches.append({'event_id': event.injury_event_id, 'pitcher_id': event.player_id,
            'control_pitcher_id': candidate.pitcher_id, 'anchor_date': event.anchor_date,
            'control_anchor_date': pseudo, 'analysis_group': event.analysis_group,
            'season': int(last.season), 'month': int(last.game_date.month), 'is_starter': bool(last.is_starter),
            'age_band': age_band, 'experience_band': experience_band,
            'case_outings': len(case), 'control_outings': len(ctrl), 'last_appearance_lead': int(case.lead_days.min())})
    aligned = pd.concat(records, ignore_index=True)
    matched = pd.DataFrame(matches)
    aligned.to_parquet(CACHE / 'aligned_trajectories.parquet', index=False)
    table(matched, 'matched_controls')
    table(pd.DataFrame(exclusions, columns=['event_id', 'reason']), 'trajectory_exclusions')
    # Paired comparison restricted to matched events, not all cases vs a convenient control pool.
    paired = aligned.loc[aligned.event_id.isin(matched.event_id)].copy()
    event_means = paired.groupby(['event_id', 'analysis_group', 'cluster_pitcher_id', 'kind', 'calendar_bin'])[features].mean().reset_index()
    event_means.to_parquet(CACHE / 'trajectory_event_means.parquet', index=False)
    summaries = []
    for (group, kind, bin_name), f in event_means.groupby(['analysis_group', 'kind', 'calendar_bin']):
        for feature in features:
            summaries.append(dict(analysis_group=group, kind=kind, window=bin_name, feature=feature,
                **cluster_mean_ci(f.rename(columns={'cluster_pitcher_id': 'pitcher_id'}), feature)))
    table(pd.DataFrame(summaries), 'calendar_trajectory_summary')
    relative = paired.loc[paired.appearance_before.le(5)]
    summaries = []
    for (group, kind, index), f in relative.groupby(['analysis_group', 'kind', 'appearance_before']):
        for feature in features:
            summaries.append(dict(analysis_group=group, kind=kind, appearance_before=index, feature=feature,
                **cluster_mean_ci(f, feature, cluster='cluster_pitcher_id')))
    table(pd.DataFrame(summaries), 'appearance_trajectory_summary')
    # Last 14-day event means give one value per episode; paired differences retain pitcher clustering.
    near = paired.loc[paired.lead_days.le(14)].groupby(['event_id', 'analysis_group', 'cluster_pitcher_id', 'kind'])[features].mean().reset_index()
    cases = near.loc[near.kind.eq('Injury')].drop(columns='kind')
    controls = near.loc[near.kind.eq('Control')].drop(columns='kind')
    pairs = cases.merge(controls, on=['event_id', 'analysis_group', 'cluster_pitcher_id'], suffixes=('_case', '_control'))
    effects = []
    for feature in features:
        pairs[feature] = pairs[feature + '_case'] - pairs[feature + '_control']
    for group, f in pairs.groupby('analysis_group'):
        for feature in features:
            effects.append(dict(analysis_group=group, feature=feature,
                **cluster_mean_ci(f, feature, cluster='cluster_pitcher_id')))
    table(pd.DataFrame(effects), 'paired_preinjury_effects')
    anatomy = events[['injury_event_id','body_group','throwing_arm_injury']]
    side_pairs = pairs.merge(anatomy,left_on='event_id',right_on='injury_event_id')
    side_pairs['comparison_group'] = np.where(side_pairs.body_group.eq('Non-arm'), 'Non-arm (trunk/lower body)',
        np.where(side_pairs.throwing_arm_injury.fillna(False), 'Explicit throwing-side arm', 'Other or uncertain side'))
    body_effects = []
    for group, f in side_pairs.groupby('comparison_group'):
        for feature in features:
            body_effects.append(dict(group=group,feature=feature,
                **cluster_mean_ci(f,feature,cluster='cluster_pitcher_id')))
    table(pd.DataFrame(body_effects), 'body_group_preinjury_effects')
    pairs.to_parquet(CACHE / 'paired_event_features.parquet', index=False)
    # All adequately observed case events retained for severity/recurrence, independently of controls.
    all_cases = aligned.loc[aligned.kind.eq('Injury') & aligned.lead_days.le(14)]
    event_features = all_cases.groupby(['event_id', 'analysis_group', 'cluster_pitcher_id'])[features].mean().reset_index()
    event_features.to_parquet(CACHE / 'event_preinjury_features.parquet', index=False)
    return aligned, matched, pd.DataFrame(effects)
