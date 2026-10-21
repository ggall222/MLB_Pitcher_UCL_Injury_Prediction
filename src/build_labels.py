"""Retrospective targets and risk-set eligibility; never feature inputs."""
import pandas as pd

DAY_WINDOWS = (14, 30, 60, 90)
APPEARANCE_WINDOWS = (3, 5)
LABEL_COLUMNS = [f'ucl_injury_next_{d}d' for d in DAY_WINDOWS] + [f'ucl_injury_next_{n}_appearances' for n in APPEARANCE_WINDOWS]


def build(appearances, injuries, transactions, box, cfg, sources_complete=True):
    injury_groups = {pid: g for pid, g in injuries.dropna(subset=['pitcher_id']).groupby('pitcher_id')}
    transaction_groups = {pid: g for pid, g in transactions.groupby('player_id')}
    box_groups = {pid: g.game_date.sort_values() for pid, g in box.groupby('pitcher_id')}
    start, observation_end = pd.Timestamp(cfg['start_date']), pd.Timestamp(cfg['observation_end'])
    rows = []
    for row in appearances.itertuples():
        day = row.game_date
        evidence = injury_groups.get(row.pitcher_id, injuries.iloc[:0])
        evidence = evidence[evidence.confirmed_ucl_surgery & evidence.position.fillna('P').eq('P')]
        relevant = evidence[evidence.surgery_date_upper.ge(start)]
        known = relevant[relevant.known_onset_date.notna()].sort_values('known_onset_date')
        t = transaction_groups.get(row.pitcher_id, transactions.iloc[:0])
        elbows = t[t.is_elbow_related & t.is_il_placement & t.transaction_date.le(observation_end)]
        stops = elbows[elbows.transaction_date.ge(start)].transaction_date.tolist()
        prior_elbows = elbows[elbows.transaction_date.lt(start)]
        if not prior_elbows.empty:
            latest = prior_elbows.transaction_date.max()
            activated = t[t.is_il_activation & t.transaction_date.between(latest, start, inclusive='neither')]
            if activated.empty:
                stops.append(latest)
        stops += known[known.known_onset_date.le(observation_end)].known_onset_date.tolist()
        stop = min(stops) if stops else pd.NaT
        future_events = known[known.known_onset_date.gt(day) & known.known_onset_date.le(observation_end)]
        next_event = future_events.iloc[0] if len(future_events) else None
        event_day = next_event.known_onset_date if next_event is not None else pd.NaT
        usable_event = pd.notna(event_day) and (pd.isna(stop) or event_day == stop)
        uncertain = relevant[relevant.known_onset_date.isna()]
        uncertain_starts = [r.surgery_date_lower - pd.Timedelta(days=cfg['uncertain_onset_lookback_days'])
                            for r in uncertain.itertuples() if pd.notna(r.surgery_date_lower)]
        observed = box_groups.get(row.pitcher_id, pd.Series(dtype='datetime64[ns]'))
        observed = observed[observed.le(observation_end)]
        last_seen = observed.max() if len(observed) else pd.NaT
        post_known = pd.notna(stop) and day >= stop
        uncertain_current = any(left <= day for left in uncertain_starts)
        eligible = bool(not post_known and not uncertain_current and sources_complete)
        out = {'game_id': row.game_id, 'pitcher_id': row.pitcher_id, 'label_eligible': eligible,
               'post_known_injury': post_known, 'uncertain_injury_timing': uncertain_current,
               'risk_set_stop_date': stop, 'next_known_ucl_onset': event_day,
               'next_injury_id': next_event.injury_id if next_event is not None else None,
               'last_observed_appearance': last_seen, 'label_observation_end': observation_end,
               'label_sources_complete': sources_complete,
               'label_onset_basis': next_event.onset_basis if next_event is not None else None}
        for days in DAY_WINDOWS:
            horizon = day + pd.Timedelta(days=days)
            val = None
            ambiguous = any(left <= horizon for left in uncertain_starts)
            interrupted = pd.notna(stop) and stop <= horizon
            if eligible:
                if usable_event and day < event_day <= horizon and not ambiguous:
                    val = True
                elif horizon <= observation_end and pd.notna(last_seen) and horizon <= last_seen and not interrupted and not ambiguous:
                    val = False
            out[f'ucl_injury_next_{days}d'] = val
        for n in APPEARANCE_WINDOWS:
            val = None
            cap = min(day + pd.Timedelta(days=cfg['followup_days']), observation_end)
            future = observed[(observed > day) & (observed <= cap)]
            event_count = int(((observed > day) & (observed < event_day)).sum()) if pd.notna(event_day) else None
            ambiguous = any(left <= cap for left in uncertain_starts)
            if eligible and not ambiguous:
                if usable_event and event_day <= cap and event_count < n:
                    val = True
                elif len(future) >= n:
                    nth = future.iloc[n - 1]
                    if (pd.isna(stop) or nth < stop) and (pd.isna(event_day) or nth < event_day):
                        val = False
            out[f'ucl_injury_next_{n}_appearances'] = val
        rows.append(out)
    result = pd.DataFrame(rows)
    for col in LABEL_COLUMNS + ['label_eligible', 'post_known_injury', 'uncertain_injury_timing', 'label_sources_complete']:
        result[col] = result[col].astype('boolean')
    for col in ['risk_set_stop_date', 'next_known_ucl_onset', 'last_observed_appearance', 'label_observation_end']:
        result[col] = pd.to_datetime(result[col])
    return result
