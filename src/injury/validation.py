"""Assertions, population audits and reproducible source-text review samples."""
import json
import pandas as pd

from .acquire import save_frame, atomic_json
from .taxonomy import UNKNOWN, REGIONS


def validate_and_report(raw, normalized, linked, events, links, comparison, matches, status, cfg):
    report = cfg['report_path']
    report.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, violations):
        checks.append(dict(check=name, violations=int(violations), passed=int(violations) == 0))

    check('duplicate_transaction_ids', raw.transaction_id.duplicated().sum())
    check('duplicate_event_ids', events.injury_event_id.duplicated().sum())
    check('missing_player_ids', events.player_id.isna().sum())
    check('missing_event_anchor_transactions', ~events.anchor_transaction_id.isin(raw.transaction_id).all())
    check('missing_transaction_links', ~links.transaction_id.isin(raw.transaction_id).all())
    check('transaction_linked_to_multiple_episodes', (links.groupby('transaction_id').injury_event_id.nunique() > 1).sum())
    check('event_without_supporting_transaction', (~events.injury_event_id.isin(links.injury_event_id)).sum())
    included = linked[linked.included_primary_population]
    check('unaccounted_initial_placement', (~included.loc[included.il_action.eq('placement'),'transaction_id'].isin(links.transaction_id)).sum())
    check('unaccounted_il_transfer', (~included.loc[included.il_action.eq('transfer'),'transaction_id'].isin(links.transaction_id)).sum())
    anchor_players = linked.set_index('transaction_id').player_id
    check('event_player_differs_from_anchor_transaction', events.player_id.ne(events.anchor_transaction_id.map(anchor_players)).sum())
    check('invalid_nonpositive_duration', (events.days_out_observed.dropna() <= 0).sum())
    check('censored_with_duration', (events.right_censored & events.days_out_observed.notna()).sum())
    check('valid_activation_without_date', (events.activation_match_valid & events.activation_date.isna()).sum())
    check('activation_not_after_placement', (events.activation_match_valid & (events.activation_date <= events.transaction_date)).sum())
    check('future_event_date_after_announcement', (events.injury_event_date > events.transaction_date).sum())
    check('raw_resolution_used_as_return_without_activation', (events.activation_date.notna() & events.activation_transaction_id.isna()).sum())
    check('new_episode_anchored_by_activation', events.anchor_transaction_id.isin(linked.loc[linked.il_action.eq('activation'), 'transaction_id']).sum())
    check('non_mlb_or_nonpitcher_in_primary', (~events.anchor_transaction_id.isin(linked.loc[linked.included_primary_population, 'transaction_id'])).sum())
    check('event_transaction_outside_request_window', (~events.transaction_date.between(pd.Timestamp(cfg['start_date']), pd.Timestamp(cfg['end_date']))).sum())
    check('nonexistent_injury_region', (~events.injury_region.isin(REGIONS)).sum())
    check('activation_reused_between_episodes', events.activation_transaction_id.dropna().duplicated().sum())
    delta = (events.activation_date - events.injury_event_date).dt.days
    check('duration_formula_mismatch', (events.duration_observed & events.days_out_observed.ne(delta)).sum())
    check('orphan_transfer_with_invented_placement_date', (events.orphan_transfer & events.injury_event_date.notna()).sum())
    check('orphan_transfer_with_invented_placement_id', (events.orphan_transfer & events.placement_transaction_id.notna()).sum())
    check('primary_diagnosis_known_proxy_before_initial_announcement', (events.diagnosis_known_at < events.public_known_date).sum())
    check('source_count_inconsistent', events.source_count.ne(events.injury_source_secondary.fillna('').apply(lambda s: 1+len(set(s.split(';'))-{''}))).sum())
    frame = pd.DataFrame(checks)
    save_frame(frame, report / 'validation_checks.csv')
    if not frame.passed.all():
        raise RuntimeError('Injury validation failed: ' + ', '.join(frame.loc[~frame.passed, 'check']))
    n = len(events)
    complete = all(s['status'] == 'ok' for s in status.values())
    metrics = {
        'raw_transactions': len(raw), 'raw_il_related_transactions': int(normalized.il_related.sum()),
        'initial_placements_all_positions': int(normalized.il_action.eq('placement').sum()),
        'pitcher_il_transactions': int(linked.included_primary_population.sum()),
        'pitcher_initial_placements': int((linked.included_primary_population & linked.il_action.eq('placement')).sum()),
        'canonical_events': n, 'unique_pitchers': events.player_id.nunique(),
        'events_with_statcast_history': int(events.has_statcast_history.sum()),
        'unknown_region_events': int(events.injury_region.eq(UNKNOWN).sum()),
        'unknown_region_percentage': 100 * events.injury_region.eq(UNKNOWN).mean(),
        'unknown_diagnosis_percentage': 100 * events.injury_diagnosis.eq(UNKNOWN).mean(),
        'laterality_coverage_percentage': 100 * events.laterality.ne('Unknown').mean(),
        'matched_activation_percentage': 100 * events.activation_match_valid.mean(),
        'right_censored_percentage': 100 * events.right_censored.mean(),
        'orphan_transfers': int(events.orphan_transfer.sum()),
        'inferred_longer_il_continuations': int(events.inferred_continuation_count.sum()),
        'duplicate_placement_records_consolidated': int(events.duplicate_placement_count.sum()),
        'overlapping_episode_percentage': 100 * events.overlapping_episode.mean(),
        'review_required_percentage': 100 * events.qa_review_required.mean(),
        'extreme_duration_events': int(events.extreme_duration.sum()),
        'duplicate_event_percentage': 100 * events.injury_event_id.duplicated().mean(),
        'supplemental_confirmation_percentage': 100 * ((events.source_count > 1) & ~events.source_conflict).mean(),
        'supplemental_linkage_percentage': 100 * (events.source_count > 1).mean(),
        'source_conflict_events': int(events.source_conflict.sum()),
        'existing_ucl_in_both': int(comparison.comparison_status.eq('both').sum()),
        'existing_ucl_only': int(comparison.comparison_status.eq('existing_ucl_only').sum()),
        'general_explicit_ucl_only': int(comparison.comparison_status.eq('general_ucl_only').sum()),
        'monthly_windows_requested': len(status), 'monthly_windows_failed': sum(s['status'] != 'ok' for s in status.values()),
        'all_transaction_windows_available': int(complete), 'automatic_validation_checks_passed': len(checks),
    }
    duration = events.days_out_observed.dropna().astype(float).describe(percentiles=[.05, .25, .5, .75, .9, .95, .99])
    for name, value in duration.items():
        metrics['days_out_observed_' + str(name)] = value
    metrics = {k: v.item() if hasattr(v,'item') else v for k,v in metrics.items()}
    save_frame(pd.DataFrame([dict(metric=k, value=v) for k, v in metrics.items()]), report / 'mlb_pitcher_injury_validation_summary.csv')
    for column, name in [('injury_region', 'events_by_region'), ('injury_diagnosis', 'events_by_diagnosis')]:
        save_frame(events.groupby(column).size().rename('events').reset_index(), report / (name + '.csv'))
    by_year = events.assign(season=events.injury_event_date.fillna(events.transaction_date).dt.year).groupby('season').agg(
        events=('injury_event_id', 'size'), pitchers=('player_id', 'nunique'),
        matched_activations=('activation_match_valid', 'sum'), right_censored=('right_censored', 'sum'),
        supplemental_ucl=('supplemental_ucl_confirmed', 'sum'))
    save_frame(by_year.reset_index(), report / 'events_by_season.csv')
    save_frame(events[events.extreme_duration].sort_values('days_out_observed', ascending=False), report / 'extreme_durations.csv')
    save_frame(events[events.overlapping_episode | events.duplicate_placement_count.gt(0)], report / 'overlapping_and_duplicate_placements.csv')
    save_frame(events[events.source_conflict], report / 'source_conflicts.csv')
    save_frame(events[events.timing_uncertain], report / 'timing_uncertain_events.csv')
    save_frame(comparison.groupby(['comparison_status','comparison_reason'],dropna=False).size().rename('records').reset_index(), report / 'ucl_comparison_reason_counts.csv')
    save_frame(events[events.injury_region.eq(UNKNOWN) | events.injury_diagnosis.eq(UNKNOWN)],
               report / 'mlb_pitcher_injury_parser_unknowns.csv')
    samples = []
    for region, g in events.groupby('injury_region'):
        sample = g.sort_values('injury_event_id').head(3).copy()
        sample['review_stratum'] = 'region: ' + region
        samples.append(sample)
    for label, g in [('unknown', events[events.injury_region.eq(UNKNOWN)]),
                     ('extreme_duration', events[events.extreme_duration].sort_values('days_out_observed', ascending=False)),
                     ('overlap', events[events.overlapping_episode]), ('source_conflict', events[events.source_conflict]),
                     ('UCL', events[events.explicit_ucl | events.supplemental_ucl_confirmed])]:
        sample = g.head(8).copy()
        sample['review_stratum'] = label
        samples.append(sample)
    sample = pd.concat(samples, ignore_index=True)
    sample['review_status'] = 'pending_text_review'
    review_path = cfg['output_path'] / 'interim/manual_review_annotations.csv'
    if review_path.exists():
        reviews = pd.read_csv(review_path)
        sample = sample.drop(columns='review_status').merge(reviews, on='injury_event_id', how='left', validate='many_to_one')
    save_frame(sample, report / 'manual_review_samples.csv')
    atomic_json(report / 'validation_summary.json', metrics)
    lines = ['# General MLB pitcher injury build', '',
             f"Observation window: {cfg['start_date']} through {cfg['end_date']} (2026 is incomplete).", '',
             f"Collected {len(raw):,} distinct official transactions across {len(status)} monthly windows; {metrics['monthly_windows_failed']} failed windows.",
             f"Built {n:,} IL episodes involving {metrics['unique_pitchers']:,} pitchers. {metrics['orphan_transfers']} episodes start with an orphan transfer and retain an unknown initial event date.", '',
             '## Validation', '', pd.DataFrame([dict(metric=k, value=v) for k,v in metrics.items()]).to_markdown(index=False), '',
             '## Season coverage', '', by_year.to_markdown(), '',
             '## Interpretation', '',
             '- Canonical records are observed MLB IL episodes, including documented illnesses. They are not a census of every injury or a biological onset standard.',
             '- `injury_event_date` follows explicit onset, parsed retroactive date, effective date, then announcement. `public_known_date` preserves the announcement needed for prospective prediction cutoffs.',
             '- Historical descriptions can be revised. `*_known_at` fields for MLB are reported-date proxies, not verified publication timestamps; `*_observed_at` records actual retrieval. `historical_publication_verified` is false unless separately adjudicated. Use snapshot dates for strict information-availability guarantees.',
             '- Activations require a valid later transaction, no intervening unrelated placement, a bounded interval, and no contradictory observed MLB pitching. Offseason activations can be administrative roster moves.',
             '- Resolution dates never establish return. Censored durations remain NULL; an orphan transfer also lacks an initial placement date.',
             '- Official anatomy is preserved. Supplemental diagnoses, snapshot dates and surgery evidence are separate; disagreements require review.',
             '- Broad elbow/forearm IL evidence is compatible with a later UCL procedure but is not proof that the original complaint was a confirmed UCL tear.',
             '- Existing UCL-only and general-UCL-only cases are comparison findings, not automatic false positives/negatives. The legacy surgery snapshot includes only its own coverage and eligibility.',
             '- Features and labels in the existing DuckDB were not modified. Future outcome fields in this dataset must never be used as pre-injury predictors.',
             '- See `docs/general_injury_pipeline.md`, acquisition coverage, supplemental coverage and review CSVs for methods and limitations.', '']
    (report / 'build_summary.md').write_text('\n'.join(lines))
    return metrics
