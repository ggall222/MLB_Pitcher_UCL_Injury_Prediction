"""Deterministic IL episodes with bounded, auditable activation matching."""
import json
from bisect import bisect_right

import pandas as pd

from ..utils import stable_id
from .taxonomy import UNKNOWN


def known(value):
    return value is not None and not pd.isna(value)


def day(value):
    return pd.Timestamp(value) if known(value) else pd.NaT


def new_episode(t, orphan=False):
    event = dict(injury_event_id=stable_id('general_mlb_il_v1', int(t['player_id']), int(t['transaction_id'])),
                 player_id=int(t['player_id']), player_name=t['player_name'], primary_position=t['primary_position'],
                 throws=t['throws'], two_way_player=t['two_way_player'], has_statcast_history=t['has_statcast_history'],
                 pitcher_identity_basis=t['pitcher_identity_basis'], identity_source=t['identity_source'],
                 team_id=t['team_id'], team_name=t['team_name'],
                 placement_transaction_id=None if orphan else int(t['transaction_id']),
                 anchor_transaction_id=int(t['transaction_id']), placement_description=None if orphan else t['description'],
                 transaction_date=t['transaction_date'], effective_date=t['effective_date'], resolution_date=t['resolution_date'],
                 retroactive_date=t.get('retroactive_date'), explicitly_documented_onset_date=t.get('explicitly_documented_onset_date'),
                 injury_event_date=pd.NaT if orphan else t['injury_event_date'],
                 injury_event_date_basis='orphan_transfer_unknown_placement' if orphan else t['injury_event_date_basis'],
                 public_known_date=t['transaction_date'], event_date_known_at=t['transaction_date'],
                 diagnosis_known_at=t['transaction_date'], initial_injury_region=t['injury_region'],
                 historical_publication_verified=False, known_at_basis='reported_transaction_date_proxy',
                 evidence_observed_at=t['source_retrieved_at'], event_date_observed_at=t['source_retrieved_at'],
                 diagnosis_observed_at=t['source_retrieved_at'],
                 initial_injury_structure=t['injury_structure'], initial_injury_diagnosis=t['injury_diagnosis'],
                 initial_laterality=t['laterality'], injury_region=t['injury_region'],
                 injury_structure=t['injury_structure'], injury_diagnosis=t['injury_diagnosis'],
                 laterality=t['laterality'], injury_text=t['injury_text'], explicit_ucl=t['explicit_ucl'],
                 initial_explicit_ucl=t['explicit_ucl'], ucl_diagnosis_reported_at=t['transaction_date'] if t['explicit_ucl'] else pd.NaT,
                 all_injury_regions=t['all_injury_regions'], all_injury_diagnoses=t['all_injury_diagnoses'],
                 multiple_injury_regions=t['multiple_injury_regions'], diagnosis_uncertain=t['diagnosis_uncertain'],
                 surgery_flag=t['surgery_flag'], surgery_type=t['surgery_type'], surgery_date=t.get('surgery_date'),
                 surgery_source='mlb_stats_api_transactions' if t['surgery_flag'] else None,
                 surgery_known_at=t['transaction_date'] if t['surgery_flag'] else pd.NaT,
                 surgery_observed_at=t['source_retrieved_at'] if t['surgery_flag'] else None,
                 initial_il_length=None if orphan else t['il_length'], max_il_length=t['il_length'],
                 transferred_to_60_day=bool(orphan and known(t['il_length']) and t['il_length'] == 60),
                 transfer_count=int(orphan), orphan_transfer=orphan, left_censored=orphan, duplicate_placement_count=0,
                 inferred_continuation_count=0, episode_boundary_date=pd.NaT,
                 activation_transaction_id=None, activation_description=None, activation_date=pd.NaT,
                 activation_transaction_date=pd.NaT, activation_effective_date=pd.NaT, activation_known_at=pd.NaT,
                 activation_observed_at=None,
                 activation_match_valid=False, activation_candidate_id=None, activation_rejection_reason=None,
                 offseason_activation=False, days_out_observed=None, duration_observed=False,
                 right_censored=True, censor_reason='observation_end', overlapping_episode=False,
                 timing_uncertain=bool(orphan or t['date_parse_uncertain']),
                 qa_review_required=bool(orphan or t['parser_review_required'] or t['identity_review_required'] or t['source_record_conflict']),
                 qa_flags=[], source_conflict=False, injury_source_primary='mlb_stats_api_transactions',
                 injury_source_secondary=None, source_count=1, source_agreement='single_source',
                 diagnosis_source=t['source_url'], onset_source=t['source_url'],
                 transaction_ids=[int(t['transaction_id'])], transfer_transaction_ids=[int(t['transaction_id'])] if orphan else [],
                 descriptions=[t['description']], evidence_urls=[t['source_url']],
                 source_retrieved_at=t['source_retrieved_at'])
    if t['parser_review_required']:
        event['qa_flags'].append('parser_uncertain_or_multiple_regions')
    if t['identity_review_required']:
        event['qa_flags'].append('identity_requires_review')
    if t['date_parse_uncertain']:
        event['qa_flags'].append('date_parse_uncertain')
    if orphan:
        event['qa_flags'].append('orphan_transfer_missing_placement')
    return event


def attach_transaction(event, t, relationship, links):
    tid = int(t['transaction_id'])
    if tid not in event['transaction_ids']:
        event['transaction_ids'].append(tid)
        event['descriptions'].append(t['description'])
        event['evidence_urls'].append(t['source_url'])
    links.append(dict(injury_event_id=event['injury_event_id'], transaction_id=tid,
                      player_id=event['player_id'], relationship=relationship))


def add_evidence(event, t):
    """Later details are separately dated. Conflicts do not overwrite initial facts."""
    different = (event['injury_region'] != UNKNOWN and t['injury_region'] != UNKNOWN and
                 event['injury_region'] != t['injury_region'])
    if different:
        event['source_conflict'] = True
        event['qa_flags'].append('conflicting_anatomy_within_episode')
        event['qa_review_required'] = True
    if event['injury_region'] == UNKNOWN and t['injury_region'] != UNKNOWN:
        for col in ('injury_region', 'injury_structure', 'injury_diagnosis', 'laterality',
                    'injury_text', 'explicit_ucl', 'all_injury_regions', 'all_injury_diagnoses'):
            event[col] = t[col]
        event['diagnosis_known_at'] = t['transaction_date']
        event['diagnosis_observed_at'] = t['source_retrieved_at']
        event['diagnosis_source'] = t['source_url']
    elif not different and not event['explicit_ucl'] and t['explicit_ucl']:
        event['explicit_ucl'] = True
        event['ucl_diagnosis_reported_at'] = t['transaction_date']
        event['injury_structure'] = t['injury_structure']
        event['diagnosis_known_at'] = t['transaction_date']
        event['diagnosis_observed_at'] = t['source_retrieved_at']
        event['diagnosis_source'] = t['source_url']
    if t['explicit_ucl'] and not event['explicit_ucl']:
        event['explicit_ucl'] = True
        event['ucl_diagnosis_reported_at'] = t['transaction_date']
    if t['surgery_flag']:
        if not event['surgery_flag']:
            event['surgery_known_at'] = t['transaction_date']
            event['surgery_observed_at'] = t['source_retrieved_at']
        event['surgery_flag'] = True
        if not event['surgery_type'] or event['surgery_type'] == 'Other / unspecified surgery':
            event['surgery_type'] = t['surgery_type']
        event['surgery_source'] = 'mlb_stats_api_transactions'
        if known(t.get('surgery_date')):
            event['surgery_date'] = t['surgery_date']


def build_episodes(transactions, cfg, observed_appearances=None):
    """Match chronological IL actions, never resolutionDate or a later unrelated episode."""
    events, links, unpaired = [], [], []
    observed = {}
    if observed_appearances is not None:
        observed = {int(pid): sorted(set(pd.to_datetime(g.game_date)))
                    for pid, g in observed_appearances.groupby('pitcher_id')}
    ordered = transactions.sort_values(['player_id', 'transaction_date', 'transaction_id'])
    for pid, group in ordered.groupby('player_id', sort=True):
        active = None
        for t in group.to_dict('records'):
            action = t['il_action']
            if action == 'transfer' and active is not None:
                dates = observed.get(int(pid), [])
                index = bisect_right(dates, day(active['transaction_date']))
                if index < len(dates) and dates[index] < day(t['transaction_date']):
                    active['censor_reason'] = 'observed_pitching_without_activation'
                    active['episode_boundary_date'] = dates[index]
                    active['qa_flags'].append('missing_activation_before_observed_pitching')
                    active['qa_review_required'] = True
                    active = None
            if action == 'placement':
                if active is not None:
                    dates = observed.get(int(pid), [])
                    index = bisect_right(dates, day(active['transaction_date']))
                    observed_return = dates[index] if index < len(dates) and dates[index] < day(t['transaction_date']) else None
                    if observed_return is not None:
                        active['censor_reason'] = 'observed_pitching_without_activation'
                        active['episode_boundary_date'] = observed_return
                        active['qa_flags'].append('missing_activation_before_observed_pitching')
                        active['qa_review_required'] = True
                        active = None
                if active is not None:
                    compatible_region = (t['injury_region'] == active['injury_region'] or UNKNOWN in (t['injury_region'],active['injury_region']))
                    same = (t['transaction_date'] == active['transaction_date'] and
                            compatible_region and t['team_id'] == active['team_id'])
                    if same:
                        active['duplicate_placement_count'] += 1
                        active['qa_review_required'] = True
                        active['qa_flags'].append('duplicate_same_day_placement')
                        attach_transaction(active, t, 'duplicate_placement', links)
                        add_evidence(active, t)
                        continue
                    # Some official records say 'placed' for a longer designation.
                    # Require same-season anatomy, compatible side, no observed
                    # return, a larger IL designation, and a bounded interval.
                    same_reference = any(known(t.get(c)) and known(active['injury_event_date']) and
                                         day(t[c]) == day(active['injury_event_date']) for c in ('injury_event_date','retroactive_date','effective_date'))
                    continuation = (t['transaction_date'].year == active['transaction_date'].year and
                        0 < (t['transaction_date']-active['transaction_date']).days <= cfg.get('continuation_max_gap_days',180) and
                        ((t['injury_region'] == active['injury_region'] and t['injury_region'] != UNKNOWN) or
                         (same_reference and compatible_region)) and
                        (t['laterality'] == active['laterality'] or 'Unknown' in (t['laterality'],active['laterality'])) and
                        known(t['il_length']) and known(active['max_il_length']) and t['il_length'] > active['max_il_length'])
                    if continuation:
                        active['inferred_continuation_count'] += 1
                        active['max_il_length'] = t['il_length']
                        active['transferred_to_60_day'] |= t['il_length'] == 60
                        active['qa_flags'].append('longer_il_placement_inferred_continuation')
                        active['qa_review_required'] = True
                        attach_transaction(active, t, 'inferred_il_continuation', links)
                        add_evidence(active, t)
                        continue
                    active['overlapping_episode'] = True
                    active['right_censored'] = True
                    active['censor_reason'] = 'superseded_by_new_placement'
                    active['qa_review_required'] = True
                    active['qa_flags'].append('new_placement_before_activation')
                    active['episode_boundary_date'] = t['transaction_date']
                    previous = active
                else:
                    previous = None
                active = new_episode(t)
                if previous:
                    active['overlapping_episode'] = True
                    active['qa_review_required'] = True
                    active['qa_flags'].append('previous_episode_unresolved')
                events.append(active)
                attach_transaction(active, t, 'initial_placement', links)
            elif action == 'transfer':
                if active is None:
                    active = new_episode(t, orphan=True)
                    events.append(active)
                    attach_transaction(active, t, 'orphan_transfer', links)
                else:
                    active['transfer_count'] += 1
                    active['transfer_transaction_ids'].append(int(t['transaction_id']))
                    if known(t['il_length']):
                        active['max_il_length'] = max([int(x) for x in (active['max_il_length'], t['il_length']) if known(x)])
                    active['transferred_to_60_day'] |= bool(known(t['il_length']) and t['il_length'] == 60)
                    attach_transaction(active, t, 'transfer', links)
                    add_evidence(active, t)
            elif action == 'activation':
                if active is None:
                    unpaired.append(dict(transaction_id=int(t['transaction_id']), player_id=int(pid),
                                          action=action, reason='activation_without_observed_open_episode', description=t['description']))
                    continue
                activation = day(t['effective_date']) if known(t['effective_date']) else day(t['transaction_date'])
                anchor = day(active['transaction_date'])
                gap = (activation - anchor).days
                rejection = None
                if activation > day(t['transaction_date']):
                    rejection = 'activation_effective_after_announcement'
                elif gap <= 0 or (known(active['injury_event_date']) and activation <= day(active['injury_event_date'])):
                    rejection = 'activation_not_strictly_after_placement'
                elif gap > cfg['activation_max_gap_days']:
                    rejection = 'activation_exceeds_maximum_matching_gap'
                # Observed MLB return before purported activation implies a missing
                # lifecycle boundary; a later activation must not silently bridge it.
                dates = observed.get(int(pid), [])
                index = bisect_right(dates, anchor)
                if not rejection and index < len(dates) and dates[index] < activation:
                    rejection = 'observed_mlb_pitching_before_activation'
                if rejection:
                    active['activation_candidate_id'] = int(t['transaction_id'])
                    active['activation_rejection_reason'] = rejection
                    active['timing_uncertain'] = True
                    active['qa_review_required'] = True
                    active['qa_flags'].append(rejection)
                    active['censor_reason'] = 'activation_match_rejected'
                    active['episode_boundary_date'] = dates[index] if rejection == 'observed_mlb_pitching_before_activation' else activation
                    attach_transaction(active, t, 'rejected_activation_candidate', links)
                    # That activation is an observed boundary, even if duration
                    # cannot safely be assigned to the open injury episode.
                    active = None
                    continue
                active.update(activation_transaction_id=int(t['transaction_id']), activation_description=t['description'],
                              activation_date=activation, activation_transaction_date=t['transaction_date'],
                              activation_effective_date=t['effective_date'], activation_known_at=t['transaction_date'],
                              activation_observed_at=t['source_retrieved_at'],
                              activation_match_valid=True, right_censored=False, censor_reason=None,
                              episode_boundary_date=activation,
                              offseason_activation=bool(activation.month in (11, 12, 1, 2)))
                if known(active['injury_event_date']):
                    active['days_out_observed'] = (activation - day(active['injury_event_date'])).days
                    active['duration_observed'] = True
                attach_transaction(active, t, 'activation', links)
                active = None
            elif active is not None:
                attach_transaction(active, t, 'other_il_status', links)
    for e in events:
        length = e['days_out_observed']
        e['extreme_duration'] = bool(length is not None and length > cfg['long_duration_review_days'])
        if e['extreme_duration']:
            e['qa_flags'].append('long_observed_duration')
        if e['right_censored']:
            e['qa_flags'].append('unresolved_duration')
        e['qa_review_required'] |= e['extreme_duration'] or e['right_censored']
        ref = day(e['injury_event_date'])
        # A descriptive roster proxy only; no assumed medical season-ending date.
        season_end = pd.Timestamp(ref.year, 11, 1) if known(ref) else pd.NaT
        e['season_ending_proxy'] = (bool(pd.Timestamp(cfg['end_date']) >= season_end and
                                       (not known(e['activation_date']) or day(e['activation_date']) >= season_end))
                                    if known(ref) and ref.month <= 10 else None)
        e['observation_end_date'] = pd.Timestamp(cfg['end_date'])
        for field in ('transaction_ids', 'transfer_transaction_ids', 'descriptions', 'evidence_urls', 'qa_flags'):
            e[field] = json.dumps(e[field] if field != 'qa_flags' else sorted(set(e[field])))
    frame = pd.DataFrame(events)
    for c in ('player_id', 'team_id', 'placement_transaction_id', 'activation_transaction_id', 'activation_candidate_id',
              'anchor_transaction_id', 'initial_il_length', 'max_il_length', 'days_out_observed'):
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors='coerce').astype('Int64')
    return frame, pd.DataFrame(links), pd.DataFrame(unpaired, columns=['transaction_id', 'player_id', 'action', 'reason', 'description'])
