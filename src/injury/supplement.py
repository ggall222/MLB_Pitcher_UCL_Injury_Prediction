"""Independent evidence and UCL comparisons; official fields are never overwritten."""
import json
import hashlib
import re

import pandas as pd

from ..utils import stable_id
from .acquire import save_frame
from .parser import parse_injury
from .taxonomy import UNKNOWN

FG_URL = 'https://www.fangraphs.com/roster-resource/injury-report'


def fetch_rosterresource(client, cfg):
    records, coverage = [], []

    def fetch(year):
        body = client.get(FG_URL, {'season': year, 'timeframe': 'all'},
                          force_refresh=bool(cfg.get('refresh_supplemental') and year == cfg['end_date'].year))
        match = re.search(rb'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', body, re.S)
        if not match:
            raise ValueError('Public page lacks structured __NEXT_DATA__; no bypass attempted')
        payload = json.loads(match[1])
        queries = payload.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
        rows = []
        for q in queries:
            data = q.get('state', {}).get('data')
            if isinstance(data, list):
                rows.extend(r for r in data if isinstance(r, dict) and 'injurySurgery' in r and 'mlbamid' in r)
        sha = hashlib.sha256(body).hexdigest()
        meta = next(r for r in reversed(client.records) if r.get('sha256') == sha)
        return [dict(r, _retrieved_at=meta['retrieved_at']) for r in rows if str(r.get('season')) == str(year)]

    for year, payload, error in client.many(range(cfg['start_date'].year, cfg['end_date'].year+1), fetch):
        coverage.append(dict(source='fangraphs_rosterresource', season=year,
                             status='failed' if error else 'ok' if payload else 'no_records_for_requested_season',
                             rows=len(payload or []), detail=error))
        for row in payload or []:
            retrieved_at = row.pop('_retrieved_at')
            raw = json.dumps(row, sort_keys=True)
            pid = row.get('mlbamid') or row.get('xMLBAMID')
            if not pid:
                continue
            parsed = parse_injury('Placed player on injured list. ' + str(row.get('injurySurgery') or ''))
            # This source deliberately combines injury and surgery dates. Never
            # reinterpret the combined field as biological onset.
            records.append(dict(supplement_record_id=stable_id('rosterresource', year, raw), player_id=int(pid),
                                player_name=row.get('playerName'), source='fangraphs_rosterresource',
                                source_url=FG_URL + f'?season={year}&timeframe=all', season=year,
                                source_snapshot_at=row.get('loaddate'),
                                source_retrieved_at=retrieved_at,
                                injury_or_surgery_date_raw=row.get('date'), date_display=row.get('dateDisplay'),
                                il_retro_date_raw=row.get('retrodate'), return_date_raw=row.get('returndate'),
                                description=row.get('injurySurgery'), position=row.get('position'),
                                status=row.get('status'), team=row.get('team'), raw_json=raw,
                                injury_region=parsed['injury_region'], injury_structure=parsed['injury_structure'],
                                injury_diagnosis=parsed['injury_diagnosis'], laterality=parsed['laterality'],
                                surgery_flag=parsed['surgery_flag'], surgery_type=parsed['surgery_type']))
    frame = pd.DataFrame(records).drop_duplicates('supplement_record_id') if records else pd.DataFrame()
    if not frame.empty:
        for raw, target in [('injury_or_surgery_date_raw', 'injury_or_surgery_date'),
                            ('il_retro_date_raw', 'il_retro_date'), ('return_date_raw', 'return_date')]:
            frame[target] = pd.to_datetime(frame[raw], format='%m/%d/%y', errors='coerce')
        # dateDisplay often encodes a month-only estimate despite a day-1 raw date.
        frame['date_precision_uncertain'] = frame.date_display.fillna('').astype(str).str.len().gt(0)
        frame['source_snapshot_at'] = pd.to_datetime(frame.source_snapshot_at, errors='coerce')
        frame = frame.sort_values(['season','player_id','il_retro_date','injury_or_surgery_date','supplement_record_id'], na_position='last').reset_index(drop=True)
        digest = stable_id(*sorted(frame.supplement_record_id))
        immutable = cfg['output_path'] / 'raw/supplemental' / f'rosterresource_{digest}.parquet'
        # Snapshot schema additions also receive a new filename; older bodies
        # and their projections are never replaced by a changed transformation.
        immutable = immutable.with_name(immutable.stem + '_v2.parquet')
        if not immutable.exists():
            save_frame(frame, immutable)
    save_frame(pd.DataFrame(coverage).sort_values('season'), cfg['report_path'] / 'supplemental_coverage.csv')
    return frame, coverage


def flag(event, reason):
    flags = json.loads(event['qa_flags'])
    event['qa_flags'] = json.dumps(sorted(set(flags + [reason])))
    event['qa_review_required'] = True


def add_source(event, name, agreement):
    sources = set(str(event.get('injury_source_secondary') or '').split(';')) - {''}
    sources.add(name)
    event['injury_source_secondary'] = ';'.join(sorted(sources))
    event['source_count'] = 1 + len(sources)
    event['source_agreement'] = 'conflict' if event['source_conflict'] else agreement


def enrich_and_compare(events, links, rosterresource, con, cfg):
    records = events.to_dict('records')
    by_id = {r['injury_event_id']: r for r in records}
    by_player = {}
    for e in records:
        by_player.setdefault(int(e['player_id']), []).append(e)
        for c in ('surgery_flag', 'surgery_type', 'surgery_date'):
            e['official_' + c] = e[c]
        e.update(supplemental_ucl_confirmed=False, supplemental_record_ids=[], supplemental_diagnoses=[],
                 supplemental_surgery_dates=[], supplemental_known_at=None, surgery_date_conflict=False)
    for group in by_player.values():
        group.sort(key=lambda e: (e['transaction_date'], e['anchor_transaction_id']))
    associations, rr_unmatched = [], []
    for r in rosterresource.to_dict('records'):
        candidates = by_player.get(int(r['player_id']), [])
        retro = r['il_retro_date']
        exact = [e for e in candidates if pd.notna(retro) and pd.notna(e['injury_event_date']) and
                 abs((retro - pd.Timestamp(e['injury_event_date'])).days) <= cfg.get('rosterresource_retro_tolerance_days',3)]
        method = 'mlbam_and_il_retro_within_3d'
        if not exact and pd.notna(r['injury_or_surgery_date']) and not r['date_precision_uncertain']:
            ref = r['injury_or_surgery_date']
            exact = [e for e in candidates if e['injury_region'] == r['injury_region'] and
                     pd.notna(e['injury_event_date']) and abs((ref - pd.Timestamp(e['injury_event_date'])).days) <= cfg.get('rosterresource_combined_date_tolerance_days',7)]
            method = 'mlbam_compatible_region_and_combined_date_within_7d'
        if len(exact) != 1:
            rr_unmatched.append(dict(**r, match_status='ambiguous' if exact else 'no_unique_official_episode', candidate_count=len(exact)))
            continue
        e = exact[0]
        region_conflict = e['injury_region'] != UNKNOWN and r['injury_region'] != UNKNOWN and e['injury_region'] != r['injury_region']
        diagnosis_conflict = (not region_conflict and e['injury_diagnosis'] not in (UNKNOWN,'Other','Surgery / Postoperative Recovery') and
                              r['injury_diagnosis'] not in (UNKNOWN,'Other','Surgery / Postoperative Recovery') and
                              e['injury_diagnosis'] != r['injury_diagnosis'])
        return_delta = (r['return_date']-pd.Timestamp(e['activation_date'])).days if pd.notna(r['return_date']) and pd.notna(e['activation_date']) else None
        return_conflict = return_delta is not None and abs(return_delta) > 3
        conflict = region_conflict or diagnosis_conflict or return_conflict
        e['source_conflict'] |= conflict
        if region_conflict:
            flag(e, 'rosterresource_anatomy_conflict')
        if diagnosis_conflict:
            flag(e, 'rosterresource_diagnosis_difference')
        if return_conflict:
            flag(e, 'rosterresource_return_date_difference')
        e['supplemental_record_ids'].append(r['supplement_record_id'])
        e['supplemental_diagnoses'].append(dict(source=r['source'], region=r['injury_region'],
                                               structure=r['injury_structure'], diagnosis=r['injury_diagnosis'],
                                               description=r['description'], source_snapshot_at=str(r['source_snapshot_at']),
                                               evidence_observed_at=r.get('source_retrieved_at')))
        add_source(e, r['source'], 'compatible' if not conflict else 'conflict')
        if r.get('source_retrieved_at'):
            observed_at = pd.Timestamp(r['source_retrieved_at']).tz_localize(None)
            e['supplemental_known_at'] = max(pd.Timestamp(e['supplemental_known_at']), observed_at) if e['supplemental_known_at'] else observed_at
        # Independent surgery evidence is separate from official anatomy and onset.
        if r['surgery_flag'] and not conflict:
            e['surgery_flag'] = True
            if not e['surgery_type']:
                e['surgery_type'] = r['surgery_type']
                e['surgery_source'] = r['source']
                e['surgery_known_at'] = r['source_snapshot_at']
                e['surgery_observed_at'] = r.get('source_retrieved_at')
            if pd.notna(r['injury_or_surgery_date']) and not r['date_precision_uncertain']:
                e['supplemental_surgery_dates'].append(dict(date=str(r['injury_or_surgery_date'].date()), source=r['source']))
                if pd.isna(e['surgery_date']):
                    e['surgery_date'] = r['injury_or_surgery_date']
                    e['surgery_source'] = r['source']
                    e['surgery_observed_at'] = r.get('source_retrieved_at')
        associations.append(dict(injury_event_id=e['injury_event_id'], supplemental_record_id=r['supplement_record_id'],
                                  source=r['source'], match_method=method, source_conflict=conflict,
                                  region_conflict=region_conflict, diagnosis_conflict=diagnosis_conflict,
                                  return_date_conflict=return_conflict,
                                  official_diagnosis=e['injury_diagnosis'], supplemental_diagnosis=r['injury_diagnosis'],
                                  official_injury_region=e['injury_region'], supplemental_injury_region=r['injury_region'],
                                  date_difference_days=(retro - pd.Timestamp(e['injury_event_date'])).days if pd.notna(retro) and pd.notna(e['injury_event_date']) else None,
                                  activation_difference_days=return_delta))
    # Snapshot the existing reconciled surgery list and UCL event tables; read only.
    old = con.execute('SELECT * FROM injury_events').fetchdf()
    surgeries = con.execute('SELECT * FROM injuries').fetchdf()
    digest = stable_id(*surgeries.sort_values('injury_id').raw_json.fillna('').tolist())
    snapshot = cfg['output_path'] / 'raw/supplemental' / f'existing_ucl_surgeries_{digest}.parquet'
    if not snapshot.exists():
        save_frame(surgeries, snapshot)
    surgery_index = surgeries.set_index('injury_id').to_dict('index')
    observed_pitchers = set(int(x[0]) for x in con.execute('SELECT DISTINCT pitcher_id FROM appearances').fetchall())
    pitcher_ids = observed_pitchers | set(by_player)
    old = old[old.pitcher_id.isin(pitcher_ids) & old.confirmed_ucl_surgery.fillna(False) &
              old.surgery_date_upper.ge(pd.Timestamp(cfg['start_date'])) & old.surgery_date_lower.le(pd.Timestamp(cfg['end_date']))]
    old = old.sort_values(['pitcher_id','surgery_date_lower','injury_event_id'])
    linked_transactions = {int(r.transaction_id): r.injury_event_id for r in links.itertuples()}
    comparison, linked_general = [], set()
    for r in old.itertuples():
        candidate, method = None, None
        if pd.notna(r.onset_transaction_id):
            eid = linked_transactions.get(int(r.onset_transaction_id))
            if eid and by_id[eid]['player_id'] == int(r.pitcher_id):
                candidate, method = by_id[eid], 'same_mlbam_and_source_transaction'
        if candidate is None:
            possible = []
            group = by_player.get(int(r.pitcher_id), [])
            for index, e in enumerate(group):
                if e['injury_region'] not in ('Elbow / UCL', 'Forearm'):
                    continue
                anchor = pd.Timestamp(e['transaction_date'])
                stop = min(anchor + pd.Timedelta(days=cfg['activation_max_gap_days']), pd.Timestamp(cfg['end_date']))
                if pd.notna(e['activation_date']):
                    stop = min(stop, pd.Timestamp(e['activation_date']))
                if pd.notna(e.get('episode_boundary_date')):
                    stop = min(stop, pd.Timestamp(e['episode_boundary_date']))
                if index + 1 < len(group):
                    stop = min(stop, pd.Timestamp(group[index+1]['transaction_date']) - pd.Timedelta(days=1))
                if r.surgery_date_lower <= stop and r.surgery_date_upper >= anchor:
                    possible.append(e)
            if len(possible) == 1:
                candidate, method = possible[0], 'unique_mlbam_active_arm_episode_and_surgery_interval'
        s = surgery_index.get(r.source_injury_id, {})
        item = dict(existing_ucl_event_id=r.injury_event_id, player_id=int(r.pitcher_id), player_name=r.pitcher_name,
                    general_injury_event_id=candidate['injury_event_id'] if candidate else None,
                    comparison_status='both' if candidate else 'existing_ucl_only', match_method=method,
                    existing_label_reference_date=r.label_reference_date, existing_onset_confidence=r.onset_confidence,
                    general_injury_event_date=candidate['injury_event_date'] if candidate else None,
                    general_public_known_date=candidate['public_known_date'] if candidate else None,
                    date_difference_days=(pd.Timestamp(candidate['injury_event_date'])-r.label_reference_date).days if candidate and pd.notna(candidate['injury_event_date']) and pd.notna(r.label_reference_date) else None,
                    general_injury_region=candidate['injury_region'] if candidate else None,
                    general_injury_diagnosis=candidate['injury_diagnosis'] if candidate else None,
                    general_explicit_ucl=candidate['explicit_ucl'] if candidate else None,
                    surgery_date=r.surgery_date, surgery_date_lower=r.surgery_date_lower, surgery_date_upper=r.surgery_date_upper,
                    procedure_category=r.procedure_category, surgery_source=s.get('source_url'), surgery_list_agreement=bool(candidate),
                    surgery_source_level=s.get('level'),
                    comparison_reason=('matched_official_episode' if candidate else
                        'surgery_source_reports_non_mlb_level' if s.get('level') != 'MLB' else
                        'pre_study_public_onset' if pd.notna(r.label_reference_date) and r.label_reference_date < pd.Timestamp(cfg['start_date']) else
                        'legacy_surgery_only_unknown_onset' if pd.isna(r.label_reference_date) else 'no_unique_compatible_official_episode'))
        comparison.append(item)
        if candidate:
            e = candidate
            linked_general.add(e['injury_event_id'])
            e['supplemental_ucl_confirmed'] = True
            e['surgery_flag'] = True
            e['supplemental_record_ids'].append(str(r.source_injury_id))
            e['supplemental_surgery_dates'].append(dict(date=str(r.surgery_date), lower=str(r.surgery_date_lower),
                                                       upper=str(r.surgery_date_upper), precision=r.surgery_date_precision, source='roegele_ucl_list'))
            if not e['surgery_type']:
                e['surgery_type'] = r.procedure_category
                e['surgery_source'] = 'roegele_ucl_list'
                e['surgery_known_at'] = s.get('retrieved_at')
                e['surgery_observed_at'] = str(s.get('retrieved_at'))
            if pd.notna(e['surgery_date']) and pd.notna(r.surgery_date) and pd.Timestamp(e['surgery_date']) != r.surgery_date:
                e['surgery_date_conflict'] = True
                e['source_conflict'] = True
                flag(e, 'different_surgery_dates_or_multiple_procedures')
            if pd.isna(e['surgery_date']) and pd.notna(r.surgery_date):
                e['surgery_date'] = r.surgery_date
                e['surgery_source'] = 'roegele_ucl_list'
                # Retrospective curated dates are not historically observable at
                # the operation date; keep the snapshot availability timestamp.
                e['surgery_known_at'] = s.get('retrieved_at')
                e['surgery_observed_at'] = str(s.get('retrieved_at'))
            if e['injury_region'] not in ('Elbow / UCL', 'Forearm', UNKNOWN):
                e['source_conflict'] = True
                flag(e, 'ucl_surgery_anatomy_conflict')
            add_source(e, 'roegele_ucl_list', 'compatible_arm_episode')
    for e in records:
        if e['explicit_ucl'] and e['injury_event_id'] not in linked_general:
            comparison.append(dict(existing_ucl_event_id=None, player_id=e['player_id'], player_name=e['player_name'],
                                   general_injury_event_id=e['injury_event_id'], comparison_status='general_ucl_only',
                                   general_injury_event_date=e['injury_event_date'], general_public_known_date=e['public_known_date'],
                                   general_injury_region=e['injury_region'], general_injury_diagnosis=e['injury_diagnosis'],
                                   general_explicit_ucl=True, surgery_list_agreement=False))
        for field in ('supplemental_record_ids', 'supplemental_diagnoses', 'supplemental_surgery_dates'):
            e[field] = json.dumps(e[field], default=str)
    save_frame(pd.DataFrame(comparison), cfg['report_path'] / 'ucl_pipeline_comparison.csv')
    save_frame(pd.DataFrame(associations), cfg['report_path'] / 'supplemental_source_matches.csv')
    save_frame(pd.DataFrame(rr_unmatched), cfg['report_path'] / 'supplemental_unmatched_records.csv')
    return pd.DataFrame(records), pd.DataFrame(comparison), pd.DataFrame(associations)
