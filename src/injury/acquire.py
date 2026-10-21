"""Restartable monthly transactions using the project's immutable HTTP cache."""
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
import yaml

from ..config import ROOT
from ..http import HttpClient
from ..ingest_mlb import API

LOG = logging.getLogger(__name__)


def load_settings(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / 'config/injury.yaml').read_text())
    cfg['start_date'] = date.fromisoformat(str(cfg['start_date']))
    end = cfg['end_date']
    cfg['end_date'] = min(date.today(), date.fromisoformat(cfg['maximum_study_date'])) if end == 'latest' else date.fromisoformat(str(end))
    if cfg['end_date'] < cfg['start_date']:
        raise ValueError('Injury study end must be on/after start')
    for key in ('database_path', 'output_path', 'report_path'):
        cfg[key] = ROOT / cfg[key]
    cfg['cache_path'] = cfg['output_path'] / 'raw/http'
    return cfg


def months(start: date, end: date) -> Iterator[Tuple[date, date]]:
    while start <= end:
        stop = min(end, date(start.year, start.month, monthrange(start.year, start.month)[1]))
        yield start, stop
        start = stop + timedelta(days=1)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    temp.replace(path)


def save_frame(frame, path):
    """Replace derived materializations atomically; immutable HTTP bodies stay intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    if path.suffix == '.parquet':
        frame.to_parquet(temp, index=False)
    else:
        frame.to_csv(temp, index=False, date_format='%Y-%m-%d')
    temp.replace(path)


def flatten_transaction(t, source_sha, retrieved_at, request_start, request_end):
    return dict(transaction_id=t.get('id'), player_id=t.get('person', {}).get('id'),
                player_name=t.get('person', {}).get('fullName'), transaction_date=t.get('date'),
                effective_date=t.get('effectiveDate'), resolution_date=t.get('resolutionDate'),
                from_team_id=t.get('fromTeam', {}).get('id'), from_team_name=t.get('fromTeam', {}).get('name'),
                to_team_id=t.get('toTeam', {}).get('id'), to_team_name=t.get('toTeam', {}).get('name'),
                type_code=t.get('typeCode'), type_description=t.get('typeDesc'), description=t.get('description'),
                raw_json=json.dumps(t, ensure_ascii=False, sort_keys=True),
                source_url=API + '/transactions?transactionIds=' + str(t.get('id')),
                source_sha256=source_sha, source_retrieved_at=retrieved_at,
                request_start=str(request_start), request_end=str(request_end))


def acquire_transactions(cfg, offline=False, refresh_recent=False):
    """A failure never discards completed windows. Every source version is retained."""
    root = cfg['output_path'] / 'raw'
    root.mkdir(parents=True, exist_ok=True)
    client = HttpClient(cfg, offline=offline)
    jobs = list(months(cfg['start_date'], cfg['end_date']))
    state_path = root / 'transaction_request_status.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    status, rows = {}, []

    def fetch(pair):
        left, right = pair
        force = refresh_recent and right >= cfg['end_date'] - timedelta(days=cfg['refresh_recent_days'])
        body = client.get(API + '/transactions', dict(sportId=1, startDate=str(left), endDate=str(right)), force_refresh=force)
        payload = json.loads(body)
        if not isinstance(payload.get('transactions'), list):
            raise ValueError('MLB response lacks transactions list')
        # API currently has no pagination metadata; detect a changed contract.
        if payload.get('totalItems', len(payload['transactions'])) > len(payload['transactions']):
            raise ValueError('Truncated transaction response; pagination must be implemented before acceptance')
        sha = hashlib.sha256(body).hexdigest()
        meta = next(r for r in reversed(client.records) if r.get('sha256') == sha)
        return payload['transactions'], sha, meta['retrieved_at']

    for i, (pair, result, error) in enumerate(client.many(jobs, fetch), 1):
        left, right = pair
        key = f'{left}:{right}'
        if error:
            status[key] = dict(start=str(left), end=str(right), status='failed', error=error,
                               previous_success=old.get(key), rows=0)
            # A failed refresh must not erase previously retrieved transactions.
            # They remain explicitly stale; this window is still reported failed.
            prior_path = root / 'mlb_transactions_2015_2026_raw.parquet'
            if prior_path.exists():
                prior = pd.read_parquet(prior_path)
                keep = pd.to_datetime(prior.transaction_date, errors='coerce').between(pd.Timestamp(left),pd.Timestamp(right))
                retained = prior.loc[keep].to_dict('records')
                rows.extend(retained)
                status[key]['previous_rows_retained'] = len(retained)
        else:
            transactions, sha, retrieved = result
            rows.extend(flatten_transaction(t, sha, retrieved, left, right) for t in transactions)
            status[key] = dict(start=str(left), end=str(right), status='ok', rows=len(transactions),
                               sha256=sha, retrieved_at=retrieved)
        old[key] = status[key]
        atomic_json(state_path, old)
        with (root / 'request_history.jsonl').open('a') as f:
            f.write(json.dumps(dict(key=key, checked_at=datetime.now(timezone.utc).isoformat(), **status[key])) + '\n')
        if i % 12 == 0 or error or i == len(jobs):
            LOG.info('Transactions: %s/%s monthly windows; %s failures', i, len(jobs), sum(s['status'] != 'ok' for s in status.values()))
    if not rows:
        raise RuntimeError('No transaction windows available; raw cache and failure status preserved')
    frame = pd.DataFrame(rows)
    for col in ('transaction_id', 'player_id', 'from_team_id', 'to_team_id'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce').astype('Int64')
    # Keep full response occurrences/variants in immutable snapshots and audit conflicts.
    digest = hashlib.sha256('|'.join(sorted(r.get('sha256', 'failed') for r in status.values())).encode()).hexdigest()[:20]
    snapshot = root / 'snapshots' / f'transaction_occurrences_{digest}.parquet'
    if not snapshot.exists():
        save_frame(frame.sort_values(['transaction_id', 'request_start']), snapshot)
    conflicts = frame.groupby('transaction_id').raw_json.nunique()
    conflicts = frame[frame.transaction_id.isin(conflicts[conflicts > 1].index)]
    save_frame(conflicts, cfg['report_path'] / 'transaction_source_conflicts.csv')
    invalid = frame[frame.transaction_id.isna() | frame.player_id.isna()]
    save_frame(invalid, cfg['report_path'] / 'transactions_missing_identifiers.csv')
    # Identical IDs are one transaction. Latest retrieved source version wins,
    # while conflicting original versions remain available in the snapshot/cache.
    raw = frame.sort_values(['source_retrieved_at', 'request_start']).drop_duplicates('transaction_id', keep='last')
    raw = raw.sort_values(['transaction_date', 'transaction_id']).reset_index(drop=True)
    raw['source_record_conflict'] = raw.transaction_id.isin(conflicts.transaction_id)
    save_frame(raw, root / 'mlb_transactions_2015_2026_raw.parquet')
    save_frame(pd.DataFrame(status.values()).sort_values('start'), cfg['report_path'] / 'acquisition_coverage.csv')
    return raw, client, status


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--refresh-recent', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    data, _, states = acquire_transactions(load_settings(args.config), args.offline, args.refresh_recent)
    LOG.info('Raw acquisition: %s unique transactions; %s failed windows', len(data), sum(s['status'] != 'ok' for s in states.values()))
