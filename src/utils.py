from datetime import datetime, timedelta, timezone
import hashlib
import json
import pandas as pd


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)


def stable_id(*values):
    return hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:32]


def date_chunks(start, end, days=31):
    while start <= end:
        stop = min(end, start + timedelta(days=days - 1))
        yield start, stop
        start = stop + timedelta(days=1)


def innings_to_outs(value):
    """Baseball 5.2 means 17 outs, not 5.2 decimal innings."""
    if value is None or pd.isna(value):
        return None
    whole, _, remainder = str(value).partition('.')
    remainder = remainder or '0'
    if remainder not in ('0', '1', '2'):
        raise ValueError('Invalid baseball innings: ' + str(value))
    return int(whole) * 3 + int(remainder)


def state(db, source, key, status, rows=0, detail=None):
    db.upsert('ingestion_state', [{'source': source, 'item_key': str(key), 'status': status,
              'row_count': rows, 'detail': detail, 'updated_at': now()}], ['source', 'item_key'])


def issue(db, source, key, reason, raw=None):
    payload = json_text(raw) if raw is not None else None
    db.upsert('ingestion_issues', [{'issue_id': stable_id(source, key, reason, payload),
              'source': source, 'item_key': str(key), 'reason': reason, 'raw_json': payload,
              'recorded_at': now()}], ['issue_id'])
