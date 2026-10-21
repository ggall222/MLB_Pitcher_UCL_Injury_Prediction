"""Build general MLB pitcher injury episodes without altering the UCL database.

    .venv/bin/python -m src.injury_pipeline
    .venv/bin/python -m src.injury_pipeline --update
    .venv/bin/python -m src.injury_pipeline --offline
"""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from .injury.acquire import load_settings, acquire_transactions, save_frame, atomic_json
from .injury.identity import normalize_transactions, lookup_identities
from .injury.episodes import build_episodes
from .injury.supplement import fetch_rosterresource, enrich_and_compare
from .injury.validation import validate_and_report

LOG = logging.getLogger(__name__)


def fingerprint(path):
    st = path.stat()
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def run(cfg, offline=False, update=False):
    cfg['refresh_supplemental'] = update
    for name in ('raw', 'interim', 'processed'):
        (cfg['output_path'] / name).mkdir(parents=True, exist_ok=True)
    cfg['report_path'].mkdir(parents=True, exist_ok=True)
    before = fingerprint(cfg['database_path'])
    raw, client, status = acquire_transactions(cfg, offline, update)
    LOG.info('Classifying transaction actions and injury text')
    normalized = normalize_transactions(raw, cfg)
    con = duckdb.connect(str(cfg['database_path']), read_only=True)
    try:
        con.execute("SET memory_limit='2GB'")
        linked, identity_failures = lookup_identities(normalized, con, client, cfg)
        tx = linked[linked.included_primary_population].copy()
        observed = con.execute("SELECT pitcher_id,game_date FROM appearances WHERE NOT timing_uncertain AND game_type='R'").fetchdf()
        events, links, unpaired = build_episodes(tx, cfg, observed)
        LOG.info('Constructed %s episodes; fetching independent RosterResource evidence', len(events))
        rr, supplemental_coverage = fetch_rosterresource(client, cfg)
        events, comparison, matches = enrich_and_compare(events, links, rr, con, cfg)
        # Enforce stable nullable types after record-oriented evidence integration.
        dates = [c for c in events if c.endswith('_date') or c.endswith('_known_at') or c in ('supplemental_known_at',)]
        for c in dates:
            events[c] = pd.to_datetime(events[c], errors='coerce')
        for c in ('player_id', 'team_id', 'placement_transaction_id', 'activation_transaction_id',
                  'activation_candidate_id', 'anchor_transaction_id', 'initial_il_length', 'max_il_length', 'days_out_observed'):
            events[c] = pd.to_numeric(events[c], errors='coerce').astype('Int64')
        events = events.sort_values(['transaction_date', 'player_id', 'injury_event_id']).reset_index(drop=True)
        source_complete = all(s['status'] == 'ok' for s in status.values())
        events['transaction_sources_complete'] = source_complete
        if not source_complete:
            events['qa_review_required'] = True
        metrics = validate_and_report(raw, normalized, linked, events, links, comparison, matches, status, cfg)
    finally:
        con.close()
    if fingerprint(cfg['database_path']) != before:
        raise RuntimeError('Source database metadata changed during the build; inspect external writers')
    processed = cfg['output_path'] / 'processed'
    datasets = {'mlb_pitcher_injury_events_2015_2026': events,
                'mlb_pitcher_injury_transactions_2015_2026': tx,
                'mlb_pitcher_injury_unresolved_2015_2026': events[events.qa_review_required | events.right_censored],
                'injury_event_transaction_links': links}
    for name, frame in datasets.items():
        for suffix in ('.csv', '.parquet'):
            save_frame(frame, processed / (name + suffix))
    save_frame(unpaired, cfg['report_path'] / 'unpaired_il_transactions.csv')
    code_files = sorted((Path(__file__).parent / 'injury').glob('*.py')) + [Path(__file__)]
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), config={k:str(v) if isinstance(v,(Path,date)) else v for k,v in cfg.items()},
                    database_fingerprint_before=before, database_fingerprint_after=fingerprint(cfg['database_path']),
                    source_database_read_only=True, metrics=metrics, identity_failures=identity_failures,
                    supplemental_coverage=supplemental_coverage,
                    http_evidence=[dict(url=r['url'],sha256=r['sha256'],retrieved_at=r['retrieved_at']) for r in client.records if 'sha256' in r],
                    raw_transactions_sha256=hashlib.sha256((cfg['output_path']/'raw/mlb_transactions_2015_2026_raw.parquet').read_bytes()).hexdigest(),
                    review_annotations_sha256=hashlib.sha256((cfg['output_path']/'interim/manual_review_annotations.csv').read_bytes()).hexdigest() if (cfg['output_path']/'interim/manual_review_annotations.csv').exists() else None,
                    source_code_sha256={str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                    outputs={n:{'rows':len(f),'columns':list(f.columns),'parquet_sha256':hashlib.sha256((processed/(n+'.parquet')).read_bytes()).hexdigest()} for n,f in datasets.items()})
    atomic_json(processed / 'build_manifest.json', manifest)
    LOG.info('%s: %s episodes, %s pitchers; outputs in %s', 'Finished' if source_complete else 'PARTIAL DATASET; failed transaction windows require resume', len(events), events.player_id.nunique(), processed)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config')
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--update', action='store_true', help='Refresh recent transaction windows; use cache for older periods')
    args = parser.parse_args(argv)
    cfg = load_settings(args.config)
    if args.start_date:
        cfg['start_date'] = date.fromisoformat(args.start_date)
    if args.end_date:
        cfg['end_date'] = date.fromisoformat(args.end_date)
    if cfg['start_date'] > cfg['end_date'] or cfg['end_date'] > date.today():
        parser.error('Requested dates must be ordered and cannot extend into the future')
    if not cfg['database_path'].is_file():
        parser.error('Existing project database is required: ' + str(cfg['database_path']))
    cfg['report_path'].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(cfg['report_path']/'pipeline.log')])
    result = run(cfg, args.offline, args.update)
    return 0 if result['metrics']['all_transaction_windows_available'] else 2


if __name__ == '__main__':
    sys.exit(main())
