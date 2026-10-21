"""CLI: historical research build, resumable ingestion, cached offline reproduction."""
import argparse
from datetime import date, timedelta
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from .config import load_config, ROOT
from .database import Database
from .http import HttpClient
from . import ingest_mlb, ingest_statcast, ingest_injuries
from . import build_appearances, build_workload_features, build_mechanical_features, build_baselines
from . import build_labels, build_model_dataset, validation
from .utils import now

LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--update', action='store_true', help='Resume missing days and ingest newer configured dates (default)')
    mode.add_argument('--full-refresh', action='store_true', help='Resume all configured sources and rebuild derived tables; preserve raw records')
    parser.add_argument('--refresh-cache', action='store_true', help='Explicitly refetch source responses, including stored pitch dates and boxscores')
    parser.add_argument('--historical', action='store_true', help='Expand/audit the configured historical central database')
    parser.add_argument('--ingest-only', action='store_true', help='Populate raw sources and preserve existing derived tables')
    parser.add_argument('--derive-only', action='store_true', help='Recompute derived research tables from stored sources')
    parser.add_argument('--phase', choices=['all','statcast','support','tracking'], default='all')
    parser.add_argument('--offline', action='store_true', help='Use cached responses only')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--start', type=date.fromisoformat)
    parser.add_argument('--end', type=date.fromisoformat, help='Use --end YYYY-MM-DD or configure end_date: null for yesterday')
    mode.add_argument('--validate-only', action='store_true')
    return parser.parse_args()


def run(args):
    cfg = load_config(args.config)
    cfg['start_date'] = args.start or cfg['start_date']
    cfg['end_date'] = args.end or cfg['end_date']
    if cfg['end_date'] >= date.today() or cfg['start_date'] > cfg['end_date']:
        raise ValueError('Date range must end before today and start on/before end')
    cfg['workload_start_date'] = date(cfg['start_date'].year, 1, 1)
    cfg['observation_end'] = min(date.today() - timedelta(days=1), cfg['end_date'] + timedelta(days=cfg['followup_days']))
    for key in ['report_path', 'export_path', 'cache_path']:
        cfg[key].mkdir(parents=True, exist_ok=True)
    stamp = now().strftime('%Y%m%dT%H%M%S')
    log_dir = cfg['report_path'] / 'logs'; log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / (stamp + '.log'))])
    # Full refresh now refreshes derived research tables; historical raw records stay intact.
    db = Database(cfg['database_path']); db.initialize()
    client = HttpClient(cfg, offline=args.offline, refresh=args.refresh_cache)
    try:
        if args.validate_only:
            features = json.loads((cfg['export_path'] / 'feature_columns.json').read_text())
            if cfg.get('historical_build'):
                from .research_reports import validate
                return validate(db,cfg,features,db.query('SELECT * FROM model_features'))
            return validation.run(db, cfg, features)
        if args.historical or cfg.get('historical_build'):
            from .historical_ingestion import ingest as historical_ingest
            if not args.derive_only:
                result = historical_ingest(db, client, cfg, args.phase)
                if args.ingest_only:
                    return result
            from .research_pipeline import build as build_research
            return build_research(db, client, cfg)
        stored_start = db.scalar('SELECT min(game_date) FROM pitches')
        stored_end = db.scalar('SELECT max(game_date) FROM pitches')
        if stored_start and cfg['start_date'] > stored_start:
            cfg['start_date'] = stored_start
            cfg['workload_start_date'] = date(stored_start.year, 1, 1)
        if stored_end and cfg['end_date'] < stored_end:
            cfg['end_date'] = stored_end
            cfg['observation_end'] = min(date.today() - timedelta(days=1), stored_end + timedelta(days=cfg['followup_days']))
        for year in range(cfg['start_date'].year, cfg['end_date'].year + 1):
            ingest_mlb.ingest_teams(db, client, year)
        ingest_mlb.ingest_schedule(db, client, cfg['workload_start_date'], cfg['observation_end'])
        ingest_statcast.ingest(db, client, cfg)
        ingest_mlb.ingest_boxscores(db, client, cfg['workload_start_date'], cfg['observation_end'])
        ingest_mlb.ingest_pitch_tracking(db, client, cfg)
        ingest_mlb.ingest_transactions(db, client, cfg['start_date'] - timedelta(days=cfg['injury_link_lookback_days']), cfg['observation_end'])
        ingest_mlb.ingest_rosters(db, client, cfg['start_date'])
        ingest_mlb.ingest_previous_seasons(db, client, cfg['start_date'].year, cfg['end_date'].year)
        ingest_mlb.ingest_people(db, client)
        injury_ok = ingest_injuries.ingest(db, client, cfg)
        ingest_mlb.ingest_people(db, client, missing_only=True)
        ingest_injuries.link_onsets(db, cfg)
        LOG.info('Building appearance metrics')
        appearances, types = build_appearances.build(db, cfg)
        box = db.query('SELECT b.*, g.game_date FROM boxscore_appearances b JOIN games g USING(game_id) WHERE NOT g.timing_uncertain')
        seasons = db.query('SELECT * FROM season_pitching')
        injuries = db.query('SELECT * FROM injuries')
        transactions = db.query('SELECT * FROM transactions')
        LOG.info('Building historical workloads, mechanics and pitcher baselines for %s appearances', len(appearances))
        workload = build_workload_features.build(appearances, box, seasons, cfg)
        mechanical = build_mechanical_features.build(appearances, cfg)
        baselines = build_baselines.build(appearances, cfg)
        complete = injury_ok and not db.scalar("SELECT count(*) FROM ingestion_state WHERE source IN ('schedule','transactions','boxscore') AND status='failed'")
        labels = build_labels.build(appearances, injuries, transactions, box, cfg, sources_complete=bool(complete))
        appearances['valid_starting_appearance'] = appearances.is_starter & appearances.quality_eligible
        model, features = build_model_dataset.build(appearances, workload, mechanical, baselines, labels)
        LOG.info('Persisting derived tables atomically')
        with db.transaction():
            for name, frame, keys in [('appearances', appearances, ['game_id','pitcher_id']),
                    ('appearance_pitch_metrics', types, ['game_id','pitcher_id','pitch_type']),
                    ('workload_features', workload, ['game_id','pitcher_id']),
                    ('mechanical_features', mechanical, ['game_id','pitcher_id']),
                    ('pitcher_baselines', baselines, ['game_id','pitcher_id']),
                    ('injury_labels', labels, ['game_id','pitcher_id']),
                    ('model_features', model, ['game_id','pitcher_id'])]:
                db.replace_derived(name, frame, keys)
            db.con.execute((ROOT / 'sql/views.sql').read_text())
        model.to_parquet(cfg['export_path'] / 'model_ready_pitcher_appearances.parquet', index=False)
        (cfg['export_path'] / 'feature_columns.json').write_text(json.dumps(features, indent=2) + '\n')
        from .dictionary import write_dictionary
        write_dictionary(db, cfg, features)
        report = validation.run(db, cfg, features)
        ddl = db.query("SELECT sql FROM duckdb_tables() WHERE table_name IN ('appearances','appearance_pitch_metrics','workload_features','mechanical_features','pitcher_baselines','injury_labels','model_features') ORDER BY table_name")
        (cfg['report_path'] / 'derived_schema.sql').write_text('\n\n'.join(ddl.sql) + '\n')
        LOG.info('Complete: %s pitches, %s appearances; %s eligible labeled rows; validation=%s',
                 report['row_counts']['pitches'], len(appearances), report['training_rows_30d'], report['passed'])
        return report
    finally:
        manifest = {'configuration': {k: str(v) if isinstance(v, (date, Path)) else v for k,v in cfg.items()},
                    'source_requests': client.records, 'python': sys.version,
                    'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for folder in ['src', 'sql', 'config'] for p in (ROOT / folder).glob('*') if p.is_file()}}
        (cfg['report_path'] / ('run_manifest_' + stamp + '.json')).write_text(json.dumps(manifest, indent=2, default=str))
        db.close()


def main():
    result = run(parse_args())
    if not result['passed']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
