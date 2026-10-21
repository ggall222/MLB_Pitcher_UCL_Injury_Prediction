"""Executable integrity checks and independent SQL temporal checks, with retained flags."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import ROOT


def run(db, cfg, feature_columns):
    checks = []
    for statement in (ROOT / 'sql/validation.sql').read_text().split(';'):
        if statement.strip():
            checks.extend(db.query(statement).to_dict('records'))
    temporal = db.query('''SELECT a.game_id, a.pitcher_id, m.avg_fb_velo_prev_5,
        (SELECT CASE WHEN count(*)=5 AND count(avg_fb_velo)>=2 THEN avg(avg_fb_velo) END
         FROM (SELECT b.avg_fb_velo FROM appearances b WHERE b.pitcher_id=a.pitcher_id
               AND b.game_date<a.game_date ORDER BY b.game_date DESC,b.scheduled_start DESC,b.game_id DESC LIMIT 5)) AS expected
        FROM appearances a JOIN mechanical_features m USING(game_id,pitcher_id)''')
    bad = ~np.isclose(temporal.avg_fb_velo_prev_5.to_numpy(dtype=float), temporal.expected.to_numpy(dtype=float), equal_nan=True, atol=1e-8)
    checks.append({'check_name': 'independent_prior_5_recalculation', 'severity': 'error', 'violations': int(bad.sum())})
    leakage_terms = ['until_next', 'return_date', 'surgery_date', 'diagnosis', 'next_known', 'label_', 'ucl_injury_next']
    bad_features = [c for c in feature_columns if any(term in c for term in leakage_terms)]
    checks.append({'check_name': 'feature_allowlist_temporal_fields', 'severity': 'error', 'violations': len(bad_features)})
    counts = {name: int(db.scalar(f'SELECT count(*) FROM "{name}"'))
              for name in db.query("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").table_name}
    states = db.query('SELECT source,status,count(*) AS items,sum(row_count) AS rows FROM ingestion_state GROUP BY source,status ORDER BY source,status')
    failed = int(db.scalar("SELECT count(*) FROM ingestion_state WHERE status='failed'"))
    checks.append({'check_name': 'failed_source_items', 'severity': 'warning', 'violations': failed})
    model = db.query('SELECT * FROM model_features')
    missingness = pd.DataFrame({'variable': model.columns, 'missing_rows': model.isna().sum().values,
                               'missing_fraction': model.isna().mean().values})
    missingness.to_csv(cfg['report_path'] / 'model_missingness.csv', index=False)
    labels = {}
    for col in model.columns:
        if col.startswith('ucl_injury_next_'):
            labels[col] = {'positive': int(model[col].eq(True).sum()), 'negative': int(model[col].eq(False).sum()),
                           'unknown': int(model[col].isna().sum())}
    positives = db.query('''SELECT DISTINCT pitcher_id,pitcher_name,next_known_ucl_onset,label_onset_basis
                            FROM model_features WHERE training_eligible AND ucl_injury_next_30d''')
    positives.to_csv(cfg['report_path'] / 'positive_cases_review.csv', index=False)
    db.query('SELECT * FROM ingestion_issues').to_csv(cfg['report_path'] / 'ingestion_issues.csv', index=False)
    db.query("SELECT * FROM injuries WHERE known_onset_date IS NOT NULL").to_parquet(cfg['report_path'] / 'linked_injury_evidence.parquet', index=False)
    checks = [{**r, 'violations': int(r['violations'])} for r in checks]
    errors = sum(r['violations'] for r in checks if r['severity'] == 'error')
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'passed': errors == 0,
              'error_violations': errors, 'checks': checks, 'row_counts': counts,
              'source_status': states.to_dict('records'), 'labels': labels,
              'training_rows_30d': int(db.scalar('SELECT count(*) FROM training_pitcher_appearances')),
              'starter_training_rows_30d': int(db.scalar('SELECT count(*) FROM starter_training_appearances')),
              'positive_pitchers_30d': len(positives), 'feature_count': len(feature_columns),
              'note': 'Passing checks establish internal consistency, not clinical validity or complete injury ascertainment.'}
    (cfg['report_path'] / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str))
    lines = ['# Validation report', '', f"Structural/temporal checks: {'PASS' if report['passed'] else 'FAIL'}", '',
             f"Training rows with observed 30-day outcome: {report['training_rows_30d']:,}; starters: {report['starter_training_rows_30d']:,}.",
             f"Distinct positive pitchers: {len(positives)}. Candidate features: {len(feature_columns)}.", '',
             '| Check | Severity | Flagged rows/items |', '|---|---|---:|']
    lines += [f"| {r['check_name']} | {r['severity']} | {r['violations']} |" for r in checks]
    lines += ['', '## Table counts', '', '| Table/view | Rows |', '|---|---:|']
    lines += [f'| {k} | {v:,} |' for k, v in counts.items()]
    lines += ['', '## Outcome coverage', '', '| Target | Positive | Negative | Unknown |', '|---|---:|---:|---:|']
    lines += [f"| {k} | {v['positive']} | {v['negative']} | {v['unknown']} |" for k, v in labels.items()]
    lines += ['', 'Unknown outcomes are retained as NULL. Warnings remain in raw data or ingestion_issues.',
              'Inspect model_missingness.csv and positive_cases_review.csv before fitting. A short sample has few independent events.',
              'MLB elbow-IL-to-surgery linkage is a time-based heuristic; clinical diagnosis and actual onset can predate public evidence.',
              'Roegele explicitly describes incomplete media coverage and first-of-month date placeholders.']
    (cfg['report_path'] / 'validation_report.md').write_text('\n'.join(lines) + '\n')
    write_build_summary(db, cfg, report)
    return report


def write_build_summary(db, cfg, report):
    counts = report['row_counts']
    failed = db.query("SELECT source,item_key,detail FROM ingestion_state WHERE status='failed'")
    source_text = 'All configured data sources were retrieved successfully; no failed source items.' if failed.empty else failed.to_string(index=False)
    lines = ['# Executed build summary', '',
        f"Pitch sample: {cfg['start_date']} through {cfg['end_date']}. Label follow-up ends {cfg.get('observation_end', 'see run manifest')}.", '',
        '## Retrieved and populated', '',
        f"- {counts['pitches']:,} raw Statcast records; {counts['appearances']:,} pitcher appearances.",
        f"- {counts['pitch_tracking']:,} supplemental physical pitches from MLB game feeds.",
        f"- {counts['games']:,} scheduled games; {counts['boxscore_appearances']:,} official pitching appearances including follow-up.",
        f"- {counts['pitchers']:,} identity records; {counts['teams']} teams; {counts['roster_snapshots']:,} historical roster records.",
        f"- {counts['transactions']:,} transactions; {counts['season_pitching']:,} prior-season pitching totals.",
        f"- {counts['injuries']:,} Roegele procedure records (including historical, minor-league and non-pitcher records for audit).",
        f"- {report['training_rows_30d']:,} eligible observed 30-day targets, including {report['starter_training_rows_30d']:,} starting appearances.",
        f"- {report['labels']['ucl_injury_next_30d']['positive']} positive 30-day appearance rows representing {report['positive_pitchers_30d']} pitchers; {report['feature_count']} allowlisted candidate features.", '',
        '## Source availability', '', source_text, '',
        '## Assumptions and quality concerns', '',
        '- Public elbow-IL announcement is the onset proxy; administrative retroactive IL dates remain separate. Injury-to-procedure linkage is temporal and needs case review.',
        '- First-of-month source surgery dates may be placeholders and are represented as uncertain ranges. Unknown-onset procedures do not create precise injury targets.',
        '- Post-known-injury rows are excluded from the risk set. No in-study reentry after elbow episodes; missing follow-up remains NULL.',
        '- Last healthy appearance and clinical diagnosis/discomfort dates are not fabricated. Only last observed pre-report appearance is calculated.',
        '- IVB uses native MLB telemetry, aligned by physical-pitch ordinal to handle automatic ball/strike numbering differences; it is separate from Savant movement.',
        '- Early 3/5/10-appearance histories and minimum-5-observation baselines are necessarily missing. Current web snapshots can revise historical reports.',
        '- This sample contains too few independent injury events for a reliable predictive-performance claim. Public injury reporting is incomplete.', '',
        '| Validation warning | Retained/flagged records |', '|---|---:|']
    lines += [f"| {r['check_name']} | {r['violations']} |" for r in report['checks'] if r['severity']=='warning']
    lines += ['', '## Recommended next steps', '',
        'Expand to multiple seasons, audit each positive episode and uncertain/name-matched procedure, select a defensible starter cohort, and evaluate with chronological/pitcher-aware splits plus a purge gap at least as long as the target horizon. Fit missing-value handling and feature selection inside training folds.', '',
        'Refresh with `python -m src.pipeline --update`; extend dates in config or with `--start`/`--end`. See README for cached offline rebuilds and source correction refreshes.']
    (cfg['report_path'] / 'build_summary.md').write_text('\n'.join(lines) + '\n')
