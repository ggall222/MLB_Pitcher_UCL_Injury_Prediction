"""Prepare or run the bounded Kang reference stage; preserve all older stages.

Usage: python -m src.injury_analysis.kang_benchmark [--fit]
"""
import argparse
import json
import platform
from importlib.metadata import version

import duckdb
import pandas as pd

from .kang_spec import ROOT, FEATURE_NAMES, load_config, atomic_json, digest
from .kang_cohort import load_inputs, construct_cohorts


def reporting(cfg, tables, metrics=None, curves=None):
    output = ROOT / cfg['output']
    support = pd.read_csv(tables / 'cohort_support.csv')
    primary = support[support.control_rule.eq(cfg['primary_control_rule']) & support.window_days.eq(28)]
    body = [
        '# Kang benchmark foundation',
        'Approximate reference implementation, 2017–2025 regular seasons. No 2026 modeling rows or outcomes were loaded. '
        'This random-game benchmark is separate from the final chronological UCL study.',
        'See [method specification](../../docs/kang_benchmark_spec.md), '
        '[feature crosswalk](tables/feature_crosswalk.csv), and '
        '[comparison table](../../docs/kang_benchmark_comparison.csv).',
        '## Primary cohort support at the fixed 28-day audit window',
        primary[['endpoint', 'role', 'rows', 'pitcher_seasons', 'unique_pitchers', 'case_seasons',
                 'control_seasons', 'represented_events', 'positive_rows']].to_markdown(index=False),
        'Counts use no-IL control seasons. Alternative target-specific controls, each window, '
        'season exclusions, ambiguous first events and residual missingness are tabulated separately. '
        'The 28-day audit does not select the fitted window.',
    ]
    if metrics is not None:
        body += ['## Bounded reference fits', metrics[['endpoint', 'role', 'selected_window_days',
            'development_cv_auc', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high',
            'average_precision', 'prevalence', 'represented_events', 'selected_features']].round(4).to_markdown(index=False),
            'Each window and feature subset was chosen inside the random development partition; '
            'only the frozen model was scored on its reserved random-game holdout. '
            'Windows maximize mean development CV AUC on the declared weekly grid. They are **not** '
            'replications of the published GAM plateau estimates. Bootstrap intervals resample pitcher-seasons, '
            'but do not remove overlap of pitchers/events across training and holdout.',
            'These discrimination results do not establish prospective UCL warning, biological onset, '
            'calibration, or an actionable alert policy. Class weighting changes the probability scale; '
            'Brier/mean-risk are descriptive diagnostics, not claims of calibrated population risk.',
            f"{int(metrics.selected_window_days.eq(max(cfg['windows_days'])).sum())} of {len(metrics)} tasks selected "
            f"the {max(cfg['windows_days'])}-day grid boundary. This is a descriptive development result, "
            'not evidence for an optimal biological window or replication of the published plateau.',
        ]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        for ax, ((endpoint, role), sub) in zip(axes.flat, curves.groupby(['endpoint', 'role'], sort=True)):
            s = sub.groupby('window_days').roc_auc.agg(['mean', 'std'])
            ax.errorbar(s.index, s['mean'], yerr=s['std'], marker='o', capsize=3)
            row = metrics[metrics.endpoint.eq(endpoint) & metrics.role.eq(role)].iloc[0]
            ax.axvline(row.selected_window_days, color='grey', ls='--')
            ax.set_title(endpoint.title() + ' / ' + role)
            ax.set_xlabel('Cumulative pre-IL window (days)')
            ax.set_ylabel('Development CV ROC-AUC')
            ax.set_xticks(cfg['windows_days'])
        fig.suptitle('Kang-style reference: window comparison\nBars are fold SD, not confidence intervals; no GAM plateau inferred')
        fig.tight_layout()
        (output / 'figures').mkdir(exist_ok=True)
        fig.savefig(output / 'figures/development_window_comparison.png', dpi=180)
        plt.close(fig)
    body += ['## Next implementation step',
             'Resolve control/label and release-Y/extension ambiguities against author code if obtainable, '
             'then freeze the reference specification. Next add an adjudicated UCL endpoint comparison '
             'while holding features/model/window policy fixed; subsequently introduce chronological evaluation '
             'and a limited family-by-lead-time analysis. Preserve 2026 until development is frozen.',
             'Existing PCA/PLS, consolidation, nonlinear and temporal results remain exploratory context; '
             'their selected features and scores are not inputs to this benchmark.']
    (output / 'summary.md').write_text('\n\n'.join(body) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--fit', action='store_true', help='Run four fixed-parameter reference tasks and weekly development CV.')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out = ROOT / cfg['output']
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    db = ROOT / cfg['database']
    before = db.stat()
    appearances, events, mapping, manifest = load_inputs(cfg)
    cohorts, seasons, first, support, flow = construct_cohorts(appearances, events, cfg)
    # An eligible placement may have no preceding observed outing that season.
    # Keep it visible rather than confusing cohort events with represented events.
    primary_frames = [f for (rule, _, _), f in cohorts.items() if rule == cfg['primary_control_rule']]
    opportunities = pd.concat(primary_frames).loc[lambda x: x.is_case_season].groupby('injury_event_id').size()
    first['preinjury_appearances'] = first.injury_event_id.map(opportunities).fillna(0).astype(int)
    first['has_preinjury_appearance'] = first.preinjury_appearances.gt(0)
    mapping.to_csv(tables / 'feature_crosswalk.csv', index=False)
    support.to_csv(tables / 'cohort_support.csv', index=False)
    flow.to_csv(tables / 'cohort_flow.csv', index=False)
    seasons.to_csv(ROOT / cfg['cache'] / 'season_eligibility.csv', index=False)
    first.to_parquet(ROOT / cfg['cache'] / 'first_event_audit.parquet', index=False)
    season_summary = seasons.groupby('season').agg(pitcher_seasons=('pitcher_id', 'size'),
        eligible_pitcher_seasons=('eligible', 'sum'), fewer_than_five=('passes_appearances', lambda x: (~x).sum()),
        above_missingness_limit=('passes_missingness', lambda x: (~x).sum()),
        ambiguous_role=('season_role', lambda x: x.eq('ambiguous').sum())).reset_index()
    season_summary.to_csv(tables / 'season_eligibility.csv', index=False)
    missing = []
    for season, frame in appearances[appearances.position.isin(cfg['positions'])].groupby('season'):
        for feature in FEATURE_NAMES:
            missing.append(dict(season=season, feature=feature, rows=len(frame),
                                missing_rows=int(frame[feature].isna().sum()), missing_fraction=frame[feature].isna().mean()))
    pd.DataFrame(missing).to_csv(tables / 'feature_missingness.csv', index=False)
    per_season = []
    for (rule, endpoint, role), f in cohorts.items():
        for season, part in f.groupby('season'):
            per_season.append(dict(control_rule=rule, endpoint=endpoint, role=role, season=season,
                rows=len(part), pitcher_seasons=part.pitcher_id.nunique(),
                case_seasons=part.loc[part.is_case_season, 'pitcher_id'].nunique(),
                control_seasons=part.loc[part.is_control, 'pitcher_id'].nunique()))
    pd.DataFrame(per_season).to_csv(tables / 'cohort_by_season.csv', index=False)
    first.groupby(['endpoint', 'eligible_case', 'has_preinjury_appearance', 'case_exclusion'], dropna=False).size().rename('events').reset_index().to_csv(
        tables / 'event_eligibility.csv', index=False)
    print(flow.to_string(index=False), flush=True)
    print(support[support.control_rule.eq(cfg['primary_control_rule']) & support.window_days.eq(28)].to_string(index=False), flush=True)
    atomic_json(out / 'benchmark_spec.json', dict(config=cfg, input_manifest=manifest,
        packages={p: version(p) for p in ['duckdb', 'pandas', 'numpy', 'scikit-learn', 'xgboost']}, python=platform.python_version(),
        feature_names=FEATURE_NAMES, development_period='2017–2025 random games; not chronological',
        protected_test_year=2026, executed_modeling=args.fit))
    metrics, curves = None, None
    if args.fit:
        from .kang_models import run_task
        results, curve_list, selections = [], [], []
        for (rule, endpoint, role), frame in cohorts.items():
            if rule != cfg['primary_control_rule']:
                continue
            result, curve, selection = run_task(frame, cfg, endpoint, role, manifest['fingerprint'])
            results.append(result)
            curve_list.append(curve)
            selections.append(selection)
        metrics, curves = pd.DataFrame(results), pd.concat(curve_list, ignore_index=True)
        metrics.to_csv(tables / 'reference_metrics.csv', index=False)
        curves.to_csv(tables / 'development_cv.csv', index=False)
        pd.concat(selections).to_csv(tables / 'retained_reference_features.csv', index=False)
    reporting(cfg, tables, metrics, curves)
    after = db.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Database changed during benchmark execution; investigate concurrent writers.')
    atomic_json(out / 'verification.json', dict(database_unchanged=True, read_only=True,
        earliest_date=str(appearances.game_date.min()), latest_date=str(appearances.game_date.max()),
        no_2026_rows=bool(appearances.game_date.max() < pd.Timestamp('2026-01-01')),
        duplicate_appearance_keys=int(appearances.duplicated(['game_id', 'pitcher_id']).sum()),
        cohort_reports_written=True, reference_models_completed=4 if args.fit else 0,
        source_fingerprint=manifest['fingerprint']))
    print('Kang foundation outputs: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
