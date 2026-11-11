"""Bounded XGBoost reference, with all selection inside development folds.

Random pitcher-game splitting is deliberately confined to this paper benchmark.
It must never replace the chronological UCL pipeline or its embargo rules.
"""
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..evaluation import rank_metrics
from .kang_cohort import window_labels
from .kang_spec import ROOT, FEATURE_NAMES, DOMAINS, atomic_json, digest


def xgb_class():
    try:
        from xgboost import XGBClassifier
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError('Install requirements-benchmark.txt. macOS needs libomp; see docs/kang_benchmark_spec.md.') from exc
    return XGBClassifier


def fit_xgb(x, y, cfg, seed):
    if y.nunique() < 2:
        raise ValueError('Insufficient outcome classes for the requested fit.')
    counts = y.value_counts()
    weights = y.map({k: len(y) / (2 * v) for k, v in counts.items()}).to_numpy()
    model = xgb_class()(**cfg['xgboost'], random_state=seed)
    # No test-based early stopping; XGBoost handles residual NaN natively.
    model.fit(x, y, sample_weight=weights)
    return model


def select_features(x, y, cfg, seed):
    """Training-only gain ranking plus greedy within-domain Pearson pruning.

    This is an explicit small approximation, not the paper's undocumented
    50-repetition stability screen. Transitive chains do not imply all pairs
    are redundant: each candidate is compared with already retained features.
    """
    available = [f for f in FEATURE_NAMES if x[f].notna().any() and x[f].nunique() > 1]
    gains = []
    for repeat in range(cfg['selector']['gain_refits']):
        subset, _ = train_test_split(np.arange(len(y)), train_size=cfg['selector']['subsample_fraction'],
                                    stratify=y, random_state=seed + repeat)
        model = fit_xgb(x.iloc[subset][available], y.iloc[subset], cfg, seed + repeat)
        gains.append(pd.Series(model.feature_importances_, index=available))
    mean_gain = pd.concat(gains, axis=1).mean(axis=1)
    order = sorted(available, key=lambda f: (-mean_gain[f], f))
    correlation = x[available].corr().abs()
    retained, records = [], []
    for feature in order:
        redundant = next((r for r in retained if DOMAINS[feature] == DOMAINS[r]
                          and correlation.loc[feature, r] > cfg['selector']['pearson_threshold']), None)
        if redundant is None:
            retained.append(feature)
        records.append(dict(feature=feature, mean_gain=float(mean_gain[feature]),
                            retained=redundant is None, redundant_with=redundant or '',
                            correlation=float(correlation.loc[feature, redundant]) if redundant else np.nan))
    for f in set(FEATURE_NAMES) - set(available):
        records.append(dict(feature=f, mean_gain=0., retained=False, redundant_with='constant/all missing', correlation=np.nan))
    return retained, pd.DataFrame(records)


def reserve_random_holdout(frame, cfg):
    if pd.to_datetime(frame.game_date).max() > pd.Timestamp('2025-12-31'):
        raise ValueError('2026 cannot enter benchmark training or holdout.')
    if frame.duplicated(['game_id', 'pitcher_id']).any():
        raise ValueError('Duplicate pitcher-games.')
    y = window_labels(frame, max(cfg['windows_days']))
    train, holdout = train_test_split(np.arange(len(frame)), test_size=cfg['test_fraction'],
                                     random_state=cfg['seed'], stratify=y)
    return np.sort(train), np.sort(holdout)


def cluster_auc_ci(frame, y, scores, repetitions, seed):
    clusters = frame[['pitcher_id', 'season']].astype(str).agg(':'.join, axis=1)
    codes, names = pd.factorize(clusters, sort=True)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        multiplicities = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        # Resample whole pitcher-seasons, retaining all their held-out rows.
        ix = np.repeat(np.arange(len(frame)), multiplicities[codes])
        if len(ix) and np.unique(y[ix]).size == 2:
            samples.append(rank_metrics(y[ix], scores[ix])['roc_auc'])
    if not samples:
        return np.nan, np.nan, 0
    low, high = np.quantile(samples, [.025, .975])
    return float(low), float(high), len(samples)


def run_task(frame, cfg, endpoint, role, input_fingerprint):
    task = endpoint + '_' + role
    directory = ROOT / cfg['cache'] / 'models' / task
    directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256((input_fingerprint + digest(__file__) + task).encode()).hexdigest()
    checkpoint = directory / 'complete.json'
    artifacts = ['cv.csv', 'selection.csv', 'predictions.parquet', 'model.ubj', 'freeze.json', 'splits.parquet']
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if result.get('signature') == signature and all((directory / p).exists() and
                digest(directory / p) == result.get('files', {}).get(p) for p in artifacts):
            print('Verified completed benchmark task: ' + task, flush=True)
            return result['metrics'], pd.read_csv(directory / 'cv.csv'), pd.read_csv(directory / 'selection.csv')
    print('Development CV and fold-local selection: ' + task, flush=True)
    development_ix, holdout_ix = reserve_random_holdout(frame, cfg)
    development = frame.iloc[development_ix].reset_index(drop=True)
    split = frame[['game_id', 'pitcher_id', 'season', 'game_date']].copy()
    split['partition'] = 'random_holdout'
    split.loc[development_ix, 'partition'] = 'development'
    split.to_parquet(directory / 'splits.parquet', index=False)
    rows = []
    for window in cfg['windows_days']:
        target = window_labels(development, window)
        if target.value_counts().min() < cfg['cv_folds']:
            raise ValueError('Too few positives/negatives for configured stratified CV: ' + task)
        folds = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg['seed'])
        for fold, (fit_ix, valid_ix) in enumerate(folds.split(development, target)):
            xfit = development.iloc[fit_ix][FEATURE_NAMES]
            yfit = target.iloc[fit_ix]
            features, _ = select_features(xfit, yfit, cfg, cfg['seed'] + fold)
            model = fit_xgb(xfit[features], yfit, cfg, cfg['seed'] + fold)
            probability = model.predict_proba(development.iloc[valid_ix][features])[:, 1]
            metrics = rank_metrics(target.iloc[valid_ix].to_numpy(), probability)
            rows.append(dict(endpoint=endpoint, role=role, window_days=window, fold=fold,
                             features=len(features), validation_rows=len(valid_ix),
                             positive_rows=int(target.iloc[valid_ix].sum()), **metrics))
        print('  ' + task + ': completed ' + str(window) + '-day development folds', flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(directory / 'cv.csv', index=False)
    means = cv.groupby('window_days').roc_auc.mean().sort_index()
    selected_window = int(means.idxmax())  # sorted index resolves exact ties to shorter window
    ytrain = window_labels(development, selected_window)
    selected, selection = select_features(development[FEATURE_NAMES], ytrain, cfg, cfg['seed'])
    model = fit_xgb(development[selected], ytrain, cfg, cfg['seed'])
    selection['endpoint'], selection['role'] = endpoint, role
    selection.to_csv(directory / 'selection.csv', index=False)
    model.save_model(directory / 'model.ubj')
    # Persist the chosen window/features/model BEFORE reading holdout outcomes.
    atomic_json(directory / 'freeze.json', dict(signature=signature, config=cfg,
        selected_window_days=selected_window, selected_features=selected,
        selection_rule=cfg['window_selection'], model_sha256=digest(directory / 'model.ubj'),
        split_sha256=digest(directory / 'splits.parquet'), evaluated_2026=False))
    holdout = frame.iloc[holdout_ix].reset_index(drop=True)
    target = window_labels(holdout, selected_window).to_numpy()
    probability = model.predict_proba(holdout[selected])[:, 1]
    low, high, nboot = cluster_auc_ci(holdout, target, probability, cfg['bootstrap_iterations'], cfg['seed'])
    metrics = dict(endpoint=endpoint, role=role, selected_window_days=selected_window,
        development_cv_auc=float(means.loc[selected_window]), selected_features=len(selected),
        development_rows=len(development), holdout_rows=len(holdout), positive_rows=int(target.sum()),
        represented_events=int(holdout.loc[target == 1, 'injury_event_id'].nunique()),
        prevalence=float(target.mean()), **rank_metrics(target, probability),
        roc_auc_ci_low=low, roc_auc_ci_high=high, valid_cluster_bootstraps=nboot,
        brier=float(np.mean((probability - target) ** 2)), mean_risk=float(probability.mean()),
        shared_pitchers=len(set(development.pitcher_id) & set(holdout.pitcher_id)),
        holdout_unique_pitchers=holdout.pitcher_id.nunique(),
        shared_case_events=len(set(development.injury_event_id.dropna()) & set(holdout.injury_event_id.dropna())))
    metrics['pr_lift'] = metrics['average_precision'] / metrics['prevalence']
    predictions = holdout[['game_id', 'pitcher_id', 'season', 'game_date', 'injury_event_id', 'lead_days']].copy()
    predictions['target'], predictions['probability'] = target, probability
    predictions.to_parquet(directory / 'predictions.parquet', index=False)
    atomic_json(checkpoint, dict(signature=signature, metrics=metrics,
                                files={p: digest(directory / p) for p in artifacts}))
    return metrics, cv, selection
