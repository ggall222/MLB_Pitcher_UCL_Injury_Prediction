"""Fixed exploratory baselines, chronological partitions, and alert-budget evaluation.

No hyperparameter search or test-driven feature selection. These comparisons are
category-versus-IL-free, not estimates of intrinsic biological predictability.
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.evaluation import rank_metrics, evaluate_predictions
from .common import CFG, CACHE, CATEGORIES, table
from .cohort import label_frame


def ap_cluster_ci(frame):
    """Pitcher bootstrap for AP and AP/prevalence, preserving all that pitcher's rows."""
    f = frame.sort_values('score', ascending=False, kind='stable')
    y = f.target.to_numpy(dtype=int)
    _, cluster = np.unique(f.pitcher_id, return_inverse=True)
    n = int(cluster.max()) + 1
    scores = f.score.to_numpy()
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    rng = np.random.default_rng(CFG['seed'])
    ap, lifts = [], []
    for _ in range(CFG['bootstrap_replicates']):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        weights = counts[cluster]
        tp = np.cumsum(weights * y)[ends]
        total = np.cumsum(weights)[ends]
        if tp[-1] == 0:
            continue
        precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
        value = np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1]
        ap.append(value)
        lifts.append(value / (tp[-1] / total[-1]))
    return dict(ap_low=np.quantile(ap, .025), ap_high=np.quantile(ap, .975),
                lift_low=np.quantile(lifts, .025), lift_high=np.quantile(lifts, .975)) if ap else {}


def fit_scores(train, targets, features):
    """Fit preprocessing and regularization on TRAIN ONLY; NULLs remain intact on disk."""
    pipe = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
        LogisticRegression(C=CFG['logistic_C'], max_iter=1500, solver='lbfgs', random_state=CFG['seed']))
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe.fit(train[features].replace([np.inf, -np.inf], np.nan), train.target)
        scores = [pipe.predict_proba(f[features].replace([np.inf, -np.inf], np.nan))[:, 1] for f in targets]
    return pipe, scores


def run_predictions(a, events, spec):
    features = spec.loc[spec.selected_for_prediction, 'name'].tolist()
    metrics, inventory, predictions, budgets, leads, curves, coefficients, thresholds = [], [], [], [], [], [], [], []
    for category in CATEGORIES:
        for horizon in CFG['horizons']:
            f = label_frame(a, category, horizon)
            partitions = {name: f.loc[f.partition.eq(name)].copy() for name in ['train', 'validation', 'test']}
            counts = {name: part.loc[part.target.eq(1), 'next_event_id'].nunique() for name, part in partitions.items()}
            enough = (counts['train'] >= CFG['minimum_training_events'] and counts['validation'] >= CFG['minimum_validation_events']
                and counts['test'] >= CFG['minimum_test_events'])
            for name, part in partitions.items():
                inventory.append(dict(category=category, horizon=horizon, partition=name, rows=len(part),
                    positives=int(part.target.sum()), events=counts[name], pitchers=part.pitcher_id.nunique(),
                    prevalence=part.target.mean(), model_run=enough))
            if not enough:
                print(f'Skip {category} {horizon}d: event counts {counts}', flush=True)
                continue
            print(f'Fit {category}, {horizon}d: event counts {counts}', flush=True)
            train, valid, test = [partitions[n] for n in ['train', 'validation', 'test']]
            pipe, (vs, ts) = fit_scores(train, [valid, test], features)
            valid['score'], test['score'] = vs, ts
            for name, part in [('validation', valid), ('test', test)]:
                result = dict(category=category, horizon=horizon, partition=name, rows=len(part),
                    positives=int(part.target.sum()), events=counts[name], prevalence=part.target.mean(),
                    **rank_metrics(part.target, part.score))
                result['pr_lift'] = result['average_precision'] / result['prevalence']
                if name == 'test':
                    result.update(ap_cluster_ci(part))
                metrics.append(result)
            predictions.append(test[['row_id','pitcher_id','game_id','game_date','season','is_starter','target',
                'score','next_event_id','next_event_date','next_event_lead_days','next_event_strict_qa']].assign(category=category, horizon=horizon))
            # All eligible date-partition events are the primary event denominator, including
            # those with no warning outing. Observable-event recall is reported separately.
            ev = events.loc[events.target_eligible & events.analysis_group.eq(category)
                & events.anchor_date.gt(CFG['validation_end']) & events.anchor_date.le(CFG['test_end'])].copy()
            ev = ev.rename(columns={'player_id': 'pitcher_id', 'anchor_date': 'label_reference_date'})
            b, l = evaluate_predictions(test, ev, target='target', rates=CFG['alert_rates'], warning_window=(7, horizon))
            budgets.append(b.assign(category=category, horizon=horizon))
            leads.append(l.assign(category=category, horizon=horizon))
            # A deployable-style threshold is fixed from validation; test alert rates can drift.
            for rate in CFG['alert_rates']:
                threshold = float(valid.score.quantile(1 - rate, interpolation='higher'))
                mask = test.score.ge(threshold)
                tp = int(test.loc[mask, 'target'].sum())
                thresholds.append(dict(category=category, horizon=horizon, validation_alert_budget=rate,
                    threshold=threshold, validation_actual_rate=valid.score.ge(threshold).mean(),
                    test_actual_rate=mask.mean(), test_precision=tp / mask.sum() if mask.sum() else np.nan,
                    test_recall=tp / test.target.sum(), alerts=int(mask.sum()), positives=int(test.target.sum())))
            if horizon == 30:
                precision, recall, _ = precision_recall_curve(test.target, test.score)
                sample = np.unique(np.linspace(0, len(precision)-1, min(600, len(precision))).astype(int))
                curves.append(pd.DataFrame({'category':category, 'precision':precision[sample], 'recall':recall[sample],
                                            'prevalence':test.target.mean()}))
                names = pipe.named_steps['simpleimputer'].get_feature_names_out(features)
                coefficients.append(pd.DataFrame({'category':category, 'feature':names,
                    'standardized_log_odds':pipe.named_steps['logisticregression'].coef_[0]}))
    table(pd.DataFrame(inventory), 'prediction_cohort_counts')
    table(pd.DataFrame(metrics), 'prediction_metrics')
    table(pd.concat(budgets, ignore_index=True), 'alert_budget_metrics')
    table(pd.concat(leads, ignore_index=True), 'warning_lead_times')
    table(pd.concat(curves, ignore_index=True), 'precision_recall_curves')
    table(pd.concat(coefficients, ignore_index=True), 'model_coefficients')
    table(pd.DataFrame(thresholds), 'validation_fixed_thresholds')
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(CACHE / 'test_predictions.parquet', index=False)
    return pd.DataFrame(metrics)


def run_family_analysis(a, spec):
    """Paired validation-only ablations; no additional test-set winner selection."""
    features = spec.loc[spec.selected_for_prediction, 'name'].tolist()
    families = sorted(spec.loc[spec.selected_for_prediction, 'family'].unique())
    results = []
    for category in ['Elbow / UCL', 'Shoulder', 'Forearm', 'Lower body']:
        f = label_frame(a, category, 30)
        train = f.loc[f.partition.eq('train')].copy()
        valid = f.loc[f.partition.eq('validation')].copy()
        if min(train.target.sum(), valid.target.sum()) < 10:
            continue
        # Two predeclared interactions built only from already-observed existing features.
        interaction_names = ['velocity_x_workload', 'release_variability_x_workload']
        for part in [train, valid]:
            part[interaction_names[0]] = part.velo_z * part.workload_acceleration_7d_vs_21d
            part[interaction_names[1]] = part.release_z_sd_z * part.workload_acceleration_7d_vs_21d
        variants = [('All selected', features), ('With two interactions', features + interaction_names)]
        for family in families:
            removed = set(spec.loc[spec.family.eq(family), 'name'])
            variants.append(('Without ' + family, [c for c in features if c not in removed]))
        for name, columns in variants:
            _, (score,) = fit_scores(train, [valid], columns)
            metric = rank_metrics(valid.target, score)
            results.append(dict(category=category, variant=name, partition='validation', predictors=len(columns),
                rows=len(valid), positives=int(valid.target.sum()), events=valid.loc[valid.target.eq(1),'next_event_id'].nunique(),
                prevalence=valid.target.mean(), **metric))
        print(f'Completed validation-only family comparisons: {category}', flush=True)
    result = pd.DataFrame(results)
    baseline = result.loc[result.variant.eq('All selected')].set_index('category').average_precision
    result['ap_change_vs_all'] = result.average_precision - result.category.map(baseline)
    table(result, 'feature_family_ablation')
    return result


def sensitivity_metrics():
    """Frozen-model test evaluation on predeclared subgroups, without refitting."""
    predictions = pd.read_parquet(CACHE / 'test_predictions.parquet')
    rows = []
    for (category, h), f in predictions.groupby(['category','horizon']):
        if h != 30:
            continue
        for name, mask in [('All', np.ones(len(f), dtype=bool)), ('Starting outings', f.is_starter),
            ('Relief outings', ~f.is_starter),
            ('Strict-QA positives and common negatives', f.target.eq(0) | f.next_event_strict_qa)]:
            s = f.loc[mask]
            if not s.target.sum():
                continue
            metric = rank_metrics(s.target, s.score)
            rows.append(dict(category=category, subgroup=name, rows=len(s), positives=int(s.target.sum()),
                events=s.loc[s.target.eq(1), 'next_event_id'].nunique(), prevalence=s.target.mean(),
                pr_lift=metric['average_precision']/s.target.mean(), **metric))
    return table(pd.DataFrame(rows), 'prediction_sensitivity')
