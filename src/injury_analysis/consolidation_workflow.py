"""Resume-aware orchestration; completed artifacts are read, not regenerated.

Run ``python -m src.injury_analysis.consolidation_workflow``. Database reads use
the original read-only input loader. A changed source signature stops the run.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .consolidation_common import CACHE, OUT, CFG, inputs, read_table, table, snapshot, save_manifest


def complete(*names):
    """A zero-byte/truncated CSV is never an accepted stage checkpoint."""
    for name in names:
        path = OUT / 'tables' / (name + '.csv')
        if not path.exists():
            return False
        try:
            if pd.read_csv(path).empty:
                return False
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return False
    return True


def reconcile_metadata():
    """Repair descriptive metadata without replacing any original numerical table."""
    from .consolidation_inventory import classify
    if not complete('reconciled_feature_inventory', 'feature_definition_updates',
                    'dictionary_and_timing_caveats', 'cross_group_summary_canonical', 'event_support'):
        old = read_table('feature_inventory')
        records = [classify(r.feature, r.source_layer) for r in old.itertuples()]
        inventory = old.copy()
        corrections = []
        for i, record in enumerate(records):
            for field, value in record.items():
                previous = old.loc[i, field]
                if pd.isna(previous):
                    previous = ''
                if previous != value:
                    corrections.append(dict(feature=record['feature'], field=field,
                        original=previous, corrected=value, reason='Builder-verified descriptive metadata finalized after interruption'))
                inventory.loc[i, field] = value
        # These changes must not affect model eligibility or family membership.
        assert inventory.feature_group.equals(old.feature_group)
        assert inventory.eligible_predictor.equals(old.eligible_predictor)
        table(inventory, 'reconciled_feature_inventory')
        table(pd.DataFrame(corrections), 'feature_definition_updates')
        table(inventory.loc[inventory.classification_ambiguity.fillna('').ne('')], 'dictionary_and_timing_caveats')
        pairs = pd.read_parquet(CACHE / 'all_correlation_pairs.parquet')
        a, b = pairs.group_a.copy(), pairs.group_b.copy()
        pairs['group_a'] = [min(x, y) for x, y in zip(a, b)]
        pairs['group_b'] = [max(x, y) for x, y in zip(a, b)]
        pairs['absolute_spearman'] = pairs.spearman.abs()
        cross = pairs.groupby(['group_a', 'group_b']).agg(pairs=('high_correlation', 'size'),
            high_pairs=('high_correlation', 'sum'), high_pair_fraction=('high_correlation', 'mean'),
            max_abs_spearman=('absolute_spearman', 'max'), median_abs_spearman=('absolute_spearman', 'median')).reset_index()
        table(cross, 'cross_group_summary_canonical')
        f, events, _ = inputs()
        support = []
        for part, g in f.groupby('partition'):
            ids = set(g.loc[g.target.eq(1), 'next_event_id'])
            ev = events.loc[events.injury_event_id.isin(ids)]
            support.append(dict(partition=part, rows=len(g), positive_rows=int(g.target.sum()),
                represented_events=len(ids), initial_explicit_ucl_events=int(ev.initial_explicit_ucl.sum()),
                later_supplemental_ucl_confirmation=int(ev.supplemental_ucl_confirmed.sum()),
                first_appearance=g.game_date.min(), last_appearance=g.game_date.max()))
        table(pd.DataFrame(support), 'event_support')


def finalize_elastic():
    """Recover selection from all six valid grid checkpoints without refitting."""
    from .consolidation_models import source_coefficients, scores, feature_sets, run_elastic_tuning
    if (CACHE / 'elastic_selection.json').exists() and complete('elastic_net_selected_coefficients'):
        return
    if not complete('elastic_net_tuning'):
        run_elastic_tuning()
        return
    grid = read_table('elastic_net_tuning')
    expected = {(c, m) for c in CFG['elastic_net_C'] for m in CFG['elastic_net_l1_ratio']}
    if set(zip(grid.C, grid.l1_ratio)) != expected:
        run_elastic_tuning()  # fit_named reuses each verified existing model.
        return
    chosen = grid.sort_values(['average_precision', 'nonzero_features', 'C', 'l1_ratio'],
                             ascending=[False, True, True, False]).iloc[0].to_dict()
    obj = joblib.load(CACHE / 'models' / (chosen['model'] + '.joblib'))
    assert obj['encoder'].features == feature_sets()[0]
    f, _, _ = inputs()
    valid = f.loc[f.partition.eq('validation')]
    assert np.isclose(average_precision_score(valid.target, scores(obj, valid)), chosen['average_precision'], atol=1e-12)
    (CACHE / 'elastic_selection.json').write_text(json.dumps(chosen, indent=2))
    table(source_coefficients(obj), 'elastic_net_selected_coefficients')


def run(until='complete'):
    snapshot()
    from .consolidation_inventory import run_inventory
    from .consolidation_redundancy import run_redundancy
    from .consolidation_models import run_family_models, run_family_permutation
    from .consolidation_representations import run_representations, run_supported_cohorts
    from .consolidation_stability import run_stability
    from .consolidation_selection import (run_candidate_selection, run_conditional_checks,
        run_coverage_checks, freeze_candidate, run_frozen_transfer)
    if not complete('feature_inventory', 'deterministic_dependencies', 'feature_group_inventory'):
        run_inventory()
    if not complete('correlation_clusters', 'high_correlation_pairs', 'cluster_summary'):
        run_redundancy()
    reconcile_metadata()
    if not complete('family_performance', 'family_paired_differences'):
        run_family_models()
    if not complete('grouped_permutation_summary', 'grouped_permutation_repeats'):
        run_family_permutation()
    if not complete('representation_comparisons'):
        run_representations()
    if not complete('supported_cohort_sensitivity'):
        run_supported_cohorts()
    finalize_elastic()
    stable = complete('feature_stability', 'cluster_selection_frequency', 'training_refit_performance')
    if stable:
        stable = read_table('training_refit_performance').repeat.nunique() == CFG['training_refits']
    if not stable:
        run_stability()
    if until == 'stability':
        return
    if not complete('candidate_reduction_curve') or not (CACHE / 'candidate_selection.json').exists():
        run_candidate_selection()
    if not complete('conditional_feature_checks'):
        run_conditional_checks()
    if not complete('coverage_comparison', 'coverage_role_season_audit'):
        run_coverage_checks()
    if not (OUT / 'candidate_freeze.json').exists():
        freeze_candidate()
    if not complete('frozen_performance', 'frozen_alert_budgets', 'validation_fixed_thresholds', 'calibration_bins'):
        run_frozen_transfer()
    save_manifest()
    if until == 'models':
        return
    from .consolidation_plots import create_figures
    from .consolidation_report import write_report
    create_figures()
    write_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', choices=['stability', 'models', 'complete'], default='complete')
    run(parser.parse_args().until)
