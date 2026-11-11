"""Build the small, report-reading benchmark synthesis (no model execution)."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = [
        nb.v4.new_markdown_cell('''# Kang-style benchmark foundation

**Question:** Can the existing data architecture support a defensible reference
for Kang et al. before the UCL, chronological and temporal-family extensions?

[Parent study](https://doi.org/10.1177/23259671261477532) ·
[Method specification](../../docs/kang_benchmark_spec.md) ·
[Research direction](../../RESEARCH_DESIGN.md)

This is an **approximate retrospective reference**, not an exact reproduction.
It uses 2017–2025 only. The 2026 chronological test remains outside this stage.
All displayed values are read from the reproducible benchmark outputs.'''),
        nb.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'config/kang_benchmark.yaml').exists())
REPORT = ROOT / 'reports/kang_benchmark'
TABLES = REPORT / 'tables'
assert (TABLES / 'reference_metrics.csv').exists(), 'Run the benchmark --fit first; see the methods.'
verification = json.loads((REPORT / 'verification.json').read_text())
assert verification['no_2026_rows'] and verification['database_unchanged']
display(pd.Series(verification, name='Verification'))'''),
        nb.v4.new_markdown_cell('''## Cohort and comparison

One MLBAM pitcher × regular-season game; P/TWP identities; at least five season
appearances; at most 10% games with any missing required field. Season-majority
role separates starters and relievers. First shoulder/elbow IL placement per
season anchors labels. Only pre-placement case games are used; controls contribute
their season. This season-based design uses retrospective ascertainment.

The primary control definition excludes all IL seasons. A second table follows
the paper's broader Methods wording while excluding unknown-anatomy seasons.
The alternative is counted, not selected based on its predictive performance.'''),
        nb.v4.new_code_cell('''display(pd.read_csv(TABLES / 'cohort_flow.csv'))
support = pd.read_csv(TABLES / 'cohort_support.csv')
display(support[support.window_days.eq(28)][['control_rule','endpoint','role','rows',
    'pitcher_seasons','unique_pitchers','case_seasons','control_seasons',
    'represented_events','positive_rows','earlier_case_negative_rows']])
display(pd.read_csv(TABLES / 'event_eligibility.csv'))'''),
        nb.v4.new_markdown_cell('''## Predictor crosswalk and measurement support

Twenty-three candidate inputs use the dictionary and existing family inventory.
Five are minimal raw-pitch aggregates absent from the appearance table. All-pitch
SD must not be replaced by fastball SD; longitudinal release Y must not be
replaced by extension. Their separate inclusion is explicitly approximate.
Residual NaNs remain available to XGBoost; no full-dataset imputation is fitted.'''),
        nb.v4.new_code_cell('''crosswalk = pd.read_csv(TABLES / 'feature_crosswalk.csv')
display(crosswalk[['kang_concept','feature','project_feature_group','availability','timing_availability']])
missing = pd.read_csv(TABLES / 'feature_missingness.csv')
overview = missing.groupby('feature')[['rows','missing_rows']].sum()
overview['missing_percent'] = 100 * overview.missing_rows / overview.rows
display(overview.sort_values('missing_percent', ascending=False).head(10))'''),
        nb.v4.new_markdown_cell('''## Development window comparison

Fixed XGBoost settings; seven weekly cumulative windows; fold-local gain/correlation
selection inside five-fold development CV. One reserved random-game holdout is
shared across windows. Window and feature decisions are persisted before holdout
scoring. This bounded maximum-CV-AUC procedure **does not reconstruct the published
GAM plateau**. Bars show fold SD, not confidence intervals.'''),
        nb.v4.new_code_cell("display(Image(filename=str(REPORT / 'figures/development_window_comparison.png')))"),
        nb.v4.new_markdown_cell('''## Reserved random-game performance

ROC-AUC is the parent-study metric; AP and prevalence provide rare-event context.
Intervals resample held-out pitcher-seasons 500 times. Class-weighted scores
are not calibrated future injury probabilities. The same pitchers and injury
episodes can be represented in development and holdout; that dependence is
reported, not claimed to be removed by bootstrapping.'''),
        nb.v4.new_code_cell('''metrics = pd.read_csv(TABLES / 'reference_metrics.csv')
display(metrics[['endpoint','role','selected_window_days','selected_features',
    'development_cv_auc','roc_auc','roc_auc_ci_low','roc_auc_ci_high',
    'average_precision','prevalence','pr_lift','represented_events']].round(4))
display(metrics[['endpoint','role','shared_pitchers','holdout_unique_pitchers','shared_case_events',
    'brier','mean_risk','prevalence']].round(4))
display(pd.read_csv(TABLES / 'retained_reference_features.csv').query('retained == True')
    .groupby(['endpoint','role']).feature.agg(list).to_frame())'''),
        nb.v4.new_markdown_cell('''## Interpretation and next step

The benchmark is executable, with tracking missingness and feature support
quantified above. A best-scoring window at the grid boundary is **not** evidence
that Kang's plateau estimates were replicated. Endpoint,
control, earlier-case-game and feature-name ambiguities remain material.

Random holdout discrimination cannot distinguish learned persistent pitcher
profiles from imminent injury signals, and does not validate useful alert lead
time. No alert thresholds, new predictors, alternative learner search, PCA/PLS
or 2026 model-selection decisions were introduced.

Resolve and freeze the qualified reference. Then change only the broad elbow
endpoint to adjudicated UCL, followed by the chronological and limited temporal
family comparisons. Preserve the earlier analyses as hypothesis-generating
context, not independent confirmation.'''),
        nb.v4.new_code_cell("display(pd.read_csv(ROOT / 'docs/kang_benchmark_comparison.csv'))"),
    ]
    notebook = nb.v4.new_notebook(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3 (project .venv)', language='python', name='python3'),
        language_info=dict(name='python', version='3.9')))
    path = ROOT / 'notebooks/kang_benchmark/01_benchmark_foundation.ipynb'
    nb.write(notebook, path)
    print(path)


if __name__ == '__main__':
    main()
