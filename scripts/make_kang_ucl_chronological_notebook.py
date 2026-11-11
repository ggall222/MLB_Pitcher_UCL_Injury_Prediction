"""Create a saved-result synthesis of Extension 2; no hidden model fitting."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# Extension 2: Does UCL discrimination survive later-time evaluation?

Train **2017–2023**, evaluate **2024–2025**, then test **2026** using the same
frozen model. The endpoint, 22 candidate predictors, XGBoost settings, training
CV/selection algorithms, GAM window procedure and ROC-AUC remain those of
Extension 1. All fitted quantities are learned again from training only.

[Methods](../../docs/kang_ucl_chronological_methods.md) ·
[Configuration](../../config/kang_ucl_chronological.yaml)

This is retrospective. The 2026 test is locked within this experiment; the year
has appeared elsewhere in the project and is not prospective confirmation.
This notebook only reads results.'''),
        code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/kang_ucl_chronological.yaml').exists())
R=ROOT/'reports/kang_ucl_chronological';T=R/'tables'
display(pd.Series(json.loads((R/'verification.json').read_text())))
freeze=json.loads((R/'development_freeze.json').read_text())
display(pd.DataFrame(freeze['roles']).T)
print('Training freeze:',freeze['frozen_at_utc'])'''),
        md('''## Cohort support and follow-up

Games are split solely by dates. Pitchers can recur in later seasons; games,
pitcher-seasons and positive injury events cannot cross periods. Fixed season
eligibility and retrospective roles are inherited from the parent.

The verified injury observation cutoff is 2026-10-20. Negative labels require
the selected calendar horizon to finish by that cutoff. Observed positives are
retained; unresolved late negatives remain censored. Source uncertainties are
reported without altering adjudication or pooling roles.'''),
        code('''display(pd.read_csv(T/'cohort_support.csv'))
uncertain=pd.read_csv(T/'test_endpoint_uncertainty.csv')
display(uncertain.groupby(['evidence_category','endpoint_primary','in_model_cohort'],dropna=False).size().rename('events').reset_index())
display(pd.read_csv(T/'validation_checks.csv'))'''),
        md('''## Training-only window development

Every 7–49-day candidate uses five-fold training CV. The exact parent functions
produce the smoothed AUC curve, slope band and rounded midpoint. Disconnected
bands do not identify a unique optimum. A missing band receives no fallback.
These CV curves use training games, not validation or test scores.'''),
        code('''display(pd.read_csv(T/'window_selection.csv'))
display(pd.read_csv(T/'training_window_performance.csv'))
display(Image(filename=str(R/'figures/01_training_windows.png')))'''),
        md('''## Random-game versus temporal discrimination

Extension 1 is read only as the comparator. Both later periods use the model
fitted on 2017–2023, with no updating after validation. Intervals use the same
500 pitcher-season bootstrap attempts; interpret them alongside event counts.
Conditional intervals with few events do not establish population-level stability.'''),
        code('''display(pd.read_csv(T/'primary_comparison.csv').round(4))
display(Image(filename=str(R/'figures/03_auc_comparison.png')))
display(Image(filename=str(R/'figures/02_temporal_roc.png')))
display(pd.read_csv(T/'seen_unseen_pitchers.csv').round(5))'''),
        md('''## Model reliance and descriptive drift

Gain is calculated from training-only bootstrap refits. The comparison concerns
model reliance, not injury mechanisms. Score distributions include all scored
cohort games; training scores are in-sample. Predictor shifts are summarized
without adapting, recalibrating or retraining the model.

Reliability bins are learned from training scores. Class weighting means the
scores should not be assumed to estimate absolute UCL risk.'''),
        code('''display(Image(filename=str(R/'figures/04_importance_comparison.png')))
display(pd.read_csv(T/'feature_importance_comparison.csv'))
display(Image(filename=str(R/'figures/05_score_drift.png')))
display(pd.read_csv(T/'score_distribution.csv'))
display(pd.read_csv(T/'predictor_drift.csv'))
display(Image(filename=str(R/'figures/06_reliability.png')))
display(pd.read_csv(T/'reliability.csv'))'''),
        md('''## Synthesis and limits

The conclusion follows the interpretation policy frozen before later-period
access. A weak chronological result is informative. No alternative windows,
seeds, predictors, cases or models are substituted to improve it. The report
distinguishes chance-compatible estimates, small event support, window ambiguity,
retrospective endpoint limitations and incomplete follow-up.'''),
        code("display(Markdown((R/'summary.md').read_text()))")]
    p=ROOT/'notebooks/kang_ucl_chronological/01_temporal_evaluation.ipynb';p.parent.mkdir(parents=True,exist_ok=True)
    n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3 (project .venv)')))
    nb.write(n,p);print(p)


if __name__=='__main__':main()
