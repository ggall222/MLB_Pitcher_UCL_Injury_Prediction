"""Generate an Extension 1 synthesis notebook from completed results only."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    markdown=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[markdown('''# Extension 1: Does Kang-style signal persist for documented UCL injury?

**One experimental change:** the broad elbow endpoint becomes the established
documented-UCL endpoint. We preserve the completed Kang replication's 2017–2025
pitcher-game design, season-role strata, predictors, random-game split, R
XGBoost, feature selection, window selection and ROC-AUC procedure.

[Methods and scope](../../docs/kang_ucl_endpoint_methods.md) ·
[Configuration](../../config/kang_ucl_endpoint.yaml)

This notebook reads saved results. It does not fit models or change settings.
The published-method approximations documented in the parent remain in force.'''),
        code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/kang_ucl_endpoint.yaml').exists())
R=ROOT/'reports/kang_ucl_endpoint'; T=R/'tables'
display(pd.Series(json.loads((R/'verification.json').read_text())))
display(pd.read_csv(T/'benchmark_equivalence.csv'))'''),
        markdown('''## Endpoint specificity and cohort composition

Use the saved primary adjudication from `ucl_dynamic_endpoint.run_endpoint`,
including official diagnosis or bounded procedure evidence and established QA
exclusions. It is operational documentation, not independent clinical chart review.

Preserve the first combined shoulder/elbow-event rule; add compatible confirmed
UCL forearm presentations to the same event universe. Unsupported elbow cases
are excluded, not moved to healthy controls. Other confirmed UCL episodes outside
the original eligibility/first-event framing remain in the audit, not the model.
The established UCL reference date can precede IL placement; surgery is not onset.'''),
        code('''display(pd.read_csv(T/'cohort_comparison.csv'))
display(pd.read_csv(T/'cohort_differences.csv'))
display(pd.read_csv(T/'endpoint_overlap_summary.csv'))
display(pd.read_csv(T/'ucl_source_coverage.csv'))
display(pd.read_csv(T/'cohort_membership_changes.csv'))'''),
        markdown('''## Class composition and available event support

The 30-day table compares a shared, already-existing candidate window. It is
descriptive, not a new selection rule. The complete table retains every window.
Technical five-fold support does not mean sufficient independent events for
stable ranks or precisely located optimal windows.'''),
        code('''display(pd.read_csv(T/'common_30_day_prevalence.csv'))
display(pd.read_csv(T/'window_support.csv'))
display(pd.read_csv(T/'preflight_support.csv').query('window in [7, 30, 49]'))'''),
        markdown('''## Window-selection behavior

Points show mean training CV AUC; lines are the unchanged `mgcv` smoother.
Shading marks actual qualifying derivative-band segments; dotted lines mark
rounded envelope midpoints. Bands are not confidence intervals or biological
progression timelines. A disconnected band is not a unique optimum.
No maximum-AUC fallback or endpoint-specific tuning is allowed.'''),
        code('''display(Image(filename=str(R/'figures/01_endpoint_window_comparison.png')))
display(pd.read_csv(T/'window_performance.csv'))
display(pd.read_csv(T/'primary_comparison.csv').round(4))'''),
        markdown('''## Feature reliance within the same model family

The original 500 training pitcher-season refits estimate mean normalized total
Gain. Compare selected variables, common leading predictors and rank movement.
Unselected features have missing Gain, not evidence of a physiological null.
Correlated variables can trade importance. No causal interpretation is implied.'''),
        code('''display(Image(filename=str(R/'figures/02_endpoint_feature_importance.png')))
display(pd.read_csv(T/'feature_importance_comparison.csv'))
display(pd.read_csv(T/'prior_broad_elbow_history_supplement.csv'))'''),
        markdown('''## Interpretation and limits

The random partition algorithm and seed are fixed; actual assignments can change
with endpoint labels and membership. Shared pitchers remain in train and test.
The contrast therefore includes changes in case composition, reference timing
and class weights induced by prevalence. It is neither a paired causal estimate
nor chronological or operational-warning validation. The source's retrospective
2026 retrieval/confirmation metadata do not enter predictors or select models.

Read the computed findings and preservation checks below. Extension 2 has not
been run.'''),
        code('''display(pd.read_csv(T/'validation_checks.csv'))
display(Markdown((R/'summary.md').read_text()))''')]
    notebook=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(
        name='python3',language='python',display_name='Python 3 (project .venv)')))
    path=ROOT/'notebooks/kang_ucl_endpoint/01_ucl_endpoint_specificity.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True);nb.write(notebook,path);print(path)


if __name__=='__main__':main()
