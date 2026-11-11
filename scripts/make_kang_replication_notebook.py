"""Build the publication-focused synthesis; results are read, never refitted."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[
        nb.v4.new_markdown_cell('''# Kang parent-paper replication

**Question:** How closely can the published experimental setup and results be
reproduced using the current data?

[Kang et al.](https://doi.org/10.1177/23259671261477532) ·
[Protocol](../../docs/kang_replication_methods.md) ·
[Configuration](../../config/kang_replication.yaml)

This stage uses broad shoulder/elbow IL outcomes, 2017–2025 pitcher-games,
season-role strata, random-game train/test splitting, R XGBoost and ROC-AUC.
It includes the published GAM slope structure and prior-injury sensitivity.
It introduces no UCL adjudication, chronological validation, warning metrics,
new predictor concepts or alternative injury model. Unknown implementation
details are disclosed approximations, not claims of exact replication.'''),
        nb.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/kang_replication.yaml').exists())
R=ROOT/'reports/kang_replication'; T=R/'tables'
display(pd.Series(json.loads((R/'verification.json').read_text())))
print((R/'software.txt').read_text())'''),
        nb.v4.new_markdown_cell('''## Cohort and source differences

At least five outings per season; no more than 10% games missing required
Appendix A1 inputs. Season-majority role with innings tie-breaking. First
shoulder/elbow placement each season; only pre-placement case games.

The paper's Methods allow controls with other IL injuries; Results describe
controls without any IL. The Methods interpretation is primary here. The
alternative is counted, not chosen based on scores. Existing UCL episode-QA
filters and primary-position exclusions are not added. Unknown-diagnosis
control seasons remain source-uncertain, not certified healthy.'''),
        nb.v4.new_code_cell('''display(pd.read_csv(T/'cohort_flow.csv'))
display(pd.read_csv(T/'cohort_counts.csv'))
display(pd.read_csv(T/'event_qa_descriptive.csv'))
display(pd.read_csv(T/'exclusions_by_season.csv'))'''),
        nb.v4.new_markdown_cell('''## Published predictors

Use the 22 Appendix A1 inputs. Longitudinal release Y is tabulated elsewhere in
the article but absent from that list; extension is included and Y remains
crosswalk-only. The source has all required concepts. Match quality refers to
variable definition, not identical observations or preprocessing to the authors.'''),
        nb.v4.new_code_cell("display(pd.read_csv(T/'feature_crosswalk.csv')[['kang_concept','feature','match_quality','used_in_model','notes']])"),
        nb.v4.new_markdown_cell('''## Candidate windows and inferred slope bands

Training-only five-fold AUC over integer windows 7–49 days; the publication
does not specify the candidate increment. `mgcv` smooths AUC, and the declining
derivative is compared with 25–35% of its maximum. Green shading shows actual
qualifying segments; orange shows published intervals. The dotted line marks
the rounded envelope midpoint used for the final model, an unpublished-detail
approximation. Disconnected green regions do not identify a unique optimal
window. Bands are not confidence intervals; no maximum-AUC fallback is used.'''),
        nb.v4.new_code_cell('''display(pd.read_csv(T/'optimal_windows.csv'))
display(Image(filename=str(R/'figures/01_window_and_slope.png')))'''),
        nb.v4.new_markdown_cell('''## ROC-AUC and published comparison

50 repetitions of stratified five-fold fits on training subsets support gain/
consistency selection, followed by within-domain Pearson pruning at |r| > .7.
The final XGBoost specification is untuned because author tuning settings are
unavailable. ROC-AUC intervals resample held-out pitcher-seasons 500 times.
The same pitchers can appear in both train and test, as in the paper.'''),
        nb.v4.new_code_cell('''display(pd.read_csv(T/'roc_auc_results.csv').round(4))
display(Image(filename=str(R/'figures/03_roc_curves.png')))
display(pd.read_csv(T/'published_comparison.csv'))'''),
        nb.v4.new_markdown_cell('''## Feature importance and prior-injury sensitivity

Importance is R XGBoost's normalized total split Gain averaged over 500 training
pitcher-season bootstrap refits. Correlated features can exchange importance;
these rankings are not physiological explanations. The sensitivity adds only
the paper's corresponding-region IL history from previous study seasons.'''),
        nb.v4.new_code_cell('''display(Image(filename=str(R/'figures/02_feature_importance.png')))
display(pd.read_csv(T/'feature_importance.csv').groupby('task',sort=False).head(5))
display(pd.read_csv(T/'prior_injury_sensitivity.csv'))'''),
        nb.v4.new_markdown_cell('''## Fidelity and interpretation

The source differences and unknown author settings prevent exact numerical
replication. In particular, control wording, release-Y/extension naming, earlier
case-game handling, candidate increments, tuning/selection cutoffs and GAM
settings require qualification. The article's feature-specific rolling-risk
attribution is insufficiently defined for a numerical reproduction; no new
method was substituted. Read the computed report below before deciding whether
the structural benchmark is adequate for subsequent one-change comparisons.'''),
        nb.v4.new_code_cell("display(Markdown((R/'summary.md').read_text()))"),
    ]
    n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3 (project .venv)')))
    path=ROOT/'notebooks/kang_replication/01_parent_paper_replication.ipynb'
    nb.write(n,path); print(path)


if __name__=='__main__':main()
