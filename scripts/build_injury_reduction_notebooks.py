"""Generate the three dimensionality-reduction notebooks; execute after regenerating."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/injury_dimensionality_reduction'
DEST.mkdir(parents=True,exist_ok=True)
SETUP='''from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/injury_reduction.yaml').exists())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.injury_analysis.reduction import CFG,OUT,CACHE,inputs,read_table
from src.injury_analysis import reduction_plots as plots
from src.injury_analysis.common import conclusion
pd.set_option('display.max_columns',16)
'''


def write(name,title,question,cells):
    n=nbf.v4.new_notebook()
    n.metadata.update(kernelspec={'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},
                      language_info={'name':'python','version':'3.9'})
    n.cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n**Question.** {question}'),nbf.v4.new_code_cell(SETUP)]
    for kind,text in cells:
        n.cells.append(nbf.v4.new_code_cell(text) if kind=='code' else nbf.v4.new_markdown_cell(text))
    nbf.write(n,DEST/name)


write('01_feature_space_and_components.ipynb','01 — Redundancy, profiles and departures',
      'How much lower-dimensional structure exists, and what do the principal dimensions represent?',[
('markdown','''## Cohort and train-only design
Reuse the existing regular-season appearance cohort. The common descriptive fitting sample consists of analysis-eligible **training** appearances through 2021, including starters and relievers. No injury labels, validation outcomes or test observations select axes. This sample differs from category-specific supervised risk sets; predictive PCA will be refitted within each such training set.

Three complementary views:

1. The 60 continuous original predictors (two binary role/hand fields are excluded from PCA).
2. Eight **strictly prior pitcher baseline means**: velocity, release X/Z and their variability, extension, spin and zone percentage.
3. The corresponding eight **current departures from prior pitcher baselines** (existing z-scores).

Historical NULLs remain intact. A training-fitted median and scale are used only for the analytical transform. Missingness flags stay outside predictive PCA; they are not interpreted as pitching components. Arm angle is excluded from this long-history comparison. PCA is unwhitened and uses deterministic full SVD. Component signs are arbitrary.'''),
('code','''from src.injury_analysis.reduction import run_structure
a,events,spec,matches=inputs()
summary=run_structure(a,spec)
display(summary)
display(read_table('feature_blocks'))
print(f'{len(a):,} cached pitcher appearances; {events.target_eligible.sum():,} eligible injury targets; {len(matches):,} original matched pairs.')'''),
('code','''plots.spectrum_figure();plt.show()
display(read_table('high_correlations').sort_values('correlation',key=lambda s:s.abs(),ascending=False).head(20))
display(read_table('training_missingness').sort_values('train_missing_fraction',ascending=False).head(12))'''),
('markdown','''## Interpret the axes from their contributing measurements
The heatmap displays training correlations with PC scores, not causal importance or injury discrimination. All projection weights, including omitted lower-ranked axes, are saved. A two-dimensional picture is useful only in proportion to the variance it retains. Current-feature space mixes role/workload, persistent levels and within-pitcher changes, so the baseline/departure views disentangle the input meanings without claiming independent biological mechanisms.'''),
('code','''plots.loading_figure();plt.show()
plots.persistence_figure();plt.show()
display(read_table('component_persistence'))'''),
('markdown','''## Sensitivity to repeated observations, missingness and training era
The leading bases are checked under (a) at most 20 deterministic sampled outings per pitcher, (b) complete measurement rows, and (c) early/recent training years. These use the same training-fitted units, so subspace overlap measures basis changes rather than differences in units. Mean squared principal-angle cosine is 1 for identical subspaces and 0 for orthogonal ones. It ignores arbitrary signs and rotations within a subspace. This is a descriptive basis sensitivity, not a new validation split.'''),
('code','''display(read_table('subspace_sensitivity'))
r=summary.set_index('block')
p=read_table('component_persistence')
conclusion(f"The original continuous space needs {int(r.loc['Current feature space','components_90'])}/60 components for 90% of training variance; its first two capture {r.loc['Current feature space','first_two_fraction']:.1%}. Baseline and departure blocks each need {int(r.loc['Prior pitcher profile','components_90'])} and {int(r.loc['Within-pitcher departure','components_90'])} of 8 components, respectively. This quantifies compression without presuming a two-dimensional injury manifold. Between-pitcher variance and role/hand correlations identify what the displayed axes actually encode.")''')])

write('02_latent_profiles_and_preinjury_change.ipynb','02 — Latent injury profiles and approaching-event changes',
      'Do injury groups occupy different regions of the frozen latent maps, and is separation persistent or changing?',[
('markdown','''## Matched, out-of-training comparisons
Reuse the **exact original matched case/control identities and outing keys**. The primary curves use pairs whose entire observed windows and injury anchors fall in 2022–2025, after PCA fitting. All-year curves, including partial 2026, are saved separately as description. These are already examined periods, not fresh confirmation.

A plotted score requires all eight source measurements for its block. For each component/category, balanced curves keep both members of the **same pairs at all points**. Calendar time uses 31–60, 15–30, 8–14 and 1–7 days; appearance time uses the final five outings in the existing 60-day alignment. Scores are expressed in training-PC standard deviations. No smoothing, future baselines or reindexing invent observations. Source dates remain recorded injury/IL proxies, not biological onset.'''),
('code','''from src.injury_analysis.reduction_trajectories import run_latent_trajectories
a,events,spec,matches=inputs()
trajectories=run_latent_trajectories(events,matches)
display(read_table('latent_alignment_coverage'))
display(trajectories.query("cohort=='Heldout 2022–2025' and kind=='difference' and point==0")[[
 'feature','mode','category','events','pitchers']])'''),
('code','''plots.latent_map_figure();plt.show()
plots.latent_trajectory_figure('appearance');plt.show()
plots.latent_trajectory_figure('calendar');plt.show()'''),
('markdown','''## Persistent offset versus evolving separation
Each event contributes one paired summary. Average separation describes the offset across the window. Final-step change is (case last minus case previous two) minus (control last minus control previous two). These answer different questions. Profile baselines are smooth by construction; a stable profile offset is not proof of susceptibility. Departure PCs may mix changing velocity with release, spin and extension, so interpret them with notebook 01's loadings.

Uncertainty resamples whole case pitchers, retaining their episodes. Time bands are simultaneous **within a component/category**; they do not adjust across every component or injury comparison. The scatter shows event overlap, not clustering accuracy. A person's participation as a control elsewhere can retain some cross-pair dependence.'''),
('code','''shapes=read_table('latent_shape_contrasts')
display(shapes.query("cohort=='Heldout 2022–2025' and mode=='appearance' and metric in ['average_separation','final_step_vs_previous_two']")[[
 'feature','category','metric','mean','low','high','events']])
display(read_table('latent_sensitivity').query("feature in ['profile_PC1','departure_PC2']"))
s=shapes.query("cohort=='Heldout 2022–2025' and mode=='appearance' and feature=='departure_PC2' and metric=='final_step_vs_previous_two'").set_index('category')
conclusion(f"The velocity/release/spin/extension departure PC2 changes by {s.loc['Shoulder','mean']:.3f} training-PC SD in the shoulder final-step contrast (interval {s.loc['Shoulder','low']:.3f} to {s.loc['Shoulder','high']:.3f}; {int(s.loc['Shoulder','events'])} pairs), versus {s.loc['Elbow / UCL','mean']:.3f} for elbow/UCL ({s.loc['Elbow / UCL','low']:.3f} to {s.loc['Elbow / UCL','high']:.3f}). All axes, uncertainty and support are retained; these group trajectories do not establish individual early-warning thresholds.")''')])

write('03_reduced_models_and_stability.ipynb','03 — Prediction, compression and stability',
      'Does dimensionality reduction help prediction or model stability, and what does it reveal about the original model?',[
('markdown','''## Identical temporal evaluation and common rows
Use the existing seven category-versus-IL-free risk sets at the 30-day horizon. Train through 2021, validate in 2022–2023 and evaluate in the previously examined 2024–2025 period, preserving the 60-day embargo and event-overlap safeguards. All variants use identical rows within category/partition; train-only imputation, scaling and PCA are refitted per category. Logistic C=1 is unchanged; numerical convergence is tightened for **all** new fits.

Compare original 62 features, workload/history, global PCA with 5/10/20 components or a training-learned 90%-variance cutoff, workload/history plus 10-PC or 90%-variance Statcast, and a role/hand-plus-missingness diagnostic. All PCA designs pass binary context and training-observed missingness flags through. Report total columns, not just PC count. A full-rank, unwhitened rotation is the numerical equivalence control; no component rescaling changes the regularization geometry.

Choose a global-PCA dimension by **validation AP only**, with smaller design as tie-break. Test comparisons remain exploratory; neither test AP nor trajectory appearance selects dimensions. Conditional pitcher-bootstrap intervals exclude retraining and dimension-selection uncertainty.'''),
('code','''from src.injury_analysis.reduction_prediction import run_prediction,run_stability,event_concentration_audit
a,events,spec,matches=inputs()
performance=run_prediction(a,events,spec)
display(read_table('predictive_cohort'))
display(read_table('predictive_dimensions'))
display(read_table('full_rotation_sanity'))
display(read_table('validation_dimension_selection'))'''),
('code','''plots.prediction_figure();plt.show()
plots.alert_figure();plt.show()
display(performance.query("partition=='test'")[[
 'category','variant','design_dimensions','average_precision','pr_lift','ap_delta_original','delta_low_original','delta_high_original']])
display(read_table('event_recall_differences').query("partition=='test'"))
display(read_table('validation_fixed_thresholds'))'''),
('markdown','''## Stability is an empirical outcome, not a promised PCA benefit
Twelve **paired whole-pitcher training bootstrap refits** relearn all preprocessing, PCA and logistic coefficients. The original validation-selected specification is fixed; a 90% rule recomputes retained dimension inside the resampled training set. Compare score-rank stability and overlap of the top-5% alert sets against each model's unperturbed fit. These diagnostics are not 12 independent tests or a sampling confidence interval for generalization.

Frozen-score checks report starter/reliever and seen/unseen-pitcher performance, and omit final-six-day positive rows without calling them healthy. A leave-one-positive-pitcher-out audit checks whether an apparent gain is concentrated in a few injured pitchers. Neither diagnostic reselects the model. Event warning requires 7–30 days of lead; all qualifying events remain in the denominator. Random alert expectations account for repeated opportunities per event.'''),
('code','''stability=run_stability(a,spec)
influence=event_concentration_audit()
plots.stability_figure();plt.show()
display(stability)
display(read_table('predictive_subgroups'))
display(influence[influence.removed_positive_pitcher.notna()].groupby('category').ap_delta.agg(['min','max']))'''),
('code','''from src.injury_analysis.reduction_report import write_synthesis
report=write_synthesis()
display(Markdown(report))
s=read_table('selected_pca_comparison').query("partition=='test'")
conclusion(f"Validation-selected global PCA improves test AP by point estimate in {s.ap_delta_original.gt(0).sum()}/7 injury categories; paired intervals exclude zero on the positive side in {s.delta_low_original.gt(0).sum()}/7. Interpret this alongside event warning, the small-group concentration audit, training-refit stability and the profile/departure trajectories. PCA is an interpretable compression experiment, not proof of latent physiological mechanisms or a newly validated injury-warning system.")''')])

print(f'Wrote three notebooks to {DEST}')

