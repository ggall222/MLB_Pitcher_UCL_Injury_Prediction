"""Generate three focused supervised-compression notebooks."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/injury_supervised_reduction';DEST.mkdir(parents=True,exist_ok=True)
SETUP="""from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/injury_supervised_reduction.yaml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.injury_analysis.supervised_reduction import inputs,read_table,OUT
from src.injury_analysis.supervised_workflow import ensure_stage
from src.injury_analysis import supervised_plots as plots
FORCE_RECOMPUTE=False
a,events,spec,matches=inputs()
pd.set_option('display.max_columns',18)
"""
def write(name,title,question,cells):
    n=nbf.v4.new_notebook()
    n.metadata.update(kernelspec={'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},
                      language_info={'name':'python','version':'3.9'})
    n.cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n**Question.** {question}'),nbf.v4.new_code_cell(SETUP)]
    n.cells.extend(nbf.v4.new_code_cell(x) if k=='code' else nbf.v4.new_markdown_cell(x) for k,x in cells)
    nbf.write(n,DEST/name)

write('01_supervised_compression_and_prediction.ipynb','01 — Does supervision make predictive information more compact?',
      'Compare training-only PLS, PCA and original features with validation-selected complexity.',[
('markdown',"""## Existing cohorts and comparison design
Retain the seven category-specific 30-day targets, starters and relievers, observed-follow-up negatives, exclusions and 60-day temporal embargo. Train through 2021; validation 2022–2023; previously examined test 2024–2025. Every method within a representation uses identical rows. No outcome or feature screening outside training contributes to the learned axes.

Four source representations reuse existing definitions: original 62 features; 13 profile fields (eight prior means + five slow context variables); eight current pitcher-relative departures; and the combined 21 fields. Workload remains in the original-feature comparator rather than being forced into the profile/departure split. Current measurements describe an already completed outing.

**PLS** fits continuous predictors to the binary training outcome. We orthonormalize each nested PLS span with QR and use the same L2 logistic C=1 as PCA and unreduced models. QR preserves each supervised subspace while avoiding an arbitrary axis-scale change to the regularization penalty. It does not whiten scores. Raw PLS regression values are not used as probabilities. All imputation/scaling fits on training only.

Binary context and training-observed missingness flags pass through for both projection methods. Thus *k components is not k total model columns*. The context/availability-only diagnostic tests what survives without any continuous projection. A full-span completion control should approximately reproduce original predictions."""),
('code',"""ensure_stage('prediction',a,events,spec,matches,force=FORCE_RECOMPUTE)
display(read_table('validation_selection'))
display(read_table('selected_performance').query("block=='Original 62' and variant=='Unreduced'")[[
 'category','partition','rows','positive_rows','represented_events','pitchers','prevalence']])
display(read_table('full_span_control').groupby('block')[['max_score_difference','ap_difference']].max())"""),
('markdown',"""## Component selection and empirical near-peak behavior
PLS and PCA use the same tested dimensions within each block, augmented with the PCA-90% count learned from training. Pick maximum **validation AP**, breaking ties toward fewer components. Neither test performance nor matched trajectories selects k.

The near-peak summary is the smallest tested k achieving 95% of the best validation AP *gain over prevalence*. A second column asks whether all later tested counts also remain above that threshold. These are descriptive summaries, not confidence-based plateaus. A small gain above chance can be unstable even if its representation is compact. The first axis of a binary supervised learner is not evidence for one underlying biological mechanism."""),
('code',"""from src.injury_analysis.supervised_report import compression_summary
display(compression_summary())
plots.curve_figure('Original 62');plt.show()
plots.curve_figure('Combined');plt.show()
plots.compression_figure();plt.show()"""),
('markdown',"""## Predictive and practical comparisons
Paired intervals resample evaluation pitchers using the established 300 shared draws and condition on trained models and selected k. They do not include tuning uncertainty or correct every multiple comparison. Validation selection makes validation estimates optimistic.

Keep 7–30-day event warnings, all-event denominators and the existing 1%, 2.5%, 5% and 10% alert budgets. A model can improve AP while reducing useful event warnings. Uniform-random expected recall and conditional warning leads are retained. Retrospective top-k alerts do not guarantee a deployable fixed threshold. Brier is total probability error, not calibration alone."""),
('code',"""plots.comparison_figure();plt.show()
display(read_table('selected_differences').query("block=='Original 62' and variant=='PLS selected'"))
display(read_table('alert_budgets').query("block=='Original 62' and alert_rate==0.05")[[
 'category','partition','variant','qualifying_events','observable_events','alerted_events','injury_event_recall',
 'random_expected_event_recall','median_warning_lead_days','false_alerts_per_pitcher_season']])
display(read_table('selected_performance').query("category=='Shoulder'")[[
 'block','partition','variant','average_precision','pr_lift','roc_auc','brier_score']])"""),
('markdown',"""## Scope of the inference
Compare compression, prediction and event coverage separately. Any apparent compression is conditional on the unprojected context/missingness columns. Preserve the original model as the benchmark; one favorable category or held-out estimate cannot establish a generally better method.""")])

write('02_components_and_injury_patterns.ipynb','02 — What do supervised components emphasize?',
      'Interpret feature families, profile/departure variation and the prior shoulder PCA pattern.',[
('markdown',"""## Leading directions and interpretation
Fit axes on category-specific training risk sets only. The weights describe standardized source coordinates of the actual orthonormalized projection. Signs for plots are oriented toward higher training injury outcome; native weights remain available. Family sums of squared axis weights are geometric summaries, not explained injury risk or causal importance.

For profile/departure comparison, distinguish prior means from current relative deviations. Profile means use up to 20 prior outings, not a fixed lifetime trait. PLS is linear: mixing the two blocks does not test nonlinear interactions."""),
('code',"""ensure_stage('prediction',a,events,spec,matches,force=False)
ensure_stage('interpretation',a,events,spec,matches,force=FORCE_RECOMPUTE)
plots.loadings_figure();plt.show()
plots.family_figure();plt.show()
display(read_table('component_family_weights').query("method=='PLS' and component==1"))
display(read_table('component_persistence').query("partition=='test' and method=='PLS' and component==1"))"""),
('markdown',"""## Relationship to the earlier PCA
Compare frozen supervised departure scores against the prior unsupervised departure PCs on the *same training rows*. Both transforms were fit in training, though the original descriptive PCA used a common rather than category-specific sample. Sign is arbitrary, so assess magnitudes and contributing measurements. These correlations establish overlap between representations, not an independent replication of predictive association."""),
('code',"""display(read_table('prior_pca_score_correlations').query("category=='Shoulder'"))
display(read_table('component_loadings').query("category=='Shoulder' and block=='Departure' and method=='PLS' and component==1").sort_values('squared_axis_weight',ascending=False))"""),
('markdown',"""## Original matched trajectories outside training
Reuse the exact prior matched case/control identities and appearance keys. Both windows lie in 2022–2025. Require common complete P+D source measurements for each plotted score, and the same matched events at every point. No imputed value creates a physiological trajectory. Predictive evaluation itself retains incomplete rows.

Display the first supervised direction for profile, departure and combined inputs, in training-score SD units. These are not full-model risk scores when selected k exceeds one. Calendar bins and outing positions remain distinct. Bands are simultaneous within each trajectory; comparisons across many injury groups remain exploratory. Every panel reports its balanced event support. Recorded injury/IL dates are not verified biological onset."""),
('code',"""plots.trajectory_figure('appearance');plt.show()
plots.trajectory_figure('calendar');plt.show()
display(read_table('component_shape_contrasts').query("category in ['Shoulder','Elbow / UCL','Lower body']"))"""),
('markdown',"""## Interpretation
Ask whether apparent case/control separation is a persistent offset or a changing departure, and whether the uncertainty supports its timing. A direction optimized for injury labels can be informative without representing a distinct physiological mechanism. Continue to the resampling and near-event checks before drawing conclusions.""")])

write('03_stability_sensitivity_and_summary.ipynb','03 — Are supervised patterns stable and useful?',
      'Check training-sample dependence, shoulder timing sensitivity and the final research interpretation.',[
('markdown',"""## Component and model stability
Use 12 paired whole-pitcher training bootstraps for shoulder and lower body, in the original and combined representations. Refit imputation, scale, PLS/PCA and classifier each time. The original validation-selected count remains fixed; these diagnostics do not include dimension-reselection uncertainty.

Compare directions in the same original training units after adjusting for refitted scales. Absolute first-axis cosine removes arbitrary sign flips. Subspace overlap is sign/rotation invariant, but larger or full-rank subspaces are easier to reproduce and should not be directly compared as equivalent tasks. Report score ranks and top-5% alert-set overlap as separate predictive stability outcomes. Inspect the number of distinct positive training events retained in each draw."""),
('code',"""ensure_stage('prediction',a,events,spec,matches,force=False)
ensure_stage('stability',a,events,spec,matches,force=FORCE_RECOMPUTE)
plots.stability_figure();plt.show()
display(read_table('stability_summary'))
display(read_table('refit_family_weights').groupby(['category','block','method','family']).first_axis_squared_weight.agg(['mean','min','max']))
display(read_table('training_refits').groupby(['category','block']).distinct_train_positive_events.agg(['min','max']))"""),
('markdown',"""## Focused shoulder sensitivity
The previous profile/departure analysis raised a specific concern: apparent gains may be concentrated just before the recorded event. Refit original, departure and combined representations after excluding final-six-day positive rows; never label those rows healthy. Separately require complete P+D measurement support, targeting the baseline/departure missingness concern without demanding that every original structural feature be observed.

For each sensitivity, both methods use the same rows, training-only transforms and validation-selected component counts. Changes across conditions therefore include changed samples and prevalence; interpret within-condition contrasts rather than raw AP differences across conditions."""),
('code',"""ensure_stage('sensitivity',a,events,spec,matches,force=FORCE_RECOMPUTE)
display(read_table('sensitivity_selection'))
display(read_table('sensitivity_differences'))
display(read_table('sensitivity_performance')[[
 'analysis','block','partition','variant','rows','positive_rows','prevalence','average_precision','brier_score']])"""),
('code',"""from src.injury_analysis.supervised_report import write_synthesis
display(Markdown(write_synthesis()))""")])
print(f'Wrote three notebooks to {DEST}')

