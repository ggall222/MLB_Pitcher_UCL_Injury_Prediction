"""Generate the focused profile/departure notebooks; execute after generation."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/injury_profile_departure';DEST.mkdir(parents=True,exist_ok=True)
SETUP="""from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/injury_profile_departure.yaml').exists())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.injury_analysis.profile_departure import inputs,read_table,OUT
from src.injury_analysis.profile_workflow import ensure_stage
from src.injury_analysis import profile_plots as plots
FORCE_RECOMPUTE=False
a,events,spec,matches=inputs()
pd.set_option('display.max_columns',16)
"""
def write(name,title,question,cells):
    n=nbf.v4.new_notebook()
    n.metadata.update(kernelspec={'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},
                      language_info={'name':'python','version':'3.9'})
    n.cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n**Question.** {question}'),nbf.v4.new_code_cell(SETUP)]
    n.cells.extend(nbf.v4.new_code_cell(x) if k=='code' else nbf.v4.new_markdown_cell(x) for k,x in cells)
    nbf.write(n,DEST/name)

write('01_predictive_increment.ipynb','01 — Does departure add predictive information?',
      'Compare profile alone, departure alone and their combination on identical risk sets.',[
('markdown',"""## Design and meanings
Reuse the existing 30-day injury targets, starters and relievers, observed-follow-up negatives, competing-injury exclusions, temporal partitions and 60-day embargo. Train through 2021; validate 2022–2023; evaluate the already examined 2024–2025 period. All preprocessing is learned on training rows.

**P:** eight strictly prior baseline means plus five slowly changing context fields. Baselines use at most 20 prior outings, minimum five valid measurements, and exclude the current date. They are evolving profiles, not lifetime traits.
**D:** the corresponding eight current pitcher z-scores. These become observable after the current outing; each includes the prior SD denominator.
**C:** 15 workload fields, current starter role and month, held common in adjusted comparisons because they mix persistent and situational information.

Use the existing median + missing-indicator + scaling + L2 logistic framework (C=1), with numerical tolerance 1e-9. Compare P, D, P+D; C+P, C+D, C+P+D; P+D plus eight prior slopes; workload/history; and the original 62 features. No field from future injury/activation outcomes enters a predictor. NULLs remain unchanged in source data. Arm angle is omitted for long-history availability."""),
('code',"""ensure_stage('models',a,events,spec,matches,force=FORCE_RECOMPUTE)
display(read_table('model_specifications'))
display(read_table('cohort_counts'))
display(read_table('feature_definitions'))"""),
('markdown',"""## Average Precision and incremental information
Intervals resample whole evaluation pitchers with shared draws across models (300 resamples). They condition on the fitted models and do not adjust across all categories or metrics. The black marks show prevalence; raw accuracy is not an outcome of interest. Positive rows and distinct represented events are reported separately."""),
('code',"""plots.performance_figure();plt.show()
plots.incremental_figure();plt.show()
display(read_table('incremental_comparisons').query("reference in ['Profile only','Context + Profile']"))"""),
('markdown',"""## Practical warnings and repeated alerts
The established warning window is 7–30 days, evaluated at 1%, 2.5%, 5% and 10% of eligible outings. All qualifying events remain in the denominator, including those without warning opportunities. Uniform-random expected recall accounts for each event's number of opportunities. Improving over a profile-only alert policy does not establish performance beyond random alerts.

Top-k budgets are retrospective. The separate validation-95th-percentile threshold table reports actual test alert rates. Warning lead comparisons among jointly detected events avoid confusing an altered detected-event mix with genuinely earlier warning."""),
('code',"""ensure_stage('scores',a,events,spec,matches,force=FORCE_RECOMPUTE)
plots.alert_figure();plt.show()
display(read_table('alert_budget_performance').query("alert_rate==0.05 and variant in ['Profile only','Departure only','Profile + Departure']"))
display(read_table('alert_concentration').query("partition=='test' and variant in ['Profile only','Profile + Departure']"))
display(read_table('validation_fixed_thresholds'))"""),
('markdown',"""## Probability quality
Brier loss measures total probability error, not calibration alone. Reliability bins use validation-score quantiles and the same fixed edges on test. These curves are descriptive; bin counts and positive support are saved. Mean bias and Brier differences complement them. No calibration map is fit here. Because follow-up requirements and competing events select the risk sets, estimated probabilities refer to this analysis cohort, not all MLB pitcher outings."""),
('code',"""plots.calibration_figure();plt.show()
display(read_table('model_performance').query("variant in ['Profile only','Departure only','Profile + Departure']")[[
 'category','partition','variant','rows','positive_rows','prevalence','average_precision','pr_lift','brier_score','calibration_bias']])
display(read_table('reliability_bins').query("partition=='test' and variant=='Profile + Departure'"))"""),
('markdown',"""## Interpretation
Read the uncertainty on paired differences rather than treating every numerical increase as evidence. The workload-adjusted comparison answers a stricter question than P versus P+D. Continue to notebook 02 for within-pitcher score behavior, missingness and timing checks, and a computed synthesis.""")])

write('02_score_timing_and_robustness.ipynb','02 — What are the models learning?',
      'Separate persistent model scores from changing scores and test whether the increment is robust.',[
('markdown',"""## Exact score decomposition and original matched controls
The combined logistic predictor equals intercept + profile contribution + departure contribution. Missing flags belong to their source block. This identity is verified numerically; contributions are log-odds terms, not causal risk shares.

Use the exact original matched case/control identities and outing keys. Both observed windows must fall in 2022–2025, outside training. Main curves require complete P+D measurements for both members and the same pairs at every plotted point. This avoids imputed measurements or changing pair composition creating a trajectory. Prediction evaluation itself retains incomplete rows.

Calendar bins are 31–60, 15–30, 8–14 and 1–7 days before the recorded event. Appearance positions are the final five outings within 60 days; they are not equally spaced in calendar time. Bands are pitcher-bootstrap simultaneous bands within each curve, without smoothing. Event counts are printed in each panel."""),
('code',"""ensure_stage('models',a,events,spec,matches,force=False)
ensure_stage('scores',a,events,spec,matches,force=FORCE_RECOMPUTE)
display(read_table('score_alignment_coverage'))
display(read_table('contribution_identity'))
display(read_table('score_variation').query("partition=='test'"))
plots.score_trajectory_figure('appearance');plt.show()
plots.score_trajectory_figure('calendar');plt.show()"""),
('code',"""display(read_table('score_shape_contrasts').query("metric in ['average_separation','last_minus_first_separation']"))
display(read_table('score_appearance_lead_days'))
coeff=read_table('model_coefficients').query("variant=='Profile + Departure'")
display(coeff[~coeff.feature.str.startswith('missingindicator')].assign(
 absolute_coefficient=lambda f:f.standardized_log_odds.abs()).sort_values('absolute_coefficient',ascending=False).groupby('category').head(8))"""),
('markdown',"""## Sensitivities motivated by the observed comparisons
Refit both representations on identical filtered rows for: complete P+D measurements; the same explicit P+D availability flags in both models; removal of final-six-day positive rows (never relabel them healthy); and measurement-only profiles without the five slow context fields. Repeat the missingness/timing checks with common workload/role/calendar adjustment for elbow/UCL and shoulder.

These checks address possible missingness artifacts, near-event diagnosis behavior and stable history effects. They are not new independent confirmatory studies. Shared-availability models disable duplicate automatic missingness flags. No learned transform sees validation/test outcomes."""),
('code',"""ensure_stage('sensitivity',a,events,spec,matches,force=FORCE_RECOMPUTE)
plots.robustness_figure();plt.show()
display(read_table('sensitivity_coverage'))
display(read_table('adjusted_sensitivity_increment'))"""),
('markdown',"""## Training stability
Twelve paired whole-pitcher training bootstraps refit every preprocessing step and coefficient. The feature definitions and regularization remain frozen. Score-rank correlation and top-5% alert-set Jaccard compare each refit with that representation's unperturbed fit. Refit AP differences are descriptive perturbations, not 12 independent tests or an uncertainty interval for future generalization."""),
('code',"""ensure_stage('stability',a,events,spec,matches,force=FORCE_RECOMPUTE)
plots.stability_figure();plt.show()
display(read_table('stability_summary'))
display(read_table('training_refit_ap_differences').groupby(['category','partition']).ap_difference.agg(['mean','min','max']))"""),
('code',"""from src.injury_analysis.profile_report import write_synthesis
display(Markdown(write_synthesis()))""")])
print(f'Wrote two notebooks to {DEST}')

