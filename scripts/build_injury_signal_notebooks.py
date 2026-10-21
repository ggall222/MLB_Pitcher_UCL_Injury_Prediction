"""Create the three notebook entry points for the injury signal-analysis extension."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/injury_signal_analysis'
DEST.mkdir(parents=True,exist_ok=True)
SETUP='''from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/injury_signals.yaml').exists())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.injury_analysis.signals_common import CFG,OUT,CACHE,inputs,read_table
from src.injury_analysis.common import conclusion
from src.injury_analysis import signal_plots as plots
pd.set_option('display.max_columns',16)
'''

def write(name,title,question,cohort,cells):
 n=nbf.v4.new_notebook()
 n.metadata.update(kernelspec={'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},language_info={'name':'python','version':'3.9'})
 n.cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n**Question.** {question}\n\n**Cohort.** {cohort}'),nbf.v4.new_code_cell(SETUP)]
 for kind,text in cells:n.cells.append(nbf.v4.new_code_cell(text) if kind=='code' else nbf.v4.new_markdown_cell(text))
 nbf.write(n,DEST/name)

write('01_feature_family_decomposition.ipynb','01 — Decomposing information by injury type',
'Which feature families carry ranking and warning information, and does Statcast add value beyond workload/history?',
'The existing seven-category, 30-day comparison cohorts are reused. Each category compares eligible impending injuries against the same observed IL-free pool; competing injuries are excluded. Training ends in 2021, validation covers 2022–2023, and the already examined 2024–2025 test period remains exploratory. No new claim of untouched confirmation is made. Model variants use identical rows within each comparison.',[
('markdown','''## Design and safeguards
Compare every single family, meaningful combinations, workload + history (WH), and all 62 historical predictors. Fit the same logistic regression with C=1 for every specification; no hyperparameter search. Training-only median imputation/scaling leaves stored NULLs intact. Covariates contain no future injury/activation/surgery information. The prediction time remains after a completed outing.

Paired pitcher bootstrap draws compare AP on the **same rows** for all variants. Intervals condition on fitted models; they do not refit the training procedure or correct for all exploratory comparisons. Event recall requires warnings 7–30 days before injury and retains events without observed warning outings in its denominator. Random-alert expectations account for multiple opportunities per event.'''),
('code','''from src.injury_analysis.family_decomposition import run_decomposition,run_modern
a,events,spec,matches=inputs()
metrics=run_decomposition(a,events,spec)
display(read_table('family_definitions')[['variant','features']])
display(read_table('family_cohort_counts'))
print(f'Existing source: {len(a):,} pitcher appearances; {events.target_eligible.sum():,} eligible injury targets; {len(matches):,} matched trajectory pairs.')'''),
('code','''plots.family_overview();plt.show()
plots.incremental_effects();plt.show()
display(metrics[metrics.variant.eq('Full')][['category','partition','average_precision','pr_lift','ap_delta','delta_low','delta_high']])'''),
('markdown','''## Practical alert performance
Average precision and event detection answer different questions. An above-prevalence row-ranking score need not beat random event detection by much. The following compares the simple baseline, a velocity/mechanics extension and the full model at identical budgets. All-event and observable-event denominators, precision, recall, lead times and non-actionable burden are saved. Seven-day eligibility means an alert one day before placement is too late for this practical objective.

The event-recall difference intervals resample a pitcher's events conditional on fixed observed top-5% selections. Uniform-random values are exact expectations, not a simulated deployed policy. Validation-fixed thresholds are saved separately because test alert rates can drift.'''),
('code','''plots.alert_comparison();plt.show()
budgets=read_table('family_alert_budgets')
display(budgets.query("partition=='test' and alert_rate==0.05 and variant in ['Workload + history','Full']")[[
 'category','variant','qualifying_events','observable_events','injury_event_recall','random_expected_event_recall','precision','median_warning_lead_days','non_actionable_alerts_per_pitcher_season']])
display(read_table('family_event_recall_differences').query("partition=='test' and variant=='Full'"))'''),
('markdown','''## Variable inspection and modern arm angle
Inspect variables in the top two individual-family **validation point estimates**, using the full model's standardized coefficients and validation permutations within season/current-role strata. Five-shuffle variability is not a sampling confidence interval. Selection and inspection use validation, so this is hypothesis generation. Correlated variables can mask or split importance; role itself is fixed by the permutation strata.

Arm angle is evaluated separately: training starts in 2020, original split boundaries remain, and angle-only/full-minus-angle/full-plus-angle models use identical complete-angle rows. Require ≥30 training and ≥10 validation/test events. This smaller, shorter-era study must not be compared as if it used the full historical training cohort.'''),
('code','''plots.importance_figure();plt.show()
display(read_table('variable_permutation_summary').sort_values('mean',ascending=False).groupby('category').head(4))
modern=run_modern(a,events,spec)
display(read_table('modern_cohort_counts'))
display(modern.query("variant=='Full + arm angle'")[['category','partition','average_precision','ap_delta','delta_low','delta_high']])
f=metrics.query("partition=='test' and variant=='Full'").set_index('category')
v=metrics.query("partition=='validation' and variant=='Full'").set_index('category')
conclusion(f"The full model exceeds WH by point estimate in {f.ap_delta.gt(0).sum()}/7 test categories and {v.ap_delta.gt(0).sum()}/7 validation categories. Elbow/UCL test AP changes by {f.loc['Elbow / UCL','ap_delta']:+.4f}; shoulder by {f.loc['Shoulder','ap_delta']:+.4f}; lower body by {f.loc['Lower body','ap_delta']:+.4f}. Forearm's validation/test changes are {v.loc['Forearm','ap_delta']:+.4f}/{f.loc['Forearm','ap_delta']:+.4f}. The paired intervals and period differences do not establish a universal Statcast advantage or a stable ranking of biological injury specificity.")''')])

write('02_preinjury_trajectories.ipynb','02 — Balanced pre-injury trajectories',
'Do pitching measures drift gradually, change late, or simply differ in persistent pitcher profiles?',
'Reuse exactly the 1,225 existing matched event/control pairs and their selected outing keys. No rematching or future baseline reconstruction occurs. Expand only with existing database fields. Calendar bins cover 31–60, 15–30, 8–14 and 1–7 days; appearance time covers the final five outings within 60 days. All-year trajectories include descriptive 2026 data, unlike predictive evaluation.',[
('markdown','''## Paired support and feature selection
Each plotted balanced feature/category keeps the **same events and both members of each pair at every point**. A second table retains all available pairs at each point, making attrition visible. This corrects changing-composition ambiguity in the initial descriptive curves. It also selects more continuously observed histories, so its smaller cohort is not automatically more representative.

The core feature set spans velocity, variability, release, extension, angle, spin/movement, command, pitch mix and workload. Additional dynamic predictors are selected transparently from notebook 01's validation inspection. Their selection is exploratory and recorded. Zero for a pitcher z-score means the pitcher's recent baseline; units are not mph. Workload acceleration remains in pitches/day.'''),
('code','''from src.injury_analysis.trajectory_shapes import expand_alignment,run_balanced_trajectories,run_shapes
a,events,spec,matches=inputs()
aligned,features=expand_alignment(a,events,matches)
summary=run_balanced_trajectories(aligned,features)
shapes=run_shapes(aligned,features)
display(read_table('trajectory_cohort_counts'))
display(read_table('trajectory_feature_selection'))
print(f'{len(aligned):,} aligned appearances; {aligned.event_id.nunique():,} event pairs; {len(features)} features.')'''),
('code','''plots.case_control_trajectories('calendar');plt.show()
plots.case_control_trajectories('appearance');plt.show()
display(summary.query("feature=='velo_z' and kind=='difference'")[['mode','category','point','mean','simultaneous_low','simultaneous_high','events','pitchers']])
display(read_table('velocity_appearance_calendar_support').query('appearance_before==1'))'''),
('markdown','''## Mean versus variability; mechanic and workload contrasts
Unsmoothed point/line plots display case minus control. A final negative velocity difference can be distinct from a sustained negative slope. Variability is examined separately rather than assumed to worsen whenever mean velocity falls.

Bands in the paired panels are simultaneous across displayed time points within each feature/category, using pitcher-cluster resampling. They do not correct for all categories/features. Counts in each legend are constant across that curve but differ by feature and category. Modern angle is based on observed same-season pairs, with no backfilled zeros.'''),
('code','''plots.paired_trajectory_panels('appearance');plt.show()
plots.paired_trajectory_panels('calendar');plt.show()
plots.paired_trajectory_panels('appearance',modern=True);plt.show()
display(read_table('available_pair_trajectory_summary').query("feature=='velo_z' and mode=='calendar' and kind=='difference'")[['category','point','events','mean']])'''),
('markdown','''## Heterogeneity and stable risk profiles
The final-step distribution shows whether an average change describes most events or masks wide heterogeneity. Baseline fastball speed can distinguish the kinds of pitchers in the injury cohort even when it does not suddenly change before the event. That is a plausible statistical explanation for predictive information without a clear mean warning trajectory, not evidence of a causal mechanism or nonlinear relationship.'''),
('code','''plots.heterogeneity_figure();plt.show()
v=shapes.query("feature=='velo_z' and metric=='final_step_vs_previous_two'").set_index('category')
display(v[['mean','low','high','events','median','q10','q90']])
conclusion(f"The balanced final-outing velocity change exceeds the corresponding control change by {v.loc['Elbow / UCL','mean']:.3f} z-units for elbow/UCL and {v.loc['Shoulder','mean']:.3f} for shoulder, versus {v.loc['Lower body','mean']:.3f} for lower body. Event-level distributions are broad. The shapes and simultaneous intervals distinguish a late average step from a claim of sustained early deterioration; calendar and appearance samples have different support.")''')])

write('03_trajectory_timing_and_sensitivity.ipynb','03 — Timing, robustness and synthesis',
'How early is divergence supported, which findings survive plausible restrictions, and how do trajectories relate to feature-family performance?',
'Use the balanced/available matched trajectories from notebook 02 and the fixed model family definitions from notebook 01. Sensitivities address uncertain episodes, final-week information, role, historical baseline strength, prior injuries and period changes. They do not manufacture clinical onset dates or infer recovery from roster return.',[
('markdown','''## Quantify shape without forcing breakpoints
One event contributes a paired slope/contrast: linear slope across five outings, final-two minus first-two mean, last-outing minus previous-two mean, and late-versus-early slope change. Calendar slopes use actual irregular days and require at least three distinct dates spanning an earlier ≥30-day observation and a last observation ≤14 days before the event.

For z-score features, maximum absolute departure and the fraction of outings beyond |z|≥1 describe heterogeneity/persistence; this reference is **not a clinical abnormality threshold**. No breakpoint is forced from five irregular observations. Timing separately reports isolated supported points and two adjacent same-direction simultaneous intervals excluding zero, requiring ≥30 balanced events. An isolated point does not establish onset, and support at the earliest point can predate the observed window.'''),
('code','''from src.injury_analysis.trajectory_shapes import run_trajectory_sensitivity
from src.injury_analysis.family_decomposition import run_predictive_sensitivity
from src.injury_analysis.signal_report import write_synthesis
a,events,spec,matches=inputs()
aligned=pd.read_parquet(CACHE/'expanded_alignment.parquet')
shapes=read_table('trajectory_shape_metrics')
plots.shape_figure();plt.show()
display(shapes.query("feature=='velo_z'")[['category','metric','mean','low','high','events']])
timing=read_table('trajectory_supported_timing')
display(timing.query("feature=='velo_z'")[['mode','category','balanced_events','individually_supported_points','first_sustained_point']])'''),
('markdown','''## Targeted sensitivity checks
Strict QA does not redefine the primary cohort; it tests dependence on unresolved flags. Removing final 3 or 7 days retains the original event/pseudo-event anchors and does not reindex earlier outings as the last outing. Same-pair contrasts separately address attrition. Strong baseline means ≥15 measured prior observations for the corresponding z-score, not merely long career experience.

First/prior episode uses the as-of reported IL count, not lifetime injury status. Starter/reliever strata use the matched index-outing role. Explicit throwing-side analysis requires known side for arm injuries. Period-specific trajectories reveal small samples rather than claiming a fresh validation. The direct arm-minus-non-arm contrast uses shared pitcher bootstrap draws across groups.'''),
('code','''sensitivity=run_trajectory_sensitivity(aligned)
plots.sensitivity_figure();plt.show()
display(sensitivity.query("feature=='velo_z'")[['condition','category','mean','low','high','events']])
display(read_table('trajectory_sensitivity_same_pair_contrasts').query("feature=='velo_z'"))
display(read_table('arm_nonarm_direct_contrasts'))'''),
('markdown','''## Does the predictive conclusion survive timing restrictions?
Refit WH, WH+velocity/mechanics and Full using the same rows under strict QA or excluding all positive appearances in the final six days. Those late rows become excluded, not healthy negatives. Restricting positive definitions changes prevalence, so compare models **within** each sensitivity, not raw AP across cohorts. Frozen model role comparisons reuse the original fitted models.

The 2024–2025 period has already informed this exploration; it is not an untouched final test. Model and trajectory conclusions should be replicated using adjudicated timing and a new held-out period.'''),
('code','''refits=run_predictive_sensitivity(a,events,spec)
display(refits[refits.variant.eq('Full')][['category','analysis','rows','represented_events','average_precision','ap_delta','delta_low','delta_high']])
display(read_table('family_role_sensitivity'))
text=write_synthesis()
display(Markdown(text))
v=sensitivity.query("feature=='velo_z' and condition=='Exclude final 7 days'").set_index('category')
conclusion(f"After excluding the final week, the matched shoulder velocity difference is {v.loc['Shoulder','mean']:.3f} (95% interval {v.loc['Shoulder','low']:.3f} to {v.loc['Shoulder','high']:.3f}); elbow/UCL is {v.loc['Elbow / UCL','mean']:.3f} ({v.loc['Elbow / UCL','low']:.3f} to {v.loc['Elbow / UCL','high']:.3f}). These results refine the timing claim and should be read alongside the smaller balanced samples, paired model increments and random-alert reference in the computed synthesis. No single best model or biological onset estimate is selected.")''')])
print(f'Wrote three notebooks to {DEST}')
