"""Create additive nonlinear notebooks without replacing executed versions."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'notebooks/ucl_nonlinear';DEST.mkdir(parents=True,exist_ok=True)
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)
setup="""from pathlib import Path
import sys, json
ROOT=Path.cwd()
while not (ROOT/'src/injury_analysis').exists() and ROOT!=ROOT.parent:
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
import pandas as pd
from IPython.display import display, Image, Markdown
from src.injury_analysis.ucl_nonlinear_common import OUT, CACHE, CFG, read_table
pd.set_option('display.max_columns',20)
protocol=json.loads((OUT/'frozen_protocol.json').read_text())
def show(name,n=25):display(read_table(name).head(n))
def plot(name):display(Image(filename=str(OUT/'figures'/(name+'.png')),width=1150))
"""
notebooks={
'01_model_form_and_validation.ipynb':[
md('# Does nonlinear model form improve UCL risk prediction?\n\nCompare linear logistic regression, additive cubic spline logistic models and constrained histogram boosting. Stumps provide an additive tree comparison. Reuse the refined UCL endpoint, frozen representations and 30-day appearance landmarks. No previous stage is redone.'),
code(setup),
code("from src.injury_analysis.ucl_nonlinear_models import run_models\nrun_models()  # verified checkpoint reuse\nshow('cohort_counts')\ndisplay({k:protocol[k] for k in ['warning_leaders','ranking_leaders','frozen_at_utc']})"),
md('## Fixed design and modest search\n\nTraining: 2015–2021; validation: 2022–2023; previously examined exploratory transfer: 2024–2025. Preserve the 60-day embargo, first-event competing-injury outcomes, observed follow-up, source NULLs and post-completed-outing cutoff. Fit 35 nonlinear candidates across seven existing representations: two spline penalties and three tree sizes. Knots, imputation, clipping and scaling are training-only. No class rebalancing, synthetic injuries or automatic random early-stopping split.\n\nThe primary selection uses validation ≥7-day warning recall at 5% budget; an AP leader is frozen separately. Choices and thresholds were saved before transfer predictions. The two leaders answer different questions.'),
code("show('development_search',n=20)\nshow('representation_comparison',n=25)"),
code("plot('01_ranking_by_representation')\nshow('leader_comparison',n=20)"),
md('**Result:** Nonlinear departure models do not improve AP. Profile splines and some profile trees can produce large validation AP, while nonlinear broader representations do not uniformly outperform linear benchmarks. The profile spline’s AP gain must be examined for event concentration before it can be considered a reproducible finding.'),
code("plot('03_overfitting_and_calibration')\nshow('training_validation_gap',n=20)"),
md('**Overfitting:** The profile-tree AP leader has training AP 0.1295, validation AP 0.0294 and transfer AP 0.00646. A strong training fit does not establish a generalizable nonlinear risk relationship. The transfer period remains exploratory; it did not choose these models.'),
code("show('additivity_comparison',n=25)")],
'02_nonlinear_effects_and_stability.ipynb':[
md('# What do the flexible models learn, and how stable is it?\n\nDistinguish curved single-feature response shapes from interactions. These plots describe fitted log-odds associations, not physiological injury thresholds or causal mechanisms.'),
code(setup),
code("from src.injury_analysis.ucl_nonlinear_analysis import run_analysis\nfrom src.injury_analysis.ucl_nonlinear_probes import run_probes\nfrom src.injury_analysis.ucl_nonlinear_diagnostics import run_diagnostics\nrun_analysis();run_probes();run_diagnostics()\nref=pd.read_parquet(CACHE/'training_effect_reference.parquet')\ndisplay(pd.Series({'reference_outings':len(ref),'reference_pitchers':ref.pitcher_id.nunique(),'maximum_outings_per_pitcher':ref.groupby('pitcher_id').size().max(),'latest_reference_date':ref.game_date.max()}))"),
md('## Training-supported local effects\n\nALE uses observed values within training-reference quantile bins. The reference caps each pitcher at five outings; no outcomes choose the sample. Curves use centered local log-odds differences. Shading is the range of eight whole-pitcher refits, not a confidence interval. Missing source values do not enter the plotted observed-value distribution.'),
code("plot('04_nonlinear_effect_curves')\nshow('effect_shape_stability',n=20)"),
md('**Interpretation:** The lower-velocity/higher-risk direction is already present in the linear model. Flexible release, extension and spin curves do not consistently improve ranking or warning. Historical profile release variability and extension can have curved effects, but broad refit ranges and concentrated AP limit interpretation. A flat workload curve in a selected stump model means that fit did not use that measurement; it does not prove workload is irrelevant.'),
code("plot('05_interactions_and_stability')\nshow('interaction_stability',n=25)"),
md('Local four-corner log-odds contrasts cancel additive main effects, including curved GAM effects. They are a local nonadditivity diagnostic, **not** full second-order ALE. Cells require ≥40 reference observations and ≥15 pitchers. Selected profile+departure and previous-62 boosting models are stumps and therefore have zero log-odds interactions. Other tree fits learn nonadditivity, but this alone does not show incremental predictive value. Profile-only pairs are supplemental interpretation probes, not new model selection.'),
code("show('permutation_summary',n=25)\nshow('permutation_stability',n=25)"),
md('Permutation is marginal and may break correlated predictor relationships. Importance is not causal or unique information. Stable-looking curves are not enough if performance depends on a few pitchers.'),
code("plot('06_composition_and_refit_uncertainty')\ng=read_table('event_ap_contributions')\ndisplay(g.loc[g.model_id.eq(protocol['ranking_leaders']['Spline'])].sort_values(['partition','fraction_total_ap'],ascending=[True,False]).groupby('partition').head(3))"),
code("g=read_table('leave_one_injured_pitcher_out')\ndisplay(g.loc[g.model_id.eq(protocol['ranking_leaders']['Spline'])].sort_values(['partition','ap_drop_without_pitcher'],ascending=[True,False]).groupby('partition').head(3))\nshow('high_influence_pitcher_profiles',n=15)"),
md('**Composition sensitivity:** One event contributes 68% of validation spline-profile AP; another contributes 79% of transfer AP. Removing its pitcher lowers AP to 0.00598 and 0.00731, respectively. These cases should remain in the dataset. The diagnostic changes the interpretation of aggregate AP rather than selecting a convenient cohort.')],
'03_warning_transfer_and_conclusions.ipynb':[
md('# Do nonlinear scores provide more useful UCL warning?\n\nThe practical question is event detection far enough ahead, under a controlled alert burden. Compare ranking and warning separately. 2024–2025 is exploratory transfer, not prospective confirmation.'),
code(setup),
code("plot('02_warning_and_alert_budget')\nw=read_table('leader_warning_comparison')\ndisplay(w.loc[w.alert_rate.eq(.05)&w.policy.eq('retrospective_budget'),['partition','model_id','minimum_lead','alerts','alerted_events','qualifying_events','median_warning_lead_days','alerts_per_detected_event','false_alert_fraction']])"),
md('At a 5% budget, departure trees warn on 8/34 validation events and 3/43 transfer events ≥7 days ahead, versus 7/34 and 2/43 for logistic regression. Spline departure warns on 7/34 and 3/43. The differences are small relative to uncertainty. The tree’s ≥21-day warnings drop from 5/34 to 0/43. Conditional lead-time medians among a few detected events do not imply broadly earlier detection.'),
code("show('warning_uncertainty',n=30)\ng=read_table('paired_uncertainty')\ndisplay(g.loc[g.model_id.isin(list(protocol['warning_leaders'].values())+list(protocol['ranking_leaders'].values()))])"),
code("show('operational_burden',n=30)\nshow('subgroup_diagnostics',n=25)"),
md('Retrospective top-N budgets differ from frozen numeric thresholds. Realized alert rates can change in transfer. The operational table includes all eligible landmarks and preserves unknown outcome alerts; it does not assume that censored rows were healthy. Subgroup recall denominators use observable events and are not directly comparable to primary all-event recall.'),
code("g=read_table('training_refit_metrics')\ndisplay(g.groupby('model_id')[['average_precision','injury_event_recall']].agg(['mean','min','max']))"),
md('**Conclusion:** Flexible models reveal curved profile associations and some fitted interactions, but no robust, broadly transferable improvement in practical UCL warning is established. The most impressive spline AP result is concentrated in very few pitchers. Keep the original linear departure/broader benchmarks, retain frozen nonlinear departure challengers, and treat the profile spline as a separate composition-sensitive hypothesis. Do not choose a new winner from transfer results.'),
code("from src.injury_analysis.ucl_nonlinear_reporting import validate\nchecks=validate()\ndisplay(pd.Series({'validation_checks':len(checks),'passed':sum(c['passed'] for c in checks)}))\nshow('validation_checks',n=12)"),
code("display(Markdown((OUT/'summary.md').read_text()))")]
}
for name,cells in notebooks.items():
    p=DEST/name
    if p.exists():continue
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'}})
    nbf.write(nb,p);print(p)
