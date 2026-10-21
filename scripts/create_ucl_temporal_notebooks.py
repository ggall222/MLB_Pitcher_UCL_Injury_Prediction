"""Create focused temporal notebooks, preserving any existing executed versions."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/ucl_temporal';DEST.mkdir(parents=True,exist_ok=True)
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
setup="""from pathlib import Path
import sys, json
ROOT=Path.cwd()
while not (ROOT/'src/injury_analysis').exists() and ROOT!=ROOT.parent:
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
import pandas as pd
from IPython.display import display, Image, Markdown
from src.injury_analysis.ucl_temporal_common import OUT, CACHE, read_table
pd.set_option('display.max_columns',24)
def show(name,n=30): display(read_table(name).head(n))
def plot(name): display(Image(filename=str(OUT/'figures'/(name+'.png')),width=1150))
"""
notebooks={
'01_temporal_family_decomposition.ipynb':[
md('# Do feature families carry different information at different UCL lead times?\n\nFirst-pass landmark logistic analysis. The refined UCL endpoint, frozen representations and chronological cohort are unchanged. The four intervals are disjoint: 1–7, 8–14, 15–30 and 31–60 days. No temporal ordering is assumed.'),
code(setup),
code("from src.injury_analysis.ucl_temporal_models import run_development\nrun_development()  # verify/reuse completed fits\nshow('cohort_support',n=20)\nshow('censoring',n=30)"),
md('All intervals share the observed 60-day first-event cohort. This keeps the appearance population constant, at the cost of restricting follow-up. A UCL event in another interval is a zero for this specific outcome; a known intervening injury is a competing first event. Uncertain outcomes remain censored. Only 10 validation events support 31–60 days.'),
code("show('family_definitions',n=15)\nshow('omitted_groups',n=12)"),
md('The 113-field union spans 11 usable families. Frozen Profile, Background, Departure-8, Previous-62 and Stable-62 are unchanged. Arm angle and within-outing velocity decline inherit historical-coverage exclusions; prior UCL history inherits its publication-timing exclusion. Fixed L2 logistic C=1 is used everywhere; all learned preprocessing is training-only.'),
code("plot('01_family_standalone')\ng=read_table('family_matrix_validation')\ndisplay(g[['family','interval','standalone_lift','standalone_lift_ci_low','standalone_lift_ci_high','represented_events']])"),
md('Large standalone PR lift is not evidence of uniquely useful information, nor proof that one family is acute or chronic. Intervals differ in width and represented events. The next comparison removes each family from a broader model.'),
code("plot('02_family_incremental')\ndisplay(g[['family','interval','ablation_ap_loss','ablation_ci_low','ablation_ci_high','permutation_ap_loss','permutation_ci_low','permutation_ci_high']])"),
md('Negative ablation/permutation values are retained. Refitting after removal can improve a correlated or overfit model. Joint within-season family shuffling measures marginal reliance, not causal or fully conditional importance. Asterisks are pointwise uncertainty only.'),
code("plot('03_frozen_representations')\nshow('reference_temporal_contrasts_validation',n=10)\nshow('followup_sensitivity',n=12)"),
md('Departure-8 has a larger near-versus-far point estimate, but its paired uncertainty includes zero. Profile does not establish constant susceptibility across horizons. The interval-specific follow-up sensitivity helps separate signal from the common cohort restriction. Transfer remains exploratory and did not define intervals or feature groups.')
],
'02_temporal_uncertainty_and_composition.ipynb':[
md('# Are apparent timing patterns stable or driven by a few pitchers?\n\nUse whole-pitcher validation resampling, joint temporal contrast bands, six whole-pitcher training refits, event-concentration diagnostics and a shared-event sensitivity. These checks are distinct from one another.'),
code(setup),
code("from src.injury_analysis.ucl_temporal_analysis import analyze_partition, run_refits\nanalyze_partition('validation'); run_refits()\nshow('candidate_ranking',n=15)\ndisplay(json.loads((OUT/'development_freeze.json').read_text())['development_summary'])"),
md('The candidate nomination rule uses validation only: mean ablation AP loss divided by interval prevalence. Candidates are narrow follow-up hypotheses, not established timing-specific mechanisms. The refits are conditional on nomination rather than a nested selection experiment.'),
code("plot('04_temporal_stability')\nshow('temporal_contrasts_validation',n=33)"),
md('The joint bands cover all 33 predeclared family contrasts (each nearer interval versus 31–60 days). None excludes zero. Two pointwise role contrasts are insufficient to declare a longer-horizon role signal after considering the full contrast family.'),
code("show('training_refit_family_losses',n=25)\nshow('role_composition_validation',n=8)"),
code("c=read_table('event_ap_contributions_validation')\nfor name,h in [('Only | Velocity','1-7'),('Only | Workload and rest','31-60')]:\n    display(c.loc[c.model.eq(name)&c.interval.eq(h)].nlargest(4,'fraction_total_ap'))\ni=read_table('pitcher_influence_validation')\ndisplay(i.loc[(i.model.eq('Only | Velocity')&i.interval.eq('1-7'))|(i.model.eq('Only | Workload and rest')&i.interval.eq('31-60'))].sort_values('ap_drop',ascending=False).head(8))"),
md('One event supplies about 81% of near-term velocity-family AP, and another supplies about 69% of longer-horizon workload-family AP. These cases remain in the primary cohort. The diagnostics constrain interpretation of the visually attractive temporal ordering.'),
code("s=read_table('shared_event_sensitivity_validation')\ndisplay(s.loc[s.model.isin(['Profile','Departure','Full union','Only | Velocity','Only | Workload and rest'])])"),
md('The shared-event sensitivity evaluates only UCL events represented in all four intervals, keeping observed comparison rows and saved models. It is retrospective and small. It tests whether interval composition may contribute to apparent heterogeneity; it is not a replacement cohort or a new fit.'),
code("show('family_matrix_test',n=16)\nshow('reference_temporal_contrasts_test',n=10)"),
md('2024–2025 is previously examined exploratory transfer. It cannot be used to revise the candidate families, interval boundaries or model settings. Null, opposing and unstable results remain in the outputs.')
],
'03_matched_trajectories_warning_and_summary.ipynb':[
md('# Do family scores change before UCL episodes, and do they provide useful warning?\n\nRetrospective matched trajectories complement the predictive matrix. Calendar-time and appearance-time alignment answer different questions.'),
code(setup),
code("from src.injury_analysis.ucl_temporal_trajectories import run_trajectories\nrun_trajectories()\nshow('trajectory_score_scales',n=12)\nshow('appearance_calendar_support',n=30)"),
md('Each family uses a single training-derived 60-day logistic score at every plotted time point, standardized by its training log-odds mean/SD. This avoids changing the score definition as the event approaches. Original matched controls are reused. Injury events and controls contribute equal weight after within-bin averaging.'),
code("plot('05_matched_family_trajectories')\nt=read_table('trajectory_shape')\ndisplay(t.loc[t.partition.eq('validation')])"),
md('Calendar plots require the same matched pairs in all four bins. Outing plots require the same pairs in final/2–3/4–5 outings; available-pair farther-out results are retained separately. Bands are pointwise dependency-cluster bootstrap intervals, not simultaneous physiological evidence. A persistent offset is not imminent tissue damage.'),
code("g=read_table('trajectory_summary')\ndisplay(g.loc[g.partition.eq('validation')&g.kind.eq('difference')&g.support.eq('available_pairs')&g.family.isin(['Velocity','Workload and rest'])])"),
code("plot('06_interval_warning')\nfor name in ['development_metrics','transfer_metrics']:\n    d=read_table(name)\n    display(d.loc[d.partition.ne('train')&d.model.isin(['Departure','Profile','Stable 62']),['partition','interval','model','alerted_events','qualifying_events','observable_events','alerts_per_detected_event','non_actionable_alert_fraction']])"),
md('These are interval-specific retrospective top-5% budgets. The 1–7-day outcome includes warnings too late for the earlier ≥7-day practical target. All qualifying events remain in the denominator, including those without opportunities. No deployable threshold was selected.'),
code("display(Markdown((OUT/'summary.md').read_text()))"),
code("from src.injury_analysis.ucl_temporal_reporting import validate\nchecks=validate()\ndisplay(pd.Series({'checks':len(checks),'passed':sum(x['passed'] for x in checks)}))"),
md('The first pass does not establish a reproducible acute/intermediate/longer-horizon ordering. Freeze definitions and test narrow hypotheses on a genuinely future chronological cohort. More flexible temporal models should follow adequate event support and replication, not the largest exploratory cell.')
]}
for name,cells in notebooks.items():
    p=DEST/name
    if p.exists():continue
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'}})
    nbf.write(nb,p);print(p)
