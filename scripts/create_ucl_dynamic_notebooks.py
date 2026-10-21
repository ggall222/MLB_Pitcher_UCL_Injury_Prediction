"""Create the three additive notebook entrypoints; never overwrite executed work."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/ucl_dynamic';DEST.mkdir(parents=True,exist_ok=True)
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)
setup="""from pathlib import Path
import sys, json
ROOT = Path.cwd()
while not (ROOT / 'src/injury_analysis').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display, Image, Markdown
from src.injury_analysis.ucl_dynamic_common import OUT, CACHE, read_table, valid_checkpoint
pd.set_option('display.max_columns', 20)
def show(name, **kwargs):
    display(read_table(name).head(kwargs.get('rows', 30)))
def plot(name):
    display(Image(filename=str(OUT / 'figures' / (name + '.png')), width=1100))
"""
notebooks={
'01_endpoint_and_landmarks.ipynb':[
md('# UCL-specific endpoint and appearance landmarks\n\n**Question:** Which documented UCL episodes support a defensible ahead-of-time prediction study? This stage starts after completed feature consolidation. The database and previous results are preserved. Events are source-documented, not independently clinically adjudicated.'),
code(setup),
code("from src.injury_analysis.ucl_dynamic_data import prepare_data\na, events, specs = prepare_data()\ndisplay(pd.Series({'source_appearances':len(a),'primary_UCL_events':int(events.endpoint_primary.sum()),'unique_primary_pitchers':events.loc[events.endpoint_primary,'player_id'].nunique(),'strict_official_events':int(events.endpoint_strict_official.sum()),'expanded_probable_events':int(events.endpoint_expanded.sum())}))"),
md('## Outcome specificity and knowledge time\n\nAn official non-procedure UCL diagnosis or a bounded compatible MLB procedure link supports the primary endpoint. Procedure-only/recovery text is not enough to establish a new injury. Forearm/flexor presentations can qualify with subsequent compatible evidence. The retrospective event label may use later confirmation; predictors cannot. Surgery date, report date, diagnosis date and earliest event proxy remain separate. No date is called exact biological onset.'),
code("show('endpoint_counts', rows=40)\nshow('ucl_events_by_season', rows=20)"),
code("cols=['player_name','prediction_reference_date','first_public_indication_date','diagnosis_date','evidence_category','initial_injury_region','linked_procedure_ids','source_publication_archive_verified']\ndisplay(events.loc[events.endpoint_primary,cols].head(15))\ndisplay(events.loc[events.endpoint_primary,'timing_uncertainty_days'].describe())"),
md('## Appearance cohort and first-event outcomes\n\nPredict after a completed eligible regular-season MLB outing. Reuse quality/history, active-episode and 60-day chronological embargo rules. Known non-UCL first injuries are competing negatives; suspected UCL/ambiguous arm episodes censor the row. Ordinary negatives require observed MLB follow-up. Starters and relievers remain included. Never skip over an intervening episode to a later UCL event.'),
code("from src.injury_analysis.ucl_dynamic_data import risk_frame\nf = risk_frame(a,events)\ndisplay(f.loc[f.target.notna()].groupby('partition').agg(rows=('target','size'),positives=('target','sum'),pitchers=('pitcher_id','nunique')))\nshow('event_observation_opportunities', rows=12)"),
code("plot('01_endpoint_and_opportunity')"),
md('**Interpretation:** 194 eligible primary events remain, but only 20/34 validation and 28/43 exploratory-transfer events have a 7–30-day prediction opportunity. All eligible events stay in the principal recall denominator. Surgery-linked ascertainment and unknown contemporaneous publication history remain limitations. The official-only subset is too small for confident independent modeling.'),
code("show('feature_representations', rows=16)\nshow('trajectory_feature_support', rows=15)")],
'02_dynamic_models_and_warning.ipynb':[
md('# Dynamic landmark models and useful warning\n\n**Question:** Do departures, their recent trajectories or profile×departure interactions improve future UCL warning? Compare frozen representations under the same endpoint/risk set within each horizon. 2015–2021 is training, 2022–2023 validation, and 2024–2025 previously examined exploratory transfer.'),
code(setup),
code("from src.injury_analysis.ucl_dynamic_models import run_models\nrun_models()  # reuses verified checkpoints; does not refit completed models\nprotocol=json.loads((OUT/'frozen_protocol.json').read_text())\ndisplay({k:protocol[k] for k in ['selected_model','primary_horizon','threshold_5_percent_validation','selection_period','frozen_at_utc']})"),
md('Fixed L2 logistic models use train-only clipping, median imputation, missingness indicators and scaling. Twelve new trajectory fields summarize four z-scores over the current and three strictly earlier-day outings within 30 days. Three interactions use training-standardized profile and departure main effects. Independent 7/14/21/30-day models are not a jointly coherent survival curve. Selection maximizes validation 7–30-day event recall at 5% budget, with AP above prevalence; ties favor fewer fields, then AP. Selection was frozen before new transfer predictions.'),
code("show('summary_model_comparison', rows=20)\nplot('02_horizon_discrimination')"),
md('## Warning, rather than discrimination alone\n\nWarning credit requires the next event be the target UCL episode, at least 7/14/21 days ahead and within 30 days. Each alert counts. Top-N budgets are retrospective; the numeric validation threshold is the future application rule. Conditional warning lead time is not a lead time for all injuries.'),
code("show('summary_warning_comparison', rows=20)\nplot('03_warning_lead_and_burden')"),
code("w=read_table('warning_policies')\ndisplay(w.loc[w.model.eq(protocol['selected_model']),['partition','policy','minimum_lead','alerts','actual_alert_rate','alerted_events','qualifying_events','median_warning_lead_days','recall_ci_low','recall_ci_high']])"),
code("show('operational_all_landmark_burden', rows=20)"),
md('**Findings:** Departure-8 was selected by validation warning (7/34 events), but warns on only 2/43 transfer events at 5% budget. Stable-62 has the best AP among these fixed hypotheses and warns on 3/34 and 9/43; that transfer result cannot change the prior selection. Trajectories add no validation warning events over profile+departure. More than 98% of labeled departure alerts are negative for the first-event UCL outcome within 30 days, including known competing injuries. The all-landmark table also preserves unknown outcomes rather than assuming they were false alarms. No reliable prospective detector has been established.'),
code("show('paired_increment_uncertainty', rows=20)")],
'03_trajectories_robustness_and_freeze.ipynb':[
md('# Longitudinal risk, heterogeneity and the frozen next study\n\n**Question:** Is elevated predicted risk persistent or increasing as injury approaches, and do findings survive reasonable checks? Matched plots are descriptive development comparisons, not a new independent validation.'),
code(setup),
code("from src.injury_analysis.ucl_dynamic_robustness import run_robustness\nfrom src.injury_analysis.ucl_dynamic_trajectories import run_trajectories\nrun_robustness();run_trajectories()\nmatches=read_table('matched_controls')\ndisplay(matches.groupby('origin').size().rename('matched_events'))"),
md('Existing matched controls are preserved. Newly eligible events receive same-stratum matches only when necessary. Case/control windows use no control appearances after its index outing. Complete balanced support is required across all five outings or all four calendar bins. Uncertainty resamples connected sets sharing either a case or control pitcher. No smoothed curve or forced injury phenotype is used.'),
code("plot('04_matched_risk_trajectories')\ns=read_table('risk_trajectory_shapes')\ndisplay(s.loc[s.partition.eq('combined')&s.kind.eq('difference')])"),
code("plot('05_individual_risk_heterogeneity')"),
md('**Findings:** The five-outing plots contain 33 balanced events; calendar plots contain 16. Case-control change intervals cross zero. Departure risk does not rise consistently toward injury. Individual paths vary substantially. Persistent offsets cannot be interpreted as injury imminence, and risk scores are not physiological measurements.'),
code("plot('06_calibration_and_refit_stability')\nc=read_table('coefficient_stability')\ndisplay(c.loc[c.feature.str.contains(' × ')])"),
md('Eight whole-pitcher training refits show uncertain interaction directions: velocity, extension and release-variability interaction coefficients all change sign across refits. A tiny conditional AP gain is therefore insufficient evidence of a stable profile-dependent physiological relationship.'),
code("show('sensitivity_results', rows=40)\nshow('sensitivity_skips')"),
md('Competing-event censoring and public-date timing retain the principal small-sample warning pattern. Probable inclusion changes denominators. Strict official, unseen-pitcher and role subsets are too small to establish broad robustness; subgroup observable-event denominators differ from primary all-event recall. No confirmation is claimed from incomplete 2026 data.'),
code("from src.injury_analysis.ucl_dynamic_reporting import validate_stage\nchecks=validate_stage()\ndisplay(pd.Series({'checks':len(checks),'passed':sum(x['passed'] for x in checks)}))\nshow('validation_checks', rows=60)"),
code("display(Markdown((OUT/'summary.md').read_text()))"),
md('**Freeze:** Preserve departure-8 as the validation-selected hypothesis and all prespecified comparators and exact fit signatures. Apply the frozen numeric validation threshold without retuning to a prospectively acquired, mature post-freeze cohort. Archive sources as they become available, adjudicate outcomes blind to scores, and report unknown follow-up and realized alert burden. Do not present the current model as a clinical detector.')]
}
for filename,cells in notebooks.items():
    path=DEST/filename
    if path.exists():continue
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'}})
    nbf.write(nb,path)
    print(path)
