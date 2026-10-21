"""Generate the maintained notebook entry points from compact, reviewable cell sources."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/injury_type_analysis'
DEST.mkdir(parents=True,exist_ok=True)

SETUP='''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config/injury_analysis.yaml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.injury_analysis.common import CFG, CACHE, OUT, read_table, table, conclusion, style
from src.injury_analysis.cohort import load
from src.injury_analysis import plots
style()
pd.set_option('display.max_columns', 14)
'''


def write(name,title,objective,cohort,cells):
    nb=nbf.v4.new_notebook()
    nb.metadata.update(kernelspec={'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},
                       language_info={'name':'python','version':'3.9'})
    nb.cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n**Objective.** {objective}\n\n**Cohort and interpretation.** {cohort}'),nbf.v4.new_code_cell(SETUP)]
    for kind,source in cells:
        nb.cells.append(nbf.v4.new_code_cell(source) if kind=='code' else nbf.v4.new_markdown_cell(source))
    nbf.write(nb,DEST/name)


write('01_overview_and_categories.ipynb','01 — Injury dataset and category definitions',
'Inspect source coverage, detailed anatomy and diagnosis, then define auditable comparison groups.',
'All canonical MLB pitcher IL episodes remain visible. The analytical cohort uses regular-season pitcher/two-way appearances, both starters and relievers, with reliable appearance timing and at least five prior outings. Targets use initial anatomy, exclude uncertain timing/overlap/postoperative/ambiguous cases, and do not require a resolved return date. Run this notebook first. No source table is changed.',[
('code','''from src.injury_analysis.cohort import prepare
a, events, spec = prepare(force=True)
display(read_table('appearance_coverage'))
display(read_table('category_counts'))
print(f"{len(events):,} canonical episodes; {events.target_eligible.sum():,} eligible targets; {len(a):,} pitcher appearances.")
print(f"{a.analysis_eligible.sum():,} analysis-eligible outings; {a.pitcher_id.nunique():,} observed pitchers.")
print(f"Statcast through {a.game_date.max().date()}; injuries observed through {events.observation_end_date.max().date()}.")'''),
('markdown','''## Detailed classifications and exclusions
Broad groups pool upper arm with lat, trunk with neck, and the lower-body regions. Elbow/UCL is **not** synonymous with confirmed UCL injury. Original region, structure and diagnosis remain in the event cache and canonical source. Unknowns are not negative controls. Exclusion reason counts overlap; return-date-only QA does not exclude primary targets.'''),
('code','''display(read_table('initial_injury_region_counts'))
display(read_table('initial_injury_structure_counts').sort_values('episodes',ascending=False).head(18))
display(read_table('initial_injury_diagnosis_counts').sort_values('episodes',ascending=False))
display(read_table('event_exclusions_overlapping_counts'))
display(events.groupby(['body_group','initial_laterality'],dropna=False).size().rename('episodes').reset_index())
print('Explicit throwing-side eligible events:', events.loc[events.target_eligible,'throwing_arm_injury'].fillna(False).sum())
plots.overview_figure(events); plt.show()'''),
('markdown','''## Historical availability and temporal contract
The prediction timestamp is **after the completed outing**. Existing workload, rolling and pitcher baselines retain the complete prior history actually available; strict prior-date maxima are asserted. Initial label date is the earlier of the recorded injury proxy and reported placement date. Later surgery, activation and duration fields never enter X. Historical descriptions can be revised: reported dates are not verified historical publication timestamps.

Arm angle has a shorter observed history. The long-history model excludes its four fields. No source values are imputed; model-local training-only imputation is documented in notebook 03.'''),
('code','''plots.missingness_figure(); plt.show()
display(spec.loc[~spec.selected_for_prediction,['name','era','train_missing_fraction','rationale']])
display(read_table('analysis_validation_checks'))
parts = a.groupby('partition').agg(outings=('row_id','size'), observed_30d=('any_injury_30d','count'), positives_30d=('any_injury_30d','sum'))
display(parts)
unknown = events.analysis_group.eq('Unknown').mean()
conclusion(f"{events.target_eligible.sum():,}/{len(events):,} episodes meet the analysis target rules. Initial anatomy is unknown for {unknown:.1%} of all episodes. {spec.selected_for_prediction.sum()} features qualify for the common historical model. Coverage differences and the partially observed 2026 season require era and follow-up restrictions; eligibility does not establish biological onset.")''')])

write('02_preinjury_trajectories.ipynb','02 — Pre-injury trajectories and matched controls',
'Compare changes from each pitcher’s prior baseline as documented injuries approach, including non-arm negative comparisons.',
'Eligible events require at least three quality outings in the preceding 60 days and one within 14 days. Controls match year/month, index-outing role, five-year age band and prior-appearance experience band. Each control pitcher is used once, with no observed nearby IL episode and horizon follow-up. Pair selection uses outcome information solely to define the comparison, never predictors. All CIs are exploratory pitcher-cluster bootstrap intervals; no row-independent significance tests.',[
('code','''from src.injury_analysis.trajectories import build_trajectories
a, events, spec = load()
aligned, matches, effects = build_trajectories(a, events)
display(matches.groupby(['analysis_group','is_starter']).agg(pairs=('event_id','size'),case_pitchers=('pitcher_id','nunique')).reset_index())
display(read_table('trajectory_exclusions').reason.value_counts().rename('events'))
print(f"{len(matches):,} matched event/control pairs; {matches.control_pitcher_id.nunique():,} unique control pitchers.")'''),
('markdown','''## Calendar time and appearance-relative time
Calendar bins summarize 31–60, 15–30, 8–14 and 1–7 days before the index date. Each event contributes one mean per bin; frequent relievers do not gain extra weight within a bin. Appearance-relative plots show the last five outings within the 60-day window. Solid lines are cases, dashed lines their controls. Bands resample case pitchers. The number of observed events varies across bins/features; the full count columns are saved.'''),
('code','''plots.trajectory_figure('calendar'); plt.show()
plots.trajectory_figure('appearance'); plt.show()
display(read_table('calendar_trajectory_summary').query("feature == 'velo_z'")[['analysis_group','kind','window','mean','n','pitchers']])'''),
('markdown','''## Matched effect sizes and anatomical specificity
The next figure uses one 1–14-day feature mean per matched event and plots injury minus control. Trunk/lower-body injuries test whether similar warning patterns also occur outside the arm. Arm-region labels alone do not establish throwing-side involvement; the second table requires explicit laterality for that comparison. There is no multiplicity-adjusted hypothesis test or causal interpretation.'''),
('code','''plots.effects_figure(); plt.show()
display(effects.query("feature == 'velo_z'"))
body = read_table('body_group_preinjury_effects')
display(body[body.feature.isin(['velo_z','release_z_sd_z','extension_z'])])
v=effects.query("feature == 'velo_z'").set_index('analysis_group')
conclusion(f"Matched shoulder outings differ from controls by {v.loc['Shoulder','mean']:.2f} velocity z-units (95% interval {v.loc['Shoulder','low']:.2f} to {v.loc['Shoulder','high']:.2f}); elbow/UCL differs by {v.loc['Elbow / UCL','mean']:.2f}. Lower-body difference is {v.loc['Lower body','mean']:.2f}. These are group-level pre-IL associations, not proof of a specific injury mechanism or reliable individual warning.")''')])

write('03_comparative_predictability.ipynb','03 — Comparative predictability with chronological evaluation',
'Use one fixed, modest baseline to compare injury categories and horizons without choosing a final model.',
'One injury category versus a common observed IL-free pool; other impending injuries are excluded. Positive rows must precede an eligible event by 1–14/30/60 days. Negative rows require no episode in the window and an MLB outing at/after the horizon within a 30-day grace interval. This selects active pitchers and does not prove absence of symptoms. Include all eligible rows without class sampling.',[
('markdown','''## Fixed experiment
Train through 2021; validation 2022–2023; test 2024–2025. A 60-day boundary embargo and event-partition assertion prevent an event crossing retained partitions. 2026 is excluded from modeling. Categories need at least 30 training, 10 validation and 10 test events. The same feature set and regularized logistic regression (C=1) apply to all categories/horizons. No hyperparameter search.

Median imputation, missingness indicators and standardization fit **only on training data**, leaving all saved feature NULLs unchanged. Arm angle, within-game fatigue with inadequate coverage, old UCL history without publication verification, outcome dates and future metadata are excluded. The new prior-IL count uses only placements reported before the outing. Repeated pitchers can appear in later time partitions, consistent with forecasting existing players; this does not measure unseen-pitcher generalization.'''),
('code','''from src.injury_analysis.prediction import run_predictions, sensitivity_metrics
a, events, spec = load()
metrics = run_predictions(a, events, spec)
display(read_table('prediction_cohort_counts').query('horizon == 30'))
test = metrics.query("partition == 'test'")
display(test[['category','horizon','events','rows','positives','prevalence','average_precision','pr_lift','ap_low','ap_high']])'''),
('markdown','''## Ranking performance and uncertainty
Average precision is the reported PR-AUC summary. PR lift divides AP by the observed category prevalence; it is not prevalence-invariant proof of superior predictability. Intervals resample entire pitchers, preserving their repeated outings. ROC-AUC remains in the CSV as a secondary metric. There is no accuracy-based ranking.'''),
('code','''plots.prediction_figure(); plt.show()
sensitivity = sensitivity_metrics()
display(sensitivity[sensitivity.category.isin(['Elbow / UCL','Shoulder','Lower body'])])
m = test.query('horizon == 30').set_index('category')
conclusion(f"30-day test PR lift ranges from {m.pr_lift.min():.2f} to {m.pr_lift.max():.2f}. Shoulder lift is {m.loc['Shoulder','pr_lift']:.2f}, elbow/UCL {m.loc['Elbow / UCL','pr_lift']:.2f}, and lower body {m.loc['Lower body','pr_lift']:.2f}. These modest, uncertain comparisons do not establish an arm-specific predictive signature; small categories and differing prevalence limit rankings.")''')])

write('04_severity_and_censoring.ipynb','04 — Severity proxies and incomplete outcomes',
'Examine observed time out, long IL designations and documented surgery without treating them as medical recovery.',
'All target-eligible episodes remain in denominators, including unresolved cases. Usable durations require a validated activation, observed duration, and non-offseason activation. For trajectories, only events with sufficient preceding appearance history contribute. Outcome strata are descriptive labels only; none enters a predictor.',[
('code','''from src.injury_analysis.outcomes import severity_analysis
a, events, spec = load()
counts, summaries = severity_analysis(events)
display(counts)
display(read_table('severity_duration_groups'))
plots.severity_figure(); plt.show()'''),
('markdown','''## Compare within anatomical categories
Duration groups are explicit observed-absence bins (≤30, 31–90, >90 days), not invented clinical severity. Unresolved/administrative returns form a separate group; no return is imputed and no Kaplan–Meier interpretation is asserted under informative missing activation. Surgery documentation may occur after injury and does not imply every undocumented case had no surgery. CIs use pitcher clusters and one pre-injury mean per event.'''),
('code','''display(summaries[(summaries.comparison=='duration_group') & summaries.category.isin(['Elbow / UCL','Shoulder','Lower body']) & summaries.feature.eq('velo_z')])
display(summaries[(summaries.comparison=='surgery_documentation') & summaries.category.isin(['Elbow / UCL','Shoulder']) & summaries.feature.eq('velo_z')])
usable=int(counts.usable_durations.sum()); n=int(counts.events.sum())
conclusion(f"Only {usable:,}/{n:,} target-eligible episodes have usable activation-based durations for these comparisons. {n-usable:,} remain unresolved or have an administrative return. Observed-duration feature differences are therefore conditional descriptions, not evidence that longer biological recovery is more predictable; missing follow-up and category composition must remain visible.")''')])

write('05_recurrence_and_transitions.ipynb','05 — Repeated injuries and anatomical transitions',
'Quantify observed repeat episodes, intervals, same-region recurrence, and pre-injury differences.',
'Recurrence is among qualifying recorded episodes starting in 2015. A transition requires observed MLB pitching between the two dates; otherwise it is separately flagged and excluded from the transition matrix. First means first observed qualifying episode, not lifetime first injury. Unequal career observation and missing non-IL injuries prevent a causal recurrence-risk interpretation.',[
('code','''from src.injury_analysis.outcomes import recurrence_analysis
a, events, spec = load()
rec, repeats, transitions, crude = recurrence_analysis(a, events)
display(rec.repeat_status.value_counts().rename('episodes'))
display(rec.groupby('player_id').size().value_counts().sort_index().rename('pitchers_with_n_episodes'))
display(read_table('recurrence_summary'))
plots.recurrence_figure(transitions); plt.show()'''),
('markdown','''## Prior history and warning patterns
The crude appearance rates below use **as-of prior reported IL counts**, never future recurrence rank. Role strata reduce a visible source of confounding, but age, exposure duration and health selection still differ. The first/repeat feature table is descriptive, one mean per event, with pitcher-bootstrap intervals. Outcome-derived episode rank is used only for these comparisons.'''),
('code','''display(crude)
patterns=read_table('recurrence_preinjury_features')
display(patterns[patterns.category.isin(['Elbow / UCL','Shoulder','Lower body']) & patterns.feature.eq('velo_z')])
conclusion(f"{len(repeats):,} repeat qualifying episodes have observed intervening MLB pitching. {repeats.same_detailed_region.mean():.1%} involve the same detailed region; the median interval is {repeats.interval_days.median():.0f} days. These transitions establish repeated documented burden, but left truncation, changing observation time and episode-boundary uncertainty prevent interpreting them as clinical recurrence probabilities.")''')])

write('06_warning_timing_and_alerts.ipynb','06 — Warning time, event recall and alert burden',
'Determine how much advance warning the fixed models provide at constrained alert budgets.',
'Use frozen 2024–2025 predictions from notebook 03. Require a 7-day minimum warning and the model’s 14/30/60-day maximum horizon. Denominators include every eligible test-period injury event, even when no eligible warning outing exists; observable-event recall is separately shown. Multiple alerts are counted separately with no cooldown. This is the selected category-versus-IL-free cohort, not an operational full-league alert simulation.',[
('code','''from src.injury_analysis.outcomes import warning_timing
a, events, spec = load()
risk = warning_timing(events)
budgets = read_table('alert_budget_metrics')
display(budgets.query('alert_rate == 0.05')[['category','horizon','qualifying_events','observable_events','alerted_events','injury_event_recall','observable_event_recall','precision','recall','median_warning_lead_days','false_alerts_per_pitcher_season']])
plots.warning_figure(); plt.show()
plots.risk_trajectory_figure(); plt.show()'''),
('markdown','''## Retrospective budgets versus validation-fixed thresholds
Top-5% test ranking fixes the retrospective test alert budget; it is not a deployable threshold. A separate validation-score quantile fixes the threshold before test evaluation, and its realized test alert rate may drift. Warning lead time below is the earliest qualifying alert among **detected** events, not all events; missed events remain in recall denominators.'''),
('code','''fixed = read_table('validation_fixed_thresholds')
display(fixed.query('horizon == 30 and validation_alert_budget == 0.05'))
leads = read_table('warning_lead_times')
display(leads.query('horizon == 30 and alert_rate == 0.05').groupby('category').warning_lead_days.agg(['count','median','min','max']))
b=budgets.query('horizon == 30 and alert_rate == 0.05')
conclusion(f"At a retrospective 5% budget, 7–30-day warning detects {b.injury_event_recall.min():.1%}–{b.injury_event_recall.max():.1%} of all qualifying category events. False alerts range from {b.false_alerts_per_pitcher_season.min():.2f} to {b.false_alerts_per_pitcher_season.max():.2f} per represented pitcher-season. Lead-time medians describe detected events only; substantial missed-event burden remains.")''')])

write('07_feature_families_and_interactions.ipynb','07 — Feature families, modern measurements and interactions',
'Assess whether removing a feature family changes validation performance and whether two prespecified workload interactions help.',
'30-day elbow/UCL, shoulder, forearm and lower-body comparisons. Every variant uses the same chronological training/validation rows and train-only preprocessing. Family ablations and interactions are **validation-only**; the test set is not used to choose a winner. Correlated families make coefficient magnitudes and removal effects conditional, not causal.',[
('code','''from src.injury_analysis.prediction import run_family_analysis
a, events, spec = load()
family = run_family_analysis(a, spec)
display(spec[spec.selected_for_prediction].groupby('family').agg(predictors=('name','size')))
display(family[['category','variant','predictors','events','average_precision','ap_change_vs_all']])
plots.family_figure(); plt.show()'''),
('markdown','''## Interpretation and the modern-feature era
Interactions are velocity z-score × recent workload acceleration, and release-variability z-score × workload acceleration. They are known after the outing and cannot include future injury or recovery information. An ablation can improve AP when a correlated/noisy family is removed; this is not a definitive estimate of biological relevance.

Arm-angle features are excluded from this long-history model. Their matched event effects below use only pairs with nonmissing measurements in the same season. No historical zeros are substituted. Within-game velocity decline requires disjoint 20-fastball segments and is sparse among relievers. Existing full feature tables preserve both for later focused studies.'''),
('code','''availability=read_table('feature_availability_by_season')
display(availability[availability.feature.isin(['arm_angle_avg','arm_angle_z','velo_drop_game'])].pivot(index='season',columns='feature',values='available_fraction'))
effects=read_table('paired_preinjury_effects')
display(effects[effects.feature.eq('arm_angle_z')])
coef=read_table('model_coefficients')
display(coef[coef.category.eq('Elbow / UCL')].assign(magnitude=lambda d:d.standardized_log_odds.abs()).nlargest(12,'magnitude').drop(columns='magnitude'))
inter=family[family.variant.eq('With two interactions')]
conclusion(f"Removing feature families or adding the two interactions changes validation AP by {family.ap_change_vs_all.min():+.4f} to {family.ap_change_vs_all.max():+.4f}. The interaction changes range from {inter.ap_change_vs_all.min():+.4f} to {inter.ap_change_vs_all.max():+.4f}. These single-split comparisons identify follow-up hypotheses, not a final selected feature set; arm-angle evidence uses a smaller modern-era sample.")''')])

write('08_capstone_summary.ipynb','08 — Computed findings and capstone figures',
'Assemble the actual results, their practical implications, and unresolved validity risks.',
'Uses the outputs of notebooks 01–07 without additional fitting or test-set selection. Tables and PNG/PDF figures are saved under reports/injury_type_analysis. The original database and canonical injury files remain unchanged.',[
('code','''from src.injury_analysis.report import write_summary
a, events, spec = load()
summary = write_summary(a, events, spec)
display(Markdown(summary))'''),
('markdown','''## Selected presentation figures
These existing figures connect the central questions: group differences before injury, comparative held-out ranking, and event detection at limited alert rates. Reuse the saved PDFs for presentations rather than screenshots.'''),
('code','''from IPython.display import Image
for name in ['02_matched_effects','03_predictability','06_warning_budgets']:
    display(Image(filename=str(OUT/'figures'/f'{name}.png'),width=1000))
display(read_table('analysis_validation_checks'))
print('Full computed narrative:', OUT/'SUMMARY.md')
print('Tables:', OUT/'tables')
print('PNG/PDF figures:', OUT/'figures')
conclusion(f"The workflow analyzed {events.target_eligible.sum():,} target-eligible documented episodes with {spec.selected_for_prediction.sum()} shared historical predictors. The computed summary above reports the observed effects, uncertain category differences, warning burden and recurrence patterns. The next step is targeted date/anatomy adjudication and a prespecified replication; these results do not yet justify a final injury-specific warning system.")''')])
print(f'Wrote 8 notebooks to {DEST}')
