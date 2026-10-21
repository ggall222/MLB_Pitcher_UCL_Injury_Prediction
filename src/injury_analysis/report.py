"""Write the capstone summary from computed tables, never prewritten numerical claims."""
import json
import pandas as pd
from .common import OUT, CACHE, CFG, read_table, source_signature


def write_summary(a, events, spec):
    m=read_table('prediction_metrics').query("partition == 'test' and horizon == 30")
    alert=read_table('alert_budget_metrics').query('horizon == 30 and alert_rate == 0.05')
    effects=read_table('paired_preinjury_effects').query("feature == 'velo_z'")
    controls=read_table('matched_controls')
    rec=read_table('recurrence_events')
    repeat=rec[rec.repeat_status.eq('Repeat after observed MLB pitching')]
    severity=read_table('severity_counts')
    ablation=read_table('feature_family_ablation')
    predictors=spec.loc[spec.selected_for_prediction,'name'].tolist()
    metrics=m[['category','events','prevalence','average_precision','pr_lift','lift_low','lift_high']].copy()
    for c in ['prevalence','average_precision']:
        metrics[c]=(100*metrics[c]).map(lambda x:f'{x:.2f}%')
    text=f'''# Exploratory injury-type analysis — computed results

The existing source database was opened read-only. This is an exploratory comparison of documented MLB IL episodes, not a clinical prediction system or a final tuned model.

## Cohort and source coverage

- {len(events):,} canonical episodes; {int(events.target_eligible.sum()):,} pass the prespecified initial-anatomy/timing rules.
- {len(a):,} regular-season appearances by official pitcher/two-way IDs; {int(a.analysis_eligible.sum()):,} pass appearance/history/active-episode rules.
- Statcast features end {a.game_date.max().date()}; injury observations end {events.observation_end_date.max().date()}. 2026 is descriptive only.
- {len(controls):,} event/control pairs, matched on year, month, current role, five-year age band and prior-appearance experience band. Each control pitcher is used at most once. Unmatched cases remain available for outcome descriptions.
- {len(predictors)} prespecified v1/context predictors survive training-period availability checks. Arm angle is descriptive in its observed era; within-game velocity decline is too sparse in this mixed-role population for the shared model.

## What the results show

Matched 1–14-day velocity differences (injury minus control, pitcher-relative z-score; exploratory pitcher bootstrap):

{effects[['analysis_group','mean','low','high','n']].to_markdown(index=False,floatfmt='.3f')}

30-day chronological test results (training through 2021, validation 2022–2023, test 2024–2025; 60-day boundary embargo):

{metrics.to_markdown(index=False,floatfmt='.2f')}

PR lift ranges from {m.pr_lift.min():.2f} to {m.pr_lift.max():.2f}. Lower-body lift is {m.loc[m.category.eq('Lower body'),'pr_lift'].iloc[0]:.2f}, so these results do not establish that the warning signals are specific to arm injuries. Category prevalences and event counts differ; overlapping uncertainty intervals do not support a reliable ranking of biological predictability.

At a retrospective 5% alert budget, requiring a warning 7–30 days before injury:

{alert[['category','qualifying_events','observable_events','alerted_events','injury_event_recall','observable_event_recall','median_warning_lead_days','false_alerts_per_pitcher_season']].to_markdown(index=False,floatfmt='.3f')}

The median lead time is conditional on detection. Events without an eligible warning outing remain in the all-event denominator. Alert budgets are retrospective ranking summaries; deployable-style thresholds fixed from validation are separately reported in `tables/validation_fixed_thresholds.csv`.

Among {len(repeat):,} repeat episodes with intervening observed MLB pitching, {repeat.same_detailed_region.mean():.1%} return to the same detailed region. Median interval is {repeat.interval_days.median():.0f} days. These are first/repeat *observed qualifying episodes*, not lifetime first injuries or medically adjudicated recurrences.

Severity descriptions have {int(severity.usable_durations.sum()):,} usable activation-based durations among {int(severity.events.sum()):,} eligible events. Unresolved/administrative returns are retained as separate outcomes. Surgery documentation is evidence of surgery; lack of documentation is not proof that surgery did not occur.

## Interpretation and next work

Feature-family ablations are validation-only. Their AP changes range from {ablation.ap_change_vs_all.min():+.4f} to {ablation.ap_change_vs_all.max():+.4f}; correlated families and this one time split limit attribution. Standardized model coefficients are conditional associations, not causal feature importance. Two prespecified velocity/workload and release-variability/workload interactions are exploratory.

Before a final modeling study, adjudicate a sample of onset dates and revised source descriptions, lock the target definition, and repeat comparisons with documented throwing-side anatomy, alternative eras, and a new held-out period. Do not choose the final feature set or injury category by these test scores and then reuse the same test set as untouched evidence.

## Material limitations

- IL/effective dates are observable roster proxies, not established biological onset. No historical publication timestamp is verified; retrospective text revisions remain a source of outcome misclassification.
- Initial anatomy defines labels. Later UCL diagnoses/surgery do not silently turn forearm complaints into early known UCL events. Arm-region categories include unknown laterality; `throwing_arm_injury` identifies explicitly supported sides.
- Timing/overlap/postoperative/ambiguous anatomy exclusions reduce generalizability. Return-date QA alone does not exclude a primary event, avoiding selection only for rapid documented recovery. A strict-QA sensitivity is provided.
- Category models compare one injury category to a shared IL-free follow-up pool. Other impending injuries are excluded. Thus test prevalence, alert rates and false-alert burden apply to this selected comparison cohort, not the entire MLB population.
- Negatives require an observed MLB outing at/after the horizon within 30 days. This reduces end-of-career/season uncertainty but selects active pitchers. IL-free does not mean symptom-free.
- Raw NULLs remain in source and analytical files. Median imputation, missingness indicators and scaling are fitted on training data only inside exploratory models. Coverage changes across eras may still influence results.
- Confidence intervals resample pitchers; case-control intervals cluster by case pitcher with unique control pitchers. Some individuals can also appear as cases elsewhere, so residual network dependence and matching selection remain limitations. Multiple exploratory comparisons are not multiplicity-adjusted significance tests.
- Severity and recurrence follow-up are incomplete and potentially informative; observed time out is not biological recovery. No censored return date is imputed. The study is left-truncated in 2015.
- A completed appearance is the prediction timestamp. Current-outing features cannot support a pregame prediction.

## Artifacts

Eight executed notebooks: `notebooks/injury_type_analysis/01_*.ipynb` through `08_*.ipynb`.
Tables: `reports/injury_type_analysis/tables/`. Figures: `reports/injury_type_analysis/figures/` (PNG and PDF).
Analysis-only Parquet: `data/injury/analysis/`. Existing UCL pipeline and DuckDB are preserved.
'''
    (OUT/'SUMMARY.md').write_text(text)
    original=json.loads((CACHE/'manifest.json').read_text())['source_signature']
    assert source_signature()==original, 'Source database or injury snapshot changed during analysis'
    return text
