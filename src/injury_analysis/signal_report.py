"""Join computed decomposition, trajectory timing and sensitivity results."""
import json
import pandas as pd
from .signals_common import CFG,OUT,read_table,table,save_manifest


def write_synthesis():
    perf=read_table('family_performance')
    def part(name,variant):return perf.loc[perf.partition.eq(name)&perf.variant.eq(variant)].set_index('category')
    variants=read_table('family_definitions')
    family_names=['workload','pitcher_history','velocity','release_mechanics','extension','spin_movement','command','pitch_mix']
    singles=perf.loc[perf.partition.eq('validation')&perf.variant.isin(family_names)].sort_values('average_precision',ascending=False)
    best=singles.groupby('category').head(2).copy()
    table(best[['category','variant','average_precision','pr_lift','ap_low','ap_high']],'validation_family_inspection_candidates')
    v=part('validation','Full');t=part('test','Full');reference=part('test','Workload + history')
    shapes=read_table('trajectory_shape_metrics')
    step=shapes.loc[shapes.feature.eq('velo_z')&shapes.metric.eq('final_step_vs_previous_two')].set_index('category')
    bridge=t[['average_precision','ap_delta','delta_low','delta_high']].rename(columns={'average_precision':'full_test_ap'}).copy()
    bridge['workload_history_test_ap']=reference.average_precision
    bridge['validation_full_ap_delta']=v.ap_delta
    for field in ['mean','low','high','events']:bridge['velocity_final_step_'+field]=step[field]
    table(bridge.reset_index(),'prediction_trajectory_bridge')
    sensitivity=read_table('trajectory_sensitivity').query("feature == 'velo_z'")
    late=sensitivity.loc[sensitivity.condition.eq('Exclude final 7 days')].set_index('category')
    arm=read_table('arm_nonarm_direct_contrasts').query("feature == 'velo_z'").iloc[0]
    modern=read_table('modern_family_performance').query("partition == 'test' and variant == 'Full + arm angle'")
    timing=read_table('trajectory_supported_timing')
    velocity_timing=timing.loc[timing.feature.eq('velo_z')]
    bets=read_table('family_alert_budgets').query("partition == 'test' and alert_rate == 0.05 and variant in ['Full','Workload + history']")
    coeff=read_table('variable_permutation_summary')
    topvars=coeff.sort_values('mean',ascending=False).groupby('category').head(4)
    msg=f'''# Feature-family decomposition and pre-injury trajectory shape

## Scope and design

This extends the existing analysis without changing the database, original cohort, labels, historical features, or 1,225 matched pairs. It compares {len(variants)} family/combination specifications across seven injury categories at the 30-day horizon. The original train-through-2021 / validation-2022–2023 / test-2024–2025 boundaries and 60-day embargo are retained. The 2024–2025 period was already examined: these are **exploratory temporal comparisons**, not a new untouched confirmatory test.

All models use the same fixed regularized logistic framework and category-specific rows, with training-only imputation/scaling. A separate common-row modern study adds arm angle using training from 2020–2021. Small modern categories are explicitly skipped. Current-outing features are known after the outing. Labels describe recorded injury/IL-date proxies, not verified biological onset.

## 1. Which information carries signal?

The following are the top two *individual-family validation point estimates* for inspection, not statistically established winners:

{best[['category','variant','average_precision','pr_lift']].to_markdown(index=False,floatfmt='.4f')}

The full model's incremental AP over workload + history is:

{bridge[['workload_history_test_ap','full_test_ap','ap_delta','delta_low','delta_high','validation_full_ap_delta']].to_markdown(floatfmt='.4f')}

Full-model AP is higher by point estimate in {int(v.ap_delta.gt(0).sum())}/7 validation categories and {int(t.ap_delta.gt(0).sum())}/7 test categories. Paired pitcher-bootstrap intervals and the different results across periods are more informative than choosing the largest score. The observed intervals condition on the fitted models; they do not include refitting uncertainty or multiple-comparison adjustment.

Workload/history already explains much of the ranking information. For elbow/UCL the full-minus-baseline test AP difference is {t.loc['Elbow / UCL','ap_delta']:+.4f}; for shoulder it is {t.loc['Shoulder','ap_delta']:+.4f}; for lower body {t.loc['Lower body','ap_delta']:+.4f}. Forearm has a test difference of {t.loc['Forearm','ap_delta']:+.4f} but a validation difference of {v.loc['Forearm','ap_delta']:+.4f}, which limits a stable injury-specific interpretation.

Selected variable inspection within the strongest validation families:

{topvars[['category','family','feature','mean']].to_markdown(index=False,floatfmt='.4f')}

Permutation occurs within validation season/role strata, with five shuffles. These are conditional AP losses within the full model, not causal effects. Correlation can distribute importance across variables. The displayed shuffle variability is not a sampling confidence interval. The role indicator is held fixed by the shuffle design and is therefore assessed only through its coefficient/family models.

## 2. General versus throwing-arm-specific signals

Workload/context signal appears in both arm and non-arm comparisons. The incremental modeling results do not establish consistent arm-specific added value from velocity/mechanics. Models compare each injury category against an IL-free pool; they do not directly classify arm versus non-arm injury.

The direct matched 1–14-day velocity contrast is stronger in explicitly throwing-side cases: arm minus non-arm difference {arm.arm_minus_nonarm:.3f} z-units (exploratory 95% interval {arm.low:.3f} to {arm.high:.3f}; {int(arm.arm_events)} arm and {int(arm.nonarm_events)} non-arm pairs). Both groups have negative matched velocity differences. This pooled descriptive contrast is not a validated injury-specific prediction rule and is not adjusted for the many feature comparisons.

Modern arm-angle incremental results on identical rows:

{modern[['category','average_precision','ap_delta','delta_low','delta_high']].to_markdown(index=False,floatfmt='.4f')}

These compare with a full model *without* arm angle on the same shorter-era cohort, not with the long-history full model. The short 2020–2021 training period and missing-angle selection limit inference.

## 3. What do the trajectories look like?

Balanced analyses keep the same matched events at every calendar bin or every one of the final five appearances, separately for each feature. Available-pair curves and attrition tables remain available. No smoothing or interpolation hides support.

Velocity's excess final step (last outing minus previous two, case change minus control change) is:

{step[['mean','low','high','events','q10','q90']].to_markdown(floatfmt='.3f')}

Elbow/UCL has an excess final step of {step.loc['Elbow / UCL','mean']:.3f} z-units; shoulder {step.loc['Shoulder','mean']:.3f}; lower body {step.loc['Lower body','mean']:.3f}. These contrasts, slopes and trajectory plots distinguish late changes from sustained early divergence. Event-specific changes span wide distributions: a group mean is not an individual warning threshold.

Pitcher baseline speed, season workload and recent role can differ well before the event, helping models rank persistent profiles without a strong mean approach-to-injury trajectory. The heterogeneity figure displays this distinction. The existing logistic baseline cannot establish nonlinear mechanisms; no biological mechanism is inferred from coefficient or curve shape.

## 4. How early is there evidence of change?

Timing uses simultaneous bootstrap bands across points **within a feature/category** and at least 30 balanced events. We separately report isolated supported points and two adjacent, same-direction supported points. Bands are not adjusted across all features/categories. The velocity timing results are:

{velocity_timing[['mode','category','balanced_events','individually_supported_points','first_sustained_point']].to_markdown(index=False)}

Calendar point 0=31–60d, 1=15–30d, 2=8–14d, 3=1–7d. Appearance point 0=5th-last and 4=last. An isolated earlier point is not an estimated onset. Support already at point 0 means the change may predate the observed window. Irregular, sparse histories do not justify a forced breakpoint model.

The final-appearance signal cannot be equated with a fixed day lead; actual lead-day distributions are in `velocity_appearance_calendar_support.csv`. The balanced samples are smaller and selected for sustained recent pitching, so compare them with the available-pair and sensitivity results rather than declaring a precise universal warning time.

## 5. What survives the sensitivity checks?

Excluding the final seven days leaves these velocity contrasts over days 8–14:

{late[['mean','low','high','events']].to_markdown(floatfmt='.3f')}

The shoulder estimate is {late.loc['Shoulder','mean']:.3f} (interval {late.loc['Shoulder','low']:.3f} to {late.loc['Shoulder','high']:.3f}); elbow/UCL is {late.loc['Elbow / UCL','mean']:.3f} (interval {late.loc['Elbow / UCL','low']:.3f} to {late.loc['Elbow / UCL','high']:.3f}). Same-pair contrasts distinguish removal of late observations from changes in which events remain. Strict QA, starter/reliever, first/prior reported injury, stronger measured baseline, throwing-side evidence, and period-specific results are tabulated. First means first observed report, not first lifetime injury.

The predictive refits remove final-six-day positives rather than relabeling them negative. They compare the full and simpler models on the same restricted rows. Strict-QA results have a changed prevalence. Frozen role-specific evaluations assess ranking differences only and do not assign unobserved event roles.

## 6. Practical warning and remaining uncertainty

At the retrospective 5% budget, all-event warning recall and random expectations are:

{bets[['category','variant','qualifying_events','injury_event_recall','random_expected_event_recall','median_warning_lead_days','non_actionable_alerts_per_pitcher_season']].to_markdown(index=False,floatfmt='.3f')}

Event recall requires at least seven days of warning. All qualifying events remain in the denominator, including those with no observable warning outing; observable-event recall is separately saved. Uniform-random expected recall accounts for multiple opportunities per event. Bootstrap event differences condition on the fixed observed alert selections and do not include model training or random-policy realization uncertainty. Lead times are conditional on detection; non-actionable alerts include warnings too late for the seven-day objective.

Major limits remain historical text revisions and unverified publication times; IL dates as outcome proxies; incomplete follow-up and selection of active pitchers; shared controls/case identity across some pairs; multiple exploratory comparisons; small forearm/other-arm samples; and differing calendar versus appearance sampling. Feature importance does not establish causality. Apparent stable profiles and late changes should not be presented as a clinically usable early-warning system.

## Artifacts and next step

Three executed notebooks are in `notebooks/injury_signal_analysis/`. Tables and PNG/PDF figures are in `reports/injury_signal_analysis/`; separate analysis Parquet is in `data/injury/analysis/signal_analysis/`. Original notebooks and central DuckDB are preserved.

The next substantive step is prospective-style date adjudication and a prespecified replication of the simplest context/workload baseline versus a small set of additional velocity features, with an actionable lead-time target. Use a new evaluation period for confirmation after these exploratory comparisons.
'''
    (OUT/'SUMMARY.md').write_text(msg)
    save_manifest(dict(computed_summary='SUMMARY.md'))
    return msg
