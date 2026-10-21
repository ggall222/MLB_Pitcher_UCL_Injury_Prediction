"""Generate a findings narrative from the computed comparisons."""
from .profile_departure import OUT,CFG,read_table,table,save_manifest


def write_synthesis():
    m=read_table('model_performance')
    d=read_table('incremental_comparisons')
    b=read_table('alert_budget_performance')
    b=b.loc[b.alert_rate.eq(.05)]
    main=d.loc[d.variant.eq('Profile + Departure')&d.reference.eq('Profile only')]
    adjusted=d.loc[d.variant.eq('Context + Profile + Departure')&d.reference.eq('Context + Profile')]
    test=m.loc[m.partition.eq('test')].set_index(['category','variant'])
    rows=[]
    for cat in m.category.unique():
        def val(name,col):return test.loc[(cat,name),col]
        event=b.loc[b.partition.eq('test')&b.category.eq(cat)].set_index('variant')
        rows.append(dict(category=cat,profile_ap=val('Profile only','average_precision'),
            departure_ap=val('Departure only','average_precision'),combined_ap=val('Profile + Departure','average_precision'),
            original_ap=val('Original 62','average_precision'),
            profile_event_recall=event.loc['Profile only','injury_event_recall'],
            combined_event_recall=event.loc['Profile + Departure','injury_event_recall'],
            random_expected_event_recall=event.loc['Profile only','random_expected_event_recall']))
    import pandas as pd
    summary=table(pd.DataFrame(rows),'research_question_summary')
    lines=['# Profile information versus short-term departure: findings',
        '',
        'The fitted models use both types of information, but their incremental value differs by injury group and metric. '
        'Persistent profiles account for much of the Average Precision in several categories. Departures change which outings '
        'are alerted and offer the clearest additional evidence for shoulder injuries. This supports a mixture of profile association '
        'and short-term signal, not a demonstrated general injury-imminence detector.',
        '',
        '## Direct comparison',
        '',
        'All models use the original category-specific 30-day risk sets: train through 2021, validation 2022–2023, '
        'test 2024–2025, with the existing 60-day embargo. Both starters and relievers are included. '
        'The source cohort has 233,433 appearances; the supervised risk set is smaller and differs by injury category. '
        'No test outcomes choose feature blocks or model hyperparameters. These periods have already been examined in earlier analyses.',
        '',
        '| Injury group | Profile AP | Departure AP | Combined AP | Original 62 AP | Profile → combined event recall |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.category} | {r.profile_ap:.4f} | {r.departure_ap:.4f} | {r.combined_ap:.4f} | {r.original_ap:.4f} | {r.profile_event_recall:.1%} → {r.combined_event_recall:.1%} |')
    primary_test=main.loc[main.partition.eq('test')]
    lines+=['',
        'Event recall above requires an alert **7–30 days** before an event, at a budget of **5% of eligible outings**. '
        'Its denominator contains all qualifying events, including events with no usable warning opportunity.',
        '',
        f'Combined AP exceeds profile-only AP by point estimate for {int(primary_test.ap_difference.gt(0).sum())}/7 test categories, '
        f'but **{int(primary_test.ap_low.gt(0).sum())}/7 paired 95% intervals exclude zero on the positive side**. '
        'Shoulder is the only category with a positive primary AP increment in both validation and test. '
        'Forearm and non-arm categories do not show a consistent, well-supported AP gain from departures.',
        '',
        '## Event warnings and the workload adjustment',
        '']
    for cat in ['Elbow / UCL','Shoulder']:
        h=main.loc[main.category.eq(cat)].set_index('partition')
        t=h.loc['test'];v=h.loc['validation']
        x=adjusted.loc[adjusted.category.eq(cat)].set_index('partition')
        lines.append(f'- **{cat}:** primary event-recall change is {100*v.event_mean:+.1f} percentage points in validation '
            f'and {100*t.event_mean:+.1f} in test (test interval {100*t.event_low:+.1f} to {100*t.event_high:+.1f}). '
            f'With workload, role and calendar held common, it becomes {100*x.loc["validation","event_mean"]:+.1f} and '
            f'{100*x.loc["test","event_mean"]:+.1f} points; adjusted AP changes are '
            f'{x.loc["validation","ap_difference"]:+.4f} and {x.loc["test","ap_difference"]:+.4f}.')
    shoulder=adjusted.loc[adjusted.category.eq('Shoulder')&adjusted.partition.eq('test')].iloc[0]
    concentration=read_table('alert_concentration').query("partition=='test' and category=='Shoulder'").set_index('variant')
    lines+=['',
        f'The adjusted shoulder test AP difference is {shoulder.ap_difference:+.4f} '
        f'(95% interval {shoulder.ap_low:+.4f} to {shoulder.ap_high:+.4f}); the validation interval includes zero. '
        'Elbow/UCL incremental gains are largely accounted for by shared workload/context. Adding eight strictly prior slopes '
        'to profile-plus-departure provides no clear additional AP benefit.',
        '',
        f'Profile scores repeatedly alert the same pitchers. For shoulder in test, the profile model alerts '
        f'{int(concentration.loc["Profile only","alerted_pitchers"])} distinct pitchers, versus '
        f'{int(concentration.loc["Profile + Departure","alerted_pitchers"])} for combined, with the same 1,390 alerted outings. '
        'This helps explain why event recall can improve without a clear AP gain. It does not establish injury-specific timing: '
        'the unadjusted combined elbow/UCL and shoulder event-recall point estimates remain below the uniform-random alert '
        'expectation in both periods. The full budget curves retain that benchmark.',
        '',
        '## Persistent scores versus approaching-event changes',
        '',
        'An exact additive decomposition reconstructs the combined linear predictor as intercept + profile term + departure term, '
        'including source-specific missingness indicators. This is a decomposition of log-odds, not a fraction of causal risk or '
        'independent predictive importance. In test, 86–98% of profile-term variance lies between pitchers, compared with '
        '4–9% for departure terms. Predictor blocks are correlated and are not biologically orthogonal.',
        '']
    shapes=read_table('score_shape_contrasts')
    sh=shapes.loc[shapes.category.eq('Shoulder')&shapes.feature.eq('departure_score')&
        shapes['mode'].eq('appearance')&shapes.metric.eq('last_minus_first_separation')].iloc[0]
    lower=shapes.loc[shapes.category.eq('Lower body')&shapes.feature.eq('profile_score')&
        shapes['mode'].eq('appearance')&shapes.metric.eq('average_separation')].iloc[0]
    lines+=[
        f'Frozen combined-model scores are applied to the original **443 out-of-training matched pairs**, without rematching. '
        f'Common complete measurement support leaves **{int(sh.events)} shoulder pairs** with all five outings. '
        f'The shoulder departure case-minus-control score increases {sh["mean"]:.3f} log-odds from the fifth-last to last outing '
        f'(pitcher-bootstrap interval {sh.low:.3f} to {sh.high:.3f}). '
        'Velocity departure has the strongest negative non-missingness standardized coefficient in the shoulder combined model: '
        'a lower-than-baseline velocity increases its score, conditional on the other predictors. Correlated coefficients are not causal effects.',
        '',
        'The smaller balanced calendar cohort contains 38 shoulder pairs. Only the final **1–7-day** bin has a simultaneous '
        'difference band above zero; 8–14 days remains uncertain. The curves suggest late divergence, but do not establish a reliable '
        'earlier warning threshold. Appearance positions cannot be translated into fixed calendar lead time. Elbow/UCL, forearm and '
        'lower-body departure trajectories do not establish a comparable changing score pattern.',
        '',
        f'Lower-body profile scores show a persistent matched offset averaging {lower["mean"]:.3f} log-odds '
        f'({lower.low:.3f} to {lower.high:.3f}; {int(lower.events)} pairs). '
        'This is compatible with stable profile/history associations; it does not prove underlying susceptibility. '
        'The means used for “profile” are rolling summaries of at most 20 prior outings and can adapt to emerging injury.',
        '',
        '## Robustness, probability quality and stability',
        '',
        '- The primary arm event-recall improvements survive complete measurement rows, equal availability indicators, removal of '
        'final-six-day positive rows, and removal of the five history/context fields. However, **after workload adjustment and '
        'excluding final-six-day positives**, shoulder AP gains shrink to about +0.0006 in validation and +0.0005 in test, '
        'with intervals spanning zero. Its adjusted test event-recall increment is about +1.9 points, also uncertain. '
        'Thus the strongest adjusted discrimination result depends materially on the near-event window.',
        '- Probability error changes are small. Shoulder test Brier loss improves by about 0.000091 in the primary comparison, '
        'but the corresponding validation interval includes zero. Reliability plots and mean bias do not support broad calibration '
        'improvement. For example, the combined lower-body test model predicts 4.32% mean risk against 2.30% observed. '
        'These are conditional risks in the selected analysis cohort, not population injury probabilities.',
        '- Adding departures improves test rank/alert stability in elbow/UCL and modestly in shoulder, but slightly reduces it in '
        'forearm and lower body across 12 paired training-pitcher bootstrap refits. All transforms are refitted on each training resample. '
        'Twelve perturbations are diagnostics, not 12 independent generalization tests.',
        '- There is no consistent evidence of **earlier** warnings for the same events. Jointly detected test shoulder events have a '
        'mean lead change of −1.8 days, with an interval spanning zero. Comparing conditional median lead across different detected '
        'events can be misleading; the paired common-event comparison is saved separately.',
        '- Validation-derived thresholds produce different test alert rates; retrospective top-5% performance is not a deployable '
        'fixed-threshold promise.',
        '',
        '## Interpretation and next experiment',
        '',
        'The answer is **some combination of both**, with strong dependence on injury category. Profile/history information explains '
        'most of the observed AP for lower-body and several other categories. Shoulder provides the best evidence that departures '
        'add information beyond profile and workload, although much of its discrimination gain is very close to the recorded event. '
        'Elbow/UCL benefits in event coverage relative to a static-profile warning policy, but has weak evidence of additional information '
        'once workload/context is included. These results do not establish that the model detects biological onset or causal susceptibility.',
        '',
        'A defensible next step is to freeze a shoulder-focused, explicitly ≥7-day-warning comparison before evaluating a new chronological '
        'period. Retain profile/workload and uniform-random alert benchmarks, inspect near-event reporting dates, and calibrate only within '
        'training/validation data if calibration is added. The 2024–2025 period here is exploratory, not a fresh confirmatory holdout.',
        '',
        '## Scope and limitations',
        '',
        'P is 8 prior measurement means + age, handedness, observed starts, days since debut and prior reported IL count. '
        'D is the corresponding 8 current pitcher z-scores. C is 15 workload fields + current role + month, held common rather than '
        'forced into either conceptual block. Missing flags are extra learned design columns; all raw NULLs remain untouched. '
        'Arm angle is omitted because long-history coverage is unavailable. Current-outing departures are available only after the outing. '
        'Pitcher z-scores include the pitcher-specific denominator and do not mathematically remove every stable characteristic.',
        '',
        'All models retain the existing median/scaling/L2-logistic C=1 framework, with train-only preprocessing and tight numerical '
        'convergence. This tests these representations in the existing linear model; it is not a universal test that nonlinear or '
        'interaction-based departure signals cannot help. AP/Brier intervals condition on frozen fitted models; event intervals also '
        'condition on fixed budget allocations. Matched time bands adjust within each trajectory, not across every injury and outcome. '
        'Small groups, repeated use of controls, multiple comparisons and retrospectively refined injury records limit inference. '
        'IL/event dates are observable proxies, not verified biological onset.',
        '',
        'Brier loss measures both discrimination and probability reliability; it is not a calibration-only score. '
        'See the [scikit-learn calibration guide](https://scikit-learn.org/1.8/modules/calibration.html).',
        '',
        '## Figures and reproducibility',
        '',
        'Figures are exported as PNG and PDF in [figures](figures/). Recommended for presentation: '
        '[incremental AP/event recall](figures/02_incremental_ap_and_event_recall.pdf), '
        '[alert-budget curves](figures/03_event_recall_alert_budgets.pdf), and '
        '[calendar score trajectories](figures/05_score_trajectories_calendar.pdf). '
        'The tables retain every model, category, partition, event denominator and uncertainty interval.',
        '',
        'Run the two notebooks with `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-profiles`. '
        'Cached stages verify source/config/code/output fingerprints before reuse; set FORCE_RECOMPUTE=True in the notebooks '
        'to reproduce fitting. The DuckDB database, canonical injuries, original cohorts, matching and preceding results remain unchanged.',
        '']
    text='\n'.join(lines)
    (OUT/'findings_summary.md').write_text(text)
    save_manifest()
    return text

