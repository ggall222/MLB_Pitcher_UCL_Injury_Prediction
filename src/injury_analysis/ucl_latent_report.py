"""Computed interpretation and publication captions for the focused UCL figures."""
import pandas as pd
from .ucl_latent_visuals import OUT,read_table,table


def write_summary():
    """Build all numbers from exported analyses, without choosing axes from outcomes."""
    support=read_table('cohort_support').set_index('cohort')
    outing=support.loc['Balanced five outings'];calendar=support.loc['Balanced calendar bins']
    summary=read_table('balanced_trajectories')
    shapes=read_table('trajectory_shape_summary')
    cells=summary.loc[summary.kind.eq('difference')]
    supported=int(((cells.simultaneous_low>0)|(cells.simultaneous_high<0)).sum())
    def contrast(component,metric):
        return shapes.loc[shapes.component.eq(component)&shapes['mode'].eq('appearance')&shapes.metric.eq(metric)].iloc[0]
    def interval(row):return f'{row["mean"]:+.2f} [{row.low:+.2f}, {row.high:+.2f}]'
    d=contrast('D1','last_minus_first_separation');c=contrast('C1','last_minus_first_separation')
    dmean=contrast('D1','average_separation');cmean=contrast('C1','average_separation')
    individual=read_table('individual_event_cells')
    heterogeneity=[]
    for kind in ['Injury','Control','difference']:
        mat=individual.pivot(index='event_id',columns='point',values=kind)
        delta=mat[4]-mat[[0,1]].mean(axis=1)
        heterogeneity.append(dict(kind=kind,events=len(delta),increasing_events=int(delta.gt(0).sum()),
            decreasing_events=int(delta.lt(0).sum()),unchanged_events=int(delta.eq(0).sum()),
            mean=delta.mean(),median=delta.median(),q25=delta.quantile(.25),q75=delta.quantile(.75),
            minimum=delta.min(),maximum=delta.max(),contrast='Final outing minus mean of first two outings'))
    heterogeneity=table(pd.DataFrame(heterogeneity),'individual_shape_distribution')
    case=heterogeneity.set_index('kind').loc['Injury'];control=heterogeneity.set_index('kind').loc['Control']
    selection=read_table('ucl_validation_selection')
    performance=read_table('ucl_selected_performance')
    comparisons=read_table('ucl_selected_differences')
    budgets=read_table('ucl_alert_budgets')
    compression=selection[['block','method','selected_components','continuous_features','passthrough_columns',
        'design_columns','validation_ap','training_variance_retained','plateau_components']].copy()
    table(compression,'compression_summary')
    primary=selection.loc[selection.block.eq('Original 62')].set_index('method')
    dep=selection.loc[selection.block.eq('Departure')].set_index('method')
    comb=selection.loc[selection.block.eq('Combined')].set_index('method')
    raw=performance.query("partition=='validation' and block=='Original 62' and variant=='Unreduced'").iloc[0]
    pdiff=comparisons.query("partition=='validation' and block=='Original 62' and variant=='PLS selected' and reference=='Unreduced'").iloc[0]
    test=budgets.query("partition=='test' and block=='Original 62' and alert_rate==0.05").set_index('variant')
    ediff=comparisons.query("partition=='test' and block=='Original 62' and variant=='PLS selected' and reference=='Unreduced'").iloc[0]
    loads=read_table('component_loadings');definitions=read_table('component_selection')
    component_rows=[]
    for component,g in loads.groupby('component',sort=False):
        top=g.nlargest(4,'squared_axis_weight')
        component_rows.append(dict(component=component,
            profile_squared_weight=g.loc[g.representation.eq('Profile'),'squared_axis_weight'].sum(),
            departure_squared_weight=g.loc[g.representation.eq('Departure'),'squared_axis_weight'].sum(),
            top_features='; '.join(f'{r.feature} ({r.injury_oriented_weight:+.3f}; {100*r.squared_axis_weight:.1f}% squared weight)'
                                  for r in top.itertuples()),
            training_absolute_standardized_logit_weight=definitions.set_index('component').loc[component,'training_absolute_standardized_logit_weight']))
    interpretation=table(pd.DataFrame(component_rows),'component_interpretation')
    leads=read_table('outing_calendar_support').set_index('appearance_before')
    out_clusters=int(summary.loc[summary['mode'].eq('appearance'),'bootstrap_clusters'].iloc[0])
    cal_clusters=int(summary.loc[summary['mode'].eq('calendar'),'bootstrap_clusters'].iloc[0])
    records=[dict(figure='01_ucl_pls_matched_trajectories',question='Persistent offset or increasing case–control divergence?',
        main_result=f'{int(outing.events)} balanced pairs. D1 paired last-minus-first change {interval(d)} SD; C1 {interval(c)} SD.',
        profile_or_imminence='Neither a stable susceptibility offset nor a shared imminent-warning rise is clearly established in these matched means.',
        development_use='D1 prioritizes lower velocity relative to personal baseline; retain as a candidate, not a validated warning marker.'),
      dict(figure='02_ucl_component_lead_time_heatmap',question='When does matched separation emerge across retained supervised axes?',
        main_result=f'{supported}/{len(cells)} cells have joint 95% bands excluding zero; {int(outing.events)} outing pairs and {int(calendar.events)} calendar pairs.',
        profile_or_imminence='Inspect persistent versus late patterns, but no supported onset of divergence can be assigned from color alone.',
        development_use='Prevents selecting a lead-time window from the largest unadjusted heatmap cell; calendar and outing cohorts differ.'),
      dict(figure='03_ucl_pca_pls_compression',question='How compact is predictive information compared with PCA?',
        main_result=f'Original inputs: PLS {int(primary.loc["PLS","selected_components"])} versus PCA {int(primary.loc["PCA","selected_components"])} selected axes; '
            f'validation AP {primary.loc["PLS","validation_ap"]:.4f} versus {primary.loc["PCA","validation_ap"]:.4f}, unreduced {raw.average_precision:.4f}.',
        profile_or_imminence='Supervision can compress ranking information; the curve does not distinguish susceptibility from imminence.',
        development_use='A smaller candidate representation is feasible, but context/missingness columns remain and predictive benefit is uncertain.'),
      dict(figure='04_individual_ucl_latent_trajectories',question='Does the average hide heterogeneous individual patterns?',
        main_result=f'D1 final-minus-early score rises in {int(case.increasing_events)}/{int(case.events)} cases and '
            f'{int(control.increasing_events)}/{int(control.events)} matched controls; case IQR [{case.q25:+.2f}, {case.q75:+.2f}] SD.',
        profile_or_imminence='Persistent, rising, falling and irregular paths coexist; ordering exposes variation without identifying phenotypes.',
        development_use='Requires prospective event-level warning evaluation; a rise or positive score is not a calibrated risk threshold.')]
    table(pd.DataFrame(records),'figure_summary')
    lines=['# Focused elbow/UCL latent visuals','',
        '**Main result.** PLS retains validation ranking with fewer continuous axes than PCA. '
        f'The matched pre-event analysis has {supported} of {len(cells)} component/time cells with a joint interval excluding zero; '
        'these figures do not establish a common ahead-of-time injury warning.','',
        '**Endpoint scope.** The saved models predict the established **Elbow / UCL** category, which includes nonspecific elbow '
        'inflammation, soreness and strains. They are not confirmed-UCL-only models. '
        f'Of {int(outing.events)} five-outing matched events, only {int(outing.initial_explicit_ucl_events)} explicitly name UCL injury at initial placement; '
        f'{int(outing.later_supplemental_ucl_confirmation)} have later supplemental UCL confirmation. '
        'Later confirmation is audit metadata, never a predictor or a substituted injury date. A four-event subgroup cannot support a separate trajectory claim.','',
        '## Figure guide','',
        '| Figure / question | Computed result | Profile versus imminent warning | Use in further development |',
        '|---|---|---|---|']
    for r in records:
        lines.append(f'| [{r["figure"]}](figures/{r["figure"]}.pdf): {r["question"]} | {r["main_result"]} | {r["profile_or_imminence"]} | {r["development_use"]} |')
    lines += ['', '## Component meaning and time scale','',
        'D1 is principally a **lower fastball-velocity departure** direction: `velo_z` supplies '
        f'{100*loads.loc[loads.component.eq("D1")&loads.feature.eq("velo_z"),"squared_axis_weight"].iloc[0]:.1f}% of squared axis weight, '
        'with smaller spin and extension contributions. D2 emphasizes spin and extension departures. '
        'C1 mixes velocity departure with prior profile and career/context fields. C2 is more profile-weighted. '
        'These are descriptions of standardized linear coordinates, not tissue mechanisms or fractions of explained injury risk.','',
        '| Axis | Profile squared weight | Departure squared weight | Absolute training logit weight per score SD |',
        '|---|---:|---:|---:|']
    for r in interpretation.itertuples():
        lines.append(f'| {r.component} | {100*r.profile_squared_weight:.1f}% | {100*r.departure_squared_weight:.1f}% | {r.training_absolute_standardized_logit_weight:.4f} |')
    lines += ['', 'The nearly zero classifier weight on C5 illustrates why an apparently colored latent trajectory is not necessarily predictive. '
        'All seven validation-retained axes remain in the heatmap; the four emphasized axes and D1 individual plot were selected by training logit weight times training score SD. '
        'Higher plotted scores point toward higher fitted logit *holding other model coordinates fixed*. They are not probabilities or complete model predictions.','',
        f'Average paired offset: D1 {interval(dmean)} SD; C1 {interval(cmean)} SD (descriptive 95% intervals). '
        'Their uncertainty includes zero. A flat group offset, even if established later, would not by itself be an imminent warning.','',
        f'The last observed outing is a median **{leads.loc[1,"median"]:g} day** before the recorded event '
        f'(IQR {leads.loc[1,"q25"]:g}–{leads.loc[1,"q75"]:g}); the fifth-last is {leads.loc[5,"median"]:g} days before. '
        'An apparent last-outing change therefore cannot substantiate a seven-day warning. Calendar bins use actual elapsed days, '
        'with one mean per event and group in each bin; no interpolation or smoothing is applied.','',
        '## Compression versus useful warning','',
        f'On {int(raw.rows):,} validation appearances ({int(raw.positive_rows)} positive rows, '
        f'{int(raw.represented_events)} represented events; prevalence {100*raw.prevalence:.2f}%), '
        f'PLS reaches its best tested original-input validation AP with {int(primary.loc["PLS","selected_components"])} axes, '
        f'versus {int(primary.loc["PCA","selected_components"])} for PCA. These retain '
        f'{100*primary.loc["PLS","training_variance_retained"]:.1f}% and {100*primary.loc["PCA","training_variance_retained"]:.1f}% '
        'of continuous training variance, respectively. This supports compact outcome-directed ranking relative to the tested PCA representations; '
        'it does not establish a biological low-dimensional injury process.','',
        f'The selected departure model uses {int(dep.loc["PLS","selected_components"])} PLS versus '
        f'{int(dep.loc["PCA","selected_components"])} PCA axes; combined inputs use {int(comb.loc["PLS","selected_components"])} versus '
        f'{int(comb.loc["PCA","selected_components"])}. Each method retains the same unprojected binary context/missingness fields within a representation. '
        f'The original-input PLS classifier therefore has **{int(primary.loc["PLS","design_columns"])} total columns**, '
        f'not {int(primary.loc["PLS","selected_components"])}. Refer to `compression_summary.csv` for the full accounting.','',
        f'PLS-minus-unreduced validation AP is {pdiff.ap_difference:+.4f} '
        f'(paired pitcher-bootstrap interval [{pdiff.ap_low:+.4f}, {pdiff.ap_high:+.4f}]). '
        'These conditional intervals do not account for validation selection. The plotted candidate grid is finite; neither a precise optimal dimension nor a statistically established plateau is claimed.','',
        'For the already examined 2024–2025 period, the existing 5% appearance-alert budget detects '
        f'{100*test.loc["PLS selected","injury_event_recall"]:.1f}% of broad elbow/UCL events 7–30 days ahead using PLS, '
        f'versus {100*test.loc["Unreduced","injury_event_recall"]:.1f}% with unreduced features. '
        f'The paired difference is {100*ediff.event_mean:+.1f} percentage points '
        f'[{100*ediff.event_low:+.1f}, {100*ediff.event_high:+.1f}]. '
        'Thus compression does not demonstrate a clear practical-warning improvement. These inherited metrics use the prior evaluable risk set, '
        'all-event denominator and retrospective top-score budget; they are not a prospective deployment trial.','',
        '## Design and uncertainty','',
        '- Frozen PLS/PCA transforms, imputers, scales and classifiers were learned through 2021. Existing 2022–2023 validation selection is reused; no new fitting or test-based component selection occurs.',
        '- Both matched windows and anchors are in the previously examined 2022–2025 interval. Cases and controls retain their original identities and outing alignment.',
        f'- Complete profile-plus-departure measurements are required for descriptive scores. The same {int(outing.events)} events contribute at each of five outings; '
        f'{int(calendar.events)} events contribute at every calendar bin. These are different subsets and may be selected by data/history availability. Predictive curves retain the original larger imputed risk sets.',
        f'- Uncertainty resamples connected sets of matched pairs sharing either case or control pitchers: {out_clusters} outing clusters and {cal_clusters} calendar clusters, '
        '300 deterministic replicates. Recurrent cases and pitchers switching case/control roles remain together.',
        '- Joint 95% max-t bands cover all retained axes and time points within each alignment and separately for case means, control means and paired differences. '
        'Shape-contrast intervals are descriptive and are not jointly corrected across all contrasts. Model fitting, matching and dimension selection uncertainty are not resampled.',
        '- Per-event scores use frozen training-score SD units; there is no event-specific centering or scaling. The individual plot shares row order across cases, controls and differences. '
        'Sorting by final-minus-early case score necessarily displays a gradient and is not evidence of discrete phenotypes. Display colors clip at ±3 SD; exported values remain unclipped.',
        '- Predictors describe completed outings. Personal baselines use prior observations. Event dates remain recorded injury/IL proxies, not verified biological onset, surgery date or return date.',
        '- Matched separation is descriptive, not a sensitivity/specificity estimate. The 2024–2025 period has been inspected repeatedly and supplies no new independent confirmation.','',
        '## What to freeze next','',
        'First adjudicate the intended UCL-specific endpoint against the existing procedure/source audit, retaining known-at-the-time dates separately from later confirmation. '
        'D1 provides a transparent velocity-departure candidate; test whether strictly prior persistence/trends add to profile risk at a predeclared 7–30-day warning window and controlled alert budget. '
        'Freeze features, transformations, outcome rules and validation-selected thresholds before evaluating a genuinely untouched chronological cohort. '
        'The current figures support a compact representation and heterogeneous trajectories; they do not validate an imminent UCL warning system.','',
        '## Scope and files','',
        'Four PNG/PDF figure pairs are exported. Loading interpretation is supplied as a table instead of another family plot. '
        'No relevance-versus-stability plot is added because prior training refits covered shoulder/lower-body models, not this elbow/UCL target. '
        'No horizon-specific predictive heatmap is fabricated from calendar-aligned means; that requires separately defined horizon outcomes and evaluation.','',
        '- [Figure summary CSV](tables/figure_summary.csv)',
        '- [Component interpretation](tables/component_interpretation.csv) and [all loadings](tables/component_loadings.csv)',
        '- [Balanced trajectories and uncertainty](tables/balanced_trajectories.csv)',
        '- [Endpoint audit](tables/endpoint_definition_audit.csv) and [cohort support](tables/cohort_support.csv)',
        '- [Event row lookup](tables/individual_event_order.csv) and [unclipped event cells](tables/individual_event_cells.csv)',
        '- [Executed notebook](../../../notebooks/ucl_latent_visuals/01_ucl_latent_visuals.ipynb)',
        '- [Methods and rerun commands](../../../docs/ucl_latent_visuals.md)','']
    text='\n'.join(lines)
    (OUT/'SUMMARY.md').write_text(text)
    captions=f'''# Suggested figure captions

All four figures use the established **broad Elbow / UCL** outcome, including nonspecific elbow diagnoses. They must not be described as confirmed-UCL-only results. The evaluated periods have already been examined; these are exploratory findings.

**Figure 1. Matched pre-event supervised-component trajectories.** Frozen training-derived PLS scores for departure (D1–D2) and combined profile/departure (C1–C2) inputs, aligned to the final five observed outings before the recorded injury/IL event or matched control anchor. The same {int(outing.events)} event pairs ({int(outing.pitchers)} injured pitchers; {out_clusters} connected pair clusters) contribute to every point. Only {int(outing.initial_explicit_ucl_events)} cases explicitly name UCL injury at initial placement. Axes are oriented by the training logistic coefficients and expressed in training-score SD units. Shading is a joint 95% bootstrap band across all seven retained axes and five positions, separately within cases and controls; annotations give paired last-minus-first separation changes with descriptive 95% intervals. D1 change is {interval(d)} SD and C1 is {interval(c)} SD. These contrasts do not establish a common approaching-injury rise. The last outing is a median {leads.loc[1,"median"]:g} day before the recorded event, so late separation would not itself demonstrate a seven-day warning.

**Figure 2. Supervised-component separation by lead time.** Equal-event-weighted case-minus-control differences for all validation-retained departure and combined PLS axes, in training component SD units on a shared zero-centered color scale. Outing alignment uses {int(outing.events)} complete matched pairs; calendar alignment uses a different set of {int(calendar.events)} pairs, complete in all four bins. Calendar values are event/group means within actual elapsed-day bins. A dot would indicate that a joint 95% bootstrap interval excludes zero; {supported} of {len(cells)} cells meet that criterion. Joint bands span all components and positions within each alignment. Color magnitude is not predictive importance, and these means do not estimate horizon-specific discrimination or biological onset.

**Figure 3. PCA versus PLS validation compression.** Average Precision versus retained continuous dimensions for the original 62 inputs, departure inputs, and combined profile/departure inputs. All transforms and classifiers were trained through 2021. Curves reuse {int(raw.rows):,} validation appearances from 2022–2023 ({int(raw.positive_rows)} positive rows; {int(raw.represented_events)} represented broad elbow/UCL events). Squares mark the prior validation-selected component counts; dashed lines show unreduced models using the same panel's inputs and dotted lines show prevalence. PLS selects {int(primary.loc["PLS","selected_components"])} versus {int(primary.loc["PCA","selected_components"])} PCA axes for original inputs, {int(dep.loc["PLS","selected_components"])} versus {int(dep.loc["PCA","selected_components"])} for departure, and {int(comb.loc["PLS","selected_components"])} versus {int(comb.loc["PCA","selected_components"])} for combined inputs. Context and missingness columns bypass projection, so latent dimension is not total model size. Selection-period estimates are optimistic; the curve supports compact ranking but does not demonstrate a statistically clear practical-warning improvement.

**Figure 4. Individual matched-event departure trajectories.** D1 scores for {int(outing.events)} individual elbow/UCL events, their matched controls, and paired differences across the final five outings. D1 was selected by its training standardized classifier weight, not the displayed trajectories, and is dominated by lower fastball-velocity departure. All panels share event order, sorted by case score at the final outing minus the average at the first two outings. Scores retain a common training-derived origin and SD without per-event centering. Colors clip at ±3 SD; exported cells retain exact unclipped values and the row lookup supplies event identities. Case scores increase under this descriptive contrast in {int(case.increasing_events)}/{int(case.events)} events; controls increase in {int(control.increasing_events)}/{int(control.events)}. Rising, falling, persistently different, and irregular paths coexist. Sorting creates a gradient by construction and does not identify injury phenotypes or establish an alert threshold.
'''
    (OUT/'FIGURE_CAPTIONS.md').write_text(captions)
    return text
