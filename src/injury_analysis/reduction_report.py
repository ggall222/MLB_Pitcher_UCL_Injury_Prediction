"""Findings generated from executed dimensionality-reduction comparisons."""
import pandas as pd
from .reduction import OUT,read_table,table,save_manifest


def write_synthesis():
    r=read_table('redundancy_summary').set_index('block')
    persistence=read_table('component_persistence')
    loadings=read_table('component_loadings').query('component <= 3')
    top=loadings.assign(importance=loadings.weight.abs()).sort_values('importance',ascending=False).groupby(['block','component']).head(3)
    top=top.sort_values(['block','component','importance'],ascending=[True,True,False])
    perf=read_table('predictive_performance')
    selected=read_table('validation_dimension_selection')
    comparison=perf.merge(selected[['category','selected_variant']],on='category')
    comparison=comparison.loc[comparison.variant.eq(comparison.selected_variant)]
    table(comparison,'selected_pca_comparison')
    test=comparison.loc[comparison.partition.eq('test')]
    shapes=read_table('latent_shape_contrasts')
    shapes=shapes.loc[shapes.cohort.eq('Heldout 2022–2025')&shapes['mode'].eq('appearance')&
        shapes.feature.isin(['profile_PC1','departure_PC2','departure_PC3'])&
        shapes.category.isin(['Elbow / UCL','Shoulder','Forearm','Lower body'])&
        shapes.metric.isin(['average_separation','final_step_vs_previous_two'])]
    shoulder_offset=shapes.loc[shapes.category.eq('Shoulder')&shapes.feature.eq('departure_PC3')&shapes.metric.eq('average_separation')].iloc[0]
    sensitivity=read_table('latent_sensitivity')
    angle=read_table('subspace_sensitivity')
    diagnostic=perf.loc[perf.partition.eq('test')&perf.variant.isin(['Original 62','Workload + history','Role/hand + missingness'])]
    stability=read_table('training_refit_summary')
    budgets=read_table('alert_budget_performance')
    budgets=budgets.loc[budgets.partition.eq('test')&budgets.alert_rate.eq(.05)]
    budgets=budgets.merge(selected[['category','selected_variant']],on='category')
    budgets=budgets.loc[budgets.variant.eq('Original 62')|budgets.variant.eq(budgets.selected_variant)]
    dims=read_table('predictive_dimensions')
    sanity=read_table('full_rotation_sanity')
    influence=read_table('positive_pitcher_influence')
    concentration=influence.loc[influence.removed_positive_pitcher.notna()].groupby('category').ap_delta.agg(['min','max']).reset_index()
    shoulder_step=shapes.loc[shapes.category.eq('Shoulder')&shapes.feature.eq('departure_PC2')&shapes.metric.eq('final_step_vs_previous_two')].iloc[0]
    shoulder_budget=budgets.loc[budgets.category.eq('Shoulder')].set_index('variant')
    shoulder_variant=selected.set_index('category').loc['Shoulder','selected_variant']
    shoulder_delta=read_table('event_recall_differences').query("partition=='test' and category=='Shoulder'").set_index('variant').loc[shoulder_variant]
    hand_validation=comparison.loc[comparison.category.eq('Hand / wrist')&comparison.partition.eq('validation')].iloc[0]
    hand_test=test.loc[test.category.eq('Hand / wrist')].iloc[0]
    p=persistence.loc[persistence.block.eq('Prior pitcher profile'),'between_pitcher_fraction']
    d=persistence.loc[persistence.block.eq('Within-pitcher departure'),'between_pitcher_fraction']
    text=f'''# Dimensionality reduction: executed exploratory findings

## Design and scope

This extension reuses the 62-feature cohort, 30-day injury definitions, category-versus-IL-free risk sets, and original 1,225 matched pairs. Training ends in 2021, validation covers 2022–2023, and the previously examined test period is 2024–2025, with the existing 60-day embargo. There is no new independent confirmatory test. All central data and preceding analyses remain intact.

The three notebooks distinguish (1) redundancy in 60 continuous selected predictors, keeping the two binary context variables outside PCA; (2) eight strictly prior pitcher baseline means; and (3) the corresponding eight current pitcher-relative departures. Arm angle remains outside the long-history experiment because of structural availability. Predictive comparisons use the original 62 features; the extra baseline measurements support description only.

All medians, scales and PCA bases are fitted on training rows. Descriptive maps use one common label-free training cohort; predictive PCA is refitted separately inside each category's training cohort. PCA is linear, unwhitened and unsupervised. Components maximize feature variance, not injury discrimination. No label-colored embedding is treated as evidence of natural injury clusters.

## Main findings

- **Moderate redundancy, not a tiny latent space:** {int(r.loc['Current feature space','components_90'])} of 60 continuous dimensions retain 90% of training variance. The first two retain only {r.loc['Current feature space','first_two_fraction']:.1%}.
- **Distinct temporal representations:** prior-profile PCs vary mainly between pitchers; departure PCs vary mainly within pitchers. In the held-out matched comparison, the displayed profile axes show no clear large case/control separation, while shoulder departure components contain both a sustained mixed-measurement offset and a late change.
- **Shoulder's late multivariate change:** departure PC2's excess final step is {shoulder_step['mean']:.3f} training-PC SD ({shoulder_step.low:.3f} to {shoulder_step.high:.3f}; {int(shoulder_step.events)} pairs). It combines velocity, release height, spin and extension and is not a physiological diagnosis.
- **No broad AP improvement:** validation-selected global PCA improves test AP by point estimate in {int(test.ap_delta_original.gt(0).sum())}/7 categories. Hand/wrist has a large test gain ({hand_test.ap_delta_original:+.4f}) but a validation change of {hand_validation.ap_delta_original:+.4f}; the small-event and pitcher-influence audits remain important.
- **AP and practical event warning can disagree:** for shoulder, the selected PCA model warns {shoulder_budget.loc[shoulder_variant,'injury_event_recall']:.1%} of events versus {shoulder_budget.loc['Original 62','injury_event_recall']:.1%} for the original features at the same 5% budget. The paired event-recall difference is {shoulder_delta['mean']:.1%} (conditional interval {shoulder_delta.low:.1%} to {shoulder_delta.high:.1%}), despite lower shoulder AP. This is an exploratory metric-specific benefit, not confirmation of a superior warning system.

## 1. Redundancy and dimension

{r[['features','training_rows','training_pitchers','first_two_fraction','components_80','components_90','components_95','participation_rank','pairs_abs_corr_90']].to_markdown(floatfmt='.3f')}

The full continuous feature space needs {int(r.loc['Current feature space','components_90'])}/60 components for 90% of variance and {int(r.loc['Current feature space','components_95'])}/60 for 95%. The first two explain {r.loc['Current feature space','first_two_fraction']:.1%}. Correlated transformations permit compression, but a two-dimensional scatter omits most variation. The participation rank is a variance-concentration summary, not an estimate of biological dimensionality.

The baseline and departure blocks each retain broad structure rather than collapsing to one or two factors. `high_correlations.csv` records the actual redundant pairs. Binary fields and missingness flags are not rotated into measurement components.

## 2. What drives the axes?

The three largest absolute projection weights for each displayed component are:

{top[['block','component','feature','weight','feature_score_correlation']].to_markdown(index=False,floatfmt='.3f')}

Weights refer to standardized original variables; correlations describe feature/score association. Component signs are arbitrary. Names such as profile or departure describe the input construction, not validated physiological mechanisms. Current-feature PC1's correlation with starter status is {persistence.loc[persistence.block.eq('Current feature space')&persistence.component.eq(1),'corr_starter'].iloc[0]:.3f}. Its large workload/role contribution can dominate an unsupervised map even when the research question concerns mechanics.

Training between-pitcher variance fractions are:

{persistence.to_markdown(index=False,floatfmt='.3f')}

For the first three prior-profile PCs, {p.min():.1%}–{p.max():.1%} of score variance is between pitchers, versus {d.min():.1%}–{d.max():.1%} for departure PCs. This supports keeping the two representations separate. It is also partly expected from their construction: a rolling prior mean is smooth, whereas a z-score measures departures. These fractions are descriptive sums of squares, not random-effects ICC estimates or proof of susceptibility.

## 3. Injury-group separation and approach-to-event changes

Primary latent trajectories use only original matched pairs whose complete observed windows and injury anchors lie in 2022–2025, outside PCA training. Each scored trajectory point requires all eight measured inputs for its block. Balanced curves keep the same case/control pairs across all five outings or four calendar bins. All-year results and all-available-pair curves are separate sensitivity outputs. Bands resample whole case pitchers and are simultaneous across time within a component/category, not across every comparison.

Selected paired contrasts (training-PC SD units) are:

{shapes[['feature','category','metric','mean','low','high','events']].to_markdown(index=False,floatfmt='.3f')}

Read `average_separation` as a persistent case/control offset; read `final_step_vs_previous_two` as the excess final-outing change after subtracting the control's own change. The first two axes show broad event overlap. Absence of separation there does not establish absence of signal in omitted components. Departure PC2 includes velocity, release height, spin and extension; a negative change is a weighted combination, not a pure velocity decline or injury-onset estimate.

The departure view also contains a sustained shoulder offset in PC3: {shoulder_offset['mean']:.3f} training-PC SD averaged across the final five outings (95% interval {shoulder_offset.low:.3f} to {shoulder_offset.high:.3f}; {int(shoulder_offset.events)} pairs). This component loads positively on spin, release height and velocity and negatively on extension. It shows why a within-pitcher departure need not be an abrupt change: departures can persist across several outings. The smaller balanced calendar cohort and multiple-comparison uncertainty prevent assigning a precise onset or universal warning time to this offset.

All six displayed axes are exported, including uncertain/null comparisons. Same-throwing-hand, strict-QA, role and final-week-exclusion checks are in `latent_sensitivity.csv`. A visual mean trajectory cannot be interpreted as a reliable individual warning threshold. Recorded dates remain injury/IL proxies; current-outing measurements become available after the outing.

## 4. Predictive comparison

The logistic model, C=1 penalty, labels and rows remain the same. Numerical convergence is tightened consistently for every new fit. Full-rank unwhitened PCA is an orthogonal-rotation control and should preserve the same linear prediction problem; the largest observed AP discrepancy is {sanity.ap_difference.abs().max():.8f}. `full_rotation_sanity.csv` also compares the tighter original-feature refit with the preserved earlier default-tolerance results. This is a numerical refinement, not a change to injury labels or historical results.

Global PCA dimensions 5, 10, 20 and a training-only 90%-variance cutoff are predeclared. A single global-PCA specification per category is selected by validation AP only. Workload/history plus compressed Statcast is a separate fixed comparison. The table below reports test results for the validation-selected global specification:

{test[['category','variant','design_dimensions','average_precision','pr_lift','ap_delta_original','delta_low_original','delta_high_original']].to_markdown(index=False,floatfmt='.4f')}

Selected global PCA has higher test AP by point estimate in {int(test.ap_delta_original.gt(0).sum())}/7 categories; its paired 95% interval excludes zero on the positive side in {int(test.delta_low_original.gt(0).sum())}/7. These are conditional fitted-model intervals, without validation-selection or multiple-comparison adjustment. Compare the full validation/test table rather than calling the largest test result a winner.

“PCA 5” is not a five-predictor model: the two binary fields and all training-observed missingness flags remain outside PCA. Actual design sizes, including these columns, are:

{dims.groupby('variant')[['source_features','retained_components','missingness_indicators','unrotated_columns','design_dimensions']].first().to_markdown()}

The role/hand-plus-missingness diagnostic tests whether metadata and feature availability carry information without continuous feature values:

{diagnostic[['category','variant','average_precision','pr_lift']].to_markdown(index=False,floatfmt='.4f')}

This diagnostic is not pure missingness: it explicitly retains role and throwing hand. Missingness also reflects historical availability and prior observation support. It must not be assigned a physiological interpretation.

## 5. Generalization, stability and practical warning

{stability.to_markdown(index=False,floatfmt='.4f')}

Twelve paired whole-pitcher training bootstrap refits relearn imputation, scaling, PCA and logistic coefficients. The originally selected dimension specification is fixed; a 90% variance rule still learns its retained count inside each refit. Rank correlation and alert-set Jaccard measure similarity to the original fit; larger values indicate greater stability. AP SD across these refits measures training perturbation sensitivity on a fixed test cohort, not a confidence interval for generalization. Original-feature coefficient stability is separately exported to avoid interpreting arbitrary PCA sign changes as instability.

At a retrospective 5% alert budget:

{budgets[['category','variant','qualifying_events','injury_event_recall','random_expected_event_recall','median_warning_lead_days']].to_markdown(index=False,floatfmt='.3f')}

Events need a warning 7–30 days before the recorded reference; all eligible events remain in the denominator, including events lacking a warning opportunity. Random expectations account for repeated opportunities. Event-difference intervals, false/non-actionable alert burden, warning-lead distributions and validation-fixed score thresholds are saved separately. Stability or row AP improvement need not translate into better event warning.

Frozen subgroup checks cover starters, relievers, seen/unseen training pitchers and exclusion of final-six-day positives. `positive_pitcher_influence.csv` removes each positive pitcher's rows in turn without refitting or selecting models. The range of the selected model's AP difference across those exclusions is:

{concentration.to_markdown(index=False,floatfmt='.4f')}

This identifies concentration in a few injured pitchers; it is not another model-selection procedure. Small categories remain uncertain even if one test-period point estimate is large.

## 6. Limits and implications

PCA supplies interpretable combinations and a controlled compression experiment; it does not turn components into physiological constructs. The variance objective can discard low-variance injury information. Prior profiles and departures have different temporal behavior, but persistent profiles are not automatically causal susceptibility factors, and late departures may reflect existing symptoms.

PCA bases are additionally checked under capped pitcher contributions, complete-measurement training and early/recent training eras. `subspace_sensitivity.csv` reports sign-invariant subspace overlap. Shared scaling isolates basis changes; this sensitivity does not refit a deployable model. Near-equal eigenvalues can make individual axes less stable than their combined subspace.

No nonlinear embedding is added merely to create apparent clusters. Given weak low-dimensional variance concentration, repeated observations, and the need for auditable held-out transformation and loadings, linear PCA answers the present questions more directly. This does not rule out nonlinear structure.

Remaining limits include selected active-pitcher controls; revised/vague injury descriptions; unverified contemporaneous reporting; missing historical measurements; partial 2026 observation (all-year description only); cross-pair dependence when a person appears elsewhere as a control; and many exploratory comparisons. A new independent evaluation period and adjudicated event timing remain necessary before a clinical or operational warning claim.

## Artifacts

Three executed notebooks: `notebooks/injury_dimensionality_reduction/`. Tables and publication-size PNG/PDF figures: `reports/injury_dimensionality_reduction/`. Fitted transforms, narrow scores, matched latent cells and predictions: `data/injury/analysis/dimensionality_reduction/`. Source DuckDB and previous analysis results are preserved.

Method references: [scikit-learn 1.6 PCA](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html) and [training-only preprocessing guidance](https://scikit-learn.org/1.6/common_pitfalls.html).
'''
    (OUT/'SUMMARY.md').write_text(text)
    save_manifest()
    return text
