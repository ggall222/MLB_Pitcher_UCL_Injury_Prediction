"""Generate findings from finished numerical artifacts, with explicit limits."""
import json
import numpy as np
import pandas as pd
from .consolidation_common import ROOT, OUT, CACHE, read_table, table


def md(frame):
    def cell(value):
        if pd.isna(value):return '—'
        if isinstance(value,(float,np.floating)):return f'{value:.5g}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write_report():
    inv=read_table('reconciled_feature_inventory');clusters=read_table('correlation_clusters')
    pairs=read_table('high_correlation_pairs');deps=read_table('deterministic_dependencies')
    family=read_table('family_performance');ab=read_table('family_paired_differences')
    perm=read_table('grouped_permutation_summary');stability=read_table('feature_stability')
    curve=read_table('candidate_reduction_curve');coverage=read_table('coverage_comparison')
    keep=read_table('retained_features');selection=json.loads((CACHE/'candidate_selection.json').read_text())
    selected=curve.loc[curve.selected].iloc[0];full=curve.set_index('variant').loc['Full feasible']
    reps=curve.set_index('variant').loc['Correlation representatives']
    frame=read_table('frozen_performance');trans=read_table('frozen_paired_differences')
    cond=read_table('conditional_feature_checks');support=read_table('event_support')
    clear=ab.loc[ab.variant.str.startswith('Without')&ab.low.gt(0)]
    stable=stability.loc[stability.selection_frequency.ge(.9)&stability.conditional_sign_stability.ge(.9)]
    unstable=stability.loc[stability.selection_frequency.gt(0)&stability.conditional_sign_stability.lt(.8)]
    roles=keep.groupby('signal_interpretation').size()
    evidence=inv[['feature','feature_group','underlying_concept','representation','training_missing_fraction',
        'training_observed_pitchers','near_zero_variance_flag','screen_exclusion','timing_availability']].merge(
        stability,on='feature',how='left',validate='one_to_one').merge(
        clusters[['feature','correlation_cluster','representative_feature']],on='feature',how='left',validate='one_to_one')
    evidence['retained']=evidence.feature.isin(keep.feature)
    evidence['weak_sparse_support']=evidence.selection_frequency.lt(.25)
    evidence['unstable_selected_sign']=evidence.conditional_sign_stability.lt(.8)&evidence.selection_frequency.ge(.25)
    evidence['marginal_permutation_not_consistently_positive']=evidence.permutation_positive_frequency.lt(.5)
    table(evidence,'noise_and_stability_assessment')
    # Post-fit review annotations do not alter frozen statistical decisions.
    # Preserve the frozen inventory and publish precise type-filter provenance.
    from .consolidation_inventory import classify
    review=inv.copy();clarifications=[]
    for i,r in inv.iterrows():
        correct=classify(r.feature,r.source_layer)['source_variables']
        if correct!=r.source_variables:
            clarifications.append(dict(feature=r.feature,original_source_variables=r.source_variables,
                corrected_source_variables=correct,reason='Explicit pitch-type filter; type-specific counts use Statcast rows, not official appearance pitch_count'))
            review.loc[i,'source_variables']=correct
    table(review,'feature_inventory_review')
    table(pd.DataFrame(clarifications),'source_field_clarifications')
    recommendations=evidence.copy()
    cluster_sizes=clusters.groupby('correlation_cluster').size()
    reasons=[]
    for r in recommendations.itertuples():
        if r.retained:
            reason='Retained in validation-selected departure block; individual conditional evidence remains uncertain except where reported.'
        elif isinstance(r.screen_exclusion,str) and r.screen_exclusion:
            reason=r.screen_exclusion
        elif r.feature in set(inv.loc[inv.support_only_sensitivity,'feature']):
            reason='Explicit support field: isolated sensitivity showed no clear compact-model AP gain and fewer warning events.'
        else:
            details=[]
            if r.selection_frequency<.25:details.append(f'Low measured selection frequency ({r.selection_frequency:.0%}) across training refits')
            if r.conditional_sign_stability<.8:details.append(f'Coefficient direction unstable when selected ({r.conditional_sign_stability:.0%} sign agreement)')
            if cluster_sizes.get(r.correlation_cluster,1)>1:details.append('Shares a high-correlation cluster; candidate representative: '+str(r.representative_feature))
            if r.permutation_positive_frequency<.5:details.append('Marginal permutation loss positive in fewer than half of refits; not a conditional no-value claim')
            if not details:details.append('Plausible distinct or stable predictor; larger joint candidates did not meet the declared compact warning selection rule')
            reason='; '.join(details)+'. Exclusion applies to this candidate, not the master table.'
        reasons.append(reason)
    recommendations['recommendation_rationale']=reasons
    table(recommendations,'feature_recommendation_details')
    # One comparison map per concept, preserving all order-dependent contrasts.
    comparisons=read_table('representation_comparisons')
    rows=[]
    for concept,g in comparisons.groupby('concept',sort=False):
        fields=set(';'.join(g.added_features.fillna('')).split(';'))-{''}
        cg=clusters.loc[clusters.feature.isin(fields)]
        strong=g.loc[g.increment_low.gt(0),'addition'].tolist()
        rows.append(dict(concept=concept,candidate_representations=';'.join(g.addition),
            compared_fields=';'.join(sorted(fields)),distinct_correlation_clusters=cg.correlation_cluster.nunique(),
            positive_increment_intervals=';'.join(strong) or 'None; all positive contrasts uncertain',
            largest_observed_ap_gain=float(g.ap_increment.max()),
            retained_fields=';'.join(keep.loc[keep.feature.isin(fields),'feature']),
            stability=';'.join(f'{r.feature}: {r.selection_frequency:.2f}' for r in keep.loc[keep.feature.isin(fields)].itertuples()),
            interpretation='Nested order-specific validation comparisons; no multiplicity correction; absence of a positive interval is not proof of no value'))
    table(pd.DataFrame(rows),'concept_representation_decisions')
    compact=keep[['feature','feature_group','representation','selection_frequency','conditional_sign_stability','ap_loss']]
    text=f"""# Feature consolidation: completed development analysis

## What was resumed

The interruption left 19 readable CSV tables, seven fully readable Parquets, five JSON artifacts and 94 loadable fitted models. Inventory, correlations, 27 family comparisons, 56 nested representation models, four supported-cohort models, and all six Elastic Net tuning fits were complete. The last completed numerical step was the Elastic Net grid. Its selection metadata was missing; this was reconstructed from the saved fits. Sixteen whole-pitcher sparse refits, reduced candidates, conditional checks, coverage analyses, freezing, transfer evaluation, figures and notebooks were then completed.

All original numerical artifacts were reused. Thirty source-variable descriptions and eight timing descriptions were finalized in a **separate reconciled inventory**. A final provenance addendum explicitly identifies pitch-type filters and distinguishes Statcast pitch-type counts from official appearance pitch counts; `feature_inventory_review.csv` is the complete annotated inventory. The frozen statistical inventory remains unchanged. An unordered family-pair summary was also corrected in a separate file; the underlying correlations were unchanged. The original database, previous PCA/PLS results and 125 audited checkpoint artifacts remain protected by fingerprints. See `resume/` and `validation_checks.csv` for verification.

## Cohort and endpoint

{md(support)}

The registered **Elbow / UCL** target includes non-UCL elbow conditions. Only 15 training, three validation and five test represented events explicitly mention UCL in their initial text. Later surgical confirmation is outcome information, not a predictor and not proof that initial diagnoses were complete. This stage therefore selects features for a **broad elbow proxy**, pending UCL-specific adjudication. The primary prediction is after a completed appearance, for the next 30 days.

The original chronological boundaries and 60-day embargo remain intact: train through 2021; select on 2022–2023; inspect 2024–2025 only after freezing. Pitchers may recur chronologically, as intended for predicting future outings of known pitchers; there are no pitcher ID predictors. Whole-pitcher uncertainty and refits preserve within-pitcher dependence. This is not an unseen-pitcher generalization experiment. 2024–2025 has already been examined extensively and is **not independent confirmation**.

## 1. How much redundancy exists?

The inventory contains {len(inv)} fields: 536 registered predictors and two existing context fields. Training-only screening leaves {int(inv.eligible_predictor.sum())} measured candidates, of which 333 are substantive and 20 describe support. {int(deps.verified_exact.sum())}/{len(deps)} tested algebraic relationships hold on their co-observed training rows. Examples include outs/innings, usage/count identities, current-minus-prior changes, z-score definitions and current-inclusive versus prior-only workload totals. Missingness can still differ between algebraically related fields.

Among the 353 feasible fields, {len(pairs)} pairs exceed |Pearson| or |Spearman| ≥0.90: {int(pairs.same_group.sum())} within families and {int((~pairs.same_group).sum())} across families. Complete linkage on training Spearman correlations produces {clusters.correlation_cluster.nunique()} within-family clusters; protecting distinct temporal roles gives {int(clusters.representative.sum())} representatives, 189 substantive. This is substantial duplication, not evidence that all members are interchangeable.

## 2. Which families provide unique information?

Extension is the only family whose removal has an unadjusted paired AP-loss interval narrowly above zero. Its point loss is {clear.ap_loss.iloc[0]:.5f} (95% interval {clear.low.iloc[0]:.5f} to {clear.high.iloc[0]:.5f}). This exploratory result does not survive an assumed guarantee about multiple comparisons; no multiplicity-adjusted claim is made. Velocity, background and movement show positive point losses with wider uncertainty. The full model relies on several families under marginal permutation, including extension, spin, movement, background and release, but reliance is not the same as unique contribution.

{md(ab.loc[ab.variant.str.startswith('Without'),['family','ap_loss','low','high']])}

The background-only model has high validation AP relative to several larger representations. This supports examining persistent/contextual risk, not interpreting every Statcast coefficient as imminent injury. AP and advance-warning counts sometimes move in opposite directions.

## 3. Which representations duplicate each other, and which differ temporally?

Current levels, adjacent rolling windows and baseline means often cluster; pitch-type counts duplicate usage plus total volume, and innings/outs and overlapping workload windows add exact or near-exact dependencies. Departures and slopes explicitly encode a different time relationship, so they were protected during correlation-only reduction. Nevertheless, none of the positive nested representation increments has a 95% paired interval wholly above zero in this validation cohort. The one clearly negative AP increment is additional seven-day workload after longer workload context; its warning count moves differently. These are order-dependent comparisons, not a declaration that departures, slopes or workload have no value.

Full contrasts and alternative choices are in `representation_comparisons.csv`, `concept_representation_decisions.csv` and `conditional_feature_checks.csv`. The latter drops each retained feature and, where available, replaces/adds a correlated alternative; these are diagnostic checks and do not initiate further test-driven pruning.

## 4. Stability and sparse selection

Validation chose Elastic Net C=0.03, l1_ratio=0.25, with 213 nonzero measured fields (222 when selected missing flags count). Sixteen whole-pitcher training refits hold those settings fixed. {len(stable)} fields have both selection frequency ≥90% and conditional sign agreement ≥90%; {len(unstable)} selected-at-least-once fields have sign agreement below 80%. Even stable selection is not proof of incremental value. Rank SD, measured-versus-flag selection, marginal permutation mean/SD and cluster selection frequency are saved for all 333 fields.

Top reproducibly selected fields:

{md(stable.sort_values(['selection_frequency','median_absolute_coefficient'],ascending=False).head(12)[['feature','selection_frequency','conditional_sign_stability','mean_rank','rank_sd','permutation_mean']])}

These are 16 conditional refits, not a new hyperparameter search in every bootstrap or formal error-controlled stability selection. Sparse coefficients for binary flags use raw-unit scaling, while continuous fields are standardized. The final candidate comparisons all use the same L2 preprocessing and fitting rule.

## 5. Reduction and the recommended candidate

The declared validation rule selected **{selection['selected']}**, containing **{len(keep)} input fields and {int(selected.design_columns)} fitted design columns** including missingness indicators. It first permits an AP loss of at most one paired-bootstrap SE relative to the highest-AP candidate, then requires warning count within two events of the best AP-competitive model, and chooses the smallest set. This is a pragmatic development tolerance, not a noninferiority test.

Validation AP: full 333 fields **{full.average_precision:.5f}**; correlation-only 189 fields **{reps.average_precision:.5f}**; selected **{selected.average_precision:.5f}**. The selected-minus-full point difference is **{selected.average_precision-full.average_precision:+.5f}**. At a retrospective 5% alert budget, the candidate warns **{int(selected.alerted_events)}/120** events 7–30 days ahead, versus **{int(full.alerted_events)}/120** for full features. The original denominator includes 120 qualifying events, of which 80 have an eligible opportunity in that warning window. Lead time is conditional on detection, not achievable for every event.

{md(curve[['variant','features','average_precision','ap_low','ap_high','alerted_events','injury_event_recall','selected']])}

The retained set contains: {', '.join(f'{n} {k.lower()}' for k,n in roles.items())}. These labels describe construction, not physiology. Background/history can include contextual seasonality, and a rolling mean is slowly changing, not an immutable pitcher trait.

**This is a warning-focused tradeoff, not a uniformly superior model.** The AP tolerance is broad because validation uncertainty is large: the selected AP is about one-third below the background-only point estimate. Correlation reduction by itself performs worse. The 62-field 90%-stability representative set offers a different development tradeoff (AP 0.03410, 18/120 events warned); it remains a documented alternative, not another test-selected winner. The eight-field set is retained as the predefined departure block, not because all eight individually show strong conditional evidence. Only `velo_z` has a positive unadjusted drop-one AP interval in that block; `release_x_sd_z` and `spin_z` have unstable signs in the full sparse refits. This prevents interpreting the final map as eight independently established biomarkers.

{md(compact)}

Every retained field has source, concept, representation, timing, correlated alternatives, conditional AP evidence and stability in `retained_features.csv`. All exclusions, including unavailable modern measurements, remain documented in `excluded_features.csv`; `feature_recommendation_details.csv` adds specific correlation, weak-selection and sign-instability reasons for each field. No master feature was deleted. The exact candidate is frozen in `config/ucl_candidate_features_v2.yaml` with fitted model and decision-table hashes in `candidate_freeze.json`.

## 6. Measurement support and low-value fields

{md(coverage[['variant','features','average_precision','ap_difference_from_no_flags','ap_difference_low','ap_difference_high','alerted_events','brier_score','calibration_bias']])}

Coverage and missingness may encode measurement availability, role, experience or historical era. Their performance cannot establish a physiological signal. The primary selection excluded explicit support fields; this sensitivity reports what changes when support and missingness flags are introduced separately. Season/role prediction and missingness summaries are in `coverage_role_season_audit.csv`. Brier score measures probability error, not calibration alone; mean bias and validation-defined calibration bins are also provided.

Arm angle and within-outing fastball decline are missing in about 82% and 80% of training rows respectively. They were not silently filled with zero or declared useless. Common-row modern and starter sensitivities show no clear AP gain, with limited event support. Raw circular spin-axis values require sine/cosine treatment before linear selection; prior UCL surgery fields require verified publication dates. `noise_and_stability_assessment.csv` reports constant/missing/rare-support flags alongside instability and conditional evidence rather than deleting individually weak fields automatically.

## 7. Temporal transfer and what to freeze

`frozen_performance.csv`, `frozen_paired_differences.csv` and `frozen_alert_budgets.csv` report already examined 2024–2025 transfer for fixed models. `validation_fixed_thresholds.csv` applies validation-derived numeric cutoffs to that period: ranking the top 5% retrospectively is different from maintaining 5% alerts prospectively. Calibration bins also use validation-defined edges. Test differences must not be used to revise this selected feature list under an untouched-test claim.

In 2024–2025, the departure candidate has AP **0.03098**, versus **0.02797** for full features and **0.04254** for the previous 62-field benchmark. Its AP difference from that benchmark is −0.01156 (paired interval −0.02687 to −0.00284). It warns **18/145** events, versus **26/145** for the previous benchmark and 15/145 for full features. The previous 62-field model must therefore remain a frozen comparator; the new candidate is not a validated replacement. The departure model's validation-fixed threshold produces **4.57%** test alerts. At the retrospective 5% budget its row precision is only **3.04%**, and median warning lead time of **23.5 days** applies to detected events only. This remains an exploratory warning system with substantial false-alert burden.

{md(frame[[c for c in ['partition','variant','average_precision','pr_lift','brier_score','calibration_bias'] if c in frame]])}

Freeze the **feature list, exact derivations, training preprocessing, L2 fit, endpoint definition, 30-day horizon and 7–30-day/5%-budget evaluation** now. Retain profile/departure and the previous 62-field comparators. Before confirmatory UCL work, adjudicate UCL-specific events and knowledge dates, then register an untouched future cohort and a prospective threshold policy. Any endpoint revision requires a new development/freeze cycle; these elbow-proxy results cannot validate confirmed ligament injury warning.

## Deliverables and limitations

Three executed notebooks are in `notebooks/feature_consolidation/`. Six figures, each PNG and PDF, cover within-family correlations, cross-family redundancy, ablation/permutation, sparse stability, reduced-candidate performance and support sensitivity. Tables preserve all intermediate evidence and final decisions. `methods.md` documents the exact analysis; `validation_checks.csv` and `validation_summary.json` record integrity and execution checks.

Feature screening and repeated comparisons consume validation information. Unadjusted intervals are descriptive and conditional on chosen models; they do not remove tuning optimism. Linear models may miss nonlinear interactions. Correlated predictors can exchange weights. Only 16 training refits and limited events support stability conclusions. The selection rule balances AP, warning count and size, not all possible operational costs. No feature importance establishes causality, tissue damage or biological onset. The remaining scientific tasks are endpoint adjudication and genuinely future chronological validation, not unfinished numerical stages of this consolidation.
"""
    (OUT/'summary.md').write_text(text)
    descriptions=[
      ('01_within_family_redundancy','Which measured representations repeat information?',f'{len(pairs)} high pairs; {clusters.correlation_cluster.nunique()} clusters; temporal roles separately protected.','Redundancy does not establish exchangeability.'),
      ('02_cross_family_redundancy','Do group names hide shared information?',f'{int((~pairs.same_group).sum())} high-correlation cross-family pairs.','Context, role and coverage can duplicate measured signal.'),
      ('03_family_ablation_and_permutation','Which families contribute beyond the rest?','Extension has the clearest but narrow unadjusted ablation interval; permutation reliance is broader.','Exploratory, correlated and model-dependent evidence.'),
      ('04_selection_and_sign_stability','Which coefficients survive different training pitchers?',f'{len(stable)} fields meet ≥90% selection and sign-agreement thresholds across 16 refits.','Stable coefficients need not imply incremental value.'),
      ('05_reduction_performance','Can a smaller set preserve ranking and advance warnings?',f'{len(keep)} fields selected; validation AP {selected.average_precision:.5f}; {int(selected.alerted_events)}/120 warned.','Validation-selected development candidate; no equivalence claim.'),
      ('06_coverage_sensitivity','Does support explain apparent gains?','Separate explicit support fields from automatic missingness indicators; compare paired AP and warnings.','Availability may proxy role, experience and era; not physiology.')]
    table(pd.DataFrame(descriptions,columns=['figure','question','result','interpretation']),'figure_summary')
    return text
