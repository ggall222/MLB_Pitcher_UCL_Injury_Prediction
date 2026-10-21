"""Computed compression tables and interpretation for the supervised experiment."""
import numpy as np
import pandas as pd
from .supervised_reduction import OUT,read_table,table,save_manifest


def compression_summary():
    selected=read_table('validation_selection')
    curves=read_table('component_curves')
    metrics=read_table('selected_performance')
    rows=[]
    for r in selected.itertuples():
        c=curves.loc[curves.category.eq(r.category)&curves.block.eq(r.block)&curves.method.eq(r.method)&curves.partition.eq('validation')].sort_values('components')
        eligible=[int(row.components) for row in c.itertuples() if c.loc[c.components.ge(row.components),'average_precision'].ge(r.plateau_cutoff_ap).all()]
        values=dict(r._asdict());values.pop('Index',None)
        values['sustained_95pct_gain_components']=min(eligible) if eligible else np.nan
        for part in ['validation','test']:
            f=metrics.loc[metrics.category.eq(r.category)&metrics.block.eq(r.block)&metrics.partition.eq(part)].set_index('variant')
            values[part+'_ap']=f.loc[r.method+' selected','average_precision']
            values[part+'_unreduced_ap']=f.loc['Unreduced','average_precision']
            values[part+'_passthrough_ap']=f.loc['Context/availability only','average_precision']
            values[part+'_prevalence']=f.loc['Unreduced','prevalence']
        rows.append(values)
    return table(pd.DataFrame(rows),'compression_summary')


def write_synthesis():
    summary=compression_summary()
    m=read_table('selected_performance');d=read_table('selected_differences')
    main=m.loc[m.block.eq('Original 62')]
    delta=d.loc[d.block.eq('Original 62')&d.variant.eq('PLS selected')]
    test=delta.loc[delta.partition.eq('test')]
    raw=test.loc[test.reference.eq('Unreduced')];pca=test.loc[test.reference.eq('PCA selected')]
    choices=summary.loc[summary.block.eq('Original 62')].set_index(['category','method'])
    values=main.loc[main.partition.eq('test')].set_index(['category','variant'])
    lines=['# Supervised dimensionality reduction: findings','',
        'Supervision finds compact predictive directions in the **departure and combined profile/departure blocks**, '
        'but it does not establish a small universal injury signature. In the full feature representation, PLS usually preserves '
        'the original model’s performance and sometimes improves over PCA. It does not demonstrate a consistent new performance '
        'advantage over the original features.',
        '',
        '## Comparison and model selection','',
        'PLS learns continuous-feature directions from the binary 30-day injury outcome in **training only**. '
        'An orthonormal basis for each nested PLS span feeds the same L2 logistic classifier as the original and PCA models. '
        'This removes arbitrary latent scaling from the L2 comparison. Binary context and training-observed missingness flags '
        'pass through unchanged for both projection methods. All imputers and scales fit on training only.',
        '',
        'Train through 2021; validation 2022–2023; previously examined test 2024–2025. The original 60-day embargo, category risk '
        'sets, all-event warning denominators and case/control matching are retained. Candidate dimensions are identical for '
        'PLS and PCA within a block, including the training-learned PCA-90% count. Maximum validation AP selects k; test outcomes '
        'never choose it. A separate full-span numerical control checks the projection/regularization implementation.',
        '',
        '| Injury group | PCA k | PLS k | Original test AP | PCA test AP | PLS test AP |',
        '|---|---:|---:|---:|---:|---:|']
    for category in main.category.unique():
        lines.append(f'| {category} | {int(choices.loc[(category,"PCA"),"selected_components"])} | '
            f'{int(choices.loc[(category,"PLS"),"selected_components"])} | '
            f'{values.loc[(category,"Unreduced"),"average_precision"]:.4f} | '
            f'{values.loc[(category,"PCA selected"),"average_precision"]:.4f} | '
            f'{values.loc[(category,"PLS selected"),"average_precision"]:.4f} |')
    lines+=['',
        f'PLS test AP is higher than the unreduced comparator in {int(raw.ap_difference.gt(0).sum())}/7 groups by point estimate, '
        f'but **{int(raw.ap_low.gt(0).sum())}/7 paired 95% intervals exclude zero on the positive side**. '
        f'Against validation-selected PCA, PLS is higher in {int(pca.ap_difference.gt(0).sum())}/7 groups, '
        f'with positive intervals in {int(pca.ap_low.gt(0).sum())}/7: elbow/UCL, shoulder and lower body. '
        'Shoulder and lower body also have positive PLS-versus-PCA AP intervals in validation. Hand/wrist and other-arm '
        'test comparisons favor PCA; thus supervision is not uniformly better.',
        '',
        '## Is the injury-related information more compressible?','',
        'In the full 60-continuous-feature space, selected PLS counts range from 3 to 30. The shoulder model selects 30 '
        '(20 first reaches 95% of its best validation AP gain over prevalence); lower body selects 30. '
        'Trunk/neck selects 3, but its test AP is lower than the original model. These patterns do not support a universal '
        'one- or two-component injury representation.',
        '',
        'Compression is clearer when level and departure representations are separated. Shoulder departure selects **1 of 8** '
        'continuous axes, versus 7 for PCA, with test AP **0.0435 versus 0.0431** for unreduced departures. '
        'Shoulder combined selects **2 of 20** continuous axes, versus 20 for PCA, with test AP **0.0529 versus 0.0518** '
        'for the unreduced combined model. Those AP increments are uncertain; the main finding is preservation with compression.',
        '',
        '**Latent count is not total model size.** The original representation retains 47 binary/missingness columns outside '
        'the projection: a 30-axis model has 77 design columns, compared with 107 in the unreduced model. '
        'The one-axis shoulder departure model still has 9 total columns; the two-axis combined model has 20. '
        'Context/availability-only comparisons and complete-measurement checks are saved. A binary linear classifier can ultimately '
        'produce a single decision score; compact prediction is not proof of a low-dimensional biological injury manifold.',
        '',
        'The component curves and compression table report both the first tested count reaching 95% of the best validation '
        '**AP gain above prevalence** and whether all later tested counts remain above that threshold. This is a descriptive '
        'near-peak criterion, not a statistical plateau or an additional test-selected tuning rule. Curves can be irregular.',
        '',
        '## What does supervision prioritize?','',
        'The first shoulder departure direction puts about **88% of its squared coefficient weight on velocity z-score**, '
        'with smaller release-Z, spin and zone-percentage contributions. Its training-score correlations with the earlier '
        'departure PCA components 2 and 3 have absolute values **0.60 and 0.52**. This reinforces part of the earlier shoulder '
        'pattern while showing that the supervised direction is primarily velocity departure, rather than a new complex '
        'physiological construct. Signs are oriented toward higher training injury outcome only for interpretation; native '
        'weights and correlations are retained.',
        '',
        'In the first combined direction, squared weights are approximately **50% departure / 50% prior profile for shoulder**, '
        '**46% / 54% for elbow/UCL**, and **5% / 95% for lower body**. These are coordinate-weight summaries, not percentages '
        'of predictive performance or causal injury risk. PLS remains linear; mixing blocks does not test nonlinear interactions.',
        '',
        'For shoulder, about **95%** of the first supervised profile score’s test variance is between pitchers, versus '
        '**10%** for the departure score and **70%** for the combined score. This preserves the distinction from the previous '
        'PCA/profile analysis: stable profile associations and within-pitcher variation are different inputs, and the combined '
        'predictor uses both.',
        '',
        '## Timing, warning performance and robustness','',
        'The original out-of-training matches are reused. Common complete P+D support leaves 68 balanced shoulder pairs across '
        'five outings and 38 across all calendar bins. Shoulder departure PLS1 shows case/control separation, especially near '
        'the recorded event. Its last-minus-first separation change is about **+0.44 training-score SD** across outings, '
        'but the interval **−0.01 to +0.98** includes zero. Calendar change is also uncertain. Supervision therefore does not '
        'establish an earlier onset or a reliably earlier warning window than the preceding analyses.',
        '',
        'Better AP need not improve the practical warning metric. For the full shoulder representation at a 5% outing alert '
        'budget, test 7–30-day event recall is **18.0% PLS, 23.0% PCA, 17.4% original**. PLS’s difference versus PCA is about '
        '**−5.0 percentage points** (paired interval approximately −9.8 to −0.3). Median warning lead among detected events '
        'is 16 days for PLS versus 21 for PCA; differing detected-event sets prevent treating that as a paired timing effect.',
        '',
        'The shoulder complete-measurement sensitivity largely preserves the primary comparisons. Excluding final-six-day '
        'positive rows makes the full-representation PLS-versus-PCA AP gain small and uncertain. Some reduced-block test gains '
        'remain, but their validation counterparts are not consistently supported. These are targeted checks of the near-event '
        'concern raised by the prior analysis, not new independent confirmations.',
        '',
        'Across 12 paired whole-pitcher training refits for shoulder and lower body, PLS leading directions are less stable than '
        'PCA directions. Shoulder first-axis absolute cosine is about 0.90 in the full block and 0.89 in the combined block, '
        'versus about 0.999 and 0.992 for PCA. Full-block PLS score ranks and alert sets are also less stable than PCA. '
        'Combined-block prediction stability is mixed. Report loading stability alongside performance; full-rank PCA subspaces '
        'are trivially stable, and subspace overlap is not directly comparable across different retained dimensions.',
        '',
        '## Research interpretation','',
        'The evidence supports **category-specific, partly compact predictive structure**, particularly in within-pitcher '
        'departure measurements. It does not support a single compact injury mechanism or broad predictive improvement over '
        'the existing original-feature models. Shoulder contains the clearest departure direction; lower-body representation '
        'is much more profile dominated. Supervision helps choose useful combinations of measurements, but it does not convert '
        'associations with recorded IL/injury dates into evidence of biological susceptibility or short-term injury onset.',
        '',
        'A sensible next step is to retain the one-axis shoulder departure and two-axis combined models as compact comparators, '
        'alongside original features and PCA, and freeze an explicit ≥7-day-warning evaluation in a new chronological period. '
        'Choose any alert-oriented tuning rule in training/validation before that evaluation. Do not replace the original pipeline '
        'on the basis of these exploratory results.',
        '',
        '## Reproducibility and limits','',
        'The database, canonical injury events and every preceding report/notebook are preserved. '
        'The new [notebooks](../../notebooks/injury_supervised_reduction/README.md), [tables](tables/) and [PNG/PDF figures](figures/) '
        'include all categories, candidate curves, selected dimensions, paired AP/Brier/event differences, warning leads, '
        'loadings, PCA correlations, matched support, sensitivity results and training-refit diagnostics.',
        '',
        'Evaluation bootstrap intervals condition on fitted models and selected dimensions; they do not include tuning '
        'uncertainty or adjust across every reported comparison. Validation selection makes validation performance optimistic. '
        '2024–2025 has been examined repeatedly, and the small injury groups remain uncertain. Calibration/Brier values refer '
        'to the selected observed-follow-up cohort, not all MLB outings. Current measurements are available after a completed '
        'appearance. The “profile” means use at most 20 prior outings and are not immutable traits. Matching can retain '
        'cross-pair dependence when pitchers participate in more than one role.',
        '',
        'PLS is implemented with the existing scikit-learn dependency; see the '
        '[official cross-decomposition guide](https://scikit-learn.org/1.9/modules/cross_decomposition.html). '
        'The implementation projects onto orthonormalized nested PLS spans and then uses logistic probabilities, rather than '
        'treating raw PLS regression outputs as calibrated probabilities.',
        '']
    text='\n'.join(lines);(OUT/'SUMMARY.md').write_text(text);save_manifest()
    return text

