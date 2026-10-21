# Supervised reduction methods

This stage tests whether outcome-directed directions preserve useful information with fewer continuous axes than PCA, using the existing injury pipeline. It does not rebuild features or source data.

## Sources and cohorts

The analysis reuses `src/injury_analysis/cohort.py`, canonical injury definitions, existing model features, predictor allowlists and original matched case/control identities. All database projections are read-only. The seven 30-day category risk sets include starters and relievers, observed-follow-up negatives, exclusions for competing/uncertain injuries and the existing 60-day embargo. Train through 2021, validation 2022–2023, exploratory test 2024–2025. Current 2026 data is not an additional fitted/test partition.

Compare original 62 features, 13 profile fields, eight departure z-scores and the combined 21 fields. Profile/departure definitions are identical to the preceding analysis. Workload is retained in the original-feature comparator. Arm angle is not included in the long-history feature space.

## Supervised projection

Fit training medians, training-observed missingness indicators and standard deviations. Continuous fields enter `sklearn.cross_decomposition.PLSRegression(scale=False)` with the binary injury outcome. PCA uses the same standardized continuous matrix. PLS is supervised only by training rows; validation/test targets never enter projection fitting.

Native PLS rotations need not be orthonormal. A nested QR basis spans the same first-k PLS directions, with unit Euclidean geometry. Project the standardized continuous fields onto that basis, pass binary context and missing flags through, and fit the existing L2 logistic C=1. No whitening or extra latent scaling changes the regularization penalty. This is a PLS-subspace classifier, not unbounded PLS regression probabilities.

Completing the basis to full rank preserves all original information. A numerical control compares its predictions to the original feature model, using the existing tight convergence tolerance. The maximum observed probability difference is about 0.000262; AP differences stay below 0.00001. Unit tests also check nested span preservation, train-only transforms, binary/indicator handling and test-label independence.

The [official scikit-learn cross-decomposition guide](https://scikit-learn.org/1.9/modules/cross_decomposition.html) describes the underlying estimator.

## Complexity and evaluation

Both methods share grids within each source block, augmented with the PCA-90% dimension learned in training. Maximum validation AP selects k, with fewer axes as the tie-break. Test performance is not used to select a method or dimension.

The descriptive near-peak count reaches 95% of the best validation AP gain above prevalence. A sustained version requires all later tested counts to remain above the threshold. These are not significance-based plateaus. A selected dimension at the grid's upper edge does not prove that the optimum has been reached.

Binary/missingness columns are not compressed. Current data retain 47 such columns for original features, 10 for profile, eight for departure and 18 for combined. Tables show both k and total design dimension. An availability/context-only diagnostic makes this retained information visible.

All selected models use the prior evaluation functions: AP, PR lift, ROC-AUC, probability loss, precision/recall/F2, event recall at controlled outing alert budgets, warning leads and false alerts. Warnings require 7–30 days; all eligible events stay in the denominator. The exact uniform-random alert expectation accounts for repeated opportunities. Retrospective top-k budgets are not deployment thresholds.

Paired 300-replicate evaluation-pitcher bootstrap intervals hold the model and selection fixed. The same resampling seed/settings as the existing signal analysis are required. They do not include component-selection uncertainty, training uncertainty or multiple-comparison correction.

## Interpretation and timing

Save native axis weights, injury-oriented weights, feature/score correlations and family sums of squared axis weights. Orientation uses training outcome covariance only. Neither coefficient energy nor latent coordinates represent causal injury-risk shares.

Compare new departure scores with the existing training-fitted descriptive departure PCs on the same category-specific training rows. The earlier PCA used a broader common training sample; the correlation therefore compares two frozen training maps, not identical fitting cohorts.

For trajectories, reuse the original 443 pairs whose case/control windows lie entirely in 2022–2025. Require common complete profile+departure fields and balanced pair support at each plotted point. No rematching, smoothing, invented observations or imputed physiology is used. Plot first-component scores in training-score SD units. The first component is not the whole selected model when k exceeds one. Bootstrap bands are simultaneous within each trajectory only.

## Stability and sensitivity

Twelve paired whole-pitcher training refits cover shoulder and lower body in original and combined inputs. Refit preprocessing, axes and classifier, holding the original validation-selected k fixed. Convert directions into the original training units before comparing first-axis absolute cosine and subspace overlap. Larger/full-rank subspaces naturally overlap more. Prediction-rank and alert-set stability are separate outcomes.

Focused shoulder checks refit after requiring complete profile/departure measurements, or excluding final-six-day positive rows without relabelling them negative. Both methods use the same filtered cohort and each selects k only in its validation subset. Original features may still contain structurally missing fields in the complete-P+D sensitivity; it deliberately targets the paired measurement block.

## Limits and reproducibility

PLS is linear and tests additive mixtures, not nonlinear interactions between profiles and departures. With a binary outcome, even a linear classifier ends in one scalar score; successful compression cannot establish an intrinsic biological dimension. Baseline profiles are recent rolling means, not immutable susceptibility. Injury dates remain recorded proxies. Short warning signals can coincide with reporting/diagnosis behavior.

See the [executed notebooks](../notebooks/injury_supervised_reduction/README.md), [computed findings](../reports/injury_supervised_reduction/SUMMARY.md) and [verification](../reports/injury_supervised_reduction/verification.json). New artifacts have separate directories; all previous reports and notebooks are fingerprinted and preserved. Source data revisions or different analysis settings require a fresh, appropriately versioned comparison.

