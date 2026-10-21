# Methods: dimensionality reduction and injury signals

## Integration and data preservation

This phase extends `src.injury_analysis` and the shared notebook runner. It reuses `cohort.load`, `label_frame`, original matched controls/aligned outing keys, pitcher-cluster bootstrap helpers, and the existing evaluation functions. It does not acquire new raw data, change injury taxonomy or labels, recompute historical features, or write to the central database. A narrow read-only projection adds eight existing baseline means where absent from the appearance cache. Outputs live under `reports/injury_dimensionality_reduction` and `data/injury/analysis/dimensionality_reduction`.

`reduction.py` handles input projection, PCA encoding, structure diagnostics and provenance. `reduction_trajectories.py` applies frozen maps to matches. `reduction_prediction.py` fits and evaluates reduced models. Plot/report helpers and three focused notebooks preserve the existing architecture.

## Why PCA

PCA provides deterministic transformations for unseen observations, explicit contributions from original variables, and a direct compression experiment for correlated engineered features. Components maximize variance, not separation between injury labels. The analysis uses standardized numeric inputs, deterministic full SVD and no whitening. A full-rank rotation preserves the original linear L2-regularized problem and serves as a numerical control. Nonlinear visualization is not added solely to make clusters appear; the current questions require interpretable loadings and temporal out-of-training comparisons. This choice does not establish that the underlying structure is linear.

Implementation references: [scikit-learn 1.6 PCA](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html) and [training-only preprocessing guidance](https://scikit-learn.org/1.6/common_pitfalls.html).

## Three descriptive representations

The original selected predictors contain 60 continuous and two binary fields (current starter status and throwing hand). The continuous space includes workload, history, pitch characteristics, levels and changes. Binary values and missing flags are excluded from its PCA axes.

The profile representation contains strictly prior baseline means for fastball velocity, release X/Z, release X/Z SD, extension, spin and zone percentage. The departure representation contains the corresponding existing current pitcher z-scores. They use the source pipeline's prior observations; no whole-career mean or future appearance is introduced. The two blocks share the same underlying measurements, which supports interpretation, but do not mathematically partition all original features or all injury risk. Arm angle is excluded from this long-history experiment because of structural coverage changes.

A common label-free set of analysis-eligible training appearances through 2021 fits descriptive median imputation, StandardScaler and PCA. All later data use those frozen transforms. Scores plotted in trajectories are divided by the training PC standard deviation; predictive scores are not rescaled after PCA. A latent trajectory is shown only when every input in that eight-measurement block is observed. NULLs remain unchanged in source data.

Saved diagnostics include cumulative explained variance; 80/90/95% component counts; participation rank `1 / sum(explained_fraction ** 2)`; high pairwise correlations; projection weights; feature/score correlations; training missingness; and component associations with role/hand. Participation rank is a variance-concentration summary, not a physiological intrinsic dimension estimate.

Between-pitcher variation is the share of total score sum of squares explained by pitcher-specific training means. It is a descriptive, appearance-weighted statistic, not a random-effects ICC. Profile means are smooth by construction, so a high between-pitcher fraction does not independently validate susceptibility. Sensitivities cap training contributions at 20 deterministic sampled outings per pitcher, restrict to complete measurements, or use early/recent training years. Shared full-training scaling isolates differences in PCA bases. Mean squared cosines of principal angles compare subspaces without sign/axis-rotation artifacts.

## Matched latent trajectories

Case/control identities and selected appearances come from the original 1,225 pairs. Primary results restrict to pairs whose entire observed windows and injury anchors lie in 2022–2025, after PCA fitting. All-year trajectories are exported separately and include partial 2026. This primary restriction avoids interpreting training-space fit as held-out anatomical separation; it does not produce a new untouched confirmatory period.

Calendar bins are 31–60, 15–30, 8–14 and 1–7 days. Appearance positions are the final five outings within the original 60-day window. Values are averaged within event/kind/bin before pairing, so relief pitchers with more outings do not automatically receive more weight. Balanced curves require both case and control and the same event set at every point for a component. Counts and all-available-pair summaries remain explicit. Different blocks can have different complete-measurement support.

Case-minus-control differences are summarized per event, including average separation, first-to-last change, linear appearance slope, final-two minus first-two change, and last-outing minus previous-two change. Whole case pitchers are resampled, keeping their episodes together. Bands use the original max-standardized-deviation bootstrap across time within a component/category; they do not control all exploratory comparisons. A control can also appear elsewhere as a case, leaving some cross-pair dependence.

The latent-space figure plots one matched event difference per point. Arrows show mean movement from fifth-last to last outing. It is not a clustering or classification accuracy plot. The primary plots show the first three axes by training variance, with no target-based axis selection. For an interpretable connection to the previous velocity result, the summary also reports departure PC2, which has its largest weight on velocity in the executed training fit. All six axes remain available, including null results.

Frozen-map sensitivities address same throwing hand, strict QA, starter/reliever role, and removing the final seven days. These preserve original anchors and appearance indices; missing earlier observations remain absent rather than reindexed as the last outing. Recorded dates remain injury/IL proxies rather than biological onset. Current measurements become observable after the outing; surgery and return dates never enter the features.

## Predictive experiment

All models use the original seven category-versus-IL-free comparisons at 30 days, training through 2021, validation in 2022–2023 and testing in 2024–2025, retaining the 60-day embargo. Competing/uncertain impending injury observations remain excluded according to the existing cohort rules. All variants within a category use identical rows. Minimum represented-event requirements remain 30 training, 10 validation and 10 testing. Negative follow-up and source limitations are unchanged.

Predeclared specifications are original 62 features, workload/history, global PCA with 5/10/20 components or a training-learned 90%-variance cutoff, full-rank rotation, workload/history plus 10-component or 90%-variance Statcast, and role/hand plus missingness only. Every category gets its own training-fitted median, scale and PCA. Logistic C=1 and L2 penalty are retained. Solver tolerance is tightened to 1e-9 consistently for all new models after full-rank rotation exposed default-tolerance discrepancies. The earlier fitted results are preserved; numerical differences from them are reported explicitly.

PCA rotates continuous columns only. Original binary columns, uncompressed workload/history in the hybrid designs, and training-observed missing flags bypass PCA while keeping their training scaling. Thus a five-PC model still has more than five predictors. The role/hand-plus-missingness diagnostic discards all continuous values but keeps those explicit contextual fields/flags. It is not a pure physiological signal or pure missingness model.

One global PCA specification per category is chosen by validation AP alone, with smaller total design as tie-break. The test period does not select components. AP and PR lift, paired AP differences against original and workload/history, event warning, lead times and alert burden are saved for all specifications. The full-rank unwhitened control should reproduce original-feature predictions within numerical tolerance; whitening would change the regularization geometry and is deliberately avoided.

Paired AP intervals resample whole test/validation pitchers, using the same draws for all models. They condition on fitted models and do not include validation selection or refitting uncertainty. Warning recall requires an alert 7–30 days before the event; every qualifying event stays in the denominator, including those lacking an observed warning opportunity. The random reference accounts for repeated opportunities per event. Event-recall difference intervals condition on the observed top-5% selection; score thresholds fixed from validation are also exported because a retrospective alert budget is not a known deployment threshold.

## Stability, concentration and generalization

For four focal categories, 12 paired whole-pitcher training bootstrap refits relearn every preprocessing/PCA/model step. The selected specification remains fixed, while a 90% variance rule relearns the retained count in each resampled training set. Scores on the same fixed test cohort are compared with their corresponding unperturbed fit using rank correlation, average absolute score change and top-5% alert-set Jaccard overlap. AP SD across refits is a perturbation diagnostic, not a confidence interval for generalization. Coefficients are mapped back to original standardized feature/indicator coordinates before assessing stability, avoiding arbitrary component sign changes.

Frozen-score subgroup checks include starters, relievers, pitchers seen/unseen in training and exclusion of final-six-day positive observations. Late positives are excluded, never relabeled healthy. Leave-one-positive-pitcher-out AP differences identify concentration in a few injured pitchers without refitting or model reselection. The small resampling study and multiple comparisons remain exploratory.

## Reproducibility and interpretation

Run `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite injury-reduction` from the project root. The three notebooks and source generator are described in `notebooks/injury_dimensionality_reduction/README.md`. Rerun the full suite after changing configuration, code or inputs; update upstream analyses first when their source fingerprint changes. Manifests retain source fingerprints, package versions, notebook execution status and code hashes. Tests cover frozen train-only transforms, full-rotation equivalence, missing-flag routing, variance-cutoff fitting, subspace invariance and validation-only dimension choice.

The analysis can reveal profile differences, late changes, redundancy and prediction/stability tradeoffs. It does not establish natural injury clusters, causal susceptibility factors, biological onset or a clinically validated warning rule. Source revisions, uncertain contemporaneous reporting, active-pitcher selection, nonrandom missingness, sparse injury categories, changing tracking eras and previously examined test data remain material limits.
