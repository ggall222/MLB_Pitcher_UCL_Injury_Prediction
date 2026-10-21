# Feature consolidation methods

## Source and preservation

This is an additive stage in `src/injury_analysis/consolidation_*.py`, configured by `config/feature_consolidation.yaml`. DuckDB is opened read-only. The appearance-grain cache joins the existing labeled cohort to the 536 registered fields in `model_features`; it does not scan or modify raw pitches. The data dictionary and actual `appearances`, `workload_features`, `mechanical_features`, and `pitcher_baselines` schemas are retained as provenance. The existing cohort contributes `calendar_month` and `prior_reported_il_count`.

The resume audit fully read all seven saved Parquets, parsed 19 CSVs and five JSONs, loaded 94 models, and checked finite coefficients and dimensions. The last completed step was the six-model Elastic Net grid; selected-model metadata was recovered without refitting. Original numerical files are preserved. A new inventory reconciles 30 source descriptions and eight timing descriptions against finalized builder rules; another table canonicalizes unordered family pairs. These are descriptive corrections, not changed correlations or feature-selection rules.

The final review inventory (`feature_inventory_review.csv`) additionally makes every pitch-type filter explicit and distinguishes Statcast pitch-type counts from the official aggregate appearance pitch count. `source_field_clarifications.csv` records these provenance annotations without modifying the frozen decision inventory. `feature_recommendation_details.csv` supplies per-field redundancy, sparse-support and sign-stability rationale; these annotations do not change the frozen feature list.

`prior_artifacts_snapshot.json` protects 591 earlier artifacts; `resume/resume_snapshot.json` protects 125 valid interrupted-run artifacts. Database/source size, mtime and hashes are checked independently. Writes use temporary files for new CSV/model checkpoints. The workflow skips complete stages; fitted models require exact configuration, input fingerprints, feature order and training/validation row-key signatures, including repeated bootstrap rows. It stops on a changed source signature rather than silently using old analysis with new source data.

## Cohort, timing and endpoint

Reuse the existing broad initial **Elbow / UCL** episode label, 30-day horizon, eligible MLB starter and relief appearances, known injury exclusions, competing/intervening IL policy, conservative negative follow-up and 60-day split embargo. Train through 2021; validation 2022–2023; already examined transfer period 2024–2025. 2026 is not used for new selection or claimed as an untouched confirmation set.

The primary outcome is not confirmed UCL-only injury. Broad elbow events have only 15/3/5 initially explicit UCL events represented in train/validation/test; later surgery corroboration is retained as outcome provenance. IL/event timing is an observable reporting proxy, not biological onset. Documented prior UCL procedure fields are excluded because source publication timing was not established. The existing count of strictly prior reported IL episodes retains its original source-publication assumptions.

Prediction occurs after a completed appearance. Current performance, mechanics and inclusive workload are available then. Slopes and prior means use strictly earlier dates; z-scores use the current aggregate relative to up to 20 prior outings (minimum five valid values for baseline mean/SD). Same-day other appearances and future dates cannot enter prior histories. Pitchers can appear in later chronological partitions; this evaluates future outings of known pitchers, not unseen-pitcher transfer. No identifier is a predictor. Grouped uncertainty uses pitchers, not independent appearance rows.

## Inventory and screening

Fifteen families follow baseball concepts and verified builders. Representation types separate current levels, historical levels, changes, slopes, z-scores, baseline means/SD/counts, workload windows, ratios, counts, usage, support and context. Generic data-dictionary descriptions are supplemented where suffix rules obscure precise windows. Calendar month is seasonal context, not persistent biological susceptibility; other profile/history fields are slowly changing rather than immutable.

Exclude constant/unobserved training fields, training missingness above 60%, fewer than 100 observed training pitchers, unverified prior surgery knowledge dates, unresolved classifications, and raw circular spin-axis coordinates requiring sine/cosine encoding. Do not zero-fill missing measurements. Near-zero variance is flagged at >99.5% modal fraction but not automatically removed. There are 353 feasible fields: 333 substantive and 20 explicit support fields. The latter are evaluated separately. Arm angle and within-appearance fastball decline are assessed on appropriate common-row modern/starter subsets rather than dismissed as intrinsically uninformative.

Sixty-three known algebraic relationships are evaluated on co-observed training rows, including counts/usage, workload totals, changes and z-scores. Exactness on co-observed rows does not mean identical missingness.

## Correlation and clustering

Draw up to 12 appearances per training pitcher, seed 20261008, without consulting outcomes (13,427 rows). Calculate pairwise Pearson and pairwise-observed Spearman with at least 250 co-observed measurements. Missing values remain missing. Cluster each family with complete linkage on `1 - abs(Spearman)`, cutting at 0.10. Unavailable correlations imply maximal distance rather than similarity. Complete linkage prevents weak-endpoint correlation chains from forming a single redundant cluster.

Representatives within cluster × temporal role prefer broadly measured baseline/z/current forms, five-outing history and lower training missingness; this ordering does not use outcomes. Adjacent historical levels and current states may correlate yet deserve separate tests. Full matrices, co-observation counts, all pairs and linkage trees are cached. Figures show readable representatives in hierarchical order, not an unreadable global matrix. Cross-family summaries count each unordered pair once.

## Model comparisons

All family and reduced-set comparisons use L2 logistic regression (C=1, unweighted classes, LBFGS, tolerance 1e-6). Training-only preprocessing learns 0.5%/99.5% clipping bounds, median imputation, missing indicators and standardization. Degenerate quantile bounds do not flatten rare binary values. These are model-input transformations; the source database remains nullable and unchanged. Missing flags are included in grouped permutation alongside their source fields.

Family ablation refits the remaining fields. Family-only models measure available marginal signal. Grouped permutation shuffles a family's combined logit contribution within season and role, except role is shuffled within season alone. This is marginal importance conditional only on those strata, not a full conditional permutation method; it breaks serial structure and can generate unusual predictor combinations. Five perturbation SDs are not confidence intervals. Correlation can make ablation and permutation disagree.

Nested concept comparisons use common contextual adjustment and add baseline → departure → slope → recent level → current state in fixed order. Workload and pitch-mix variations follow corresponding semantic sequences. These are order-dependent, exploratory contrasts, without multiple-testing correction. Modern arm-angle and starter fatigue sensitivities compare identical measured rows, preserving all qualifying events in the population warning denominator.

## Sparse screening and stability

Elastic Net logistic regression spans C={0.003,0.03,0.3} and l1_ratio={0.25,0.75}; SAGA tolerance 1e-4, maximum 5,000 iterations, with convergence failures raised. Six settings are selected by validation AP, then fewer nonzero fields and stronger regularization for ties. All 333 substantive fields remain available, so correlation alternatives can exchange weight.

In this sparse model only, binary measurements and missing flags remain on raw unit scale to prevent rare standardized flags from dominating SAGA's step size; continuous features retain training standardization. This changes penalty geometry relative to L2 and is not interpreted as a pure algorithm comparison.

Sixteen whole-pitcher bootstrap training refits (seeds 20261008 through 20261023) keep chosen C=0.03/l1_ratio=0.25 fixed. Every appearance of each sampled pitcher is included with the same multiplicity. Selection uses |coefficient|>1e-5, with measured and missing-flag selection separated. Report sign agreement conditional on selection, top-decile frequency, ranks, coefficient variation, and one fixed-seed per-feature permutation in each refit. Permutation across refits is a perturbation diagnostic, not a causal or multiplicity-adjusted test. Cluster-any-member selection accounts for correlated substitution. Sixteen refits give 6.25-percentage-point frequency resolution; no formal stability-selection error bound is claimed.

## Candidate selection and freezing

Candidates: full feasible space, previous 62 fields, correlation representatives, prior profile and departure blocks, their combination, background context, Elastic Net nonzero values, and cluster × temporal-role representatives at 90%,75%,50%,25% selection frequency. Stability representatives rank by selection frequency then median absolute coefficient within the cluster. Identical feature sets are deduplicated; existing identical fits are reused.

The validation rule was specified before candidate fits: identify highest AP; accept AP loss no larger than one paired-bootstrap SE; among those, accept warning count within two events of the best at a 5% alert budget; choose smallest input set, then AP. The tolerance is pragmatic and can be wide when event support is limited. It does not prove equivalence, and selection-period estimates are optimistic. The eight-field departure block is the selected warning-focused candidate; 62 stable representatives are another documented validation tradeoff. Drop-one and correlated-alternative add/replace comparisons diagnose the selected model but do not trigger additional test-informed pruning.

Coverage sensitivity separately removes automatic missing flags and adds the 20 explicit support fields; season/role missingness and prediction summaries accompany AP and warning metrics. Source values are not changed. The feature list, fitted model, preprocessing and decision-table hashes are frozen before this stage generates 2024–2025 predictions. That period was already examined in earlier analyses, so the result is exploratory transfer, not independent validation. The previous 62-field model remains a comparator; the compact candidate is not a validated replacement.

## Evaluation and uncertainty

Reuse AP, PR lift, ROC-AUC, Brier, mean prediction bias, validation-edge calibration bins, and alert budgets of 1%,2.5%,5%,10%. Practical warning requires an alert 7–30 days before the recorded event. Denominators include all qualifying events, with observable events also reported. Median lead time is conditional on detection. False alerts not followed by an event within 30 days differ from non-actionable alerts under the narrower warning window.

AP and pairwise differences use 300 whole-validation-pitcher bootstrap draws with the existing analysis seed 20260919. Event intervals use the inherited whole-pitcher event framework, including no-opportunity events. These conditional intervals do not refit models or repeat selection and do not remove tuning optimism. Training-refit stability and validation sampling uncertainty are different analyses.

Top-budget ranking retrospectively uses the period's score distribution. A separate validation-derived numeric 95th-percentile threshold is transferred unchanged, reporting actual alert-rate drift. Neither thresholds nor features are selected from test outcomes. Freeze a prospective threshold policy and adjudicated UCL endpoint before a future confirmatory cohort.

## Reproduce or resume

```bash
.venv/bin/python -m src.injury_analysis.consolidation_workflow
.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite feature-consolidation
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_feature_consolidation.py
```

The workflow supports `--until stability` or `--until models`. Complete stages are reused; source fingerprint changes stop execution for review. New reports are in `reports/feature_consolidation/`, models/cache in `data/injury/analysis/feature_consolidation/`, and the candidate specification is `config/ucl_candidate_features_v2.yaml`. Notebook construction is separate (`scripts/build_feature_consolidation_notebooks.py`) and preserves existing notebooks.
