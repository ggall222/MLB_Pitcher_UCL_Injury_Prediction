# Methods: feature decomposition and trajectory shape

## Reuse and scope

This extension reads the existing analytical cohort through `src.injury_analysis.cohort.load`, which checks its source/config/code fingerprint. MLBAM keys and the original 1,225 matched pairs are preserved. A small read-only DuckDB projection adds existing measured-baseline counts, velocity-variability change and any requested existing trajectory field absent from the narrow cache. There is no pitch reaggregation, raw-data retrieval, source imputation, surgery-date substitution or database mutation.

New helpers are `signals_common.py` (paths, projection, shared statistics), `family_decomposition.py` (models/evaluation), `trajectory_shapes.py` (paired trajectories and shape), `signal_plots.py` (figures), and `signal_report.py` (computed synthesis). The existing notebook runner now accepts `--suite injury-signals`; its original default is preserved. New artifacts live separately under `reports/injury_signal_analysis` and `data/injury/analysis/signal_analysis`.

## Family experiment

The original 62 selected historical predictors are partitioned into workload, velocity, release mechanics, extension, spin/movement, command, pitch mix, and pitcher history/context. `family_definitions.csv` records every exact column. “History/context” includes role, calendar month, age, handedness, prior observed starts, elapsed time since debut and as-of prior reported IL count; it is not purely medical injury history.

Eight family-only specifications and ten combinations/full specifications are fixed in code. The combinations include workload+history (WH), velocity+mechanics, Statcast-only, WH plus individual Statcast families, WH plus velocity/mechanics, and Full. Models reuse fixed C=1 logistic regression and training-only median imputation, missingness indicators and scaling. All variants within each category share the same train/validation/test rows and targets. No new hyperparameter optimization or negative sampling occurs.

The original 2015–2021 / 2022–2023 / 2024–2025 chronology and 60-day embargo remain. The test period is already examined and therefore exploratory. Categories require 30/10/10 represented training/validation/test events. Source NULLs, immutable identifiers, original outcomes, and current-outing prediction time are unchanged. The category-versus-IL-free design excludes competing impending injury types and requires observed negative follow-up; it is not a head-to-head multiclass injury classifier or a complete-league deployment cohort.

## Paired model uncertainty and warning evaluation

AP and PR lift use 300 pitcher-cluster bootstrap draws. Within a category/partition, exactly the same pitcher draws evaluate all model variants. Therefore Full-minus-WH and incremental-family intervals are paired on the same rows; comparing overlap of two separate intervals is unnecessary. AP/lift intervals and differences condition on the fitted models and omit training-refit uncertainty. They are not multiplicity-adjusted across categories/specifications. Cross-category raw AP differences also reflect differing prevalence and cannot directly establish anatomical specificity.

The existing evaluation functions calculate alert budgets, precision, recall, F2, event recall, lead times and false/non-actionable burden. Warnings must precede the event by 7–30 days. All qualifying date-partition events, including those without usable warning outings, remain in the all-event denominator; observable-event recall is separately reported. Lead times condition on detection. A row-label-positive alert within six days is too late for this practical target and is included in non-actionable burden.

The exact random reference selects the same K alerts uniformly among N eligible rows. For an event with m warning opportunities, detection probability is `1 - C(N-m,K)/C(N,K)`. Average over all events, including m=0. The random curve is an expectation, not a 5%-event-recall baseline or a deployed policy. Event-level model-minus-WH/random intervals resample whole pitchers' events conditional on the fixed observed top-5% alert selections. They do not re-rank alerts inside each bootstrap, refit models, or model random-policy realization uncertainty.

Validation-fixed score thresholds are also saved, because a retrospectively fixed test alert fraction is not a threshold that was known before testing. The role-specific sensitivity evaluates frozen scores and reports AP differences; it does not invent starter/reliever classifications for events with no observed outing.

## Variable inspection

For four focal categories (elbow/UCL, shoulder, forearm, lower body), the top two individual-family validation point estimates identify variables for inspection. Those variables are permuted within validation season/current-role strata in the **full** model, with five repeats. The same unchanged target rows are scored. The output is conditional AP loss after perturbation, not a causal effect or an independent validation of selected families. Standard deviation across five shuffles describes perturbation variability, not sampling uncertainty. Role is fixed by the strata and its coefficient remains available rather than reporting a meaningless shuffled-role importance.

Correlated features can mask or distribute importance; row permutation also breaks some serial dependence. All standardized full-model coefficients are saved to make signs and missingness-indicator dependence inspectable. Neither importance method chooses a final model. Selected dynamic variables supplement the trajectory feature list transparently; `trajectory_feature_selection.csv` distinguishes the original scientific list from validation-driven additions.

## Separate modern arm-angle experiment

Training begins in 2020 with the original train/validation/test end dates. Require nonmissing arm-angle average, SD, change and z-score to create one common comparison cohort. Compare angle-only, Full without angle, and Full plus angle on exactly the same rows. The latter's paired AP difference uses Full without angle as reference. Retain the 30/10/10 event rule and report skipped categories; never silently pool an underpowered category.

The short 2020–2021 training history and complete-angle selection limit interpretation. Baselines themselves still use the original available strictly prior history. The modern model is not directly interchangeable with a model trained across 2015–2021, and missing historical angle is never filled with zero.

## Balanced trajectories

The expansion reuses the exact existing aligned game/pitcher keys. Every observation precedes its injury/pseudo-event anchor. For each feature, average within each event/kind/calendar bin, then require both the case and its matched control to be observed. The balanced curves additionally require the same event pair at all four calendar bins or all final five appearance positions. Per-feature attrition and the all-available-pair summaries remain exported.

Chronological calendar points are 31–60, 15–30, 8–14 and 1–7 days. Appearance points are fifth-last through last within the existing 60-day window; no interpolation creates missing outings. Case/control means are equally event-weighted, not weighted by the number of relief appearances in a bin. Pointwise case/control bands and simultaneous paired-difference bands are clearly distinguished. Counts accompany every curve and each exported point.

For simultaneous bands, resample case pitchers, retain the whole vector of time-point differences for every selected event, and use the 95th percentile of the maximum absolute standardized bootstrap deviation over those points. This controls the displayed time-point family within a feature/category, not all comparisons in the study. As in the original matching, a person can also be a control elsewhere; unique control pitchers prevent reuse but do not remove every source of cross-pair dependence.

## Shape and timing summaries

Shape metrics use one paired summary per event:

- Linear slope across the five appearance indices.
- Mean of final two minus mean of first two appearances.
- Final appearance minus mean of previous two appearances.
- Late slope minus early slope across two-appearance intervals; this is a discrete slope contrast, not an estimated biological acceleration.
- For z-score features only, maximum absolute departure and fraction of appearances with |z|≥1. This is a descriptive reference, not a clinical abnormality threshold.
- An actual-calendar-day slope per 10 days, requiring at least three distinct measured dates, an observation at least 30 days before the event, and one within 14 days. Case and control slopes use their own actual dates and are differenced; no evenly spaced calendar assumption is imposed.

Report bootstrap means/intervals, medians, 10th/90th percentiles and positive-change fractions. These expose heterogeneity instead of pretending the average is an individual trajectory. Raw workload/profile measures are kept in their original units; pitcher z-scores reference existing strictly prior baselines, not a new normalization fit using future observations.

“First sustained support” requires at least 30 balanced events and two adjacent same-direction simultaneous bands excluding zero. Isolated supported points are recorded separately. Neither is an onset estimate, and support already at the earliest observed point can precede the window. There is no invented clinical effect-size threshold. Sparse/irregular observations do not justify a forced change-point model. Actual calendar lead distributions for appearance positions are exported so an outing index cannot be mistaken for a fixed day lead.

## Sensitivity and connection to prediction

Trajectory checks address strict QA; exclusion of final three/seven days; matched index starter/relief role; first/prior reported IL episode; ≥15 measured baseline observations; explicitly documented throwing side; complete years through 2025; and the distinct validation/test eras. Restricted final-day analyses retain original anchors and outing indices. Same-pair contrasts separate removed late observations from loss of events lacking earlier observations. These restrictions change sample support and must not be read as randomized comparisons.

A direct arm-minus-non-arm velocity/mechanics/workload contrast resamples shared pitcher IDs across groups, rather than declaring a difference because one group's interval crosses zero and another's does not. It pools explicitly throwing-side arm regions versus trunk/lower-body regions; it is exploratory and not adjusted for all features.

Predictive sensitivities refit WH, WH+velocity/mechanics and Full under strict-QA positives or exclusion of final-six-day positives. Excluded positives are not relabeled healthy. Compare increments within the restricted cohort, since prevalence changes. The bridge table connects category-level AP increments with trajectory shape and count information; the populations are not identical, so it supports interpretation rather than mediation or a causal explanation.

The heterogeneity/profile figure examines a concrete alternative explanation when a family predicts without a clear average trajectory: stable pitcher baseline differences and a wide distribution of event-specific changes. The existing linear logistic framework does not establish nonlinear mechanisms, and the extension does not select a complex model to force that conclusion.

## Limits for a paper

Reported IL/effective dates remain proxies; historical source descriptions can be revised and contemporaneous publication timing is not verified. Labels can reflect already-existing symptoms. Injury and surgery are distinct. Return dates/durations never enter predictors. First recorded episode is not lifetime first injury; all-year trajectories include a partially observed 2026 season. The selected IL-free comparison pool does not prove absence of symptoms or represent every MLB outing. Multiple exploratory comparisons, sparse categories, nonrandom missingness, matched-cohort selection and previously examined test data remain material limitations.

The next confirmatory study should adjudicate timing, predeclare a small comparison against the simple workload/context baseline, use a practical lead-time target and an independent evaluation period. Do not describe the current decomposition as a final validated injury-warning model.
