# UCL-specific endpoint and dynamic-warning methods

## Source and endpoint

Reuse the canonical general-injury episodes, official MLB transaction links and existing identifier-based legacy UCL/procedure comparison. Original source fields and every exclusion flag are retained in the new adjudication table. No name-based match is added. This stage does not claim new contemporaneous news verification.

Primary eligibility requires reliable existing episode timing and one of:

- Official UCL diagnosis text attached to the episode, excluding procedure/recovery-only text as independent confirmation of a new event.
- A previously linked MLB-level UCL procedure with compatible initial anatomy, a surgery interval wholly within 180 days after the episode reference, before activation/episode boundary, and no observed intervening MLB return. A procedure cannot support multiple primary episodes.

Unknown diagnoses, broader elbow inflammation/sprains without UCL evidence, postoperative entries, overlapping/continuation episodes, incompatible laterality, unresolved anatomy/identity/timing conflicts, and incomplete sources are excluded from the primary endpoint. Initial forearm/flexor descriptions followed by compatible UCL evidence are explicitly allowed; original anatomy is never overwritten. `confirmed` means documented under these operational rules, not clinical confirmation. Public UCL text that only describes surgery is probable unless a bounded procedure independently validates its episode. Strict official and expanded-probable definitions are sensitivity outcomes.

The prediction reference uses the existing earliest defensible canonical injury/retroactive/effective/report proxy. It is not biological onset. `first_public_indication_date`, original transaction/effective/retroactive dates, `diagnosis_date`, procedure intervals, `ucl_confirmation_reported_date`, activation and source retrieval timestamps remain distinct. Supplemental confirmation learned later affects retrospective outcome adjudication only. None of these future fields enters the predictor list. Historical publication archives were not independently verified; transaction dates may reflect retrospectively updated records.

## Risk set and follow-up

The grain remains one official MLB pitcher (including relievers/two-way players under the existing identity rules) × regular-season appearance. Existing quality, at least five prior outings, chronology, and active-episode exclusions are reused. Predictors are available after the completed outing. Unresolved absence is not a healthy negative; reentry follows the existing valid activation/observed MLB return logic.

The next chronological canonical episode is evaluated. A documented target UCL event within the horizon is positive. A known non-UCL first injury is a competing negative for the first-event cumulative-risk question. Ambiguous elbow/forearm/unknown conditions and excluded possible UCL episodes censor the row when they occur within the horizon. Ordinary negatives require an observed MLB appearance on/after the horizon within the existing 30-day follow-up grace, before source observation ends. No appearance after an intervening event can retrospectively make that event disappear. Censoring known competitors is a separately refit sensitivity.

Training is 2015–2021, validation 2022–2023, exploratory transfer 2024–2025. The existing 60-day end-of-partition embargo and episode partition checks are preserved. Earlier historical observations remain available for baselines. Pitchers can recur across chronological partitions, as in prior longitudinal analyses; this is not an unseen-pitcher experiment. Whole-pitcher resampling and a clearly labeled unseen-pitcher sensitivity address dependence and limited generalization. 2026 receives no model performance/selection evaluation.

## Predictors and fitting

Frozen hypotheses: background-7; profile-13; departure-8; profile+departure-21; profile+departure+trajectory-33; that block plus three interactions; previous-62; stable-representative-62. The two 62-field sets are different. Exact ordered fields are in `feature_representations.csv`, `representations.json` and the freeze. No renewed feature selection is performed.

Departure trajectories use velocity z, release-Z-variability z, extension z and spin z. Each has a slope per observed outing, mean absolute z and proportion with absolute z ≥1 across the current outing and three earlier quality-valid outings within 30 days. All four values must exist; otherwise the derived field is missing. Earlier-day selection excludes other same-day games. This is persistence of magnitude, not necessarily same-direction worsening. Slopes retain direction. Sparse or irregular histories do not justify formal change points.

Interactions multiply training-standardized profile velocity×velocity departure, profile release-Z variability×its departure, and profile extension×extension departure; both main effects remain. Product centering/scaling is training-only. Logistic regression uses fixed L2 C=1, no class rebalancing and no new hyperparameter search. Existing train-only 0.5%/99.5% clipping, median imputation, missingness indicators and scaling are reused. Source NULLs are unchanged. Thus encoded dimension can exceed raw feature count. Separate horizon models need not yield monotone cross-horizon probabilities; they are landmark comparisons, not a jointly fitted survival distribution.

## Selection, warning and uncertainty

Primary horizon: 30 days. The predeclared candidate rule maximizes validation 7–30d all-event recall at a 5% appearance budget, requiring AP above prevalence; ties prefer fewer raw features then higher AP. The full protocol, signatures and validation 95th-percentile score thresholds are written before any new transfer predictions. The selected model is not declared statistically superior.

Report AP, PR lift, ROC-AUC, Brier score and calibration tables. Horizon-specific follow-up changes prevalence and risk-set membership, so AP across horizons is not a same-sample contest. Calibration bins use validation score quantiles and unchanged boundaries in transfer; no validation/test outcome calibration is fit.

Warning credit requires that the target be the appearance's next episode and be at least 7/14/21 days ahead, within the target horizon. No skipping of competing injuries. All eligible events are in the principal denominator, including unobservable off-season/short-history events; opportunity-conditional recall is separately labeled. First-warning lead is the largest lead among qualifying alerts, conditional on detected events. Each alert counts; no silent cooldown. Event-level uncertainty resamples whole pitchers while holding scores/thresholds fixed. AP bootstrap uses the existing paired whole-pitcher utility; eight training refits assess fit instability separately. These are descriptive uncertainty intervals, without multiplicity-adjusted hypothesis claims.

Top-floor(rate×N) budgets with immutable-key tie breaking are retrospective ranking comparisons. Frozen numeric validation thresholds are separately applied to transfer. Labeled performance excludes unknown outcomes. The operational burden table also applies those thresholds to all prospectively eligible appearances, retains unknown alert outcomes, and measures two-consecutive actual-outing alerts with a ≤30-day gap. The simpler warning table's repeat fraction describes consecutive evaluated appearances; it is not the same as consecutive actual outings. Empirical bootstrap intervals can degenerate at zero when no event was detected; this does not establish a zero population uncertainty bound.

## Matching and trajectory uncertainty

Keep existing controls for newly adjudicated eligible events. Only events without a prior match receive a deterministic new match using existing season/month, role, five-year age band and experience band rules, a different control pitcher and no reused control pitcher. Controls require observed follow-up and distance from other injury episodes. Control windows stop at their matched index outing; no future outings are included. Cases cannot borrow appearances before an intervening injury.

Require ≥3 eligible outings in 60 days and a final outing within 14 days. Each plotted series additionally uses the same complete event support across all five outings, or across all four calendar bins, in both case and control. Events receive equal weight. Connected sets sharing a case or control pitcher form bootstrap units. Plots use point estimates and 95% simultaneous time bands; shape tables report last-minus-first differences and uncertainty. No smoothing or phenotype clustering is imposed. Controls are selected retrospectively healthy for comparison, not a deployable risk population. Average trajectory separation is not an individual-warning guarantee.

## Sensitivities and limits

Refit training-only models for competing-event censoring, probable-inclusive endpoints and public-date reference. Public-date sensitivity conservatively keeps the original earliest-date active-episode exclusions; it does not reinstate post-retroactive appearances. Strict official-only, exclusion of final six days, full trajectory support, starter/reliever and unseen-pitcher evaluations apply frozen primary models. Strict-event results are too sparse for a new tuned model. Subgroup recall denominators explicitly use subgroup observable events and cannot be compared directly to all-event primary recall.

IL episodes miss non-IL UCL problems; event timing is uncertain; surgery-linked ascertainment favors procedures; repeated episodes and vague descriptions remain imperfect; winter events often have no eligible warning opportunity; injury/roster observation is not complete medical surveillance. Current results cannot establish tissue failure, causal mechanisms or clinical readiness. A truly future cohort needs contemporaneous archived sources and mature follow-up after the recorded freeze.
