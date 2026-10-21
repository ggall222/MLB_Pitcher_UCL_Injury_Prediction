# Methods for the injury-type notebook workflow

## Source reuse and data grain

`model_features` supplies existing appearance aggregates, workload, prior rolling windows and pitcher-relative baselines. `pitchers.position` identifies official P/TWP identities, including relief and starting appearances. Current metadata may not capture every historical position change. Only regular-season rows are used. `data/injury/processed/mlb_pitcher_injury_events_2015_2026.parquet` supplies canonical IL episodes. MLBAM `player_id` joins directly to `pitcher_id`; names are never join keys. Source schemas, feature selection, source hashes and database size/mtime are recorded. The database is opened with `read_only=True` and never initialized or migrated.

The overview includes every episode and its original detailed fields. Broad analysis groups are elbow/UCL, shoulder, forearm, other arm (upper arm/lat), trunk/neck, lower body, and hand/wrist. Illness, Other and Unknown remain visible but are not prediction targets. Upper-arm region does not automatically mean throwing-side injury; explicit injured laterality is compared with throwing hand in a separate comparison.

## Primary event cohort

Targets require a known initial date and one of the seven groups. Exclude left-censored placements, uncertain timing, overlapping episodes, multi-region ambiguity, postoperative/pre-existing surgery descriptions, inferred longer-IL continuations, anatomy-source conflicts, identity-review flags, and incomplete transaction coverage. Exclusion counts can overlap.

Initial recorded anatomy controls the target. Later UCL or surgery information never silently changes an initially known forearm complaint into early known UCL disease. The event anchor is the earlier of the event-date proxy and reported placement date. Outings on or after this date are excluded from that event's pre-injury window. Missing-date/orphan events still block purported healthy intervals using the reported transaction date, without becoming positive targets.

Return-date QA and long duration alone **do not** exclude primary targets. Requiring complete short-term recovery would select against severe injuries. A strict-QA sensitivity retains common negatives and only positives with no canonical QA flags; its changed prevalence makes it a sensitivity analysis, not a directly exchangeable cohort.

## Appearances and temporal availability

Keep the source quality rule and require at least five prior observed appearances. Check `history_max_date`, `workload_max_date` and `baseline_max_date` are strictly earlier than each outing. The prediction timestamp is after completion of the outing. Current velocity/release/command/workload measures cannot support a pregame claim.

An analytical active-episode interval begins at the conservative event anchor. It ends at the earliest validated activation or observed MLB pitching outing after the reported placement. The latter is only a boundary for subsequent exposure, **not** an imputed medical return or duration. Appearance data and canonical injury outcomes remain unchanged.

Historic API text may have been revised. `public_known_date` is a reported-date proxy and `historical_publication_verified` is false. The dataset is suitable for retrospective exploration, not a claim of fully reconstructed contemporaneous information. As-of prior-IL count uses only canonical placements whose reported dates precede the appearance, not future episode count/rank. The original UCL-history predictor is omitted because a procedure date alone does not verify historical publication.

## Labels, follow-up and competing events

For 14/30/60-day horizons, a positive precedes the next eligible episode by 1 through H days. Any next excluded/unknown episode in that window makes its outcome unknown rather than negative. Outings during active episodes are excluded.

A negative needs no recorded episode in the window, complete source observation through its end, and a recorded MLB appearance at or after H days within a 30-day grace period. This avoids labeling end-of-observation or disappearance from MLB as known health. It selects active pitchers and still cannot establish symptom-free health, continuous MLB roster membership, or absence of non-IL injuries.

Each category is compared with the same eligible negative pool. Impending other-category injuries are excluded, not silently relabeled healthy. Category prevalences differ. PR lift and alert burden describe these selected comparisons; they are not full-league operational estimates or definitive rankings of biological predictability.

## Matched trajectories

An event needs at least three eligible appearances in the previous 60 days and at least one in the previous 14 days. Event-level means are calculated for 31–60, 15–30, 8–14 and 1–7 calendar days, plus the last five observed outings within the 60-day window. Feature history remains the existing full-history computation; the 60-day alignment does not reset baselines.

Controls match year/month of the last pre-injury outing, current starter/relief role, five-year age band and observed experience band (≤20, 21–100, >100 prior outings). Require no observed IL event in the preceding 60 days or following 90 days and observed 60-day follow-up. Pseudo-event date is the matched outing plus the case's last-outing lead time. Subsequent control outings before the pseudo-date are not included, ensuring the index outing is matched. Each control pitcher is used once, and no same-pitcher self-match is allowed. Event processing and selection use a fixed seed.

Matching does not condition on current velocity or workload, which are quantities being compared. Unmatched cases remain in severity/recurrence descriptions; matched-effect claims only use matched pairs. Selection of controls with future health is a retrospective design criterion, not information supplied to a predictor. Controls may have injuries elsewhere in their careers.

Each event contributes one mean per time bin. Confidence intervals resample entire case pitchers using 300 bootstrap replicates, preserving repeated events. In paired comparisons the associated control remains attached; unique control pitchers prevent control reuse. A person may be a control and a case elsewhere, leaving residual cross-pair dependence. These are exploratory intervals with no multiplicity adjustment, not formal independent-row p-values.

## Predictive experiment

- Train through 2021; validation 2022–2023; test 2024–2025. 2026 remains descriptive.
- A 60-day pre-boundary embargo and event-ID partition assertion prevent related event labels crossing retained partitions.
- At least 30 training, 10 validation and 10 test positive events are required; skipped combinations are recorded.
- Use the existing curated v1 predictors plus role, calendar month and as-of prior reported IL count. Select availability/constant columns using only training-period data. Keep columns with ≤40% training missingness. Exclude modern arm-angle measures from the common long-history model. Additional existing release-variability z-scores are descriptive and used in one declared interaction, not silently added to the baseline feature set.
- Fixed logistic regression C=1, no class sampling, no hyperparameter search. Median imputation with missingness indicators and standard scaling fit on training only. All raw/analytical nullable values remain unchanged.
- Regularization and correlated feature families mean standardized coefficients are conditional associations. Repeated pitchers can appear in later partitions: this evaluates forecasting existing pitchers, not unseen-person generalization.

Average precision summarizes PR performance. Report prevalence, PR lift, ROC-AUC secondarily, precision/recall at fixed alert budgets, F2, event recall, warning lead times and false alerts per represented pitcher-season. AP/lift uncertainty resamples complete test-pitcher histories. No accuracy-based criterion is used.

Event recall requires a warning 7–H days before injury. The primary denominator contains all eligible category events in the test date range, including events without an eligible warning outing; observable-event recall is separately reported. A lead-time median only describes detected events. Multiple alerts are counted without cooldown. Retrospective top-percent budgets use held-out ranking and are explicitly distinct from validation-quantile thresholds fixed before test evaluation; both are reported.

Feature-family ablations and two prespecified interactions (velocity z-score × workload acceleration; vertical release-variability z-score × workload acceleration) are evaluated on validation only. No final feature set is chosen using test performance. Arm-angle matched effects are descriptive within available same-season pairs; a separate richer-era prediction study remains future work.

## Severity and recurrence

Usable observed durations require a valid activation and a non-offseason activation. Report ≤30, 31–90 and >90-day bins explicitly as IL absence, with unresolved/administrative returns separate. Long-IL transfer and surgery documentation are separate proxies. No censored return is imputed; no survival model assumes censoring is noninformative. Lack of surgery documentation does not establish no surgery.

First means first **observed qualifying** episode since 2015, not first lifetime injury. Transitions require distinct chronological episodes and an observed MLB pitching outing between them; repeats without that evidence are separately reported. Original detailed-region recurrence and broader transitions are both retained. Interval distributions and crude appearance-level risk by as-of prior count are descriptions, not causal recurrence probabilities. Career duration, role, age, source completeness and left truncation affect these comparisons.

## Reproducibility and extension

Use `config/injury_analysis.yaml`, `requirements-analysis.txt` (or the executed environment's `requirements-analysis-lock.txt`), and the commands in `notebooks/injury_type_analysis/README.md`. Run 01–08 in order. The notebook runner saves successful/error outputs and an execution manifest, and asserts that source file fingerprints did not change. A changed source/config/cohort implementation invalidates the analytical cache.

The helpers are divided into cohort construction, trajectories, prediction/evaluation, outcomes, figures and report generation. Existing `src/evaluation.py` supplies ranking metrics, chronological embargoes and event/alert evaluation. Existing ingestion, injury parsing and feature engineering are untouched.

Before confirmatory modeling, adjudicate onset/publication dates and ambiguous episodes, predeclare the final target and estimand, and reserve a fresh replication period. The current test set has now been inspected for exploratory comparisons and should not be represented as untouched after model/category selection.
