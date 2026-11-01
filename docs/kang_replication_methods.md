# Kang parent-paper replication protocol

This separate stage attempts the published experiment, not a methodological
extension. The earlier `kang_benchmark` foundation and all UCL work are preserved.
Protocol choices below are fixed before this stage's fits.

Source: Kang LS et al., OJSM 2026;14(9),
[10.1177/23259671261477532](https://journals.sagepub.com/doi/10.1177/23259671261477532),
Methods, Tables 1–3, Figures 2–4 and Appendix A1–A4; reviewed 2026-10-30.

| Element | Published description | Replication status / implementation |
|---|---|---|
| Population / time | MLB regular-season pitching, 2017–2025 | Exact dates/game type; existing Statcast/official boxscores instead of the authors' extract |
| Grain | Pitcher-game; repeated pitcher-seasons | Exact grain, MLBAM identity |
| Outcomes | First shoulder or elbow IL event per season | Close approximation: general official IL episode anatomy, not adjudicated UCL |
| Event date | IL placement | Transaction date; no surgery/onset substitution |
| Eligibility | At least 5 outings, at most 10% games missing any required field | Exact thresholds; Appendix A1 predictor list |
| Role | Season majority starter/reliever; innings break ties | Exact conceptual rule; unresolved ties unavailable |
| Controls | Methods: no shoulder/elbow IL; Results: no IL | Ambiguous; follow Methods, count Results interpretation separately |
| Case games | Only before placement; binary indicator inside window | Earlier case games retained as negative; exact outside-window handling ambiguous |
| Windows | Cumulative windows spanning 7–49 days | Exact endpoints; daily integer grid is a disclosed approximation because increment is not supplied |
| Inputs | Age, workload, velocity/spin summaries, release, pitch fractions | 22 Appendix A1 predictors; no project-specific engineered predictors |
| Release coordinates | Tables name Y; appendix/importance name extension | Ambiguous; model appendix extension, retain Y in crosswalk only |
| Preprocessing | Absolute X; domain-wise Pearson pruning above 0.7 | Same operations; existing source QC, native XGBoost NaNs are source/software approximations |
| Split | Stratified random games, approximately 67/33 | Exact ratio/grain; same pitcher allowed in both; seed unavailable |
| Window CV | Training-only stratified 5-fold AUC | Exact CV type; full published inputs used before final-window selection |
| Selection | 50 repetitions of 5-fold CV on random subsets; gain/consistency | Exact repetitions/folds; 80% subsamples and use-in-tree frequency ≥0.5 are explicit unknown-detail approximations |
| Pruning | Within-domain highest average Gain among correlated variables | Same rule via gain-ordered pairwise greedy pruning; domains declared in crosswalk |
| Learner | Class-weighted XGBoost; training-only tuning | Same R model family; author tuning procedure/parameters unavailable; one untuned default-style specification |
| Class weights | Inverse class frequency | Same relative weights, normalized to mean one within each fit; normalization not reported |
| Window selection | GAM AUC curve; derivative between 25% and 35% of maximum slope | Same structure with R `mgcv`; basis/penalty/grid settings unavailable, use documented defaults |
| Integer final window | Not provided | Rounded midpoint of inferred derivative band; approximation, never maximum AUC fallback |
| Primary metric | Holdout ROC-AUC, 500 pitcher-season bootstrap CIs | Exact metric/resampling unit/count |
| Importance | XGBoost Gain; Figure 4 mean gain over bootstrap resamples | R total-Gain fraction; 500 training pitcher-season bootstrap refits; exact importance-resample details not reported |
| Prior-injury sensitivity | Corresponding anatomy IL in previous seasons within study | Reproduced; same frozen window/features plus prior-region binary |
| Rolling variability / dynamic risk | Top 15 features; rolling SD, predicted-risk IQR; 500 cluster resamples | Unavailable for exact numerical replication: mapping of rolling SD to trained level-feature predictions / feature-specific attribution is not specified; no invented substitute |

## Necessary choices and limits

All recorded MLB pitching appearances enter season screening; the paper does
not describe a primary-position filter, and its raw 7,491 pitcher-seasons match
the all-pitching source. Position-player pitching outings that survive its five-
outing rule remain included. Role is a retrospective season label, not a
real-time predictor. No pitcher split, embargo, competing-event censoring,
minimum prior history, injury-episode QA exclusion, or UCL adjudication is added.
Missing placement IDs cannot locate an initial IL event and remain unclassifiable.
Overlaps, timing flags and multiple anatomy flags are reported, not excluded.

Existing episode construction avoids counting IL transfers as new placements.
It is rerun through 2025 on the already normalized official transactions and
appearances; this is the study-period boundary, not a chronological model
redesign. Broad episode anatomy may use later clarification within that period;
initial anatomy is also saved. Controls lack a recorded target placement during
the regular season; unknown diagnoses are not silently turned into confirmed
healthy controls and are explicitly counted among source-uncertain controls.
They are nevertheless retained under the publication's no-recorded-target rule.
Case games on/after placement are excluded. Preseason/offseason events are not
regular-season incident cases. Source/coding differences are not tuned away.

The crosswalk uses the requested labels **exact**, **close proxy**, **unavailable**.
“Exact” refers to a field's definition, not identical raw observations. Movement,
arm angle, pitcher z-scores and profile/departure predictors are not listed as
model inputs in this paper and are not added. Previously derived all-pitch SD,
spin maximum and absolute X implement paper concepts, not new research features.

A fixed game reservation, stratified on the 49-day label, is reused across
windows. The paper does not specify partition reuse. Only the training portion
provides window curves, selection, correlation and model fitting. The window
search uses five-fold XGBoost on the complete published predictor set; the paper
describes optimal-window construction before final model development but does
not fully specify nesting of selection in window CV. At the selected window,
50 independent 80% stratified subsets each receive stratified five-fold fits
(250 fits). Mean normalized total Gain and frequency used in a tree determine
the retained candidates; correlated candidates lose to higher average gain.
Neither the frequency cutoff nor subset fraction is claimed to be the authors'.

R `mgcv::gam(auc ~ s(days, bs='tp', k=10), method='GCV.Cp')` smooths the mean
training CV AUC. Finite-difference slopes use step 1e-4 on a .01-day grid. The
qualifying range is after the maximum slope and strictly between its 25% and
35% levels; lower/upper are the first/last qualifying days. Disconnected bands
are flagged rather than hidden. No positive declining-slope band means the
optimal window is unavailable: no replacement with the AUC maximum. Rounded
band midpoint supplies the final integer window; that conversion is not stated
in the article. GAM is a performance-curve smoother, **not a challenger injury model**.

Primary final models and the paper's prior-injury sensitivity use the same
unmodified reserved games. Only ROC-AUC and its cluster interval are reported
as predictive performance. Importance uses `xgb.importance` total-Gain shares,
not SHAP, permutation importance or Python average gain. Conditional bootstrap
intervals do not remove the train/test pitcher overlap inherent in the paper.

The fixed, unreported choices are in `config/kang_replication.yaml`. They cannot
be advertised as exact hyperparameter or feature-selection replication. A faithful
structural benchmark is possible; exact numerical reproduction requires author
code/data, especially control coding, candidate grid, feature selection and GAM
settings. No settings are adjusted to match the article's AUCs or windows.

## Run and outputs

```bash
.venv/bin/python -m src.injury_analysis.kang_replication --prepare-only
.venv/bin/python -m src.injury_analysis.kang_replication
.venv/bin/python -m pytest tests/test_kang_replication.py -q
```

Python uses the existing project requirements. R requires `xgboost`, `mgcv` and
`jsonlite`; installed versions are recorded. The runner supports `--config`
for Rscript path, worker count and separate output directories. Checkpoints are
input/code/config fingerprinted. No original database object is written.

Outputs: `reports/kang_replication` (methods comparison, counts, windows, AUCs,
gain, prior-injury sensitivity, figures and summary),
`data/injury/analysis/kang_replication` (regenerable cohorts, splits, R models,
CV/resampling checkpoints), and `notebooks/kang_replication` (synthesis).
