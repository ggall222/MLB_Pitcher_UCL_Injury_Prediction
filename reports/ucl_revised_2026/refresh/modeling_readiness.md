# Modeling readiness



1. Seasons represented: 12 (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026); check source gaps before interpreting this as complete.

2. Raw pitch/event records: 8,150,993.

3. Total pitcher appearances: 236,628.

4. Official starting appearances: 55,244; quality-eligible model starts: 55,097.

5. Unique starting pitchers in the model population: 1,351.

6. Qualifying UCL episodes: 167; represented by a positive 30-day model row: 98.

7. Unique injured starters: 161.

8. 30-day positive prevalence: 0.7044% (355 / 50395 eligible rows).

9. 249 master features exceed 20% missing among valid starts. Arm angle, pitch-specific absence, insufficient history and native IVB need separate consideration. See feature_missingness.csv and raw_feature_availability_by_season.csv.

10. 21 unresolved identities: Adrian Perez, J.J. Picollo, Adam Meier, Nick Conte, Danny Santana, Sam Horn, Derrick Lewis, Jeff Knox, Glenn Carter, Eli White, Garrett Cooper, Drew Robinson, Ronnie Williams, Gunner Leger, Gabe Gaeckle, Sean Johnson, Jakob Marsee, Blake Dunn, England Bryan, Austin Nye, Trevor Story. Candidate evidence and scope categories are in unmatched_injury_audit.csv.

11. The observed event count supports beginning an exploratory baseline after case-level label review and the checks below; it does not establish generalization.

12. Train 2015–2022; validation 2023–2024; test 2025–2026 (latest year partial). Chosen to maximize the smallest partition event count; inspect temporal_split_audit.csv. This compares event support without fitting a model. Use 90-day embargoes for cross-horizon comparisons. Purge events spanning retained partitions. Repeated pitchers may legitimately recur in future years; also report a separate unseen-pitcher sensitivity analysis.

13. model_features_v1.yaml selects 65 features with rationales: 61 long-history candidates plus 4 optional modern arm-angle features. Per-era missingness determines the final training-fold subset; no imputation performed.

14. Remaining risks: public reporting can lag symptoms; generic forearm/elbow IL-to-surgery linkage is retrospective and requires review; surgery-only events have unknown onset and create censoring intervals; ascertainment is incomplete for minor leagues; missing follow-up is not a negative. Openers and role changes remain in the official-start population. Earlier data and 2026 location definitions differ. Cohort quality filters and incomplete telemetry can select healthier/measurable appearances.

## Label reference and event review



A qualifying starter event has a confirmed procedure, a public reference inside the study/follow-up period, and at least one quality-eligible MLB start before that reference. This broad count can include former starters without a recent prediction opportunity; positive-window and observable-event counts are reported separately.

The reference is the first public arm-IL announcement in the linked active episode. Effective/retroactive IL date is preserved separately. Clinical diagnosis, first symptom and last healthy appearance remain NULL unless explicitly sourced; last observed pre-reference appearance is not certified healthy. An official activation followed by an observed MLB appearance permits risk-set reentry. Source return dates never enter predictors. Multiple procedures within one active episode count as one event. Surgery-only records are lower confidence and do not generate fabricated precise targets.

The next-3/5 appearance outcomes count all MLB pitching appearances, including relief, with a 90-day maximum. Calendar negatives need subsequent MLB observation through the horizon and no intervening arm episode or uncertain-surgery interval. Active arm-IL rows are excluded. This conservative observation rule must be examined for selection bias.

## Feature eras



Keep 2015-present as the full tracking-history cohort. Compare broad features across 2015-2016 and 2017-2019 separately because source velocity definitions changed. Hawk-Eye began in 2020; use 2020-2025 as an enhanced-tracking sensitivity era, subject to measured coverage. Arm-angle modern-era boundaries are empirical: first_season_meaningful_availability uses >=50% nonmissing appearances and must be checked each subsequent year. Treat 2026 as a location-definition sensitivity era because plate reference and strike-zone definitions changed. Earlier seasons are preserved.

[MLB Statcast overview](https://www.mlb.com/glossary/statcast); [MLB 2020 tracking update](https://www.mlb.com/news/2020-statcast-update-includes-pose-tracking-capabilities); [Savant CSV definitions](https://baseballsavant.mlb.com/csv-docs); [Chadwick identifier register](https://github.com/chadwickbureau/register).

## Evaluation preparation



src/evaluation.py accepts held-out predictions and a predeclared event cohort. It computes AP, ROC-AUC, precision, recall, F2, alert-budget curves at 1%, 2.5%, 5%, 10%, event recall, first warning lead times, and false alerts per pitcher-season. Primary practical target: event recall with a 7-30-day warning at 5% appearance alerts. Report both all qualifying and observable event denominators. Retrospective top-score budgets are not deployable thresholds; choose deployment thresholds on validation data only.

Any future imputer, scaler, feature selector or alert threshold must be fitted only on training/validation history. Retain structural NULLs and consider explicit era-specific feature subsets. Never fill historical arm angle with zero.


Historical measurement findings: arm_angle_avg is missing for 45.01% of valid starts; velo_drop_game is missing for 22.30% because many starts do not supply 40 measured fastballs. Native IVB is sparsely supplied in 2015–2016 (2,931 measured native records). The conservative cross-source speed gate rejects 1,303,136 earlier-era records, even though physical pitch counts match. This is a measurement-comparability limitation, not a reason to fill IVB with zero. IVB is excluded from the curated v1 list. See pitch_tracking_alignment_by_season.csv.
