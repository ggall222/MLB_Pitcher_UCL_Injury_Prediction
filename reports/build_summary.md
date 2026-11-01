# Full historical build summary



Executed coverage: 2015-04-05 through 2026-09-08; 12 represented seasons.



8,077,715 raw pitch/event records; 234,393 appearances; 2,773 observed pitchers.

54,746 official starting appearances, 179,647 relief, 0 ambiguous; 54,599 quality-eligible starts from 1,342 pitchers.

Database: /Users/garrettgall/Desktop/MLB pitch stats/mlb_ucl_project/database/mlb_ucl.duckdb (16.67 GiB).



|   season |   pitches |   physical_pitches | earliest_game_date   | latest_game_date    |   pitch_games |   appearances |   pitchers |   starter_appearances |   relief_appearances |   ambiguous_appearances |   valid_starting_appearances |   unique_starting_pitchers |   official_final_games |
|---------:|----------:|-------------------:|:---------------------|:--------------------|--------------:|--------------:|-----------:|----------------------:|---------------------:|------------------------:|-----------------------------:|---------------------------:|-----------------------:|
|     2015 |    702306 |             702301 | 2015-04-05 00:00:00  | 2015-10-04 00:00:00 |          2429 |         19962 |        735 |                  4858 |                15104 |                       0 |                         4850 |                        313 |                   2429 |
|     2016 |    715823 |             715821 | 2016-04-03 00:00:00  | 2016-10-02 00:00:00 |          2428 |         20158 |        742 |                  4856 |                15302 |                       0 |                         4854 |                        309 |                   2428 |
|     2017 |    724627 |             721244 | 2017-04-02 00:00:00  | 2017-10-01 00:00:00 |          2430 |         20516 |        755 |                  4860 |                15656 |                       0 |                         4848 |                        315 |                   2430 |
|     2018 |    724452 |             721190 | 2018-03-29 00:00:00  | 2018-10-01 00:00:00 |          2431 |         21196 |        799 |                  4862 |                16334 |                       0 |                         4846 |                        343 |                   2431 |
|     2019 |    735090 |             732473 | 2019-03-20 00:00:00  | 2019-09-29 00:00:00 |          2429 |         21429 |        831 |                  4858 |                16571 |                       0 |                         4820 |                        368 |                   2429 |
|     2020 |    264262 |             263584 | 2020-07-23 00:00:00  | 2020-09-27 00:00:00 |           898 |          7959 |        735 |                  1796 |                 6163 |                       0 |                         1792 |                        294 |                    898 |
|     2021 |    712320 |             709852 | 2021-04-01 00:00:00  | 2021-10-03 00:00:00 |          2429 |         21540 |        909 |                  4858 |                16682 |                       0 |                         4839 |                        396 |                   2429 |
|     2022 |    710210 |             708540 | 2022-04-07 00:00:00  | 2022-10-05 00:00:00 |          2430 |         20878 |        871 |                  4860 |                16018 |                       0 |                         4856 |                        367 |                   2430 |
|     2023 |    720684 |             717945 | 2023-03-30 00:00:00  | 2023-10-01 00:00:00 |          2430 |         20629 |        863 |                  4860 |                15769 |                       0 |                         4844 |                        382 |                   2430 |
|     2024 |    711899 |             709512 | 2024-03-20 00:00:00  | 2024-09-30 00:00:00 |          2429 |         20693 |        855 |                  4858 |                15835 |                       0 |                         4847 |                        370 |                   2429 |
|     2025 |    712528 |             710084 | 2025-03-18 00:00:00  | 2025-09-28 00:00:00 |          2430 |         20868 |        873 |                  4860 |                16008 |                       0 |                         4850 |                        368 |                   2430 |
|     2026 |    643514 |             641413 | 2026-03-25 00:00:00  | 2026-09-08 00:00:00 |          2180 |         18565 |        852 |                  4360 |                14205 |                       0 |                         4353 |                        360 |                   2180 |



## Injury population



| metric                                     |   count |
|:-------------------------------------------|--------:|
| total_ucl_procedure_records                |    2797 |
| matched_mlbam_records                      |    2777 |
| records_with_statcast_history              |    1028 |
| records_with_surgery_in_study_period       |    1598 |
| records_involving_pitchers_with_mlb_starts |     594 |
| qualifying_unique_starter_events           |     166 |
| unique_injured_starters                    |     160 |
| repeat_procedures                          |     211 |
| internal_brace_procedures                  |      73 |
| tommy_john_reconstruction_procedures       |    2724 |
| uncertain_surgery_dates                    |    1384 |
| unconfirmed_or_out_of_scope_procedures     |       5 |
| surgery_only_events_unknown_onset          |    2542 |



## Starting appearance labels



| label                         |   eligible_rows |   positive_rows |   negative_rows |   positive_percentage |   unique_injury_events |   unique_injured_pitchers |   unknown_or_ineligible_rows |
|:------------------------------|----------------:|----------------:|----------------:|----------------------:|-----------------------:|--------------------------:|-----------------------------:|
| ucl_injury_next_14d           |           51212 |             218 |           50994 |              0.425681 |                     94 |                        92 |                         3387 |
| ucl_injury_next_30d           |           49880 |             355 |           49525 |              0.711708 |                     98 |                        96 |                         4719 |
| ucl_injury_next_60d           |           47957 |             518 |           47439 |              1.08013  |                     99 |                        97 |                         6642 |
| ucl_injury_next_90d           |           46317 |             613 |           45704 |              1.32349  |                    101 |                        99 |                         8282 |
| ucl_injury_next_3_appearances |           44646 |             244 |           44402 |              0.546522 |                     95 |                        93 |                         9953 |
| ucl_injury_next_5_appearances |           39304 |             357 |           38947 |              0.908304 |                     95 |                        93 |                        15295 |



Unmatched identities: 20; original-55 reconciliation outcomes are individually recorded in unmatched_injury_audit.csv.

| status     | reason_category                |   records |
|:-----------|:-------------------------------|----------:|
| resolved   | reconciled_identifier          |        35 |
| unresolved | incorrect_or_missing_player_id |         4 |
| unresolved | missing_mlbam_id               |         1 |
| unresolved | name_matching_issue            |        11 |
| unresolved | outside_statcast_period        |         3 |
| unresolved | procedure_not_in_scope         |         1 |



Failed source items: 0; finalized games without Statcast records: 0. Inspect failed_source_ranges.csv and missing_statcast_games.csv; represented seasons do not imply complete coverage.

Database integrity and temporal checks: PASS.



Primary population is regular-season MLB official starts, including openers. Official boxscore gamesStarted is preferred; first pitcher in official pitching order is fallback. Teams must have exactly one identified starter. Quality eligibility requires an official boxscore, reliable game timing, >=80% measured velocity, and physical-vs-official pitch-count difference <=2. No pitch-count threshold defines a starter.

Prediction time is after appearance completion. Same-day and future observations are excluded from historical features; current-game metrics are available at this prediction time. Raw relief appearances, rejected quality rows and the full feature master are retained.

No final model was fitted and no missing measurements were imputed.


Historical measurement findings: arm_angle_avg is missing for 45.42% of valid starts; velo_drop_game is missing for 22.15% because many starts do not supply 40 measured fastballs. Native IVB is sparsely supplied in 2015–2016 (2,931 measured native records). The conservative cross-source speed gate rejects 1,303,136 earlier-era records, even though physical pitch counts match. This is a measurement-comparability limitation, not a reason to fill IVB with zero. IVB is excluded from the curated v1 list. See pitch_tracking_alignment_by_season.csv.
