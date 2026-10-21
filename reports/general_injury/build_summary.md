# General MLB pitcher injury build

Observation window: 2015-01-01 through 2026-09-18 (2026 is incomplete).

Collected 166,203 distinct official transactions across 141 monthly windows; 0 failed windows.
Built 5,381 IL episodes involving 1,616 pitchers. 49 episodes start with an orphan transfer and retain an unknown initial event date.

## Validation

| metric                                   |        value |
|:-----------------------------------------|-------------:|
| raw_transactions                         | 166203       |
| raw_il_related_transactions              |  18917       |
| initial_placements_all_positions         |   9691       |
| pitcher_il_transactions                  |  10705       |
| pitcher_initial_placements               |   5574       |
| canonical_events                         |   5381       |
| unique_pitchers                          |   1616       |
| events_with_statcast_history             |   5332       |
| unknown_region_events                    |    729       |
| unknown_region_percentage                |     13.5477  |
| unknown_diagnosis_percentage             |     18.6954  |
| laterality_coverage_percentage           |     74.4657  |
| matched_activation_percentage            |     67.8313  |
| right_censored_percentage                |     32.1687  |
| orphan_transfers                         |     49       |
| inferred_longer_il_continuations         |    185       |
| duplicate_placement_records_consolidated |     57       |
| overlapping_episode_percentage           |      7.24772 |
| review_required_percentage               |     53.7447  |
| extreme_duration_events                  |    318       |
| duplicate_event_percentage               |      0       |
| supplemental_confirmation_percentage     |     50.0093  |
| supplemental_linkage_percentage          |     59.1897  |
| source_conflict_events                   |    504       |
| existing_ucl_in_both                     |    258       |
| existing_ucl_only                        |    456       |
| general_explicit_ucl_only                |    219       |
| monthly_windows_requested                |    141       |
| monthly_windows_failed                   |      0       |
| all_transaction_windows_available        |      1       |
| automatic_validation_checks_passed       |     26       |
| days_out_observed_count                  |   3626       |
| days_out_observed_mean                   |     61.4774  |
| days_out_observed_std                    |     64.4013  |
| days_out_observed_min                    |      1       |
| days_out_observed_5%                     |      9       |
| days_out_observed_25%                    |     16       |
| days_out_observed_50%                    |     34       |
| days_out_observed_75%                    |     81       |
| days_out_observed_90%                    |    163.5     |
| days_out_observed_95%                    |    217       |
| days_out_observed_99%                    |    258.75    |
| days_out_observed_max                    |    533       |

## Season coverage

|   season |   events |   pitchers |   matched_activations |   right_censored |   supplemental_ucl |
|---------:|---------:|-----------:|----------------------:|-----------------:|-------------------:|
|     2015 |      286 |        253 |                   207 |               79 |                 15 |
|     2016 |      298 |        259 |                   228 |               70 |                 15 |
|     2017 |      375 |        295 |                   258 |              117 |                 13 |
|     2018 |      431 |        338 |                   301 |              130 |                 25 |
|     2019 |      398 |        317 |                   284 |              114 |                 15 |
|     2020 |      311 |        279 |                   248 |               63 |                 18 |
|     2021 |      742 |        496 |                   327 |              415 |                 19 |
|     2022 |      586 |        421 |                   297 |              289 |                 23 |
|     2023 |      507 |        393 |                   416 |               91 |                 30 |
|     2024 |      507 |        381 |                   411 |               96 |                 30 |
|     2025 |      469 |        362 |                   413 |               56 |                 29 |
|     2026 |      471 |        386 |                   260 |              211 |                 26 |

## Interpretation

- Canonical records are observed MLB IL episodes, including documented illnesses. They are not a census of every injury or a biological onset standard.
- `injury_event_date` follows explicit onset, parsed retroactive date, effective date, then announcement. `public_known_date` preserves the announcement needed for prospective prediction cutoffs.
- Historical descriptions can be revised. `*_known_at` fields for MLB are reported-date proxies, not verified publication timestamps; `*_observed_at` records actual retrieval. `historical_publication_verified` is false unless separately adjudicated. Use snapshot dates for strict information-availability guarantees.
- Activations require a valid later transaction, no intervening unrelated placement, a bounded interval, and no contradictory observed MLB pitching. Offseason activations can be administrative roster moves.
- Resolution dates never establish return. Censored durations remain NULL; an orphan transfer also lacks an initial placement date.
- Official anatomy is preserved. Supplemental diagnoses, snapshot dates and surgery evidence are separate; disagreements require review.
- Broad elbow/forearm IL evidence is compatible with a later UCL procedure but is not proof that the original complaint was a confirmed UCL tear.
- Existing UCL-only and general-UCL-only cases are comparison findings, not automatic false positives/negatives. The legacy surgery snapshot includes only its own coverage and eligibility.
- Features and labels in the existing DuckDB were not modified. Future outcome fields in this dataset must never be used as pre-injury predictors.
- See `docs/general_injury_pipeline.md`, acquisition coverage, supplemental coverage and review CSVs for methods and limitations.
