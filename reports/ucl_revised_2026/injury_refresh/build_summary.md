# General MLB pitcher injury build

Observation window: 2015-01-01 through 2026-10-20 (2026 is incomplete).

Collected 167,178 distinct official transactions across 142 monthly windows; 0 failed windows.
Built 5,410 IL episodes involving 1,618 pitchers. 49 episodes start with an orphan transfer and retain an unknown initial event date.

## Validation

| metric                                   |        value |
|:-----------------------------------------|-------------:|
| raw_transactions                         | 167178       |
| raw_il_related_transactions              |  19106       |
| initial_placements_all_positions         |   9739       |
| pitcher_il_transactions                  |  10819       |
| pitcher_initial_placements               |   5603       |
| canonical_events                         |   5410       |
| unique_pitchers                          |   1618       |
| events_with_statcast_history             |   5361       |
| unknown_region_events                    |    731       |
| unknown_region_percentage                |     13.512   |
| unknown_diagnosis_percentage             |     18.7061  |
| laterality_coverage_percentage           |     74.4732  |
| matched_activation_percentage            |     68.8725  |
| right_censored_percentage                |     31.1275  |
| orphan_transfers                         |     49       |
| inferred_longer_il_continuations         |    185       |
| duplicate_placement_records_consolidated |     57       |
| overlapping_episode_percentage           |      7.28281 |
| review_required_percentage               |     52.976   |
| extreme_duration_events                  |    324       |
| duplicate_event_percentage               |      0       |
| supplemental_confirmation_percentage     |     50.0924  |
| supplemental_linkage_percentage          |     59.3161  |
| source_conflict_events                   |    509       |
| existing_ucl_in_both                     |    259       |
| existing_ucl_only                        |    472       |
| general_explicit_ucl_only                |    220       |
| monthly_windows_requested                |    142       |
| monthly_windows_failed                   |      0       |
| all_transaction_windows_available        |      1       |
| automatic_validation_checks_passed       |     26       |
| days_out_observed_count                  |   3702       |
| days_out_observed_mean                   |     61.1777  |
| days_out_observed_std                    |     64.5884  |
| days_out_observed_min                    |      1       |
| days_out_observed_5%                     |      9       |
| days_out_observed_25%                    |     16       |
| days_out_observed_50%                    |     34       |
| days_out_observed_75%                    |     79.75    |
| days_out_observed_90%                    |    163       |
| days_out_observed_95%                    |    217       |
| days_out_observed_99%                    |    258.99    |
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
|     2025 |      469 |        362 |                   414 |               55 |                 29 |
|     2026 |      500 |        396 |                   335 |              165 |                 27 |

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
