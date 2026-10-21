# Historical database validation

| check_name                                          | severity   |   violations |
|:----------------------------------------------------|:-----------|-------------:|
| duplicate_pitches                                   | error      |            0 |
| duplicate_appearances                               | error      |            0 |
| invalid_player_ids                                  | error      |            0 |
| missing_pitch_dates                                 | error      |            0 |
| impossible_velocities_raw_retained                  | warning    |            1 |
| extreme_velocity_raw_retained                       | warning    |         2967 |
| impossible_release_raw_retained                     | warning    |          425 |
| missing_pitch_game_join                             | error      |            0 |
| appearance_missing_boxscore                         | warning    |            0 |
| pitch_count_disagreement_gt_2                       | warning    |            0 |
| unmatched_injury_identity                           | warning    |           21 |
| injury_missing_pitcher_fk                           | error      |            0 |
| post_injury_training_rows                           | error      |            0 |
| post_injury_negative_labels                         | error      |            0 |
| historical_feature_dates                            | error      |            0 |
| positive_label_window                               | error      |            0 |
| negative_without_followup                           | error      |            0 |
| unknown_date_precision_fabricated                   | error      |            0 |
| impossible_outs_conversion                          | error      |            0 |
| supplemental_tracking_unaligned                     | warning    |            0 |
| supplemental_pitch_alignment_invalid                | warning    |      1303137 |
| supplemental_invalid_alignment_used                 | error      |            0 |
| extreme_induced_vertical_break_retained             | warning    |            5 |
| predictor_temporal_allowlist                        | error      |            0 |
| nonregular_starter_cohort                           | error      |            0 |
| positive_14d_strictly_pre_reference                 | error      |            0 |
| positive_30d_strictly_pre_reference                 | error      |            0 |
| positive_60d_strictly_pre_reference                 | error      |            0 |
| positive_90d_strictly_pre_reference                 | error      |            0 |
| injury_event_foreign_key                            | error      |            0 |
| independent_prior_5_recalculation_stratified_sample | error      |            0 |
