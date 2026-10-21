-- Each statement returns one check row. Python also checks temporal feature values.
SELECT 'duplicate_pitches' AS check_name, 'error' AS severity,
       count(*) AS violations FROM (SELECT game_id, at_bat_number, pitch_number FROM pitches GROUP BY ALL HAVING count(*) > 1);
SELECT 'duplicate_appearances' AS check_name, 'error' AS severity,
       count(*) AS violations FROM (SELECT game_id, pitcher_id FROM appearances GROUP BY ALL HAVING count(*) > 1);
SELECT 'invalid_player_ids' AS check_name, 'error' AS severity,
       count(*) AS violations FROM pitches WHERE pitcher_id IS NULL OR pitcher_id <= 0;
SELECT 'missing_pitch_dates' AS check_name, 'error' AS severity,
       count(*) AS violations FROM pitches WHERE game_date IS NULL;
SELECT 'impossible_velocities_raw_retained' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM pitches WHERE release_speed NOT BETWEEN 30 AND 110;
SELECT 'extreme_velocity_raw_retained' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM pitches WHERE release_speed < 50 OR release_speed > 105;
SELECT 'impossible_release_raw_retained' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM pitches WHERE abs(release_pos_x)>7 OR release_pos_z NOT BETWEEN 0 AND 9 OR release_extension NOT BETWEEN 2 AND 9;
SELECT 'missing_pitch_game_join' AS check_name, 'error' AS severity,
       count(*) AS violations FROM pitches p LEFT JOIN games g USING(game_id) WHERE g.game_id IS NULL;
SELECT 'appearance_missing_boxscore' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM appearances WHERE boxscore_missing;
SELECT 'pitch_count_disagreement_gt_2' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM appearances WHERE abs(pitch_count_difference) > 2;
SELECT 'unmatched_injury_identity' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM injuries WHERE pitcher_id IS NULL;
SELECT 'injury_missing_pitcher_fk' AS check_name, 'error' AS severity,
       count(*) AS violations FROM injuries i LEFT JOIN pitchers p USING(pitcher_id) WHERE i.pitcher_id IS NOT NULL AND p.pitcher_id IS NULL;
SELECT 'post_injury_training_rows' AS check_name, 'error' AS severity,
       count(*) AS violations FROM model_features WHERE training_eligible AND (post_known_injury OR game_date >= risk_set_stop_date);
SELECT 'post_injury_negative_labels' AS check_name, 'error' AS severity,
       count(*) AS violations FROM model_features WHERE post_known_injury AND ucl_injury_next_30d = FALSE;
SELECT 'historical_feature_dates' AS check_name, 'error' AS severity,
       count(*) AS violations FROM model_features WHERE history_max_date >= game_date OR baseline_max_date >= game_date OR workload_max_date >= game_date;
SELECT 'positive_label_window' AS check_name, 'error' AS severity,
       count(*) AS violations FROM model_features WHERE ucl_injury_next_30d AND NOT (next_known_ucl_onset > game_date AND next_known_ucl_onset <= game_date + INTERVAL 30 DAY);
SELECT 'negative_without_followup' AS check_name, 'error' AS severity,
       count(*) AS violations FROM model_features WHERE ucl_injury_next_30d = FALSE AND (game_date + INTERVAL 30 DAY > last_observed_appearance OR game_date + INTERVAL 30 DAY > label_observation_end);
SELECT 'unknown_date_precision_fabricated' AS check_name, 'error' AS severity,
       count(*) AS violations FROM injuries WHERE surgery_date_precision <> 'day' AND surgery_date IS NOT NULL;
SELECT 'impossible_outs_conversion' AS check_name, 'error' AS severity,
       count(*) AS violations FROM boxscore_appearances WHERE abs(innings_pitched * 3 - outs_recorded) > 0.00001;
SELECT 'supplemental_tracking_unaligned' AS check_name, 'warning' AS severity,
       (SELECT count(*) FROM pitch_tracking) - (SELECT count(*) FROM aligned_pitch_tracking) AS violations;
SELECT 'supplemental_pitch_alignment_invalid' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM aligned_pitch_tracking WHERE NOT alignment_valid;
SELECT 'supplemental_invalid_alignment_used' AS check_name, 'error' AS severity,
       count(*) AS violations FROM aligned_pitch_tracking WHERE NOT alignment_valid AND ivb_inches IS NOT NULL;
SELECT 'extreme_induced_vertical_break_retained' AS check_name, 'warning' AS severity,
       count(*) AS violations FROM pitch_tracking WHERE ivb_inches NOT BETWEEN -50 AND 50;
