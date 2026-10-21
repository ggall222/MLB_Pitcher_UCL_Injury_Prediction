# Model data dictionary

Complete table-level dictionary: [data_dictionary.csv](data_dictionary.csv).
A safe feature can still be NULL or constant in a short sample. Fit imputation/selection only on the training fold.

| Variable | Type | Unit | Feature | Definition |
|---|---|---|---|---|
| game_id | BIGINT | identifier | False | MLB gamePk identifier |
| pitcher_id | BIGINT | identifier | False | MLBAM person identifier |
| total_pitches | BIGINT | count | True | Total pitches; see source/table methodology and builder for exact aggregation |
| statcast_batters_faced | BIGINT | count | True | Statcast batters faced; see source/table methodology and builder for exact aggregation |
| avg_velocity | DOUBLE | mph | True | Avg velocity; see source/table methodology and builder for exact aggregation |
| max_velocity | DOUBLE | mph | True | Max velocity; see source/table methodology and builder for exact aggregation |
| effective_speed_avg | DOUBLE | mph | True | Effective speed avg; see source/table methodology and builder for exact aggregation |
| release_x_avg | DOUBLE | feet | True | Release x avg; see source/table methodology and builder for exact aggregation |
| release_z_avg | DOUBLE | feet | True | Release z avg; see source/table methodology and builder for exact aggregation |
| release_x_sd | DOUBLE | feet | True | Release x sd; see source/table methodology and builder for exact aggregation |
| release_z_sd | DOUBLE | feet | True | Release z sd; see source/table methodology and builder for exact aggregation |
| extension_avg | DOUBLE | feet | True | Extension avg; see source/table methodology and builder for exact aggregation |
| extension_sd | DOUBLE | feet | True | Extension sd; see source/table methodology and builder for exact aggregation |
| arm_angle_avg | DOUBLE | degrees | True | Arm angle avg; see source/table methodology and builder for exact aggregation |
| arm_angle_sd | DOUBLE | degrees | True | Arm angle sd; see source/table methodology and builder for exact aggregation |
| spin_avg | DOUBLE | rpm | True | Spin avg; see source/table methodology and builder for exact aggregation |
| ivb_inches | DOUBLE | inches | True | Native MLB game-feed pitchData.breaks.breakVerticalInduced, averaged over measured physical pitches; distinct from Savant pfx_z |
| horizontal_movement_avg | DOUBLE | feet | True | Horizontal movement avg; see source/table methodology and builder for exact aggregation |
| vertical_movement_avg | DOUBLE | feet | True | Vertical movement avg; see source/table methodology and builder for exact aggregation |
| zone_pct | DOUBLE | fraction 0–1 | True | Fraction of measured-zone pitches with Savant zone 1 through 9; missing zones excluded |
| strike_pct | DOUBLE | fraction 0–1 | True | Fraction of classified pitches with S or X result, including contact in play |
| first_pitch_strike_pct | DOUBLE | fraction 0–1 | True | Fraction of pitch_number=1 classified pitches with S or X result |
| hbp_count | BIGINT | count | True | Hbp count; see source/table methodology and builder for exact aggregation |
| hbp_rate | DOUBLE | fraction 0–1 | True | Hbp rate; see source/table methodology and builder for exact aggregation |
| plate_x_sd | DOUBLE | feet | True | Plate x sd; see source/table methodology and builder for exact aggregation |
| plate_z_sd | DOUBLE | feet | True | Plate z sd; see source/table methodology and builder for exact aggregation |
| high_velocity_pitch_count | BIGINT | mph | True | High velocity pitch count; see source/table methodology and builder for exact aggregation |
| velocity_measured_count | BIGINT | mph | True | Velocity measured count; see source/table methodology and builder for exact aggregation |
| arm_angle_measured_count | BIGINT | degrees | True | Arm angle measured count; see source/table methodology and builder for exact aggregation |
| location_variability | DOUBLE | feet | True | sqrt(plate_x_sd^2 + plate_z_sd^2) |
| velocity_coverage | DOUBLE | mph | True | Velocity coverage; see source/table methodology and builder for exact aggregation |
| arm_angle_coverage | DOUBLE | degrees | True | Arm angle coverage; see source/table methodology and builder for exact aggregation |
| avg_fb_velo | DOUBLE | mph | True | Avg fb velo; see source/table methodology and builder for exact aggregation |
| max_fb_velo | DOUBLE | mph | True | Max fb velo; see source/table methodology and builder for exact aggregation |
| median_fb_velo | DOUBLE | mph | True | Median fb velo; see source/table methodology and builder for exact aggregation |
| fb_velo_sd | DOUBLE | mph | True | Fb velo sd; see source/table methodology and builder for exact aggregation |
| fastball_count | BIGINT | count | True | Fastball count; see source/table methodology and builder for exact aggregation |
| fastball_usage_pct | DOUBLE | fraction 0–1 | True | Fastball usage pct; see source/table methodology and builder for exact aggregation |
| first_n_fb_velo | DOUBLE | mph | True | First n fb velo; see source/table methodology and builder for exact aggregation |
| last_n_fb_velo | DOUBLE | mph | True | Last n fb velo; see source/table methodology and builder for exact aggregation |
| velo_drop_game | DOUBLE | mph | True | Last 20 measured FF/SI/FC velocities minus first 20; NULL unless at least 40 measured fastballs |
| ff_pitch_count | DOUBLE | count | True | Ff pitch count; see source/table methodology and builder for exact aggregation |
| ff_avg_velocity | DOUBLE | mph | True | Ff avg velocity; see source/table methodology and builder for exact aggregation |
| ff_spin_avg | DOUBLE | rpm | True | Ff spin avg; see source/table methodology and builder for exact aggregation |
| ff_horizontal_movement_avg | DOUBLE | feet | True | Ff horizontal movement avg; see source/table methodology and builder for exact aggregation |
| ff_vertical_movement_avg | DOUBLE | feet | True | Ff vertical movement avg; see source/table methodology and builder for exact aggregation |
| ff_release_x_avg | DOUBLE | feet | True | Ff release x avg; see source/table methodology and builder for exact aggregation |
| ff_release_z_avg | DOUBLE | feet | True | Ff release z avg; see source/table methodology and builder for exact aggregation |
| ff_extension_avg | DOUBLE | feet | True | Ff extension avg; see source/table methodology and builder for exact aggregation |
| ff_arm_angle_avg | DOUBLE | degrees | True | Ff arm angle avg; see source/table methodology and builder for exact aggregation |
| ff_zone_pct | DOUBLE | fraction 0–1 | True | Ff zone pct; see source/table methodology and builder for exact aggregation |
| ff_strike_pct | DOUBLE | fraction 0–1 | True | Ff strike pct; see source/table methodology and builder for exact aggregation |
| ff_spin_axis_avg | DOUBLE | degrees | True | Ff spin axis avg; see source/table methodology and builder for exact aggregation |
| ff_usage_pct | DOUBLE | fraction 0–1 | True | Ff usage pct; see source/table methodology and builder for exact aggregation |
| ff_ivb_inches | DOUBLE | inches | True | Ff ivb inches; see source/table methodology and builder for exact aggregation |
| si_pitch_count | DOUBLE | count | True | Si pitch count; see source/table methodology and builder for exact aggregation |
| si_avg_velocity | DOUBLE | mph | True | Si avg velocity; see source/table methodology and builder for exact aggregation |
| si_spin_avg | DOUBLE | rpm | True | Si spin avg; see source/table methodology and builder for exact aggregation |
| si_horizontal_movement_avg | DOUBLE | feet | True | Si horizontal movement avg; see source/table methodology and builder for exact aggregation |
| si_vertical_movement_avg | DOUBLE | feet | True | Si vertical movement avg; see source/table methodology and builder for exact aggregation |
| si_release_x_avg | DOUBLE | feet | True | Si release x avg; see source/table methodology and builder for exact aggregation |
| si_release_z_avg | DOUBLE | feet | True | Si release z avg; see source/table methodology and builder for exact aggregation |
| si_extension_avg | DOUBLE | feet | True | Si extension avg; see source/table methodology and builder for exact aggregation |
| si_arm_angle_avg | DOUBLE | degrees | True | Si arm angle avg; see source/table methodology and builder for exact aggregation |
| si_zone_pct | DOUBLE | fraction 0–1 | True | Si zone pct; see source/table methodology and builder for exact aggregation |
| si_strike_pct | DOUBLE | fraction 0–1 | True | Si strike pct; see source/table methodology and builder for exact aggregation |
| si_spin_axis_avg | DOUBLE | degrees | True | Si spin axis avg; see source/table methodology and builder for exact aggregation |
| si_usage_pct | DOUBLE | fraction 0–1 | True | Si usage pct; see source/table methodology and builder for exact aggregation |
| si_ivb_inches | DOUBLE | inches | True | Si ivb inches; see source/table methodology and builder for exact aggregation |
| fc_pitch_count | DOUBLE | count | True | Fc pitch count; see source/table methodology and builder for exact aggregation |
| fc_avg_velocity | DOUBLE | mph | True | Fc avg velocity; see source/table methodology and builder for exact aggregation |
| fc_spin_avg | DOUBLE | rpm | True | Fc spin avg; see source/table methodology and builder for exact aggregation |
| fc_horizontal_movement_avg | DOUBLE | feet | True | Fc horizontal movement avg; see source/table methodology and builder for exact aggregation |
| fc_vertical_movement_avg | DOUBLE | feet | True | Fc vertical movement avg; see source/table methodology and builder for exact aggregation |
| fc_release_x_avg | DOUBLE | feet | True | Fc release x avg; see source/table methodology and builder for exact aggregation |
| fc_release_z_avg | DOUBLE | feet | True | Fc release z avg; see source/table methodology and builder for exact aggregation |
| fc_extension_avg | DOUBLE | feet | True | Fc extension avg; see source/table methodology and builder for exact aggregation |
| fc_arm_angle_avg | DOUBLE | degrees | True | Fc arm angle avg; see source/table methodology and builder for exact aggregation |
| fc_zone_pct | DOUBLE | fraction 0–1 | True | Fc zone pct; see source/table methodology and builder for exact aggregation |
| fc_strike_pct | DOUBLE | fraction 0–1 | True | Fc strike pct; see source/table methodology and builder for exact aggregation |
| fc_spin_axis_avg | DOUBLE | degrees | True | Fc spin axis avg; see source/table methodology and builder for exact aggregation |
| fc_usage_pct | DOUBLE | fraction 0–1 | True | Fc usage pct; see source/table methodology and builder for exact aggregation |
| fc_ivb_inches | DOUBLE | inches | True | Fc ivb inches; see source/table methodology and builder for exact aggregation |
| sl_pitch_count | DOUBLE | count | True | Sl pitch count; see source/table methodology and builder for exact aggregation |
| sl_avg_velocity | DOUBLE | mph | True | Sl avg velocity; see source/table methodology and builder for exact aggregation |
| sl_spin_avg | DOUBLE | rpm | True | Sl spin avg; see source/table methodology and builder for exact aggregation |
| sl_horizontal_movement_avg | DOUBLE | feet | True | Sl horizontal movement avg; see source/table methodology and builder for exact aggregation |
| sl_vertical_movement_avg | DOUBLE | feet | True | Sl vertical movement avg; see source/table methodology and builder for exact aggregation |
| sl_release_x_avg | DOUBLE | feet | True | Sl release x avg; see source/table methodology and builder for exact aggregation |
| sl_release_z_avg | DOUBLE | feet | True | Sl release z avg; see source/table methodology and builder for exact aggregation |
| sl_extension_avg | DOUBLE | feet | True | Sl extension avg; see source/table methodology and builder for exact aggregation |
| sl_arm_angle_avg | DOUBLE | degrees | True | Sl arm angle avg; see source/table methodology and builder for exact aggregation |
| sl_zone_pct | DOUBLE | fraction 0–1 | True | Sl zone pct; see source/table methodology and builder for exact aggregation |
| sl_strike_pct | DOUBLE | fraction 0–1 | True | Sl strike pct; see source/table methodology and builder for exact aggregation |
| sl_spin_axis_avg | DOUBLE | degrees | True | Sl spin axis avg; see source/table methodology and builder for exact aggregation |
| sl_usage_pct | DOUBLE | fraction 0–1 | True | Sl usage pct; see source/table methodology and builder for exact aggregation |
| sl_ivb_inches | DOUBLE | inches | True | Sl ivb inches; see source/table methodology and builder for exact aggregation |
| st_pitch_count | DOUBLE | count | True | St pitch count; see source/table methodology and builder for exact aggregation |
| st_avg_velocity | DOUBLE | mph | True | St avg velocity; see source/table methodology and builder for exact aggregation |
| st_spin_avg | DOUBLE | rpm | True | St spin avg; see source/table methodology and builder for exact aggregation |
| st_horizontal_movement_avg | DOUBLE | feet | True | St horizontal movement avg; see source/table methodology and builder for exact aggregation |
| st_vertical_movement_avg | DOUBLE | feet | True | St vertical movement avg; see source/table methodology and builder for exact aggregation |
| st_release_x_avg | DOUBLE | feet | True | St release x avg; see source/table methodology and builder for exact aggregation |
| st_release_z_avg | DOUBLE | feet | True | St release z avg; see source/table methodology and builder for exact aggregation |
| st_extension_avg | DOUBLE | feet | True | St extension avg; see source/table methodology and builder for exact aggregation |
| st_arm_angle_avg | DOUBLE | degrees | True | St arm angle avg; see source/table methodology and builder for exact aggregation |
| st_zone_pct | DOUBLE | fraction 0–1 | True | St zone pct; see source/table methodology and builder for exact aggregation |
| st_strike_pct | DOUBLE | fraction 0–1 | True | St strike pct; see source/table methodology and builder for exact aggregation |
| st_spin_axis_avg | DOUBLE | degrees | True | St spin axis avg; see source/table methodology and builder for exact aggregation |
| st_usage_pct | DOUBLE | fraction 0–1 | True | St usage pct; see source/table methodology and builder for exact aggregation |
| st_ivb_inches | DOUBLE | inches | True | St ivb inches; see source/table methodology and builder for exact aggregation |
| cu_pitch_count | DOUBLE | count | True | Cu pitch count; see source/table methodology and builder for exact aggregation |
| cu_avg_velocity | DOUBLE | mph | True | Cu avg velocity; see source/table methodology and builder for exact aggregation |
| cu_spin_avg | DOUBLE | rpm | True | Cu spin avg; see source/table methodology and builder for exact aggregation |
| cu_horizontal_movement_avg | DOUBLE | feet | True | Cu horizontal movement avg; see source/table methodology and builder for exact aggregation |
| cu_vertical_movement_avg | DOUBLE | feet | True | Cu vertical movement avg; see source/table methodology and builder for exact aggregation |
| cu_release_x_avg | DOUBLE | feet | True | Cu release x avg; see source/table methodology and builder for exact aggregation |
| cu_release_z_avg | DOUBLE | feet | True | Cu release z avg; see source/table methodology and builder for exact aggregation |
| cu_extension_avg | DOUBLE | feet | True | Cu extension avg; see source/table methodology and builder for exact aggregation |
| cu_arm_angle_avg | DOUBLE | degrees | True | Cu arm angle avg; see source/table methodology and builder for exact aggregation |
| cu_zone_pct | DOUBLE | fraction 0–1 | True | Cu zone pct; see source/table methodology and builder for exact aggregation |
| cu_strike_pct | DOUBLE | fraction 0–1 | True | Cu strike pct; see source/table methodology and builder for exact aggregation |
| cu_spin_axis_avg | DOUBLE | degrees | True | Cu spin axis avg; see source/table methodology and builder for exact aggregation |
| cu_usage_pct | DOUBLE | fraction 0–1 | True | Cu usage pct; see source/table methodology and builder for exact aggregation |
| cu_ivb_inches | DOUBLE | inches | True | Cu ivb inches; see source/table methodology and builder for exact aggregation |
| kc_pitch_count | DOUBLE | count | True | Kc pitch count; see source/table methodology and builder for exact aggregation |
| kc_avg_velocity | DOUBLE | mph | True | Kc avg velocity; see source/table methodology and builder for exact aggregation |
| kc_spin_avg | DOUBLE | rpm | True | Kc spin avg; see source/table methodology and builder for exact aggregation |
| kc_horizontal_movement_avg | DOUBLE | feet | True | Kc horizontal movement avg; see source/table methodology and builder for exact aggregation |
| kc_vertical_movement_avg | DOUBLE | feet | True | Kc vertical movement avg; see source/table methodology and builder for exact aggregation |
| kc_release_x_avg | DOUBLE | feet | True | Kc release x avg; see source/table methodology and builder for exact aggregation |
| kc_release_z_avg | DOUBLE | feet | True | Kc release z avg; see source/table methodology and builder for exact aggregation |
| kc_extension_avg | DOUBLE | feet | True | Kc extension avg; see source/table methodology and builder for exact aggregation |
| kc_arm_angle_avg | DOUBLE | degrees | True | Kc arm angle avg; see source/table methodology and builder for exact aggregation |
| kc_zone_pct | DOUBLE | fraction 0–1 | True | Kc zone pct; see source/table methodology and builder for exact aggregation |
| kc_strike_pct | DOUBLE | fraction 0–1 | True | Kc strike pct; see source/table methodology and builder for exact aggregation |
| kc_spin_axis_avg | DOUBLE | degrees | True | Kc spin axis avg; see source/table methodology and builder for exact aggregation |
| kc_usage_pct | DOUBLE | fraction 0–1 | True | Kc usage pct; see source/table methodology and builder for exact aggregation |
| kc_ivb_inches | DOUBLE | inches | True | Kc ivb inches; see source/table methodology and builder for exact aggregation |
| ch_pitch_count | DOUBLE | count | True | Ch pitch count; see source/table methodology and builder for exact aggregation |
| ch_avg_velocity | DOUBLE | mph | True | Ch avg velocity; see source/table methodology and builder for exact aggregation |
| ch_spin_avg | DOUBLE | rpm | True | Ch spin avg; see source/table methodology and builder for exact aggregation |
| ch_horizontal_movement_avg | DOUBLE | feet | True | Ch horizontal movement avg; see source/table methodology and builder for exact aggregation |
| ch_vertical_movement_avg | DOUBLE | feet | True | Ch vertical movement avg; see source/table methodology and builder for exact aggregation |
| ch_release_x_avg | DOUBLE | feet | True | Ch release x avg; see source/table methodology and builder for exact aggregation |
| ch_release_z_avg | DOUBLE | feet | True | Ch release z avg; see source/table methodology and builder for exact aggregation |
| ch_extension_avg | DOUBLE | feet | True | Ch extension avg; see source/table methodology and builder for exact aggregation |
| ch_arm_angle_avg | DOUBLE | degrees | True | Ch arm angle avg; see source/table methodology and builder for exact aggregation |
| ch_zone_pct | DOUBLE | fraction 0–1 | True | Ch zone pct; see source/table methodology and builder for exact aggregation |
| ch_strike_pct | DOUBLE | fraction 0–1 | True | Ch strike pct; see source/table methodology and builder for exact aggregation |
| ch_spin_axis_avg | DOUBLE | degrees | True | Ch spin axis avg; see source/table methodology and builder for exact aggregation |
| ch_usage_pct | DOUBLE | fraction 0–1 | True | Ch usage pct; see source/table methodology and builder for exact aggregation |
| ch_ivb_inches | DOUBLE | inches | True | Ch ivb inches; see source/table methodology and builder for exact aggregation |
| fs_pitch_count | DOUBLE | count | True | Fs pitch count; see source/table methodology and builder for exact aggregation |
| fs_avg_velocity | DOUBLE | mph | True | Fs avg velocity; see source/table methodology and builder for exact aggregation |
| fs_spin_avg | DOUBLE | rpm | True | Fs spin avg; see source/table methodology and builder for exact aggregation |
| fs_horizontal_movement_avg | DOUBLE | feet | True | Fs horizontal movement avg; see source/table methodology and builder for exact aggregation |
| fs_vertical_movement_avg | DOUBLE | feet | True | Fs vertical movement avg; see source/table methodology and builder for exact aggregation |
| fs_release_x_avg | DOUBLE | feet | True | Fs release x avg; see source/table methodology and builder for exact aggregation |
| fs_release_z_avg | DOUBLE | feet | True | Fs release z avg; see source/table methodology and builder for exact aggregation |
| fs_extension_avg | DOUBLE | feet | True | Fs extension avg; see source/table methodology and builder for exact aggregation |
| fs_arm_angle_avg | DOUBLE | degrees | True | Fs arm angle avg; see source/table methodology and builder for exact aggregation |
| fs_zone_pct | DOUBLE | fraction 0–1 | True | Fs zone pct; see source/table methodology and builder for exact aggregation |
| fs_strike_pct | DOUBLE | fraction 0–1 | True | Fs strike pct; see source/table methodology and builder for exact aggregation |
| fs_spin_axis_avg | DOUBLE | degrees | True | Fs spin axis avg; see source/table methodology and builder for exact aggregation |
| fs_usage_pct | DOUBLE | fraction 0–1 | True | Fs usage pct; see source/table methodology and builder for exact aggregation |
| fs_ivb_inches | DOUBLE | inches | True | Fs ivb inches; see source/table methodology and builder for exact aggregation |
| sv_pitch_count | DOUBLE | count | True | Sv pitch count; see source/table methodology and builder for exact aggregation |
| sv_avg_velocity | DOUBLE | mph | True | Sv avg velocity; see source/table methodology and builder for exact aggregation |
| sv_spin_avg | DOUBLE | rpm | True | Sv spin avg; see source/table methodology and builder for exact aggregation |
| sv_horizontal_movement_avg | DOUBLE | feet | True | Sv horizontal movement avg; see source/table methodology and builder for exact aggregation |
| sv_vertical_movement_avg | DOUBLE | feet | True | Sv vertical movement avg; see source/table methodology and builder for exact aggregation |
| sv_release_x_avg | DOUBLE | feet | True | Sv release x avg; see source/table methodology and builder for exact aggregation |
| sv_release_z_avg | DOUBLE | feet | True | Sv release z avg; see source/table methodology and builder for exact aggregation |
| sv_extension_avg | DOUBLE | feet | True | Sv extension avg; see source/table methodology and builder for exact aggregation |
| sv_arm_angle_avg | DOUBLE | degrees | True | Sv arm angle avg; see source/table methodology and builder for exact aggregation |
| sv_zone_pct | DOUBLE | fraction 0–1 | True | Sv zone pct; see source/table methodology and builder for exact aggregation |
| sv_strike_pct | DOUBLE | fraction 0–1 | True | Sv strike pct; see source/table methodology and builder for exact aggregation |
| sv_spin_axis_avg | DOUBLE | degrees | True | Sv spin axis avg; see source/table methodology and builder for exact aggregation |
| sv_usage_pct | DOUBLE | fraction 0–1 | True | Sv usage pct; see source/table methodology and builder for exact aggregation |
| sv_ivb_inches | DOUBLE | inches | True | Sv ivb inches; see source/table methodology and builder for exact aggregation |
| team_id | BIGINT | identifier | False | Team id; see source/table methodology and builder for exact aggregation |
| opponent_id | BIGINT | identifier | False | Opponent id; see source/table methodology and builder for exact aggregation |
| is_starter | BOOLEAN | boolean | True | Official boxscore gamesStarted; first pitcher listed is fallback. Openers count as starters. |
| outs_recorded | INTEGER | count | True | Outs recorded; see source/table methodology and builder for exact aggregation |
| innings_pitched | DOUBLE | innings | True | Official outs recorded divided by 3; 5.2 baseball notation becomes 5.6667 |
| pitch_count | INTEGER | count | True | Pitch count; see source/table methodology and builder for exact aggregation |
| batters_faced | INTEGER | count | True | Batters faced; see source/table methodology and builder for exact aggregation |
| walks | INTEGER | count | True | Walks; see source/table methodology and builder for exact aggregation |
| strikeouts | INTEGER | count | True | Strikeouts; see source/table methodology and builder for exact aggregation |
| hit_batters | INTEGER | count | True | Hit batters; see source/table methodology and builder for exact aggregation |
| game_date | TIMESTAMP | date | False | Official game date; resumed games flagged as timing uncertain |
| scheduled_start | TIMESTAMP | numeric | False | Scheduled start; see source/table methodology and builder for exact aggregation |
| season | INTEGER | numeric | False | Season; see source/table methodology and builder for exact aggregation |
| timing_uncertain | BOOLEAN | numeric | False | Timing uncertain; see source/table methodology and builder for exact aggregation |
| team | VARCHAR | numeric | False | Team; see source/table methodology and builder for exact aggregation |
| opponent | VARCHAR | numeric | False | Opponent; see source/table methodology and builder for exact aggregation |
| pitcher_name | VARCHAR | numeric | False | Pitcher name; see source/table methodology and builder for exact aggregation |
| birth_date | TIMESTAMP | date/timestamp | False | Birth date; see source/table methodology and builder for exact aggregation |
| mlb_debut | TIMESTAMP | numeric | False | Mlb debut; see source/table methodology and builder for exact aggregation |
| throws | VARCHAR | numeric | False | Throws; see source/table methodology and builder for exact aggregation |
| position | VARCHAR | numeric | False | Position; see source/table methodology and builder for exact aggregation |
| boxscore_missing | BOOLEAN | numeric | False | Boxscore missing; see source/table methodology and builder for exact aggregation |
| pitch_count_difference | BIGINT | count | True | Pitch count difference; see source/table methodology and builder for exact aggregation |
| walk_rate | DOUBLE | fraction 0–1 | True | Official walks divided by official batters faced |
| strikeout_rate | DOUBLE | fraction 0–1 | True | Official strikeouts divided by official batters faced |
| age_at_appearance | DOUBLE | years | True | Appearance date minus birth date divided by 365.2425; avoids present-day age |
| days_since_mlb_debut | BIGINT | numeric | True | Days since mlb debut; see source/table methodology and builder for exact aggregation |
| flagged_pitch_count | BIGINT | count | True | Flagged pitch count; see source/table methodology and builder for exact aggregation |
| quality_eligible | BOOLEAN | boolean | False | Boxscore present, unambiguous game timing, >=80% tracked velocity and pitch counts within 2 |
| workload_max_date | TIMESTAMP_NS | date | False | Latest contributing historical source date; must precede prediction date; audit-only |
| rest_days | DOUBLE | days | True | Full calendar days without a pitching appearance since previous prior-day MLB outing; first observed is NULL |
| workload_history_appearances | BIGINT | count | True | Workload history appearances; see source/table methodology and builder for exact aggregation |
| pitches_prev_7d | BIGINT | pitches | True | Official pitches in [date-N days, date), excluding all current-day appearances |
| pitches_7d | BIGINT | count | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| innings_prev_7d | DOUBLE | innings | True | Official innings in [date-N days, date), excluding all current-day appearances |
| innings_7d | DOUBLE | innings | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| appearances_7d | BIGINT | count | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| workload_7d_complete | BOOLEAN | boolean | True | Workload 7d complete; see source/table methodology and builder for exact aggregation |
| pitches_prev_14d | BIGINT | pitches | True | Official pitches in [date-N days, date), excluding all current-day appearances |
| pitches_14d | BIGINT | count | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| innings_prev_14d | DOUBLE | innings | True | Official innings in [date-N days, date), excluding all current-day appearances |
| innings_14d | DOUBLE | innings | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| appearances_14d | BIGINT | count | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| workload_14d_complete | BOOLEAN | boolean | True | Workload 14d complete; see source/table methodology and builder for exact aggregation |
| pitches_prev_28d | BIGINT | pitches | True | Official pitches in [date-N days, date), excluding all current-day appearances |
| pitches_28d | BIGINT | count | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| innings_prev_28d | DOUBLE | innings | True | Official innings in [date-N days, date), excluding all current-day appearances |
| innings_28d | DOUBLE | innings | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| appearances_28d | BIGINT | count | True | Named workload in [date-N days,date) plus the current appearance; other same-day games omitted |
| workload_28d_complete | BOOLEAN | boolean | True | Workload 28d complete; see source/table methodology and builder for exact aggregation |
| starts_prev_30d | BIGINT | count | True | Starts prev 30d; see source/table methodology and builder for exact aggregation |
| starter_share_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| starter_role_prev_5 | BOOLEAN | boolean | True | At least 3 of previous 5 prior-day appearances were starts; NULL before 5 appearances |
| season_pitch_count | BIGINT | count | True | Named workload from January 1 through prior day plus current appearance |
| season_innings | DOUBLE | innings | True | Named workload from January 1 through prior day plus current appearance |
| season_appearances | BIGINT | count | True | Named workload from January 1 through prior day plus current appearance |
| season_starts | BIGINT | count | True | Named workload from January 1 through prior day plus current appearance |
| previous_season_innings | DOUBLE | innings | True | Complete official innings in calendar season t-1; missing player total stays NULL |
| current_vs_previous_season_workload | DOUBLE | ratio | True | Partial season-to-date innings divided by complete prior-season innings; not pace-adjusted |
| workload_acceleration_7d_vs_21d | DOUBLE | pitches/day | True | Prior 7-day pitches/7 minus prior days 8–28 pitches/21 |
| acute_chronic_pitch_ratio | DOUBLE | ratio | True | Current-inclusive 7-day daily pitch rate divided by current-inclusive 28-day daily pitch rate |
| prior_appearance_count | BIGINT | count | True | Prior appearance count; see source/table methodology and builder for exact aggregation |
| history_max_date | TIMESTAMP_NS | date | False | Latest contributing historical source date; must precede prediction date; audit-only |
| history_count_prev_3 | BIGINT | count | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| avg_fb_velo_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_sd_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_sd_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| arm_angle_avg_prev_3 | DOUBLE | degrees | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| zone_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_avg_velocity_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_spin_avg_prev_3 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_release_x_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_release_z_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_extension_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| history_count_prev_5 | BIGINT | count | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| avg_fb_velo_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_sd_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_sd_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| arm_angle_avg_prev_5 | DOUBLE | degrees | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| zone_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_avg_velocity_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_spin_avg_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_release_x_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_release_z_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_extension_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| history_count_prev_10 | BIGINT | count | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| avg_fb_velo_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_sd_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_sd_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| arm_angle_avg_prev_10 | DOUBLE | degrees | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| zone_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_avg_velocity_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_spin_avg_prev_10 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_release_x_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_release_z_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_extension_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fb_velo_change_vs_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fb_velo_slope_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_slope_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_slope_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_sd_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_x_sd_slope_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_sd_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| release_z_sd_slope_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| extension_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| extension_slope_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| arm_angle_change_vs_prev_5 | DOUBLE | degrees | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| arm_angle_slope_prev_5 | DOUBLE | degrees | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| spin_change_vs_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| spin_slope_prev_5 | DOUBLE | rpm | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| zone_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| zone_pct_slope_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| baseline_max_date | TIMESTAMP_NS | date | False | Latest contributing historical source date; must precede prediction date; audit-only |
| avg_fb_velo_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| avg_fb_velo_baseline_mean | DOUBLE | mph | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| avg_fb_velo_baseline_sd | DOUBLE | mph | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| velo_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| release_x_avg_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| release_x_avg_baseline_mean | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_x_avg_baseline_sd | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_x_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| release_z_avg_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| release_z_avg_baseline_mean | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_z_avg_baseline_sd | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_z_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| release_x_sd_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| release_x_sd_baseline_mean | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_x_sd_baseline_sd | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_x_sd_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| release_z_sd_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| release_z_sd_baseline_mean | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_z_sd_baseline_sd | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| release_z_sd_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| extension_avg_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| extension_avg_baseline_mean | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| extension_avg_baseline_sd | DOUBLE | feet | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| extension_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| arm_angle_avg_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| arm_angle_avg_baseline_mean | DOUBLE | degrees | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| arm_angle_avg_baseline_sd | DOUBLE | degrees | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| arm_angle_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| spin_avg_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| spin_avg_baseline_mean | DOUBLE | rpm | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| spin_avg_baseline_sd | DOUBLE | rpm | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| spin_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| zone_pct_baseline_count | BIGINT | count | True | Valid observations in last 20 strictly prior-day outings for the named metric |
| zone_pct_baseline_mean | DOUBLE | fraction 0–1 | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| zone_pct_baseline_sd | DOUBLE | fraction 0–1 | True | Named mean or sample SD over last 20 strictly prior-day outings; at least 5 valid observations |
| zone_pct_z | DOUBLE | standard deviations | True | Current measurement minus historical pitcher mean, divided by historical SD; minimum 5 valid prior observations, SD > 1e-8 |
| label_eligible | BOOLEAN | boolean | False | Retrospective at-risk eligibility with sufficient source retrieval; not a feature |
| post_known_injury | BOOLEAN | boolean | False | Appearance during a known active arm-IL episode before documented return; never a healthy negative |
| uncertain_injury_timing | BOOLEAN | numeric | False | Uncertain injury timing; see source/table methodology and builder for exact aggregation |
| risk_set_stop_date | TIMESTAMP_NS | date | False | Current active arm-IL episode start or next arm-IL episode start; NULL if none observed |
| next_known_ucl_onset | TIMESTAMP_NS | numeric | False | Next known ucl onset; see source/table methodology and builder for exact aggregation |
| next_injury_id | VARCHAR | identifier | False | Legacy alias of next_injury_event_id in historical builds |
| last_observed_appearance | TIMESTAMP_NS | numeric | False | Last observed appearance; see source/table methodology and builder for exact aggregation |
| label_observation_end | TIMESTAMP_NS | numeric | False | Label observation end; see source/table methodology and builder for exact aggregation |
| label_sources_complete | BOOLEAN | boolean | False | Both configured Roegele tabs and MLB schedule/transaction/boxscore requests succeeded |
| label_onset_basis | VARCHAR | numeric | False | Label onset basis; see source/table methodology and builder for exact aggregation |
| ucl_injury_next_14d | BOOLEAN | nullable boolean target | False | Confirmed public-evidence onset in (appearance date, appearance date + N days]; negatives require complete follow-up; otherwise NULL |
| ucl_injury_next_30d | BOOLEAN | nullable boolean target | False | Confirmed public-evidence onset in (appearance date, appearance date + N days]; negatives require complete follow-up; otherwise NULL |
| ucl_injury_next_60d | BOOLEAN | nullable boolean target | False | Confirmed public-evidence onset in (appearance date, appearance date + N days]; negatives require complete follow-up; otherwise NULL |
| ucl_injury_next_90d | BOOLEAN | nullable boolean target | False | Confirmed public-evidence onset in (appearance date, appearance date + N days]; negatives require complete follow-up; otherwise NULL |
| ucl_injury_next_3_appearances | BOOLEAN | nullable boolean target | False | Confirmed public-evidence onset before the Nth subsequent appearance, capped at 90 days; NULL if censored |
| ucl_injury_next_5_appearances | BOOLEAN | nullable boolean target | False | Confirmed public-evidence onset before the Nth subsequent appearance, capped at 90 days; NULL if censored |
| training_eligible | BOOLEAN | boolean | False | Quality-eligible and at risk before known/uncertain injury; target still must be nonnull |
| starter_source | VARCHAR | category | False | Official gamesStarted or official pitching-order fallback; never pitch count |
| pitching_order | INTEGER | numeric | False | Pitching order; see source/table methodology and builder for exact aggregation |
| game_type | VARCHAR | numeric | False | Game type; see source/table methodology and builder for exact aggregation |
| appearance_throws | VARCHAR | numeric | False | Appearance throws; see source/table methodology and builder for exact aggregation |
| throws_left | BOOLEAN | boolean | True | Left hand observed in this appearance; NULL for unavailable or conflicting hands |
| official_starters_on_team | BIGINT | numeric | False | Official starters on team; see source/table methodology and builder for exact aggregation |
| first_observed_pitcher_id | BIGINT | identifier | False | First observed pitcher id; see source/table methodology and builder for exact aggregation |
| starter_crosscheck_disagrees | BOOLEAN | numeric | False | Starter crosscheck disagrees; see source/table methodology and builder for exact aggregation |
| starter_status | VARCHAR | category | False | starter, relief or ambiguous after official-team-count and first-observed-pitcher checks |
| valid_starting_appearance | BOOLEAN | boolean | False | Regular-season unambiguous official start passing current appearance quality rules |
| observed_prior_mlb_appearances | BIGINT | count | True | MLB pitching appearances strictly before this date within observed history |
| observed_prior_mlb_starts | BIGINT | count | True | MLB starts strictly before this date within observed history; left-truncated at study start |
| documented_prior_ucl_count | BIGINT | count | True | Count of documented prior procedures, using conservative upper date bounds; not count of unique clinical episodes |
| documented_prior_ucl | BOOLEAN | boolean | True | At least one documented confirmed procedure with upper date bound strictly before this appearance; missing ascertainment is possible |
| appearances_prev_7d | BIGINT | count | True | Appearances prev 7d; see source/table methodology and builder for exact aggregation |
| appearances_prev_14d | BIGINT | count | True | Appearances prev 14d; see source/table methodology and builder for exact aggregation |
| appearances_prev_28d | BIGINT | count | True | Appearances prev 28d; see source/table methodology and builder for exact aggregation |
| fb_velo_sd_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| horizontal_movement_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| vertical_movement_avg_prev_3 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| location_variability_prev_3 | DOUBLE | numeric | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| strike_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| first_pitch_strike_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fastball_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_usage_pct_prev_3 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fb_velo_sd_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| horizontal_movement_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| vertical_movement_avg_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| location_variability_prev_5 | DOUBLE | numeric | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| strike_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| first_pitch_strike_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fastball_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_usage_pct_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fb_velo_sd_prev_10 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| horizontal_movement_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| vertical_movement_avg_prev_10 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| location_variability_prev_10 | DOUBLE | numeric | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| strike_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| first_pitch_strike_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fastball_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_usage_pct_prev_10 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fb_velo_change_vs_prev_3 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fb_velo_sd_change_vs_prev_5 | DOUBLE | mph | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| horizontal_movement_avg_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| vertical_movement_avg_change_vs_prev_5 | DOUBLE | feet | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| location_variability_change_vs_prev_5 | DOUBLE | numeric | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| strike_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| first_pitch_strike_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fastball_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ff_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| si_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fc_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sl_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| st_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| cu_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| kc_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| ch_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| fs_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| sv_usage_pct_change_vs_prev_5 | DOUBLE | fraction 0–1 | True | Named metric averaged over previous N prior-day appearances; full N outings and at least max(2,floor(N/2)) nonnull values |
| next_injury_event_id | VARCHAR | identifier | False | Next linked event used only for outcome/evaluation audit, never X |
