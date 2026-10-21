-- Core contract. NULL means unknown, never an invented zero.
CREATE TABLE IF NOT EXISTS teams (
    team_id BIGINT PRIMARY KEY, team_name VARCHAR, abbreviation VARCHAR,
    raw_json JSON, retrieved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pitchers (
    pitcher_id BIGINT PRIMARY KEY CHECK (pitcher_id > 0), pitcher_name VARCHAR,
    birth_date DATE, mlb_debut DATE, throws VARCHAR, position VARCHAR,
    fangraphs_id VARCHAR, metadata_source VARCHAR, raw_json JSON, retrieved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS games (
    game_id BIGINT PRIMARY KEY, game_date DATE NOT NULL, scheduled_start TIMESTAMP,
    season INTEGER, game_type VARCHAR, status VARCHAR, home_team_id BIGINT REFERENCES teams(team_id),
    away_team_id BIGINT REFERENCES teams(team_id), double_header VARCHAR, game_number INTEGER,
    timing_uncertain BOOLEAN DEFAULT FALSE, raw_json JSON
);
CREATE TABLE IF NOT EXISTS pitches (
    game_id BIGINT REFERENCES games(game_id), at_bat_number INTEGER, pitch_number INTEGER,
    pitcher_id BIGINT REFERENCES pitchers(pitcher_id), game_date DATE NOT NULL, batter_id BIGINT,
    inning INTEGER, inning_topbot VARCHAR, stand VARCHAR, p_throws VARCHAR, balls INTEGER, strikes INTEGER,
    pitch_type VARCHAR, description VARCHAR, events VARCHAR, pitch_result VARCHAR,
    release_speed DOUBLE, effective_speed DOUBLE, release_pos_x DOUBLE, release_pos_y DOUBLE,
    release_pos_z DOUBLE, release_extension DOUBLE, arm_angle DOUBLE, release_spin_rate DOUBLE,
    pfx_x DOUBLE, pfx_z DOUBLE, spin_axis DOUBLE, plate_x DOUBLE, plate_z DOUBLE,
    sz_top DOUBLE, sz_bot DOUBLE, zone INTEGER, outs_when_up INTEGER,
    api_break_z_with_gravity DOUBLE, is_hbp BOOLEAN, is_actual_pitch BOOLEAN,
    quality_flags VARCHAR, raw_json JSON NOT NULL,
    PRIMARY KEY (game_id, at_bat_number, pitch_number)
);
CREATE TABLE IF NOT EXISTS boxscore_appearances (
    game_id BIGINT REFERENCES games(game_id), pitcher_id BIGINT REFERENCES pitchers(pitcher_id),
    team_id BIGINT REFERENCES teams(team_id), opponent_id BIGINT REFERENCES teams(team_id),
    is_starter BOOLEAN, outs_recorded INTEGER, innings_pitched DOUBLE,
    pitch_count INTEGER, batters_faced INTEGER, walks INTEGER, strikeouts INTEGER,
    hit_batters INTEGER, raw_json JSON,
    PRIMARY KEY (game_id, pitcher_id)
);
CREATE TABLE IF NOT EXISTS roster_snapshots (
    team_id BIGINT REFERENCES teams(team_id), snapshot_date DATE, player_id BIGINT,
    player_name VARCHAR, position VARCHAR, status VARCHAR, raw_json JSON,
    PRIMARY KEY (team_id, snapshot_date, player_id)
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT PRIMARY KEY, player_id BIGINT, player_name VARCHAR,
    transaction_date DATE, effective_date DATE, resolution_date DATE,
    from_team_id BIGINT, to_team_id BIGINT, type_code VARCHAR, type_description VARCHAR,
    description VARCHAR, is_il_placement BOOLEAN, is_il_activation BOOLEAN,
    is_elbow_related BOOLEAN, raw_json JSON, source_url VARCHAR
);
CREATE TABLE IF NOT EXISTS season_pitching (
    pitcher_id BIGINT REFERENCES pitchers(pitcher_id), season INTEGER,
    innings_pitched DOUBLE, pitch_count INTEGER, games_played INTEGER, games_started INTEGER,
    raw_json JSON, PRIMARY KEY (pitcher_id, season)
);
CREATE TABLE IF NOT EXISTS injuries (
    injury_id VARCHAR PRIMARY KEY, pitcher_id BIGINT REFERENCES pitchers(pitcher_id),
    player_name VARCHAR, fangraphs_id VARCHAR, team VARCHAR, level VARCHAR, position VARCHAR,
    throws VARCHAR, age DOUBLE, procedure_category VARCHAR, repeat_procedure BOOLEAN,
    surgery_date DATE, surgery_date_raw VARCHAR, surgery_date_precision VARCHAR,
    return_date DATE, recovery_months DOUBLE, surgeon VARCHAR, source_url VARCHAR,
    source_tab VARCHAR, source_reference VARCHAR, identity_method VARCHAR,
    confirmed_ucl_surgery BOOLEAN, first_reported_elbow_date DATE, il_placement_date DATE,
    diagnosis_date DATE, known_onset_date DATE, onset_basis VARCHAR, onset_transaction_id BIGINT,
    last_pre_onset_appearance DATE, last_healthy_mlb_appearance DATE,
    quality_flags VARCHAR, raw_json JSON, retrieved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ingestion_state (
    source VARCHAR, item_key VARCHAR, status VARCHAR, row_count BIGINT, detail VARCHAR,
    updated_at TIMESTAMP, PRIMARY KEY (source, item_key)
);
CREATE TABLE IF NOT EXISTS ingestion_issues (
    issue_id VARCHAR PRIMARY KEY, source VARCHAR, item_key VARCHAR, reason VARCHAR,
    raw_json JSON, recorded_at TIMESTAMP
);
-- Wide derived schemas are deterministically generated from config by the builders.
-- reports/derived_schema.sql records the actual persisted DDL after every build.
ALTER TABLE injuries ADD COLUMN IF NOT EXISTS surgery_date_lower DATE;
ALTER TABLE injuries ADD COLUMN IF NOT EXISTS surgery_date_upper DATE;
-- Optional supplemental telemetry; Statcast remains the primary pitch source.
CREATE TABLE IF NOT EXISTS pitch_tracking (
    game_id BIGINT REFERENCES games(game_id), at_bat_number INTEGER, pitch_number INTEGER,
    ivb_inches DOUBLE, horizontal_break_inches DOUBLE, vertical_break_inches DOUBLE,
    pitch_start TIMESTAMP, pitch_end TIMESTAMP, raw_json JSON,
    PRIMARY KEY (game_id, at_bat_number, pitch_number)
);
ALTER TABLE boxscore_appearances ADD COLUMN IF NOT EXISTS starter_source VARCHAR;
ALTER TABLE boxscore_appearances ADD COLUMN IF NOT EXISTS pitching_order INTEGER;
CREATE TABLE IF NOT EXISTS historical_checkpoints (
    season INTEGER PRIMARY KEY, phase VARCHAR, started_at TIMESTAMP,
    completed_at TIMESTAMP, pitches BIGINT, appearances BIGINT, detail VARCHAR
);

CREATE TABLE IF NOT EXISTS team_seasons (
    team_id BIGINT REFERENCES teams(team_id), season INTEGER, team_name VARCHAR,
    abbreviation VARCHAR, raw_json JSON, retrieved_at TIMESTAMP, PRIMARY KEY(team_id,season)
);
-- MLB's abstract Final category also contains unplayed cancellations. Keep raw
-- status evidence and distinguish these schedules from actual completed games.
UPDATE games SET status='Cancelled'
WHERE json_extract_string(raw_json,'$.status.codedGameState')='C'
   OR json_extract_string(raw_json,'$.status.detailedState')='Cancelled';
UPDATE ingestion_state AS s SET status='not_applicable',row_count=0,
    detail='MLB detailed status is Cancelled; no pitching appearance occurred'
WHERE source='boxscore' AND status='failed'
  AND EXISTS(SELECT 1 FROM games g WHERE g.game_id=try_cast(s.item_key AS BIGINT) AND g.status='Cancelled');
UPDATE ingestion_state AS s SET status='not_applicable',row_count=0,
    detail='Only cancelled MLB schedule entries on this date; raw schedule retained'
WHERE source='statcast' AND status='failed'
  AND EXISTS(SELECT 1 FROM games g WHERE g.game_date=try_cast(s.item_key AS DATE) AND g.status='Cancelled')
  AND NOT EXISTS(SELECT 1 FROM games g WHERE g.game_date=try_cast(s.item_key AS DATE) AND g.status='Final');
