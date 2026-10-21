CREATE OR REPLACE VIEW model_ready_pitcher_appearances AS
SELECT * FROM model_features;
CREATE OR REPLACE VIEW training_pitcher_appearances AS
SELECT * FROM model_features
WHERE training_eligible AND ucl_injury_next_30d IS NOT NULL;
CREATE OR REPLACE VIEW starter_training_appearances AS
SELECT * FROM training_pitcher_appearances WHERE is_starter;
CREATE OR REPLACE VIEW model_ready_starting_pitcher_appearances AS
SELECT * FROM model_features WHERE valid_starting_appearance;
CREATE OR REPLACE VIEW training_starting_pitcher_appearances AS
SELECT * FROM model_ready_starting_pitcher_appearances
WHERE training_eligible AND ucl_injury_next_30d IS NOT NULL;
