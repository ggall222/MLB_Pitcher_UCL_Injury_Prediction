-- Savant numbers administrative automatic strikes/balls; MLB numbers physical pitches.
-- Align physical ordinals within each plate appearance, not the two native pitch numbers.
CREATE OR REPLACE VIEW aligned_pitch_tracking AS
WITH physical AS (
    SELECT game_id, at_bat_number, pitch_number, release_speed,
           row_number() OVER (PARTITION BY game_id,at_bat_number ORDER BY pitch_number) AS physical_ordinal,
           count(*) OVER (PARTITION BY game_id,at_bat_number) AS physical_count
    FROM pitches WHERE is_actual_pitch
), native AS (
    SELECT *, row_number() OVER (PARTITION BY game_id,at_bat_number ORDER BY pitch_number) AS physical_ordinal,
           count(*) OVER (PARTITION BY game_id,at_bat_number) AS physical_count,
           TRY_CAST(json_extract_string(raw_json,'$.pitchData.startSpeed') AS DOUBLE) AS mlb_speed
    FROM pitch_tracking
), matched AS (
    SELECT p.game_id,p.at_bat_number,p.pitch_number,t.pitch_number AS mlb_pitch_number,p.physical_ordinal,
           p.physical_count=t.physical_count AS counts_match,
           abs(p.release_speed-t.mlb_speed) AS velocity_difference,
           p.physical_count=t.physical_count AND
             (p.release_speed IS NULL OR t.mlb_speed IS NULL OR abs(p.release_speed-t.mlb_speed)<=0.5) AS alignment_valid,
           t.ivb_inches
    FROM physical p JOIN native t USING(game_id,at_bat_number,physical_ordinal)
)
SELECT game_id,at_bat_number,pitch_number,mlb_pitch_number,physical_ordinal,
       counts_match,velocity_difference,alignment_valid,
       CASE WHEN alignment_valid THEN ivb_inches END AS ivb_inches
FROM matched;
