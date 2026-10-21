"""Export regressions use tiny temporary databases, never the research database."""
import hashlib
import json

import duckdb
import pytest

import generate_exploratory_csvs as exporter


def make_source(path):
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE pitches AS
        SELECT * FROM (VALUES
        (1, 1, 1, 123456, TIMESTAMP '2023-12-31', 'FF', 99., true),
        (2, 1, 1, 123456, TIMESTAMP '2024-01-01', 'FF', 90., true),
        (2, 1, 2, 123456, TIMESTAMP '2024-01-01', 'FF', 94., true),
        (2, 1, 3, 123456, TIMESTAMP '2024-01-01', 'CH', 80., true),
        (2, 1, 4, 123456, TIMESTAMP '2024-01-01', NULL, NULL, false),
        (3, 1, 1, 765432, TIMESTAMP '2025-12-31 23:59:59', 'SL', 85., true),
        (3, 1, 2, 765432, TIMESTAMP '2025-12-31 23:59:59', 'FF', NULL, true),
        (3, 1, 3, 765432, TIMESTAMP '2025-12-31 23:59:59', 'FF', 900., true),
        (4, 1, 1, 123456, TIMESTAMP '2026-01-01', 'FF', 98., true),
        (5, 1, 1, 123456, TIMESTAMP '2024-10-01', 'FF', 98., true)
        ) t(game_id,at_bat_number,pitch_number,pitcher_id,game_date,pitch_type,release_speed,is_actual_pitch)""")
    con.execute("""CREATE TABLE games AS SELECT DISTINCT game_id,game_date,
        CASE WHEN game_id=5 THEN 'P' ELSE 'R' END AS game_type FROM pitches""")
    con.execute("""CREATE TABLE appearances AS SELECT DISTINCT p.game_id,p.pitcher_id,p.game_date,
        g.game_type, p.game_id=2 AS is_starter, 'boxscore' AS starter_source,
        3 AS total_pitches FROM pitches p JOIN games g USING(game_id)""")
    con.execute("""CREATE TABLE model_features AS SELECT a.*,
        TIMESTAMP '2023-09-25' AS history_max_date,
        TIMESTAMP '2023-09-25' AS baseline_max_date,
        TIMESTAMP '2023-09-25' AS workload_max_date,
        93.25 AS avg_fb_velo_prev_5, false AS training_eligible,
        CASE WHEN game_id=3 THEN true ELSE NULL END AS ucl_injury_next_30d,
        CASE WHEN game_id=3 THEN 'event-1' ELSE NULL END AS next_injury_event_id
        FROM appearances a""")
    con.close()


def test_export_join_sort_dates_optional_fields_labels_and_source_preservation(tmp_path):
    source = tmp_path / "source.duckdb"
    out = tmp_path / "exports"
    make_source(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert exporter.main(["--database", str(source), "--output-dir", str(out), "--pitch-sample-rows", "4"]) == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["pitch_sample_quotas"] == {"2024": 2, "2025": 2}
    assert manifest["any_injury_positive_model_rows"] == 1
    con = duckdb.connect()
    app = exporter.csv_relation(out / "pitcher_appearances_2024_2025.csv",
                               manifest["exports"]["pitcher_appearances_2024_2025.csv"]["columns"])
    # Exercises the original ambiguous ORDER BY through the joined appearance query.
    assert con.execute(f"SELECT game_id,total_pitches,fastball_count,avg_fb_velo FROM {app} ORDER BY game_id").fetchall() == [
        (2, 3, 2, 92.0), (3, 3, 2, None)]
    assert con.execute(f"SELECT fb_velo_sd FROM {app} WHERE game_id=2").fetchone()[0] == pytest.approx(2 ** 1.5)
    model = exporter.csv_relation(out / "model_features_2024_2025.csv",
                                 manifest["exports"]["model_features_2024_2025.csv"]["columns"])
    assert con.execute(f"SELECT game_id,season,ucl_injury_next_30d,avg_fb_velo_prev_5 FROM {model} ORDER BY game_id").fetchall() == [
        (2, 2024, None, 93.25), (3, 2025, True, 93.25)]
    assert con.execute(f"SELECT is_starter,training_eligible FROM {model} WHERE ucl_injury_next_30d").fetchone() == (False, False)
    assert manifest["history_validation"]["history_max_date"]["rows_using_pre_2024_history"] == 2
    assert "arm_angle" in manifest["missing_optional_columns"]["appearance_raw_fields"]
    assert "ucl_injury_next_30d" not in manifest["model_predictor_columns"]
    assert not (source.parent / (source.name + ".wal")).exists()
    con.close()


def test_sampling_is_stable_across_scan_order_and_thread_count(tmp_path):
    source = tmp_path / "source.duckdb"
    make_source(source)
    con = duckdb.connect(str(source), read_only=True)
    _, relations = exporter.inspect_schema(con)
    raw = next(r for r in relations if r.name == "pitches")
    games = next(r for r in relations if r.name == "games")
    raw_query = exporter.raw_window_query(raw, games, [])
    q1, quotas, _, _ = exporter.pitch_sample_query(con, raw, raw_query, 4, 42, {})
    con.execute("SET threads=1")
    first = sorted(con.execute(q1).fetchall())
    q2, _, _, _ = exporter.pitch_sample_query(con, raw, raw_query + " ORDER BY pitch_number DESC,game_id DESC", 4, 42, {})
    con.execute("SET threads=4")
    assert sorted(con.execute(q2).fetchall()) == first
    assert quotas == {2024: 2, 2025: 2}
    _, capped, _, _ = exporter.pitch_sample_query(con, raw, raw_query, 100, 42, {})
    assert capped == {2024: 4, 2025: 3}
    con.close()


def test_duplicate_pitch_keys_fail_before_export(tmp_path):
    source = tmp_path / "source.duckdb"
    make_source(source)
    con = duckdb.connect(str(source))
    con.execute("INSERT INTO pitches SELECT * FROM pitches WHERE game_id=2 AND pitch_number=1")
    con.close()
    out = tmp_path / "exports"
    with pytest.raises(ValueError, match="duplicate keys"):
        exporter.main(["--database", str(source), "--output-dir", str(out)])
    assert not list(out.glob("*.csv"))


def test_fallback_without_engineered_table_does_not_invent_features(tmp_path):
    source = tmp_path / "source.duckdb"
    make_source(source)
    con = duckdb.connect(str(source))
    con.execute("DROP TABLE model_features")
    con.close()
    out = tmp_path / "exports"
    exporter.main(["--database", str(source), "--output-dir", str(out), "--pitch-sample-rows", "4"])
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["selected_sources"]["model_features"] is None
    assert manifest["exports"]["model_features_2024_2025.csv"]["rows"] == 2
    assert "ucl_injury_next_30d" not in manifest["exports"]["model_features_2024_2025.csv"]["columns"]
    assert any("No finished feature table" in n for n in manifest["notes"])


def test_invalid_path_does_not_create_database_or_outputs(tmp_path):
    with pytest.raises(SystemExit) as error:
        exporter.main(["--database", str(tmp_path / "missing.duckdb"), "--output-dir", str(tmp_path / "exports")])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
