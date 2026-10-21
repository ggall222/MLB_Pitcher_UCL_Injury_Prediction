#!/usr/bin/env python3
"""Export 2024–2025 exploration CSVs without changing the source DuckDB.

Run with the project's environment:
    .venv/bin/python generate_exploratory_csvs.py
    .venv/bin/python generate_exploratory_csvs.py --database /path/to/mlb_ucl.duckdb

Only DuckDB and the Python standard library are required. Queries operate on disk;
no pandas frames or full-database copies are created. Existing derived CSVs in
the chosen output directory are replaced atomically after validation.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import duckdb


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "database/mlb_ucl.duckdb"
DEFAULT_OUTPUT = ROOT / "data/exports/exploratory_2024_2025"
START_DATE = "2024-01-01"
END_DATE_EXCLUSIVE = "2026-01-01"  # Includes every time on December 31, 2025.
SEASONS = (2024, 2025)
FASTBALL_TYPES = ("FF", "SI", "FC")  # Four-seam, sinker, cutter; excludes off-speed.
PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV")
LABELS = ["ucl_injury_next_" + h for h in
          ("14d", "30d", "60d", "90d", "3_appearances", "5_appearances")]
CONTEXT = """pitcher_name team_id opponent_id team opponent game_type is_starter
starter_status starter_source valid_starting_appearance appearance_throws
quality_eligible timing_uncertain boxscore_missing""".split()
AUDIT = """training_eligible label_eligible post_known_injury uncertain_injury_timing
label_sources_complete label_onset_basis next_injury_event_id next_injury_id
next_known_ucl_onset risk_set_stop_date last_observed_appearance label_observation_end
history_max_date baseline_max_date workload_max_date""".split()

# The project's 65-feature v1 selection, plus a limited set of interpretable
# rolling means, slopes, history counts, and current measurements. No automatic
# numeric-column selection: future label/audit information must never become X.
V1_FEATURES = """avg_fb_velo fb_velo_sd fb_velo_change_vs_prev_3
fb_velo_change_vs_prev_5 velo_drop_game velo_z fb_velo_slope_prev_5
avg_fb_velo_baseline_mean release_x_avg release_z_avg release_x_sd release_z_sd
extension_avg extension_sd release_x_change_vs_prev_5 release_z_change_vs_prev_5
extension_change_vs_prev_5 release_x_z release_z_z extension_z arm_angle_avg
arm_angle_sd arm_angle_change_vs_prev_5 arm_angle_z spin_avg spin_change_vs_prev_5
spin_z horizontal_movement_avg vertical_movement_avg
horizontal_movement_avg_change_vs_prev_5 vertical_movement_avg_change_vs_prev_5
fastball_usage_pct fastball_usage_pct_change_vs_prev_5 sl_usage_pct
sl_usage_pct_change_vs_prev_5 ch_usage_pct cu_usage_pct zone_pct location_variability
strike_pct first_pitch_strike_pct zone_pct_change_vs_prev_5
location_variability_change_vs_prev_5 walk_rate hbp_rate total_pitches
innings_pitched rest_days pitches_prev_7d pitches_prev_14d pitches_prev_28d
appearances_prev_14d appearances_prev_28d season_pitch_count season_innings
previous_season_innings workload_acceleration_7d_vs_21d acute_chronic_pitch_ratio
starts_prev_30d starter_share_prev_5 age_at_appearance throws_left
documented_prior_ucl observed_prior_mlb_starts days_since_mlb_debut""".split()


def qi(value):
    """Quote an inspected SQL identifier, including unusual schema/column names."""
    return '"' + value.replace('"', '""') + '"'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def window(date_expression):
    return (f"{date_expression} >= DATE '{START_DATE}' AND "
            f"{date_expression} < DATE '{END_DATE_EXCLUSIVE}'")


@dataclass
class Relation:
    schema: str
    name: str
    kind: str
    columns: dict

    @property
    def sql(self):
        return qi(self.schema) + "." + qi(self.name)


def inspect_schema(con):
    """Inspect every available schema/table before choosing source relations."""
    schemas = con.execute("SELECT DISTINCT catalog_name, schema_name FROM "
                          "information_schema.schemata ORDER BY 1, 2").fetchall()
    tables = con.execute("SELECT table_schema, table_name, table_type FROM "
                         "information_schema.tables ORDER BY 1, 2").fetchall()
    columns = con.execute("SELECT table_schema, table_name, column_name, data_type "
                          "FROM information_schema.columns ORDER BY "
                          "table_schema, table_name, ordinal_position").fetchall()
    lookup = {}
    for schema, table, column, dtype in columns:
        lookup.setdefault((schema, table), {})[column] = dtype
    relations = [Relation(s, t, k, lookup[(s, t)]) for s, t, k in tables]
    print(f"Inspected {len(schemas)} catalog/schema entries and {len(relations)} tables/views.",
          flush=True)
    return schemas, relations


def choose(relations, names, required, signature=(), mandatory=False):
    """Prefer actual project names, validating columns instead of assuming them."""
    compatible = [r for r in relations if set(required) <= set(r.columns)]
    for name in names:
        found = [r for r in compatible if r.name == name]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"Ambiguous source {name}: found in multiple schemas")
    found = [r for r in compatible if r.kind == "BASE TABLE"
             and signature and set(signature) <= set(r.columns)]
    if len(found) == 1:
        return found[0]
    if mandatory:
        raise ValueError(f"Cannot identify {names[0]}; required fields: {sorted(required)}. "
                         "Inspect the database schema or supply the correct --database.")
    return None


def select_optional(relation, names, prefix, missing, category):
    available = [n for n in dict.fromkeys(names) if n in relation.columns]
    missing[category] = [n for n in dict.fromkeys(names) if n not in relation.columns]
    return [f"{prefix}.{qi(n)}" for n in available], available


def assert_unique(con, query, keys, name):
    columns = ", ".join(qi(k) for k in keys)
    count = con.execute(f"SELECT count(*) FROM (SELECT {columns} FROM ({query}) "
                        f"GROUP BY {columns} HAVING count(*) > 1)").fetchone()[0]
    if count:
        raise ValueError(f"{name} has {count:,} duplicate keys; refusing a multiplied export")


def raw_window_query(raw, games, notes):
    query = f"SELECT p.* FROM {raw.sql} p WHERE {window('p.game_date')}"
    if games and "game_type" in games.columns:
        # EXISTS avoids duplicate-producing joins. The pipeline's games source is
        # MLB (sportId=1); make the regular-season restriction explicit here.
        query += (f" AND EXISTS (SELECT 1 FROM {games.sql} g WHERE "
                  "g.game_id=p.game_id AND g.game_type='R')")
    else:
        notes.append("Game-type metadata unavailable: raw population cannot be explicitly restricted to regular season.")
    return query


def appearance_query(raw, raw_query, metadata, missing, notes):
    """Reaggregate physical pitches; clean only derived measurements, never raw."""
    if "is_actual_pitch" in raw.columns:
        physical = "coalesce(is_actual_pitch, false)"
    elif "description" in raw.columns:
        physical = "coalesce(description NOT IN ('automatic_ball', 'automatic_strike'), true)"
        notes.append("Physical pitches inferred from description; no is_actual_pitch field.")
    else:
        physical = "true"
        notes.append("No physical-pitch flag/description: appearance counts include all raw events.")
    # Match the existing project's documented plausibility bounds. Invalid
    # measurements remain NULL, sample SD requires >=2 observed measurements.
    bounds = {"release_speed": (30, 110), "release_pos_x": (-7, 7),
              "release_pos_z": (0, 9), "release_extension": (2, 9),
              "release_spin_rate": (0, 4500), "arm_angle": (-90, 180)}
    replacements = [f"CASE WHEN {qi(c)} BETWEEN {lo} AND {hi} THEN {qi(c)} END AS {qi(c)}"
                    for c, (lo, hi) in bounds.items() if c in raw.columns]
    clean = "*" + (" REPLACE (" + ", ".join(replacements) + ")" if replacements else "")
    agg = ["game_id", "pitcher_id", "CAST(min(game_date) AS DATE) AS game_date",
           "CAST(year(min(game_date)) AS INTEGER) AS season", "count(*) AS total_pitches"]
    missing["appearance_raw_fields"] = []

    def metric(field, expression, alias):
        if field in raw.columns:
            agg.append(f"{expression} AS {qi(alias)}")
        else:
            missing["appearance_raw_fields"].append(field)

    metric("at_bat_number", "count(DISTINCT at_bat_number)", "statcast_batters_faced")
    for field, stem in [("release_pos_x", "release_x"), ("release_pos_z", "release_z"),
                        ("release_extension", "extension"), ("arm_angle", "arm_angle")]:
        metric(field, f"avg({qi(field)})", stem + "_avg")
        metric(field, f"stddev_samp({qi(field)})", stem + "_sd")
    for field, alias in [("release_speed", "avg_velocity"), ("release_spin_rate", "spin_avg"),
                         ("pfx_x", "horizontal_movement_avg"), ("pfx_z", "vertical_movement_avg"),
                         ("plate_x", "plate_x_avg"), ("plate_z", "plate_z_avg")]:
        metric(field, f"avg({qi(field)})", alias)
    metric("release_speed", "count(release_speed)", "velocity_measured_count")
    metric("release_speed", "count(*) FILTER (WHERE release_speed >= 95)", "high_velocity_pitch_count")
    metric("arm_angle", "count(arm_angle)", "arm_angle_measured_count")
    metric("zone", "avg(CASE WHEN zone IS NOT NULL THEN (zone BETWEEN 1 AND 9)::INT END)", "zone_pct")
    metric("pitch_result", "avg(CASE WHEN pitch_result IS NOT NULL THEN (pitch_result IN ('S','X'))::INT END)", "strike_pct")
    if "pitch_number" in raw.columns:
        metric("pitch_result", "avg(CASE WHEN pitch_number=1 AND pitch_result IS NOT NULL "
               "THEN (pitch_result IN ('S','X'))::INT END)", "first_pitch_strike_pct")
    for axis in ("x", "z"):
        metric("plate_" + axis, f"stddev_samp(plate_{axis})", f"plate_{axis}_sd")
    if {"plate_x", "plate_z"} <= set(raw.columns):
        agg.append("sqrt(pow(stddev_samp(plate_x),2)+pow(stddev_samp(plate_z),2)) AS location_variability")
    if "pitch_type" in raw.columns:
        fb = "pitch_type IN (" + ",".join(qs(t) for t in FASTBALL_TYPES) + ")"
        agg.extend([f"count(*) FILTER (WHERE {fb}) AS fastball_count",
                    f"count(*) FILTER (WHERE {fb})::DOUBLE/count(*) AS fastball_usage_pct"])
        for func, alias in [("avg", "avg_fb_velo"), ("max", "max_fb_velo"), ("stddev_samp", "fb_velo_sd")]:
            metric("release_speed", f"{func}(release_speed) FILTER (WHERE {fb})", alias)
        for pt in PITCH_TYPES:
            test = f"pitch_type={qs(pt)}"
            agg.extend([f"count(*) FILTER (WHERE {test}) AS {pt.lower()}_pitch_count",
                        f"count(*) FILTER (WHERE {test})::DOUBLE/count(*) AS {pt.lower()}_usage_pct"])
            metric("release_spin_rate", f"avg(release_spin_rate) FILTER (WHERE {test})", pt.lower() + "_spin_avg")
        # Unknown/rare codes are retained in the denominator and explicitly counted.
        listed = ",".join(qs(t) for t in PITCH_TYPES)
        agg.append(f"count(*) FILTER (WHERE pitch_type IS NULL OR pitch_type NOT IN ({listed})) AS other_pitch_count")
    else:
        missing["appearance_raw_fields"].append("pitch_type")
    missing["appearance_raw_fields"] = sorted(set(missing["appearance_raw_fields"]))
    metadata_fields = [
    n for n in (
        CONTEXT + [
            "innings_pitched",
            "outs_recorded",
            "pitch_count",
            "batters_faced",
            "walks",
            "strikeouts",
        ]
    )
    if n not in {"game_id", "pitcher_id", "game_date", "season"}
]

    joined, _ = select_optional(
        metadata,
        metadata_fields,
        "m",
        missing,
        "appearance_context"
    ) if metadata else ([], [])
    query = (f"WITH pitch_window AS ({raw_query}), physical AS "
             f"(SELECT {clean} FROM pitch_window WHERE {physical}), aggregated AS "
             f"(SELECT {', '.join(agg)} FROM physical GROUP BY game_id,pitcher_id) "
             "SELECT a.*" + (", " + ", ".join(joined) if joined else "") + " FROM aggregated a")
    if metadata:
        query += f" LEFT JOIN {metadata.sql} m ON a.game_id=m.game_id AND a.pitcher_id=m.pitcher_id"
    else:
        notes.append("Appearance metadata unavailable: official role/team/innings omitted.")
    return query


def feature_candidates():
    features = V1_FEATURES + """max_fb_velo fastball_count first_n_fb_velo last_n_fb_velo
high_velocity_pitch_count velocity_measured_count arm_angle_measured_count
velocity_coverage arm_angle_coverage plate_x_sd plate_z_sd statcast_batters_faced
batters_faced strikeout_rate ivb_inches prior_appearance_count workload_history_appearances
observed_prior_mlb_appearances documented_prior_ucl_count starter_role_prev_5
season_appearances season_starts current_vs_previous_season_workload""".split()
    metrics = "avg_fb_velo fb_velo_sd release_x_avg release_z_avg release_x_sd release_z_sd extension_avg arm_angle_avg spin_avg zone_pct".split()
    for n in (3, 5, 10):
        features.append(f"history_count_prev_{n}")
        features.extend(f"{m}_prev_{n}" for m in metrics)
    for stem in "release_x release_z release_x_sd release_z_sd extension arm_angle spin zone_pct".split():
        features.append(stem + "_slope_prev_5")
    for metric in metrics:
        if metric != "fb_velo_sd":
            features.extend(metric + suffix for suffix in ("_baseline_count", "_baseline_mean", "_baseline_sd"))
    for n in (7, 14, 28):
        features.extend([f"pitches_prev_{n}d", f"innings_prev_{n}d", f"appearances_prev_{n}d",
                         f"workload_{n}d_complete"])
    for pt in PITCH_TYPES:
        features.extend(pt.lower() + suffix for suffix in
                        ("_usage_pct", "_usage_pct_change_vs_prev_5", "_spin_avg"))
    return list(dict.fromkeys(features))


def model_query(features, appearances_sql, labels, missing, notes):
    if not features:
        notes.append("No finished feature table found: model CSV falls back to current appearance aggregates; "
                     "rolling features, trends, baseline z-scores and historical workload are unavailable.")
        extra, found = select_optional(labels, LABELS + AUDIT, "l", missing, "fallback_labels") if labels else ([], [])
        query = "SELECT a.*" + (", " + ", ".join(extra) if extra else "") + f" FROM ({appearances_sql}) a"
        if labels:
            query += f" LEFT JOIN {labels.sql} l ON a.game_id=l.game_id AND a.pitcher_id=l.pitcher_id"
        return query, [], [f for f in found if f in LABELS]
    predictors, names = select_optional(features, feature_candidates(), "f", missing, "model_predictors")
    model_context_fields = [
    n for n in (CONTEXT + AUDIT + LABELS)
    if n not in {"game_id", "pitcher_id", "game_date", "season"}
]
    model_context_fields = [
        n for n in (CONTEXT + AUDIT + LABELS)
        if n not in {"game_id", "pitcher_id", "game_date", "season"}
    ]
    context, _ = select_optional(features, model_context_fields, "f",missing, "model_context_and_labels")
    selected = ["f.game_id", "f.pitcher_id", "CAST(f.game_date AS DATE) AS game_date",
                "CAST(year(f.game_date) AS INTEGER) AS season"] + context + predictors
    query = "SELECT " + ", ".join(selected) + f" FROM {features.sql} f WHERE {window('f.game_date')}"
    if "game_type" in features.columns:
        query += " AND f.game_type='R'"
    notes.append("No rolling high-intensity pitch-count feature is selected; available high_velocity_pitch_count "
                 "is current-appearance workload at >=95 mph. No unavailable historical measurements are imputed.")
    return query, names, [n for n in LABELS if n in features.columns]


def pitch_sample_query(con, raw, raw_query, target, seed, missing):
    columns, _ = select_optional(raw, """at_bat_number pitch_number batter_id pitch_type
release_speed effective_speed release_spin_rate release_pos_x release_pos_y release_pos_z
release_extension arm_angle pfx_x pfx_z spin_axis plate_x plate_z sz_top sz_bot zone
description events pitch_result balls strikes inning inning_topbot stand p_throws
outs_when_up is_actual_pitch quality_flags""".split(), "p", missing, "pitch_sample_fields")
    counts = dict(con.execute(f"SELECT year(game_date),count(*) FROM ({raw_query}) GROUP BY 1").fetchall())
    desired = min(target, sum(counts.values()))
    quotas = {y: min(counts.get(y, 0), desired // 2 + (desired % 2 if y == 2024 else 0)) for y in SEASONS}
    for y in SEASONS:  # Transfer any unavailable season quota without over-sampling.
        quotas[y] += min(desired - sum(quotas.values()), counts.get(y, 0) - quotas[y])
    keys = [k for k in ("game_id", "pitcher_id", "at_bat_number", "pitch_number") if k in raw.columns]
    # MD5 over explicit stable keys is independent of scan order, DuckDB threads,
    # Python's randomized hash, and DuckDB's internal hash implementation.
    key_text = ", ".join([qs(str(seed))] + [f"CAST(p.{qi(k)} AS VARCHAR)" for k in keys])
    ranking = f"md5(concat_ws(':', {key_text})), " + ", ".join("p." + qi(k) for k in keys)
    query = (f"WITH pitch_window AS ({raw_query}) SELECT p.game_id, p.pitcher_id, "
             "CAST(p.game_date AS DATE) AS game_date, CAST(year(p.game_date) AS INTEGER) AS season" +
             (", " + ", ".join(columns) if columns else "") + " FROM pitch_window p "
             f"QUALIFY row_number() OVER (PARTITION BY year(p.game_date) ORDER BY {ranking}) <= "
             f"CASE year(p.game_date) WHEN 2024 THEN {quotas[2024]} WHEN 2025 THEN {quotas[2025]} ELSE 0 END")
    return query, quotas, counts, keys


def csv_relation(path, columns):
    # Explicit types preserve MLBAM IDs and all-null optional fields. CSV dates
    # are ISO YYYY-MM-DD; empty cells represent NULL, never imputed zero.
    types = ", ".join(qs(n) + ": " + qs(t) for n, t in columns.items())
    return f"read_csv({qs(path)}, header=true, nullstr='', columns={{{types}}})"


def summary(con, query, columns):
    result = {}
    row = con.execute(f"SELECT count(*),min(game_date),max(game_date),count(DISTINCT pitcher_id) FROM ({query})").fetchone()
    result.update(rows=row[0], earliest_game_date=str(row[1]), latest_game_date=str(row[2]), unique_pitchers=row[3])
    result["by_season"] = {str(y): n for y, n in con.execute(
        f"SELECT season,count(*) FROM ({query}) GROUP BY 1 ORDER BY 1").fetchall()}
    result["injury_labels"] = {}
    for label in LABELS:
        if label not in columns:
            continue
        event = ",count(DISTINCT CASE WHEN " + qi(label) + " THEN next_injury_event_id END)" if "next_injury_event_id" in columns else ""
        row = con.execute(f"SELECT count({qi(label)}),count(*) FILTER (WHERE {qi(label)}),"
                          f"count(DISTINCT CASE WHEN {qi(label)} THEN pitcher_id END){event} FROM ({query})").fetchone()
        result["injury_labels"][label] = dict(nonnull_rows=row[0], positive_rows=row[1],
                                             negative_rows=row[0]-row[1], unknown_rows=result["rows"]-row[0],
                                             positive_pitchers=row[2])
        if event:
            result["injury_labels"][label]["positive_events"] = row[3]
    for flag in ("is_starter", "valid_starting_appearance", "training_eligible", "timing_uncertain"):
        if flag in columns:
            result[flag + "_rows"] = con.execute(f"SELECT count(*) FROM ({query}) WHERE {qi(flag)}").fetchone()[0]
    return result


def export_csv(con, query, path, keys):
    print(f"Generating {path.name} ...", flush=True)
    columns = {r[0]: r[1] for r in con.execute(f"DESCRIBE ({query})").fetchall()}
    order = ["game_date", "game_id", "pitcher_id"] + [k for k in keys if k not in ("game_id", "pitcher_id")]
    # Order the projected result, not joined source columns: appearance metadata
    # and pitch aggregates both contain game_date/game_id/pitcher_id.
    ordered = "SELECT * FROM (" + query + ") exported ORDER BY " + ", ".join(qi(k) for k in order)
    staging = path.with_suffix(".csv.part")
    try:
        copied = con.execute(f"COPY ({ordered}) TO {qs(staging)} (FORMAT CSV, HEADER true, NULL '', DATEFORMAT '%Y-%m-%d')").fetchone()[0]
        relation = csv_relation(staging, columns)
        read_query = f"SELECT * FROM {relation}"
        stats = summary(con, read_query, columns)
        if stats["rows"] != copied:
            raise ValueError(f"CSV row count does not match COPY for {path.name}")
        invalid = con.execute(f"SELECT count(*) FROM {relation} WHERE game_date IS NULL OR NOT ({window('game_date')}) "
                              "OR season IS DISTINCT FROM year(game_date) OR game_id IS NULL OR pitcher_id IS NULL").fetchone()[0]
        if invalid:
            raise ValueError(f"{path.name}: {invalid} rows have invalid dates/IDs/seasons")
        assert_unique(con, read_query, keys, path.name)
        stats["bytes"] = staging.stat().st_size
        stats["columns"] = columns
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
    print(f"  {stats['rows']:,} rows; {stats['bytes']/1_000_000:.1f} MB; seasons: {stats['by_season']}", flush=True)
    return stats


def source_fingerprint(path):
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pitch-sample-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=202425)
    parser.add_argument("--memory-limit", default="2GB", help="DuckDB execution memory limit; default 2GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    database, out = args.database.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not database.is_file():
        parser.error(f"Database does not exist or is not a file: {database}. Supply --database /path/to/existing.duckdb")
    if args.pitch_sample_rows < 2 or args.threads < 1:
        parser.error("--pitch-sample-rows must be >=2 and --threads must be >=1")
    # Never permit an output to alias the source, including through a symlink.
    output_names = ["pitcher_appearances_2024_2025.csv", "model_features_2024_2025.csv",
                    "pitch_level_sample_2024_2025.csv", "export_manifest.json", "README.md"]
    if any((out / n).resolve() == database for n in output_names):
        parser.error("Output path aliases the source database; choose another --output-dir")
    before = source_fingerprint(database)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.executable}; DuckDB: {duckdb.__version__}\n"
          f"Opening READ ONLY: {database}\nExport dates: {START_DATE} through 2025-12-31", flush=True)
    notes, missing = [], {}
    with tempfile.TemporaryDirectory(prefix=".duckdb_export_", dir=out) as spill:
        con = duckdb.connect(str(database), read_only=True)
        try:
            con.execute(f"SET memory_limit={qs(args.memory_limit)}")
            con.execute(f"SET threads={args.threads}")
            con.execute(f"SET temp_directory={qs(spill)}")
            schemas, relations = inspect_schema(con)
            required = {"game_id", "pitcher_id", "game_date"}
            raw = choose(relations, ["pitches", "statcast_pitches", "raw_statcast"], required,
                         {"pitch_number", "at_bat_number", "release_speed"}, mandatory=True)
            games = choose(relations, ["games"], {"game_id", "game_date"}, {"home_team_id", "away_team_id"})
            appearances = choose(relations, ["appearances"], required, {"total_pitches", "starter_source"})
            # Avoid starter-only or training-only views: all positives must survive.
            features = choose(relations, ["model_features", "model_ready_pitcher_appearances"], required,
                              {"history_max_date", "baseline_max_date", "training_eligible"})
            labels = choose(relations, ["injury_labels"], {"game_id", "pitcher_id"}, {"ucl_injury_next_30d"})
            sources = {"raw_pitches": raw, "games": games, "appearances": appearances,
                       "model_features": features, "injury_labels": labels}
            print("Sources: " + "; ".join(f"{role}={r.sql if r else 'unavailable'}" for role, r in sources.items()), flush=True)
            raw_sql = raw_window_query(raw, games, notes)
            raw_keys = [k for k in ("game_id", "at_bat_number", "pitch_number") if k in raw.columns]
            if len(raw_keys) < 3:
                raise ValueError("Raw pitch source lacks stable (game_id, at_bat_number, pitch_number) keys needed for reproducible sampling")
            assert_unique(con, raw_sql, raw_keys, "raw pitch window")
            conflict = con.execute(f"SELECT count(*) FROM (SELECT game_id,pitcher_id FROM ({raw_sql}) "
                                   "GROUP BY 1,2 HAVING count(DISTINCT game_date)>1)").fetchone()[0]
            if conflict:
                raise ValueError("Raw pitcher/game groups contain conflicting dates; cannot assign a safe appearance date")
            for r in (appearances, features, labels):
                if r:
                    assert_unique(con, f"SELECT game_id,pitcher_id FROM {r.sql}", ["game_id", "pitcher_id"], r.name)
            app_sql = appearance_query(raw, raw_sql, appearances or features, missing, notes)
            model_sql, predictors, present_labels = model_query(features, app_sql, labels, missing, notes)
            # Prior history is inherited from the complete source, then filtered.
            history_checks = {}
            for h in ("history_max_date", "baseline_max_date", "workload_max_date"):
                if features and h in features.columns:
                    invalid = con.execute(f"SELECT count(*) FROM ({model_sql}) WHERE {qi(h)} >= game_date").fetchone()[0]
                    if invalid:
                        raise ValueError(f"Temporal leakage: {h} is not strictly prior on {invalid} rows")
                    history_checks[h] = {"nonprior_rows": invalid, "rows_using_pre_2024_history":
                        con.execute(f"SELECT count(*) FROM ({model_sql}) WHERE {qi(h)} < DATE '{START_DATE}'").fetchone()[0]}
            pitch_sql, quotas, raw_counts, sample_keys = pitch_sample_query(con, raw, raw_sql, args.pitch_sample_rows, args.seed, missing)
            queries = [app_sql, model_sql, pitch_sql]
            exports = {}
            for filename, query, keys in zip(output_names[:3], queries,
                                              [["game_id", "pitcher_id"], ["game_id", "pitcher_id"], sample_keys]):
                exports[filename] = export_csv(con, query, out / filename, keys)
            model_stats = exports[output_names[1]]
            source_model_stats = summary(con, model_sql, model_stats["columns"])
            for key in ("rows", "by_season", "injury_labels"):
                if source_model_stats[key] != model_stats[key]:
                    raise ValueError(f"Model CSV did not retain source {key}")
            sample_stats = exports[output_names[2]]
            if sample_stats["by_season"] != {str(y): n for y, n in quotas.items() if n}:
                raise ValueError("Pitch sample does not match the requested seasonal quotas")
            app_read = csv_relation(out / output_names[0], exports[output_names[0]]["columns"])
            model_read = csv_relation(out / output_names[1], model_stats["columns"])
            absent = con.execute(f"SELECT count(*) FROM {model_read} m ANTI JOIN {app_read} a USING(game_id,pitcher_id)").fetchone()[0]
            if absent:
                raise ValueError(f"{absent} model appearances are missing from the appearance export")
            positives = " OR ".join(qi(n) for n in present_labels)
            any_positive = con.execute(f"SELECT count(*) FROM {model_read} WHERE {positives}").fetchone()[0] if positives else None
        finally:
            con.close()
    after = source_fingerprint(database)
    if after != before:
        raise RuntimeError("Source file metadata changed during the export (possibly an external writer); verify the source and rerun")
    notes.extend([
        "All available regular-season appearances are retained, including relief and quality-ineligible rows. For the starter population, filter valid_starting_appearance=true; separately apply training_eligible and require the chosen label to be non-NULL.",
        "Predictions are defined AFTER the appearance. Current pitch metrics and season workload include the current appearance; prior rolling/baseline dates must be strictly earlier. Earlier full-history source observations remain available to engineered features.",
        "Labels and future event/follow-up dates are outcomes/audit metadata, not predictors. Use only model_predictor_columns for X. Injury date uncertainty and censoring are inherited unchanged from the source.",
        "Pitch sample contains raw event rows, including automatic balls/strikes; use is_actual_pitch=true to reproduce appearance aggregation. Source outliers are preserved in this raw sample. The sample is balanced by season, not by injury status, and individual appearances are generally incomplete.",
        "Appearance aggregates exclude nonphysical pitches and apply the existing measurement bounds: speed 30–110 mph, release X -7–7 ft, release Z 0–9 ft, extension 2–9 ft, spin 0–4500 rpm, arm angle -90–180 degrees. SD is sample SD; unused pitch types have count/usage zero and mean spin NULL. Usage/rates are fractions 0–1.",
        "Arm angle and pitch-type-specific metrics may be missing; historical baselines can be NULL despite 2024–2025 current measurement availability. No imputation is applied. Observed MLB experience is left-truncated at the source study start.",
        "Official starters include openers. Timing-uncertain suspended/resumed games remain visible with their source eligibility flags. Do not randomly split these appearance rows; this two-season exploration extract is not an independent chronological evaluation split.",
        "The three CSVs are derived exports. DuckDB was opened read_only=True; source size, inode and modification time were unchanged. No source tables or files were updated."
    ])
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "database": str(database),
                "database_fingerprint_before": before, "database_fingerprint_after": after,
                "duckdb_version": duckdb.__version__, "start_date": START_DATE, "end_date_inclusive": "2025-12-31",
                "filter_fields": {"pitcher_appearances": f"{raw.sql}.game_date", "model_features": f"{features.sql}.game_date" if features else "derived appearance game_date", "pitch_sample": f"{raw.sql}.game_date"},
                "schemas": schemas, "schema_inventory": [{"schema": r.schema, "name": r.name, "type": r.kind, "columns": r.columns} for r in relations],
                "selected_sources": {k: r.sql if r else None for k, r in sources.items()},
                "pitch_sample_seed": args.seed, "pitch_sample_quotas": quotas, "raw_window_rows_by_season": raw_counts,
                "exports": exports, "model_predictor_columns": predictors,
                "v1_feature_columns": [n for n in V1_FEATURES if n in predictors],
                "model_nonpredictor_columns": [n for n in model_stats["columns"] if n not in predictors],
                "any_injury_positive_model_rows": any_positive, "history_validation": history_checks,
                "missing_optional_columns": missing, "notes": notes,
                "queries": dict(zip(output_names[:3], queries))}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = ["# Exploratory MLB exports, 2024–2025", "", "Generated by `generate_exploratory_csvs.py` from a read-only DuckDB connection.",
              "All row dates are 2024-01-01 through 2025-12-31. SQL, inspected schemas, exact column types, omitted optional fields, label counts, and the predictor allowlist are in `export_manifest.json`.", "",
              "| CSV | Rows | MB (decimal) | 2024 | 2025 |", "|---|---:|---:|---:|---:|"]
    for name, s in exports.items():
        readme.append(f"| {name} | {s['rows']:,} | {s['bytes']/1_000_000:.1f} | {s['by_season'].get('2024',0):,} | {s['by_season'].get('2025',0):,} |")
    readme.extend(["", f"Model CSV: {len(predictors)} selected predictor columns, including {len(manifest['v1_feature_columns'])} v1 features; {any_positive} rows positive under at least one exported horizon. No appearance sampling.", "",
                   "| Label | Positive rows | Known negative rows | Unknown rows | Unique positive events |", "|---|---:|---:|---:|---:|"])
    for name, s in model_stats["injury_labels"].items():
        readme.append(f"| {name} | {s['positive_rows']} | {s['negative_rows']} | {s['unknown_rows']} | {s.get('positive_events','unavailable')} |")
        print(f"Retained {name}: {s['positive_rows']:,} positive rows ({s.get('positive_events', 'unknown')} events)")
    readme.extend(["", "## Interpretation", ""] + ["- " + n for n in notes])
    readme.extend(["", "## Missing optional fields", "", "```json", json.dumps(missing, indent=2), "```", ""])
    (out / "README.md").write_text("\n".join(readme))
    print(f"Source unchanged. All {any_positive} rows positive under any exported horizon retained.\nOutputs: {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (duckdb.Error, OSError, ValueError, RuntimeError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        sys.exit(1)
