# General MLB pitcher injury episodes, 2015–2026

This extension builds documented MLB injured-list episodes across all body regions and illnesses. It does not replace the existing UCL study, redefine its labels, fit models, or change the central DuckDB. Its canonical grain is **one MLB pitcher × one inferred roster injury episode**, not a pitch, appearance, individual transaction, or surgery.

The executed results are in [`reports/general_injury/build_summary.md`](../reports/general_injury/build_summary.md). All 2026 results are through the configured observation date and are incomplete for the full year. A successful request means the API returned that window; it does not prove the API recorded every historical injury or activation.

## Architecture and reuse

The existing `src/http.py` supplies HTTP retry/backoff, bounded concurrency, checksum validation, immutable compressed response bodies and atomic cache indexing. MLB endpoints and MLBAM IDs reuse `src/ingest_mlb.py` conventions; event IDs reuse `src/utils.stable_id`. Existing `pitchers`, `appearances`, `injuries`, `injury_events`, and `injury_event_procedures` are read from DuckDB for identity, observed returns and UCL validation. No call to `Database.initialize()` or the original feature/label pipeline is made.

New modules:

| Module | Responsibility |
|---|---|
| `src/injury_pipeline.py` | CLI, read-only source access, orchestration, exports and build manifest |
| `src/injury/acquire.py` | Calendar-month requests, lossless normalization, cache/resume and coverage |
| `src/injury/identity.py` | IL action normalization, official people/team lookups and pitcher inclusion |
| `src/injury/taxonomy.py` | Ordered anatomical structure, region and diagnosis patterns |
| `src/injury/parser.py` | Injury text, laterality, dates and lifecycle action parsing |
| `src/injury/episodes.py` | Placement/transfer/activation state machine and boundary validation |
| `src/injury/supplement.py` | RosterResource evidence and comparisons with the existing UCL pipeline |
| `src/injury/validation.py` | Assertions, population/duration summaries and review samples |

Configuration is separate in [`config/injury.yaml`](../config/injury.yaml). Dependencies are already installed by the project; no new package is needed.

## Run and update

From `mlb_ucl_project`:

```bash
# Full reproducible derivation; completed HTTP requests are reused.
.venv/bin/python -m src.injury_pipeline

# Refresh the last 62 days of transaction windows and current-year RosterResource.
.venv/bin/python -m src.injury_pipeline --update

# Rebuild entirely from existing immutable cached responses.
.venv/bin/python -m src.injury_pipeline --offline

# Explicit observation window; override output/report paths in a separate config
# when retaining several derived study windows alongside one another.
.venv/bin/python -m src.injury_pipeline --start-date 2015-01-01 --end-date 2026-09-18
.venv/bin/python -m src.injury_pipeline --config config/injury.yaml

# Acquisition only, useful for warming the cache before a later full build.
.venv/bin/python -m src.injury.acquire

.venv/bin/python -m pytest -q
```

`latest` resolves to the current date, capped at December 31, 2026. The current day's response may change; use `--update` to refresh recent periods. Output basenames retain the study label `2015_2026`; the manifest and actual row dates identify the precise requested window. Neither a `--full-refresh` of Statcast nor a model rebuild is necessary.

## Storage and provenance

```text
data/injury/
  raw/
    http/                         # Immutable SHA-256 gzip bodies + URL cache indexes
    snapshots/                    # Immutable lossless transaction occurrence snapshots
    supplemental/                 # Immutable normalized source snapshots
    mlb_transactions_2015_2026_raw.parquet
    transaction_request_status.json
    request_history.jsonl
  interim/
    player_identities.parquet
    mlb_team_seasons.parquet
    il_transactions_all_positions.parquet
    manual_review_annotations.csv
  processed/
    mlb_pitcher_injury_events_2015_2026.csv
    mlb_pitcher_injury_events_2015_2026.parquet
    mlb_pitcher_injury_transactions_2015_2026.csv
    mlb_pitcher_injury_transactions_2015_2026.parquet
    mlb_pitcher_injury_unresolved_2015_2026.csv
    mlb_pitcher_injury_unresolved_2015_2026.parquet
    injury_event_transaction_links.csv
    injury_event_transaction_links.parquet
    build_manifest.json
reports/general_injury/
  build_summary.md
  acquisition_coverage.csv
  validation_checks.csv
  mlb_pitcher_injury_validation_summary.csv
  mlb_pitcher_injury_parser_unknowns.csv
  events_by_season.csv
  events_by_region.csv
  events_by_diagnosis.csv
  ucl_pipeline_comparison.csv
  ucl_comparison_reason_counts.csv
  supplemental_source_matches.csv
  supplemental_unmatched_records.csv
  supplemental_coverage.csv
  source_conflicts.csv
  timing_uncertain_events.csv
  extreme_durations.csv
  overlapping_and_duplicate_placements.csv
  unpaired_il_transactions.csv
  excluded_il_transactions.csv
  manual_review_samples.csv
```

Raw JSON responses are preserved before any filter. The flattened raw Parquet includes complete `raw_json`, source URL, request bounds, source SHA and retrieval timestamp, including fields the adapter does not recognize. Its current materialization can be regenerated, but prior HTTP response bodies and occurrence snapshots are never overwritten. Transaction-ID duplicates are deduplicated; conflicting versions remain in the conflict report and snapshots. The latest retrieved version is selected deterministically.

Requests use `https://statsapi.mlb.com/api/v1/transactions?sportId=1&startDate=...&endDate=...`. They are monthly, retry transient HTTP failures through the existing client, and persist completion/failure status after each window. Failed refreshes retain previously materialized rows for that window and remain explicitly failed. A partial build is flagged in console/reports, marks its events' source coverage incomplete, and exits with status 2. It must be resumed before treating its lifecycle associations as fully supported.

## Sources and their limits

1. **MLB Stats API transactions, people, and season-specific teams** are the official backbone. IDs are retained as nullable integers. Names are display fields, never linkage keys. API date fields and source text can contain errors or later edits; official does not mean infallible.
2. **[FanGraphs / RosterResource](https://www.fangraphs.com/roster-resource/injury-report)** public season pages contain structured `__NEXT_DATA__` records with MLBAM IDs. The adapter requests `season=YYYY&timeframe=all`, checks each returned record's actual season, and caches the full response. This execution retrieved records for 2020–2026; 2015–2019 returned no records for the requested season. HTML/schema changes become explicit source failures, not guessed data.
3. **Existing reconciled Roegele UCL/Tommy John and internal-brace records** are used as an independent surgery validation snapshot. Their original URLs, date precision and identity methods are retained. This pipeline does not refresh or mutate the original surgery/UCL tables. Newer procedures absent from that snapshot may therefore be missing from this comparison.
4. The **[Baseball Prospectus recovery dashboard introduction](https://www.baseballprospectus.com/news/article/69824/visualizing-the-infirmary-introducing-the-recovery-dashboard/)** was retrievable, but an MLBAM-keyed bulk dataset adapter was not established. It is not claimed as an ingested source.
5. **[Pro Sports Transactions](https://www.prosportstransactions.com/baseball/Search/Search.php)** returned HTTP 403 in this environment. No access bypass was attempted. The observed result is recorded in `other_source_evaluation.json`.

RosterResource is matched by MLBAM ID plus a unique IL retro date within three days. A fallback requires compatible anatomy and a unique combined injury/surgery date within seven days. Ambiguous matches stay unmatched. These tolerances are configurable. Its combined `Injury / Surgery Date` field is never silently interpreted as biological onset. Month-level `dateDisplay` records do not create exact surgery dates.

Different anatomy, specific diagnoses, return dates more than three days apart, and differing surgery dates are retained as conflicts. Some reflect evolving diagnoses, different return definitions, or multiple procedures rather than factual mistakes. Official fields are not replaced. Supplemental-only and ambiguous records remain in the unmatched inventory; they are not automatically converted into presumed MLB IL episodes. Thus the canonical population has a clear IL-based ascertainment rule, while the independent inventory supports later expansion to documented non-IL injuries.

To add another adapter, return ID-keyed evidence with source URL, original text, raw payload, date precision, retrieval timestamp and separate injury/return/surgery dates; add a conservative match rule and conflict report in `supplement.py`. Do not join on names or overwrite official fields.

## Pitcher population

An IL record must belong to a verified MLB team for the transaction's season. Destination team controls membership when present; a minor-league destination is not made MLB merely because the source team is an MLB club.

The primary population includes people whose official primary position is `P` or `TWP`. It also retains a historical pitching-role case when the same MLBAM ID has an observed MLB pitching appearance and the transaction identifies a pitching role; this is flagged for identity review when current primary position differs. This prevents both dependence on the words RHP/LHP alone and automatic inclusion of every position player who once pitched. Two-way players are included and explicitly flagged when official metadata identifies them as TWP. Current people metadata is a snapshot, not a complete historical position history.

Pitchers on an MLB IL without observed Statcast history remain eligible as documented MLB roster cases, with `has_statcast_history=false`. This prevents exclusion of pitchers who were already hurt before debut or missed the entire observed playing window. Starters and relievers are both included.

## Parsing and taxonomy

The parser first removes the player/team/throwing-hand portion of a transaction, then parses the injury clause. It recognizes injured list, disabled list, IL and DL, distinguishes placement, transfer, activation/reinstatement and other IL status, and retains generic placements without a diagnosis. Rehabilitation assignment alone is not activation.

Specific structures precede generic anatomy. Examples include rotator cuff → Shoulder, olecranon → Elbow / UCL, pronator → Forearm, meniscus → Knee and Achilles → Foot / Ankle. Context overrides address thumb UCL, hip labrum, biceps explicitly described at the shoulder, and flexor complaints explicitly located at the elbow. Generic flexor strain is not falsely labeled a documented tendon injury. Pectoral and quadriceps injuries use Other with the named structure rather than being forced into a wrong anatomical region.

The configured regions are Elbow / UCL, Forearm, Shoulder, Upper Arm, Lat, Core / Oblique / Abdomen, Back / Spine, Hip / Groin, Hamstring, Knee, Lower Leg, Foot / Ankle, Hand / Wrist / Finger, Neck, Head / Concussion, Illness / non-musculoskeletal, Other, and Unknown / Unspecified. Specific diagnoses include strain, sprain, tear/rupture, inflammatory and tendinopathy terms, impingement, stress fracture/reaction, fracture, contusion, dislocation/subluxation, tightness, soreness, spasm, concussion and postoperative recovery. Unknown remains valid when a mechanism is not supplied.

Word boundaries prevent `lat` or `ACL` substring matches. Laterality comes from the injury clause, not RHP/LHP. Bilateral requires explicit bilateral/both wording; conflicting sides otherwise remain Unknown. Multi-region and multi-player descriptions, negation and uncertain diagnoses are flagged. All detected regions/diagnoses and matched text remain available. `explicit_ucl` requires explicit unambiguous UCL/Tommy John language, rather than any elbow complaint.

## Dates, episodes and activation matching

`injury_event_date` uses explicitly documented injury date, description's retroactive date, MLB effective date, then transaction date. It is an **observable roster/injury-date proxy**, not automatically biological onset. Dates in the future of their reported transaction or implausibly old retroactive/effective dates (over 31 days by default) are retained as original fields, flagged, and excluded from date selection. The next defensible source date is used; the pipeline never repairs a year by guessing. Long genuine symptom histories require explicit onset evidence, not a retroactive IL assumption.

Transactions are ordered by player, reported transaction date and transaction ID. An initial placement opens an episode; transfer records extend it. A later activation must be strictly after placement, before another unrelated placement, within a configurable 730-day association bound, and not contradicted by a reliable observed MLB pitching return. Long accepted durations over 180 days remain present with a review flag. Offseason activations are flagged because they can represent administrative roster restoration.

Same-day compatible duplicate placements are consolidated with their IDs and descriptions retained. A later record phrased as a new placement on a longer IL can be an inferred continuation only with same-season compatible anatomy or the same explicit effective date, a compatible side, a bounded interval of 180 days, a larger IL designation, and no observed intervening MLB return. These inferred continuations are flagged for review. A genuinely different or unresolved new placement opens a separate episode; the old one is censored rather than matched to an arbitrary later activation.

An observed MLB pitching appearance can establish a boundary where the API lacks an activation. It does **not** become an imputed activation date or duration. Such events stay duration-censored. The existing appearance history only extends through its own database coverage; the manifest identifies that source snapshot. A transfer with no observed initial placement becomes a left-censored episode with NULL initial injury date, rather than an invented initial placement. Same-day activation/transfer ambiguity remains flagged.

`resolutionDate` is retained for audit only and is never used as return. For validated matches with a known initial event date:

```text
days_out_observed = activation_date - injury_event_date
```

This includes calendar days and offseasons; it is observed roster absence, not medical recovery. Unresolved durations are NULL. `season_ending_proxy` describes remaining on the observed IL through November 1 for an in-season event, when follow-up permits; it is not a clinical severity diagnosis. No arbitrary minor/moderate/severe bins are assigned.

## Output schema and time availability

The manifest lists every actual output column. Major groups are:

| Group | Fields |
|---|---|
| Identity | `injury_event_id`, `player_id`, `player_name`, `primary_position`, `throws`, `two_way_player`, `has_statcast_history`, `pitcher_identity_basis` |
| Timing | `injury_event_date`, `injury_event_date_basis`, original transaction/effective/retroactive/resolution dates, `explicitly_documented_onset_date`, `public_known_date` |
| Anatomy/diagnosis | `injury_region`, `injury_structure`, `injury_diagnosis`, `laterality`, initial values, all detected regions/diagnoses, `explicit_ucl`, `initial_explicit_ucl` |
| Lifecycle | Placement/anchor/activation IDs and descriptions, transfer IDs/counts, all linked transaction IDs, `episode_boundary_date`, activation candidate/rejection reason |
| Severity/outcomes | `initial_il_length`, `max_il_length`, `transferred_to_60_day`, `days_out_observed`, `duration_observed`, `extreme_duration`, `season_ending_proxy` |
| Surgery | `surgery_flag`, `surgery_type`, `surgery_date`, `surgery_source`, `official_surgery_*`, supplemental dates/precision, `surgery_date_conflict` |
| Provenance | Primary/secondary sources, `source_count`, `source_agreement`, `source_conflict`, evidence URLs, supplemental record IDs/diagnoses, retrieval timestamps |
| QA | `activation_match_valid`, `right_censored`, `left_censored`, `orphan_transfer`, `overlapping_episode`, `timing_uncertain`, `qa_review_required`, `qa_flags`, source-coverage flag |

**Temporal distinction:** `public_known_date` and MLB `*_known_at` values are **reported transaction-date proxies**, not verified historical publication timestamps. MLB can revise the text attached to an older record. Accordingly `known_at_basis='reported_transaction_date_proxy'` and `historical_publication_verified=false` are explicit. Actual `evidence_observed_at`, `event_date_observed_at`, `diagnosis_observed_at`, `activation_observed_at` and `surgery_observed_at` preserve the snapshot retrieval time. RosterResource retains both its load timestamp and retrieval time. Curated surgery evidence uses its source snapshot time. For strict prospective availability, use observed snapshot timestamps or independently dated publication evidence; do not assume later diagnosis wording was known on the original IL date.

`initial_*` fields remain separate from details first supplied in later transactions. A later UCL diagnosis has its own reported date. Supplemental diagnosis dates, activation dates and surgery information are outcomes/audit fields and must not enter pre-injury predictors. The event table contains no safe automatic predictor allowlist because it is primarily outcome evidence.

## UCL integration and comparison

The legacy UCL pipeline links reconciled surgery records to arm-IL episodes and creates future appearance labels. Its tables and outputs remain unchanged. The new comparison first tries the exact same MLBAM ID and source transaction; otherwise it accepts only a unique compatible active arm episode bounded by activation, observed return, a subsequent placement, and the matching-gap limit.

`ucl_pipeline_comparison.csv` separates both-pipeline matches, existing-UCL-only cases, and new explicitly documented UCL cases without a legacy match. It includes surgery-level source, anatomy/diagnosis differences, public-reference versus injury-proxy date differences, and matching method. Existing-only cases are separated into non-MLB procedure level, legacy surgery-only/unknown onset, pre-study onset and unresolved association. A pitcher with MLB history may have had a procedure in the minors; that alone does not establish a missed MLB IL event. Conversely, postseason/postoperative IL re-placements do not prove a new UCL injury or new operation. UCL injury, UCL repair with an internal brace, and reconstruction remain distinct concepts.

## Validation, review and limitations

The build checks stable IDs, primary population membership, duplicate events and transactions, complete placement/transfer accounting, linked-player consistency, positive durations, exact duration arithmetic, source counts, date bounds, censored durations, activation ordering/reuse, and non-invented orphan dates. Reports provide injury region and diagnosis counts, season coverage, laterality coverage, matching/censoring/overlap rates, duration percentiles, extreme cases, independent-source matching, and UCL agreement. Source-text review samples cover every represented region plus unknowns, conflicts, overlapping placements, long durations and UCL cases.

The first audit found and fixed stale retroactive-year inflation, unsupported specificity for generic flexor complaints, false multi-region flags for local fatigue/blisters, and new-event inflation from longer-IL continuation records. Review annotations live separately in `interim/manual_review_annotations.csv`; reruns do not overwrite them. The notes are source-text/lifecycle audits, not clinical adjudication, and do not automatically clear QA flags.

Remaining limitations:

- Many injuries never generate IL transactions. The canonical primary population is IL-ascertained; unmatched supplemental records remain available for later adjudication.
- Vague, missing and misspelled descriptions cannot always establish anatomy or diagnosis. Blank values are not imputed.
- The same roster absence may include evolving diagnoses or multiple conditions; a single canonical region cannot capture every clinical detail. Evidence arrays and conflict flags retain that complexity.
- Missing activations, administrative offseason moves and same-day transactions limit observed duration. A return to the roster does not establish full recovery.
- Public source text can be revised, misdated or incomplete. Some retained extreme cases require external publication-date review before onset modeling.
- People metadata is not a historical position census. Two-way and converted players need explicit handling.
- Supplemental coverage is uneven across seasons. The UCL validation source is an existing snapshot, and RosterResource provides no usable 2015–2019 records in this run.
- 2026 is incomplete, and recent records may change. No returned rows beyond the observation date are represented as complete follow-up.

Before creating multi-injury labels, choose an explicit target (reported IL event, earliest defensible retrospective injury proxy, or adjudicated clinical onset), resolve the high-impact timing/conflict cases, and apply temporal splits. The presence of a censored return does not invalidate a well-documented initial injury event, but it cannot be used to infer a negative recovery outcome.

## Join to Statcast without altering the database

Use MLBAM IDs and a deliberate time relationship. This example lists all documented episodes for pitchers in the project; it deliberately does not attach future event fields as predictors:

```sql
SELECT i.*
FROM read_parquet('data/injury/processed/mlb_pitcher_injury_events_2015_2026.parquet') i
WHERE EXISTS (SELECT 1 FROM appearances a WHERE a.pitcher_id = i.player_id);
```

Any appearance-level label builder should be a separate transformation with tests for timing, censoring, repeated episodes and source availability. No final predictive model has been trained.
