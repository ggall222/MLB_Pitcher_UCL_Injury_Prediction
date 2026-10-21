# UCL endpoint refinement and dynamic early warning

Completed from the interrupted schema/evidence review, reusing all completed feature-consolidation artifacts. No prior numerical stage was rerun or replaced. This is development and exploratory transfer analysis, not prospective confirmation.

## Endpoint and cohort

The primary endpoint contains **194 eligible documented UCL episodes** (98 training, 34 validation, 43 exploratory transfer, 19 descriptive 2026). Official non-procedure UCL diagnosis supports 60; the other 134 require a bounded, identifier-linked UCL procedure with compatible anatomy, no intervening MLB return, and no incompatible episode boundary. There are 221 eligible events when probable evidence is included. Twenty-two compatible forearm-to-UCL progressions were retained with explicit source adjudication. Generic elbow pain, unconfirmed forearm injury, postoperative entries and unresolved episode ambiguity are not automatically UCL outcomes.

The date is the earliest canonical reported/retroactive/effective injury proxy, never a substituted surgery date. Biological onset is generally unknown. Source retrieval occurred retrospectively; contemporaneous archived publication was not independently verified. The strict official sensitivity has only seven validation events, of which three have labeled appearances, and is underpowered.

There are 233,433 source appearances. The 30-day labeled risk sets contain 87,332 training, 28,875 validation and 29,349 exploratory-transfer appearances. Positive events represented are 59, 25 and 28; not every eligible UCL episode has a preceding MLB appearance. At 7–30 days, only 20/34 validation and 28/43 transfer events have an eligible warning opportunity. All eligible endpoint events remain in the principal recall denominator.

## Modeling and findings

At each completed appearance, fixed L2 logistic landmark models estimate first-event UCL risk over 7, 14, 21 or 30 days. These are separate horizon models, not a coherent survival curve. They preserve profile, departure-8, prior-62 and stable-62 representations. Twelve new trajectory fields and three standardized profile×departure interactions test dynamic additions. Missing values are imputed only inside training-fitted preprocessing; source data remain nullable.

The predeclared validation warning rule selected **Departure** before any new 2024–2025 predictions. It detected **7/34 validation events (20.6%)** at 7–30 days and a 5% appearance budget, versus **2/43 (4.7%)** in exploratory transfer. Conditional median first-warning lead was 12 and 14 days. These medians describe detected events only. The result does not establish dependable warning.

Profile alone had validation AP 0.00612 versus 0.00727 for departure, 0.00737 combined, 0.00769 with trajectories and 0.00789 with interactions. Paired uncertainty includes zero for the profile/departure and trajectory AP increments. The small interaction AP increment is positive under conditional validation resampling, but adds no validation warning events; training-refit uncertainty and multiple exploratory comparisons limit interpretation.

Stable-62 achieved validation AP 0.01092 (2.28× prevalence) and transfer AP 0.01846 (3.82×). Its 7–30d event warnings were 3/34 and 9/43. Previous-62 warned on 3/34 and 9/43. Better ranking and better practical warning are distinct, and transfer results must not retroactively change the selected candidate.

Adding trajectories to profile+departure left validation warnings at 3/34; exploratory-transfer warnings rose from 5/43 to 6/43. Interactions added no warnings in either period. The evidence does not justify claiming a robust incremental early-warning benefit.

Matched risk plots use 33 balanced five-outing pairs and 16 balanced calendar-window pairs, unchanged existing controls where available, and same-stratum matches only for newly eligible episodes. Whole dependency-cluster intervals account for shared case/control pitchers. Average departure risk does not show a consistent rise as injury approaches; individual paths are heterogeneous. A flat group offset is a profile association, not evidence of imminence. No latent or risk score is interpreted as tissue damage. All three interaction coefficient signs vary across the eight training refits, weakening any interpretation of the small AP increment.

## Warning burden, uncertainty and robustness

More than 98% of labeled alerts are negative for the first-event UCL outcome within 30 days in the principal departure comparison. This includes known competing first injuries, not a claim of complete medical surveillance. Unknown/censored outcomes are excluded from labeled performance, not called healthy. `operational_all_landmark_burden.csv` separately applies the frozen numeric threshold to every prospectively eligible appearance and retains unknown alert outcomes: 1,980 validation alerts include 536 unknown outcomes; 1,875 transfer alerts include 518. Numeric-threshold alert rates can drift away from 5%; retrospective top-N budgets are not deployment thresholds.

Whole-pitcher paired resampling, eight whole-pitcher training refits, endpoint expansion, strict official diagnoses, public-date timing, competing-event censoring, exclusion of the last six days, complete trajectory support, role and unseen-pitcher comparisons are reported. Subgroup analyses use explicitly identified observable-event denominators. The strict/unseen subsets are too small to establish generalization. No formal universal change point is supported by the irregular and limited event histories.

Known non-UCL first injuries are competing negatives for the primary cumulative first-event question. Suspected UCL/ambiguous arm episodes are censored. Censoring all known competitors is a sensitivity analysis. No alert may skip an intervening episode to receive credit for a later UCL event.

Endpoint specificity and the risk set changed from the previous broad Elbow/UCL analysis, so its AP and warning counts are not directly comparable to these results. The same-endpoint 30-day frozen-feature benchmarks provide the comparison here. Multi-horizon modeling has not established an independent improvement over ordinary 30-day appearance-level classification.

## Frozen next step

Keep the validation-selected departure-8 model, profile and prior-62 comparators, and all prespecified model artifacts frozen in `frozen_protocol.json`; do not retune them on transfer outcomes. Retain stable-62 as the already specified broader comparison, not a newly validated winner. Acquire a genuinely post-freeze chronological cohort with contemporaneous source snapshots, complete outcome follow-up and adjudicated UCL episodes. Use the frozen validation-derived threshold, report realized alert burden, and audit injuries and dates blind to prediction scores. Current results warrant further study, not clinical deployment or a confirmatory performance claim.

## Files

- `notebooks/ucl_dynamic/`: endpoint/cohort; models/warnings; trajectories/robustness.
- `reports/ucl_dynamic/tables/`: complete adjudication, evidence links, event opportunities, comparisons, bootstrap/refit uncertainty, sensitivities, calibration and QA.
- `reports/ucl_dynamic/figures/`: six PNG/PDF figures, numbered 01–06.
- `data/injury/analysis/ucl_dynamic/`: immutable-source derived data, predictions and fitted model caches.
- `reports/ucl_dynamic/frozen_protocol.json`: exact representations, fit signatures, validation thresholds and freeze timestamp.

Rerun/resume: `.venv/bin/python -m src.injury_analysis.ucl_dynamic_pipeline` then `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-dynamic`. Checkpoint hashes fail loudly on changed inputs. The database is opened read-only.
