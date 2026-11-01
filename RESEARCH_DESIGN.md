# Final research direction: a focused extension of Kang et al.

For the **replication-only parent-paper experiment**, use the separate
[Kang replication stage](notebooks/kang_replication/README.md), its
[protocol](docs/kang_replication_methods.md), and
[results](reports/kang_replication/summary.md). It uses R XGBoost, the published
repeated-selection structure and GAM slope rule. The earlier foundation below
is retained as a preliminary approximation, not overwritten or relabeled.

Start with the [Kang benchmark specification](docs/kang_benchmark_spec.md),
[computed foundation results](reports/kang_benchmark/summary.md), and
[benchmark notebook](notebooks/kang_benchmark/01_benchmark_foundation.ipynb).

The parent study is Kang LS, Mavrommatis S, Ina JG, et al., *Orthopaedic Journal
of Sports Medicine* 2026;14(9), [doi:10.1177/23259671261477532](https://doi.org/10.1177/23259671261477532).
This repository contains an approximate, explicitly qualified reference implementation;
it does not claim an exact reproduction of the authors' unpublished code.

| Stage | Research comparison | Status |
|---|---|---|
| Reference | Broad shoulder/elbow IL events, pitcher-games, role strata, XGBoost, cumulative windows, random-game evaluation | Foundation implemented; cohort and modeling differences documented |
| Endpoint extension | Adjudicated UCL events versus the broad elbow reference, with other choices held fixed | Existing endpoint machinery reusable; controlled comparison not yet implemented |
| Evaluation extension | Strict chronological development and final evaluation versus random-game evaluation | Existing embargo/censoring machinery reusable; benchmark extension not yet implemented |
| Temporal extension | A few prespecified feature-family comparisons across preinjury lead times | Prior analyses provide hypotheses; no new family-by-window search in this stage |

The intended chronological design remains training 2015–2023, validation 2024–2025,
and reserved 2026 evaluation. **No 2026 modeling observations or outcomes enter
this benchmark.** Historical 2026 analyses elsewhere are preserved and cannot
be redescribed as never examined or prospective evidence.

PCA/PLS, profile/departure, consolidation, nonlinear, dynamic UCL, temporal and
replication work remain intact as supporting exploratory research. The new
reference does not import their selected predictors, fitted models or scores.
The common architecture is still MLBAM-linked Statcast appearances plus
provenance-preserving official IL episodes. No database objects were changed.

Next gate: settle the documented control/label and release-Y/extension questions,
freeze the benchmark specification, then make one attributable extension at a
time. Do not interpret the random-game reference as a deployment-ready warning
model or infer an optimal biological injury window from a cumulative AUC curve.

The existing root README has unresolved merge markers predating this stage.
It is preserved; this file provides a clean research entry point without
discarding either version of that earlier documentation.
