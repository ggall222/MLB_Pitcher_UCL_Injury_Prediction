# Initial UCL temporal signal decomposition

This is development-stage signal discovery, not an independently confirmed early-warning system. The refined endpoint, source dates, frozen feature sets, chronological splits and database are unchanged.

## Design and support

Four disjoint future intervals: **1–7, 8–14, 15–30 and 31–60 days**. All primary models use the same observed 60-day first-event cohort: 64,646 training, 22,269 validation and 22,782 exploratory-transfer appearances. A known other injury ends the first-event risk interval; ambiguous outcomes remain censored. A UCL event in a different interval is a negative for the specific interval, not a healthy-pitcher designation.

| Validation interval | Positive appearances | Represented events | All qualifying events |
|---|---:|---:|---:|
| 1-7 days | 52 | 25 | 34 |
| 8-14 days | 28 | 20 | 34 |
| 15-30 days | 58 | 17 | 34 |
| 31-60 days | 51 | 10 | 34 |

The 31–60-day estimate is supported by only 10 validation events. Interval widths and event composition differ; PR lift normalizes prevalence but does not make the prediction tasks identical. All-event warning denominators include events without eligible opportunities.

The 113-field union of Background, Profile, Departure-8, Previous-62 and Stable-62 defines 11 families. Fixed C=1 logistic regression is used for all 112 interval fits. Scaling, clipping, imputation and missingness flags are training-derived. No interval boundaries, representations or regularization were tuned. Arm angle and within-outing velocity decline remain outside this long-history analysis because of prior coverage exclusions; documented prior UCL history remains excluded for unverified publication timing. Prior publicly reported IL count is included in background.

## What the estimates show

| Family/reference | 1–7d PR lift | 8–14d | 15–30d | 31–60d |
|---|---:|---:|---:|---:|
| Velocity | 10.66 | 1.25 | 1.77 | 4.86 |
| Workload and rest | 1.80 | 2.26 | 3.52 | 7.24 |
| Pitcher background and experience | 2.26 | 1.69 | 1.09 | 1.50 |
| Role and recent starts | 1.69 | 1.35 | 2.15 | 2.47 |
| Departure | 2.26 | 1.91 | 1.11 | 1.31 |
| Profile | 1.31 | 1.42 | 1.29 | 3.51 |
| Previous 62 | 11.79 | 2.08 | 1.17 | 2.08 |
| Stable 62 | 4.06 | 3.36 | 1.95 | 2.15 |

**0 of 33 predeclared family timing contrasts exclude zero under joint 95% bootstrap bands.** Point estimates alone do not establish acute, intermediate or persistent categories. Family ablation and grouped permutation sometimes disagree; the families share correlated information, and refitting after ablation can reduce overfitting. The permutation measure is marginal within-season reliance, not a conditional or causal effect.

## Event concentration and stability

- Velocity, 1-7 days: one event contributes **80.6%** of AP. Excluding that pitcher from evaluation changes AP from 0.02489 to 0.00403. This is an influence diagnostic; the event remains in the dataset.
- Workload and rest, 31-60 days: one event contributes **69.4%** of AP. Excluding that pitcher from evaluation changes AP from 0.01659 to 0.00397. This is an influence diagnostic; the event remains in the dataset.

Two families nominated using development-only mean normalized ablation loss: **Velocity, Pitcher background and experience**. These are follow-up candidates, not established temporal signals. Six common whole-pitcher training resamples preserve histories and repeat preprocessing.

| Nominated family | Interval | Positive ablation in training refits | Refit loss range (AP / prevalence) |
|---|---|---:|---:|
| Velocity | 1-7 | 5/6 | -0.03 to 1.31 |
| Velocity | 8-14 | 4/6 | -0.01 to 1.77 |
| Velocity | 15-30 | 5/6 | -0.04 to 0.26 |
| Velocity | 31-60 | 4/6 | -0.08 to 7.87 |
| Pitcher background and experience | 1-7 | 5/6 | -2.17 to 0.92 |
| Pitcher background and experience | 8-14 | 5/6 | -0.03 to 1.06 |
| Pitcher background and experience | 15-30 | 1/6 | -1.55 to 0.41 |
| Pitcher background and experience | 31-60 | 5/6 | -0.15 to 10.03 |

Only **10 validation events** have eligible opportunities in every interval. A shared-event evaluation sensitivity is saved for every family/model; it changes event composition without fitting or selecting a new model. Interval-upper-follow-up sensitivity for Full union, Profile and Departure is also retained. Neither sensitivity is a prospective validation.

## Matched trajectories and interpretation

Each family trajectory uses one fixed 60-day training-only logistic score, standardized using its training log-odds distribution. The score does not change depending on the retrospectively known lead interval. Original matched controls are reused. Calendar plots balance the same pairs across all four bins; outing plots balance final, 2–3 and 4–5 appearances. Available-pair summaries for 6–10 and 11+ outings are separate and never treated as the same calendar durations. Bootstrap clusters connect events sharing either case or control pitchers.

- calendar: 7–7 balanced events per family; 2/11 unadjusted family difference-change intervals exclude zero. These exploratory trajectory contrasts are not corrected across families.
- outings: 17–17 balanced events per family; 3/11 unadjusted family difference-change intervals exclude zero. These exploratory trajectory contrasts are not corrected across families.

Persistent case-control offsets are profile associations, not proof of an approaching injury. Score direction combines the family’s current levels, historical levels and departures; it is not a physiological latent construct. Sparse balanced support and the same previously examined cohorts limit inference about shape.

Velocity-family matched difference changes are near zero: +0.027 training-score SD in calendar alignment (95% interval −0.224 to +0.331), and +0.017 SD in outing alignment (−0.203 to +0.236). Workload-family case-control scores decrease toward the event in both alignments (−0.404 and −0.448 SD; unadjusted intervals below zero). This is not evidence of a general rise in workload risk near injury; the score blends workload level, rest and changes, and only seven calendar-balanced events contribute.

## Transfer, warning and next step

The development family choices, model signatures, interval definitions and contrast results were saved before scoring 2024–2025. All frozen families and null findings were transferred; transfer cannot redefine a candidate or claim independent confirmation. Warning comparisons use retrospective top-5% appearance budgets for each interval, not deployable thresholds. The 1–7-day endpoint is more proximal than the existing ≥7-day practical-warning objective.

Departure-8 PR lift is **2.26, 1.91, 1.11, 1.31** across validation intervals and **1.74, 1.07, 1.26, 1.47** in exploratory transfer. Its near-minus-far validation contrast is +0.95 (95% interval −0.52 to +2.94), so stronger near-term discrimination remains uncertain. Profile’s 31–60-day lift drops from **3.51** in validation to **0.90** in transfer. At the 5% budget, Departure-8 warns on 9/34 validation events at 1–7 days but 4/43 transfer events; the corresponding 8–14-day figures are 4/34 and 1/43. Broader models have stronger transfer ranking in some intervals, but this does not validate the proposed temporal family ordering.

**The temporal heterogeneity hypothesis remains unconfirmed in this first pass.** Use the nominated families only for narrow, prespecified replication questions. Do not launch flexible lag searches or freeze an acute/intermediate/chronic physiological narrative from these heatmaps. Freeze the interval definitions and family mapping for a genuinely post-freeze chronological cohort, and improve event/timing support before escalating model complexity.

See [methods](../../docs/ucl_temporal_methods.md), [notebooks](../../notebooks/ucl_temporal/README.md), `tables/family_matrix_validation.csv`, `tables/temporal_contrasts_validation.csv`, `tables/followup_sensitivity.csv`, and `tables/shared_event_sensitivity_validation.csv`.
