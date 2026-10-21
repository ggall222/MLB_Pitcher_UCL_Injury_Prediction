# Interpreting the injury-type figures for a paper or presentation

The eight notebooks produced **11 distinct figures**. Notebook 08 repeats three selected figures rather than introducing a new experiment. This guide groups the figures into eight analytical messages. It describes the saved results and adds a separately identified random-alert reference calculation; the original figures and model results have not been changed.

**Central finding:** Lower pitcher-relative velocity preceded several categories of documented IL episodes, particularly shoulder and elbow/UCL episodes. These associations did not establish injury-type-specific mechanisms or strong practical advance-warning performance.

The population includes **both starters and relievers**, not only starting pitchers. Injury timing means the conservative recorded injury/IL-date proxy, not verified biological onset. The prediction timestamp is after a completed appearance. Elbow/UCL is a broad anatomical category, not a set composed exclusively of confirmed UCL tears or Tommy John procedures.

## 1. Cohort composition and historical coverage

Figures: [cohort overview](figures/01_cohort_overview.pdf), [feature coverage](figures/01_feature_coverage.pdf).

The gray bars are all documented episodes; blue bars are the subset passing the analysis rules. There are **5,381 canonical episodes and 3,552 eligible targets**. Among eligible targets, shoulder (844), lower body (763), and elbow/UCL (650) are the largest groups. The difference between gray and blue reflects exclusions, including postoperative episodes and uncertain timing/anatomy; it is not a change in underlying injury frequency.

The year panel describes recorded episode counts. It does **not** estimate exposure-adjusted injury incidence. Changing source ascertainment, the shortened 2020 season, and incomplete 2026 observation prevent a direct claim that injury rates rose or fell.

The coverage heatmap shows the fraction of appearances with nonmissing measurements. Arm angle is absent before 2020 in this snapshot, partially available in 2020, and broadly available from 2021. The standardized arm-angle baseline needs prior observations, so it becomes usable more slowly. Velocity-decline-within-game is available in only about 16–22% of appearances because its definition requires two disjoint 20-fastball segments. That is largely structural in a mixed starter/reliever cohort.

**Suggested caption:** “Composition of documented MLB pitcher IL episodes and availability of existing appearance-level measurements across the study period. Eligibility applies timing, anatomy and episode-quality rules; counts are not exposure-adjusted incidence estimates. The 2026 season is incomplete.”

**Use:** Main-paper cohort description; feature availability is useful in methods or supplementary material.

## 2. Calendar and appearance-relative trajectories

Figures: [calendar trajectories](figures/02_calendar_trajectories.pdf), [appearance trajectories](figures/02_appearance_trajectories.pdf).

Read left to right as the recorded episode approaches. Solid lines are injury-associated observations; dashed lines are matched controls. Shaded bands are exploratory 95% pitcher-bootstrap intervals around the case means, not prediction intervals for an individual pitcher. Each event contributes one mean per calendar bin; the appearance plot shows up to five preceding outings within 60 days.

For velocity, zero means the appearance matches the pitcher's recent prior baseline. A negative z-score means slower than that baseline. Baselines use up to 20 strictly prior outings, with at least five valid observations. These units are **not mph**, and they are not pooled between-pitcher Cohen's d. Workload acceleration is plotted in pitches/day, unlike the standardized panels.

The clearest visual pattern is lower velocity near shoulder and elbow/UCL documentation, with controls closer to their baselines. Extension, release variability and workload do not show a consistent deterioration shared by all categories. The forearm trajectory is not a smooth monotonic decline across every time bin.

These are event-aligned group means. Changing numbers of events with data in each bin mean they are not complete longitudinal trajectories of an identical set of pitchers. Strong claims about changes from one bin to another need a balanced-event analysis or direct within-event contrasts. They can also reflect already-existing symptoms before documentation.

**Suggested caption:** “Event-weighted pitching trajectories before the recorded injury/IL-date proxy and matched control dates. Solid and dashed lines indicate injury and control observations, respectively. Standardized measures reference each pitcher's strictly prior history. Shading shows exploratory pitcher-bootstrap 95% intervals; available event counts vary by time bin and feature.”

**Use:** A velocity-focused presentation panel; retain the complete panel set in the paper or supplement.

## 3. Matched pre-injury effects

Figure: [matched effects](figures/02_matched_effects.pdf).

Each point is the mean **injury minus matched-control** difference for the preceding 1–14 days. Zero is no average matched difference. Bars are exploratory pitcher-bootstrap 95% intervals. An interval crossing zero leaves the direction uncertain; it does not establish no effect. An interval excluding zero is not a multiplicity-adjusted significance finding.

| Category | Velocity difference, pitcher-standardized units | 95% interval | Event pairs with observed velocity |
|---|---:|---:|---:|
| Shoulder | −0.389 | −0.546 to −0.241 | 276 |
| Elbow/UCL | −0.378 | −0.554 to −0.207 | 225 |
| Forearm | −0.299 | −0.542 to −0.033 | 84 |
| Lower body | −0.157 | −0.314 to 0.002 | 280 |

Velocity is the clearest average signal in this figure. Most extension and release-variability intervals overlap zero. Forearm vertical release variability is **lower**, not higher, than controls (−0.280; interval −0.512 to −0.018); this isolated exploratory result does not support a universal “increasing mechanical instability” explanation.

Arm effects may look larger than lower-body effects, but the figure does not contain a direct arm-versus-non-arm statistical contrast. Nor does an effect appearing clearer in one group establish a difference between groups. In the separate pooled comparison table, explicitly throwing-side arm cases and non-arm cases both have negative mean velocity differences.

**Suggested caption:** “Matched differences in pitcher-standardized measurements during the 14 days preceding documented IL episodes. Velocity was lower relative to controls for shoulder and elbow/UCL episodes, whereas extension and vertical release-variability differences were generally less consistent. Points show event-level mean paired differences and bars show exploratory 95% pitcher-bootstrap intervals.”

**Use:** The strongest single figure for the descriptive research finding.

## 4. Comparative predictability

Figure: [predictability](figures/03_predictability.pdf).

The left panel is average precision divided by positive-row prevalence: **PR lift = 1** is the prevalence reference. Lift 2 does not mean 200% accuracy, twice the actual injury probability, or detection of twice as many injury events. Average precision summarizes the precision–recall ranking over thresholds; it is not precision at the chosen 5% alert threshold.

| Category | Positive-row prevalence | Average precision | PR lift |
|---|---:|---:|---:|
| Elbow/UCL | 2.56% | 4.32% | 1.69 |
| Shoulder | 3.05% | 5.65% | 1.85 |
| Forearm | 1.02% | 2.15% | 2.10 |
| Lower body | 2.30% | 4.75% | 2.06 |

These are chronological **2024–2025 test** results after training through 2021 and validation in 2022–2023. The right panel shows that precision remains low over much of the recall range. Its display is clipped at precision 0.5, as the title states.

The defensible conclusion is modest ranking information with substantial uncertainty. Lower-body performance is comparable to arm-category performance, so the model results do not establish throwing-arm specificity. The forearm point estimate is based on only 38 represented test events and has a wide interval; it is not established as the “most predictable” category.

Each model compares its category with a common observed IL-free pool; other impending injuries are excluded. These are **not head-to-head injury-type classifiers**, and their differing prevalences/cohorts constrain comparisons. Scores have not been established as calibrated absolute MLB injury probabilities.

**Suggested caption:** “Chronological test performance of fixed 30-day logistic baselines by injury category. Average precision is normalized by category-specific positive-row prevalence; intervals resample test pitchers. Models show modest ranking improvement, with overlapping uncertainty and comparable performance for lower-body injuries. Category-versus-IL-free comparisons do not establish discrimination between injury types.”

**Use:** Main results, paired with the warning-budget figure rather than presented alone as practical utility.

## 5. Observed absence and incomplete severity outcomes

Figure: [severity follow-up](figures/04_severity_followup.pdf).

Each bar sums to all eligible episodes in that category. Gray means the duration is unresolved or the return was administrative, **not** that recovery was fast or no return occurred. Only **2,252 of 3,552 episodes (63.4%)** have usable activation-based durations.

Among those with usable durations, the median observed absence is 33.5 days for elbow/UCL, 33 days for shoulder, and 21 days for lower-body episodes. These are conditional descriptions. Elbow/UCL has substantial missing/administrative follow-up, so this figure cannot establish its true recovery-time distribution or reliably compare biological severity across categories.

The plot does not test whether severe injuries are more predictable. That requires the separate pre-injury severity-stratified analyses, adequate follow-up, and careful handling of selective missing outcomes.

**Suggested caption:** “Distribution of observed IL absence and unresolved or administrative returns among eligible episodes. Observed-duration bins are descriptive absence proxies rather than clinical severity categories; missing durations remain visible and are not imputed.”

**Use:** A useful limitations/severity figure; avoid treating it as proof of prognostic severity differences.

## 6. Recurrence and anatomical transitions

Figure: [recurrence transitions](figures/05_recurrence_transitions.pdf).

Read a row as the previous qualifying category and a column as the next category. Cell text is the number of transitions; color is the fraction **within that row**. The analysis includes only repeat episodes with observed MLB pitching between them. It conditions on a subsequent recorded event, rather than following every injured pitcher to estimate recurrence probability.

There are **2,031 observed transitions**. Of these, **584 (28.8%)** remain in the same broad heatmap category. The summary's **22.3%** refers to the same *detailed anatomical region*, a narrower definition. These numbers must not be used interchangeably. A hip-to-knee transition remains within the broad lower-body group but changes detailed region.

For example, 162 shoulder-to-shoulder transitions are 35.2% of the observed transitions beginning with shoulder. That does **not** mean 35.2% of all pitchers with a shoulder injury will have another shoulder injury. The median interval between these observed qualifying repeat episodes is 350 days, not an estimated waiting time for all injured pitchers.

Darker diagonal cells may also reflect common categories. Establishing excess same-region recurrence requires a comparison accounting for category prevalence, exposure and follow-up. First observed injury is not necessarily lifetime first injury because observation begins in 2015.

**Suggested caption:** “Transitions between successive qualifying IL episodes separated by observed MLB pitching. Numbers indicate transition counts and shading indicates row-normalized proportions. Diagonal cells reflect repeated broad categories among pitchers with observed subsequent episodes, not unconditional clinical recurrence probabilities.”

**Use:** A strong secondary presentation figure with its conditioning stated explicitly.

## 7. Warning budgets and model scores approaching events

Figures: [warning budgets](figures/06_warning_budgets.pdf), [risk trajectories](figures/06_risk_trajectories.pdf).

The warning plot asks how many distinct events receive at least one alert **7–30 days before** the episode when only a fixed fraction of eligible outings can be alerted. Its primary denominator includes eligible events with no usable warning outing. Observable-event recall is separately available in the table. This is different from appearance-level recall.

At a retrospective 5% budget, event detection is 17.2% for elbow/UCL, 18.0% for shoulder, and 22.4% for lower body. Most events remain undetected. Reported median lead times of 17, 18 and 24.5 days, respectively, apply **only to detected events**.

### Additional reference check for publication

The original figure has no random-alert reference. A 5% appearance budget does not imply 5% expected event recall: one event can have several warning opportunities. I computed the exact expected recall of choosing the same number of alerts uniformly at random from each existing comparison cohort, using the same 7–30-day window and event denominator.

| Category | Model event recall | Uniform-random expected recall | Difference, percentage points |
|---|---:|---:|---:|
| Elbow/UCL | 17.2% | 15.7% | +1.5 |
| Shoulder | 18.0% | 17.0% | +1.0 |
| Forearm | 17.4% | 19.6% | −2.2 |
| Other arm | 8.9% | 18.4% | −9.5 |
| Trunk/neck | 18.8% | 17.4% | +1.4 |
| Lower body | 22.4% | 16.2% | +6.2 |
| Hand/wrist | 21.2% | 17.4% | +3.8 |

These are point estimates against exact random expectations, not significance tests or confidence intervals for improvement. They show why above-prevalence **row-ranking** performance cannot be presented as established strong **event-warning** performance. The elbow/UCL and shoulder event-level advantages are small; forearm and other-arm point estimates fall below the uniform-random reference. Add this reference and uncertainty for the differences before making an event-level superiority claim.

The diagnostic is reproducible with:

```bash
.venv/bin/python scripts/compare_random_alert_benchmark.py
```

Output: [random_alert_event_benchmark.csv](tables/random_alert_event_benchmark.csv). With N eligible outings, K selected alerts, and m opportunities for an event, expected detection is `1 - C(N-m,K)/C(N,K)`. Averaging this probability over all eligible events gives the reference recall. Independence across events is not required for the expected value; it would matter for uncertainty.

The false-alert panel reports roughly 1.1 false alerts per represented pitcher-season at 5%, **within the selected comparison cohort**. It is not a complete-season league-wide workload estimate. Most alerts are false by the category's 30-day row label: precision at this budget is 5.4% for elbow/UCL and 7.8% for shoulder. Also, the existing false-alert definition counts 1–6-day positives as correct row alerts even though they fail the 7-day practical-warning requirement; use the separately saved non-actionable-alert burden for that stricter interpretation.

Budgets are retrospective top-percent rankings. Thresholds fixed from validation give different test alert rates: approximately 4.0% for elbow/UCL, 5.5% for shoulder and 7.8% for lower body at a nominal validation 5% budget. Do not call the plotted test budget a verified deployment threshold.

The risk-trajectory figure shows a near-event increase in mean shoulder and elbow/UCL scores, but not a universal rise across groups. Lower-body scores stay elevated rather than increasing toward the event; forearm scores are relatively flat. Scores are uncalibrated within selected category cohorts, and available events differ across bins. Avoid interpreting a score of 0.05 as a validated 5% absolute injury probability or treating cross-category score height as intrinsic risk.

**Suggested caption:** “Event detection and alert burden at retrospective appearance-level alert budgets, requiring a warning 7–30 days before the documented episode. Recall includes events without eligible warning outings; lead times are conditional on detection. These selected-cohort results require comparison with an equal-budget random-alert reference before inferring practical predictive benefit.”

**Use:** Essential practical-performance figure, but add the random reference and distinguish actionable from merely label-positive alerts for publication.

## 8. Feature families and interactions

Figure: [feature families](figures/07_feature_families.pdf).

Each cell is validation AP for the variant minus AP for the full model. For a “Without…” row, **negative/red** means removing that family hurt performance; **positive/blue** means removing it improved performance. A blue “Without release mechanics” cell is not evidence that release mechanics was the most useful family.

Examples: removing workload reduces lower-body AP by 0.0046 (about **0.46 percentage points**), whereas elbow/UCL AP is almost unchanged. Removing pitcher-history/context reduces elbow/UCL AP by 0.0033. Removing release mechanics improves elbow/UCL AP by 0.0037 in this split. The two specified interactions change AP very little, with no consistent benefit across categories.

Families overlap in information. Removing one need not remove its signal from correlated remaining predictors. Small, single-validation-split differences without uncertainty cannot establish a definitive feature ranking, a causal role, or that a physiological quantity is unimportant. Arm angle is not assessed as a historical prediction family in this plot; it was excluded for coverage reasons.

**Suggested caption:** “Change in 30-day validation average precision following removal of a feature family or addition of two prespecified workload interactions. Negative values indicate worse performance than the complete baseline; positive values indicate improvement. Effects are conditional on correlated retained predictors and are exploratory single-split comparisons.”

**Use:** Secondary results or appendix; useful for explaining the next research questions rather than announcing a final feature set.

## Suggested presentation sequence

1. Cohort overview: what data and cases are represented.
2. Velocity trajectory or matched-effects figure: the clearest descriptive signal.
3. Predictability: modest chronological ranking information, including non-arm comparisons.
4. Warning budget **with random reference added**: practical event-level benefit remains uncertain and limited.
5. Recurrence or feature-family figure, if time permits.

For a paper, retain the matched-effects and practical-evaluation figures as central results. Put the coverage heatmap, redundant time alignment, severity incompleteness and detailed ablations in methods/supplementary material as appropriate to the question.

Use the saved PDFs for scalable publication output. Captions should state the unit of analysis, event/pitcher counts, date proxy, window, comparison population, uncertainty method and time split. Use “associated with subsequent documented IL episodes,” rather than “causes injury,” “detects tissue damage,” or “predicts UCL tears.” Do not report this as a starters-only or a calibrated clinical model.

## Reusable results paragraph

“In a retrospective analysis of documented MLB pitcher IL episodes, velocity relative to each pitcher's prior baseline was lower before shoulder and elbow/UCL episodes than in matched control observations. Extension and release-variability differences were less consistent across categories. Fixed models evaluated chronologically showed modest average-precision lift, with comparable performance for lower-body injuries. Event-level warning performance was limited: at a 5% appearance alert budget, 17.2% of elbow/UCL and 18.0% of shoulder episodes were flagged 7–30 days in advance, compared with uniform-random expected recalls of 15.7% and 17.0%, respectively. These findings support further study of pre-documentation pitching changes but do not establish injury-specific or operationally useful early-warning prediction.”
