# Focused elbow/UCL latent visuals

**Main result.** PLS retains validation ranking with fewer continuous axes than PCA. The matched pre-event analysis has 0 of 63 component/time cells with a joint interval excluding zero; these figures do not establish a common ahead-of-time injury warning.

**Endpoint scope.** The saved models predict the established **Elbow / UCL** category, which includes nonspecific elbow inflammation, soreness and strains. They are not confirmed-UCL-only models. Of 62 five-outing matched events, only 4 explicitly name UCL injury at initial placement; 15 have later supplemental UCL confirmation. Later confirmation is audit metadata, never a predictor or a substituted injury date. A four-event subgroup cannot support a separate trajectory claim.

## Figure guide

| Figure / question | Computed result | Profile versus imminent warning | Use in further development |
|---|---|---|---|
| [01_ucl_pls_matched_trajectories](figures/01_ucl_pls_matched_trajectories.pdf): Persistent offset or increasing case–control divergence? | 62 balanced pairs. D1 paired last-minus-first change +0.12 [-0.36, +0.63] SD; C1 +0.08 [-0.20, +0.39] SD. | Neither a stable susceptibility offset nor a shared imminent-warning rise is clearly established in these matched means. | D1 prioritizes lower velocity relative to personal baseline; retain as a candidate, not a validated warning marker. |
| [02_ucl_component_lead_time_heatmap](figures/02_ucl_component_lead_time_heatmap.pdf): When does matched separation emerge across retained supervised axes? | 0/63 cells have joint 95% bands excluding zero; 62 outing pairs and 32 calendar pairs. | Inspect persistent versus late patterns, but no supported onset of divergence can be assigned from color alone. | Prevents selecting a lead-time window from the largest unadjusted heatmap cell; calendar and outing cohorts differ. |
| [03_ucl_pca_pls_compression](figures/03_ucl_pca_pls_compression.pdf): How compact is predictive information compared with PCA? | Original inputs: PLS 10 versus PCA 30 selected axes; validation AP 0.0300 versus 0.0285, unreduced 0.0284. | Supervision can compress ranking information; the curve does not distinguish susceptibility from imminence. | A smaller candidate representation is feasible, but context/missingness columns remain and predictive benefit is uncertain. |
| [04_individual_ucl_latent_trajectories](figures/04_individual_ucl_latent_trajectories.pdf): Does the average hide heterogeneous individual patterns? | D1 final-minus-early score rises in 40/62 cases and 26/62 matched controls; case IQR [-0.38, +0.92] SD. | Persistent, rising, falling and irregular paths coexist; ordering exposes variation without identifying phenotypes. | Requires prospective event-level warning evaluation; a rise or positive score is not a calibrated risk threshold. |

## Component meaning and time scale

D1 is principally a **lower fastball-velocity departure** direction: `velo_z` supplies 77.4% of squared axis weight, with smaller spin and extension contributions. D2 emphasizes spin and extension departures. C1 mixes velocity departure with prior profile and career/context fields. C2 is more profile-weighted. These are descriptions of standardized linear coordinates, not tissue mechanisms or fractions of explained injury risk.

| Axis | Profile squared weight | Departure squared weight | Absolute training logit weight per score SD |
|---|---:|---:|---:|
| D1 | 0.0% | 100.0% | 0.2649 |
| D2 | 0.0% | 100.0% | 0.0505 |
| C1 | 54.2% | 45.8% | 0.4297 |
| C2 | 76.7% | 23.3% | 0.1484 |
| C3 | 64.6% | 35.4% | 0.0644 |
| C4 | 92.1% | 7.9% | 0.0500 |
| C5 | 59.4% | 40.6% | 0.0001 |

The nearly zero classifier weight on C5 illustrates why an apparently colored latent trajectory is not necessarily predictive. All seven validation-retained axes remain in the heatmap; the four emphasized axes and D1 individual plot were selected by training logit weight times training score SD. Higher plotted scores point toward higher fitted logit *holding other model coordinates fixed*. They are not probabilities or complete model predictions.

Average paired offset: D1 +0.11 [-0.12, +0.32] SD; C1 +0.14 [-0.05, +0.33] SD (descriptive 95% intervals). Their uncertainty includes zero. A flat group offset, even if established later, would not by itself be an imminent warning.

The last observed outing is a median **1 day** before the recorded event (IQR 1–1); the fifth-last is 16.5 days before. An apparent last-outing change therefore cannot substantiate a seven-day warning. Calendar bins use actual elapsed days, with one mean per event and group in each bin; no interpolation or smoothing is applied.

## Compression versus useful warning

On 26,796 validation appearances (600 positive rows, 89 represented events; prevalence 2.24%), PLS reaches its best tested original-input validation AP with 10 axes, versus 30 for PCA. These retain 41.8% and 90.0% of continuous training variance, respectively. This supports compact outcome-directed ranking relative to the tested PCA representations; it does not establish a biological low-dimensional injury process.

The selected departure model uses 2 PLS versus 8 PCA axes; combined inputs use 5 versus 20. Each method retains the same unprojected binary context/missingness fields within a representation. The original-input PLS classifier therefore has **57 total columns**, not 10. Refer to `compression_summary.csv` for the full accounting.

PLS-minus-unreduced validation AP is +0.0016 (paired pitcher-bootstrap interval [-0.0016, +0.0042]). These conditional intervals do not account for validation selection. The plotted candidate grid is finite; neither a precise optimal dimension nor a statistically established plateau is claimed.

For the already examined 2024–2025 period, the existing 5% appearance-alert budget detects 18.6% of broad elbow/UCL events 7–30 days ahead using PLS, versus 17.9% with unreduced features. The paired difference is +0.7 percentage points [-4.0, +5.3]. Thus compression does not demonstrate a clear practical-warning improvement. These inherited metrics use the prior evaluable risk set, all-event denominator and retrospective top-score budget; they are not a prospective deployment trial.

## Design and uncertainty

- Frozen PLS/PCA transforms, imputers, scales and classifiers were learned through 2021. Existing 2022–2023 validation selection is reused; no new fitting or test-based component selection occurs.
- Both matched windows and anchors are in the previously examined 2022–2025 interval. Cases and controls retain their original identities and outing alignment.
- Complete profile-plus-departure measurements are required for descriptive scores. The same 62 events contribute at each of five outings; 32 events contribute at every calendar bin. These are different subsets and may be selected by data/history availability. Predictive curves retain the original larger imputed risk sets.
- Uncertainty resamples connected sets of matched pairs sharing either case or control pitchers: 54 outing clusters and 29 calendar clusters, 300 deterministic replicates. Recurrent cases and pitchers switching case/control roles remain together.
- Joint 95% max-t bands cover all retained axes and time points within each alignment and separately for case means, control means and paired differences. Shape-contrast intervals are descriptive and are not jointly corrected across all contrasts. Model fitting, matching and dimension selection uncertainty are not resampled.
- Per-event scores use frozen training-score SD units; there is no event-specific centering or scaling. The individual plot shares row order across cases, controls and differences. Sorting by final-minus-early case score necessarily displays a gradient and is not evidence of discrete phenotypes. Display colors clip at ±3 SD; exported values remain unclipped.
- Predictors describe completed outings. Personal baselines use prior observations. Event dates remain recorded injury/IL proxies, not verified biological onset, surgery date or return date.
- Matched separation is descriptive, not a sensitivity/specificity estimate. The 2024–2025 period has been inspected repeatedly and supplies no new independent confirmation.

## What to freeze next

First adjudicate the intended UCL-specific endpoint against the existing procedure/source audit, retaining known-at-the-time dates separately from later confirmation. D1 provides a transparent velocity-departure candidate; test whether strictly prior persistence/trends add to profile risk at a predeclared 7–30-day warning window and controlled alert budget. Freeze features, transformations, outcome rules and validation-selected thresholds before evaluating a genuinely untouched chronological cohort. The current figures support a compact representation and heterogeneous trajectories; they do not validate an imminent UCL warning system.

## Scope and files

Four PNG/PDF figure pairs are exported. Loading interpretation is supplied as a table instead of another family plot. No relevance-versus-stability plot is added because prior training refits covered shoulder/lower-body models, not this elbow/UCL target. No horizon-specific predictive heatmap is fabricated from calendar-aligned means; that requires separately defined horizon outcomes and evaluation.

- [Figure summary CSV](tables/figure_summary.csv)
- [Component interpretation](tables/component_interpretation.csv) and [all loadings](tables/component_loadings.csv)
- [Balanced trajectories and uncertainty](tables/balanced_trajectories.csv)
- [Endpoint audit](tables/endpoint_definition_audit.csv) and [cohort support](tables/cohort_support.csv)
- [Event row lookup](tables/individual_event_order.csv) and [unclipped event cells](tables/individual_event_cells.csv)
- [Executed notebook](../../../notebooks/ucl_latent_visuals/01_ucl_latent_visuals.ipynb)
- [Methods and rerun commands](../../../docs/ucl_latent_visuals.md)
