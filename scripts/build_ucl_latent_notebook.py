"""Generate only the new focused notebook; never touch the earlier PCA/PLS suite."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/ucl_latent_visuals'
DEST.mkdir(parents=True,exist_ok=True)
cells=[('markdown',"""# Elbow/UCL latent signals and advance warning

**Question:** Do the existing supervised directions show persistent case/control differences, a change approaching injury, or heterogeneous individual paths? Can PLS preserve validation ranking with fewer dimensions than PCA?

This notebook adds four focused visuals to the frozen supervised-reduction study. It does **not** fit new models, replace prior results, or equate latent separation with a deployable warning system. The established **Elbow / UCL** target is broader than confirmed UCL injury; explicitly documented UCL support is audited below."""),
('code',"""from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/ucl_latent_visuals.yaml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.injury_analysis.ucl_latent_visuals import run,read_table,OUT
from src.injury_analysis.ucl_latent_report import write_summary
from src.injury_analysis import ucl_latent_plots as plots
summary=run()
write_summary()
display(read_table('cohort_support'))
display(read_table('endpoint_definition_audit'))
def finding(number):
    r=read_table('figure_summary').iloc[number-1]
    display(Markdown('**Computed result.** '+r.main_result+'\\n\\n**Interpretation.** '+r.profile_or_imminence))"""),
('markdown',"""## Frozen design and component selection

Transforms, imputation/scaling, and classifiers were fit through 2021. Component counts were selected on 2022–2023 validation AP in the prior stage. No component is chosen from the appearance of its held-out trajectory. We retain all two departure and five combined axes in the heatmap, emphasize the two largest training standardized logit weights per block, and use the largest-weight departure axis for individual events.

Scores are oriented using the frozen logistic coefficient and divided by their **training score SD**. Positive means a larger conditional logit contribution, not a calibrated probability. The bases are the actual orthonormalized nested PLS coordinates used by the existing classifier.

The matched analysis reuses original pair identities and outing keys, requires both windows/anchors in the already examined 2022–2025 period, and requires complete P+D fields. No imputed measurement creates a plotted biological trajectory. Every time point within an alignment uses the same event pairs. Calendar and outing panels have different balanced cohorts.

The 300 bootstrap draws keep connected pairs sharing **either case or control pitchers** together, including pitchers switching roles. Joint bands cover all retained components × positions within an alignment and group. Shape-contrast intervals are descriptive, without a joint correction across contrasts. Models, matching, and dimension selection remain fixed."""),
('code',"""display(read_table('component_selection')[['component','block','retained_axes','design_columns',
 'training_absolute_standardized_logit_weight','primary_trajectory','individual_heatmap']])
display(read_table('component_interpretation'))"""),
('markdown',"""## 1. Component trajectories: offset versus approach-to-event change

The case/control curves use five unsmoothed observed outings. The annotated contrast is `(case − control at −1) − (case − control at −5)`. Use paired contrasts, rather than whether separate case/control bands overlap, to assess divergence. A constant offset would concern profile/relative-level differences, not injury imminence. A null interval does not prove no effect."""),
('code',"""plots.trajectory_figure();plt.show()
finding(1)
display(read_table('trajectory_shape_summary').query("mode=='appearance' and component in ['D1','D2','C1','C2'] and metric in ['average_separation','last_minus_first_separation']"))
display(read_table('outing_calendar_support'))"""),
('markdown',"""## 2. Component × lead-time separation

Each cell is the equal-event-weighted matched difference in training component SD units. The zero-centered scale is shared across panels. A dot would indicate that the joint interval excludes zero. A colored cell without support is not an estimated onset of warning.

Calendar bins average available measurements separately within each event/group/bin before pairing; events do not gain more weight by making more appearances. No smoothing, interpolation, or per-event normalization is used. Later retained axes may have negligible classifier weight, so geometric separation and predictive contribution are distinct."""),
('code',"""plots.lead_time_heatmap();plt.show()
finding(2)
display(read_table('balanced_trajectories').groupby('mode')[['events','pitchers','bootstrap_clusters']].first())"""),
('markdown',"""## 3. Validation compression performance

Reuse the actual original validation AP curves: same labels, rows, candidate counts, L2 logistic classifier, and training-only preprocessing within each representation. Squares mark the **existing** validation-selected count. Each reference line uses unreduced inputs from that panel, not an unrelated feature set.

Binary context and missingness indicators bypass projection; component count is not total model dimension. These are selection-period curves, so apparent peaks are optimistic. The finite grid does not establish an exact optimum or a confidence-based plateau. Practical event warnings remain the relevant downstream comparison."""),
('code',"""plots.compression_figure();plt.show()
finding(3)
display(read_table('compression_summary').query("block in ['Original 62','Departure','Combined']"))
display(read_table('ucl_selected_differences').query("block=='Original 62' and variant=='PLS selected' and reference=='Unreduced'")[
 ['partition','ap_difference','ap_low','ap_high','event_mean','event_low','event_high']])
display(read_table('ucl_alert_budgets').query("block=='Original 62' and alert_rate==0.05 and variant in ['Unreduced','PLS selected']")[[
 'partition','variant','qualifying_events','observable_events','alerted_events','injury_event_recall','median_warning_lead_days']])"""),
('markdown',"""## 4. Individual-event heterogeneity

Each row is one original matched event pair. All three panels share row order, chosen descriptively by case score at the final outing minus the first two outings' average. This sorting necessarily creates an ordered gradient; it is **not** clustering or evidence for injury phenotypes.

Scores keep a fixed training-derived origin and scale. Case-minus-control values provide the comparison that is missing from an unpaired event-only heatmap. Colors clip at ±3 SD; the CSV cells are not clipped. IDs, names, dates and initially explicit UCL flags are in the row-lookup table. Increasing D1 or crossing zero is not a prespecified clinical alert rule."""),
('code',"""plots.individual_heatmap();plt.show()
finding(4)
display(read_table('individual_shape_distribution'))
display(read_table('individual_event_order').head(10))"""),
('markdown',"""## What these visuals support

Read the computed per-figure table below together with its uncertainty. Compression, group separation, and advance warning answer different questions. Components are weighted combinations of measured pitching variables, not physiological mechanisms. Recorded IL/injury timing is not biological onset, surgery timing, or return timing. Current-outing predictors are only available after an outing is completed.

Before a UCL-specific warning study, adjudicate the intended endpoint and freeze the horizon, feature/trajectory definitions, and validation-selected threshold. The last observed outing is often much less than seven days before the recorded event. Test a predeclared useful lead window and controlled alert budget in an untouched future chronological cohort.

Four figures are sufficient here. Family interpretation is tabulated; existing refit stability is not UCL-specific, and calendar mean contrasts cannot substitute for separately evaluated horizon-specific predictive relevance."""),
('code',"""display(read_table('figure_summary'))
print('Summary, captions and tables:',OUT/'SUMMARY.md')
print('Publication exports:',OUT/'figures')""")]
nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(s) if kind=='code' else nbf.v4.new_markdown_cell(s) for kind,s in cells])
nb.metadata.update(kernelspec={'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},
                   language_info={'name':'python','version':'3.9'})
nbf.write(nb,DEST/'01_ucl_latent_visuals.ipynb')
print('Generated',DEST/'01_ucl_latent_visuals.ipynb')
