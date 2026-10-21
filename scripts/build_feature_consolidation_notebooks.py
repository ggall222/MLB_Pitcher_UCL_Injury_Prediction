"""Create only the new consolidation notebooks; refuse to replace executed files."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/feature_consolidation'
SETUP="""from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/feature_consolidation.yaml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import pandas as pd
from IPython.display import display,Markdown,Image
from src.injury_analysis.consolidation_common import OUT,CACHE,read_table
assert (OUT/'candidate_freeze.json').exists(), 'First run: .venv/bin/python -m src.injury_analysis.consolidation_workflow'
def plot(name):display(Image(filename=str(OUT/'figures'/f'{name}.png')))
display(read_table('cohort_counts'))
display(read_table('event_support'))"""

NOTEBOOKS={
'01_inventory_and_redundancy':[
('markdown',"""# Feature space: concepts, timing and redundancy

**Question:** Which engineered fields represent different baseball information, and which repeat the same quantities?

This additive stage preserves the database and previous PCA/PLS work. It inventories the 536 registered predictors plus two existing cohort context fields. All screening and correlations use **2015–2021 training data**; validation is 2022–2023. The already examined 2024–2025 period is exploratory transfer, not independent confirmation.

The inherited **Elbow / UCL** endpoint is broad initial elbow anatomy. Confirmed UCL-only event support is much smaller and is reported below. Current-outing values are known after the appearance; prior histories exclude the current date. The executable, resumable companion is `python -m src.injury_analysis.consolidation_workflow`. These notebooks inspect its verified artifacts without refitting completed stages."""),
('code',SETUP),
('markdown',"""## Schema-backed inventory and deterministic relationships

Actual DuckDB schema, source builders and dictionary define classification. Builder suffix precedence corrects generic dictionary prose for slopes and three-outing changes. Source and timing clarifications are in a separate reconciled inventory; the original numerical inventory remains unchanged.

Do not equate 538 registered/context fields with 538 usable measurements. Training-only exclusions concern missingness above 60%, fewer than 100 measured pitchers, constant fields, circular spin-axis coordinates requiring proper encoding, and unverified publication timing for prior surgery fields. Coverage predictors remain in separate sensitivity analyses. These choices define the long-history analysis, not universal exclusions for future richer-era work."""),
('code',"""inventory=read_table('feature_inventory_review')
display(read_table('feature_group_inventory'))
display(inventory[['feature','feature_group','underlying_concept','representation','source_variables',
 'timing_availability','current_appearance_included','training_missing_fraction','screen_exclusion']])
display(read_table('feature_definition_updates'))
display(read_table('deterministic_dependencies'))"""),
('markdown',"""## Within-family correlation structure

Each pitcher contributes at most 12 training appearances (13,427 rows total). Pearson and Spearman correlations use pairwise observed values, with at least 250 paired measurements. No imputation or outcome screening creates these correlations.

Complete linkage at distance `1 − |Spearman| = 0.10` identifies redundant clusters. Distinct temporal roles retain separate representatives even inside one cluster. The figure shows a readable subset of representatives in hierarchical order; full matrices, trees, and membership tables remain available. Correlation is evidence of redundancy, not sufficient grounds for deleting a field."""),
('code',"""plot('01_within_family_redundancy')
display(read_table('cluster_summary'))
display(read_table('high_correlation_pairs').head(30))"""),
('markdown',"""## Cross-family redundancy

Fractions and pair counts reveal hidden overlap, including role/workload, movement/release and historical support. The canonical summary combines unordered family pairs; it corrects display grouping without recomputing correlations. A high correlation can reflect a deterministic identity, role, pitch selection or shared measurements. It does not establish interchangeable injury information."""),
('code',"""plot('02_cross_family_redundancy')
display(read_table('cross_group_high_pairs').sort_values('spearman',key=lambda x:x.abs(),ascending=False).head(25))
display(read_table('feature_availability_by_season').query("feature in ['arm_angle_avg','spin_avg','velo_drop_game','extension_avg']"))"""),
('markdown',"""**Interpretation:** Read redundancy alongside timing. A current level, strictly historical level and current departure can correlate strongly while answering different prediction questions. The next notebook tests conditional predictive value rather than assuming the correlation representatives are sufficient.""")],
'02_predictive_value_and_stability':[
('markdown',"""# Predictive contribution, representations and selection stability

**Question:** Which families contribute beyond the remaining predictors, and do the selected representatives survive changes in the training pitchers?

All models fit the same chronological training cohort, with training-fitted medians, tail bounds, missing flags and scaling. No test outcomes enter these comparisons. L2 logistic models provide the common comparison; Elastic Net is one additional selection instrument. The target is the inherited broad elbow endpoint, not confirmed UCL injury."""),
('code',SETUP),
('markdown',"""## Family ablation and grouped permutation

Removing a family and refitting asks whether the others can substitute for it. Shuffling a family asks whether the fitted full model relies on it. Both its measurements and missing flags are shuffled together within season and role (season only when testing role). This is marginal permutation within strata, not full conditional importance.

Ablation intervals resample whole validation pitchers 300 times. Permutation bars show variation over five shuffles, **not confidence intervals**. Multiple comparisons, model dependence and only 89 represented validation injury events limit uniqueness claims. Event warning uses 7–30-day lead time and 5% of eligible rows; its denominator includes all 120 qualifying events, even events without an observable warning opportunity."""),
('code',"""plot('03_family_ablation_and_permutation')
display(read_table('family_performance')[['variant','family','features','average_precision','pr_lift',
 'brier_score','alerted_events','injury_event_recall','median_warning_lead_days','non_actionable_alerts_per_pitcher_season']])"""),
('markdown',"""## Different representations of the same concept

Small nested comparisons share age, hand, prior IL history, role and workload context. They add baseline, departure, prior slope, recent level and current level in the declared order. These contrasts are order-dependent, not exhaustive searches or independent hypothesis tests. Inspect their intervals and warning counts together.

Arm angle and within-outing velocity decline lack long-history measurement support. Separate common-row modern and starter sensitivities prevent interpreting historical missingness as proof of no signal."""),
('code',"""display(read_table('representation_comparisons')[['concept','step','addition','added_features','features',
 'average_precision','ap_increment','increment_low','increment_high','alerted_events']])
display(read_table('supported_cohort_sensitivity'))"""),
('markdown',"""## Sparse selection and whole-pitcher stability

Six Elastic Net settings span three regularization strengths and two mixing parameters, selected by validation AP. All 333 feasible substantive fields enter screening, preserving alternatives for weight exchange. Rare binary fields and missing flags remain on raw unit scale for the sparse model; continuous fields are standardized. This penalty geometry differs from the common L2 comparisons and is disclosed rather than interpreted as a controlled algorithm contest.

Sixteen refits resample entire training pitchers, holding the chosen hyperparameters fixed. Measured-value selection, missing-flag selection, sign, rank and marginal permutation stability remain separate. With only 16 refits, frequencies change in steps of 6.25%; this is a stability diagnostic, not formal false-discovery-controlled stability selection. Validation-bootstrap intervals condition on fitted models and do not incorporate the full tuning search."""),
('code',"""display(read_table('elastic_net_tuning'))
plot('04_selection_and_sign_stability')
display(read_table('feature_stability'))
display(read_table('cluster_selection_frequency'))
display(read_table('training_refit_performance'))"""),
('markdown',"""**Interpretation:** A repeatably nonzero coefficient can still have weak incremental validation value. Correlated alternatives can exchange signs or ranks; high selection frequency alone is insufficient. The final stage combines stability, temporal roles, validation ranking, warning utility and compactness.""")],
'03_candidate_set_and_freeze':[
('markdown',"""# Compact candidate set and temporal transfer

**Question:** How much predictive and warning information survives substantial reduction, and what should be frozen for a genuinely future cohort?

Candidates include the full feasible space, prior 62 fields, correlation representatives, profile/departure blocks, background context, sparse selected values and cluster representatives at four selection-frequency thresholds. Identical feature sets are deduplicated. Every comparison uses the same L2 model and training-only preprocessing."""),
('code',SETUP),
('markdown',"""## Predeclared validation-only selection rule

Retain candidates within one paired-bootstrap AP standard error of the highest validation AP. Among those, require at most two fewer events warned than the best 5% alert-budget count. Choose the smallest input set that passes, breaking ties by AP. This pragmatic tolerance is not a statistical equivalence claim; repeated validation use remains a development limitation.

The connecting line below joins **nonnested** candidate sets to aid reading; it is not a monotonic feature-elimination path. Missingness indicators are additional fitted design columns."""),
('code',"""plot('05_reduction_performance')
display(read_table('candidate_reduction_curve'))
display(read_table('conditional_feature_checks'))"""),
('markdown',"""## Measurement support versus substantive predictors

Compare explicit historical support and automatic missingness indicators separately. All alternatives use identical rows. This analysis examines whether availability, role or experience can substitute for measured information; a gain is not evidence about injury physiology. These sensitivity models do not trigger another round of test-driven feature selection."""),
('code',"""plot('06_coverage_sensitivity')
display(read_table('coverage_comparison'))
display(read_table('coverage_role_season_audit'))"""),
('markdown',"""## Frozen candidate and already examined 2024–2025 transfer

The exact feature list, preprocessing, fitted model hash and decision-table hashes were frozen **before this stage** generated test scores. The period had been examined in earlier work, so these results are not independent confirmation. No feature or threshold is selected from test results.

Retrospective top-5% ranking controls the within-period alert count but requires the period's score distribution. The separate validation-fixed threshold table shows actual test alert-rate drift. Median warning lead time is among detected events; it does not describe every injury. False alerts within 30 days and non-actionable alerts under the narrower 7–30-day warning definition are reported separately."""),
('code',"""display(read_table('frozen_performance'))
display(read_table('frozen_paired_differences'))
display(read_table('validation_fixed_thresholds'))
display(read_table('calibration_bins'))
display(read_table('retained_features')[['feature','feature_group','representation','signal_interpretation',
 'selection_frequency','conditional_sign_stability','ap_loss','low','high','reason_retained']])
display(read_table('final_feature_map'))"""),
('markdown',"""## Evidence-based synthesis

The following section is generated from completed tables, rather than written in advance. Exclusions remain recommendations for this candidate representation, not deletion from the master database."""),
('code',"display(Markdown((OUT/'summary.md').read_text()))")]
}


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for name,cells in NOTEBOOKS.items():
        path=DEST/(name+'.ipynb')
        if path.exists():
            print('Preserved '+str(path.relative_to(ROOT)))
            continue
        notebook=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(text) if kind=='markdown' else nb.v4.new_code_cell(text) for kind,text in cells],
            metadata={'kernelspec':{'display_name':'MLB UCL analysis (.venv)','language':'python','name':'mlb-ucl-analysis'},
                      'language_info':{'name':'python'}})
        nb.write(notebook,path)
        print('Created '+str(path.relative_to(ROOT)))


if __name__=='__main__':main()
