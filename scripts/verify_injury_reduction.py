"""Validate executed PCA artifacts, temporal support and unchanged source data."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.injury_analysis.common import source_signature,sha,CACHE as OLD_CACHE
from src.injury_analysis.reduction import OUT,CACHE,CFG
from src.injury_analysis.reduction_prediction import choose_dimension


def main():
    checks={}
    manifest=json.loads((OUT/'notebook_execution.json').read_text())
    checks['source_unchanged']=manifest['source_signature']==source_signature()
    checks['three_notebooks_executed']=len(manifest['notebooks'])==3 and all(
        n['success'] and n['errors']==0 and n['executed_cells']==n['code_cells'] for n in manifest['notebooks'])
    checks['executed_notebook_hashes_match']=all(
        sha(ROOT/'notebooks/injury_dimensionality_reduction'/n['name'])==n['sha256'] for n in manifest['notebooks'])
    checks['analysis_code_hashes_match']=all(sha(ROOT/p)==v for p,v in manifest['code_sha256'].items())
    for suite in ['injury_type_analysis','injury_signal_analysis']:
        old=json.loads((ROOT/'reports'/suite/'notebook_execution.json').read_text())
        checks[suite+'_notebooks_preserved']=all(sha(ROOT/'notebooks'/suite/n['name'])==n['sha256'] for n in old['notebooks'])
    tables=OUT/'tables'
    f=pd.read_csv(tables/'predictive_performance.csv')
    checks['all_comparisons_present']=len(f)==140 and f.variant.nunique()==10 and f.category.nunique()==7
    checks['identical_rows_across_variants']=bool((f.groupby(['category','partition'])[
        ['rows','positive_rows','represented_events','prevalence']].nunique()==1).all().all())
    selection=pd.read_csv(tables/'validation_dimension_selection.csv').set_index('category').selected_variant
    checks['selection_reproduces_validation_only_rule']=all(
        choose_dimension(f.loc[f.category.eq(c)&f.partition.eq('validation')])==v for c,v in selection.items())
    sanity=pd.read_csv(tables/'full_rotation_sanity.csv')
    checks['full_rotation_numerically_equivalent']=bool(sanity.max_score_difference.lt(.001).all()
                                                       and sanity.ap_difference.abs().lt(.00001).all())
    predictions=pd.read_parquet(CACHE/'predictions.parquet')
    checks['unique_predictions_per_category_row']=not predictions[['category','partition','row_id']].duplicated().any()
    checks['no_positive_event_crosses_validation_test']=all(
        not set(g.loc[g.partition.eq('validation')&g.target.eq(1),'next_event_id']) &
            set(g.loc[g.partition.eq('test')&g.target.eq(1),'next_event_id'])
        for _,g in predictions.groupby('category'))
    a=pd.read_parquet(OLD_CACHE/'appearances.parquet')
    training=a.loc[a.analysis_eligible&a.partition.eq('train')]
    maps=joblib.load(CACHE/'descriptive_pca.joblib')
    checks['descriptive_fit_uses_training_row_count']=all(
        enc.fit_rows==len(training) and enc.pca.n_samples_==len(training) for enc in maps.values())
    checks['orthonormal_pca_bases']=all(np.allclose(
        enc.pca.components_@enc.pca.components_.T,np.eye(enc.pca.n_components_),atol=1e-10) for enc in maps.values())
    aligned=pd.read_parquet(CACHE/'latent_alignment.parquet')
    matches=pd.read_csv(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv')
    checks['original_match_ids_preserved']=set(aligned.event_id)==set(matches.event_id)
    checks['no_duplicate_aligned_outings']=not aligned[['event_id','kind','pitcher_id','game_id']].duplicated().any()
    checks['strictly_pre_anchor']=bool(aligned.game_date.lt(
        aligned.anchor_date.where(aligned.kind.eq('Injury'),aligned.control_anchor_date)).all())
    held=aligned.loc[aligned.heldout_2022_2025]
    checks['primary_latent_rows_outside_training']=bool(held.game_date.between('2022-01-01','2025-12-31').all())
    for prefix in ['profile','departure']:
        cols=[prefix+'_PC'+str(i) for i in range(1,4)]
        checks[prefix+'_incomplete_measurements_not_plotted']=bool(
            aligned.loc[~aligned[prefix+'_complete'],cols].isna().all().all())
    summary=pd.read_csv(tables/'latent_balanced_trajectories.csv')
    keys=['cohort','feature','mode','category']
    members=pd.read_parquet(CACHE/'latent_balanced_membership.parquet')
    counts=members.groupby(keys).event_id.nunique().rename('membership_events').reset_index()
    joined=summary.merge(counts,on=keys,validate='many_to_one')
    checks['balanced_support_matches_membership']=bool(joined.events.eq(joined.membership_events).all())
    refits=pd.read_csv(tables/'training_refit_repeats.csv')
    checks['paired_training_refits_complete']=len(refits)==4*2*CFG['pitcher_bootstrap_refits'] and bool(
        refits.groupby(['category','variant']).repeat.nunique().eq(CFG['pitcher_bootstrap_refits']).all())
    result=dict(checks=checks,all_passed=all(checks.values()),source_signature=source_signature(),
                csv_tables=len(list(tables.glob('*.csv'))),figures=len(list((OUT/'figures').glob('*.png'))),
                matched_pairs=len(matches),heldout_matched_pairs=held.event_id.nunique(),
                tests_log='tests.log')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['all_passed']:
        raise AssertionError('Failed checks: '+str([k for k,v in checks.items() if not v]))


if __name__=='__main__':
    main()

