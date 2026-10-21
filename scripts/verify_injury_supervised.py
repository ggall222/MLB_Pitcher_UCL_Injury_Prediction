"""Verify temporal integrity, selected dimensions and preservation of preceding work."""
from pathlib import Path
import sys,json
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.injury_analysis.supervised_reduction import (
    OUT,CACHE,CFG,inputs,source_signature,sha,choose_components,feature_blocks)
from src.injury_analysis.supervised_workflow import STAGES,signature
from src.injury_analysis.supervised_diagnostics import model_path
from src.injury_analysis.cohort import label_frame
from src.injury_analysis.family_decomposition import target_events

def main():
    checks={}
    snap=json.loads((OUT/'prior_artifacts_snapshot.json').read_text())
    checks['source_unchanged']=source_signature()==snap['source']
    checks['all_prior_reports_and_notebooks_unchanged']=all(sha(ROOT/p)==h for p,h in snap['artifacts'].items())
    execution=json.loads((OUT/'notebook_execution.json').read_text())
    checks['three_notebooks_executed']=len(execution['notebooks'])==3 and all(
        x['success'] and not x['errors'] and x['code_cells']==x['executed_cells'] for x in execution['notebooks'])
    checks['notebook_hashes_match']=all(sha(ROOT/'notebooks/injury_supervised_reduction'/x['name'])==x['sha256'] for x in execution['notebooks'])
    checks['analysis_code_hashes_match']=all(sha(ROOT/p)==h for p,h in execution['code_sha256'].items())
    for stage in STAGES:
        record=json.loads((CACHE/f'{stage}_execution.json').read_text())
        checks[stage+'_source_and_code_match']=record['signature']==signature(stage)
        checks[stage+'_artifact_hashes_match']=all(sha(ROOT/p)==h for p,h in record['artifacts'].items())
    performance=pd.read_csv(OUT/'tables/selected_performance.csv')
    checks['all_method_block_category_comparisons_present']=len(performance)==7*4*4*2
    checks['identical_rows_and_labels_across_representations']=bool(performance.groupby(['category','partition'])[
        ['rows','positive_rows','represented_events','prevalence']].nunique().eq(1).all().all())
    curves=pd.read_csv(OUT/'tables/component_curves.csv')
    selection=pd.read_csv(OUT/'tables/validation_selection.csv')
    checks['selection_reproduces_validation_rule']=all(
        choose_components(curves.loc[curves.category.eq(r.category)&curves.block.eq(r.block)&curves.method.eq(r.method)])==r.selected_components
        for r in selection.itertuples())
    checks['same_candidate_grid_for_pls_and_pca']=all(
        set(g.loc[g.method.eq('PLS'),'components'])==set(g.loc[g.method.eq('PCA'),'components'])
        for _,g in curves.groupby(['category','block']))
    predictions=pd.read_parquet(CACHE/'predictions.parquet')
    checks['unique_category_block_row_predictions']=not predictions[['category','block','row_id']].duplicated().any()
    a,events,spec,matches=inputs();same=True;separate=True;training=True;orthogonal=True;numeric=True;totaldims=True
    prior=pd.read_parquet(ROOT/'data/injury/analysis/profile_departure/predictions.parquet')
    comparisons={'Original 62':'Original 62','Profile':'Profile only','Departure':'Departure only','Combined':'Profile + Departure'}
    max_raw_difference=0.
    for category,g in predictions.groupby('category'):
        f=label_frame(a,category,30)
        eventsets=[set(f.loc[f.partition.eq(p)&f.target.eq(1),'next_event_id']) for p in ['train','validation','test']]
        separate &= all(not eventsets[i]&eventsets[j] for i in range(3) for j in range(i))
        train=f.loc[f.partition.eq('train')]
        for block,h in g.groupby('block'):
            original=f.set_index('row_id').reindex(h.row_id)
            same &= np.array_equal(h.target,original.target) and np.array_equal(h.game_id,original.game_id)
            old=prior.loc[prior.category.eq(category)].set_index('row_id').reindex(h.row_id)
            max_raw_difference=max(max_raw_difference,float(np.max(np.abs(h['score__Unreduced'].to_numpy()-old['score__'+comparisons[block]].to_numpy()))))
            saved=joblib.load(model_path(category,block));basis=saved['basis']
            training &= basis.fit_rows==len(train) and np.isclose(basis.training_prevalence,train.target.mean())
            training &= np.isclose(basis.pls._y_mean[0],train.target.mean())
            numeric &= all(name not in ['throws_left','is_starter'] and not name.startswith('missingindicator') for name in basis.continuous)
            for method,q in basis.bases.items():
                orthogonal &= np.allclose(q.T@q,np.eye(q.shape[1]),atol=1e-10)
                k=saved['selected'][method]
                totaldims &= saved['models'][method+' selected'].n_features_in_==k+len(basis.keep)
    checks['original_risk_sets_and_outcomes_preserved']=bool(same)
    checks['no_event_crosses_temporal_partitions']=bool(separate)
    checks['unreduced_predictions_match_previous_stage']=max_raw_difference<1e-10
    checks['supervised_fit_rows_and_target_mean_are_training_only']=bool(training)
    checks['binary_and_missingness_not_rotated']=bool(numeric)
    checks['orthonormal_pls_and_pca_bases']=bool(orthogonal)
    checks['reported_total_dimensions_correct']=bool(totaldims)
    control=pd.read_csv(OUT/'tables/full_span_control.csv')
    checks['full_span_control_equivalent_within_numerical_tolerance']=bool(
        control.max_score_difference.lt(.001).all() and control.ap_difference.abs().lt(.00001).all())
    budgets=pd.read_csv(OUT/'tables/alert_budgets.csv')
    checks['all_event_denominators_preserved']=all(
        g.qualifying_events.eq(len(target_events(events,cat,p))).all() for (cat,p),g in budgets.groupby(['category','partition']))
    aligned=pd.read_parquet(CACHE/'matched_component_alignment.parquet')
    checks['matched_analyses_outside_training']=bool(aligned.game_date.between('2022-01-01','2025-12-31').all())
    checks['matched_observations_strictly_pre_event']=bool(aligned.game_date.lt(
        aligned.anchor_date.where(aligned.kind.eq('Injury'),aligned.control_anchor_date)).all())
    checks['original_443_heldout_matches']=aligned.event_id.nunique()==443 and set(aligned.event_id)<=set(matches.event_id)
    components=[c for c in aligned if '__PLS' in c or '__PCA' in c]
    checks['incomplete_measurements_not_plotted']=bool(aligned.loc[~aligned.complete,components].isna().all().all())
    membership=pd.read_parquet(CACHE/'balanced_component_membership.parquet')
    counts=membership.groupby(['block','mode','category']).event_id.nunique().rename('members').reset_index()
    trajectory=pd.read_csv(OUT/'tables/component_trajectories.csv').merge(counts,on=['block','mode','category'],validate='many_to_one')
    checks['balanced_membership_matches_trajectories']=bool(trajectory.events.eq(trajectory.members).all())
    refits=pd.read_csv(OUT/'tables/training_refits.csv')
    checks['paired_training_refits_complete']=len(refits)==2*2*2*2*CFG['training_refits']
    checks['no_failed_or_nonfinite_scores']=bool(np.isfinite(predictions.filter(like='score__').to_numpy()).all())
    result={'checks':checks,'all_passed':all(checks.values()),'preserved_prior_artifacts':len(snap['artifacts']),
        'csv_tables':len(list((OUT/'tables').glob('*.csv'))),'figures':len(list((OUT/'figures').glob('*.png'))),
        'max_unreduced_score_difference':max_raw_difference,'source_signature':source_signature()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['all_passed']:raise AssertionError([k for k,v in checks.items() if not v])
if __name__=='__main__':main()

