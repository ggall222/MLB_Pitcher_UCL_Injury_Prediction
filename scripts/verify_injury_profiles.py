"""Audit the completed profile/departure analysis and preservation of prior results."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.injury_analysis.profile_departure import OUT,CACHE,CFG,inputs,source_signature,sha
from src.injury_analysis.profile_workflow import STAGES,stage_signature
from src.injury_analysis.cohort import label_frame
from src.injury_analysis.family_decomposition import target_events
from src.injury_analysis.common import CACHE as OLD_CACHE

def main():
    checks={}
    snap=json.loads((OUT/'prior_artifacts_snapshot.json').read_text())
    checks['source_unchanged']=source_signature()==snap['source']
    checks['all_prior_artifacts_unchanged']=all(sha(ROOT/p)==h for p,h in snap['artifacts'].items())
    execution=json.loads((OUT/'notebook_execution.json').read_text())
    checks['two_notebooks_executed']=len(execution['notebooks'])==2 and all(
        x['success'] and not x['errors'] and x['code_cells']==x['executed_cells'] for x in execution['notebooks'])
    checks['notebook_hashes_match']=all(sha(ROOT/'notebooks/injury_profile_departure'/x['name'])==x['sha256'] for x in execution['notebooks'])
    checks['analysis_code_hashes_match']=all(sha(ROOT/p)==h for p,h in execution['code_sha256'].items())
    for stage in STAGES:
        manifest=json.loads((CACHE/f'{stage}_execution.json').read_text())
        checks[stage+'_source_code_and_config_match']=manifest['signature']==stage_signature(stage)
        checks[stage+'_artifacts_match']=all(sha(ROOT/p)==h for p,h in manifest['artifacts'].items())
    f=pd.read_csv(OUT/'tables/model_performance.csv')
    checks['nine_variants_seven_categories_two_partitions']=len(f)==9*7*2
    checks['common_rows_labels_within_comparisons']=bool(f.groupby(['category','partition'])[
        ['rows','positive_rows','represented_events','prevalence']].nunique().eq(1).all().all())
    pred=pd.read_parquet(CACHE/'predictions.parquet')
    checks['unique_category_row_predictions']=not pred[['category','row_id']].duplicated().any()
    a,e,s,m=inputs()
    same=True;separate=True
    for category,g in pred.groupby('category'):
        f=label_frame(a,category,30)
        p=g.set_index('row_id');r=f.loc[f.partition.isin(['validation','test'])].set_index('row_id').reindex(p.index)
        same &= len(r)==len(p) and np.array_equal(p.target,r.target) and np.array_equal(p.game_id,r.game_id)
        event_sets=[set(f.loc[f.partition.eq(part)&f.target.eq(1),'next_event_id']) for part in ['train','validation','test']]
        separate &= all(not event_sets[i]&event_sets[j] for i in range(3) for j in range(i))
    checks['original_risk_sets_and_outcomes_preserved']=bool(same)
    checks['no_positive_event_crosses_partitions']=bool(separate)
    prior=pd.read_parquet(ROOT/'data/injury/analysis/dimensionality_reduction/predictions.parquet')
    # Previous PCA analysis stores model columns under its own spelling.
    print('Prior prediction columns:',[x for x in prior if 'Original' in x])
    old_column=next(c for c in prior if 'Original 62' in c)
    joined=pred.merge(prior[['category','partition','row_id',old_column]],on=['category','partition','row_id'],validate='one_to_one',suffixes=('_new','_old'))
    new_column='score__Original 62'
    if old_column==new_column:new_column+='_new';old_column+='_old'
    checks['original_62_scores_reproduced']=bool(len(joined)==len(pred) and np.allclose(joined[new_column],joined[old_column],atol=1e-10))
    budgets=pd.read_csv(OUT/'tables/alert_budget_performance.csv')
    checks['all_event_denominators_preserved']=all(
        (g.qualifying_events==len(target_events(e,cat,part))).all() for (cat,part),g in budgets.groupby(['category','partition']))
    aligned=pd.read_parquet(CACHE/'matched_score_alignment.parquet')
    checks['matched_rows_outside_training']=bool(aligned.game_date.between('2022-01-01','2025-12-31').all())
    checks['matched_rows_strictly_pre_event']=bool(aligned.game_date.lt(aligned.anchor_date.where(
        aligned.kind.eq('Injury'),aligned.control_anchor_date)).all())
    checks['matched_original_keys_preserved']=set(aligned.event_id)<=set(m.event_id) and aligned.event_id.nunique()==443
    checks['incomplete_measurements_not_plotted']=bool(aligned.loc[~aligned.complete,['profile_score','departure_score']].isna().all().all())
    support=pd.read_parquet(CACHE/'score_balanced_membership.parquet')
    support=support.groupby(['category','feature','mode']).event_id.nunique().rename('members').reset_index()
    curves=pd.read_csv(OUT/'tables/score_trajectories.csv').merge(support,on=['category','feature','mode'],validate='many_to_one')
    checks['balanced_pairs_match_curves']=bool(curves.events.eq(curves.members).all())
    checks['exact_logit_reconstruction']=bool(pd.read_csv(OUT/'tables/contribution_identity.csv').max_absolute_error.lt(1e-10).all())
    refits=pd.read_csv(OUT/'tables/training_refits.csv')
    checks['paired_training_refits_complete']=len(refits)==4*2*2*CFG['training_refits']
    result={'checks':checks,'all_passed':all(checks.values()),'prior_artifacts_preserved':len(snap['artifacts']),
        'csv_tables':len(list((OUT/'tables').glob('*.csv'))),'figures':len(list((OUT/'figures').glob('*.png'))),
        'source_signature':source_signature()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['all_passed']:raise AssertionError([k for k,v in checks.items() if not v])
if __name__=='__main__':main()

