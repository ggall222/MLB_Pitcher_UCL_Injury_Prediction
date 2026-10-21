"""Validate frozen transforms, matched support, reproducibility and preservation."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.injury_analysis.ucl_latent_visuals import (
    OUT,CACHE,CONFIG_PATH,read_table,source_signature,sha,inputs,select_components,
    CATEGORY,model_path,paired_dependency_clusters,balanced_summary)


def main():
    checks={}
    snapshot=json.loads((OUT/'prior_artifacts_snapshot.json').read_text())
    checks['source_database_and_events_unchanged']=source_signature()==snapshot['source']
    checks['all_prior_artifact_hashes_unchanged']=all(sha(ROOT/p)==h for p,h in snapshot['artifacts'].items())
    manifest=json.loads((OUT/'manifest.json').read_text())
    checks['configuration_hash_matches']=sha(CONFIG_PATH)==manifest['config_sha256']
    checks['frozen_models_unchanged']=all(sha(ROOT/p)==h for p,h in manifest['frozen_models'].items())
    checks['analysis_code_hashes_match']=all(sha(ROOT/p)==h for p,h in manifest['code_sha256'].items())
    checks['input_hashes_match']=all(sha(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    execution=json.loads((OUT/'notebook_execution.json').read_text())
    checks['focused_notebook_executed_without_errors']=len(execution['notebooks'])==1 and all(
        x['success'] and not x['errors'] and x['code_cells']==x['executed_cells'] for x in execution['notebooks'])
    checks['executed_notebook_hash_matches']=all(sha(ROOT/'notebooks/ucl_latent_visuals'/x['name'])==x['sha256']
        for x in execution['notebooks'])
    checks['executed_analysis_code_matches']=all(sha(ROOT/p)==h for p,h in execution['code_sha256'].items())
    a,events,spec,matches=inputs()
    aligned=pd.read_parquet(CACHE/'aligned_components.parquet')
    checks['original_match_identities_preserved']=set(aligned.event_id)<=set(matches.event_id)
    keys=['event_id','kind','pitcher_id','game_id','game_date','lead_days','appearance_before']
    original=pd.read_parquet(ROOT/'data/injury/analysis/aligned_trajectories.parquet',columns=keys)
    compare=aligned[keys].merge(original,on=keys,how='left',indicator=True,validate='one_to_one')
    checks['original_matched_outings_and_times_preserved']=compare._merge.eq('both').all()
    checks['matched_windows_and_anchors_outside_training']=all(aligned[c].between('2022-01-01','2025-12-31').all()
        for c in ['game_date','anchor_date','control_anchor_date'])
    checks['scores_strictly_pre_event_or_control_anchor']=aligned.game_date.lt(
        aligned.anchor_date.where(aligned.kind.eq('Injury'),aligned.control_anchor_date)).all()
    checks['historical_predictors_strictly_prior']=all((a[c].isna()|a[c].lt(a.game_date)).all()
        for c in ['history_max_date','baseline_max_date'])
    definitions=read_table('component_selection');training_selection=True;equivalent=True
    for block,g in definitions.groupby('block'):
        model=joblib.load(model_path(CATEGORY,block))
        expected=select_components(model,block)
        training_selection &= g.component.tolist()==expected.component.tolist()
        for col in ['axis','retained_axes','orientation','training_rank','training_absolute_standardized_logit_weight']:
            training_selection &= np.allclose(g[col],expected[col])
        raw=model['basis'].transform(aligned,'PLS',model['selected']['PLS'])
        for r in g.itertuples():
            scores=raw[:,r.axis-1]*r.orientation/r.training_score_sd
            equivalent &= np.allclose(aligned.loc[aligned.complete,r.component],scores[aligned.complete])
    checks['component_selection_matches_frozen_training_rule']=training_selection
    checks['scores_reproduce_frozen_transform_and_training_scale']=equivalent
    checks['incomplete_measurements_not_plotted']=aligned.loc[~aligned.complete,definitions.component].isna().all().all()
    stored=pd.read_parquet(CACHE/'balanced_event_cells.parquet')
    summaries=read_table('balanced_trajectories');balanced=True;dependency=True;reproducible=True
    for mode,g in stored.groupby('mode'):
        positions=5 if mode=='appearance' else 4
        balanced &= g.groupby(['event_id','component']).point.nunique().eq(positions).all()
        balanced &= g.groupby('component').event_id.nunique().nunique()==1
        ids=set(g.event_id)
        clusters=paired_dependency_clusters(aligned.loc[aligned.event_id.isin(ids)])
        dependency &= g.resample_cluster_id.eq(g.event_id.map(clusters)).all()
        linked=aligned.loc[aligned.event_id.isin(ids),['event_id','pitcher_id']].drop_duplicates()
        linked['cluster']=linked.event_id.map(clusters)
        dependency &= linked.groupby('pitcher_id').cluster.nunique().eq(1).all()
        regenerated,_=balanced_summary(aligned,definitions.component.tolist(),mode)
        keys2=['mode','component','point','kind']
        first=regenerated.set_index(keys2).sort_index()
        second=summaries.loc[summaries['mode'].eq(mode)].set_index(keys2).sort_index()
        reproducible &= first.index.equals(second.index)
        reproducible &= np.allclose(first[['mean','simultaneous_low','simultaneous_high']],
                                    second[['mean','simultaneous_low','simultaneous_high']])
    checks['common_balanced_support_all_axes_and_times']=bool(balanced)
    checks['shared_case_and_control_pitchers_resampled_together']=bool(dependency)
    checks['joint_uncertainty_reproduces']=bool(reproducible)
    order=read_table('individual_event_order');cells=read_table('individual_event_cells')
    checks['individual_cells_match_unclipped_balanced_values']=all(np.allclose(
        cells.set_index(['event_id','point'])[kind].sort_index(),
        stored.loc[stored['mode'].eq('appearance')&stored.component.eq('D1')].set_index(['event_id','point'])[kind].sort_index())
        for kind in ['Injury','Control','difference'])
    checks['individual_row_order_matches_declared_rule']=order.case_late_minus_early.is_monotonic_increasing
    checks['component_squared_weights_sum_to_one']=np.allclose(read_table('component_loadings').groupby('component').squared_axis_weight.sum(),1.)
    upstream=ROOT/'reports/injury_supervised_reduction/tables'
    curves_match=True
    for name in ['component_curves','validation_selection','selected_performance','selected_differences','alert_budgets']:
        original=pd.read_csv(upstream/(name+'.csv'))
        original=original.loc[original.category.eq(CATEGORY)].reset_index(drop=True)
        try:pd.testing.assert_frame_equal(read_table('ucl_'+name),original,check_exact=False,rtol=1e-12,atol=1e-12)
        except AssertionError:curves_match=False
    checks['predictive_curves_and_warning_results_unchanged']=curves_match
    counts=read_table('cohort_support')
    balanced_ids=set(stored.loc[stored['mode'].eq('appearance'),'event_id'])
    e=events.loc[events.injury_event_id.isin(balanced_ids)]
    expected=counts.set_index('cohort').loc['Balanced five outings']
    checks['explicit_ucl_subset_audited_without_relabeling']=int(e.initial_explicit_ucl.sum())==int(expected.initial_explicit_ucl_events)
    checks['four_nonempty_png_and_pdf_pairs']=all(len(list((OUT/'figures').glob('*.'+ext)))==4 and
        all(p.stat().st_size>1000 for p in (OUT/'figures').glob('*.'+ext)) for ext in ['png','pdf'])
    checks['four_figure_interpretations_and_captions']=len(read_table('figure_summary'))==4 and (OUT/'FIGURE_CAPTIONS.md').exists()
    checks={k:bool(v) for k,v in checks.items()}
    result={'checks':checks,'all_passed':all(checks.values()),'preserved_prior_artifacts':len(snapshot['artifacts']),
            'figure_pairs':4,'csv_tables':len(list((OUT/'tables').glob('*.csv'))),'source':source_signature()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['all_passed']:raise AssertionError([k for k,v in checks.items() if not v])


if __name__=='__main__':main()
