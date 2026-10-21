"""Content-verified, restartable notebook stages without touching prior artifacts."""
import json
from .profile_departure import ROOT,OUT,CACHE,CONFIG_PATH,source_signature,sha

STAGES={
    'models':(['profile_departure.py'],[
        'model_performance','incremental_comparisons','alert_budget_performance','warning_leads',
        'reliability_bins','cohort_counts','model_coefficients','validation_fixed_thresholds',
        'model_dimensions','feature_definitions','model_specifications']),
    'scores':(['profile_scores.py','trajectory_shapes.py'],[
        'score_alignment_coverage','score_variation','contribution_identity','score_trajectories',
        'score_shape_contrasts','score_appearance_lead_days','alert_concentration']),
    'sensitivity':(['profile_robustness.py'],[
        'sensitivity_performance','sensitivity_increment','sensitivity_coverage',
        'adjusted_sensitivity_performance','adjusted_sensitivity_increment']),
    'stability':(['profile_robustness.py'],[
        'training_refits','stability_summary','training_refit_ap_differences'])}


def stage_signature(stage):
    specific,outputs=STAGES[stage]
    modules=['profile_departure.py','reduction.py','family_decomposition.py','cohort.py','common.py','signals_common.py']
    paths=[ROOT/'src/injury_analysis'/name for name in sorted(set(modules+specific))]
    paths.append(ROOT/'src/evaluation.py')
    result={'source':source_signature(),'config':sha(CONFIG_PATH),
            'code':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'matching':sha(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv')}
    if stage in ['scores','stability']:result['trained_predictions']=sha(CACHE/'predictions.parquet')
    return result


def stage_paths(stage):
    paths=[OUT/'tables'/f'{name}.csv' for name in STAGES[stage][1]]
    if stage=='models':paths += [CACHE/'predictions.parquet']+sorted(CACHE.glob('models_*.joblib'))
    if stage=='scores':paths += [CACHE/'matched_score_alignment.parquet',CACHE/'score_balanced_membership.parquet']
    return paths


def record_completed_stage(stage):
    """Record only after the named computations successfully produced their outputs."""
    paths=stage_paths(stage)
    if any(not p.exists() for p in paths):raise ValueError(f'Incomplete {stage} artifacts')
    result={'signature':stage_signature(stage),
            'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    (CACHE/f'{stage}_execution.json').write_text(json.dumps(result,indent=2))
    return result


def ensure_stage(stage,a,events,spec,matches,force=False):
    path=CACHE/f'{stage}_execution.json'
    if path.exists() and not force:
        previous=json.loads(path.read_text())
        intact=all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in previous['artifacts'].items())
        if previous['signature']==stage_signature(stage) and intact:
            print(f'Using verified {stage} results ({len(previous["artifacts"])} artifacts).',flush=True)
            return
    before=source_signature()
    if stage=='models':
        from .profile_departure import run_models
        run_models(a,events,spec)
    elif stage=='scores':
        from .profile_scores import run_score_analysis,run_alert_diagnostics
        run_score_analysis(a,events,matches);run_alert_diagnostics()
    elif stage=='sensitivity':
        from .profile_robustness import run_sensitivity,run_adjusted_sensitivity
        run_sensitivity(a,events);run_adjusted_sensitivity(a,events,spec)
    elif stage=='stability':
        from .profile_robustness import run_stability
        run_stability(a)
    else:raise ValueError(stage)
    assert source_signature()==before,'Source changed during analysis'
    record_completed_stage(stage)

