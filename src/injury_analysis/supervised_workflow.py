"""Fingerprint-verified notebook stages for supervised compression."""
import json
from .supervised_reduction import ROOT,OUT,CACHE,CONFIG_PATH,source_signature,sha

STAGES={
 'prediction':(['supervised_reduction.py'],['selected_performance','validation_selection','component_curves',
     'alert_budgets','selected_differences','warning_leads','basis_loadings','full_span_control']),
 'interpretation':(['supervised_diagnostics.py','trajectory_shapes.py'],['component_loadings','component_family_weights',
     'prior_pca_score_correlations','component_persistence','component_trajectories','component_shape_contrasts']),
 'sensitivity':(['supervised_diagnostics.py'],['sensitivity_performance','sensitivity_differences',
     'sensitivity_selection','sensitivity_component_curves']),
 'stability':(['supervised_diagnostics.py'],['training_refits','refit_family_weights','stability_summary'])}


def signature(stage):
    modules=['supervised_reduction.py','profile_departure.py','reduction.py','family_decomposition.py',
             'cohort.py','common.py','signals_common.py']+STAGES[stage][0]
    paths=[ROOT/'src/injury_analysis'/n for n in sorted(set(modules))]+[ROOT/'src/evaluation.py']
    result={'source':source_signature(),'config':sha(CONFIG_PATH),
            'code':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'matching':sha(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv')}
    if stage!='prediction':result['trained_predictions']=sha(CACHE/'predictions.parquet')
    return result


def paths(stage):
    files=[OUT/'tables'/f'{name}.csv' for name in STAGES[stage][1]]
    if stage=='prediction':files+=[CACHE/'predictions.parquet']+sorted(CACHE.glob('model_*.joblib'))
    if stage=='interpretation':files+=[CACHE/'matched_component_alignment.parquet',CACHE/'balanced_component_membership.parquet']
    return files


def record_completed_stage(stage):
    artifacts=paths(stage)
    if any(not p.exists() for p in artifacts):raise ValueError('Missing '+stage+' artifacts')
    result={'signature':signature(stage),'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in artifacts}}
    (CACHE/f'{stage}_execution.json').write_text(json.dumps(result,indent=2))
    return result


def ensure_stage(stage,a,events,spec,matches,force=False):
    path=CACHE/f'{stage}_execution.json'
    if path.exists() and not force:
        record=json.loads(path.read_text())
        if record['signature']==signature(stage) and all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in record['artifacts'].items()):
            print(f'Using verified {stage} artifacts.',flush=True);return
    before=source_signature()
    if stage=='prediction':
        from .supervised_reduction import run_prediction
        run_prediction(a,events,spec)
    elif stage=='interpretation':
        from .supervised_diagnostics import run_interpretation
        run_interpretation(a,events,spec,matches)
    elif stage=='stability':
        from .supervised_diagnostics import run_stability
        run_stability(a,spec)
    elif stage=='sensitivity':
        from .supervised_diagnostics import run_sensitivity
        run_sensitivity(a,events,spec)
    else:raise ValueError(stage)
    assert source_signature()==before
    record_completed_stage(stage)

