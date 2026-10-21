"""Validate completed consolidation artifacts, temporal boundaries and preservation."""
from pathlib import Path
import datetime as dt
import json
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.injury_analysis.common import source_signature,sha
from src.injury_analysis.consolidation_common import CACHE,OUT,CFG,inputs,read_table,table
from src.injury_analysis.consolidation_models import scores,feature_sets
from src.injury_analysis.consolidation_selection import choose_candidate
from src.injury_analysis.family_decomposition import paired_bootstrap_ap
from src.injury_analysis.consolidation_plots import FIGURES


def main():
    checks=[]
    def check(name,passed,detail):
        checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
        print(('PASS ' if passed else 'FAIL ')+name,flush=True)
    prior=json.loads((OUT/'prior_artifacts_snapshot.json').read_text())
    resumed=json.loads((OUT/'resume/resume_snapshot.json').read_text())
    check('Database and source signature unchanged',source_signature()==prior['source']==resumed['source'],source_signature())
    for name,snapshot in [('Preceding analyses preserved',prior),('Interrupted-run checkpoints preserved',resumed)]:
        changed=[n for n,digest in snapshot['artifacts'].items() if not (ROOT/n).exists() or sha(ROOT/n)!=digest]
        check(name,not changed,f'{len(snapshot["artifacts"])} fingerprints; changed={changed}')
    inv=read_table('reconciled_feature_inventory');old=read_table('feature_inventory')
    registered=json.loads((ROOT/'data/exports/feature_columns.json').read_text())
    check('Complete schema-backed inventory',set(inv.feature)==set(registered+['prior_reported_il_count','calendar_month']) and not inv.feature.duplicated().any(),len(inv))
    check('Metadata corrections do not alter selection inputs',
        inv[['feature','feature_group','eligible_predictor','training_missing_fraction']].equals(old[['feature','feature_group','eligible_predictor','training_missing_fraction']]),'Only descriptive source/timing fields reconciled')
    reviewed=read_table('feature_inventory_review').set_index('feature')
    type_counts=[n for n in reviewed.index if n.endswith('_pitch_count') and reviewed.loc[n,'pitch_type_specific']]
    check('Pitch-type count provenance differs from official count',
        all('physical Statcast pitch rows' in reviewed.loc[n,'source_variables'] and 'pitch_type =' in reviewed.loc[n,'source_variables'] for n in type_counts)
        and 'official boxscore' in reviewed.loc['pitch_count','source_variables'],len(type_counts))
    core,reps,support=feature_sets()
    check('Separate substantive and support pools',len(core)==333 and len(support)==20 and not set(core)&set(support),f'{len(core)} core; {len(support)} support; {len(reps)} correlation representatives')
    check('No outcome or unverified prior surgery predictors',not any('next_' in n or 'documented_prior_ucl' in n or 'surgery_date' in n for n in core),'Registered predictor allowlist plus existing cohort context')
    f,events,_=inputs()
    check('Unique appearance grain',not f[['game_id','pitcher_id']].duplicated().any(),len(f))
    check('Chronological partitions',f.loc[f.partition.eq('train'),'game_date'].max()<pd.Timestamp('2022-01-01') and
          f.loc[f.partition.eq('validation'),'game_date'].between('2022-01-01','2023-12-31').all() and
          f.loc[f.partition.eq('test'),'game_date'].between('2024-01-01','2025-12-31').all(),'Original cohort rows and 60-day embargo reused')
    for name in ['history_max_date','baseline_max_date','workload_max_date']:
        t=pd.to_datetime(f[name]);check('Strictly prior '+name,(t.isna()|t.lt(f.game_date)).all(),'No current-day/future history')
    keys=read_table('correlation_sample_keys')
    check('Correlations use training-only capped sample',set(keys.row_id)<=set(f.loc[f.partition.eq('train'),'row_id']) and keys.groupby('pitcher_id').size().le(12).all(),len(keys))
    for name in ['pearson','spearman','coobserved_counts']:
        c=pd.read_parquet(CACHE/(name+'.parquet'))
        check(name+' matrix integrity',c.shape==(353,353) and np.allclose(c,c.T,equal_nan=True),c.shape)
    canonical=read_table('cross_group_summary_canonical');pairs=pd.read_parquet(CACHE/'all_correlation_pairs.parquet')
    check('Canonical pair aggregation preserves all correlations',canonical.pairs.sum()==len(pairs) and canonical.high_pairs.sum()==pairs.high_correlation.sum(),len(pairs))
    sparse=read_table('elastic_net_tuning')
    check('Complete Elastic Net grid',set(zip(sparse.C,sparse.l1_ratio))=={(c,m) for c in CFG['elastic_net_C'] for m in CFG['elastic_net_l1_ratio']},len(sparse))
    raw=read_table('elastic_net_refit_coefficients');imp=read_table('feature_refit_permutation');perf=read_table('training_refit_performance')
    check('Sixteen complete whole-pitcher refits',set(raw.repeat)==set(range(16)) and raw.groupby('repeat').feature.nunique().eq(333).all() and len(raw)==5328 and len(imp)==5328 and len(perf)==16,len(raw))
    models=list((CACHE/'models').glob('*.joblib'));bad=[]
    for path in models:
        obj=joblib.load(path);enc=obj['encoder'];model=obj['model']
        if not (np.isfinite(model.coef_).all() and model.coef_.shape[1]==len(enc.names) and pd.Timestamp(enc.fit_max_date)<pd.Timestamp('2022-01-01')):
            bad.append(path.name)
        if obj['elastic'] is not None and model.n_iter_.max()>=CFG['elastic_net_max_iter']:
            bad.append(path.name+': iteration limit')
    check('All fitted models finite, converged and train-only',not bad,f'{len(models)} models; failures={bad}')
    valid=f.loc[f.partition.eq('validation')];pred=pd.read_parquet(CACHE/'candidate_validation_predictions.parquet')
    assert valid.row_id.tolist()==pred.row_id.tolist()
    scoremap={c.removeprefix('score__'):pred[c].to_numpy() for c in pred if c.startswith('score__')}
    boots,_=paired_bootstrap_ap(valid,scoremap)
    chosen,decisions=choose_candidate(read_table('candidate_reduction_curve'),boots)
    selection=json.loads((CACHE/'candidate_selection.json').read_text())
    check('Candidate decision reproduced without test data',chosen==selection['selected'],chosen)
    freeze=json.loads((OUT/'candidate_freeze.json').read_text())
    check('Frozen feature list and model unchanged',sha(ROOT/freeze['candidate_file'])==freeze['candidate_sha256'] and
          sha(CACHE/'models'/(selection['model_name']+'.joblib'))==freeze['model_sha256'],selection['features'])
    changed=[n for n,d in freeze['decision_tables'].items() if sha(ROOT/n)!=d]
    check('Frozen decision tables unchanged',not changed,changed)
    created=dt.datetime.fromisoformat(freeze['created_at']).timestamp()
    check('Freeze predates new test-score artifact',created<=(CACHE/'frozen_predictions.parquet').stat().st_mtime,'Previously examined test period; not independent confirmation')
    predictions=pd.read_parquet(CACHE/'frozen_predictions.parquet');specs=json.loads((CACHE/'candidate_specifications.json').read_text())
    for part,g in predictions.groupby('partition'):
        source=f.set_index('row_id').loc[g.row_id]
        obj=joblib.load(CACHE/'models'/(selection['model_name']+'.joblib'))
        check('Reproduced frozen '+part+' scores',np.allclose(scores(obj,source),g['score__'+chosen],atol=1e-12),'Training transform only')
    tables=list((OUT/'tables').glob('*.csv'))
    for path in tables:pd.read_csv(path)
    check('All CSV outputs readable',True,len(tables))
    figures=[OUT/'figures'/(name+'.'+ext) for name,_ in FIGURES for ext in ['png','pdf']]
    check('Six PNG/PDF figures',all(p.exists() and p.stat().st_size>1000 for p in figures),len(figures))
    notebooks=list((ROOT/'notebooks/feature_consolidation').glob('*.ipynb'));issues=[];cells=0
    for path in notebooks:
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        for c in book.cells:
            if c.cell_type=='code':
                cells+=1
                if c.execution_count is None or any(o.output_type=='error' for o in c.outputs):issues.append(path.name)
    check('Three executed notebooks without errors',len(notebooks)==3 and not issues,f'{len(notebooks)} notebooks; {cells} executed code cells')
    tests=(OUT/'resume/pytest.log').read_text()
    check('Full repository test suite passes','165 passed' in tests and ' failed' not in tests,'165 passed (147 preceding + 18 new)')
    partial=list(CACHE.rglob('*.partial'))+list(OUT.rglob('*.partial'))
    check('No unfinished partial outputs',not partial,partial)
    table(pd.DataFrame(checks),'validation_checks')
    result=dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),checks=len(checks),
        passed=sum(r['passed'] for r in checks),models=len(models),tables=len(tables),figures=len(figures),
        notebooks=len(notebooks),tests=165,source_unchanged=source_signature()==prior['source'])
    (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2))
    assert all(r['passed'] for r in checks),'See validation_checks.csv'
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
