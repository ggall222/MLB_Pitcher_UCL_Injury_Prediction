"""Chronological separation, administrative follow-up and frozen algorithms."""
import json
import os
import subprocess

import pandas as pd
import pytest

from src.injury_analysis.kang_spec import ROOT
from src.injury_analysis.kang_ucl_endpoint_data import check_hashes
from src.injury_analysis.kang_ucl_chronological_data import settings,period_for,observed_labels,assert_separation
from src.injury_analysis.kang_ucl_chronological import training_signature


def test_fixed_calendar_partitions_and_identical_parameters():
    spec,_,base,_,cfg=settings()
    dates=pd.Series(pd.to_datetime(['2016-12-31','2017-01-01','2023-12-31','2024-01-01','2025-12-31','2026-01-01','2026-12-31']))
    assert period_for(dates,spec).tolist()==['excluded','training','training','validation','validation','test','test']
    assert {k:v for k,v in base.items() if k not in ['cache','output']}=={k:v for k,v in cfg.items() if k not in ['cache','output']}


def test_incomplete_negative_followup_is_missing_and_observed_positive_retained():
    f=pd.DataFrame(dict(game_date=pd.to_datetime(['2026-09-01','2026-09-27','2026-09-27','2026-10-18']),
                        case_season=[False,False,True,True],lead_days=[float('nan'),float('nan'),3,10]))
    y=observed_labels(f,30,'2026-10-20')
    assert y.iloc[0]==0 and pd.isna(y.iloc[1]) and y.iloc[2]==1 and pd.isna(y.iloc[3])
    assert observed_labels(f.iloc[[1]],23,'2026-10-20').iloc[0]==0


def test_same_pitcher_across_seasons_allowed_but_shared_events_rejected():
    a=pd.DataFrame(dict(pitcher_id=[1],game_id=[10],season=[2023],case_season=[True],lead_days=[5],injury_event_id=['a']))
    b=a.assign(game_id=11,season=2024,injury_event_id='b')
    assert_separation({'training':a,'validation':b})
    with pytest.raises(AssertionError,match='injury-event'):assert_separation({'training':a,'validation':b.assign(injury_event_id='a')})
    with pytest.raises(AssertionError,match='pitcher-season'):assert_separation({'training':a,'validation':b.assign(season=2023)})
    with pytest.raises(AssertionError,match='game overlap'):assert_separation({'training':a,'validation':b.assign(game_id=10)})
    with pytest.raises(AssertionError,match='game_id overlap'):assert_separation({'training':a,'validation':b.assign(game_id=10,pitcher_id=2)})


def test_future_file_mutation_does_not_change_training_signature(tmp_path):
    cfg=settings()[-1];train=tmp_path/'training.csv';later=tmp_path/'test.csv'
    train.write_text('training-only fixture');later.write_text('future 0')
    before=training_signature(train,cfg,'fixed software')
    later.write_text('future outcome 1 and different predictor values')
    assert training_signature(train,cfg,'fixed software')==before
    train.write_text('different training data')
    assert training_signature(train,cfg,'fixed software')!=before


def test_r_reuses_identical_functions_and_rejects_future_training_rows(tmp_path):
    script=tmp_path/'check.R'
    script.write_text('''source('scripts/kang_replication.R')
names<-c('fit_model','folds','stratified_subset','prune_features','window_band','bootstrap_auc','gain','auc')
before<-lapply(names,function(n)body(get(n)))
source('scripts/kang_ucl_chronological.R')
stopifnot(identical(before,lapply(names,function(n)body(get(n)))))
p<-tempfile();write.csv(data.frame(game_date='2024-01-01',game_id=1,pitcher_id=1),p,row.names=FALSE)
c<-tempfile();write_json(list(signature='unit'),c,auto_unbox=TRUE)
error<-tryCatch({train_chronological(p,c,tempdir());FALSE},error=function(e)TRUE)
stopifnot(error)
cat('Identical parent methods; future training rejected\\n')
''')
    p=subprocess.run(['Rscript','--vanilla',str(script)],cwd=ROOT,env=dict(os.environ,LC_ALL='C',LANG='C'),capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr


def test_prior_stages_and_completed_training_frozen():
    *_,cfg=settings();out=ROOT/cfg['output']
    if not (out/'protocol_freeze.json').exists():pytest.skip('Experiment not prepared')
    protocol=json.loads((out/'protocol_freeze.json').read_text());check_hashes(protocol['prior_hashes']);check_hashes(protocol['source_hashes'])
    if (out/'development_freeze.json').exists():
        f=json.loads((out/'development_freeze.json').read_text());check_hashes(f['training_hashes'])
        assert f['same_frozen_model_for_validation_and_test']


def test_actual_later_scores_have_independent_auc_and_labels():
    from sklearn.metrics import roc_auc_score
    spec,_,_,_,cfg=settings();cache=ROOT/cfg['cache'];found=False
    for role in ['starter','reliever']:
        for period in ['validation','test']:
            p=cache/role/('predictions_'+period+'.csv')
            if not p.exists():continue
            found=True;pred=pd.read_csv(p);m=pd.read_csv(p.parent/('metrics_'+period+'.csv')).iloc[0]
            known=pred[pred.target.notna()]
            if known.target.nunique()==2:assert abs(roc_auc_score(known.target,known.prediction)-m.roc_auc)<1e-10
            f=pd.read_csv(p.parent/(period+'.csv'),low_memory=False)
            assert period_for(pd.to_datetime(f.game_date),spec).eq(period).all()
            pd.testing.assert_series_equal(pred.target.astype(float),observed_labels(f,int(m.window),spec['source_observation_end_expected']),check_names=False)
    if not found:pytest.skip('No later-period scores yet')
