"""Validate saved replication artifacts without changing the published design."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .kang_spec import ROOT, digest
from .kang_replication_data import PREDICTORS


def validate(cfg):
    cache=ROOT/cfg['cache']; records=[]
    for task in ['elbow_starter','elbow_reliever','shoulder_starter','shoulder_reliever']:
        d=cache/task; cohort=pd.read_csv(d/'cohort.csv');split=pd.read_csv(d/'split.csv')
        complete=json.loads((d/'complete.json').read_text())
        for name,expected in complete['hashes'].items():
            assert digest(d/name)==expected,(task,'artifact changed',name)
        assert not split.duplicated(['pitcher_id','game_id']).any()
        assert len(split)==len(cohort)
        joined=cohort.merge(split[['pitcher_id','game_id','partition']],on=['pitcher_id','game_id'],validate='one_to_one')
        assert joined.season.between(2017,2025).all()
        train=joined[joined.partition.eq('train')]
        held=joined[joined.partition.eq('test')]
        cv=pd.read_csv(d/'window_cv.csv')
        expected_windows=set(range(cfg['window_min'],cfg['window_max']+1,cfg['window_step']))
        assert set(cv.window)==expected_windows
        assert cv.groupby('window').size().eq(cfg['cv_folds']).all()
        for w,g in cv.groupby('window'):
            assert g.games.sum()==len(train)
            assert g.positive_games.sum()==(train.case_season & train.lead_days.between(1,w)).sum()
        if complete['status']=='complete':
            freeze=json.loads((d/'training_freeze.json').read_text())
            selected=set(freeze['features'])
            assert selected<=set(PREDICTORS)
            assert complete['selection_fits']==250
            assert complete['importance_refits']==500
            pred=pd.read_csv(d/'predictions.csv')
            compare=held.merge(pred,on=['pitcher_id','game_id','season','game_date','injury_event_id'],validate='one_to_one')
            assert len(compare)==len(held)==len(pred)
            assert compare.target.eq((compare.case_season & compare.lead_days.between(1,freeze['selected_window'])).astype(int)).all()
            assert np.isfinite(pred.prediction).all() and pred.prediction.between(0,1).all()
            m=pd.read_csv(d/'metrics.csv').iloc[0]
            assert abs(roc_auc_score(pred.target,pred.prediction)-m.roc_auc)<1e-10
            importance=pd.read_csv(d/'importance.csv')
            assert set(importance.feature)==selected
            assert importance.valid_refits.eq(500).all()
            assert abs(importance.mean_gain.sum()-1)<1e-5
            records.append(dict(task=task,status='passed',training_games=len(train),test_games=len(held),
                candidate_windows=len(expected_windows),selection_fits=250,importance_refits=500,
                independently_verified_auc=float(m.roc_auc),shared_pitchers=len(set(train.pitcher_id)&set(held.pitcher_id)),
                study_period_valid=True,artifact_hashes_valid=True))
        else:
            records.append(dict(task=task,status='no_qualifying_window',candidate_windows=len(expected_windows)))
    result=pd.DataFrame(records)
    result.to_csv(ROOT/cfg['output']/'tables/validation_checks.csv',index=False)
    return result
