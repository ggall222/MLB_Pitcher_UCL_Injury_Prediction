"""Published-design invariants, without importing later UCL safeguards."""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.injury_analysis.kang_replication_data import config, PREDICTORS, season_screen, build_cohorts
from src.injury_analysis.kang_spec import ROOT


def outings():
    rows=[]
    for year in [2018,2019]:
        for pid in [1,2,3,4]:
            for i,day in enumerate(['04-01','04-05','04-10','04-15','04-20','04-25']):
                rows.append(dict({f:1. for f in PREDICTORS},pitcher_id=pid,game_id=year*100+pid*10+i,
                    season=year,game_date=pd.Timestamp(f'{year}-{day}'),is_starter=True,
                    outs_recorded=15,position='SS' if pid==4 else 'P',kang_release_y_avg=np.nan))
    return pd.DataFrame(rows)


def events():
    return pd.DataFrame([
        dict(player_id=1,transaction_date=pd.Timestamp('2018-04-16'),injury_region='Elbow / UCL',
             placement_transaction_id=1,injury_event_id='a',overlapping_episode=True,timing_uncertain=True),
        dict(player_id=1,transaction_date=pd.Timestamp('2019-04-15'),injury_region='Elbow / UCL',
             placement_transaction_id=2,injury_event_id='b',overlapping_episode=True,timing_uncertain=True),
        dict(player_id=1,transaction_date=pd.Timestamp('2019-04-23'),injury_region='Shoulder',
             placement_transaction_id=3,injury_event_id='c',overlapping_episode=False,timing_uncertain=False),
        dict(player_id=2,transaction_date=pd.Timestamp('2019-04-15'),injury_region='Hamstring',
             placement_transaction_id=4,injury_event_id='d',overlapping_episode=False,timing_uncertain=False),
        dict(player_id=3,transaction_date=pd.Timestamp('2019-04-15'),injury_region='Unknown / Unspecified',
             placement_transaction_id=5,injury_event_id='e',overlapping_episode=False,timing_uncertain=False)])


def test_appendix_inputs_no_project_innovations_or_unlisted_release_y():
    assert len(PREDICTORS)==22
    assert 'kang_release_y_avg' not in PREDICTORS
    assert 'extension_avg' in PREDICTORS
    assert not any(any(t in f for t in ['baseline','ucl','slope','rolling','zscore','arm_angle']) for f in PREDICTORS)


def test_paper_eligibility_does_not_add_position_or_history_filter():
    s=season_screen(outings(),config())
    assert s.eligible.all()  # missing unused Y does not change Appendix A1 gate
    assert 4 in set(s.pitcher_id)  # recorded pitching, not current primary position
    a=outings();a.loc[(a.pitcher_id==1)&(a.season==2019),'spin_avg']=np.nan
    assert not season_screen(a,config()).query('pitcher_id==1 and season==2019').eligible.any()


def test_no_extra_episode_qa_filter_and_first_combined_endpoint():
    c,_,first,_=build_cohorts(outings(),events(),config())
    elbow=c['elbow_starter']
    assert {'a','b'}==set(elbow.injury_event_id.dropna())
    assert 'c' not in set(first.injury_event_id)  # no second shoulder event promoted
    assert (elbow.loc[elbow.case_season,'lead_days']>0).all()
    assert pd.Timestamp('2019-04-15') not in set(elbow.query('pitcher_id==1').game_date)


def test_methods_controls_allow_other_and_unknown_injuries_and_counts_reveal_it():
    c,_,_,counts=build_cohorts(outings(),events(),config())
    f=c['elbow_starter']
    assert {2,3,4}<=set(f.loc[~f.case_season,'pitcher_id'])
    rows=counts.query("endpoint=='elbow' and role=='starter'").set_index('control_definition')
    assert rows.loc['no_shoulder_elbow_il','games']>rows.loc['no_any_il','games']
    assert rows.loc['no_shoulder_elbow_il','unknown_diagnosis_control_seasons']==1


def test_prior_injury_is_body_region_previous_season_not_current_season_or_ucl():
    c,*_=build_cohorts(outings(),events(),config())
    f=c['elbow_starter'].query('pitcher_id==1')
    assert f.loc[f.season.eq(2018),'prior_region_il'].eq(0).all()
    assert f.loc[f.season.eq(2019),'prior_region_il'].eq(1).all()


def test_replication_study_dates_are_fixed(tmp_path):
    import yaml
    cfg=config();cfg['end_date']='2026-12-31';p=tmp_path/'cfg.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='2017'):
        config(p)


@pytest.mark.skipif(shutil.which('Rscript') is None,reason='R runtime not installed')
def test_r_auc_stratification_pruning_slope_band_and_checkpoint(tmp_path):
    # Source the actual production functions, without launching the full experiment.
    script=tmp_path/'invariants.R'
    script.write_text('''source("scripts/kang_replication.R")
stopifnot(abs(auc(c(1,0,1,0),c(.8,.8,.3,.1))-.625)<1e-12)
y<-rep(c(0,0,0,1),25);z<-folds(y,5,1)
stopifnot(all(table(z,y)[,2]==5),identical(z,folds(y,5,1)))
a<-stratified_subset(y,.67,1);b<-setdiff(seq_along(y),a)
stopifnot(length(intersect(a,b))==0,length(c(a,b))==length(y))
cfg<-list(minimum_selection_frequency=.5,minimum_mean_gain=0,correlation_threshold=.7)
x<-cbind(a=1:20,b=2*(1:20),c=rep(c(1,2),10))
g<-matrix(rep(c(.2,.7,.1),5),ncol=3,byrow=TRUE,dimnames=list(NULL,colnames(x)))
p<-prune_features(x,g,cfg,list(a='v',b='v',c='w'))
stopifnot('b'%in%p$features,!('a'%in%p$features),'c'%in%p$features)
cfg<-list(window_min=7,window_max=49,gam=list(basis='tp',k=10,method='GCV.Cp',
 derivative_step=1e-4,grid_step=.01,low_fraction=.25,high_fraction=.35))
d<-data.frame(window=7:49);d$roc_auc<-.55+.25*(1-exp(-(d$window-7)/12))
w<-window_band(d,cfg)
stopifnot(is.finite(w$band$selected),w$band$lower<=w$band$upper)
q<-w$curve[w$curve$in_band,]
stopifnot(all(q$slope>q$low_slope),all(q$slope<q$high_slope))
d$roc_auc<-.9-.005*(d$window-7);w<-window_band(d,cfg)
stopifnot(is.na(w$band$selected))
path<-tempfile();checkpoint(path,'a',function() 3)
stopifnot(checkpoint(path,'a',function() stop('must reuse'))==3)
writeLines('corrupt',path);stopifnot(checkpoint(path,'a',function() 4)==4)
cfg<-list(xgboost=list(objective='binary:logistic',max_depth=2,subsample=.7,nthread=1,tree_method='hist'),nrounds=5)
x<-cbind(a=seq_len(100),b=sin(seq_len(100)));y<-as.integer(seq_len(100)%%3==0)
m1<-fit_model(x,y,cfg,14);runif(50);m2<-fit_model(x,y,cfg,14)
stopifnot(identical(predict(m1,x),predict(m2,x)))
cat('All R invariants passed\\n')
''')
    result=subprocess.run(['Rscript','--vanilla',str(script)],cwd=ROOT,text=True,capture_output=True,
                          env=dict(os.environ,LC_ALL='C',LANG='C'))
    assert result.returncode==0,result.stdout+result.stderr
