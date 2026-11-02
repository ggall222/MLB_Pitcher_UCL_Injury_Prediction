"""Extension 1: run frozen Kang R code on the established UCL endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess

import pandas as pd

from .kang_spec import ROOT, atomic_json, digest
from .kang_replication import execute_task
from .kang_replication_data import PREDICTORS
from .kang_ucl_endpoint_data import settings, freeze, prepare, check_hashes


def equivalence(spec, base, cfg, saved):
    """Fail before fitting if any experimental setting differs from the parent."""
    records=[]
    for k in base:
        if k in ['cache','output']:continue
        assert cfg[k]==base[k],k
        records.append(dict(component=k,unchanged=True,evidence='Identical inherited configuration value'))
    for component in ['split_algorithm','XGBoost_fit','inverse_frequency_weights','CV_and_feature_selection',
                      'correlation_pruning','GAM_window_selection','ROC_AUC','cluster_bootstrap','gain_importance']:
        records.append(dict(component=component,unchanged=True,evidence='Identical scripts/kang_replication.R, reused without edits'))
    assert saved['predictors']==PREDICTORS
    assert digest(ROOT/cfg['output']/'tables/feature_crosswalk.csv')==digest(ROOT/base['output']/'tables/feature_crosswalk.csv')
    records.append(dict(component='predictors_and_crosswalk',unchanged=True,evidence='22 original inputs; identical crosswalk hash'))
    check_hashes(saved['benchmark_hashes']);check_hashes(saved['endpoint_source_hashes'])
    pd.DataFrame(records).to_csv(ROOT/cfg['output']/'tables/benchmark_equivalence.csv',index=False)


def preflight(cfg,rscript):
    """Use the actual R partition function; do not impose a new split/seed."""
    cache=ROOT/cfg['cache']
    script=cache/'preflight.R'
    script.write_text('''source("scripts/kang_replication.R")
args<-commandArgs(trailingOnly=TRUE); cfg<-fromJSON(args[1]); result<-list()
for(role in c('starter','reliever')) {
 d<-read.csv(file.path(args[2],paste0('ucl_',role),'cohort.csv'))
 case<-tolower(as.character(d$case_season))=='true'
 label<-function(w) as.integer(case & !is.na(d$lead_days) & d$lead_days>=1 & d$lead_days<=w)
 tr<-stratified_subset(label(cfg$window_max),1-cfg$test_fraction,cfg$seed)
 te<-setdiff(seq_len(nrow(d)),tr)
 for(w in seq(cfg$window_min,cfg$window_max,by=cfg$window_step)) {
  y<-label(w)
  result[[length(result)+1]]<-data.frame(role=role,window=w,train_games=length(tr),test_games=length(te),
   train_positive_games=sum(y[tr]),test_positive_games=sum(y[te]),
   train_events=length(unique(na.omit(d$injury_event_id[tr][y[tr]==1]))),
   test_events=length(unique(na.omit(d$injury_event_id[te][y[te]==1]))),
   supported=sum(y[tr])>=cfg$cv_folds && sum(1-y[tr])>=cfg$cv_folds && length(unique(y[te]))==2)
 }
}
write.csv(do.call(rbind,result),args[3],row.names=FALSE)
''')
    atomic_json(cache/'effective_model_config.json',cfg)
    output=ROOT/cfg['output']/'tables/preflight_support.csv'
    subprocess.run([rscript,'--vanilla',str(script),str(cache/'effective_model_config.json'),str(cache),str(output)],
        cwd=ROOT,env=dict(os.environ,LC_ALL='C',LANG='C'),check=True,capture_output=True,text=True)
    support=pd.read_csv(output)
    return {role:bool(g.supported.all()) for role,g in support.groupby('role')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    spec,base,cfg=settings(args.config);saved=freeze(spec,base,cfg)
    cohorts,benchmark=prepare(spec,base,cfg)
    equivalence(spec,base,cfg,saved)
    rscript=shutil.which(cfg['rscript'])
    if not rscript:raise RuntimeError('Rscript required; reuse the benchmark R environment.')
    supported=preflight(cfg,rscript)
    print(pd.read_csv(ROOT/cfg['output']/'tables/cohort_comparison.csv').to_string(index=False),flush=True)
    print('Technical split/CV support:',supported,flush=True)
    if args.prepare_only:return
    software=subprocess.check_output([rscript,'--vanilla','-e',
        'cat(R.version.string,"\\n");for(p in c("xgboost","mgcv","jsonlite"))cat(p,as.character(packageVersion(p)),"\\n")'],
        env=dict(os.environ,LC_ALL='C',LANG='C'),text=True)
    assert software==(ROOT/base['output']/'software.txt').read_text(),'Different benchmark R/package versions'
    (ROOT/cfg['output']/'software.txt').write_text(software)
    with ThreadPoolExecutor(max_workers=min(2,cfg['workers'])) as pool:
        futures=[pool.submit(execute_task,'ucl_'+r,cfg,rscript,software) for r in cohorts if supported[r]]
        for f in as_completed(futures):f.result()
    from .kang_ucl_endpoint_reporting import report, validate
    report(spec,base,cfg,cohorts,benchmark,supported)
    validate(spec,base,cfg,saved,supported)


if __name__=='__main__':main()
