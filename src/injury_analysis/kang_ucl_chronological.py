"""Train, freeze, then evaluate the same model on two later calendar periods."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import subprocess

import pandas as pd

from .kang_spec import ROOT, DOMAINS, atomic_json, digest
from .kang_replication_data import PREDICTORS
from .kang_ucl_endpoint_data import check_hashes
from .kang_ucl_chronological_data import settings, freeze_protocol, load_period

DRIVER=ROOT/'scripts/kang_ucl_chronological.R'


def training_signature(training_path,cfg,software):
    # Explicitly independent of validation/test files, labels and fitted Ext1 outputs.
    return hashlib.sha256((digest(training_path)+digest(DRIVER)+digest(ROOT/'scripts/kang_replication.R')+
        json.dumps(cfg,sort_keys=True)+software).encode()).hexdigest()


def train_role(role,cfg,software):
    d=ROOT/cfg['cache']/role;sig=training_signature(d/'training.csv',cfg,software)
    run_cfg=dict(cfg,signature=sig,predictors=PREDICTORS,domains={c:DOMAINS[c] for c in PREDICTORS})
    atomic_json(d/'run_config.json',run_cfg)
    if (d/'complete.json').exists():
        done=json.loads((d/'complete.json').read_text())
        if done['signature']==sig and done.get('hashes'):
            check_hashes(done['hashes'],d);print('Verified training checkpoint:',role,flush=True);return
    print('Fitting training-only role:',role,flush=True)
    with (d/'training.log').open('w') as log:
        subprocess.run([cfg['rscript'],'--vanilla',str(DRIVER),'train',str(d/'training.csv'),str(d/'run_config.json'),str(d)],
            cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,LC_ALL='C',LANG='C'))
    done=json.loads((d/'complete.json').read_text())
    names=['training.csv','training_support.csv','window_cv.csv','gam_curve.csv','window_selection.json',
           'feature_selection.csv','model.bin','training_freeze.json','importance.csv']
    done['hashes']={name:digest(d/name) for name in names if (d/name).exists()}
    atomic_json(d/'complete.json',done)


def development_freeze(spec,cfg,protocol):
    out=ROOT/cfg['output'];cache=ROOT/cfg['cache'];files={};roles={}
    for role in ['starter','reliever']:
        done=json.loads((cache/role/'complete.json').read_text());check_hashes(done['hashes'],cache/role)
        roles[role]=dict(status=done['status'],signature=done['signature'])
        files.update({str((cache/role/n).relative_to(ROOT)):h for n,h in done['hashes'].items()})
        if done['status']=='complete':
            f=json.loads((cache/role/'training_freeze.json').read_text())
            assert f['training_max_date']<='2023-12-31'
            roles[role].update(selected_window=f['selected_window'],features=f['features'])
    p=out/'development_freeze.json'
    if p.exists():
        saved=json.loads(p.read_text());check_hashes(saved['training_hashes'])
        assert saved['roles']==roles;return saved
    value=dict(frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=digest(out/'protocol_freeze.json'),
        training_hashes=files,roles=roles,training_end=spec['training_end'],
        same_frozen_model_for_validation_and_test=True,no_later_outcomes_used=True,
        interpretation_policy=spec['interpretation_policy'])
    atomic_json(p,value);return value


def score(period,cfg,frozen,cutoff):
    cache=ROOT/cfg['cache'];out=ROOT/cfg['output']
    check_hashes(frozen['training_hashes'])
    access=out/(period+'_access.json')
    if not access.exists():atomic_json(access,dict(first_access_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        development_freeze_sha256=digest(out/'development_freeze.json')))
    for role,info in frozen['roles'].items():
        if info['status']!='complete':continue
        d=cache/role
        sig=hashlib.sha256((info['signature']+digest(d/(period+'.csv'))+cutoff+period).encode()).hexdigest()
        marker=d/(period+'_scored.json')
        if marker.exists():
            m=json.loads(marker.read_text())
            if m['signature']==sig:check_hashes(m['hashes'],d);continue
        with (d/(period+'_score.log')).open('w') as log:
            subprocess.run([cfg['rscript'],'--vanilla',str(DRIVER),'score',str(d/(period+'.csv')),
                str(d/'run_config.json'),str(d),period,cutoff],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,LC_ALL='C',LANG='C'))
        atomic_json(marker,dict(signature=sig,hashes={n:digest(d/n) for n in ['predictions_'+period+'.csv','metrics_'+period+'.csv']}))
    check_hashes(frozen['training_hashes'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train-only',action='store_true');args=parser.parse_args(argv)
    spec,endpoint,base,ext1,cfg=settings();protocol=freeze_protocol(spec,endpoint,base,ext1,cfg)
    software=subprocess.check_output([cfg['rscript'],'--vanilla','-e',
        'cat(R.version.string,"\\n");for(p in c("xgboost","mgcv","jsonlite"))cat(p,as.character(packageVersion(p)),"\\n")'],
        env=dict(os.environ,LC_ALL='C',LANG='C'),text=True)
    assert software==(ROOT/base['output']/'software.txt').read_text()
    (ROOT/cfg['output']/'software.txt').write_text(software)
    training=load_period('training',spec,endpoint,base,ext1,cfg)
    print(pd.read_csv(ROOT/cfg['output']/'tables/cohort_training.csv').to_string(index=False),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(train_role,role,cfg,software) for role in ['starter','reliever']]
        for task in as_completed(tasks):task.result()
    frozen=development_freeze(spec,cfg,protocol)
    score('training',cfg,frozen,spec['training_end'])
    print('Training models and evaluation policy frozen.',flush=True)
    if args.train_only:return
    for period in ['validation','test']:
        # Access record precedes loading new-period values; both models already frozen.
        p=ROOT/cfg['output']/(period+'_access.json')
        if not p.exists():atomic_json(p,dict(first_access_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            development_freeze_sha256=digest(ROOT/cfg['output']/'development_freeze.json')))
        load_period(period,spec,endpoint,base,ext1,cfg)
        score(period,cfg,frozen,protocol['source_cutoff'])
    from .kang_ucl_chronological_reporting import report,validate
    report(spec,base,ext1,cfg,frozen,protocol)
    validate(spec,base,ext1,cfg,frozen,protocol)


if __name__=='__main__':main()
