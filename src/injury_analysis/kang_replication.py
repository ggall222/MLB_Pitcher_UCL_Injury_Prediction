"""Run the isolated published-method replication, with resumable R tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from .kang_replication_data import config, prepare, PREDICTORS
from .kang_spec import ROOT, DOMAINS, atomic_json, digest

PUBLISHED = {
    'elbow_starter': dict(events=146, seasons=1114, games=34285, pitchers=553, low=27.5, high=29.5, auc=.868, prior_auc=.872),
    'elbow_reliever': dict(events=284, seasons=2856, games=81975, pitchers=1101, low=19.7, high=21.3, auc=.815, prior_auc=.829),
    'shoulder_starter': dict(events=212, seasons=1180, games=36354, pitchers=563, low=22.8, high=25.7, auc=.815, prior_auc=.858),
    'shoulder_reliever': dict(events=333, seasons=2905, games=83897, pitchers=1109, low=18.6, high=20.3, auc=.797, prior_auc=.835),
}


def execute_task(task, cfg, rscript, software):
    directory = ROOT / cfg['cache'] / task
    source = directory / 'cohort.csv'
    signature = hashlib.sha256((digest(source) + digest(ROOT / 'scripts/kang_replication.R') +
        json.dumps(cfg, sort_keys=True) + software).encode()).hexdigest()
    run_cfg = dict(cfg, signature=signature, predictors=PREDICTORS, domains={f: DOMAINS[f] for f in PREDICTORS})
    atomic_json(directory / 'run_config.json', run_cfg)
    completion = directory / 'complete.json'
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved.get('signature') == signature and saved.get('hashes') and all(
                (directory / p).exists() and digest(directory / p) == h for p, h in saved['hashes'].items()):
            print('Verified completed replication: ' + task, flush=True)
            return task
    print('Starting/resuming R replication: ' + task, flush=True)
    with (directory / 'run.log').open('w') as log:
        subprocess.run([rscript, '--vanilla', str(ROOT / 'scripts/kang_replication.R'), str(source),
                        str(directory / 'run_config.json'), str(directory)], check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, LC_ALL='C', LANG='C'))
    saved = json.loads(completion.read_text())
    artifacts = [p for p in directory.iterdir() if p.suffix in ['.csv', '.bin', '.json', '.txt']
                 and p.name not in ['complete.json', 'run_config.json']]
    saved['hashes'] = {p.name: digest(p) for p in artifacts}
    atomic_json(completion, saved)
    print('Completed R replication: ' + task, flush=True)
    return task


def report(cfg, cohorts):
    out, cache = ROOT / cfg['output'], ROOT / cfg['cache']
    tables, figures = out / 'tables', out / 'figures'
    figures.mkdir(exist_ok=True)
    metrics, windows, gains, sensitivities, comparison = [], [], [], [], []
    for task, f in cohorts.items():
        d = cache / task
        w = json.loads((d / 'window_selection.json').read_text())
        slope_curve = pd.read_csv(d / 'gam_curve.csv')
        w['selected_day_in_band'] = (bool(slope_curve.iloc[(slope_curve.window-w['selected']).abs().argmin()].in_band)
                                     if w['selected'] is not None else False)
        windows.append(dict(task=task, **w))
        pub = PUBLISHED[task]
        for quantity, published, actual in [
            ('pitcher-games', pub['games'], len(f)), ('pitcher-seasons', pub['seasons'], len(f[['pitcher_id','season']].drop_duplicates())),
            ('unique pitchers within role', pub['pitchers'], f.pitcher_id.nunique()), ('injury events with pre-IL games', pub['events'], f.injury_event_id.nunique())]:
            comparison.append(dict(task=task, quantity=quantity, kang=published, replication=actual,
                explanation='Official-source IL coding and Methods control definition; no additional episode QA exclusions'))
        comparison.append(dict(task=task, quantity='slope-band window days', kang=f"{pub['low']}–{pub['high']}",
            replication=f"{w['lower']}–{w['upper']}", explanation='Daily candidate grid and mgcv defaults are disclosed approximations'))
        if (d / 'metrics.csv').exists() and json.loads((d / 'complete.json').read_text())['status'] == 'complete':
            m = pd.read_csv(d / 'metrics.csv'); m['task'] = task; m['published_auc'] = pub['auc']
            m['band_segments'] = w['segments']; m['selected_day_in_band'] = w['selected_day_in_band'];metrics.append(m)
            i = pd.read_csv(d / 'importance.csv'); i['task'] = task; i['domain'] = i.feature.map(DOMAINS); gains.append(i)
            sen = pd.read_csv(d / 'prior_injury_sensitivity.csv'); sen['task'] = task; sen['published_auc'] = pub['prior_auc']; sensitivities.append(sen)
            comparison.append(dict(task=task, quantity='test ROC-AUC', kang=pub['auc'],replication=m.roc_auc.iloc[0],
                explanation='Different source cohort, unknown original hyperparameters/selection rules, selected window'))
            published_features = {'elbow_starter':'release Z, absolute X, extension, slider fraction, velocity SD',
                'elbow_reliever':'release Z, absolute X, age, extension, velocity',
                'shoulder_starter':'extension, mean spin, maximum velocity, absolute X, age',
                'shoulder_reliever':'release Z, extension, absolute X, age, maximum velocity'}
            comparison.append(dict(task=task,quantity='top five mean total-Gain features',kang=published_features[task],
                replication=', '.join(i.head(5).feature),explanation='Training bootstrap Gain; correlated alternatives can exchange rank'))
    for endpoint, pub in [('elbow', (116260,3970,1654,430)), ('shoulder', (120251,4085,1672,545))]:
        f = pd.concat([f for task,f in cohorts.items() if task.startswith(endpoint)])
        actual = [len(f),len(f[['pitcher_id','season']].drop_duplicates()),f.pitcher_id.nunique(),f.injury_event_id.nunique()]
        for quantity, published, value in zip(['pitcher-games','pitcher-seasons','unique pitchers','injury events with pre-IL games'],pub,actual):
            comparison.append(dict(task=endpoint+'_combined',quantity=quantity,kang=published,replication=value,
                explanation='Controls shared across endpoint analyses; individual pitchers counted once across role strata'))
    flow = pd.read_csv(tables/'cohort_flow.csv').set_index('stage')
    for stage,games,seasons in [('raw_pitching',175685,7491),('missingness_and_role',172022,5943)]:
        for quantity,published,column in [('pitcher-games',games,'games'),('pitcher-seasons',seasons,'pitcher_seasons')]:
            comparison.append(dict(task=stage,quantity=quantity,kang=published,replication=flow.loc[stage,column],
                explanation='Retrieved source coverage, required-field missingness and unresolved role ties'))
    pd.DataFrame(windows).to_csv(tables/'optimal_windows.csv',index=False)
    comparison_frame=pd.DataFrame(comparison)
    comparison_frame['numeric_difference']=pd.to_numeric(comparison_frame.replication,errors='coerce')-pd.to_numeric(comparison_frame.kang,errors='coerce')
    comparison_frame.to_csv(tables/'published_comparison.csv',index=False)
    metric = pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame()
    gain = pd.concat(gains,ignore_index=True) if gains else pd.DataFrame()
    metric.to_csv(tables/'roc_auc_results.csv',index=False)
    gain.to_csv(tables/'feature_importance.csv',index=False)
    if sensitivities: pd.concat(sensitivities).to_csv(tables/'prior_injury_sensitivity.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,2,figsize=(11,13))
    for row,(task,_) in enumerate(cohorts.items()):
        d=cache/task; cv=pd.read_csv(d/'window_cv.csv'); smooth=pd.read_csv(d/'gam_curve.csv')
        w=json.loads((d/'window_selection.json').read_text()); pub=PUBLISHED[task]
        means=cv.groupby('window').roc_auc.mean()
        axes[row,0].scatter(means.index,means,s=10,color='grey',label='Mean training CV AUC')
        axes[row,0].plot(smooth.window,smooth.smoothed_auc,label='mgcv GAM')
        axes[row,1].plot(smooth.window,smooth.slope)
        for key in ['low_slope','high_slope']: axes[row,1].axhline(smooth[key].iloc[0],ls='--',color='grey')
        for ax in axes[row]:
            ax.axvspan(pub['low'],pub['high'],color='orange',alpha=.18,label='Published band')
            if w['lower'] is not None:
                # Shade actual qualifying segments; do not conceal gaps with one rectangle.
                active=smooth.in_band.to_numpy(dtype=bool)
                starts=np.flatnonzero(np.diff(np.r_[False,active].astype(int))==1)
                ends=np.flatnonzero(np.diff(np.r_[active,False].astype(int))==-1)
                for j,(start,end) in enumerate(zip(starts,ends)):
                    ax.axvspan(smooth.window.iloc[start],smooth.window.iloc[end],color='green',alpha=.2,
                               label='Replication slope band' if j==0 else None)
                ax.axvline(w['selected'],color='green',ls=':',lw=1,label='Rounded band midpoint')
            ax.set_xlabel('Cumulative pre-IL window (days)')
        axes[row,0].set_title(task.replace('_',' / '));axes[row,0].set_ylabel('ROC-AUC')
        axes[row,1].set_title('GAM derivative; 25–35% maximum slope');axes[row,1].set_ylabel('AUC gain per day')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Kang-method replication: window selection\nGAM settings / grid approximate; band is not a confidence interval')
    fig.tight_layout();fig.savefig(figures/'01_window_and_slope.png',dpi=160);plt.close(fig)
    if not gain.empty:
        fig, axes=plt.subplots(2,2,figsize=(12,9))
        for ax,(task,part) in zip(axes.flat,gain.groupby('task',sort=False)):
            top=part.head(10).iloc[::-1]
            ax.barh(top.feature,top.mean_gain);ax.set_title(task.replace('_',' / '));ax.set_xlabel('Mean normalized total Gain')
        fig.suptitle('Published-style importance: 500 training pitcher-season bootstrap refits')
        fig.tight_layout();fig.savefig(figures/'02_feature_importance.png',dpi=160);plt.close(fig)
        from sklearn.metrics import roc_curve
        fig,axes=plt.subplots(2,2,figsize=(9,8))
        for ax,row in zip(axes.flat,metric.itertuples()):
            pred=pd.read_csv(cache/row.task/'predictions.csv');fpr,tpr,_=roc_curve(pred.target,pred.prediction)
            ax.plot(fpr,tpr,label=f'AUC {row.roc_auc:.3f} ({row.ci_low:.3f}–{row.ci_high:.3f})')
            ax.plot([0,1],[0,1],color='grey',ls='--');ax.set_title(row.task.replace('_',' / '))
            ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.legend(fontsize=8)
        fig.suptitle('Random-game holdout ROC; same pitchers may occur in training')
        fig.tight_layout();fig.savefig(figures/'03_roc_curves.png',dpi=160);plt.close(fig)
    counts=pd.read_csv(tables/'cohort_counts.csv').query("control_definition == 'no_shoulder_elbow_il'")
    window_interpretation=[]
    for w in windows:
        pub=PUBLISHED[w['task']]
        if w['lower'] is None:
            finding='No qualifying declining-slope band.'
        elif w['segments']>1:
            finding=f"{w['segments']} disconnected segments; the envelope does not identify a unique window."
        elif w['lower']>pub['high']:
            finding='The qualifying band is later/longer than the published band.'
        elif w['upper']<pub['low']:
            finding='The qualifying band is earlier/shorter than the published band.'
        else:
            finding='The band overlaps the published interval; this is descriptive agreement, not an equivalence test.'
        window_interpretation.append(f"- **{w['task'].replace('_',' / ')}:** {finding}")
    body=['# Kang parent-paper replication',
        'This stage reproduces the published experimental structure using existing data. It adds no UCL-specific endpoint, chronological split, new warning metric, learner comparison or project-specific predictor.',
        '## Cohort',counts.to_markdown(index=False),
        'The Methods control definition is primary; counts using the conflicting Results no-IL definition are retained in `tables/cohort_counts.csv`. Events and controls are shared across repeated games; endpoint totals must not be summed as unique people.',
        '## Window and discrimination comparison',
        metric[['task','lower','upper','band_segments','window','selected_day_in_band','roc_auc','ci_low','ci_high','published_auc','features']].round(3).to_markdown(index=False) if not metric.empty else 'No qualifying derivative bands; final models unavailable.',
        'Bands come from the published GAM derivative structure, not maximum AUC. They are not confidence intervals. The candidate increment, smoother settings and band-to-integer conversion are undisclosed in the article and explicitly approximated here.',
        'Multiple qualifying segments mean the earliest/latest interval is an envelope containing gaps; its rounded midpoint can fall outside every qualifying segment. Such a result is not a well-identified unique optimal window, even when its midpoint resembles the published window.',
        '\n'.join(window_interpretation),
        '## Feature importance',gain.groupby('task',sort=False).head(5)[['task','feature','domain','mean_gain']].round(4).to_markdown(index=False) if not gain.empty else 'Unavailable without final models.',
        'Gain is normalized total split gain averaged over training bootstrap refits; it does not identify physiological mechanisms.',
        '## Fidelity and unresolved details',
        'Matched: years, game grain, broad injury framing, role strata, random 67/33 games with pitcher overlap, XGBoost, inverse-frequency weighting, 50 repeated five-fold selection fits, domain Pearson threshold 0.7, ROC-AUC and 500 pitcher-season confidence resamples. The same GAM slope-band structure and prior-region injury sensitivity were implemented.',
        'Approximate/unavailable: author cohort records, conflicting control wording, exact feature list (Y versus extension), outside-window case handling, exact random seed/grid, selection cutoffs/subset size, tuning procedure, hyperparameters, software versions, smoother settings and final window conversion. The published feature-specific rolling-risk attribution lacks enough detail for numerical replication and was not replaced with a new method.',
        'This is a reproducible **qualified structural benchmark**, not an exact numerical replication or evidence of deployment performance. Do not change settings to force agreement. Author code/data would be needed to remove the documented ambiguities.',
        'See [protocol](../../docs/kang_replication_methods.md), [config](../../config/kang_replication.yaml), and [direct comparison](tables/published_comparison.csv). Existing analyses and the central database remain unchanged.']
    (out/'summary.md').write_text('\n\n'.join(body)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv);cfg=config(args.config)
    before=(ROOT/cfg['database']).stat()
    cohorts,source=prepare(cfg)
    out=ROOT/cfg['output']
    atomic_json(out/'replication_spec.json',dict(config=cfg,source_manifest=source,predictors=PREDICTORS,
        source_of_truth='10.1177/23259671261477532',protocol=digest(ROOT/'docs/kang_replication_methods.md')))
    print(pd.read_csv(out/'tables/cohort_counts.csv').to_string(index=False),flush=True)
    if args.prepare_only:return
    rscript=shutil.which(cfg['rscript'])
    if not rscript:raise RuntimeError('Rscript required; install R and xgboost/mgcv/jsonlite or configure its path.')
    software=subprocess.check_output([rscript,'--vanilla','-e',
        'cat(R.version.string,"\\n");for(p in c("xgboost","mgcv","jsonlite"))cat(p,as.character(packageVersion(p)),"\\n")'],
        env=dict(os.environ,LC_ALL='C',LANG='C'),text=True)
    (out/'software.txt').write_text(software)
    with ThreadPoolExecutor(max_workers=cfg['workers']) as executor:
        pending=[executor.submit(execute_task,t,cfg,rscript,software) for t in cohorts]
        for future in as_completed(pending):future.result()
    report(cfg,cohorts)
    from .kang_replication_validation import validate
    validate(cfg)
    after=(ROOT/cfg['database']).stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    atomic_json(out/'verification.json',dict(database_read_only=True,database_unchanged=True,cohorts=len(cohorts),
        completed_tasks={t:json.loads((ROOT/cfg['cache']/t/'complete.json').read_text())['status'] for t in cohorts},
        study_years=[2017,2025],primary_metric='ROC-AUC',no_ucl_endpoint=True))


if __name__=='__main__':main()
