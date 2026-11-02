"""Read-only parent comparisons, endpoint audit and independent validation."""
import json
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .kang_spec import ROOT, DOMAINS, atomic_json, digest
from .kang_replication_data import PREDICTORS
from .kang_ucl_endpoint_data import check_hashes, labels, GRAIN


def band_details(d):
    w=json.loads((d/'window_selection.json').read_text())
    curve=pd.read_csv(d/'gam_curve.csv')
    day=w['selected']
    inside=bool(curve.iloc[(curve.window-day).abs().argmin()].in_band) if day is not None else False
    if day is None:status='unavailable'
    elif w['segments']>1 or not inside:status='ambiguous_disconnected_band'
    else:status='single_qualifying_band'
    return w,inside,status


def report(spec,base,cfg,cohorts,benchmark,supported):
    out,cache,old=ROOT/cfg['output'],ROOT/cfg['cache'],ROOT/base['cache']
    tables=out/'tables';figures=out/'figures';figures.mkdir(exist_ok=True)
    primary=[];windows=[];importance=[];sensitivities=[]
    paths={}
    for role in ['starter','reliever']:
        for outcome,root,frames in [('elbow',old,benchmark),('ucl',cache,cohorts)]:
            d=root/(outcome+'_'+role);paths[(outcome,role)]=d
            f=frames[role]
            record=dict(outcome=outcome,role=role,cohort_games=len(f),cohort_events=f.injury_event_id.nunique(),
                injured_pitchers=f.loc[f.case_season,'pitcher_id'].nunique())
            if not (d/'complete.json').exists():
                primary.append(dict(record,status='insufficient_unchanged_CV_support'));continue
            cv=pd.read_csv(d/'window_cv.csv')
            aggregate=cv.groupby('window').agg(cv_roc_auc=('roc_auc','mean'),cv_fold_sd=('roc_auc','std'),
                folds=('fold','size') if 'fold' in cv else ('roc_auc','size'),
                validation_games=('games','sum'),validation_positive_games=('positive_games','sum')).reset_index()
            aggregate['outcome']=outcome;aggregate['role']=role;windows.append(aggregate)
            w,inside,status=band_details(d)
            record.update(selected_window=w['selected'],band_lower=w['lower'],band_upper=w['upper'],
                          band_segments=w['segments'],selected_day_in_band=inside,status=status)
            if json.loads((d/'complete.json').read_text())['status']=='complete':
                m=pd.read_csv(d/'metrics.csv').iloc[0].to_dict()
                record.update(m)
                y=labels(f,int(m['window']))
                record.update(cohort_positive_games=int(y.sum()),cohort_prevalence=float(y.mean()))
                partition=pd.read_csv(d/'split.csv')
                selected=f.merge(partition[GRAIN+['partition']],on=GRAIN,validate='one_to_one')
                positive=selected[labels(selected,int(m['window'])).eq(1)]
                train_ids=set(positive.loc[positive.partition.eq('train'),'injury_event_id'])
                test_ids=set(positive.loc[positive.partition.eq('test'),'injury_event_id'])
                record['shared_positive_events']=len(train_ids & test_ids)
                g=pd.read_csv(d/'importance.csv');g['outcome']=outcome;g['role']=role
                g['rank']=np.arange(1,len(g)+1);importance.append(g)
                sen=pd.read_csv(d/'prior_injury_sensitivity.csv');sen['outcome']=outcome;sen['role']=role
                sensitivities.append(sen)
            primary.append(record)
    result=pd.DataFrame(primary)
    result.to_csv(tables/'primary_comparison.csv',index=False)
    pd.concat(windows,ignore_index=True).to_csv(tables/'window_performance.csv',index=False)
    if sensitivities:pd.concat(sensitivities).to_csv(tables/'prior_broad_elbow_history_supplement.csv',index=False)
    gain=pd.concat(importance,ignore_index=True) if importance else pd.DataFrame()
    comparisons=[]
    for role in ['starter','reliever']:
        g=pd.DataFrame({'feature':PREDICTORS});g['role']=role;g['domain']=g.feature.map(DOMAINS)
        for outcome in ['elbow','ucl']:
            part=gain[(gain.role==role)&(gain.outcome==outcome)]
            p=part[['feature','rank','mean_gain','gain_sd','valid_refits']].rename(
                columns={c:outcome+'_'+c for c in ['rank','mean_gain','gain_sd','valid_refits']})
            g=g.merge(p,on='feature',how='left',validate='one_to_one')
            g[outcome+'_selected']=g[outcome+'_rank'].notna()
        g['rank_change_ucl_minus_elbow']=g.ucl_rank-g.elbow_rank
        g['selection_comparison']=np.select([g.elbow_selected & g.ucl_selected,
            g.elbow_selected & ~g.ucl_selected,~g.elbow_selected & g.ucl_selected],
            ['both','no_longer_selected','newly_selected'],default='neither')
        comparisons.append(g)
    comparison=pd.concat(comparisons,ignore_index=True)
    comparison.to_csv(tables/'feature_importance_comparison.csv',index=False)
    gain.to_csv(tables/'feature_importance.csv',index=False)
    for role in ['starter','reliever']:
        m=result[result.role.eq(role)].set_index('outcome')
        if m.loc['ucl','status']=='single_qualifying_band':
            lo,hi=m.loc['ucl',['band_lower','band_upper']]
            blo,bhi=m.loc['elbow',['band_lower','band_upper']]
            relation='longer' if lo>bhi else ('shorter' if hi<blo else 'overlapping')
        else:relation='ambiguous_or_unsupported'
        result.loc[result.role.eq(role),'ucl_vs_elbow_window']=relation
    result.to_csv(tables/'primary_comparison.csv',index=False)
    draw_figures(result,paths,comparison,figures)
    summarize(result,comparison,tables,out,cfg)


def draw_figures(result,paths,comparison,figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'elbow':'#3366a2','ucl':'#bc4b36'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex='col')
    for col,role in enumerate(['starter','reliever']):
        for outcome in ['elbow','ucl']:
            d=paths[(outcome,role)]
            if not (d/'gam_curve.csv').exists():continue
            cv=pd.read_csv(d/'window_cv.csv').groupby('window').roc_auc.mean()
            curve=pd.read_csv(d/'gam_curve.csv');w,inside,status=band_details(d)
            color=colors[outcome];label='Broad elbow' if outcome=='elbow' else 'Documented UCL'
            axes[0,col].scatter(cv.index,cv,s=9,color=color,alpha=.4)
            axes[0,col].plot(curve.window,curve.smoothed_auc,color=color,label=label)
            axes[1,col].plot(curve.window,curve.slope,color=color,label=label)
            for cutoff in ['low_slope','high_slope']:
                axes[1,col].axhline(curve[cutoff].iloc[0],color=color,ls='--',alpha=.35,lw=.8)
            active=curve.in_band.to_numpy(dtype=bool)
            starts=np.flatnonzero(np.diff(np.r_[False,active].astype(int))==1)
            ends=np.flatnonzero(np.diff(np.r_[active,False].astype(int))==-1)
            for ax in axes[:,col]:
                for start,end in zip(starts,ends):ax.axvspan(curve.window.iloc[start],curve.window.iloc[end],color=color,alpha=.1)
                if w['selected'] is not None:ax.axvline(w['selected'],color=color,ls=':',lw=1)
            r=result[(result.outcome==outcome)&(result.role==role)].iloc[0]
            axes[0,col].text(.02,.04+(0 if outcome=='elbow' else .065),
                f'{label}: {r.cohort_events:.0f} events; '+
                ('no band' if w['selected'] is None else f"{w['lower']:.1f}–{w['upper']:.1f} d; {w['segments']} segment(s)"),
                color=color,transform=axes[0,col].transAxes,fontsize=8)
        axes[0,col].set_title(role.title()+'s');axes[0,col].set_ylabel('Mean training CV ROC-AUC')
        axes[0,col].legend(loc='upper left',fontsize=8)
        axes[1,col].set_ylabel('Smoothed AUC slope per day');axes[1,col].set_xlabel('Cumulative preinjury window (days)')
    fig.suptitle('Extension 1: changing the endpoint within the frozen Kang framework\nShading: actual slope-band segments; dotted line: rounded envelope midpoint',fontsize=12)
    fig.tight_layout();fig.savefig(figures/'01_endpoint_window_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,role in zip(axes,['starter','reliever']):
        g=comparison[comparison.role.eq(role)].copy()
        top=g[(g.elbow_rank<=6)|(g.ucl_rank<=6)].copy()
        top['order']=top[['elbow_mean_gain','ucl_mean_gain']].max(axis=1)
        top=top.sort_values('order');y=np.arange(len(top))
        ax.barh(y-.18,top.elbow_mean_gain.fillna(0),height=.34,color=colors['elbow'],label='Broad elbow')
        ax.barh(y+.18,top.ucl_mean_gain.fillna(0),height=.34,color=colors['ucl'],label='Documented UCL')
        ax.set_yticks(y);ax.set_yticklabels(top.feature,fontsize=9);ax.set_title(role.title()+'s')
        ax.set_xlabel('Mean normalized total Gain');ax.legend(fontsize=8)
    fig.suptitle('Model reliance after endpoint refinement\nSame 500 training pitcher-season refits; union of each endpoint’s top six\nAbsent bar = feature not selected for that endpoint',fontsize=12)
    fig.tight_layout();fig.savefig(figures/'02_endpoint_feature_importance.png',dpi=180);plt.close(fig)


def summarize(result,comparison,tables,out,cfg):
    cohorts=pd.read_csv(tables/'cohort_comparison.csv');support=pd.read_csv(tables/'window_support.csv')
    overlap=pd.read_csv(tables/'endpoint_overlap_events.csv');mapping=pd.read_csv(tables/'all_ucl_event_mapping.csv')
    retained=int(overlap.retained_in_ucl_cohort.sum());original=len(overlap)
    additions=int(mapping.benchmark_mapping_status.eq('added_confirmed_UCL_not_benchmark_elbow').sum())
    differences=[]
    for role in ['starter','reliever']:
        c=cohorts[cohorts.role.eq(role)].set_index('outcome')
        for quantity in ['games','pitcher_seasons','unique_pitchers','injury_events','injured_pitchers','control_games','case_games']:
            differences.append(dict(role=role,quantity=quantity,broad_elbow=c.loc['elbow',quantity],
                ucl=c.loc['ucl',quantity],difference=c.loc['ucl',quantity]-c.loc['elbow',quantity]))
    pd.DataFrame(differences).to_csv(tables/'cohort_differences.csv',index=False)
    prevalence=support[support.window.eq(30)][['outcome','role','games','positive_games','negative_games','prevalence','represented_events']].copy()
    prevalence.to_csv(tables/'common_30_day_prevalence.csv',index=False)
    views=result[['outcome','role','cohort_events','selected_window','band_lower','band_upper','band_segments',
                  'selected_day_in_band','roc_auc','ci_low','ci_high','test_events','shared_positive_events','features','ucl_vs_elbow_window']]
    findings=[]
    for role in ['starter','reliever']:
        r=result[result.role.eq(role)].set_index('outcome');g=comparison[comparison.role.eq(role)]
        shared=g[(g.elbow_rank<=5)&(g.ucl_rank<=5)].feature.tolist()
        moved=g[g.rank_change_ucl_minus_elbow.abs()>=3][['feature','rank_change_ucl_minus_elbow']]
        movers=', '.join(f'{x.feature} ({x.rank_change_ucl_minus_elbow:+.0f} ranks)' for x in moved.itertuples()) or 'none with a ≥3-rank shift among shared selected features'
        new=', '.join(g.loc[g.selection_comparison.eq('newly_selected'),'feature']) or 'none'
        dropped=', '.join(g.loc[g.selection_comparison.eq('no_longer_selected'),'feature']) or 'none'
        if pd.notna(r.loc['ucl'].get('roc_auc',np.nan)):
            u=r.loc['ucl'];b=r.loc['elbow']
            direction='higher' if u.roc_auc>b.roc_auc else 'lower'
            finding=(f"**{role.title()}s:** UCL AUC {u.roc_auc:.3f} (95% cluster interval {u.ci_low:.3f}–{u.ci_high:.3f}), "
                f"versus elbow {b.roc_auc:.3f}; {direction} by {abs(u.roc_auc-b.roc_auc):.3f}. "
                f"Selected integer window {u.selected_window:.0f} days; band {u.band_lower:.2f}–{u.band_upper:.2f} "
                f"({int(u.band_segments)} segment(s), midpoint in band: {bool(u.selected_day_in_band)}). "
                f"Window relation: {u.ucl_vs_elbow_window}. {int(u.test_events)} UCL events contribute positive held-out games.")
        else:finding=f'**{role.title()}s:** final window/model unavailable under the unchanged procedure; no fallback or pooling.'
        findings += [finding, f"Shared top-five predictors: {', '.join(shared) or 'none'}. Rank changes (positive = lower UCL rank): {movers}. Newly selected: {new}. No longer selected: {dropped}."]
    good=result[result.outcome.eq('ucl') & result.roc_auc.notna()]
    if len(good)==2 and good.ci_low.gt(.5).all():
        conclusion='Ranking signal persists for both roles under the same random-game design: both UCL AUC intervals exceed 0.5. This is conditional discrimination, not evidence of ahead-of-time clinical usefulness or generalization to future seasons.'
    else:conclusion='Persistence is limited or unsupported in at least one role under the unchanged design. Report the available estimates without selecting a replacement window or model.'
    body=['# Extension 1 — UCL endpoint specificity',
        'Completed comparison against the frozen Kang replication. Endpoint/adjudication and resulting case membership/reference dates changed; model, predictors, splitting algorithm, selection, window search and ROC-AUC procedure did not.',
        '## Cohort and endpoint overlap',cohorts.to_markdown(index=False),
        f'The benchmark contains **{original} broad elbow events**. **{retained} ({100*retained/original:.1f}%)** remain UCL cases. '
        f'The same first-event rule also admits **{additions} confirmed forearm/UCL cases**, yielding **{retained+additions} UCL events**. '
        f'The established source contains {len(mapping)} accepted 2017–2025 UCL events before benchmark eligibility/first-event filtering. '
        'Non-UCL elbow cases are excluded, not reclassified as healthy controls. The endpoint-overlap and all-source mapping tables give event-level reasons.',
        '## Prevalence at the common 30-day candidate window',prevalence.round(6).to_markdown(index=False),
        'This common existing candidate window describes class composition; it is not an additional window-selection rule. All 43 windows and earlier-case negatives appear in `tables/window_support.csv`.',
        '## Selected-window results',views.round(4).to_markdown(index=False),
        '\n\n'.join(findings),
        'The intervals on ROC-AUC use the same 500 held-out pitcher-season resamples as the benchmark. The slope bands are not confidence intervals. A disconnected envelope or midpoint outside the band does not establish a unique optimal preinjury window.',
        '## Interpretation and decision',conclusion,
        'Higher or lower AUC is not automatically improvement or deterioration of a warning system. Fewer and different cases, their timing, prevalence-derived class weights and changed random-game membership all accompany the endpoint change. No paired significance test or operational-warning claim is made. Gain describes model reliance, not tissue damage or causation.',
        'The reduced independent-event count limits precision and makes feature ranks/window shapes sensitive to case composition. Shared pitchers across random train/test games remain intentional. Broad-control unknown diagnoses remain as in the benchmark. The source is retrospective operational adjudication, not an independent clinical review.',
        'The primary table also reports injury events with positive games in **both** training and test. The held-out games are not held-out injury events. Strong AUC can therefore reflect recognizable pitcher/event context; these results cannot isolate short-term warning information from persistent profile information.',
        'There is enough technical support to specify **Extension 2: chronological validation**, with event-support checks and the present benchmark frozen. These random-game findings do not predict that chronological performance will persist. Do not tune this endpoint experiment to make that later comparison favorable. Extension 2 has not been run.',
        '## Reproducibility',
        'The original R script is reused without edits. `experiment_freeze.json` hashes all benchmark artifacts and the authoritative endpoint. `tables/benchmark_equivalence.csv` verifies fixed settings; `tables/validation_checks.csv` independently verifies labels, splits and AUC. Only the two UCL model tasks are fitted. The automatic prior-broad-elbow-history sensitivity is supplemental and never used for selection.',
        'See [methods](../../docs/kang_ucl_endpoint_methods.md), [configuration](../../config/kang_ucl_endpoint.yaml), '
        '[window figure](figures/01_endpoint_window_comparison.png), [importance figure](figures/02_endpoint_feature_importance.png), '
        '[primary table](tables/primary_comparison.csv), and [importance comparison](tables/feature_importance_comparison.csv).']
    (out/'summary.md').write_text('\n\n'.join(body)+'\n')


def validate(spec,base,cfg,saved,supported):
    out,cache=ROOT/cfg['output'],ROOT/cfg['cache'];records=[]
    endpoint=pd.read_parquet(ROOT/spec['adjudicated_events'])
    accepted=set(endpoint.loc[endpoint.endpoint_primary,'injury_event_id'])
    seasons=pd.read_csv(ROOT/base['cache']/'season_eligibility.csv')
    for role in ['starter','reliever']:
        d=cache/('ucl_'+role);f=pd.read_csv(d/'cohort.csv',low_memory=False)
        assert not f.duplicated(GRAIN).any() and f.season.between(2017,2025).all()
        assert set(f.loc[f.case_season,'injury_event_id'])<=accepted
        assert f.loc[f.case_season,'lead_days'].gt(0).all()
        assert f.merge(seasons,on=['pitcher_id','season'],validate='many_to_one').role.eq(role).all()
        assert set(f.columns)==set(['game_id','pitcher_id','season','game_date','injury_event_id','lead_days','case_season','prior_region_il']+PREDICTORS)
        if not supported[role]:
            records.append(dict(role=role,status='insufficient_support'));continue
        done=json.loads((d/'complete.json').read_text())
        check_hashes(done['hashes'],d)
        cv=pd.read_csv(d/'window_cv.csv')
        assert set(cv.window)==set(range(7,50)) and cv.groupby('window').size().eq(5).all()
        split=pd.read_csv(d/'split.csv')
        assert len(split)==len(f) and not split.duplicated(GRAIN).any()
        joined=f.merge(split[GRAIN+['partition']],on=GRAIN,validate='one_to_one')
        train=joined[joined.partition.eq('train')];test=joined[joined.partition.eq('test')]
        for w,g in cv.groupby('window'):
            assert g.games.sum()==len(train) and g.positive_games.sum()==labels(train,w).sum()
        if done['status']=='complete':
            pred=pd.read_csv(d/'predictions.csv');m=pd.read_csv(d/'metrics.csv').iloc[0]
            score=roc_auc_score(pred.target,pred.prediction)
            assert abs(score-m.roc_auc)<1e-10
            z=pred.merge(test,on=GRAIN,validate='one_to_one')
            assert len(z)==len(test)==len(pred)
            assert np.array_equal(z.target,labels(z,int(m.window)))
            freeze=json.loads((d/'training_freeze.json').read_text())
            assert set(freeze['features'])<=set(PREDICTORS)
            imp=pd.read_csv(d/'importance.csv');assert imp.valid_refits.eq(500).all()
            assert done['selection_fits']==250 and done['importance_refits']==500
            records.append(dict(role=role,status='passed',independent_roc_auc=score,training_games=len(train),test_games=len(test),
                test_events=int(m.test_events),shared_pitchers=len(set(train.pitcher_id)&set(test.pitcher_id)),
                candidate_windows=43,selection_fits=250,importance_refits=500))
        else:records.append(dict(role=role,status='no_qualifying_band',candidate_windows=43))
    # Independently reconstruct actual random partitions using the frozen R helper.
    script=cache/'verify_split.R'
    script.write_text('''source("scripts/kang_replication.R")
args<-commandArgs(trailingOnly=TRUE); cfg<-fromJSON(args[1])
for(role in c('starter','reliever')) {
 p<-file.path(args[2],paste0('ucl_',role)); if(!file.exists(file.path(p,'split.csv')))next
 f<-read.csv(file.path(p,'cohort.csv')); s<-read.csv(file.path(p,'split.csv'))
 y<-as.integer(tolower(as.character(f$case_season))=='true' & !is.na(f$lead_days) & f$lead_days>=1 & f$lead_days<=cfg$window_max)
 expected<-stratified_subset(y,1-cfg$test_fraction,cfg$seed)
 stopifnot(identical(which(s$partition=='train'),expected),identical(s$game_id,f$game_id),identical(s$pitcher_id,f$pitcher_id))
 cat(role,': identical stratified random-game partition\\n')
}
''')
    p=subprocess.run([cfg['rscript'],'--vanilla',str(script),str(cache/'effective_model_config.json'),str(cache)],
        cwd=ROOT,text=True,capture_output=True,check=True,env=dict(os.environ,LC_ALL='C',LANG='C'))
    (out/'split_verification.txt').write_text(p.stdout)
    check_hashes(saved['benchmark_hashes']);check_hashes(saved['endpoint_source_hashes'])
    db=(ROOT/base['database']).stat();assert saved['database_stat']==dict(size=db.st_size,mtime_ns=db.st_mtime_ns)
    pd.DataFrame(records).to_csv(out/'tables/validation_checks.csv',index=False)
    atomic_json(out/'verification.json',dict(benchmark_hashes_unchanged=True,benchmark_files=len(saved['benchmark_hashes']),
        authoritative_endpoint_unchanged=True,database_metadata_unchanged=True,models_refit=['ucl_'+r for r in supported if supported[r]],
        no_benchmark_refits=True,identical_R_script=True,identical_predictors=True,identical_hyperparameters=True,
        identical_random_game_split_algorithm=True,chronological_split_used=False,roles=records))
    print('Extension validation passed; benchmark and endpoint hashes unchanged.',flush=True)
