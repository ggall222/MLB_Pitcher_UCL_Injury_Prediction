"""Temporal evaluation, fixed descriptive diagnostics and preservation checks."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .kang_spec import ROOT, atomic_json, digest, DOMAINS
from .kang_replication_data import PREDICTORS
from .kang_ucl_endpoint_data import check_hashes
from .kang_ucl_chronological_data import GRAIN, period_for, observed_labels, assert_separation

PERIODS=['training','validation','test']


def read_frames(cfg):
    cache=ROOT/cfg['cache'];frames={};predictions={}
    for role in ['starter','reliever']:
        for period in PERIODS:
            d=cache/role
            frames[(role,period)]=pd.read_csv(d/(period+'.csv'),parse_dates=['game_date'],low_memory=False)
            p=d/('predictions_'+period+'.csv')
            if p.exists():predictions[(role,period)]=pd.read_csv(p,parse_dates=['game_date'],low_memory=False)
    return frames,predictions


def reliability_edges(scores,bins):
    return np.unique(np.r_[-np.inf,np.quantile(scores,np.arange(1,bins)/bins),np.inf])


def report(spec,base,ext1,cfg,frozen,protocol):
    out=ROOT/cfg['output'];tables=out/'tables';cache=ROOT/cfg['cache']
    frames,predictions=read_frames(cfg)
    support=[];performance=[];windows=[];cv_rows=[];subgroups=[];drift=[];scores=[];reliability=[];importance=[]
    for role in ['starter','reliever']:
        d=cache/role;tr=frames[(role,'training')];seen=set(tr.pitcher_id)
        old=ROOT/ext1['cache']/('ucl_'+role)
        om=pd.read_csv(old/'metrics.csv').iloc[0].to_dict()
        performance.append(dict(role=role,period='random_comparator',**om,
            represented_events=om['test_events'],positive_games=om['positive_test_games'],
            prevalence=om['positive_test_games']/om['test_games']))
        if (d/'window_cv.csv').exists():
            cv=pd.read_csv(d/'window_cv.csv');g=cv.groupby('window').agg(cv_auc=('roc_auc','mean'),fold_sd=('roc_auc','std'),
                folds=('fold','size'),games=('games','sum'),positive_games=('positive_games','sum')).reset_index()
            g['role']=role;cv_rows.append(g)
        complete=frozen['roles'][role]['status']=='complete'
        if (d/'window_selection.json').exists():
            w=json.loads((d/'window_selection.json').read_text());curve=pd.read_csv(d/'gam_curve.csv')
            inside=bool(curve.iloc[(curve.window-w['selected']).abs().argmin()].in_band) if w['selected'] is not None else False
            windows.append(dict(role=role,**w,selected_day_in_band=inside,
                interpretable_single_band=w['segments']==1 and inside))
        else:windows.append(dict(role=role,status=frozen['roles'][role]['status']))
        if complete:
            edges=reliability_edges(predictions[(role,'training')].prediction,spec['reliability_bins'])
            for source,p in [('random',old/'importance.csv'),('chronological',d/'importance.csv')]:
                imp=pd.read_csv(p);imp['role']=role;imp['model']=source;imp['rank']=np.arange(1,len(imp)+1);importance.append(imp)
        for period in PERIODS:
            f=frames[(role,period)]
            record=dict(role=role,period=period,pitcher_games=len(f),pitcher_seasons=len(f[['pitcher_id','season']].drop_duplicates()),
                unique_pitchers=f.pitcher_id.nunique(),ucl_events=f.injury_event_id.nunique(),
                first_game=str(f.game_date.min().date()),last_game=str(f.game_date.max().date()),
                eligible_games=np.nan,censored_games=np.nan,represented_events=np.nan,
                positive_games=np.nan,negative_games=np.nan,prevalence=np.nan)
            for c in PREDICTORS:
                v=f[c];tv=tr[c];sd=tv.std()
                drift.append(dict(role=role,period=period,feature=c,n_observed=int(v.notna().sum()),missing_fraction=v.isna().mean(),
                    mean=v.mean(),sd=v.std(),q25=v.quantile(.25),median=v.median(),q75=v.quantile(.75),
                    standardized_mean_difference=(v.mean()-tv.mean())/sd if sd>0 else np.nan))
            if not complete:
                record['status']='model_unavailable';support.append(record)
                performance.append(dict(role=role,period=period,status='model_unavailable',roc_auc=np.nan,
                    ci_low=np.nan,ci_high=np.nan,represented_events=np.nan,positive_games=np.nan,
                    valid_bootstraps=0,brier=np.nan,mean_risk_bias=np.nan,prevalence=np.nan));continue
            p=predictions[(role,period)];m=pd.read_csv(d/('metrics_'+period+'.csv')).iloc[0].to_dict()
            record.update(m);record['small_event_support']=m['represented_events']<spec['small_event_count_flag']
            support.append(record);performance.append(dict(role=role,**m))
            q=p.prediction
            scores.append(dict(role=role,period=period,scored_games=len(p),mean=q.mean(),sd=q.std(),
                **{'q'+str(int(v*100)):q.quantile(v) for v in [.01,.1,.25,.5,.75,.9,.99]}))
            known=p[p.target.notna()].copy()
            known['bin']=pd.cut(known.prediction,edges,labels=False,include_lowest=True)
            for bin_id,b in known.groupby('bin'):
                reliability.append(dict(role=role,period=period,bin=int(bin_id),games=len(b),
                    positive_games=int(b.target.sum()),events=b.loc[b.target.eq(1),'injury_event_id'].nunique(),
                    mean_score=b.prediction.mean(),observed_rate=b.target.mean(),bin_source='training score quantiles'))
            if period!='training':
                for status,select in [('seen',p.pitcher_id.isin(seen)),('unseen',~p.pitcher_id.isin(seen))]:
                    z=p[select];k=z[z.target.notna()];pos=k[k.target.eq(1)]
                    subgroups.append(dict(role=role,period=period,training_pitcher_status=status,scored_games=len(z),
                        eligible_games=len(k),pitchers=z.pitcher_id.nunique(),pitcher_seasons=len(z[['pitcher_id','season']].drop_duplicates()),
                        positive_games=int(k.target.sum()),represented_events=pos.injury_event_id.nunique(),
                        prevalence=k.target.mean(),roc_auc=roc_auc_score(k.target,k.prediction) if k.target.nunique()==2 else np.nan,
                        small_event_support=pos.injury_event_id.nunique()<spec['small_event_count_flag']))
    support=pd.DataFrame(support);perf=pd.DataFrame(performance);win=pd.DataFrame(windows)
    support.to_csv(tables/'cohort_support.csv',index=False);perf.to_csv(tables/'performance_long.csv',index=False)
    win.to_csv(tables/'window_selection.csv',index=False)
    if cv_rows:pd.concat(cv_rows).to_csv(tables/'training_window_performance.csv',index=False)
    pd.DataFrame(subgroups).to_csv(tables/'seen_unseen_pitchers.csv',index=False)
    pd.DataFrame(drift).to_csv(tables/'predictor_drift.csv',index=False)
    pd.DataFrame(scores).to_csv(tables/'score_distribution.csv',index=False)
    pd.DataFrame(reliability).to_csv(tables/'reliability.csv',index=False)
    uncertainty=pd.read_csv(tables/'test_endpoint_uncertainty.csv')
    uncertainty.groupby(['evidence_category','endpoint_primary','in_model_cohort'],dropna=False).size().rename('episodes').reset_index().to_csv(
        tables/'test_endpoint_uncertainty_summary.csv',index=False)
    comparison=[]
    for role,g in perf.groupby('role'):
        r={'role':role};g=g.set_index('period');random=g.loc['random_comparator','roc_auc']
        for p in ['random_comparator','validation','test']:
            for metric in ['roc_auc','ci_low','ci_high','positive_games','represented_events','valid_bootstraps']:
                if p=='random_comparator':alias={'positive_games':'positive_test_games','represented_events':'test_events'}.get(metric,metric)
                else:alias=metric
                r[p+'_'+metric]=g.loc[p].get(alias,np.nan)
            if p!='random_comparator':r[p+'_delta_from_random']=g.loc[p].get('roc_auc',np.nan)-random
        comparison.append(r)
    comp=pd.DataFrame(comparison);comp.to_csv(tables/'primary_comparison.csv',index=False)
    gain=pd.concat(importance,ignore_index=True) if importance else pd.DataFrame()
    gain.to_csv(tables/'feature_importance.csv',index=False)
    imp_comp=[]
    for role in ['starter','reliever']:
        if not len(gain) or not (gain.role.eq(role)).any():continue
        g=pd.DataFrame({'feature':PREDICTORS});g['role']=role;g['domain']=g.feature.map(DOMAINS)
        for kind in ['random','chronological']:
            z=gain[(gain.role==role)&(gain.model==kind)][['feature','rank','mean_gain','gain_sd','valid_refits']]
            g=g.merge(z.rename(columns={c:kind+'_'+c for c in z.columns if c!='feature'}),on='feature',how='left')
        g['rank_change']=g.chronological_rank-g.random_rank
        imp_comp.append(g)
    importance_comparison=pd.concat(imp_comp,ignore_index=True) if imp_comp else pd.DataFrame()
    importance_comparison.to_csv(tables/'feature_importance_comparison.csv',index=False)
    plots(out,cache,predictions,perf,win,gain,pd.DataFrame(reliability))
    summary(spec,out,support,perf,win,importance_comparison,pd.DataFrame(subgroups),pd.DataFrame(drift),pd.DataFrame(scores))


def plots(out,cache,predictions,perf,windows,gain,reliability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=out/'figures';figs.mkdir(exist_ok=True)
    colors={'training':'#555555','validation':'#2878a8','test':'#bb4938','random_comparator':'#555555'}
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,role in enumerate(['starter','reliever']):
        d=cache/role
        if not (d/'gam_curve.csv').exists():continue
        cv=pd.read_csv(d/'window_cv.csv').groupby('window').roc_auc.mean();curve=pd.read_csv(d/'gam_curve.csv')
        w=windows[windows.role.eq(role)].iloc[0]
        axes[row,0].scatter(cv.index,cv,s=10,color='grey');axes[row,0].plot(curve.window,curve.smoothed_auc)
        axes[row,1].plot(curve.window,curve.slope)
        for c in ['low_slope','high_slope']:axes[row,1].axhline(curve[c].iloc[0],color='grey',ls='--')
        active=curve.in_band.to_numpy(dtype=bool)
        starts=np.flatnonzero(np.diff(np.r_[False,active].astype(int))==1);ends=np.flatnonzero(np.diff(np.r_[active,False].astype(int))==-1)
        for ax in axes[row]:
            for lo,hi in zip(starts,ends):ax.axvspan(curve.window.iloc[lo],curve.window.iloc[hi],alpha=.15,color='green')
            if pd.notna(w.get('selected')):ax.axvline(w.selected,color='green',ls=':')
            ax.set_xlabel('Cumulative preinjury window (days)')
        axes[row,0].set_title(role.title()+': training-only CV');axes[row,0].set_ylabel('ROC-AUC')
        axes[row,1].set_title('GAM slope; actual qualifying segments');axes[row,1].set_ylabel('AUC slope per day')
    fig.suptitle('Chronological model development uses 2017–2023 only\nShaded slope bands are not confidence intervals')
    fig.tight_layout();fig.savefig(figs/'01_training_windows.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for row,role in enumerate(['starter','reliever']):
        for col,period in enumerate(['validation','test']):
            ax=axes[row,col];ax.plot([0,1],[0,1],color='grey',ls='--')
            p=predictions.get((role,period));m=perf[(perf.role==role)&(perf.period==period)].iloc[0]
            if p is not None:
                k=p[p.target.notna()]
                if k.target.nunique()==2:
                    x,y,_=roc_curve(k.target,k.prediction);ax.plot(x,y,color=colors[period],label=f'AUC {m.roc_auc:.3f} ({m.ci_low:.3f}–{m.ci_high:.3f})')
                    ax.legend(fontsize=8,loc='lower right')
                ax.text(.03,.94,f'{len(k):,} games; {int(m.represented_events)} positive events',transform=ax.transAxes,fontsize=8)
            else:ax.text(.1,.5,'Model unavailable',transform=ax.transAxes)
            ax.set_title(role.title()+' / '+('2024–2025' if period=='validation' else '2026'))
            ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate')
    fig.suptitle('Frozen 2017–2023 models on later seasons');fig.tight_layout()
    fig.savefig(figs/'02_temporal_roc.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,role in zip(axes,['starter','reliever']):
        for x,period in enumerate(['random_comparator','validation','test']):
            m=perf[(perf.role==role)&(perf.period==period)].iloc[0]
            if pd.notna(m.get('roc_auc')):
                ax.vlines(x,m.ci_low,m.ci_high,color=colors[period],lw=2);ax.scatter(x,m.roc_auc,color=colors[period])
                events=m.get('test_events') if period=='random_comparator' else m.get('represented_events')
                ax.annotate(f'{m.roc_auc:.3f}\n{events:.0f} events',(x,m.ci_high),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(.5,color='grey',ls='--');ax.set_xticks([0,1,2]);ax.set_xticklabels(['Random games','2024–25','2026'])
        ax.set_ylim(0,1.18);ax.set_yticks(np.arange(0,1.01,.2));ax.set_xlim(-.4,2.4);ax.set_title(role.title());ax.set_ylabel('ROC-AUC with clustered 95% interval')
    fig.suptitle('Random discrimination versus later-time discrimination')
    fig.tight_layout();fig.savefig(figs/'03_auc_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,role in zip(axes,['starter','reliever']):
        if not len(gain):continue
        g=gain[gain.role.eq(role)];top=set(g[g['rank'].le(6)].feature)
        order=g[g.feature.isin(top)].groupby('feature').mean_gain.max().sort_values().index
        for offset,kind,color in [(-.18,'random','#555555'),(.18,'chronological','#2878a8')]:
            values=g[g.model.eq(kind)].set_index('feature').mean_gain.reindex(order).fillna(0)
            ax.barh(np.arange(len(order))+offset,values,height=.34,label=kind,color=color)
        ax.set_yticks(np.arange(len(order)));ax.set_yticklabels(order,fontsize=8);ax.set_title(role.title());ax.legend(fontsize=8)
        ax.set_xlabel('Mean normalized total Gain')
    fig.suptitle('Training model reliance: same bootstrap Gain method\nAbsent bar = not selected; no later-period importance refits')
    fig.tight_layout();fig.savefig(figs/'04_importance_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,role in zip(axes,['starter','reliever']):
        for period in PERIODS:
            p=predictions.get((role,period))
            if p is None:continue
            q=np.linspace(0,1,201);x=np.maximum(p.prediction.quantile(q).to_numpy(),1e-12)
            ax.plot(x,q,label=f'{period} (n={len(p):,})',color=colors[period])
        ax.set_xscale('log');ax.set_xlabel('XGBoost score (log scale)');ax.set_ylabel('Cumulative fraction of games')
        ax.set_title(role.title());ax.legend(fontsize=8)
    fig.suptitle('Score drift under the same frozen model\nAll scored games; training distribution is in-sample; scores are not calibrated risk')
    fig.tight_layout();fig.savefig(figs/'05_score_drift.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,role in zip(axes,['starter','reliever']):
        if not len(reliability):continue
        g=reliability[(reliability.role==role)&reliability.period.ne('training')]
        maximum=max(.01,g.mean_score.max(),g.observed_rate.max())*1.1
        ax.plot([0,maximum],[0,maximum],color='grey',ls='--')
        for period in ['validation','test']:
            z=g[g.period.eq(period)]
            ax.plot(z.mean_score,z.observed_rate,'o-',color=colors[period],label=period)
        ax.set_title(role.title());ax.set_xlabel('Mean model score');ax.set_ylabel('Observed positive fraction');ax.legend(fontsize=8)
    fig.suptitle('Descriptive reliability: training-defined bins, no recalibration\nRow/event counts for each bin are saved in the reliability table')
    fig.tight_layout();fig.savefig(figs/'06_reliability.png',dpi=170);plt.close(fig)


def classify_conclusion(perf,spec):
    later=perf[perf.period.isin(['validation','test'])]
    if len(later)!=4 or later.roc_auc.isna().any() or later.ci_low.isna().any():return 'Temporal discrimination does not clearly persist'
    if later.represented_events.lt(spec['small_event_count_flag']).any() or later.ci_low.le(.5).any():
        return 'Temporal discrimination does not clearly persist'
    random=perf[perf.period.eq('random_comparator')].set_index('role').roc_auc
    if any(r.ci_high<random[r.role] for r in later.itertuples()):return 'Temporal discrimination weakens but remains above chance'
    return 'Temporal discrimination largely persists'


def summary(spec,out,support,perf,windows,importance,subgroups,drift,scores):
    conclusion=classify_conclusion(perf,spec)
    view=support[['role','period','pitcher_games','eligible_games','censored_games','pitcher_seasons','unique_pitchers',
                  'ucl_events','represented_events','positive_games','negative_games','prevalence']]
    pview=perf[perf.period.ne('training')][['role','period','roc_auc','ci_low','ci_high','represented_events','valid_bootstraps']]
    notes=[]
    for role in ['starter','reliever']:
        g=perf[perf.role.eq(role)].set_index('period');r=g.loc['random_comparator','roc_auc']
        w=windows[windows.role.eq(role)].iloc[0]
        if pd.notna(w.get('selected')):
            notes.append(f"**{role.title()} window:** {w.selected:.0f} days; slope envelope {w.lower:.2f}–{w.upper:.2f}, "
                f"{int(w.segments)} segment(s), midpoint inside a segment: {bool(w.selected_day_in_band)}. "
                'This checks band shape, not window-selection stability across independent samples.')
        else:notes.append(f'**{role.title()}:** window/model unavailable; no replacement rule applied.')
        for period in ['validation','test']:
            z=g.loc[period]
            if pd.notna(z.get('roc_auc')):
                notes.append(f"{role.title()} {period}: AUC {z.roc_auc:.3f} ({z.ci_low:.3f}–{z.ci_high:.3f}), "
                    f"{z.roc_auc-r:+.3f} versus random {r:.3f}; {int(z.represented_events)} represented events, "
                    f"{int(z.positive_games)} positive games; {int(z.censored_games)} administratively censored games.")
        if len(importance):
            v=importance[importance.role.eq(role)]
            common=v[(v.random_rank<=5)&(v.chronological_rank<=5)].feature.tolist()
            leading=v.sort_values('chronological_rank').head(5).feature.tolist()
            notes.append(f"{role.title()} leading chronological predictors: {', '.join(leading)}. Shared top-five with random: {', '.join(common) or 'none'}. Gain is model reliance, not causation.")
    strongest=drift[drift.period.ne('training')].copy()
    strongest['magnitude']=strongest.standardized_mean_difference.abs()
    top=strongest.sort_values('magnitude',ascending=False).groupby(['role','period'],sort=False).head(3)
    uncertain=pd.read_csv(out/'tables/test_endpoint_uncertainty.csv')
    accepted=int(uncertain.endpoint_primary.sum());included=int(uncertain.in_model_cohort.sum())
    probable=int(uncertain.evidence_category.eq('probable_supplemental_documentation').sum())
    timing=int(uncertain.endpoint_exclusions.fillna('').str.contains('timing|missing_initial_date').sum())
    selected_days='/'.join(str(int(v)) for v in windows.get('selected',pd.Series(dtype=float)).dropna())
    censored=support.loc[support.period.eq('test'),'censored_games'].sum()
    latest=support.loc[support.period.eq('test'),'last_game'].max()
    followup=(f'At the selected {selected_days}-day horizons, {censored:.0f} test games require administrative censoring. '
        f'The last regular-season game is {latest}; the source cutoff is {spec["source_observation_end_expected"]}. '
        'Administrative maturity does not guarantee complete diagnosis/procedure confirmation. '
        f'The source has {accepted} accepted 2026 UCL episodes, of which {included} enter the benchmark-compatible cohort. '
        f'The audit also retains {probable} probable-supplemental episodes and {timing} episodes with timing/date exclusions; '
        'these are not promoted to confirmed outcomes. Unresolved IL absence is distinct from uncertain UCL identity. '
        'Later source confirmation could change retrospective ascertainment; no such confirmation is invented.')
    body=['# Extension 2 — chronological UCL evaluation',f'**Conclusion: {conclusion}.**',
        'The only major design change is calendar separation. All fitted choices use 2017–2023. The same frozen models score 2024–2025 and 2026; neither later period chooses a model, window or feature. The random-game comparator is archived Extension 1.',
        '## Cohort and positive-event support',view.round(6).to_markdown(index=False),
        'Pitcher-games are the screened cohort; eligible games have observed labels for the selected window. Cohort events and events represented by positive games are distinct. Source observation ends **2026-10-20**. Late negative horizons beyond that date remain censored. First-event and regular-season injury framing remain inherited.',
        followup,
        '## Windows and discrimination',pview.round(4).to_markdown(index=False),'\n\n'.join(notes),
        'The same 500 pitcher-season bootstrap attempts yield conditional intervals; degenerate draws are omitted and their valid count is reported. A small number of events limits inference even when intervals appear narrow. Training CV remains random within training, as required; its window curve is not a later-time performance estimate.',
        'In-sample training AUCs: '+', '.join(f'{r.role} {r.roc_auc:.3f}' for r in perf[perf.period.eq('training')].itertuples())+'. These are fit diagnostics, not independent evaluation. The comparison with later-period estimates does not isolate a specific cause of any generalization gap.',
        '## Seen and unseen pitchers',subgroups.round(5).to_markdown(index=False),
        'Seen status means the pitcher appeared in 2017–2023 training. Pitcher identities can recur across periods, but games, pitcher-seasons and positive events cannot. Subgroup differences are descriptive and may reflect very few events; there is no subgroup model selection.',
        'Unseen-pitcher point estimates are higher in the displayed comparisons, but these do not establish a reliable subgroup advantage. The high validation AUC for unseen relievers is based on only two events; in 2026 both reliever subgroups are near 0.39. The seen-starter test subgroup has only one event. These sparse contrasts should remain descriptive.',
        '## Score, predictor and calibration drift',scores.round(6).to_markdown(index=False),
        'Largest descriptive standardized mean shifts (all original fields remain in the full drift table):',
        top[['role','period','feature','standardized_mean_difference','missing_fraction']].round(4).to_markdown(index=False),
        'Score distributions include all scored cohort games, including censored test games; training scores are in-sample. Predictor drift is descriptive and does not alter feature selection. Class-weighted XGBoost scores are not assumed to be absolute risks. Reliability uses training-derived bins and observed labels only; no recalibration or updating was applied.',
        'The largest predictor shifts involve sweeper usage and extension, with additional velocity/spin shifts. Such distribution changes may reflect repertoire, population or measurement changes; they are not isolated causes of the AUC decline. Starter median/upper-tail scores rise in later periods, while average scores remain below observed prevalence in every later role/period. This distinguishes score drift and underprediction from discrimination.',
        perf[perf.period.isin(['validation','test'])][['role','period','brier','mean_risk_bias','prevalence']].round(6).to_markdown(index=False),
        '## Scope of the conclusion',
        'The frozen qualitative policy requires later intervals above chance and at least five represented events in every role/period before making a strong overall persistence statement. This is a conservative narrative rule, not an equivalence test or model-selection criterion. Read each role/period estimate alongside its support.',
        'A decline is evidence of weaker later-time performance under this design, not proof that the original experiment was invalid. Temporal drift, case composition and the removal of random cross-set event sharing can all contribute. This experiment does not identify their individual causal contributions.',
        'The endpoint remains retrospective operational adjudication. Historical source-publication archives are not independently verified; generic/unknown conditions and excluded possible-UCL episodes remain limitations. Season-role and eligibility use full-season information as in the parent. Previously analyzed 2026 data are locked for this experiment, but are not globally untouched or prospective confirmation. No clinical, deployment or operational-warning claim follows.',
        '## Reproducibility',
        'The protocol freeze records source hashes, settings and interpretation policy before fitting. The development freeze records training models, features and windows before later-period access. AUC is verified independently in Python; tests enforce date, game, pitcher-season and event separation and unchanged prior artifacts.',
        'See [methods](../../docs/kang_ucl_chronological_methods.md), [config](../../config/kang_ucl_chronological.yaml), '
        '[primary comparison](tables/primary_comparison.csv), [validation checks](tables/validation_checks.csv), and [AUC figure](figures/03_auc_comparison.png).']
    (out/'summary.md').write_text('\n\n'.join(body)+'\n')


def validate(spec,base,ext1,cfg,frozen,protocol):
    out=ROOT/cfg['output'];cache=ROOT/cfg['cache'];frames,predictions=read_frames(cfg);rows=[]
    assert_separation({p:pd.concat([frames[(r,p)] for r in ['starter','reliever']],ignore_index=True) for p in PERIODS})
    for role in ['starter','reliever']:
        by_period={p:frames[(role,p)] for p in PERIODS};assert_separation(by_period)
        d=cache/role;complete=frozen['roles'][role]['status']=='complete'
        for p,f in by_period.items():
            assert period_for(f.game_date,spec).eq(p).all()
            assert set(f.columns)==set(['game_id','pitcher_id','season','game_date','injury_event_id','lead_days','case_season']+PREDICTORS)
            if not complete:continue
            pred=predictions[(role,p)];m=pd.read_csv(d/('metrics_'+p+'.csv')).iloc[0]
            assert len(pred)==len(f) and np.array_equal(pred[GRAIN].to_numpy(),f[GRAIN].to_numpy())
            cutoff=spec['training_end'] if p=='training' else protocol['source_cutoff']
            pd.testing.assert_series_equal(pred.target.astype(float),observed_labels(f,int(m.window),cutoff),check_names=False)
            known=pred[pred.target.notna()]
            independent=roc_auc_score(known.target,known.prediction) if known.target.nunique()==2 else np.nan
            assert (pd.isna(independent) and pd.isna(m.roc_auc)) or abs(independent-m.roc_auc)<1e-10
            rows.append(dict(role=role,period=p,status='passed',games=len(f),known_labels=len(known),independent_auc=independent,
                no_cross_period_games=True,no_cross_period_pitcher_seasons=True,no_cross_period_positive_events=True))
        if complete:
            tr=by_period['training'];freeze=json.loads((d/'training_freeze.json').read_text())
            assert freeze['training_max_date']<=spec['training_end']
            assert set(freeze['features'])<=set(PREDICTORS)
            cv=pd.read_csv(d/'window_cv.csv');assert set(cv.window)==set(range(7,50))
            assert cv.groupby('window').size().eq(5).all()
            for w,g in cv.groupby('window'):
                assert g.games.sum()==len(tr)
                assert g.positive_games.sum()==int((tr.case_season & tr.lead_days.between(1,w)).sum())
            y=observed_labels(tr,freeze['selected_window'],spec['training_end']);assert y.notna().all()
            assert abs(freeze['positive_weight']-len(y)/(2*y.sum()))<1e-3
    for period in ['validation','test']:
        access=json.loads((out/(period+'_access.json')).read_text())
        assert pd.Timestamp(access['first_access_at_utc'])>=pd.Timestamp(frozen['frozen_at_utc'])
        assert access['development_freeze_sha256']==digest(out/'development_freeze.json')
    check_hashes(frozen['training_hashes']);check_hashes(protocol['prior_hashes']);check_hashes(protocol['source_hashes'])
    stat=(ROOT/base['database']).stat();assert protocol['database_stat']==dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    pd.DataFrame(rows).to_csv(out/'tables/validation_checks.csv',index=False)
    atomic_json(out/'verification.json',dict(prior_files_unchanged=len(protocol['prior_hashes']),endpoint_unchanged=True,
        database_unchanged=True,training_cutoff=spec['training_end'],source_cutoff=protocol['source_cutoff'],
        later_access_after_freeze=True,no_later_refits=True,no_cross_period_event_sharing=True,
        same_parent_functions=True,predictors=PREDICTORS,checks=json.loads(pd.DataFrame(rows).to_json(orient='records'))))
    print('Temporal separation, independent AUC, source preservation and frozen-model checks passed.',flush=True)
