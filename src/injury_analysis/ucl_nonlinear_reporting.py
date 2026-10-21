"""Focused figures, synthesis and invariants for the nonlinear model-form test."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ucl_nonlinear_common import ROOT,OUT,CACHE,CFG,inputs,table,read_table,figure,snapshot,checkpoint,valid_checkpoint,write_json
from .ucl_nonlinear_models import load_model,scores
from .ucl_dynamic_evaluation import event_bootstrap
from .common import sha,source_signature

def label(mid):
    if mid.startswith('Linear|'):return 'Linear / '+mid.split('|')[1]
    rep,alg,setting=mid.split('__');rep=rep.replace('_',' + ') if rep.startswith('Profile_departure') else rep.replace('_',' ')
    return rep+' / '+('spline C='+setting.replace('p','.') if alg=='Spline' else 'boosting '+setting+' leaves')

def summary_tables():
    p=json.loads((OUT/'frozen_protocol.json').read_text());m=read_table('model_metrics');w=read_table('warning_metrics')
    ids=list(dict.fromkeys(list(p['warning_leaders'].values())+list(p['ranking_leaders'].values())))
    main=m.loc[m.model_id.isin(ids)&~m.partition.eq('train')].copy()
    table(main,'leader_comparison');table(w.loc[w.model_id.isin(ids)],'leader_warning_comparison')
    selected=set(p['per_representation'].values());table(m.loc[m.model_id.isin(selected)],'representation_comparison')
    events=read_table('event_warnings');uncertainty=[]
    for (part,mid,lead),g in events.loc[events.model_id.isin(ids)].groupby(['partition','model_id','minimum_lead']):
        b=event_bootstrap(g);uncertainty.append(dict(partition=part,model_id=mid,minimum_lead=lead,detected=int(g.detected.sum()),events=len(g),
            recall=g.detected.mean(),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975)))
    table(pd.DataFrame(uncertainty),'warning_uncertainty')
    # Additive nulls should remain numerical zero. Refit maps quantify reproducibility.
    inter=pd.concat([read_table('local_interaction_cells'),read_table('mechanism_probe_interactions')],ignore_index=True).drop_duplicates(['model_id','refit','left','right','x_bin','y_bin'])
    summaries=[]
    for (mid,left,right),g in inter.groupby(['model_id','left','right']):
        base=g.loc[g.refit.eq(-1)&g.supported].set_index(['x_bin','y_bin']);refits=g.loc[g.refit.ge(0)&g.supported]
        rms=[];correlation=[];sign=[]
        for _,r in refits.groupby('refit'):
            joined=r.set_index(['x_bin','y_bin']).join(base[['mixed_log_odds']],rsuffix='_original').dropna(subset=['mixed_log_odds','mixed_log_odds_original'])
            if joined.empty:continue
            rms.append(np.sqrt(np.average(joined.mixed_log_odds**2,weights=joined.rows)))
            if joined.mixed_log_odds.std()>1e-8 and joined.mixed_log_odds_original.std()>1e-8:
                correlation.append(joined.mixed_log_odds.corr(joined.mixed_log_odds_original))
                sign.append(np.average(np.sign(joined.mixed_log_odds)==np.sign(joined.mixed_log_odds_original),weights=joined.rows))
        summaries.append(dict(model_id=mid,left=left,right=right,supported_cells=len(base),
            original_rms=np.sqrt(np.average(base.mixed_log_odds**2,weights=base.rows)) if len(base) else np.nan,
            refit_rms_mean=np.mean(rms) if rms else np.nan,refit_rms_min=min(rms) if rms else np.nan,refit_rms_max=max(rms) if rms else np.nan,
            median_map_correlation=np.median(correlation) if correlation else np.nan,mean_cell_sign_agreement=np.mean(sign) if sign else np.nan))
    table(pd.DataFrame(summaries),'interaction_stability')
    curves=pd.concat([read_table('local_effect_curves'),read_table('mechanism_probe_curves')],ignore_index=True).drop_duplicates(['model_id','refit','feature','bin'])
    shapes=[]
    for (mid,feature),g in curves.groupby(['model_id','feature']):
        base=g.loc[g.refit.eq(-1)].set_index('bin');similarity=[];amplitude=[]
        for _,r in g.loc[g.refit.ge(0)].groupby('refit'):
            j=r.set_index('bin').join(base[['ale_log_odds']],rsuffix='_original')
            if j.ale_log_odds.std()>1e-8 and j.ale_log_odds_original.std()>1e-8:similarity.append(j.ale_log_odds.corr(j.ale_log_odds_original))
            amplitude.append(j.ale_log_odds.max()-j.ale_log_odds.min())
        shapes.append(dict(model_id=mid,feature=feature,median_refit_shape_correlation=np.median(similarity) if similarity else np.nan,
            original_effect_range=base.ale_log_odds.max()-base.ale_log_odds.min(),refit_effect_range_min=min(amplitude),refit_effect_range_max=max(amplitude)))
    table(pd.DataFrame(shapes),'effect_shape_stability')
    f,ev,s=inputs();train=f.loc[f.partition.eq('train')&f.target.notna()]
    contribution=read_table('event_ap_contributions');top=contribution.loc[contribution.fraction_total_ap.ge(.2)].copy();profiles=[]
    for row in top.itertuples():
        spec=p['catalog'][row.model_id]
        positive=f.loc[f.partition.eq(row.partition)&f.pitcher_id.eq(row.pitcher_id)&f.dynamic_next_event_id.eq(row.dynamic_next_event_id)&f.target.eq(1)]
        for feature in spec['features']:
            mean=positive[feature].mean();distribution=train[feature].dropna()
            profiles.append(dict(partition=row.partition,model_id=row.model_id,pitcher_id=row.pitcher_id,injury_event_id=row.dynamic_next_event_id,
                player_name=positive.pitcher_name.iloc[0],feature=feature,positive_row_mean=mean,
                training_percentile=float((distribution<=mean).mean()),minimum_prior_outings=int(positive.prior_appearance_count.min()),fraction_total_ap=row.fraction_total_ap))
    table(pd.DataFrame(profiles),'high_influence_pitcher_profiles')

def make_figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    p=json.loads((OUT/'frozen_protocol.json').read_text());m=read_table('model_metrics');w=read_table('warning_metrics')
    colors={'Linear':'#0072B2','Spline':'#009E73','Boosting':'#D55E00'};reps=CFG['representations']
    short=['Background','Profile','Departure-8','Profile + departure','+ trajectories','Previous-62','Stable-62']
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,part in zip(axes,['validation','test']):
        for j,alg in enumerate(colors):
            ids=[p['per_representation'][r+'|'+alg] for r in reps]
            v=m.loc[m.partition.eq(part)].set_index('model_id').loc[ids]
            ax.barh(np.arange(len(reps))+(j-1)*.23,v.pr_lift,height=.22,label=alg,color=colors[alg])
        ax.set(yticks=np.arange(len(reps)),yticklabels=short,xlabel='Average Precision / prevalence',title='Validation: 2022–2023' if part=='validation' else 'Exploratory transfer: 2024–2025');ax.axvline(1,color='gray',ls='--')
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=9);fig.suptitle('Nonlinear gains depend on representation\nPer-representation warning-selected settings; AP leaders are a separate hypothesis',fontsize=12);fig.tight_layout();figure(fig,'01_ranking_by_representation');plt.close(fig)
    leaders=list(dict.fromkeys(list(p['warning_leaders'].values())+[p['ranking_leaders']['Spline']]))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));pal=['#0072B2','#009E73','#D55E00','#CC79A7']
    for ax,part in zip(axes[:2],['validation','test']):
        for mid,c in zip(leaders,pal):
            g=w.loc[w.partition.eq(part)&w.model_id.eq(mid)&w.policy.eq('retrospective_budget')&w.alert_rate.eq(.05)].sort_values('minimum_lead')
            ax.plot(g.minimum_lead,g.injury_event_recall*100,'o-',label=label(mid),color=c)
        ax.set(title='Validation' if part=='validation' else 'Exploratory transfer',xlabel='Minimum warning lead (days)',ylabel='All-event recall (%)',xticks=[7,14,21],ylim=(0,30))
    for mid,c in zip(leaders,pal):
        g=w.loc[w.partition.eq('test')&w.model_id.eq(mid)&w.policy.eq('retrospective_budget')&w.minimum_lead.eq(7)].sort_values('alert_rate')
        axes[2].plot(g.alert_rate*100,g.injury_event_recall*100,'o-',color=c,label=label(mid))
    axes[2].set(title='Transfer budget curve: ≥7d',xlabel='Appearance alert budget (%)',ylabel='All-event recall (%)')
    axes[2].legend(fontsize=7,bbox_to_anchor=(1.02,1));fig.suptitle('Flexible Departure-8 adds one detected event; longer-lead gains do not transfer');fig.tight_layout();figure(fig,'02_warning_and_alert_budget');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));dev=read_table('development_search');g=dev.pivot(index='model_id',columns='partition',values='average_precision')
    for alg,c in colors.items():
        ids=dev.loc[dev.algorithm.eq(alg),'model_id'].unique();v=g.loc[ids];axes[0].scatter(v.train,v.validation,label=alg,color=c,alpha=.7)
    lim=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([0,lim],[0,lim],'--',color='gray');axes[0].set(xlabel='Training AP',ylabel='Validation AP',title='Higher training fit can conceal overfitting');axes[0].legend()
    calibration=read_table('calibration_bins')
    for mid,c in zip(leaders,pal):
        g=calibration.loc[calibration.partition.eq('test')&calibration.model_id.eq(mid)];axes[1].plot(g.mean_score*100,g.observed_risk*100,'o-',label=label(mid),color=c)
    lim=max(axes[1].get_xlim()[1],axes[1].get_ylim()[1]);axes[1].plot([0,lim],[0,lim],'--',color='gray');axes[1].set(title='Transfer calibration: validation-defined bins',xlabel='Predicted risk (%)',ylabel='Observed first-event UCL risk (%)');axes[1].legend(fontsize=7)
    fig.tight_layout();figure(fig,'03_overfitting_and_calibration');plt.close(fig)
    curves=pd.concat([read_table('local_effect_curves'),read_table('mechanism_probe_curves')],ignore_index=True).drop_duplicates(['model_id','refit','feature','bin'])
    choices=[('velo_z',[p['warning_leaders']['Linear'],p['per_representation']['Departure|Spline'],p['per_representation']['Departure|Boosting']]),
        ('release_z_sd_z',[p['per_representation']['Departure|Spline'],p['per_representation']['Departure|Boosting']]),
        ('extension_z',[p['per_representation']['Departure|Spline'],p['per_representation']['Departure|Boosting']]),
        ('release_z_sd_baseline_mean',[p['ranking_leaders']['Spline']]),
        ('extension_avg_baseline_mean',[p['ranking_leaders']['Spline'],p['ranking_leaders']['Boosting']]),
        ('pitches_prev_28d',[p['per_representation']['Previous 62|Boosting']])]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(feature,ids) in zip(axes.flat,choices):
        for mid in ids:
            g=curves.loc[curves.model_id.eq(mid)&curves.feature.eq(feature)];base=g.loc[g.refit.eq(-1)].sort_values('bin')
            if base.empty:continue
            alg=p['catalog'][mid]['algorithm'];c=colors[alg]
            boot=g.loc[g.refit.ge(0)].groupby('bin').ale_log_odds.agg(low='min',high='max').reindex(base.bin)
            ax.plot(base.x,base.ale_log_odds,'o-',label=label(mid),color=c,ms=3)
            ax.fill_between(base.x.to_numpy(),boot.low.to_numpy(),boot.high.to_numpy(),color=c,alpha=.13)
        ax.axhline(0,color='gray',ls='--',lw=.6);ax.set(title=feature,xlabel='Observed training-supported feature value',ylabel='Centered local log-odds effect');ax.legend(fontsize=6)
    fig.suptitle('Learned response shapes: 4,000 training reference outings, at most five per pitcher\nALE log-odds associations; shading spans eight refits, not a confidence interval',fontsize=12);fig.tight_layout();figure(fig,'04_nonlinear_effect_curves');plt.close(fig)
    inter=pd.concat([read_table('local_interaction_cells'),read_table('mechanism_probe_interactions')],ignore_index=True).drop_duplicates(['model_id','refit','left','right','x_bin','y_bin'])
    dep=p['per_representation']['Departure|Boosting'];prof=p['ranking_leaders']['Boosting']
    choices=[(dep,'velo_z','release_z_sd_z'),(prof,'avg_fb_velo_baseline_mean','extension_avg_baseline_mean')]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),gridspec_kw={'width_ratios':[1,1,1.2]})
    friendly={'velo_z':'Velocity departure','release_z_sd_z':'Release variability departure','extension_z':'Extension departure',
        'avg_fb_velo_baseline_mean':'Profile velocity','extension_avg_baseline_mean':'Profile extension',
        'release_z_sd_baseline_mean':'Profile release variability','age_at_appearance':'Age'}
    for ax,(mid,left,right) in zip(axes,choices):
        g=inter.loc[inter.model_id.eq(mid)&inter.left.eq(left)&inter.right.eq(right)&inter.refit.eq(-1)]
        z=g.pivot(index='y_bin',columns='x_bin',values='mixed_log_odds');limit=max(float(np.nanmax(np.abs(z))),.01)
        im=ax.imshow(z,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        for r in g.itertuples():
            ax.text(r.x_bin,r.y_bin,f'n={r.rows}' if r.supported else 'unsupported',ha='center',va='center',fontsize=7,color='white' if r.supported and abs(r.mixed_log_odds)>.5*limit else 'black')
        ax.set(title=label(mid),xlabel=friendly[left]+' quantile bin',ylabel=friendly[right]+' quantile bin',xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5));fig.colorbar(im,ax=ax,label='Local mixed log-odds contrast',shrink=.8)
    stab=read_table('interaction_stability');g=stab.loc[stab.model_id.isin([dep,prof])&stab.original_rms.gt(1e-8)].nlargest(5,'original_rms')
    axes[2].barh(range(len(g)),g.original_rms,color='#D55E00',alpha=.8)
    axes[2].hlines(range(len(g)),g.refit_rms_min,g.refit_rms_max,color='black',lw=2);axes[2].set(yticks=range(len(g)),yticklabels=[friendly[r.left]+' ×\n'+friendly[r.right] for r in g.itertuples()],xlabel='RMS local interaction contrast',title='Original fit and eight-refit ranges');axes[2].tick_params(axis='y',labelsize=7)
    fig.suptitle('Trees can fit interactions; fitted interactions are not evidence of useful warning',fontsize=12);fig.tight_layout();figure(fig,'05_interactions_and_stability');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));mid=p['ranking_leaders']['Spline'];ec=read_table('event_ap_contributions');loo=read_table('leave_one_injured_pitcher_out')
    for j,part in enumerate(['validation','test']):
        g=ec.loc[ec.model_id.eq(mid)&ec.partition.eq(part)].sort_values('fraction_total_ap',ascending=False)
        axes[0].bar(j,g.fraction_total_ap.iloc[0]*100,color=['#0072B2','#D55E00'][j]);axes[0].text(j,2,f'{g.fraction_total_ap.iloc[0]:.1%}',ha='center',color='white',weight='bold')
        q=loo.loc[loo.model_id.eq(mid)&loo.partition.eq(part)];original=q.original_ap.iloc[0];reduced=q.ap_without_pitcher.min()
        axes[1].plot([0,1],[original,reduced],'o-',label='Validation' if part=='validation' else 'Exploratory transfer',color=['#0072B2','#D55E00'][j])
    axes[0].set(xticks=[0,1],xticklabels=['Validation','Transfer'],ylabel='Share of total AP (%)',ylim=(0,100),title='One event dominates spline-profile AP')
    axes[1].set(xticks=[0,1],xticklabels=['All pitchers','Most influential\npitcher removed'],ylabel='Average Precision',title='Gain collapses under an influence check');axes[1].legend(fontsize=8)
    refit=read_table('training_refit_metrics');ids=[p['warning_leaders']['Linear'],p['warning_leaders']['Spline'],p['warning_leaders']['Boosting'],p['ranking_leaders']['Spline']]
    axes[2].boxplot([refit.loc[refit.model_id.eq(i),'injury_event_recall']*100 for i in ids],vert=False,tick_labels=['Linear departure','Spline departure','Tree departure','Spline profile'])
    axes[2].set(xlabel='Validation ≥7d event recall (%)',title='Eight fixed-setting training refits');fig.tight_layout();figure(fig,'06_composition_and_refit_uncertainty');plt.close(fig)

def validate():
    f,e,s=inputs();prior=snapshot();checks=[]
    def check(n,v,d=''):checks.append(dict(check=n,passed=bool(v),detail=d))
    check('Source database and canonical inputs preserved',source_signature()==prior['source'])
    changed=[p for p,h in prior['artifacts'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    check('All prior reports, notebooks, source modules and fitted models preserved',not changed,';'.join(changed))
    old=pd.read_csv(ROOT/'reports/ucl_dynamic/tables/model_metrics.csv');new=read_table('model_metrics')
    for rep in CFG['representations']:
        for part in ['validation','test']:
            a=old.loc[old.model.eq(rep)&old.horizon.eq(30)&old.partition.eq(part)].iloc[0]
            b=new.loc[new.model_id.eq('Linear|'+rep)&new.partition.eq(part)].iloc[0]
            check(f'Unchanged {part} linear scores/metrics: {rep}',np.allclose([a.rows,a.positive_rows,a.average_precision,a.brier_score],[b.rows,b.positive_rows,b.average_precision,b.brier_score],rtol=1e-12,atol=1e-14))
    protocol=json.loads((OUT/'frozen_protocol.json').read_text())
    check('Freeze precedes transfer artifact',pd.Timestamp(protocol['frozen_at_utc']).timestamp()<(CACHE/'predictions.parquet').stat().st_mtime)
    preds=pd.read_parquet(CACHE/'predictions.parquet');check('No 2026 modeling',preds.game_date.dt.year.max()<=2025)
    positive=f.loc[f.target.eq(1)&f.partition.isin(['train','validation','test'])]
    check('Positive events do not span retained partitions',positive.groupby('dynamic_next_event_id').partition.nunique().le(1).all())
    for mid in protocol['transfer_models']:
        obj=load_model(mid);check('Training-only fit: '+mid,pd.Timestamp(obj['training_max_date'])<=pd.Timestamp('2021-12-31'))
        check('Frozen fitted signature: '+mid,obj['signature']==protocol['fit_signatures'][mid])
        if '__Boosting__' in mid:check('No random early-stopping split: '+mid,obj['model'].early_stopping is False)
    for stage in ['models','analysis','diagnostics','probes']:check(stage+' checkpoint verified',valid_checkpoint(stage))
    check('ALE reference is training-only',pd.read_parquet(CACHE/'training_effect_reference.parquet').partition.eq('train').all())
    i=read_table('local_interaction_cells');null=i.loc[i.model_id.str.startswith('Linear|')|i.model_id.str.contains('__Spline__'),'mixed_log_odds'].dropna()
    check('Additive model interaction contrasts numerically zero',null.abs().max()<1e-10)
    check('Unsupported interaction cells are masked',i.loc[~i.supported,'mixed_log_odds'].isna().all())
    check('No partial artifacts',not list(CACHE.rglob('*.partial')) and not list(OUT.rglob('*.partial')))
    table(pd.DataFrame(checks),'validation_checks')
    if not all(r['passed'] for r in checks):raise AssertionError([r for r in checks if not r['passed']])
    return checks

def write_summary():
    text='''# Nonlinear UCL model evaluation

## Design and models

This additive stage preserves the refined UCL endpoint, first-event competing-injury rules, chronological partitions, 60-day embargo, completed-appearance cutoff, source missingness and frozen feature representations. No previous analysis is replaced. The 30-day risk sets are unchanged: 87,332 training rows (59 represented events), 28,875 validation rows (25 represented events), and 29,349 exploratory-transfer rows (28 represented events). All-event warning denominators remain 34 validation and 43 transfer episodes; only 20 and 28 have a 7–30d opportunity.

Seven frozen representations were compared: background, profile, departure-8, profile+departure, trajectory-enhanced profile+departure, previous-62 and stable-62. Existing L2 logistic models were reused exactly. The nonlinear search fitted 35 candidates: cubic additive spline logistic models at C=0.1 or 1; histogram gradient boosting with 2, 4 or 8 leaves, depth 1/2/3, 150 iterations, learning rate .05, minimum leaf 200 and L2=10. Stumps are additive on log-odds; larger trees permit interactions. Training-only clipping, median imputation and missingness indicators were retained. There was no random validation split, synthetic injury, class weighting, neural network or new feature selection.

Settings were selected solely on 2022–2023 validation warnings at a 5% budget, with AP above prevalence and simple-model tie breaks. AP leaders were frozen separately as ranking hypotheses. Exact fits, settings and numeric validation thresholds were written before new 2024–2025 predictions. No model was refitted on validation. The already examined transfer period is exploratory; 2026 was not evaluated.

## Principal results

| Hypothesis | Validation AP | Transfer AP | Validation ≥7d warnings | Transfer ≥7d warnings |
|---|---:|---:|---:|---:|
| Linear departure-8 | 0.00727 | 0.00584 | 7/34 | 2/43 |
| Spline departure-8, C=.1 | 0.00679 | 0.00571 | 7/34 | 3/43 |
| Boosted departure-8, 8 leaves | 0.00643 | 0.00533 | 8/34 | 3/43 |
| Spline profile, C=.1 (AP leader) | 0.02466 | 0.04451 | 4/34 | 4/43 |
| Boosted profile, 4 leaves (AP leader) | 0.02943 | 0.00646 | 3/34 | 1/43 |
| Linear stable-62 | 0.01092 | 0.01846 | 3/34 | 9/43 |

The warning leaders are departure models, not the profile AP leaders. Nonlinear departure models do not improve AP; the one-event warning gain has wide paired uncertainty including zero. At ≥14 days, departure-tree warnings are 5/34 validation versus 2/43 transfer; at ≥21 days, 5/34 versus 0/43. Its conditional first-warning median is 26.5 days in validation and 17 days in transfer, among only 8 and 3 detected events. There are roughly 180 and 489 alerts per detected event at 5% budget. A longer median among a few detections is not generally earlier detection.

### A large AP gain is concentrated in very few pitchers

The profile spline is the clearest apparent nonlinear ranking improvement, but one event supplies 68.1% of validation AP and another 79.1% of transfer AP. Removing the corresponding pitcher changes validation AP from 0.02466 to 0.00598 and transfer AP from 0.04451 to 0.00731. Those pitchers are Humberto Castellanos (2022) and Justin Martinez (2025); identifiers and raw training-relative profile values are retained for audit. This diagnostic removes that pitcher's positive and negative rows; it is not a new tuned model or an excuse to discard the cases.

The paired whole-pitcher AP-increment intervals include zero in both periods. Across eight training refits, profile-spline validation AP ranges from 0.00581 to 0.03940. The profile-tree AP leader falls from training AP 0.12947 to validation 0.02943 and transfer 0.00646; its refit validation AP is 0.00446–0.01073. These are material composition sensitivity and overfitting warnings, despite impressive single-fit AP.

### Nonlinearity versus interaction

Additive splines can recover curved profile associations without interactions. The strongest profile-spline validation permutation effects involve historical release-Z variability, release-X position, extension, age and experience. These correlated, persistent profile variables do not establish short-term imminence. The spline-profile gain is too concentrated to claim a reproducible population-level susceptibility relationship.

Departure models retain a lower-velocity/higher-risk direction already present in logistic regression. Curved extension, release-variability and spin effects occur in flexible fits, but they do not consistently increase ranking or warning. Curves are training-supported accumulated local log-odds effects with eight-refit ranges, not medical thresholds. Marginal permutation importance is conditional on the fitted model and can be shared or distorted by correlated features.

The warning-selected profile+departure and previous-62 boosting models are stumps. Their learned log-odds interaction contrasts are zero by construction. Deeper alternatives did not improve their validation warning counts. Trees can fit velocity×release-variability and profile mechanical interactions, but local interaction maps and their refit variability do not establish useful incremental warning. No interaction is interpreted as a biomechanical mechanism.

The broader feature sets do not benefit uniformly: stable-62 nonlinear selections have lower validation and transfer AP than linear stable-62; previous-62 splines are approximately tied with the linear benchmark in transfer AP but lower in validation. Trajectory-enhanced models also fail to establish a consistent warning advantage. The comparison supports curved profile associations in a subset of cases, not a general need for complex injury interactions.

## Calibration, burden and uncertainty

Brier scores, calibration curves with validation-defined bins, warning budgets at 1/2.5/5/10%, ≥7/14/21d recall, first-warning event details, false/nonactionable alert fractions and alerts per detection are saved. Rare outcomes make small Brier differences hard to interpret; a high AP alone does not establish calibration. Fixed validation numeric thresholds are evaluated separately from retrospective top-N budgets. All-eligible-appearance burden includes unknown outcome alerts without assigning healthy labels.

Conditional uncertainty uses 300 paired whole-pitcher resamples; eight common whole-pitcher training refits assess fit variability without retuning. Leave-one-injured-pitcher-out diagnostics, event AP contributions, role and unseen-pitcher checks expose composition sensitivity. Eight-refit effect ranges are descriptive, not 95% confidence intervals. Selection uncertainty and repeated exploratory comparisons remain; an empirical zero-warning bootstrap interval is not a population certainty bound.

## What to retain

Nonlinear modeling does **not materially change the conclusion that reliable UCL early warning has not been demonstrated**. Model form may affect a few profile-based rankings, but it is not established as the main warning bottleneck. Endpoint uncertainty, heterogeneous events, limited positive-event counts and missing direct biomechanical measurements remain plausible constraints; this study cannot identify their causal contributions.

Keep the original linear departure-8 and broader linear benchmarks frozen. Retain spline departure-8 and the shallow departure-tree as prespecified nonlinear warning challengers, and the profile spline as a separate composition-sensitive ranking hypothesis. Do not promote it to a deployed detector based on AP. Exact model artifacts and thresholds are in `frozen_protocol.json`; the next useful test is a genuinely future, complete post-freeze cohort with prospectively archived injury evidence and no retuning.

## Artifacts and rerun

- `notebooks/ucl_nonlinear/`: design/comparison; effects/stability; warning/transfer synthesis.
- `reports/ucl_nonlinear/figures/`: six focused PNG/PDF figures.
- `reports/ucl_nonlinear/tables/`: tuning, paired uncertainty, warning/calibration, influence, local effects, interactions, refits and validation.
- `data/injury/analysis/ucl_nonlinear/`: fitted candidate/refit models, predictions and the training-only interpretation reference.
- Methods and source references: `docs/ucl_nonlinear_methods.md`.

Resume with `.venv/bin/python -m src.injury_analysis.ucl_nonlinear_pipeline`. Execute notebooks with `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-nonlinear`. Completed checkpoints are reused; changed inputs fail loudly.
'''
    (OUT/'summary.md').write_text(text)
    guide=[('01_ranking_by_representation','Which representations benefit?','Profile gains stand out; broad sets do not uniformly benefit.'),
        ('02_warning_and_alert_budget','Do nonlinear scores warn earlier?','Departure gains are small and uncertain; long-lead validation results do not transfer.'),
        ('03_overfitting_and_calibration','Are gains generalizable and calibrated?','Profile trees fit training much better; calibration and rare-event precision remain limited.'),
        ('04_nonlinear_effect_curves','What response shapes were learned?','Local shapes differ, but are model associations with composition-sensitive refit ranges.'),
        ('05_interactions_and_stability','Do interactions explain useful signal?','Local nonadditivity exists in trees; its existence does not prove warning value.'),
        ('06_composition_and_refit_uncertainty','Are gains broadly distributed?','One event dominates profile-spline AP in each period; gains collapse under influence checks.')]
    table(pd.DataFrame(guide,columns=['figure','question','interpretation']),'figure_guide')

def run_reporting():
    if valid_checkpoint('reporting'):return
    summary_tables();make_figures();validate();write_summary()
    checkpoint('reporting',[OUT/'summary.md',OUT/'tables/validation_checks.csv']+list((OUT/'figures').glob('*')))
