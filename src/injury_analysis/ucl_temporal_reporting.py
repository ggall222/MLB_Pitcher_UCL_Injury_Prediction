"""Focused temporal figures, source preservation checks and empirical synthesis."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import sha,source_signature
from .ucl_temporal_common import ROOT,OUT,CACHE,CFG,read_table,table,figure,snapshot,checkpoint,valid_checkpoint,write_json
from .ucl_temporal_data import prepare,interval_target
from .ucl_temporal_models import load

ORDER=['1-7','8-14','15-30','31-60']
SHORT={'Appearance performance':'Performance','Location and strike fractions':'Location / command',
       'Pitch mix and pitch-type counts':'Pitch mix / counts','Pitcher background and experience':'Background / experience',
       'Release position and variability':'Release / variability','Role and recent starts':'Role / recent starts'}

def heat(ax,frame,value,title,center=False,ci=None):
    z=frame.pivot(index='family',columns='interval',values=value).reindex(columns=ORDER)
    v=z.to_numpy();lim=max(float(np.nanmax(np.abs(v))),.001)
    image=ax.imshow(v,aspect='auto',cmap='RdBu_r' if center else 'YlGnBu',
                    vmin=-lim if center else 0,vmax=lim)
    for i,fam in enumerate(z.index):
        for j,h in enumerate(z.columns):
            row=frame.loc[frame.family.eq(fam)&frame.interval.eq(h)].iloc[0]
            star='' if ci is None else ('*' if row[ci+'_ci_low']>0 or row[ci+'_ci_high']<0 else '')
            ax.text(j,i,f'{v[i,j]:.2f}{star}',ha='center',va='center',fontsize=8,
                    color='white' if (abs(v[i,j])>.58*lim if center else v[i,j]>.55*lim) else 'black')
    ax.set(xticks=range(4),xticklabels=ORDER,yticks=range(len(z)),
           yticklabels=[SHORT.get(n,n) for n in z.index],title=title,xlabel='Days before recorded UCL episode')
    ax.tick_params(axis='y',labelsize=9)
    return image

def make_figures():
    m=read_table('family_matrix_validation')
    counts=read_table('cohort_support').query("partition == 'validation'").set_index('interval')
    support='; '.join(f'{h}d: {int(counts.loc[h,"represented_events"])} events / {int(counts.loc[h,"positive_rows"])} rows' for h in ORDER)
    fig,ax=plt.subplots(figsize=(9,7))
    im=heat(ax,m,'standalone_lift','Standalone family information: validation PR lift')
    fig.colorbar(im,ax=ax,label='AP / interval prevalence')
    fig.suptitle('Common 22,269-appearance cohort; point estimates are not temporal confirmation\n'+support,fontsize=9)
    fig.tight_layout();figure(fig,'01_family_standalone');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    q=m.copy();q['ablation_pp']=100*q.ablation_ap_loss;q['permutation_pp']=100*q.permutation_ap_loss
    im=heat(axes[0],q,'ablation_pp','Full minus family-ablated AP',True,'ablation')
    fig.colorbar(im,ax=axes[0],label='AP loss (percentage points)',shrink=.7)
    im=heat(axes[1],q,'permutation_pp','Full minus jointly permuted-family AP',True,'permutation')
    fig.colorbar(im,ax=axes[1],label='AP loss (percentage points)',shrink=.7)
    fig.suptitle('Incremental information can differ from standalone performance\n* Pointwise 95% pitcher-bootstrap interval excludes zero; not multiplicity-adjusted. Negative values favor removing/shuffling the family.',fontsize=10)
    fig.tight_layout();figure(fig,'02_family_incremental');plt.close(fig)
    names=['Background','Profile','Departure','Previous 62','Stable 62','Full union']
    colors=['#999999','#009E73','#D55E00','#CC79A7','#0072B2','#222222']
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    for ax,part in zip(axes,['validation','test']):
        r=read_table('reference_uncertainty_'+part)
        for j,(name,c) in enumerate(zip(names,colors)):
            g=r.loc[r.model.eq(name)].set_index('interval').reindex(ORDER)
            ax.plot(range(4),g.pr_lift,'o-',color=c,label='Departure-8' if name=='Departure' else name,ms=4)
            # Bands for the two primary hypotheses; full CIs for every model in table.
            if name in ['Profile','Departure']:
                ax.fill_between(np.arange(4),g.lift_ci_low.to_numpy(),g.lift_ci_high.to_numpy(),color=c,alpha=.12)
        ax.axhline(1,color='gray',ls='--',lw=.8)
        ax.set(xticks=range(4),xticklabels=ORDER,xlabel='Days before recorded episode',
               title='Validation 2022–2023' if part=='validation' else 'Exploratory transfer 2024–2025',ylabel='PR lift (log scale)',yscale='log')
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Frozen representations refit to disjoint outcomes; no horizon-specific tuning\nShading: conditional 95% pitcher-bootstrap intervals for Profile and Departure-8 only',fontsize=11)
    fig.tight_layout();figure(fig,'03_frozen_representations');plt.close(fig)
    chosen=json.loads((OUT/'candidate_families.json').read_text())['families']
    refit=read_table('training_refit_family_losses');cont=read_table('temporal_contrasts_validation')
    fig,axes=plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios':[1,1,1.35]})
    for ax,family in zip(axes[:2],chosen):
        g=refit.loc[refit.family.eq(family)]
        for r,gg in g.groupby('refit'):
            v=gg.set_index('interval').reindex(ORDER).ablation_lift_loss
            ax.plot(range(4),v,'o-',alpha=.45,lw=1,ms=3,color='#0072B2')
        main=m.loc[m.family.eq(family)].set_index('interval').reindex(ORDER)
        ax.plot(range(4),main.ablation_lift_loss,'o-',color='black',lw=2,label='Original fit')
        ax.axhline(0,color='gray',ls='--')
        ax.set(xticks=range(4),xticklabels=ORDER,title=SHORT.get(family,family),
               ylabel='Ablation AP loss / interval prevalence',xlabel='Days before episode')
        ax.legend(fontsize=8)
    g=cont.loc[cont.family.isin(chosen)].copy()
    axes[2].hlines(range(len(g)),g.simultaneous_low,g.simultaneous_high,color='#999999',lw=3,label='Joint 95% band')
    axes[2].hlines(range(len(g)),g.ci_low,g.ci_high,color='#0072B2',lw=2,label='Pointwise 95% interval')
    axes[2].scatter(g.ablation_lift_difference,range(len(g)),color='black',s=18)
    axes[2].axvline(0,color='black',ls='--',lw=.8)
    axes[2].set(yticks=range(len(g)),yticklabels=[SHORT.get(r.family,r.family)+'\n'+r.contrast for r in g.itertuples()],
                xlabel='Difference in normalized ablation loss',title='Timing contrasts remain uncertain')
    axes[2].tick_params(axis='y',labelsize=7);axes[2].legend(fontsize=7)
    fig.suptitle('Six whole-pitcher training refits; two families nominated using development only\nBlue lines are refits, not confidence intervals; joint bands cover all 33 family-time contrasts',fontsize=11)
    fig.tight_layout();figure(fig,'04_temporal_stability');plt.close(fig)
    trajectory=read_table('trajectory_summary')
    display=CFG['trajectory_display_families']
    fig,axes=plt.subplots(len(display),2,figsize=(12,12))
    for i,fam in enumerate(display):
        for j,mode in enumerate(['calendar','outings']):
            ax=axes[i,j]
            g=trajectory.loc[trajectory.partition.eq('validation')&trajectory.family.eq(fam)&trajectory['mode'].eq(mode)&trajectory.support.eq('balanced')]
            for kind,color in [('Injury','#D55E00'),('Control','#0072B2')]:
                z=g.loc[g.kind.eq(kind)].sort_values('point')
                ax.plot(z.point,z['mean'],'o-',color=color,label=kind,ms=4)
                ax.fill_between(z.point.to_numpy(),z.ci_low.to_numpy(),z.ci_high.to_numpy(),color=color,alpha=.13)
            if not g.empty:
                z=g.drop_duplicates('point').sort_values('point')
                ax.set(xticks=z.point,xticklabels=[f'{r.label}\nn={r.events}' for r in z.itertuples()])
            ax.axhline(0,color='gray',ls='--',lw=.7)
            ax.set(title=SHORT.get(fam,fam)+' — '+('days' if j==0 else 'outings'),
                   ylabel='Training-standardized family log-odds')
            if i==0:ax.legend(fontsize=8)
    fig.suptitle('Retrospective matched trajectories, validation events only\nOne fixed training-derived 60-day score per family; same pairs across plotted points. Bands: pointwise dependency-cluster bootstrap.',fontsize=11)
    fig.tight_layout();figure(fig,'05_matched_family_trajectories');plt.close(fig)
    metric=read_table('development_metrics').query("partition == 'validation'")
    transfer=read_table('transfer_metrics')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,part,df in zip(axes,['Validation','Exploratory transfer'],[metric,transfer]):
        for name,c in zip(names,colors):
            z=df.loc[df.model.eq(name)].set_index('interval').reindex(ORDER)
            ax.plot(range(4),100*z.injury_event_recall,'o-',label=name,color=c,ms=4)
        denominator=int(z.qualifying_events.iloc[0])
        ax.set(xticks=range(4),xticklabels=ORDER,title=f'{part}: all {denominator} qualifying events',
               ylabel='Events warned within interval (%)',xlabel='Lead interval (days)')
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Warning at a retrospective 5% appearance budget\nInterval-specific warning; 1–7 days is not the existing ≥7-day practical-warning endpoint. No deployable threshold selected.',fontsize=11)
    fig.tight_layout();figure(fig,'06_interval_warning');plt.close(fig)

def validate():
    f,events,groups,models=prepare();checks=[]
    def add(name,passed,detail=''):checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    old=snapshot()
    add('source_signature_unchanged',source_signature()==old['source'])
    changed=[name for name,h in old['artifacts'].items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
    add('prior_artifact_hashes_unchanged',not changed,f'{len(old["artifacts"])} checked; changed={changed}')
    add('appearance_keys_unique',not f[['pitcher_id','game_id']].duplicated().any())
    add('regular_season',f.game_type.eq('R').all())
    add('union_is_113_frozen_fields',len(models['Full union'])==113)
    add('families_partition_union',len([c for g in groups.values() for c in g])==113 and len(groups)==11)
    for c in ['history_max_date','baseline_max_date','workload_max_date','dynamic_history_max_date']:
        add('strict_prior_'+c,(f[c].isna()|f[c].lt(f.game_date)).all())
    outcomes=pd.concat([interval_target(f,*h).target for h in CFG['intervals']],axis=1)
    add('exclusive_interval_outcomes',(outcomes.fillna(0).sum(axis=1)<=1).all())
    add('sum_disjoint_equals_60d',(outcomes.loc[f.target.notna()].astype(int).sum(axis=1).to_numpy()==f.loc[f.target.notna(),'target'].to_numpy()).all())
    add('unknown_outcomes_preserved',outcomes.loc[f.target.isna()].isna().all().all())
    add('active_episodes_excluded',not f.loc[f.target.notna(),'active_episode'].any())
    add('no_2026_predictive_metrics',not read_table('development_metrics').partition.eq('descriptive_2026').any()
        and read_table('transfer_metrics').partition.eq('test').all())
    for interval in ORDER:
        keys=[]
        for part in ['validation','test']:
            p=pd.read_parquet(CACHE/f'predictions_{part}_{interval}.parquet')
            g=interval_target(f,*map(int,interval.split('-')))
            g=g.loc[g.partition.eq(part)&g.target.notna()]
            add(f'keys_{part}_{interval}',g.row_id.tolist()==p.row_id.tolist())
            add(f'targets_{part}_{interval}',np.array_equal(g.target.to_numpy(),p.target.to_numpy()))
            add(f'finite_{part}_{interval}',np.isfinite(p[list(models)].to_numpy()).all())
        for name in models:
            obj=load(interval,name)
            add('train_only_'+interval+'_'+name,pd.Timestamp(obj['training_max_date'])<=pd.Timestamp('2021-12-31'))
    near=f.loc[f.target.eq(1)]
    spans=near.loc[near.partition.isin(['train','validation','test'])].groupby('dynamic_next_event_id').partition.nunique()
    add('positive_events_do_not_span_partitions',spans.le(1).all())
    protocol=json.loads((OUT/'development_freeze.json').read_text())
    transfer=json.loads((OUT/'transfer_manifest.json').read_text())
    add('development_frozen_before_transfer',transfer['started_after_development_freeze']==protocol['frozen_at_utc']
        and transfer['completed_at_utc']>protocol['frozen_at_utc'])
    for part in ['validation','test']:
        x=read_table('event_ap_contributions_'+part)
        add('ap_contributions_sum_'+part,np.allclose(x.groupby(['interval','model']).fraction_total_ap.sum(),1))
    add('no_partial_artifacts',not list(CACHE.rglob('*.partial')) and not list(OUT.rglob('*.partial')))
    t=table(pd.DataFrame(checks),'validation_checks')
    if not t.passed.all():raise AssertionError(t.loc[~t.passed].to_string(index=False))
    return checks

def make_summary():
    m=read_table('family_matrix_validation');refs=read_table('reference_uncertainty_validation')
    support=read_table('cohort_support').query("partition == 'validation'").set_index('interval')
    contrast=read_table('temporal_contrasts_validation')
    significant=contrast.loc[contrast.simultaneous_low.gt(0)|contrast.simultaneous_high.lt(0)]
    selected=json.loads((OUT/'candidate_families.json').read_text())['families']
    concentration=read_table('event_ap_contributions_validation')
    influence=read_table('pitcher_influence_validation')
    shapes=read_table('trajectory_shape')
    refit=read_table('training_refit_family_losses')
    lines=['# Initial UCL temporal signal decomposition','',
        'This is development-stage signal discovery, not an independently confirmed early-warning system. '
        'The refined endpoint, source dates, frozen feature sets, chronological splits and database are unchanged.','',
        '## Design and support','',
        'Four disjoint future intervals: **1–7, 8–14, 15–30 and 31–60 days**. '
        'All primary models use the same observed 60-day first-event cohort: 64,646 training, 22,269 validation and 22,782 exploratory-transfer appearances. '
        'A known other injury ends the first-event risk interval; ambiguous outcomes remain censored. '
        'A UCL event in a different interval is a negative for the specific interval, not a healthy-pitcher designation.','',
        '| Validation interval | Positive appearances | Represented events | All qualifying events |',
        '|---|---:|---:|---:|']
    for h in ORDER:
        r=support.loc[h];lines.append(f'| {h} days | {int(r.positive_rows)} | {int(r.represented_events)} | {int(r.qualifying_events)} |')
    lines+=['','The 31–60-day estimate is supported by only 10 validation events. Interval widths and event composition differ; '
        'PR lift normalizes prevalence but does not make the prediction tasks identical. '
        'All-event warning denominators include events without eligible opportunities.','',
        'The 113-field union of Background, Profile, Departure-8, Previous-62 and Stable-62 defines 11 families. '
        'Fixed C=1 logistic regression is used for all 112 interval fits. '
        'Scaling, clipping, imputation and missingness flags are training-derived. '
        'No interval boundaries, representations or regularization were tuned. '
        'Arm angle and within-outing velocity decline remain outside this long-history analysis because of prior coverage exclusions; '
        'documented prior UCL history remains excluded for unverified publication timing. Prior publicly reported IL count is included in background.','',
        '## What the estimates show','',
        '| Family/reference | 1–7d PR lift | 8–14d | 15–30d | 31–60d |',
        '|---|---:|---:|---:|---:|']
    for name in ['Velocity','Workload and rest','Pitcher background and experience','Role and recent starts']:
        g=m.loc[m.family.eq(name)].set_index('interval')
        lines.append('| '+name+' | '+' | '.join(f'{g.loc[h,"standalone_lift"]:.2f}' for h in ORDER)+' |')
    for name in ['Departure','Profile','Previous 62','Stable 62']:
        g=refs.loc[refs.model.eq(name)].set_index('interval')
        lines.append('| '+name+' | '+' | '.join(f'{g.loc[h,"pr_lift"]:.2f}' for h in ORDER)+' |')
    lines+=['',
        f'**{len(significant)} of 33 predeclared family timing contrasts exclude zero under joint 95% bootstrap bands.** '
        'Point estimates alone do not establish acute, intermediate or persistent categories. '
        'Family ablation and grouped permutation sometimes disagree; the families share correlated information, and refitting after ablation can reduce overfitting. '
        'The permutation measure is marginal within-season reliance, not a conditional or causal effect.','',
        '## Event concentration and stability','']
    for family,h in [('Velocity','1-7'),('Workload and rest','31-60')]:
        name='Only | '+family
        c=concentration.loc[concentration.model.eq(name)&concentration.interval.eq(h)].sort_values('fraction_total_ap',ascending=False).iloc[0]
        i=influence.loc[influence.model.eq(name)&influence.interval.eq(h)&influence.pitcher_id.eq(c.pitcher_id)].iloc[0]
        lines.append(f'- {family}, {h} days: one event contributes **{c.fraction_total_ap:.1%}** of AP. '
                     f'Excluding that pitcher from evaluation changes AP from {i.original_ap:.5f} to {i.ap_without_pitcher:.5f}. '
                     'This is an influence diagnostic; the event remains in the dataset.')
    lines+=['','Two families nominated using development-only mean normalized ablation loss: **'+', '.join(selected)+'**. '
            'These are follow-up candidates, not established temporal signals. Six common whole-pitcher training resamples preserve histories and repeat preprocessing.','',
            '| Nominated family | Interval | Positive ablation in training refits | Refit loss range (AP / prevalence) |',
            '|---|---|---:|---:|']
    for fam in selected:
        for h in ORDER:
            g=refit.loc[refit.family.eq(fam)&refit.interval.eq(h)]
            lines.append(f'| {fam} | {h} | {int(g.ablation_ap_loss.gt(0).sum())}/{len(g)} | {g.ablation_lift_loss.min():.2f} to {g.ablation_lift_loss.max():.2f} |')
    shared=read_table('shared_event_sensitivity_validation')
    lines+=['',f'Only **{int(shared.shared_events.iloc[0])} validation events** have eligible opportunities in every interval. '
        'A shared-event evaluation sensitivity is saved for every family/model; it changes event composition without fitting or selecting a new model. '
        'Interval-upper-follow-up sensitivity for Full union, Profile and Departure is also retained. '
        'Neither sensitivity is a prospective validation.','',
        '## Matched trajectories and interpretation','',
        'Each family trajectory uses one fixed 60-day training-only logistic score, standardized using its training log-odds distribution. '
        'The score does not change depending on the retrospectively known lead interval. Original matched controls are reused. '
        'Calendar plots balance the same pairs across all four bins; outing plots balance final, 2–3 and 4–5 appearances. '
        'Available-pair summaries for 6–10 and 11+ outings are separate and never treated as the same calendar durations. '
        'Bootstrap clusters connect events sharing either case or control pitchers.','']
    q=shapes.loc[shapes.partition.eq('validation')]
    for mode in ['calendar','outings']:
        v=q.loc[q['mode'].eq(mode)]
        significant_shape=v.loc[v.ci_low.gt(0)|v.ci_high.lt(0)]
        if len(v):
            lines.append(f'- {mode}: {int(v.events.min())}–{int(v.events.max())} balanced events per family; '
                         f'{len(significant_shape)}/{len(v)} unadjusted family difference-change intervals exclude zero. '
                         'These exploratory trajectory contrasts are not corrected across families.')
    lines+=['',
        'Persistent case-control offsets are profile associations, not proof of an approaching injury. '
        'Score direction combines the family’s current levels, historical levels and departures; it is not a physiological latent construct. '
        'Sparse balanced support and the same previously examined cohorts limit inference about shape.','',
        'Velocity-family matched difference changes are near zero: +0.027 training-score SD in calendar alignment '
        '(95% interval −0.224 to +0.331), and +0.017 SD in outing alignment (−0.203 to +0.236). '
        'Workload-family case-control scores decrease toward the event in both alignments (−0.404 and −0.448 SD; '
        'unadjusted intervals below zero). This is not evidence of a general rise in workload risk near injury; '
        'the score blends workload level, rest and changes, and only seven calendar-balanced events contribute.','',
        '## Transfer, warning and next step','',
        'The development family choices, model signatures, interval definitions and contrast results were saved before scoring 2024–2025. '
        'All frozen families and null findings were transferred; transfer cannot redefine a candidate or claim independent confirmation. '
        'Warning comparisons use retrospective top-5% appearance budgets for each interval, not deployable thresholds. '
        'The 1–7-day endpoint is more proximal than the existing ≥7-day practical-warning objective.','',
        'Departure-8 PR lift is **2.26, 1.91, 1.11, 1.31** across validation intervals and '
        '**1.74, 1.07, 1.26, 1.47** in exploratory transfer. Its near-minus-far validation contrast is +0.95 '
        '(95% interval −0.52 to +2.94), so stronger near-term discrimination remains uncertain. '
        'Profile’s 31–60-day lift drops from **3.51** in validation to **0.90** in transfer. '
        'At the 5% budget, Departure-8 warns on 9/34 validation events at 1–7 days but 4/43 transfer events; '
        'the corresponding 8–14-day figures are 4/34 and 1/43. '
        'Broader models have stronger transfer ranking in some intervals, but this does not validate the proposed temporal family ordering.','',
        '**The temporal heterogeneity hypothesis remains unconfirmed in this first pass.** '
        'Use the nominated families only for narrow, prespecified replication questions. Do not launch flexible lag searches or freeze an acute/intermediate/chronic physiological narrative from these heatmaps. '
        'Freeze the interval definitions and family mapping for a genuinely post-freeze chronological cohort, and improve event/timing support before escalating model complexity.','',
        'See [methods](../../docs/ucl_temporal_methods.md), [notebooks](../../notebooks/ucl_temporal/README.md), '
        '`tables/family_matrix_validation.csv`, `tables/temporal_contrasts_validation.csv`, '
        '`tables/followup_sensitivity.csv`, and `tables/shared_event_sensitivity_validation.csv`.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    guide=[
        ('01_family_standalone','Does each family rank interval outcomes on its own?','PR lift can look large when a few events dominate; consult paired uncertainty.'),
        ('02_family_incremental','What remains useful conditional on a broader feature model?','Ablation and marginal permutation answer different questions; preserve negative values.'),
        ('03_frozen_representations','Does Departure-8 strengthen near injury relative to Profile?','Different patterns across intervals and transfer remain exploratory.'),
        ('04_temporal_stability','Are family-time contrasts robust to pitcher composition?','Joint contrast intervals include zero; refits can change ordering.'),
        ('05_matched_family_trajectories','Do fixed family scores diverge before recorded injury?','Balanced matched retrospective summaries with small event counts.'),
        ('06_interval_warning','Do ranking patterns translate into interval-specific alerts?','Fixed retrospective 5% budget; sparse event detection, not a prospective alarm policy.')]
    table(pd.DataFrame(guide,columns=['figure','question','interpretation']),'figure_guide')

def run_reporting():
    if valid_checkpoint('reporting'):return
    make_figures();make_summary();validate()
    checkpoint('reporting',[OUT/'summary.md',OUT/'tables/figure_guide.csv',
        OUT/'tables/validation_checks.csv']+list((OUT/'figures').glob('*')))
