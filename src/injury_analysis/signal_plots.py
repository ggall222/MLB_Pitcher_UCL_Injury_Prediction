"""Selective comparison figures with visible event support and unsmoothed points."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common import style
from .signals_common import CFG,CACHE,LABELS,read_table,figure
from .plots import COLORS

style()


def family_overview():
    f=read_table('family_performance')
    order=['pitcher_history','workload','velocity','release_mechanics','extension','spin_movement','command','pitch_mix',
        'Workload + history','WH + velocity','WH + mechanics','WH + velocity + mechanics','Statcast only','Full']
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    for ax,part in zip(axes,['validation','test']):
        mat=f.loc[f.partition.eq(part)].pivot(index='variant',columns='category',values='pr_lift').reindex(index=order,columns=CFG['categories'])
        im=ax.imshow(mat,cmap='YlGnBu',vmin=1,vmax=3,aspect='auto')
        for i in range(len(mat)):
            for j in range(len(mat.columns)):
                ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if mat.iloc[i,j]>2.5 else '#142331')
        ax.set(xticks=np.arange(len(mat.columns)),xticklabels=mat.columns,yticks=np.arange(len(mat)),
            yticklabels=mat.index if part=='validation' else [],title='Validation 2022–2023' if part=='validation' else 'Examined test period 2024–2025')
        ax.tick_params(axis='x',rotation=55);ax.grid(False)
    fig.colorbar(im,ax=axes,label='Average precision / prevalence (color clipped at 1–3)')
    return figure(fig,'01_family_pr_lift')


def incremental_effects():
    f=read_table('family_performance')
    variants=['WH + velocity','WH + mechanics','WH + velocity + mechanics','Full']
    fig,axes=plt.subplots(1,2,figsize=(13,6.5))
    for ax,part in zip(axes,['validation','test']):
        for j,name in enumerate(variants):
            s=f.loc[f.partition.eq(part)&f.variant.eq(name)].set_index('category').reindex(CFG['categories'])
            x=s.ap_delta.to_numpy()*100;y=np.arange(len(s))+(j-1.5)*.16
            ax.errorbar(x,y,xerr=np.vstack([np.maximum(x-s.delta_low*100,0),np.maximum(s.delta_high*100-x,0)]),fmt='o',ms=4,capsize=2,label=name)
        ax.axvline(0,color='grey',ls='--')
        ax.set(yticks=np.arange(len(CFG['categories'])),yticklabels=CFG['categories'] if part=='validation' else [],
            xlabel='AP difference vs workload + history (percentage points)',title=part.title()+'; paired pitcher-bootstrap intervals')
    axes[0].legend(fontsize=8,loc='lower left')
    return figure(fig,'01_incremental_statcast_value')


def alert_comparison():
    f=read_table('family_alert_budgets').query("partition == 'test'")
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,category in zip(axes.flat,CFG['focus_categories']):
        s=f.loc[f.category.eq(category)]
        for name in ['Workload + history','WH + velocity + mechanics','Full']:
            g=s.loc[s.variant.eq(name)].sort_values('alert_rate')
            ax.plot(g.alert_rate*100,g.injury_event_recall*100,'o-',label=name)
        g=s.loc[s.variant.eq('Full')].sort_values('alert_rate')
        ax.plot(g.alert_rate*100,g.random_expected_event_recall*100,'k--',label='Uniform random expectation')
        ax.set(title=category+f' (all events n={int(g.qualifying_events.iloc[0])})',
            xlabel='Retrospective alert budget (%)',ylabel='Events warned 7–30 days ahead (%)')
    axes[0,0].legend(fontsize=8)
    return figure(fig,'01_event_warning_vs_random')


def importance_figure():
    f=read_table('variable_permutation_summary')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,category in zip(axes.flat,CFG['focus_categories']):
        s=f.loc[f.category.eq(category)].nlargest(6,'mean').sort_values('mean')
        ax.barh(s.feature,s['mean']*100,color=COLORS[category])
        ax.errorbar(s['mean']*100,np.arange(len(s)),xerr=s['std']*100,fmt='none',color='#343434',capsize=2)
        ax.set(title=category,xlabel='Validation AP loss after permutation (pp)')
        ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Variables in the top two validation family-only models; bars show shuffle SD, not sampling CI',fontsize=10)
    return figure(fig,'01_variable_inspection')


def case_control_trajectories(mode='calendar',feature='velo_z'):
    f=read_table('balanced_trajectory_summary')
    f=f.loc[f['mode'].eq(mode)&f.feature.eq(feature)]
    labels=['31–60','15–30','8–14','1–7'] if mode=='calendar' else ['−5','−4','−3','−2','−1']
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for ax,category in zip(axes.flat,CFG['focus_categories']):
        for kind,color,ls in [('Injury',COLORS[category],'-'),('Control','#6f757b','--')]:
            s=f.loc[f.category.eq(category)&f.kind.eq(kind)].sort_values('point')
            ax.plot(s.point,s['mean'],marker='o',color=color,ls=ls,label=kind)
            ax.fill_between(s.point,s.low,s.high,color=color,alpha=.14)
        n=int(s.events.iloc[0]) if len(s) else 0
        ax.set(xticks=range(len(labels)),xticklabels=labels,title=f'{category}; same {n} pairs at every point',
            ylabel=LABELS.get(feature,feature),xlabel='Days before recorded episode' if mode=='calendar' else 'Outings before recorded episode')
    axes[0,0].legend()
    return figure(fig,f'02_{mode}_{feature}_cases_controls')


def paired_trajectory_panels(mode='appearance',modern=False):
    f=read_table('balanced_trajectory_summary')
    f=f.loc[f['mode'].eq(mode)&f.kind.eq('difference')]
    features=['velo_z','fb_velo_sd_change_vs_prev_5','release_z_sd_z','workload_acceleration_7d_vs_21d']
    if modern:features=['extension_z','arm_angle_z','spin_z','zone_pct_z']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,feature in zip(axes.flat,features):
        for category in CFG['focus_categories']:
            s=f.loc[f.feature.eq(feature)&f.category.eq(category)].sort_values('point')
            if s.empty:continue
            ax.plot(s.point,s['mean'],'o-',label=f'{category} (n={s.events.iloc[0]})',color=COLORS[category])
            ax.fill_between(s.point,s.simultaneous_low,s.simultaneous_high,color=COLORS[category],alpha=.07)
        ax.axhline(0,color='grey',lw=1)
        labels=['31–60','15–30','8–14','1–7'] if mode=='calendar' else ['−5','−4','−3','−2','−1']
        ax.set(xticks=range(len(labels)),xticklabels=labels,title=LABELS.get(feature,feature),
            ylabel='Case minus matched control',xlabel='Days before episode' if mode=='calendar' else 'Outings before episode')
        ax.legend(fontsize=7)
    fig.suptitle('Balanced pairs; simultaneous 95% bands across time within each feature/category; no smoothing',fontsize=10)
    return figure(fig,f'02_{mode}_paired_'+('modern_shape' if modern else 'core_shape'))


def shape_figure():
    f=read_table('trajectory_shape_metrics').query("feature == 'velo_z'")
    metrics=['slope_per_appearance','late_minus_early_mean','final_step_vs_previous_two']
    titles=['Linear slope across final five outings','Final two minus first two outings','Last outing minus previous two']
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,metric,title in zip(axes,metrics,titles):
        s=f.loc[f.metric.eq(metric)].set_index('category').reindex(CFG['focus_categories'])
        x=s['mean'].to_numpy();y=np.arange(len(s))
        ax.errorbar(x,y,xerr=np.vstack([np.maximum(x-s.low,0),np.maximum(s.high-x,0)]),fmt='o',capsize=3)
        ax.axvline(0,color='grey',ls='--')
        ax.set(yticks=y,yticklabels=[f'{k} (n={int(n)})' for k,n in zip(s.index,s.events)] if ax==axes[0] else [],
            title=title,xlabel='Case–control difference in velocity-z change')
    return figure(fig,'03_velocity_shape_contrasts')


def sensitivity_figure():
    f=read_table('trajectory_sensitivity').query("feature == 'velo_z'")
    conditions=['Primary','Strict QA','Exclude final 3 days','Exclude final 7 days','Stronger baseline',
        'Starter index','Reliever index','First reported episode','Prior reported episode']
    fig,axes=plt.subplots(1,2,figsize=(12,6.5))
    for ax,category in zip(axes,['Elbow / UCL','Shoulder']):
        s=f.loc[f.category.eq(category)].set_index('condition').reindex(conditions)
        x=s['mean'].to_numpy();y=np.arange(len(s))
        ax.errorbar(x,y,xerr=np.vstack([np.maximum(x-s.low,0),np.maximum(s.high-x,0)]),fmt='o',capsize=3,color=COLORS[category])
        ax.axvline(0,color='grey',ls='--')
        ax.set(yticks=y,yticklabels=[f'{k} (n={int(n)})' for k,n in zip(s.index,s.events)],
            title=category,xlabel='Velocity z-score: injury minus control, prior 14 days')
        ax.invert_yaxis()
    return figure(fig,'03_velocity_sensitivity')


def heterogeneity_figure():
    raw=pd.read_parquet(CACHE/'event_shape_metrics.parquet')
    raw=raw.loc[raw.feature.eq('velo_z')&raw.metric.eq('final_step_vs_previous_two')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    groups=CFG['focus_categories']
    vals=[raw.loc[raw.category.eq(g),'difference'].to_numpy() for g in groups]
    axes[0].boxplot(vals,tick_labels=groups,showfliers=False,vert=False)
    axes[0].axvline(0,color='grey',ls='--')
    axes[0].set(xlabel='Event-specific excess final velocity change (z-units)',title='Wide individual variation; outliers omitted from display')
    f=read_table('balanced_trajectory_summary')
    for category in ['Elbow / UCL','Shoulder','Lower body']:
        s=f.loc[f.feature.eq('avg_fb_velo_baseline_mean')&f['mode'].eq('appearance')&f.kind.eq('difference')&f.category.eq(category)].sort_values('point')
        if s.empty:continue
        axes[1].plot(s.point,s['mean'],'o-',label=f'{category} (n={int(s.events.iloc[0])})',color=COLORS[category])
        axes[1].fill_between(s.point,s.low,s.high,color=COLORS[category],alpha=.1)
    axes[1].axhline(0,color='grey',ls='--')
    axes[1].set(xticks=range(5),xticklabels=['−5','−4','−3','−2','−1'],xlabel='Outings before recorded episode',
        ylabel='Case minus control baseline fastball speed (mph)',title='Persistent pitcher-profile differences can carry signal')
    axes[1].legend(fontsize=8)
    return figure(fig,'03_heterogeneity_and_baseline_profiles')
