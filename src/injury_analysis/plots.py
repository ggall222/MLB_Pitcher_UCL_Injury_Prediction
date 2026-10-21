"""Selective figures shared by the notebooks; full numerical results stay in CSVs."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common import CATEGORIES, LABELS, ARM, NONARM, figure, style, read_table

style()
COLORS = dict(zip(CATEGORIES, ['#236192','#00897b','#d28128','#8561a9','#5d6d7e','#bb5566','#a27c26']))


def overview_figure(events):
    counts = events.groupby('analysis_group').agg(All=('injury_event_id','size'), Eligible=('target_eligible','sum')).sort_values('All')
    annual = events.groupby('season').agg(All=('injury_event_id','size'), Eligible=('target_eligible','sum'))
    fig, axes = plt.subplots(1,2,figsize=(12,4.7))
    counts.plot.barh(ax=axes[0],color=['#c5cbd3','#236192'])
    axes[0].set(xlabel='Documented IL episodes',ylabel='',title='Which categories support comparison?')
    annual.plot.bar(ax=axes[1],color=['#c5cbd3','#236192'])
    axes[1].set(xlabel='Year (2026 incomplete)',ylabel='Episodes',title='Timing/anatomy exclusions by year')
    return figure(fig,'01_cohort_overview')


def missingness_figure():
    f = read_table('feature_availability_by_season')
    selected = ['avg_fb_velo','velo_drop_game','release_z_sd','extension_avg','spin_avg','arm_angle_avg','arm_angle_z','velo_z']
    matrix = f[f.feature.isin(selected)].pivot(index='feature',columns='season',values='available_fraction').reindex(selected)
    fig, ax = plt.subplots(figsize=(10,3.8))
    im = ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set(xticks=np.arange(len(matrix.columns)),xticklabels=matrix.columns,
           yticks=np.arange(len(matrix)),yticklabels=matrix.index,title='Feature availability across the observed pitching population')
    ax.grid(False)
    fig.colorbar(im,ax=ax,label='Nonmissing fraction')
    return figure(fig,'01_feature_coverage')


def trajectory_figure(mode='calendar'):
    name = 'calendar_trajectory_summary' if mode=='calendar' else 'appearance_trajectory_summary'
    f = read_table(name)
    features = ['velo_z','release_z_sd_z','extension_z','workload_acceleration_7d_vs_21d']
    groups = ['Elbow / UCL','Shoulder','Forearm','Lower body']
    fig, axes = plt.subplots(2,2,figsize=(12,7))
    for ax, feature in zip(axes.flat,features):
        for group in groups:
            for kind in ['Injury','Control']:
                s = f[(f.feature==feature)&(f.analysis_group==group)&(f.kind==kind)]
                if mode=='calendar':
                    order=['31–60 days','15–30 days','8–14 days','1–7 days']
                    s=s.set_index('window').reindex(order)
                    x=np.arange(4)
                    ax.set_xticks(x, ['60–31','30–15','14–8','7–1'])
                else:
                    s=s.sort_values('appearance_before',ascending=False)
                    x=-s.appearance_before.to_numpy()
                ax.plot(x,s['mean'],color=COLORS[group],ls='-' if kind=='Injury' else '--',
                        lw=1.8 if kind=='Injury' else 1,alpha=1 if kind=='Injury' else .55,
                        label=group if kind=='Injury' else None)
                if kind=='Injury': ax.fill_between(x,s.low,s.high,color=COLORS[group],alpha=.09)
        ax.set_title(LABELS[feature])
        ax.set_xlabel('Days before index date' if mode=='calendar' else 'Outings before index date')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched pre-injury trajectories: solid = injury, dashed = controls; pitcher-bootstrap 95% intervals',fontsize=11)
    return figure(fig,'02_'+mode+'_trajectories')


def effects_figure():
    f=read_table('paired_preinjury_effects')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    for ax,feature in zip(axes,['velo_z','release_z_sd_z','extension_z']):
        s=f[f.feature==feature].set_index('analysis_group').reindex(CATEGORIES)
        x=s['mean'].to_numpy(); y=np.arange(len(s))
        ax.errorbar(x,y,xerr=np.vstack([np.maximum(x-s.low,0),np.maximum(s.high-x,0)]),fmt='o',color='#236192',capsize=3)
        ax.axvline(0,color='grey',lw=1)
        ax.set(yticks=y,yticklabels=s.index if ax==axes[0] else [],xlabel='Injury minus matched control',title=LABELS[feature])
    return figure(fig,'02_matched_effects')


def prediction_figure():
    m=read_table('prediction_metrics').query("partition == 'test' and horizon == 30").sort_values('pr_lift')
    pr=read_table('precision_recall_curves')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    y=np.arange(len(m)); val=m.pr_lift.to_numpy()
    axes[0].errorbar(val,y,xerr=np.vstack([np.maximum(val-m.lift_low,0),np.maximum(m.lift_high-val,0)]),fmt='o',capsize=3,color='#236192')
    axes[0].axvline(1,color='grey',ls='--')
    axes[0].set(yticks=y,yticklabels=m.category,xlabel='Average precision / prevalence',title='30-day test PR lift (pitcher-bootstrap interval)')
    for category in CATEGORIES:
        f=pr.loc[pr.category.eq(category)]
        axes[1].plot(f.recall,f.precision,label=category,color=COLORS[category])
    axes[1].set(xlabel='Recall',ylabel='Precision',ylim=(0,.5),title='2024–2025 precision–recall (display clipped at 0.5)')
    axes[1].legend(fontsize=8)
    return figure(fig,'03_predictability')


def severity_figure():
    counts=read_table('severity_duration_groups')
    mat=counts.pivot(index='analysis_group',columns='duration_group',values='events').fillna(0).reindex(CATEGORIES)
    fig,ax=plt.subplots(figsize=(10,4.8))
    mat.div(mat.sum(axis=1),axis=0).plot.barh(stacked=True,ax=ax,color=['#9fb8ca','#548bad','#184f71','#d4d4d4'])
    ax.set(xlabel='Fraction of eligible episodes',ylabel='',title='Observed IL absence: unresolved outcomes remain visible')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    return figure(fig,'04_severity_followup')


def recurrence_figure(transitions):
    mat=transitions.reindex(index=CATEGORIES,columns=CATEGORIES,fill_value=0)
    frac=mat.div(mat.sum(axis=1),axis=0)
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(frac,cmap='Blues',vmin=0,vmax=.6)
    for i in range(len(mat)):
        for j in range(len(mat)):
            ax.text(j,i,f'{mat.iloc[i,j]}',ha='center',va='center',color='white' if frac.iloc[i,j]>.35 else '#162a3a',fontsize=9)
    ax.set(xticks=np.arange(len(mat)),yticks=np.arange(len(mat)),xticklabels=mat.columns,yticklabels=mat.index,
        xlabel='Next qualifying episode',ylabel='Previous qualifying episode',title='Injury transitions after observed intervening MLB pitching')
    ax.tick_params(axis='x',rotation=35);ax.grid(False)
    fig.colorbar(im,ax=ax,label='Within-prior-category fraction (cells show counts)')
    return figure(fig,'05_recurrence_transitions')


def warning_figure():
    f=read_table('alert_budget_metrics').query('horizon == 30')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for category,g in f.groupby('category'):
        g=g.sort_values('alert_rate')
        axes[0].plot(g.alert_rate*100,g.injury_event_recall*100,'o-',label=category,color=COLORS[category])
        axes[1].plot(g.alert_rate*100,g.false_alerts_per_pitcher_season,'o-',color=COLORS[category])
    axes[0].set(xlabel='Retrospective alert budget (% eligible outings)',ylabel='All qualifying events detected (%)',title='Warnings 7–30 days before the episode')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Retrospective alert budget (%)',ylabel='False alerts / represented pitcher-season',title='False-alert burden (no cooldown)')
    return figure(fig,'06_warning_budgets')


def risk_trajectory_figure():
    f=read_table('event_risk_trajectories')
    fig,ax=plt.subplots(figsize=(8,4.5))
    bins=['31–60','15–30','8–14','1–7']
    for group in ['Elbow / UCL','Shoulder','Forearm','Lower body']:
        s=f[f.category==group].set_index('lead_bin').reindex(bins)
        x=np.arange(4)
        ax.plot(x,s['mean'],label=group,color=COLORS[group])
        ax.fill_between(x,s.low,s.high,color=COLORS[group],alpha=.1)
    ax.set(xticks=x,xticklabels=bins,xlabel='Days before event',ylabel='Mean 30-day model score',title='Event-weighted risk scores as documented injury approaches')
    ax.legend(fontsize=8)
    return figure(fig,'06_risk_trajectories')


def family_figure():
    f=read_table('feature_family_ablation')
    mat=f[~f.variant.eq('All selected')].pivot(index='variant',columns='category',values='ap_change_vs_all')
    fig,ax=plt.subplots(figsize=(8,5.3))
    bound=max(.001,abs(mat.to_numpy()).max())
    im=ax.imshow(mat,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    for i in range(len(mat)):
        for j in range(len(mat.columns)):
            ax.text(j,i,f'{mat.iloc[i,j]:+.3f}',ha='center',va='center',fontsize=9)
    ax.set(xticks=np.arange(len(mat.columns)),xticklabels=mat.columns,yticks=np.arange(len(mat)),yticklabels=mat.index,
           title='Validation-only changes in average precision')
    ax.grid(False)
    fig.colorbar(im,ax=ax,label='AP change vs all selected features')
    return figure(fig,'07_feature_families')
