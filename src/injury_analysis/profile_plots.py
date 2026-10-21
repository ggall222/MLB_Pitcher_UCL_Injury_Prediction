"""Publication exports for the profile/departure comparison; no model fitting."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .profile_departure import CFG,PRIMARY,read_table,figure
from .common import CATEGORIES

COLORS={'Profile only':'#355C7D','Departure only':'#D17631','Profile + Departure':'#31856A',
        'Original 62':'#777777','Workload + history':'#8D6094'}
SHORT={'Profile only':'Profile','Departure only':'Departure','Profile + Departure':'Combined'}
FOCUS=CFG['focus_categories']


def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})


def performance_figure():
    style();f=read_table('model_performance')
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True)
    names=PRIMARY+['Original 62']
    for ax,part in zip(axes,['validation','test']):
        for j,name in enumerate(names):
            s=f.loc[f.partition.eq(part)&f.variant.eq(name)].set_index('category').reindex(CATEGORIES)
            y=np.arange(len(s))+(j-1.5)*.17
            ax.plot(s.average_precision,y,'o',ms=5,color=COLORS[name],label=name)
            if name in PRIMARY:
                ax.hlines(y,s.ap_low,s.ap_high,color=COLORS[name],lw=1,alpha=.75)
        base=f.loc[f.partition.eq(part)&f.variant.eq('Profile only')].set_index('category').reindex(CATEGORIES)
        ax.scatter(base.prevalence,np.arange(len(base)),marker='|',s=120,color='black',label='Prevalence')
        ax.set_title('Validation 2022–2023' if part=='validation' else 'Test 2024–2025')
        ax.set_xlabel('Average Precision');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(np.arange(len(CATEGORIES)),CATEGORIES)
    axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('Profile information carries much of AP; departure gains vary by injury',y=1.01)
    fig.tight_layout(rect=[0,.07,1,1])
    return figure(fig,'01_profile_departure_average_precision')


def incremental_figure():
    style();f=read_table('incremental_comparisons')
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharey=True)
    for row,(reference,variant,title) in enumerate([
        ('Profile only','Profile + Departure','Add departures to profile'),
        ('Context + Profile','Context + Profile + Departure','Add departures with workload/role/calendar held common')]):
        s=f.loc[f.reference.eq(reference)&f.variant.eq(variant)]
        for col,(value,low,high,xlabel) in enumerate([
            ('ap_difference','ap_low','ap_high','Change in Average Precision'),
            ('event_mean','event_low','event_high','Change in 7–30-day event recall at 5% alerts')]):
            ax=axes[row,col]
            for part,offset,color in [('validation',-.13,'#527A9F'),('test',.13,'#B35B36')]:
                g=s.loc[s.partition.eq(part)].set_index('category').reindex(CATEGORIES)
                y=np.arange(len(g))+offset
                ax.hlines(y,g[low],g[high],color=color,lw=1.5)
                ax.plot(g[value],y,'o',color=color,label=part.capitalize())
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(np.arange(len(CATEGORIES)),CATEGORIES)
            ax.set_title(title,fontsize=10);ax.set_xlabel(xlabel)
            ax.grid(axis='x',alpha=.15)
            if col==1:ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].invert_yaxis();axes[0,0].legend()
    fig.suptitle('Incremental signal: paired pitcher-bootstrap 95% intervals')
    fig.tight_layout(rect=[0,0,1,.97])
    return figure(fig,'02_incremental_ap_and_event_recall')


def alert_figure():
    style();f=read_table('alert_budget_performance');f=f.loc[f.partition.eq('test')]
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,cat in zip(axes.flat,FOCUS):
        for name in PRIMARY+['Workload + history']:
            s=f.loc[f.category.eq(cat)&f.variant.eq(name)].sort_values('alert_rate')
            ax.plot(100*s.alert_rate,100*s.injury_event_recall,'o-',label=name,color=COLORS[name],lw=1.5,ms=4)
        r=f.loc[f.category.eq(cat)&f.variant.eq('Profile only')].sort_values('alert_rate')
        ax.plot(100*r.alert_rate,100*r.random_expected_event_recall,'--',color='black',label='Uniform random expectation')
        ax.axvline(5,color='gray',ls=':',alpha=.7)
        ax.set_title(f'{cat} ({int(r.qualifying_events.iloc[0])} events)')
        ax.set_xlabel('Alert budget (% of eligible outings)');ax.set_ylabel('7–30-day event recall (%)')
        ax.grid(alpha=.15);ax.set_xticks([1,2.5,5,10])
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Test event warning: improvement over profile is not necessarily better than random')
    fig.tight_layout(rect=[0,.09,1,.96])
    return figure(fig,'03_event_recall_alert_budgets')


def calibration_figure():
    style();f=read_table('reliability_bins')
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex='col',sharey='col')
    for j,cat in enumerate(FOCUS):
        for i,part in enumerate(['validation','test']):
            ax=axes[i,j];largest=.06
            for name in ['Profile only','Profile + Departure']:
                s=f.loc[f.category.eq(cat)&f.partition.eq(part)&f.variant.eq(name)]
                ax.plot(s.mean_prediction,s.observed_rate,'o-',ms=4,color=COLORS[name],label=SHORT[name])
                largest=max(largest,s.mean_prediction.max()*1.05,s.observed_rate.max()*1.05)
            ax.plot([0,largest],[0,largest],ls='--',lw=1,color='gray')
            ax.set_title(cat if i==0 else 'Test 2024–2025')
            if j==0:ax.set_ylabel(('Validation 2022–2023\n' if i==0 else '')+'Observed rate')
            if i==1:ax.set_xlabel('Mean predicted risk')
            ax.xaxis.set_major_formatter(PercentFormatter(1,0));ax.yaxis.set_major_formatter(PercentFormatter(1,0))
            ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Reliability within the selected analysis cohort; validation-defined bin edges')
    fig.tight_layout(rect=[0,0,1,.96])
    return figure(fig,'04_probability_reliability')


def score_trajectory_figure(mode):
    style();f=read_table('score_trajectories')
    f=f.loc[f['mode'].eq(mode)&f.kind.eq('difference')]
    labels=['5th last','4th last','3rd last','2nd last','Last'] if mode=='appearance' else ['31–60','15–30','8–14','1–7']
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey='row')
    for j,cat in enumerate(FOCUS):
        for i,(feature,color,title) in enumerate([('profile_score',COLORS['Profile only'],'Profile term'),
                                                ('departure_score',COLORS['Departure only'],'Departure term')]):
            ax=axes[i,j];s=f.loc[f.category.eq(cat)&f.feature.eq(feature)].sort_values('point')
            ax.plot(s.point,s['mean'],'o-',color=color,lw=1.6)
            ax.fill_between(s.point,s.simultaneous_low,s.simultaneous_high,color=color,alpha=.2)
            ax.axhline(0,color='gray',lw=1,ls='--')
            ax.set_title(f'{cat}\n{int(s.events.iloc[0])} balanced pairs',fontsize=10)
            ax.set_xticks(np.arange(len(labels)),labels,rotation=20 if mode=='appearance' else 0)
            if j==0:ax.set_ylabel(title+'\ncase − control log-odds contribution')
            if i==1:ax.set_xlabel('Outing before recorded event' if mode=='appearance' else 'Days before recorded event')
            ax.grid(alpha=.15)
    fig.suptitle('Frozen combined model: persistent score offset versus evolving score separation')
    fig.tight_layout(rect=[0,0,1,.95])
    return figure(fig,'05_score_trajectories_'+mode)


def robustness_figure():
    style();f=read_table('sensitivity_increment');primary=read_table('incremental_comparisons')
    primary=primary.loc[primary.variant.eq('Profile + Departure')&primary.reference.eq('Profile only')]
    import pandas as pd
    f=pd.concat([primary,f],ignore_index=True)
    conditions=['Primary','Complete P+D measurements','Shared availability flags','Exclude final six days','Measurement profiles only']
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for col,cat in enumerate(FOCUS):
        for row,(value,low,high,label) in enumerate([
            ('ap_difference','ap_low','ap_high','Change in AP'),
            ('event_mean','event_low','event_high','Change in event recall at 5%')]):
            ax=axes[row,col]
            for part,offset,color in [('validation',-.13,'#527A9F'),('test',.13,'#B35B36')]:
                s=f.loc[f.category.eq(cat)&f.partition.eq(part)].set_index('analysis').reindex(conditions)
                y=np.arange(len(s))+offset
                ax.hlines(y,s[low],s[high],color=color)
                ax.plot(s[value],y,'o',color=color,ms=4,label=part.capitalize())
            ax.axvline(0,color='gray',ls='--');ax.set_title(cat);ax.set_xlabel(label)
            ax.set_yticks(np.arange(len(conditions)),conditions);ax.grid(axis='x',alpha=.15)
            if row==1:ax.xaxis.set_major_formatter(PercentFormatter(1,0))
    axes[0,0].invert_yaxis();axes[0,0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Does adding departures survive missingness, timing and history checks?')
    fig.tight_layout(rect=[0,0,1,.96])
    return figure(fig,'06_sensitivity_increment')


def stability_figure():
    style();f=read_table('stability_summary');f=f.loc[f.partition.eq('test')]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8))
    for ax,metric,title in zip(axes,['median_rank_correlation','median_alert_jaccard'],
                              ['Score-rank correlation','Top-5% alert-set overlap (Jaccard)']):
        for offset,name in [(-.16,'Profile only'),(.16,'Profile + Departure')]:
            s=f.loc[f.variant.eq(name)].set_index('category').reindex(FOCUS)
            ax.bar(np.arange(len(s))+offset,s[metric],.3,color=COLORS[name],label=SHORT[name])
        ax.set_xticks(np.arange(len(FOCUS)),FOCUS,rotation=15);ax.set_ylim(0,1)
        ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axes[0].legend();fig.suptitle('Median stability over 12 paired whole-pitcher training refits')
    fig.tight_layout(rect=[0,0,1,.94])
    return figure(fig,'07_training_refit_stability')

