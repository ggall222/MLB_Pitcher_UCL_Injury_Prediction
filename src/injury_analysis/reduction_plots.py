"""Exportable PCA figures with explicit units, cohorts and sample support."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common import style
from .plots import COLORS
from .reduction import CFG,CACHE,read_table,figure
from .signals_common import mean_ci

style()


def spectrum_figure():
    s=read_table('variance_spectrum');r=read_table('redundancy_summary').set_index('block')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,(block,g) in zip(axes,s.groupby('block',sort=False)):
        ax.plot(g.component,g.cumulative_fraction*100,'o-',ms=3)
        ax.axhline(90,color='grey',ls='--')
        ax.axvline(r.loc[block,'components_90'],color='grey',ls=':')
        ax.set(xlabel='Retained principal components',ylabel='Training variance retained (%)',ylim=(0,103),
               title=f'{block}\n{int(r.loc[block,"components_90"])}/{int(r.loc[block,"features"])} dimensions for 90%')
    fig.suptitle('Training through 2021 only; median-imputed, standardized measurements; no injury labels used',fontsize=10)
    return figure(fig,'01_variance_retained')


def loading_figure():
    f=read_table('component_loadings').query('component <= 3')
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,(block,g) in zip(axes,f.groupby('block',sort=False)):
        names=g.feature.unique().tolist()
        if len(names)>12:
            names=list(dict.fromkeys(c for _,h in g.groupby('component') for c in h.loc[h.weight.abs().nlargest(4).index,'feature']))
        mat=g.pivot(index='feature',columns='component',values='feature_score_correlation').reindex(names)
        short=[n.replace('_baseline_mean','').replace('_change_vs_prev_5',' Δ5').replace('_acceleration_7d_vs_21d',' acceleration') for n in names]
        im=ax.imshow(mat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        for i in range(len(mat)):
            for j in range(3):
                ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,
                        color='white' if abs(mat.iloc[i,j])>.65 else '#252525')
        ax.set(xticks=range(3),xticklabels=['PC1','PC2','PC3'],yticks=range(len(names)),yticklabels=short,title=block)
        ax.tick_params(axis='y',labelsize=8);ax.grid(False)
    fig.colorbar(im,ax=axes,label='Training correlation: original feature versus PC score',shrink=.6)
    return figure(fig,'01_component_interpretation')


def persistence_figure():
    f=read_table('component_persistence')
    fig,ax=plt.subplots(figsize=(10,4.7))
    names=f.block+' PC'+f.component.astype(str)
    colors=[{'Current feature space':'#6f757b','Prior pitcher profile':'#226b99','Within-pitcher departure':'#00897b'}[g] for g in f.block]
    ax.barh(names,f.between_pitcher_fraction*100,color=colors)
    ax.set(xlim=(0,100),xlabel='Between-pitcher share of training score variation (%)',
           title='Prior profiles vary between pitchers; baseline departures vary within pitchers')
    ax.invert_yaxis()
    return figure(fig,'01_between_within_pitcher_variation')


def latent_trajectory_figure(mode='appearance'):
    f=read_table('latent_balanced_trajectories')
    f=f.loc[f.cohort.eq('Heldout 2022–2025')&f['mode'].eq(mode)&f.kind.eq('difference')]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    labels=['−5','−4','−3','−2','−1'] if mode=='appearance' else ['31–60','15–30','8–14','1–7']
    for row,prefix in enumerate(['profile','departure']):
        for j in range(1,4):
            ax=axes[row,j-1]
            for category in CFG['focus_categories']:
                g=f.loc[f.feature.eq(f'{prefix}_PC{j}')&f.category.eq(category)].sort_values('point')
                if g.empty:continue
                ax.plot(g.point,g['mean'],'o-',ms=4,color=COLORS[category],label=f'{category} (n={int(g.events.iloc[0])})')
                ax.fill_between(g.point,g.simultaneous_low,g.simultaneous_high,color=COLORS[category],alpha=.08)
            ax.axhline(0,color='grey',lw=1)
            ax.set(xticks=range(len(labels)),xticklabels=labels,title=f'{prefix.title()} PC{j}',
                   xlabel='Outings before recorded episode' if mode=='appearance' else 'Days before recorded episode',
                   ylabel='Case − control (training PC SD)')
            ax.legend(fontsize=6.7)
    fig.suptitle('Frozen training PCA; 2022–2025 matched pairs; same events at every point; simultaneous time bands',fontsize=10)
    return figure(fig,f'02_{mode}_latent_trajectories')


def latent_map_figure():
    f=pd.read_parquet(CACHE/'latent_event_cells.parquet')
    f=f.loc[f.cohort.eq('Heldout 2022–2025')&f['mode'].eq('appearance')]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,prefix in zip(axes,['profile','departure']):
        g=f.loc[f.feature.isin([f'{prefix}_PC1',f'{prefix}_PC2'])]
        mat=g.pivot(index=['event_id','analysis_group','cluster_pitcher_id','point'],columns='feature',values='difference').dropna().reset_index()
        for category in CFG['focus_categories']:
            h=mat.loc[mat.analysis_group.eq(category)]
            near=h.loc[h.point.eq(4)]
            ax.scatter(near[f'{prefix}_PC1'],near[f'{prefix}_PC2'],s=12,alpha=.10,color=COLORS[category])
            early=h.loc[h.point.eq(0)]
            xm,ym=early[f'{prefix}_PC1'].mean(),early[f'{prefix}_PC2'].mean()
            xn,yn=near[f'{prefix}_PC1'].mean(),near[f'{prefix}_PC2'].mean()
            ax.plot(xm,ym,'o',mfc='white',mec=COLORS[category],ms=6)
            ax.annotate('',xy=(xn,yn),xytext=(xm,ym),arrowprops=dict(arrowstyle='->',color=COLORS[category],lw=2))
            ax.plot(xn,yn,'D',ms=5,color=COLORS[category],label=f'{category} (n={len(near)})')
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7)
        ax.set(xlabel='PC1 case − matched control (training SD)',ylabel='PC2 case − matched control (training SD)',
               title=f'{prefix.title()}: group means and event overlap')
        ax.legend(fontsize=7)
    fig.suptitle('Heldout 2022–2025: arrows from fifth-last to last outing; faint points = individual final pair differences',fontsize=10)
    return figure(fig,'02_latent_space_overlap')


def prediction_figure():
    f=read_table('predictive_performance');sel=read_table('validation_dimension_selection').set_index('category').selected_variant
    categories=f.category.unique()
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,part in zip(axes,['validation','test']):
        for j,name in enumerate(['Selected global PCA','WH + Statcast PCA 10','PCA 90%']):
            rows=[]
            for category in categories:
                variant=sel[category] if name=='Selected global PCA' else name
                rows.append(f.loc[f.category.eq(category)&f.partition.eq(part)&f.variant.eq(variant)].iloc[0])
            s=pd.DataFrame(rows);x=s.ap_delta_original.to_numpy()*100;y=np.arange(len(s))+(j-1)*.2
            ax.errorbar(x,y,xerr=np.vstack([np.maximum(x-s.delta_low_original*100,0),np.maximum(s.delta_high_original*100-x,0)]),fmt='o',capsize=2,label=name)
        ax.axvline(0,color='grey',ls='--')
        ax.set(yticks=range(len(categories)),yticklabels=categories if part=='validation' else [],
               xlabel='AP difference from original 62 features (percentage points)',title=part.title())
    axes[0].legend(fontsize=8)
    fig.suptitle('Paired pitcher-bootstrap intervals; global dimension chosen on validation only; intervals ignore selection',fontsize=10)
    return figure(fig,'03_predictive_increment')


def alert_figure():
    f=read_table('alert_budget_performance').query("partition == 'test'")
    selected=read_table('validation_dimension_selection').set_index('category').selected_variant
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,category in zip(axes.flat,CFG['focus_categories']):
        for name in ['Original 62','Workload + history',selected[category]]:
            s=f.loc[f.category.eq(category)&f.variant.eq(name)].sort_values('alert_rate')
            ax.plot(s.alert_rate*100,s.injury_event_recall*100,'o-',label=name)
        ax.plot(s.alert_rate*100,s.random_expected_event_recall*100,'k--',label='Random expectation')
        ax.set(title=category+f' (all events n={int(s.qualifying_events.iloc[0])})',xlabel='Retrospective appearance alert budget (%)',
               ylabel='Events warned 7–30 days ahead (%)')
        ax.legend(fontsize=7)
    return figure(fig,'03_event_warning_recall')


def stability_figure():
    f=read_table('training_refit_summary')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,col,title in zip(axes,['score_rank_correlation_mean','alert_jaccard_mean'],
                           ['Score-rank stability','Top-5% alert-set stability']):
        for j,kind in enumerate(['Original 62','Selected PCA']):
            s=f.loc[f.variant.eq('Original 62') if kind=='Original 62' else ~f.variant.eq('Original 62')].set_index('category').reindex(CFG['focus_categories'])
            ax.bar(np.arange(len(s))+(j-.5)*.34,s[col],width=.34,label=kind)
        ax.set(xticks=range(len(s)),xticklabels=CFG['focus_categories'],ylim=(0,1),title=title,
               ylabel='Mean similarity to original fitted model')
        ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8)
    fig.suptitle(f'{CFG["pitcher_bootstrap_refits"]} paired whole-pitcher training refits; all preprocessing refitted; exploratory stability diagnostic',fontsize=10)
    return figure(fig,'03_training_refit_stability')
