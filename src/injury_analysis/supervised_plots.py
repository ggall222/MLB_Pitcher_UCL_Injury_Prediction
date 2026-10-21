"""Figures for supervised versus unsupervised compression."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .supervised_reduction import read_table,figure
from .common import CATEGORIES
COLORS={'PLS':'#246C85','PCA':'#D27B38','Unreduced':'#555555'}
FOCUS=['Elbow / UCL','Shoulder','Forearm','Lower body']


def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})


def curve_figure(block='Original 62'):
    style();f=read_table('component_curves');f=f.loc[f.block.eq(block)&f.partition.eq('validation')]
    selection=read_table('validation_selection')
    fig,axes=plt.subplots(2,4,figsize=(15,7.3))
    for ax,cat in zip(axes.flat,CATEGORIES):
        for method in ['PLS','PCA']:
            s=f.loc[f.category.eq(cat)&f.method.eq(method)].sort_values('components')
            ax.plot(s.components,s.average_precision,'o-',ms=4,color=COLORS[method],label=method)
            k=selection.loc[selection.category.eq(cat)&selection.block.eq(block)&selection.method.eq(method),'selected_components'].iloc[0]
            selected=s.loc[s.components.eq(k)].iloc[0]
            ax.scatter([k],[selected.average_precision],s=70,marker='s',facecolors='none',edgecolors=COLORS[method],zorder=5)
        raw=f.loc[f.category.eq(cat)&f.method.eq('Unreduced')].iloc[0]
        availability=f.loc[f.category.eq(cat)&f.method.eq('Context/availability only')].iloc[0]
        ax.axhline(raw.average_precision,ls='--',color=COLORS['Unreduced'],label='Unreduced')
        ax.axhline(availability.average_precision,ls=':',color='#8A6699',label='Context/availability only')
        ax.axhline(raw.prevalence,ls=':',lw=.8,color='black',label='Prevalence')
        ax.set_title(cat);ax.set_xlabel('Retained continuous axes');ax.set_ylabel('Validation AP')
        ax.grid(alpha=.15)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle(f'{block}: complexity selected on validation only (squares mark selected k)')
    fig.tight_layout(rect=[0,0,1,.96])
    return figure(fig,'01_validation_component_curves_'+block.replace(' ','_'))


def comparison_figure():
    style();f=read_table('selected_differences');f=f.loc[f.block.eq('Original 62')&f.variant.eq('PLS selected')]
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharey=True)
    for row,reference in enumerate(['Unreduced','PCA selected']):
        for col,(value,low,high,label) in enumerate([
            ('ap_difference','ap_low','ap_high','Change in AP'),
            ('event_mean','event_low','event_high','Change in 7–30-day event recall at 5% alerts')]):
            ax=axes[row,col]
            for part,offset,color in [('validation',-.13,'#527A9F'),('test',.13,'#B45D37')]:
                s=f.loc[f.reference.eq(reference)&f.partition.eq(part)].set_index('category').reindex(CATEGORIES)
                y=np.arange(len(s))+offset
                ax.hlines(y,s[low],s[high],color=color,lw=1.5);ax.plot(s[value],y,'o',color=color,label=part.capitalize())
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(np.arange(len(CATEGORIES)),CATEGORIES)
            ax.set_title('Selected PLS minus '+reference);ax.set_xlabel(label);ax.grid(axis='x',alpha=.15)
            if col==1:ax.xaxis.set_major_formatter(PercentFormatter(1,0))
    axes[0,0].invert_yaxis();axes[0,0].legend()
    fig.suptitle('Does supervised compression improve discrimination or practical warning?')
    fig.tight_layout(rect=[0,0,1,.97])
    return figure(fig,'02_selected_pls_increment')


def compression_figure():
    style();s=read_table('validation_selection')
    blocks=['Original 62','Profile','Departure','Combined']
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,method in zip(axes,['PCA','PLS']):
        g=s.loc[s.method.eq(method)].pivot(index='category',columns='block',values='selected_components').reindex(index=CATEGORIES,columns=blocks)
        denom=s.loc[s.method.eq(method)].pivot(index='category',columns='block',values='continuous_features').reindex(index=CATEGORIES,columns=blocks)
        ax.imshow(g/denom,cmap='Blues',vmin=0,vmax=1,aspect='auto')
        for i in range(len(g)):
            for j in range(len(blocks)):
                ratio=g.iloc[i,j]/denom.iloc[i,j]
                ax.text(j,i,f'{int(g.iloc[i,j])}/{int(denom.iloc[i,j])}',ha='center',va='center',
                        color='white' if ratio>.65 else 'black')
        ax.set_xticks(range(4),blocks,rotation=20);ax.set_yticks(range(len(g)),CATEGORIES);ax.set_title(method)
    fig.suptitle('Selected latent dimensions / continuous source fields')
    fig.text(.5,.015,'Binary context and missing flags remain outside the projection: 47 / 10 / 8 / 18 additional columns, respectively.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94])
    return figure(fig,'03_compression_by_representation')


def loadings_figure():
    style();f=read_table('component_loadings')
    f=f.loc[f.category.eq('Shoulder')&f.method.eq('PLS')&f.component.eq(1)]
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,block in zip(axes.flat,['Original 62','Profile','Departure','Combined']):
        s=f.loc[f.block.eq(block)].copy()
        s=s.loc[s.axis_weight.abs().nlargest(10).index].sort_values('injury_oriented_axis_weight')
        ax.barh(s.feature,s.injury_oriented_axis_weight,color=np.where(s.injury_oriented_axis_weight>=0,'#246C85','#BD7140'))
        ax.axvline(0,color='gray',lw=.7);ax.set_title(block+' — first supervised axis')
        ax.set_xlabel('Weight in standardized source-feature coordinates');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Shoulder PLS1: signs oriented toward higher injury outcome using training only')
    fig.tight_layout(rect=[0,0,1,.96])
    return figure(fig,'04_shoulder_pls_loadings')


def family_figure():
    style();f=read_table('component_family_weights')
    fig,axes=plt.subplots(1,2,figsize=(13,5.7))
    s=f.loc[f.block.eq('Original 62')&f.method.eq('PLS')&f.component.eq(1)].pivot(
        index='category',columns='family',values='squared_axis_weight').fillna(0).reindex(CATEGORIES)
    im=axes[0].imshow(s,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    axes[0].set_xticks(range(len(s.columns)),[c.replace('_',' ') for c in s.columns],rotation=35,ha='right')
    axes[0].set_yticks(range(len(s)),s.index);axes[0].set_title('Original representation: PLS1 family weights')
    fig.colorbar(im,ax=axes[0],shrink=.75,label='Squared axis-weight sum')
    g=f.loc[f.block.eq('Combined')&f.method.eq('PLS')&f.component.eq(1)].pivot(
        index='category',columns='family',values='squared_axis_weight').fillna(0).reindex(CATEGORIES)
    axes[1].barh(g.index,g['Prior profile'],color='#355C7D',label='Prior profile')
    axes[1].barh(g.index,g['Current departure'],left=g['Prior profile'],color='#D17631',label='Current departure')
    axes[1].invert_yaxis();axes[1].legend(loc='lower right');axes[1].set_xlim(0,1)
    axes[1].set_title('Combined representation: PLS1 blocks');axes[1].set_xlabel('Squared axis-weight sum')
    fig.suptitle('Supervised directions mix measurement families; weights are not causal importance')
    fig.tight_layout(rect=[0,0,1,.95])
    return figure(fig,'05_supervised_family_weights')


def trajectory_figure(mode):
    style();f=read_table('component_trajectories');f=f.loc[f['mode'].eq(mode)&f.kind.eq('difference')]
    labels=['5th last','4th last','3rd last','2nd last','Last'] if mode=='appearance' else ['31–60','15–30','8–14','1–7']
    fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True,sharey='row')
    for row,block in enumerate(['Profile','Departure','Combined']):
        for col,category in enumerate(FOCUS):
            ax=axes[row,col];s=f.loc[f.category.eq(category)&f.block.eq(block)].sort_values('point')
            ax.plot(s.point,s['mean'],'o-',color='#246C85')
            ax.fill_between(s.point,s.simultaneous_low,s.simultaneous_high,color='#246C85',alpha=.2)
            ax.axhline(0,color='gray',ls='--',lw=1)
            ax.set_title(f'{category} — {int(s.events.iloc[0])} pairs',fontsize=10)
            ax.set_xticks(range(len(labels)),labels,rotation=20 if mode=='appearance' else 0)
            if col==0:ax.set_ylabel(block+' PLS1\ncase − control (training score SD)')
            if row==2:ax.set_xlabel('Outing before event' if mode=='appearance' else 'Days before event')
            ax.grid(alpha=.15)
    fig.suptitle('Frozen first supervised directions: original matched pairs outside training')
    fig.tight_layout(rect=[0,0,1,.96])
    return figure(fig,'06_supervised_trajectories_'+mode)


def stability_figure():
    style();f=read_table('stability_summary');f=f.loc[f.partition.eq('test')]
    keys=[('Shoulder','Original 62'),('Shoulder','Combined'),('Lower body','Original 62'),('Lower body','Combined')]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    metrics=[('median_first_axis_cosine','First-axis direction similarity'),
             ('median_subspace_overlap','Selected-subspace overlap'),
             ('median_rank_correlation','Test score-rank correlation'),
             ('median_alert_jaccard','Test top-5% alert-set overlap')]
    for ax,(metric,title) in zip(axes.flat,metrics):
        for offset,method in [(-.17,'PCA'),(.17,'PLS')]:
            s=f.loc[f.method.eq(method)].set_index(['category','block']).reindex(keys)
            ax.bar(np.arange(len(keys))+offset,s[metric],.32,color=COLORS[method],label=method)
        ax.set_xticks(range(len(keys)),[c+'\n'+b for c,b in keys],fontsize=9)
        ax.set_ylim(0,1);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.15)
    axes[0,0].legend()
    fig.suptitle('Median over 12 paired training-pitcher refits; component counts held fixed')
    fig.tight_layout(rect=[0,0,1,.95])
    return figure(fig,'07_supervised_stability')
