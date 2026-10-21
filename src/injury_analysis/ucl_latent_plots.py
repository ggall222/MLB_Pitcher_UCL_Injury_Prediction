"""Four focused, exportable elbow/UCL figures using frozen model coordinates."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .ucl_latent_visuals import CFG,read_table,figure


def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white',
                         'pdf.fonttype':42})


def trajectory_figure():
    style();data=read_table('balanced_trajectories')
    definitions=read_table('component_selection');shapes=read_table('trajectory_shape_summary')
    selected=definitions.loc[definitions.primary_trajectory.eq(1)].component.tolist()
    support=read_table('cohort_support').set_index('cohort').loc['Balanced five outings']
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,component in zip(axes.flat,selected):
        for kind,color in [('Injury','#B96036'),('Control','#246C85')]:
            s=data.loc[data.component.eq(component)&data['mode'].eq('appearance')&data.kind.eq(kind)].sort_values('point')
            ax.plot(s.point,s['mean'],'o-',color=color,label='Cases' if kind=='Injury' else 'Matched controls',lw=1.8,ms=5)
            ax.fill_between(s.point,s.simultaneous_low,s.simultaneous_high,color=color,alpha=.14)
        definition=definitions.set_index('component').loc[component]
        shape=shapes.loc[shapes.component.eq(component)&shapes['mode'].eq('appearance')&
                         shapes.metric.eq('last_minus_first_separation')].iloc[0]
        ax.set_title(f'{component}: {definition.block} PLS{int(definition.axis)}')
        ax.text(.02,.03,f'Change in separation: {shape["mean"]:+.2f}\n95% interval [{shape.low:+.2f}, {shape.high:+.2f}]',
                transform=ax.transAxes,fontsize=9,va='bottom',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        ax.axhline(0,color='gray',lw=.7,ls=':');ax.grid(alpha=.13)
        ax.set_xticks(range(5),['−5','−4','−3','−2','−1'])
        ax.set_xlabel('Outings before recorded event')
        ax.set_ylabel('Oriented component score\n(training score SD)')
    axes[0,0].legend(loc='upper left',fontsize=9)
    fig.suptitle('Elbow/UCL: matched pre-event PLS trajectories',fontsize=14)
    fig.text(.5,.93,f'{int(support.events)} balanced event pairs / {int(support.pitchers)} injured pitchers; '
             f'{int(support.initial_explicit_ucl_events)} events explicitly UCL at initial placement.',ha='center',fontsize=10)
    fig.text(.5,.016,'Bands cover all seven axes × five positions within each group; bootstrap clusters preserve shared case/control pitchers.\n'
             'Change intervals describe paired last-minus-first differences. Previously examined 2022–2025; not independent confirmation.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.075,1,.91])
    return figure(fig,'01_ucl_pls_matched_trajectories')


def lead_time_heatmap():
    style();f=read_table('balanced_trajectories');f=f.loc[f.kind.eq('difference')]
    components=read_table('component_selection').component.tolist()
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True,gridspec_kw={'width_ratios':[5,4]})
    norm=TwoSlopeNorm(vmin=-.75,vcenter=0,vmax=.75)
    for ax,mode,labels in zip(axes,['appearance','calendar'],
                            [['−5','−4','−3','−2','−1'],['31–60','15–30','8–14','1–7']]):
        g=f.loc[f['mode'].eq(mode)]
        mat=g.pivot(index='component',columns='point',values='mean').reindex(components)
        im=ax.imshow(mat,cmap='RdBu_r',norm=norm,aspect='auto')
        for i,c in enumerate(components):
            for j in range(len(labels)):
                row=g.loc[g.component.eq(c)&g.point.eq(j)].iloc[0]
                supported=row.simultaneous_low>0 or row.simultaneous_high<0
                ax.text(j,i,f'{row["mean"]:+.2f}'+(' •' if supported else ''),ha='center',va='center',fontsize=10)
        ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(components)),components)
        ax.set_title(f'{mode.capitalize()} alignment — {int(g.events.iloc[0])} balanced pairs')
        ax.set_xlabel('Outings before event' if mode=='appearance' else 'Days before event')
    axes[0].set_ylabel('D = departure; C = combined profile + departure\nAll validation-retained PLS axes')
    fig.subplots_adjust(top=.82,bottom=.18,left=.12,right=.91,wspace=.13)
    cax=fig.add_axes([.93,.22,.017,.55])
    fig.colorbar(im,cax=cax,label='Case − control (training component SD)',extend='both')
    supported=((f.simultaneous_low>0)|(f.simultaneous_high<0)).sum()
    fig.suptitle('Elbow/UCL separation across lead time',fontsize=14)
    fig.text(.5,.88,f'{supported} cells have joint 95% bands excluding zero; color magnitude alone is not evidence of separation.',
             ha='center',fontsize=10)
    fig.text(.5,.035,'Zero-centered scale. Joint uncertainty covers all components × times within each alignment. Broad elbow/UCL proxy.\n'
             'The two panels use different balanced cohorts; calendar bins do not equal outing positions. Color is not predictive importance.',
             ha='center',fontsize=9)
    return figure(fig,'02_ucl_component_lead_time_heatmap')


def compression_figure():
    style();f=read_table('ucl_component_curves');s=read_table('ucl_validation_selection')
    f=f.loc[f.partition.eq('validation')]
    fig,axes=plt.subplots(1,3,figsize=(14,5.4),sharey=True)
    for ax,block in zip(axes,['Original 62','Departure','Combined']):
        colors={'PCA':'#CF7C37','PLS':'#246C85'}
        for method in ['PCA','PLS']:
            g=f.loc[f.block.eq(block)&f.method.eq(method)].sort_values('components')
            ax.plot(g.components,g.average_precision,'o-',color=colors[method],label=method,lw=1.7,ms=4)
            chosen=s.loc[s.block.eq(block)&s.method.eq(method)].iloc[0]
            k=int(chosen.selected_components)
            ap=g.loc[g.components.eq(k),'average_precision'].iloc[0]
            ax.scatter([k],[ap],marker='s',s=90,facecolors='none',edgecolors=colors[method],lw=1.5,zorder=5)
        raw=f.loc[f.block.eq(block)&f.method.eq('Unreduced')].iloc[0]
        ax.axhline(raw.average_precision,color='gray',ls='--',label='Unreduced same inputs')
        ax.axhline(raw.prevalence,color='black',ls=':',lw=.8,label='Prevalence')
        choices=s.loc[s.block.eq(block)].set_index('method')
        ax.set_title(block)
        ax.text(.03,.035,f'Selected axes: PCA {int(choices.loc["PCA","selected_components"])}; PLS {int(choices.loc["PLS","selected_components"])}\n'
            f'Plus {int(choices.passthrough_columns.iloc[0])} context/missingness columns',transform=ax.transAxes,fontsize=9)
        ax.set_xlabel('Retained continuous axes');ax.grid(alpha=.15)
    ymin=f.average_precision.min();ymax=f.average_precision.max()
    axes[0].set_ylim(ymin-.0025,ymax+.0006)
    axes[0].set_ylabel('Validation Average Precision')
    fig.suptitle('Elbow/UCL: validation ranking with fewer supervised axes',fontsize=14)
    cohort=read_table('ucl_selected_performance').query("partition=='validation' and block=='Original 62' and variant=='Unreduced'").iloc[0]
    fig.text(.5,.915,f'2022–2023: {int(cohort.rows):,} appearances / {int(cohort.positive_rows):,} positive rows / '
             f'{int(cohort.represented_events)} represented elbow/UCL events. Selection-period estimates are optimistic.',ha='center',fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9)
    fig.text(.5,.09,'Squares mark validation-selected dimensions. Same rows and candidate counts per representation; all transforms trained through 2021.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.88])
    return figure(fig,'03_ucl_pca_pls_compression')


def individual_heatmap():
    style();cells=read_table('individual_event_cells');order=read_table('individual_event_order')
    component=read_table('component_selection').query('individual_heatmap').component.iloc[0]
    n=len(order);limit=CFG['individual_display_limit_sd']
    fig,axes=plt.subplots(1,3,figsize=(12,10.2),sharey=True)
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit);clipped=0
    for ax,kind,title in zip(axes,['Injury','Control','difference'],['Injury cases','Matched controls','Case − control']):
        mat=cells.pivot(index='row_number',columns='point',values=kind).reindex(order.row_number)
        clipped+=int((np.abs(mat.to_numpy())>limit).sum())
        im=ax.imshow(mat,cmap='RdBu_r',norm=norm,aspect='auto',interpolation='nearest')
        ax.set_title(title);ax.set_xticks(range(5),['−5','−4','−3','−2','−1'])
        ax.set_xlabel('Outings before event / control anchor')
        ax.set_yticks(range(n),order.event_label,fontsize=6)
        ax.tick_params(axis='y',length=0)
    axes[0].set_ylabel('Individual events, ordered by case late-minus-early score\nMore increasing trajectories toward the bottom')
    fig.subplots_adjust(left=.12,right=.89,top=.88,bottom=.1,wspace=.14)
    cax=fig.add_axes([.915,.14,.018,.70])
    fig.colorbar(im,cax=cax,label=f'{component} oriented score / paired difference (training SD)',extend='both')
    fig.suptitle(f'Individual elbow/UCL trajectories: {component} across {n} matched event pairs',fontsize=14)
    fig.text(.5,.925,'One fixed training-derived scale; no per-event centering. Cases and controls share row order.',ha='center',fontsize=10)
    fig.text(.5,.025,f'Rows are ordered descriptively, not clustered into injury phenotypes. {clipped} of {n*5*3} cells exceed ±{limit:g} SD color limits.\n'
        'Unclipped values and event identifiers are exported. Broad elbow/UCL proxy; previously examined 2022–2025.',
        ha='center',fontsize=9)
    return figure(fig,'04_individual_ucl_latent_trajectories')
