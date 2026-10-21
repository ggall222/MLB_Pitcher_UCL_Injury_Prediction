"""Six publication-exportable figures; plot selection never selects model inputs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from .consolidation_common import CACHE, OUT, figure, table, read_table

SHORT = {'Release position and variability': 'Release', 'Pitch mix and pitch-type counts': 'Pitch mix',
    'Workload and rest': 'Workload', 'Pitcher background and experience': 'Background',
    'Measurement coverage and history support': 'Support', 'Location and strike fractions': 'Location',
    'Role and recent starts': 'Role', 'Appearance performance': 'Performance', 'Pitch movement': 'Movement'}


def finish(fig, name):
    figure(fig, name)
    plt.close(fig)
    return name


def correlation_figure():
    corr = pd.read_parquet(CACHE / 'spearman.parquet')
    clusters = read_table('correlation_clusters')
    order = json.loads((CACHE / 'hierarchies.json').read_text())['orders']
    groups = ['Velocity', 'Release position and variability', 'Spin', 'Pitch movement',
              'Workload and rest', 'Pitch mix and pitch-type counts']
    fig, axes = plt.subplots(2, 3, figsize=(20, 15), layout='constrained')
    plotted = []
    for ax, group in zip(axes.flat, groups):
        g = clusters.loc[clusters.feature_group.eq(group)]
        representatives = set(g.loc[g.representative, 'feature'])
        names = [n for n in order[group] if n in representatives]
        # Correlation-only representatives keep the printed panel readable.
        # Every field, full matrix and complete-linkage tree remain in artifacts.
        if len(names) > 32:
            names = [names[i] for i in np.linspace(0, len(names)-1, 32).astype(int)]
        plotted.extend(dict(group=group, position=i, feature=n) for i, n in enumerate(names))
        im = ax.imshow(corr.loc[names, names], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_title(f'{SHORT.get(group, group)}: {len(g)} fields → {g.correlation_cluster.nunique()} clusters\n{len(names)} representatives shown', fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.45, label='Pairwise Spearman correlation')
    fig.suptitle('Within-family redundancy in training data\n13,427 pitcher-capped appearances; complete-linkage order; temporal roles protected', fontsize=16)
    table(pd.DataFrame(plotted), 'correlation_figure_fields')
    return finish(fig, '01_within_family_redundancy')


def cross_group_figure():
    df = read_table('cross_group_summary_canonical')
    names = sorted(set(df.group_a) | set(df.group_b))
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    counts = pd.DataFrame(0, index=names, columns=names)
    for r in df.itertuples():
        matrix.loc[r.group_a, r.group_b] = matrix.loc[r.group_b, r.group_a] = 100*r.high_pair_fraction
        counts.loc[r.group_a, r.group_b] = counts.loc[r.group_b, r.group_a] = r.high_pairs
    fig, ax = plt.subplots(figsize=(10, 9), layout='constrained')
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, np.nanmax(matrix)))
    ax.set_xticks(range(len(names)), [SHORT.get(n, n) for n in names], rotation=55, ha='right')
    ax.set_yticks(range(len(names)), [SHORT.get(n, n) for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            if counts.iloc[i, j]:
                ax.text(j, i, f'{matrix.iloc[i,j]:.1f}%\n({counts.iloc[i,j]})', ha='center', va='center', fontsize=7,
                        color='white' if matrix.iloc[i,j] > np.nanmax(matrix)*.55 else 'black')
    fig.colorbar(im, ax=ax, shrink=.75, label='Percent of pairs with |Pearson| or |Spearman| ≥ 0.90')
    ax.set_title('Redundancy crosses feature-family boundaries\nCells: percentage (number of pairs); diagonals are within-family pairs')
    return finish(fig, '02_cross_family_redundancy')


def family_figure():
    ab = read_table('family_paired_differences').query("variant.str.startswith('Without family')", engine='python')
    perm = read_table('grouped_permutation_summary').set_index('group')
    performance = read_table('family_performance').set_index('variant')
    ab = ab.sort_values('ap_loss')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, layout='constrained')
    y = np.arange(len(ab))
    axes[0].hlines(y, ab.low, ab.high, color='#4e79a7')
    axes[0].scatter(ab.ap_loss, y, color='#4e79a7')
    for k, r in enumerate(ab.itertuples()):
        count = int(performance.loc[r.variant, 'alerted_events'])
        axes[0].annotate(f'{count} warned', (r.ap_loss, k), xytext=(4, 7), textcoords='offset points', fontsize=7)
    p = perm.reindex(ab.family)
    axes[1].errorbar(p['mean'], y, xerr=p['std'], fmt='o', color='#b47b16', capsize=3)
    for ax in axes:
        ax.axvline(0, color='gray', lw=1)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Average Precision decrease')
    axes[0].set_yticks(y, [SHORT.get(n, n) for n in ab.family])
    axes[0].set_title('Remove family and refit\nPaired pitcher-bootstrap 95% interval')
    axes[1].set_title('Jointly shuffle family in full model\nMean ± SD of five perturbations (not a CI)')
    fig.suptitle('Unique contribution and reliance are different questions\n2022–2023: 26,796 rows, 600 positives; full model warns 11/120 events at 5% alerts', fontsize=13)
    return finish(fig, '03_family_ablation_and_permutation')


def stability_figure():
    all_features = read_table('feature_stability').sort_values(['selection_frequency', 'median_absolute_coefficient'], ascending=False)
    selected = set(json.loads((CACHE/'candidate_selection.json').read_text())['features'])
    unstable = all_features.loc[all_features.selection_frequency.ge(.5)].sort_values('conditional_sign_stability').head(10)
    names = list(dict.fromkeys(all_features.loc[all_features.feature.isin(selected),'feature'].tolist()+
                              all_features.head(12).feature.tolist()+unstable.feature.tolist()))
    names += [n for n in all_features.feature if n not in names][:max(0,30-len(names))]
    summary = all_features.set_index('feature').loc[names].reset_index()
    table(summary, 'stability_figure_features')
    raw = read_table('elastic_net_refit_coefficients')
    raw['sign'] = np.where(raw.selected, np.sign(raw.coefficient), 0)
    values = raw.pivot(index='feature', columns='repeat', values='sign').reindex(summary.feature)
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, gridspec_kw={'width_ratios':[4,1]}, layout='constrained')
    cmap = ListedColormap(['#2166ac','#eeeeee','#b2182b'])
    axes[0].imshow(values, aspect='auto', cmap=cmap, norm=BoundaryNorm([-1.5,-.5,.5,1.5],3))
    axes[0].set_yticks(range(len(summary)), [('* ' if f in selected else '')+f for f in summary.feature], fontsize=8)
    axes[0].set_xticks(range(values.shape[1]), np.arange(1, values.shape[1]+1))
    axes[0].set_xlabel('Whole-pitcher training refit')
    axes[1].barh(range(len(summary)), summary.selection_frequency, color='#555555')
    axes[1].set_xlim(0, 1.05); axes[1].set_xlabel('Selection frequency')
    fig.suptitle('Sparse selection across 16 whole-pitcher resamples\nBlue: negative; red: positive; gray: zero; * retained candidate field\nRetained fields plus frequently selected and sign-unstable examples; hyperparameters fixed', fontsize=13)
    return finish(fig, '04_selection_and_sign_stability')


def reduction_figure():
    df = read_table('candidate_reduction_curve').sort_values('features')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    for ax, metric, lo, hi, label in [(axes[0], 'average_precision', 'ap_low', 'ap_high', 'Average Precision'),
        (axes[1], 'injury_event_recall', 'event_low', 'event_high', '7–30-day event recall at 5% alerts')]:
        ax.plot(df.features, df[metric], color='.75', zorder=1)
        for i, r in enumerate(df.itertuples()):
            color = '#c44e52' if r.selected else '#4e79a7'
            ax.vlines(r.features, getattr(r,lo), getattr(r,hi), color=color, alpha=.6)
            ax.scatter(r.features, getattr(r,metric), marker='*' if r.selected else 'o', s=180 if r.selected else 35, color=color, zorder=3)
            ax.annotate(str(i+1), (r.features,getattr(r,metric)), xytext=(4,4), textcoords='offset points', fontsize=8)
        ax.set_xlabel('Input fields (missing flags additional)');ax.set_ylabel(label)
        ax.grid(alpha=.2)
    legend = '\n'.join(f'{i+1}: {r.variant} ({r.features})'+(' — selected' if r.selected else '') for i,r in enumerate(df.itertuples()))
    fig.text(.99,.5,legend,va='center',ha='left',fontsize=9)
    fig.suptitle('Validation performance during feature consolidation\nPaired-pitcher 95% intervals; connecting line guides reading of nonnested candidate sets', fontsize=14)
    return finish(fig, '05_reduction_performance')


def coverage_figure():
    df = read_table('coverage_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout='constrained')
    y = np.arange(len(df))
    axes[0].hlines(y,df.ap_difference_low,df.ap_difference_high,color='#4e79a7')
    axes[0].scatter(df.ap_difference_from_no_flags,y,color='#4e79a7')
    axes[0].axvline(0,color='gray');axes[0].set_xlabel('AP difference versus core without missing flags')
    axes[0].set_yticks(y,df.variant)
    axes[1].barh(y,df.alerted_events,color='#4e79a7')
    axes[1].tick_params(axis='y',labelleft=False);axes[1].set_xlabel('Events warned 7–30 days ahead / 120 qualifying events')
    for i,r in enumerate(df.itertuples()):
        axes[1].text(r.alerted_events+.15,i,str(int(r.alerted_events)),va='center')
    fig.suptitle('Does explicit measurement support improve the compact model?\nValidation only; AP intervals are paired pitcher-bootstrap; all models use the same rows',fontsize=13)
    return finish(fig, '06_coverage_sensitivity')


FIGURES = [('01_within_family_redundancy',correlation_figure),('02_cross_family_redundancy',cross_group_figure),
    ('03_family_ablation_and_permutation',family_figure),('04_selection_and_sign_stability',stability_figure),
    ('05_reduction_performance',reduction_figure),('06_coverage_sensitivity',coverage_figure)]


def create_figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    for name, function in FIGURES:
        if not all((OUT/'figures'/(name+'.'+ext)).exists() for ext in ['png','pdf']):
            function()
            print('Saved '+name, flush=True)
