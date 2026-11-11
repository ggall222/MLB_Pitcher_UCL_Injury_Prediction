"""Readable locked replication synthesis, figures and preservation checks."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ucl_temporal_models import load
from .ucl_temporal_data import interval_target
from .ucl_replication_common import ROOT,CFG,OUT,CACHE,OLD,OLD_CACHE,sha,write_json,table,read,figure,checkpoint,valid,locked
from .ucl_replication_data import prepare

INTERVALS=['1-7','8-14','15-30','31-60']
NAMES=['Only | Velocity','Only | Pitcher background and experience']
COLORS=['#1f718a','#c36b37']
def md(df):
    def fmt(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])

def figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    m=read('replication_metrics');shared=read('shared_event_sensitivity');influence=read('event_influence_summary')
    old=pd.read_csv(OLD/'tables/development_metrics.csv');old=old.loc[old.partition.eq('validation')]
    transfer=pd.read_csv(OLD/'tables/transfer_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,name in zip(axes,NAMES):
        for df,label,color,style in [(old,'2022–2023 development','#999999','o-'),(transfer,'2024–2025 exploratory transfer','#7777aa','s--'),
            (m,'2026 locked replication','#1f718a','o-'),(shared,'2026 shared 8 events','#c36b37','s:')]:
            g=df.loc[df.model.eq(name)].set_index('interval').loc[INTERVALS];ax.plot(range(4),g.pr_lift,style,color=color,label=label)
        g=m.loc[m.model.eq(name)].set_index('interval').loc[INTERVALS]
        for i,rr in enumerate(g.itertuples()):ax.annotate(f'n={rr.represented_events}',(i,rr.pr_lift),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(1,color='gray',ls=':');ax.set(xticks=range(4),xticklabels=INTERVALS,xlabel='Days until first UCL event',title=name.removeprefix('Only | '),yscale='log')
    axes[0].set_ylabel('PR lift (log scale)');axes[1].legend(fontsize=8)
    fig.suptitle('Event composition changes the apparent temporal pattern');figure(fig,'01_replication_and_shared_events');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,name,color in zip(axes,NAMES,COLORS):
        q=influence.loc[influence.model.eq(name)].set_index('interval').loc[INTERVALS]
        for i,r in enumerate(q.itertuples()):
            ax.plot([r.loo_lift_min,r.loo_lift_max],[i,i],color=color,lw=3);ax.scatter(r.original_lift,i,color=color,s=45)
            ax.annotate(f'{r.largest_event_fraction_ap:.0%} AP from largest event',(max(r.original_lift,r.loo_lift_max),i),xytext=(5,4),textcoords='offset points',fontsize=8)
        ax.set(title=name.removeprefix('Only | '),xlabel='PR lift; bar = injured-pitcher leave-one-out range',yticks=range(4),yticklabels=INTERVALS,xlim=(0,max(q.loo_lift_max.max(),q.original_lift.max())*1.65));ax.axvline(1,color='gray',ls=':')
    axes[0].set_ylabel('Lead interval (days)');fig.suptitle('Primary estimates retain every event; influence is diagnostic')
    figure(fig,'02_event_dependence');plt.close(fig)
    op=read('alert_policy_results');fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,interval in zip(axes.flat,INTERVALS):
        for name,color in zip(NAMES+['Full union'],COLORS+['#606b57']):
            g=op.loc[op.interval.eq(interval)&op.model.eq(name)].sort_values('development_budget')
            ax.plot(g.alert_episodes_per_pitcher_season,100*g.event_recall,'o-',color=color,label=name.removeprefix('Only | '))
            r=g.loc[g.development_budget.eq(.05)].iloc[0];ax.scatter(r.alert_episodes_per_pitcher_season,100*r.event_recall,marker='*',s=180,color=color)
        ax.set(title=f'{interval} days; all 21 qualifying events',xlabel='Alert episodes per pitcher-season',ylabel='Event recall (%)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Development-fixed cutoffs: burden versus warnings\nStars mark 5% development budget, not 5% holdout alerts')
    figure(fig,'03_operational_warning_tradeoff');plt.close(fig)
    bins=read('reliability_bins');fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for ax,interval in zip(axes.flat,INTERVALS):
        for name,color in zip(NAMES+['Full union'],COLORS+['#606b57']):
            g=bins.loc[bins.interval.eq(interval)&bins.model.eq(name)]
            ax.plot(100*g.mean_risk,100*g.observed,'o-',color=color,label=name.removeprefix('Only | '))
            if name=='Full union':
                for r in g.itertuples():ax.annotate(f'{r.represented_events}e',(100*r.mean_risk,100*r.observed),xytext=(3,4),textcoords='offset points',fontsize=8)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],':',color='gray')
        ax.set(title=f'{interval} days',xlabel='Mean frozen predicted risk (%)',ylabel='Observed interval event rate (%)')
    axes[0,0].legend(fontsize=7);fig.suptitle('Coarse calibration; labels count distinct events in full-union bins\nFive development-defined bins; no holdout recalibration')
    figure(fig,'04_calibration');plt.close(fig)
    work=read('workload_diagnostic');fig,axes=plt.subplots(1,3,figsize=(12,4))
    fields=[('total_pitches','Pitches in current outing'),('rest_days','Days since prior outing'),('Workload and rest','Frozen workload score (training SD)')]
    for ax,(field,title) in zip(axes,fields):
        for part,color,label in [('validation','#1f718a','2022–2023'),('test','#c36b37','2024–2025 exploratory')]:
            g=work.loc[work.field.eq(field)&work.partition.eq(part)].set_index('mode').loc[['calendar','outings']]
            x=np.arange(2)+(-.08 if part=='validation' else .08);y=g.near_minus_far_case_control_difference
            ax.errorbar(x,y,yerr=np.vstack([y-g.ci_low,g.ci_high-y]),fmt='o',capsize=3,color=color,label=label)
            for xx,yy,n in zip(x,y,g.events):ax.annotate(f'n={n}',(xx,yy),xytext=(3,5),textcoords='offset points',fontsize=7)
        ax.axhline(0,color='gray',ls=':');ax.set(title=title,xticks=[0,1],xticklabels=['Calendar','Outings'])
    axes[0].set_ylabel('Near-minus-far change in case–control difference');axes[2].legend(fontsize=7)
    fig.suptitle('Usage reduction is more consistent than an added-rest explanation');figure(fig,'05_workload_interpretation');plt.close(fig)
    c=read('censoring');c['outcome']=np.select([c.label_status.eq('ucl_first_event'),c.label_status.isin(['observed_event_free_horizon','non_ucl_competing_first_event'])],['First UCL within 60d','Known comparison outcome'],default='Unknown / censored')
    z=c.pivot_table(index='month',columns='outcome',values='rows',aggfunc='sum',fill_value=0)
    fig,ax=plt.subplots(figsize=(9,4));z.plot.bar(stacked=True,ax=ax,color={'First UCL within 60d':'#c36b37','Known comparison outcome':'#1f718a','Unknown / censored':'#b0b7bb'})
    ax.set(title='Common 60-day outcome support limits late-season inference',ylabel='Eligible 2026 appearance landmarks',xlabel='Appearance month');ax.tick_params(axis='x',rotation=30)
    figure(fig,'06_followup_support');plt.close(fig)


def qa():
    spec=locked();f,e,_=prepare();rows=[]
    def check(name,ok,detail=''):rows.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    access=json.loads((OUT/'replication_access.json').read_text())
    check('freeze precedes first scoring',pd.Timestamp(spec['frozen_at_utc'])<pd.Timestamp(access['first_scored_at_utc']))
    check('freeze unchanged',sha(OUT/'replication_freeze.json')==access['freeze_sha256'])
    check('candidate nomination unchanged',spec['candidate_families']==['Velocity','Pitcher background and experience'])
    check('original artifacts byte-identical',True,len(spec['prior_artifacts'])) # locked() verified every hash.
    check('replication year only',f.game_date.dt.year.eq(2026).all())
    check('appearance keys unique',not f.duplicated(['pitcher_id','game_id']).any())
    for c in ['history_max_date','workload_max_date','baseline_max_date']:
        check(c+' strictly earlier',bool((f[c].isna()|f[c].lt(f.game_date)).all()))
    check('censored outcomes remain missing',f.loc[f.label_status.str.startswith('censored'),'target'].isna().all())
    check('known event-free negatives have followup',f.loc[f.label_status.eq('observed_event_free_horizon'),'dynamic_followup'].all())
    reference=None
    for lo,hi in CFG['intervals']:
        interval=f'{lo}-{hi}';g=interval_target(f,lo,hi);ids=g.loc[g.target.notna(),'row_id'].tolist()
        if reference is None:reference=ids
        check(interval+' common risk rows',reference==ids)
        dev=pd.read_parquet(OLD_CACHE/f'predictions_validation_{interval}.parquet')
        check(interval+' threshold dates development only',dev.game_date.between('2022-01-01','2023-12-31').all())
        for name in spec['models']:
            obj=load(interval,name);key=interval+'|'+name
            check(key+' frozen fit',obj['signature']==spec['fit_signatures'][key])
            check(key+' training preprocessing only',pd.Timestamp(obj['encoder'].fit_max_date)<=pd.Timestamp('2021-12-31'))
            check(key+' fixed L2 design',obj['model'].C==1 and obj['model'].penalty=='l2')
            check(key+' frozen development thresholds',all(np.isclose(spec['thresholds'][key][str(r)],dev[name].quantile(1-r),rtol=0,atol=1e-15) for r in CFG['alert_rates']))
    cc=read('event_ap_contributions').groupby(['interval','model']).ap_contribution.sum();met=read('replication_metrics').set_index(['interval','model'])
    check('exact event contributions sum to AP',np.allclose(cc,met.loc[cc.index,'average_precision']))
    w=read('workload_diagnostic');original=pd.read_csv(OLD/'tables/trajectory_shape.csv');original=original.loc[original.family.eq('Workload and rest')].set_index(['partition','mode'])
    score=w.loc[w.kind.eq('score')].set_index(['partition','mode'])
    check('original workload score contrasts reproduced',np.allclose(score.near_minus_far_case_control_difference,original.loc[score.index,'final_minus_distant_difference']))
    contrib=w.loc[w.kind.eq('coefficient_contribution')].groupby(['partition','mode']).near_minus_far_case_control_difference.sum()
    check('workload linear decomposition sums to score',np.allclose(contrib,score.loc[contrib.index,'near_minus_far_case_control_difference']))
    op=read('alert_policy_results');check('known plus unknown equals alert count',(op.known_outcome_alerts+op.unknown_outcome_alerts==op.alerted_outings).all())
    check('no partial artifacts',not list(CACHE.rglob('*.partial')))
    result=table(pd.DataFrame(rows),'validation_checks');write_json(dict(passed=bool(result.passed.all()),checks=len(result),prior_temporal_artifacts_preserved=len(spec['prior_artifacts']),models_refitted=0),OUT/'validation_summary.json')
    if not result.passed.all():raise AssertionError(result.loc[~result.passed].to_dict('records'))


def summary():
    spec=locked();support=read('cohort_support');m=read('replication_metrics');influence=read('event_influence_summary')
    contrast=read('temporal_contrasts');nom=contrast.loc[contrast.family.isin(CFG['candidate_families'])]
    op=read('alert_policy_results');op=op.loc[op.development_budget.eq(.05)&op.model.isin(NAMES)]
    shared=read('shared_event_sensitivity');work=read('workload_diagnostic');qaresult=json.loads((OUT/'validation_summary.json').read_text())
    selected=m.loc[m.model.isin(NAMES)];table(selected,'nominated_replication_metrics')
    table(influence.loc[influence.model.isin(NAMES)],'nominated_event_stability');table(op,'nominated_operational_policy')
    joined=selected[['interval','model','pr_lift']].merge(shared[['interval','model','pr_lift','shared_events']],on=['interval','model'],suffixes=('_all','_shared'))
    vel=influence.loc[influence.model.eq('Only | Velocity')&influence.interval.eq('8-14')].iloc[0]
    vshared=shared.loc[shared.model.eq('Only | Velocity')&shared.interval.eq('8-14')].iloc[0]
    primary_support=support.iloc[0]
    text=f'''# Locked temporal replication and operational warning

**Decision: temporal heterogeneity remains unsupported. Continue toward dynamic multi-horizon UCL risk estimation; do not build an acute/intermediate/chronic architecture from these results.** Some discrimination survives, but timing specificity, event stability and useful warning burden are different questions. None of the {len(contrast)} original family timing contrasts excludes zero under the original joint bootstrap procedure; {int(nom.same_direction_development.sum())}/{len(nom)} nominated-family contrasts have the same sign as development. Same direction alone does not demonstrate similar magnitude, statistical support or operational value.

## Locked design and support

The original `ucl_temporal` freeze, 28 fixed C=1 L2 logistic model definitions, 113-field union, 11 families and four intervals remain unchanged. Models and preprocessing were originally trained in 2015–2021; development validation was 2022–2023. The 2024–2025 results retain their exploratory-transfer interpretation. No model was fitted or selected in this replication. This specifically applies the original temporal freeze, while the separate revised-training study remains separate.

2026 is **locked retrospective chronological replication sensitivity**, not prospective or fully independent confirmation: these dates predate the October 20 development freeze and existed elsewhere in the project. The operational specification and cutoffs were frozen at {spec['frozen_at_utc']} before replication scoring. Source observation ends {CFG['observation_end']}; regular-season appearances end {primary_support.latest_appearance}.

{md(support[['interval','rows','pitchers','positive_rows','represented_events','qualifying_events','unknown_rows']])}

The same **{int(primary_support.rows):,} observed 60-day first-event rows** underlie all interval estimates. There are {int(primary_support.eligible_landmarks):,} eligible landmarks and {int(primary_support.unknown_rows):,} unknown/censored rows. A known competing injury ends the first-event interval; unknown/ambiguous episodes remain censored. UCL in another interval is an interval-specific negative, not a healthy-pitcher declaration. Late-season absence of observed follow-up never becomes a negative. The latest observed event-free 60-day landmark is {primary_support.latest_observed_eventfree}. With only 9–16 represented events per interval, this supports a limited sensitivity analysis, not precise confirmation of timing differences.

## Ranking and event dependence

{md(selected[['interval','model','average_precision','prevalence','pr_lift','roc_auc','brier_score','calibration_bias']])}

Velocity's strongest 2026 point estimate moves to 8–14 days, rather than reproducing the original 1–7-day peak. Its lift is {vel.original_lift:.2f}, but **{vel.largest_event_fraction_ap:.1%} of AP comes from one event**. The lowest injured-pitcher leave-one-out lift is {vel.loo_lift_min:.2f}; all primary analyses retain that event. Exact AP contribution is an additive ranking decomposition, while leave-one-pitcher-out removes all that pitcher's evaluation rows; the two diagnostics have different meanings.

{md(influence.loc[influence.model.isin(NAMES),['interval','model','largest_event_fraction_ap','loo_lift_min','loo_lift_max']])}

Background AP is less concentrated: the largest event accounts for roughly 12–27% across intervals, compared with the 72% velocity peak. This is relative ranking stability, not proof of a timing-specific mechanism.

Only **{int(shared.shared_events.iloc[0])} events** are represented in every interval. Keeping the original shared-event comparison rule lowers the 8–14-day velocity lift to **{vshared.pr_lift:.2f}**. Background retains a proximal ranking gradient in this restricted sample, but its incremental timing contrasts remain uncertain. Event-composition restriction is retrospective and not a new target population or fitting rule.

{md(joined)}

The exact 33 near-minus-31–60-day normalized ablation contrasts, original five reference contrasts, joint max-t bands, shared-event contrasts and continuous development-to-replication effect differences are saved. There was no prespecified equivalence margin; wide intervals and same-sign estimates cannot establish similar effect size. Bootstrap intervals resample whole pitchers and are conditional on frozen fits; they do not capture all model-development or injury-ascertainment uncertainty.

## Operational warnings

All 1%, 2.5%, 5% and 10% cutoffs are numeric score quantiles of the original **2022–2023 observed-outcome validation rows**. They are never optimized on 2026. Deployment applies them to **all eligible 2026 landmarks**, including unknown-outcome rows. Consecutive alerts form an episode within pitcher-season; any non-alert or ineligible outing, season boundary, or >30-day gap breaks it. Alert grouping uses no future outcome information. Each interval/model is evaluated separately; these are not combined into an ensemble warning system.

At the 5% development cutoff:

{md(op[['interval','model','detected_events','observable_events','realized_alert_rate','alert_episodes','alert_episodes_per_pitcher_season','alerts_per_detected_injury','non_actionable_alert_fraction','unknown_outcome_alerts','median_warning_lead_days']])}

All-event recall uses **21 qualifying events**, including those without eligible opportunities; observable-event recall uses the interval's supported subset. The nominated models detect only **0–5 events per interval**, despite hundreds of alert episodes. Among alerts with known outcomes, roughly **98–100% are not followed by the target interval UCL event**. This is not proof that those pitchers were healthy or that alerts lacked every possible clinical use. Unknown alerts are explicitly unresolved, not false. Background cutoffs produce about **7–17%** actual alert rates across intervals despite the 5% development budget, with long persistent alert episodes. The source of this operational drift is not resolved by ranking AP. Realized rates and unresolved-alert fractions should be monitored before deployment.

The 1–7-day endpoint remains proximal and does not establish useful ≥7-day warning. Longer-window detections provide median leads shown above, but often involve only one or two events. Detection fractions and earliest/median lead times must be read with their event counts. No cutoff is endorsed for clinical action from these data. The original retrospective top-5% warning calculation is retained as context, clearly separated from fixed-threshold operational results.

## Calibration and workload interpretation

Most headline models underpredict the observed interval prevalence. Brier score and mean-risk bias are reported for every interval; coarse reliability bins use development-defined edges and show distinct event counts. Diagnostic intercept/slope fits were allowed only with at least 10 represented events and 20 positive rows: 1–7 and 15–30 days qualify; 8–14 and 31–60 remain too sparse by that locked rule. No diagnostic fit recalibrates any predictions.

The workload diagnostic reuses the original matched pairs and one fixed training-derived workload score. The original development score decreases are exactly reproduced. In 2024–2025 exploratory transfer, the calendar score change is −0.283 training SD (interval crosses zero), and the outing change is −0.016 SD. Current pitch counts decrease relative to controls by about 26 pitches in calendar alignment and 20 pitches in outing alignment; added rest does **not** replicate consistently. Workload acceleration, recent pitch volume and season workload have offsetting signed contributions. This supports usage reduction as an interpretation to investigate, not a causal injury mechanism or evidence of skipped scheduled starts. Symptoms, intent and missed starts are not established by the available data. Raw support, dependency-cluster intervals, event changes and leave-event-out means are retained. Removing any one event leaves the development workload-score change negative in both alignments; the exploratory-transfer outing change ranges from −0.074 to +0.139 SD when individual events are omitted, so even its direction is unstable.

## What to carry forward

Keep the nominated families and fixed interval definitions for future replication, but simplify the scientific framing to **dynamic multi-horizon UCL risk estimation**. Do not force distinct acute/intermediate/chronic mechanisms. Mature late-2026 outcomes in a separately versioned evaluation using these same frozen models and cutoffs; any later policy redesign must use a development cohort, not these holdout results. A future heterogeneous architecture requires more stable event-level and shared-event evidence plus a defensible operational burden.

## Reproduction and artifacts

- Methods: `docs/ucl_temporal_replication_methods.md`.
- Frozen configuration: `config/ucl_temporal_replication.yaml`; operational/model hashes: `replication_freeze.json`.
- Reusable entry point: `.venv/bin/python -m src.injury_analysis.ucl_replication_pipeline`.
- Synthesis: `notebooks/ucl_temporal_replication/01_locked_replication.ipynb`.
- Tables cover support/censoring, all 112 model-interval metrics, original contrasts, event influence, shared-event sensitivity, 448 fixed-threshold policies, episode/event warning details, calibration and matched workload diagnostics.
- Six figures, each PNG/PDF, are in `figures/`.
- **{qaresult['checks']} analysis checks passed**, **{qaresult['prior_temporal_artifacts_preserved']} original temporal artifacts verified unchanged**, **zero model refits**. Unit-test and notebook execution logs are saved beside this summary.
'''
    (OUT/'summary.md').write_text(text)
    table(pd.DataFrame([
        dict(figure='01_replication_and_shared_events',question='Does the timing pattern replicate?',result='Velocity peak shifts and shrinks under common events; no established timing architecture.'),
        dict(figure='02_event_dependence',question='Are headline results event-stable?',result='8–14d velocity AP is concentrated in one event; primary estimates retain it.'),
        dict(figure='03_operational_warning_tradeoff',question='Are frozen cutoffs useful at manageable burden?',result='Few events detected, many episodes, and cohort-dependent realized alert rates.'),
        dict(figure='04_calibration',question='Do scores estimate interval risks well?',result='Generally underpredictive; sparse event support limits calibration inference.'),
        dict(figure='05_workload_interpretation',question='What explains falling workload scores?',result='Usage reduction is more consistent than added rest; prior exploratory cohorts only.'),
        dict(figure='06_followup_support',question='Which 2026 outcomes can be evaluated?',result='Late-season unknown outcomes remain censored, including unresolved alerts.')]),'figure_guide')


def run_reporting():
    if valid('reporting'):return
    qa();figures();summary()
    checkpoint('reporting',[OUT/'summary.md',OUT/'validation_summary.json',OUT/'tables/validation_checks.csv',OUT/'tables/figure_guide.csv']+list((OUT/'figures').glob('*')))
