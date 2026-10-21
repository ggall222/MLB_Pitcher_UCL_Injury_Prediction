"""Publication artifacts and read-only validation for dynamic UCL development."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ucl_dynamic_common import ROOT,OUT,CACHE,CFG,table,read_table,figure,write_json,checkpoint,valid_checkpoint,snapshot
from .ucl_dynamic_data import prepare_data,risk_frame,event_cohort
from .ucl_dynamic_models import scores,model_path
from .common import sha,source_signature

SHORT={'Background':'Background','Profile':'Profile','Departure':'Departure (8)',
    'Profile + departure':'Profile + departure','Profile + departure + trajectory':'+ trajectories',
    'Profile + departure + trajectory + interactions':'+ interactions','Previous 62':'Previous 62','Stable 62':'Stable 62'}
COLORS=['#7d8491','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#222222']

def operational_audit():
    a,e,specs=prepare_data();f=risk_frame(a,e);freeze=json.loads((OUT/'frozen_protocol.json').read_text())
    f=f.sort_values(['pitcher_id','game_date','game_id']);f['outing_index']=f.groupby('pitcher_id').cumcount()
    op=[];opportunities=[]
    for part in ['train','validation','test','descriptive_2026']:
        ev=event_cohort(e,part);g=f.loc[f.partition.eq(part)]
        for r in ev.itertuples():
            rows=g.loc[g.pitcher_id.eq(r.player_id)&g.game_date.lt(r.prediction_reference_date)&g.game_date.ge(r.prediction_reference_date-pd.Timedelta(days=30))]
            opportunity=rows.loc[rows.analysis_eligible&rows.dynamic_next_target&rows.dynamic_next_event_id.eq(r.injury_event_id)]
            opportunities.append(dict(injury_event_id=r.injury_event_id,player_id=r.player_id,player_name=r.player_name,partition=part,
                event_date=r.prediction_reference_date,all_outings_previous_30d=len(rows),eligible_outings_previous_30d=len(opportunity),
                eligible_outings_7_30d=int(opportunity.dynamic_lead_days.between(7,30).sum()),
                reason='observable' if len(opportunity) else ('no MLB outing in prior 30 days in retained partition' if rows.empty else 'quality/history/active episode or intervening-event exclusion')))
        if part not in ['validation','test']:continue
        g=g.loc[g.analysis_eligible].copy()
        for name in list(dict.fromkeys([freeze['selected_model'],'Profile','Previous 62','Stable 62'])):
            obj=joblib.load(CACHE/'models'/(model_path(30,name)+'.joblib'));g['score']=scores(obj,g)
            g['alert']=g.score.ge(freeze['thresholds']['30|'+name]);prior=g.groupby('pitcher_id').alert.shift(fill_value=False)
            successive=g.groupby('pitcher_id').outing_index.diff().eq(1)&g.groupby('pitcher_id').game_date.diff().dt.days.le(30)
            repeat=g.alert&prior&successive
            known_false=g.alert&g.target.eq(0);unknown=g.alert&g.target.isna()
            eligible=g.dynamic_next_target&g.dynamic_next_event_id.isin(ev.injury_event_id)&g.dynamic_lead_days.between(7,30)
            hits=g.loc[g.alert&eligible,'dynamic_next_event_id'].nunique()
            repeated_hits=g.loc[repeat&eligible,'dynamic_next_event_id'].nunique()
            op.append(dict(partition=part,model=name,all_prospectively_eligible_rows=len(g),alerts=int(g.alert.sum()),actual_alert_rate=float(g.alert.mean()),
                label_observed_rows=int(g.target.notna().sum()),unknown_outcome_alerts=int(unknown.sum()),known_non_ucl_alerts=int(known_false.sum()),
                known_ucl_within_30d_alerts=int((g.alert&g.target.eq(1)).sum()),qualifying_events=len(ev),warned_events_7_30d=hits,
                repeat_alerts_consecutive_actual_outings=int(repeat.sum()),two_consecutive_warning_events=repeated_hits,
                two_consecutive_alert_rate=float(repeat.mean()),
                burden_note='All prospectively eligible appearances; unknown outcomes preserved. Two-consecutive rule is a descriptive untuned sensitivity.'))
    table(pd.DataFrame(op),'operational_all_landmark_burden');table(pd.DataFrame(opportunities),'event_observation_opportunities')

def make_figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    m=read_table('model_metrics');w=read_table('warning_metrics');e=read_table('ucl_endpoint_adjudication')
    names=list(SHORT);freeze=json.loads((OUT/'frozen_protocol.json').read_text());selected=freeze['selected_model']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    count=e.groupby('endpoint_partition')[['endpoint_primary','endpoint_strict_official','endpoint_expanded']].sum().reindex(['train','validation','test','descriptive_2026'])
    count.plot.bar(ax=axes[0],color=['#0072B2','#009E73','#E69F00'],rot=0)
    axes[0].set_xticklabels(['2015–21','2022–23','2024–25','2026*']);axes[0].set(xlabel='',ylabel='Eligible episodes',title='UCL evidence definitions')
    axes[0].legend(['Primary documented','Official diagnosis only','Including probable'],fontsize=8)
    opp=read_table('event_observation_opportunities');counts=opp.groupby('partition').agg(all_events=('injury_event_id','size'),observed=('eligible_outings_7_30d',lambda x:(x>0).sum())).reindex(['train','validation','test'])
    counts.plot.bar(ax=axes[1],color=['#7d8491','#D55E00'],rot=0);axes[1].set_xticklabels(['2015–21','2022–23','2024–25'])
    axes[1].set(xlabel='',ylabel='Episodes',title='A seven-day warning is not always observable');axes[1].legend(['All eligible events','Has 7–30d opportunity'])
    fig.suptitle('Specificity and observation reduce the usable UCL sample');fig.text(.5,-.025,'*2026 incomplete/descriptive. Source documentation is not independent clinical adjudication.',ha='center',fontsize=9)
    fig.tight_layout();figure(fig,'01_endpoint_and_opportunity');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    for ax,part,title in zip(axes,['validation','test'],['Validation: 2022–2023','Exploratory transfer: 2024–2025']):
        for n,color in zip(names,COLORS):
            g=m.loc[m.partition.eq(part)&m.model.eq(n)].sort_values('horizon');ax.plot(g.horizon,g.pr_lift,'o-',label=SHORT[n],color=color,lw=1.5,ms=4)
        ax.axhline(1,color='grey',ls='--');ax.set(title=title,xlabel='Future horizon (days)',xticks=CFG['horizons'])
    axes[0].set_ylabel('Average precision / outcome prevalence');axes[1].legend(bbox_to_anchor=(1.01,1),fontsize=8)
    fig.suptitle('Broader models rank UCL risk better; shorter horizons have different risk sets');fig.tight_layout();figure(fig,'02_horizon_discrimination');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    show=['Profile','Departure','Profile + departure + trajectory','Previous 62','Stable 62']
    for ax,part in zip(axes[:2],['validation','test']):
        for n in show:
            g=w.loc[w.partition.eq(part)&w.horizon.eq(30)&w.model.eq(n)&w.alert_rate.eq(.05)].sort_values('minimum_lead')
            ax.plot(g.minimum_lead,100*g.injury_event_recall,'o-',label=SHORT[n],color=COLORS[names.index(n)])
        ax.set(title='Validation' if part=='validation' else 'Exploratory transfer',xlabel='Minimum warning lead (days)',xticks=[7,14,21],ylabel='All-event recall (%)',ylim=(0,40))
    axes[1].legend(fontsize=8)
    g=w.loc[w.horizon.eq(30)&w.model.eq(selected)&w.alert_rate.eq(.05)&w.minimum_lead.eq(7)]
    axes[2].bar(['Validation','Transfer'],g.non_actionable_alert_fraction*100,color=['#0072B2','#D55E00']);axes[2].set(ylim=(0,100),ylabel='Alerts outside a 7–30d warning (%)',title='Departure alert burden')
    for i,r in enumerate(g.itertuples()):axes[2].text(i,50,f'{r.alerted_events}/{r.qualifying_events} events\n{r.alerts:,} alerts',ha='center',color='white',weight='bold')
    fig.suptitle('At a 5% appearance alert budget, useful warnings remain sparse');fig.tight_layout();figure(fig,'03_warning_lead_and_burden');plt.close(fig)
    t=read_table('risk_trajectory_summary');fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,(n,mode) in enumerate([(selected,'outings'),(selected,'calendar'),('Profile + departure + trajectory','outings'),('Profile + departure + trajectory','calendar')]):
        ax=axes.flat[j];g=t.loc[t.model.eq(n)&t['mode'].eq(mode)&t.partition.eq('combined')]
        for kind,color in [('Injury','#D55E00'),('Control','#0072B2')]:
            s=g.loc[g.kind.eq(kind)].sort_values('point');x=s.point.to_numpy();ax.plot(x,s['mean']*100,'o-',color=color,label=kind)
            ax.fill_between(x,s.simultaneous_low.to_numpy()*100,s.simultaneous_high.to_numpy()*100,color=color,alpha=.15)
        labels=['−5','−4','−3','−2','−1'] if mode=='outings' else ['31–60','15–30','8–14','1–7']
        ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Predicted 30-day UCL risk (%)',xlabel='Outings before event' if mode=='outings' else 'Days before event',title=f'{SHORT[n]}: {int(g.events.iloc[0])} balanced pairs')
        ax.legend(fontsize=8)
    fig.suptitle('Matched risk trajectories do not show a uniform approach-to-injury rise\n2022–2025 development; 95% simultaneous bands by connected pitcher clusters',fontsize=12)
    fig.tight_layout();figure(fig,'04_matched_risk_trajectories');plt.close(fig)
    cells=read_table('balanced_risk_cells');g=cells.loc[cells.model.eq(selected)&cells['mode'].eq('outings')]
    values=g.pivot(index='event_id',columns='point',values='Injury');order=(values[4]-values[0]).sort_values().index;values=values.loc[order]
    fig,ax=plt.subplots(figsize=(7,8));im=ax.imshow(values.to_numpy()*100,aspect='auto',cmap='viridis',vmin=0,vmax=np.quantile(values.to_numpy(),.98)*100)
    ax.set(xticks=range(5),xticklabels=['−5','−4','−3','−2','−1'],xlabel='Outings before recorded UCL event',ylabel=f'{len(values)} individual events (ordered by final − first risk)',yticks=[])
    ax.set_title('Departure risk paths vary substantially across UCL events\nOne shared probability scale; colors capped at the 98th percentile',fontsize=11)
    fig.colorbar(im,ax=ax,label='Predicted 30-day risk (%)');fig.tight_layout();figure(fig,'05_individual_risk_heterogeneity');plt.close(fig)
    c=read_table('calibration_bins');ref=read_table('training_refit_metrics');fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for n in ['Profile','Departure','Profile + departure + trajectory','Stable 62']:
        g=c.loc[c.model.eq(n)&c.partition.eq('validation')];axes[0].plot(g.mean_score*100,g.observed_risk*100,'o-',label=SHORT[n],color=COLORS[names.index(n)])
    maxv=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([0,maxv],[0,maxv],'--',color='grey');axes[0].set(xlabel='Mean predicted risk (%)',ylabel='Observed risk (%)',title='Validation calibration (training-only fits)');axes[0].legend(fontsize=8)
    names_ref=ref.model.unique();axes[1].boxplot([ref.loc[ref.model.eq(n),'injury_event_recall']*100 for n in names_ref],tick_labels=[SHORT[n] for n in names_ref],vert=False)
    axes[1].set(xlabel='Validation event recall, 7–30d (%)',title='Eight whole-pitcher training refits')
    fig.tight_layout();figure(fig,'06_calibration_and_refit_stability');plt.close(fig)

def validate_stage():
    a,e,s=prepare_data();before=snapshot();checks=[]
    def check(name,value,detail=''):checks.append(dict(check=name,passed=bool(value),detail=detail))
    check('Database and primary source signatures preserved',before['source']==source_signature())
    changed=[p for p,h in before['artifacts'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    check('Prior reports, notebooks, models and frozen lists preserved',not changed,';'.join(changed))
    check('Unique appearance keys',not a[['pitcher_id','game_id']].duplicated().any())
    check('Unique canonical episode IDs',e.injury_event_id.is_unique)
    for col in ['history_max_date','baseline_max_date','workload_max_date','dynamic_history_max_date']:
        check(col+' is strictly prior',a.loc[a[col].notna(),col].lt(a.loc[a[col].notna(),'game_date']).all())
    check('Primary events have valid timing and documented evidence',e.loc[e.endpoint_primary,'timing_usable'].all() and e.loc[e.endpoint_primary,'evidence_category'].str.startswith('confirmed_').all())
    procedure=e.loc[e.endpoint_primary&e.procedure_link_valid,'linked_procedure_ids'].str.split(';').explode()
    check('A procedure cannot support two primary episodes',not procedure.duplicated().any())
    for h in CFG['horizons']:
        f=risk_frame(a,e,h);pos=f.loc[f.target.eq(1)&f.partition.isin(['train','validation','test'])]
        check(f'{h}d positive events confined to one partition',pos.groupby('dynamic_next_event_id').partition.nunique().le(1).all())
        check(f'{h}d positives strictly before next event',pos.dynamic_lead_days.between(1,h).all())
        check(f'{h}d unknown episode not negative',f.loc[f.label_status.eq('censored_ambiguous_or_excluded_episode'),'target'].isna().all())
    freeze=json.loads((OUT/'frozen_protocol.json').read_text())
    check('Freeze precedes first stored transfer predictions',pd.Timestamp(freeze['frozen_at_utc']).timestamp()<(CACHE/'predictions.parquet').stat().st_mtime)
    for n in s:
        obj=joblib.load(CACHE/'models'/(model_path(30,n)+'.joblib'))
        check(n+' fit ends in training',pd.Timestamp(obj['training_max_date'])<=pd.Timestamp(CFG['train_end']))
        check(n+' matches frozen fit signature',obj['signature']==freeze['fit_signatures']['30|'+n])
    aligned=pd.read_parquet(CACHE/'matched_risk_trajectories.parquet')
    check('Matched alignment unique',not aligned[['event_id','kind','game_id','pitcher_id']].duplicated().any())
    check('Matched appearances strictly pre-event',aligned.lead_days.between(1,60).all())
    check('No 2026 modeling or trajectory evaluation',aligned.game_date.dt.year.max()<=2025 and read_table('model_metrics').partition.isin(['validation','test']).all())
    for stage in ['endpoint','data','models','robustness','trajectories']:check(stage+' checkpoint hashes verified',valid_checkpoint(stage))
    for p in (CACHE/'models').glob('*.joblib'):joblib.load(p)
    check('No partial artifacts',not list(CACHE.rglob('*.partial')) and not list(OUT.rglob('*.partial')))
    table(pd.DataFrame(checks),'validation_checks')
    if not all(r['passed'] for r in checks):raise AssertionError([r for r in checks if not r['passed']])
    return checks

def write_summary():
    m=read_table('model_metrics');w=read_table('warning_policies');e=read_table('ucl_endpoint_adjudication');shape=read_table('risk_trajectory_shapes')
    freeze=json.loads((OUT/'frozen_protocol.json').read_text());selected=freeze['selected_model']
    summary=m.loc[m.horizon.eq(30),['partition','model','rows','positive_rows','represented_events','average_precision','pr_lift','brier_score']]
    warning=w.loc[w.minimum_lead.eq(7)&w.policy.eq('retrospective_budget'),['partition','model','qualifying_events','observable_events','alerted_events','injury_event_recall','median_warning_lead_days','alerts','non_actionable_alert_fraction']]
    table(summary,'summary_model_comparison');table(warning,'summary_warning_comparison')
    text=f'''# UCL endpoint refinement and dynamic early warning

Completed from the interrupted schema/evidence review, reusing all completed feature-consolidation artifacts. No prior numerical stage was rerun or replaced. This is development and exploratory transfer analysis, not prospective confirmation.

## Endpoint and cohort

The primary endpoint contains **{int(e.endpoint_primary.sum())} eligible documented UCL episodes** (98 training, 34 validation, 43 exploratory transfer, 19 descriptive 2026). Official non-procedure UCL diagnosis supports 60; the other 134 require a bounded, identifier-linked UCL procedure with compatible anatomy, no intervening MLB return, and no incompatible episode boundary. There are 221 eligible events when probable evidence is included. Twenty-two compatible forearm-to-UCL progressions were retained with explicit source adjudication. Generic elbow pain, unconfirmed forearm injury, postoperative entries and unresolved episode ambiguity are not automatically UCL outcomes.

The date is the earliest canonical reported/retroactive/effective injury proxy, never a substituted surgery date. Biological onset is generally unknown. Source retrieval occurred retrospectively; contemporaneous archived publication was not independently verified. The strict official sensitivity has only seven validation events, of which three have labeled appearances, and is underpowered.

There are 233,433 source appearances. The 30-day labeled risk sets contain 87,332 training, 28,875 validation and 29,349 exploratory-transfer appearances. Positive events represented are 59, 25 and 28; not every eligible UCL episode has a preceding MLB appearance. At 7–30 days, only 20/34 validation and 28/43 transfer events have an eligible warning opportunity. All eligible endpoint events remain in the principal recall denominator.

## Modeling and findings

At each completed appearance, fixed L2 logistic landmark models estimate first-event UCL risk over 7, 14, 21 or 30 days. These are separate horizon models, not a coherent survival curve. They preserve profile, departure-8, prior-62 and stable-62 representations. Twelve new trajectory fields and three standardized profile×departure interactions test dynamic additions. Missing values are imputed only inside training-fitted preprocessing; source data remain nullable.

The predeclared validation warning rule selected **{selected}** before any new 2024–2025 predictions. It detected **7/34 validation events (20.6%)** at 7–30 days and a 5% appearance budget, versus **2/43 (4.7%)** in exploratory transfer. Conditional median first-warning lead was 12 and 14 days. These medians describe detected events only. The result does not establish dependable warning.

Profile alone had validation AP 0.00612 versus 0.00727 for departure, 0.00737 combined, 0.00769 with trajectories and 0.00789 with interactions. Paired uncertainty includes zero for the profile/departure and trajectory AP increments. The small interaction AP increment is positive under conditional validation resampling, but adds no validation warning events; training-refit uncertainty and multiple exploratory comparisons limit interpretation.

Stable-62 achieved validation AP 0.01092 (2.28× prevalence) and transfer AP 0.01846 (3.82×). Its 7–30d event warnings were 3/34 and 9/43. Previous-62 warned on 3/34 and 9/43. Better ranking and better practical warning are distinct, and transfer results must not retroactively change the selected candidate.

Adding trajectories to profile+departure left validation warnings at 3/34; exploratory-transfer warnings rose from 5/43 to 6/43. Interactions added no warnings in either period. The evidence does not justify claiming a robust incremental early-warning benefit.

Matched risk plots use 33 balanced five-outing pairs and 16 balanced calendar-window pairs, unchanged existing controls where available, and same-stratum matches only for newly eligible episodes. Whole dependency-cluster intervals account for shared case/control pitchers. Average departure risk does not show a consistent rise as injury approaches; individual paths are heterogeneous. A flat group offset is a profile association, not evidence of imminence. No latent or risk score is interpreted as tissue damage. All three interaction coefficient signs vary across the eight training refits, weakening any interpretation of the small AP increment.

## Warning burden, uncertainty and robustness

More than 98% of labeled alerts are negative for the first-event UCL outcome within 30 days in the principal departure comparison. This includes known competing first injuries, not a claim of complete medical surveillance. Unknown/censored outcomes are excluded from labeled performance, not called healthy. `operational_all_landmark_burden.csv` separately applies the frozen numeric threshold to every prospectively eligible appearance and retains unknown alert outcomes: 1,980 validation alerts include 536 unknown outcomes; 1,875 transfer alerts include 518. Numeric-threshold alert rates can drift away from 5%; retrospective top-N budgets are not deployment thresholds.

Whole-pitcher paired resampling, eight whole-pitcher training refits, endpoint expansion, strict official diagnoses, public-date timing, competing-event censoring, exclusion of the last six days, complete trajectory support, role and unseen-pitcher comparisons are reported. Subgroup analyses use explicitly identified observable-event denominators. The strict/unseen subsets are too small to establish generalization. No formal universal change point is supported by the irregular and limited event histories.

Known non-UCL first injuries are competing negatives for the primary cumulative first-event question. Suspected UCL/ambiguous arm episodes are censored. Censoring all known competitors is a sensitivity analysis. No alert may skip an intervening episode to receive credit for a later UCL event.

Endpoint specificity and the risk set changed from the previous broad Elbow/UCL analysis, so its AP and warning counts are not directly comparable to these results. The same-endpoint 30-day frozen-feature benchmarks provide the comparison here. Multi-horizon modeling has not established an independent improvement over ordinary 30-day appearance-level classification.

## Frozen next step

Keep the validation-selected departure-8 model, profile and prior-62 comparators, and all prespecified model artifacts frozen in `frozen_protocol.json`; do not retune them on transfer outcomes. Retain stable-62 as the already specified broader comparison, not a newly validated winner. Acquire a genuinely post-freeze chronological cohort with contemporaneous source snapshots, complete outcome follow-up and adjudicated UCL episodes. Use the frozen validation-derived threshold, report realized alert burden, and audit injuries and dates blind to prediction scores. Current results warrant further study, not clinical deployment or a confirmatory performance claim.

## Files

- `notebooks/ucl_dynamic/`: endpoint/cohort; models/warnings; trajectories/robustness.
- `reports/ucl_dynamic/tables/`: complete adjudication, evidence links, event opportunities, comparisons, bootstrap/refit uncertainty, sensitivities, calibration and QA.
- `reports/ucl_dynamic/figures/`: six PNG/PDF figures, numbered 01–06.
- `data/injury/analysis/ucl_dynamic/`: immutable-source derived data, predictions and fitted model caches.
- `reports/ucl_dynamic/frozen_protocol.json`: exact representations, fit signatures, validation thresholds and freeze timestamp.

Rerun/resume: `.venv/bin/python -m src.injury_analysis.ucl_dynamic_pipeline` then `.venv/bin/python scripts/run_injury_analysis_notebooks.py --suite ucl-dynamic`. Checkpoint hashes fail loudly on changed inputs. The database is opened read-only.
'''
    (OUT/'summary.md').write_text(text)
    figure_map=[('01_endpoint_and_opportunity','Which UCL events can be evaluated?','Specificity and missing pre-event MLB opportunities reduce usable N.','Endpoint ascertainment'),
        ('02_horizon_discrimination','Does dynamic representation improve risk ranking?','Broader benchmarks rank better; horizons differ in prevalence and follow-up.','Mixed profile/departure'),
        ('03_warning_lead_and_burden','How early and at what burden?','Selected departure warnings fail to transfer consistently; false/nonactionable burden is large.','Warning usefulness'),
        ('04_matched_risk_trajectories','Does risk rise before UCL injury?','No uniform rising departure pattern; support and uncertainty are explicit.','Persistent versus changing risk'),
        ('05_individual_risk_heterogeneity','Are average paths universal?','Individual UCL trajectories vary in direction and risk level.','Heterogeneous departures'),
        ('06_calibration_and_refit_stability','Are predictions calibrated and stable?','Rare outcomes and pitcher resampling limit confidence in apparent gains.','Model stability')]
    table(pd.DataFrame(figure_map,columns=['figure','question','interpretation','research_role']),'figure_guide')

def run_reporting():
    if valid_checkpoint('reporting'):return
    operational_audit();make_figures();validate_stage();write_summary()
    checkpoint('reporting',[OUT/'summary.md',OUT/'tables/validation_checks.csv',OUT/'tables/operational_all_landmark_burden.csv']+list((OUT/'figures').glob('*')))
