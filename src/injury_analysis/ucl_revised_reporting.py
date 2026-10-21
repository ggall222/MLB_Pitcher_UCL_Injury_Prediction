"""Refresh audit and frozen chronological results; never select using test scores."""
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import sha
from .ucl_revised_common import ROOT,OUT,CACHE,CFG,table,read_table,figure,write_json,checkpoint,valid_checkpoint
from .ucl_revised_data import prepare,frame
from .ucl_revised_evaluation import frozen_protocol
from .ucl_revised_models import load


def refresh_audit():
    """Read-only central DB audit, including survival of every old raw pitch key."""
    baseline=json.loads((OUT/'refresh_baseline.json').read_text())
    with duckdb.connect(str(ROOT/CFG['database_path']),read_only=True) as c:
        counts={t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in baseline['before_counts']}
        dates=c.execute('SELECT min(game_date),max(game_date) FROM pitches').fetchone()
        fail=c.sql("SELECT * FROM ingestion_state WHERE status='failed'").df()
        missing=c.sql("""SELECT g.game_id,g.game_date,g.status FROM games g
            WHERE g.season=2026 AND g.game_type='R' AND g.status='Final'
            AND NOT EXISTS(SELECT 1 FROM pitches p WHERE p.game_id=g.game_id)""").df()
        table(fail,'refresh_failed_sources');table(missing,'refresh_missing_games')
        coverage=c.sql('''WITH p AS (SELECT year(game_date) season,count(*) pitches FROM pitches GROUP BY 1),
            a AS (SELECT season,count(*) appearances,sum(CASE WHEN is_starter THEN 1 ELSE 0 END) starters,
              count(DISTINCT pitcher_id) pitchers FROM appearances GROUP BY 1)
            SELECT * FROM p JOIN a USING(season) ORDER BY season''').df()
        table(coverage,'refreshed_season_coverage')
        c.execute("ATTACH '"+baseline['snapshot'].replace("'","''")+"' AS before_refresh (READ_ONLY)")
        absent=c.execute('''SELECT count(*) FROM before_refresh.pitches b ANTI JOIN main.pitches p
            USING(game_id,at_bat_number,pitch_number)''').fetchone()[0]
    result=dict(before=baseline['before_counts'],after=counts,
        added={t:counts[t]-baseline['before_counts'][t] for t in counts},
        first_pitch_date=str(dates[0]),latest_pitch_date=str(dates[1]),
        database_bytes=(ROOT/CFG['database_path']).stat().st_size,
        failed_sources=len(fail),missing_2026_final_games=len(missing),old_raw_pitch_keys_missing=absent,
        outcome_observation_end=CFG['observation_end'],snapshot=baseline['snapshot'])
    write_json(result,OUT/'refresh_summary.json');return result


def markdown(frame):
    """Small markdown tables without introducing an optional tabulate dependency."""
    def fmt(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None)])


def short(mid):
    return mid.replace('Profile_departure_trajectory','P+D+trajectory').replace('Profile_departure','P+D').replace('__',' / ').replace('_',' ').replace('1p0','1').replace('0p1','.1').replace('10p0','10')


def run_figures():
    p=frozen_protocol();m=read_table('model_metrics');u=read_table('performance_uncertainty')
    names=list(dict.fromkeys([p['selected_primary_model'],p['selected_ap_model'],
        'Profile__Linear__1p0','Departure__Linear__1p0','Profile_departure__Linear__1p0']))
    names=[n for n in names if n in m.model_id.unique()]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5));y=np.arange(len(names))
    for offset,part,color in [(-.14,'validation','#1c6e91'),(.14,'test','#d36d32')]:
        x=m.loc[m.partition.eq(part)&m.horizon.eq(30)].set_index('model_id').loc[names]
        ci=u.loc[u.partition.eq(part)].set_index('model_id').loc[names]
        ax.errorbar(x.pr_lift,y+offset,xerr=np.vstack([np.maximum(0,x.pr_lift-ci.lift_ci_low),np.maximum(0,ci.lift_ci_high-x.pr_lift)]),
            fmt='o',capsize=3,color=color,label='2024–2025 validation' if part=='validation' else '2026 test')
    ax.set(yticks=y,yticklabels=[short(n) for n in names],xlabel='Average Precision / cohort prevalence',title='Chronological generalization of frozen UCL models')
    ax.axvline(1,color='gray',ls=':');ax.legend();fig.text(.5,-.02,'300 whole-pitcher bootstrap intervals; conditional on fitted models. 2026 never used for selection.',ha='center',fontsize=9)
    figure(fig,'01_chronological_discrimination');plt.close(fig)
    w=read_table('warning_metrics');s=w.loc[w.horizon.eq(30)&w.minimum_lead.eq(7)&w.model_id.eq(p['selected_primary_model'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,part in zip(axes,['validation','test']):
        for policy,color,label in [('retrospective_budget','#1c6e91','Retrospective exact budget'),('frozen_validation_threshold','#d36d32','Frozen validation cutoff')]:
            g=s.loc[s.partition.eq(part)&s.policy.eq(policy)].sort_values('alert_rate')
            ax.plot(100*g.actual_alert_rate,100*g.injury_event_recall,'o-',color=color,label=label)
        n=int(s.loc[s.partition.eq(part),'qualifying_events'].iloc[0]);ax.set(title=f'{part.title()} · {n} qualifying events',xlabel='Observed-outcome appearances alerted (%)')
    axes[0].set_ylabel('UCL events warned 7–30 days ahead (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Fixed thresholds and fixed alert budgets answer different questions');figure(fig,'02_warning_policy');plt.close(fig)
    f=read_table('followup_by_month');f=f.loc[f.partition.eq('test')&f.horizon.eq(30)]
    f['outcome']=np.select([f.label_status.eq('ucl_first_event'),f.label_status.isin(['observed_event_free_horizon','non_ucl_competing_first_event'])],['Positive','Observed negative'],default='Unknown / censored')
    z=f.pivot_table(index='month',columns='outcome',values='rows',aggfunc='sum',fill_value=0)
    fig,ax=plt.subplots(figsize=(9,4));z.plot.bar(stacked=True,ax=ax,color={'Positive':'#d36d32','Observed negative':'#1c6e91','Unknown / censored':'#aab3b7'})
    ax.set(title='2026 follow-up limits the evaluable 30-day cohort',ylabel='Eligible appearance landmarks',xlabel='Appearance month');ax.tick_params(axis='x',rotation=30)
    figure(fig,'03_followup_censoring');plt.close(fig)
    tw=read_table('training_window_comparison');tw=tw.loc[tw.model_id.isin(names)]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mid,g in tw.groupby('model_id'):
        g=g.set_index('training_window').loc[['2015-2021','2015-2023']]
        axes[0].plot([0,1],g.average_precision,'o-',label=short(mid));axes[1].plot([0,1],100*g.injury_event_recall,'o-')
    for ax in axes:ax.set(xticks=[0,1],xticklabels=['2015–2021 train','2015–2023 train'])
    axes[0].set_ylabel('Validation Average Precision');axes[1].set_ylabel('Validation 7–30d event recall at 5% (%)')
    axes[0].legend(fontsize=7);fig.suptitle('More training history, identical 2024–2025 validation rows')
    figure(fig,'04_training_expansion');plt.close(fig)
    cal=read_table('calibration_bins');cal=cal.loc[cal.horizon.eq(30)&cal.model_id.eq(p['selected_primary_model'])]
    fig,ax=plt.subplots(figsize=(6,5))
    for part,color in [('validation','#1c6e91'),('test','#d36d32')]:
        g=cal.loc[cal.partition.eq(part)];ax.scatter(g.predicted,g.observed,s=np.clip(g.rows/25,12,160),color=color,label=part.title());ax.plot(g.predicted,g.observed,color=color,alpha=.5)
    lim=max(cal.predicted.max(),cal.observed.max())*1.1;ax.plot([0,lim],[0,lim],':',color='gray');ax.set(xlabel='Mean predicted 30-day risk',ylabel='Observed UCL fraction',title='Calibration with validation-defined bins');ax.legend()
    figure(fig,'05_calibration');plt.close(fig)


def run_qa():
    p=frozen_protocol();a,e,_=prepare();checks=[]
    def check(name,passed,detail=''):checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    refresh=refresh_audit()
    check('all old raw pitch keys retained',refresh['old_raw_pitch_keys_missing']==0)
    check('source retries complete',refresh['failed_sources']==0)
    check('2026 final regular games present',refresh['missing_2026_final_games']==0)
    dbcheck=json.loads((OUT/'refresh/validation_report.json').read_text());check('database validation',dbcheck['passed'])
    check('unique appearance landmarks',not a.duplicated(['pitcher_id','game_id']).any())
    for col in ['history_max_date','workload_max_date','baseline_max_date','dynamic_history_max_date']:
        check(col+' strictly prior',bool((a[col].isna()|a[col].lt(a.game_date)).all()))
    before=json.loads((OUT/'prior_artifacts.json').read_text());changed=[name for name,h in before.items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
    check('prior analysis artifacts unchanged',not changed,f'{len(before)} files; changed={changed}')
    access=json.loads((OUT/'test_access.json').read_text())
    check('freeze precedes test scoring',pd.Timestamp(p['frozen_at_utc'])<pd.Timestamp(access['first_scored_at_utc']))
    check('freeze unchanged after test scoring',sha(OUT/'frozen_protocol.json')==access['freeze_sha256'])
    check('test adaptation disabled',p['test_adaptation_allowed'] is False)
    for h,mid in p['evaluation_pairs']:
        obj=load(h,mid);enc=obj['encoder'];base=getattr(enc,'base',enc)
        check(f'H{h} {mid} training date',pd.Timestamp(obj['training_max_date'])<=pd.Timestamp(CFG['train_end']))
        check(f'H{h} {mid} preprocessing date',pd.Timestamp(base.fit_max_date)<=pd.Timestamp(CFG['train_end']))
        check(f'H{h} {mid} no outcome predictors',not any(c.startswith(('endpoint_','ucl_injury_next','dynamic_next')) or c in ['target','activation_date','surgery_date'] for c in obj['spec']['features']))
    for h in CFG['horizons']:
        f=frame(a,e,h)
        check(f'H{h} unknown outcomes not negatives',f.loc[f.label_status.str.startswith('censored'),'target'].isna().all())
        check(f'H{h} event-free negatives observed',f.loc[f.label_status.eq('observed_event_free_horizon'),'dynamic_followup'].all())
        check(f'H{h} modeled rows eligible',f.loc[f.target.notna(),'analysis_eligible'].all())
        for part,start,end in [('validation','2024-01-01','2025-12-31'),('test','2026-01-01','2026-12-31')]:
            pp=pd.read_parquet(CACHE/f'predictions_{part}_h{h}.parquet')
            check(f'H{h} {part} dates',pp.game_date.between(start,end).all())
            check(f'H{h} {part} known labels',pp.target.notna().all())
    check('no partially written modeling artifacts',not list(CACHE.rglob('*.partial')))
    result=table(pd.DataFrame(checks),'validation_checks')
    write_json(dict(passed=bool(result.passed.all()),checks=len(result),prior_artifacts_verified=len(before),
        database_validation_passed=dbcheck['passed'],raw_pitch_keys_preserved=True),OUT/'validation_summary.json')
    if not result.passed.all():raise AssertionError(result.loc[~result.passed].to_dict('records'))
    return result


def run_summary():
    p=frozen_protocol();m=read_table('model_metrics');w=read_table('warning_metrics');c=read_table('cohort_support')
    key=p['selected_primary_model'];rank=p['selected_ap_model'];refresh=json.loads((OUT/'refresh_summary.json').read_text())
    selected=m.loc[m.horizon.eq(30)&m.model_id.isin([key,rank])]
    table(selected,'selected_model_results')
    t=[]
    for h,mid in p['evaluation_pairs']:
        if h==30:t.append(dict(model=mid,role='primary warning' if mid==key else 'secondary AP' if mid==rank else 'pre-frozen comparator',threshold_5pct=p['thresholds'][f'30|{mid}']['0.05']))
    table(pd.DataFrame(t),'frozen_models_and_thresholds')
    core=c.loc[c.horizon.eq(30)].set_index('partition');test=core.loc['test'];tr=core.loc['train'];val=core.loc['validation']
    v=m.loc[m.horizon.eq(30)&m.model_id.eq(key)&m.partition.eq('validation')].iloc[0]
    te=m.loc[m.horizon.eq(30)&m.model_id.eq(key)&m.partition.eq('test')].iloc[0]
    primary_w=w.loc[w.horizon.eq(30)&w.model_id.eq(key)&w.minimum_lead.eq(7)&w.alert_rate.eq(.05)]
    tw=read_table('training_window_comparison');tw=tw.loc[tw.model_id.eq(key)].set_index('training_window')
    old=tw.loc['2015-2021'];new=tw.loc['2015-2023']
    coverage=read_table('feature_coverage');bad=coverage.loc[coverage.partition.eq('test')&coverage.missing_fraction.gt(.1)].sort_values('missing_fraction',ascending=False)
    burden=read_table('operational_alert_burden');burden=burden.loc[burden.horizon.eq(30)&burden.model_id.eq(key)]
    q=json.loads((OUT/'validation_summary.json').read_text())
    text=f'''# Revised UCL chronological evaluation

Train **2015–2023**; validation **2024–2025**; final chronological test **2026**. Source cutoff: **{CFG['observation_end']}**. All 59 candidate model/representation settings were fitted using training data only. Validation selected models and thresholds. Frozen protocol timestamp: {p['frozen_at_utc']}. Test results never changed those decisions.

## Refresh and preservation

The database now contains **{refresh['after']['pitches']:,} pitches** and **{refresh['after']['appearances']:,} appearances**, through **{refresh['latest_pitch_date']}**, with {refresh['added']['pitches']:,} new pitches and {refresh['added']['appearances']:,} new appearances. All previous raw pitch keys and {q['prior_artifacts_verified']:,} prior analysis artifacts were retained. Raw/API caches are preserved; refreshed general injuries use a new versioned directory. The old database is retained at `{refresh['snapshot']}`. A stale empty boxscore cached for a postponed game was re-fetched before any development fits; ingestion now bypasses failed cached boxscores. There are {refresh['failed_sources']} outstanding source failures and {refresh['missing_2026_final_games']} missing final regular-season games.

## Expanded training and validation

The revised 30-day model has **{int(tr.rows):,} training rows**, **{int(tr.positive_rows):,} positives**, and **{int(tr.represented_events)} represented UCL events**. Validation contains **{int(val.rows):,} observed-outcome rows**, **{int(val.positive_rows)} positives**, and **{int(val.represented_events)} represented events**; its all-event warning denominator is {int(val.qualifying_events)}.

For the eventual primary model, expanding training from 2015–2021 to 2015–2023 changes the training sample from {int(old.training_rows):,} to {int(new.training_rows):,}. On identical 2024–2025 validation rows, AP changes **{old.average_precision:.4f} → {new.average_precision:.4f}**, and 7–30-day event recall at a retrospective 5% alert budget changes **{old.injury_event_recall:.1%} → {new.injury_event_recall:.1%}**. This paired diagnostic fixes model settings; it is not a separate historical test. See `training_window_uncertainty.csv` for whole-pitcher uncertainty.

## Frozen models

Primary warning model: **{short(key)}**. Secondary AP model: **{short(rank)}**. Primary 30-day numeric alert threshold: **{p['thresholds'][f'30|{key}']['0.05']:.10g}** (the validation 95th score percentile). All fitted preprocessors, feature lists, model files, numeric thresholds, horizon settings and comparison models are SHA-registered in `frozen_protocol.json` before `test_access.json`.

{markdown(selected[['partition','model_id','rows','positive_rows','average_precision','pr_lift','roc_auc','brier_score','mean_predicted_risk','prevalence']])}

## Early warnings and generalization

{markdown(primary_w[['partition','policy','actual_alert_rate','alerted_events','qualifying_events','observable_events','injury_event_recall','median_warning_lead_days','non_actionable_alert_fraction']])}

Primary AP changes **{v.average_precision:.4f} → {te.average_precision:.4f}** from validation to test, while PR lift changes **{v.pr_lift:.2f} → {te.pr_lift:.2f}**. AP levels depend on prevalence: validation prevalence is {v.prevalence:.3%}, test prevalence {te.prevalence:.3%}. Judge temporal generalization using both lift and warning performance, with bootstrap intervals, rather than interpreting a higher AP alone as improvement. No recalibration or threshold adjustment was performed on 2026.

The exact-budget curve is retrospective ranking performance; its test top-5% cutoff is **not** a deployable retuning. The separate numeric validation threshold is the frozen operational policy. Applying it to every eligible landmark yields:

{markdown(burden[['partition','eligible_landmarks','alerts','actual_alert_rate','actionable_alerts','alerted_events','unknown_outcome_alerts','observed_nonactionable_alerts']])}

## 2026 cohort and incomplete follow-up

2026 contributes **{int(test.appearance_rows):,} regular-season pitcher appearances**, **{int(test.eligible_landmarks):,} eligible landmarks**, and **{int(test.rows):,} observed-outcome 30-day rows** from **{int(test.pitchers)} pitchers**. There are **{int(test.positive_rows)} positive rows**, **{int(test.represented_events)} represented UCL events**, and **{int(test.qualifying_events)} qualifying calendar-year events**. **{int(test.events_with_7d_warning_opportunity)} events** have at least one eligible 7–30-day warning opportunity (**{int(test.warning_opportunity_rows)} appearances**). Events without an observable opportunity remain in the all-event denominator.

**{int(test.unknown_outcome_rows):,} eligible 2026 landmarks remain unknown/censored**, not negatives. Latest event-free negative landmark: {test.last_eventfree_negative_date}. A negative needs observed subsequent MLB participation through the horizon within the existing 30-day grace, as well as sufficient injury-source follow-up; the end of the regular season further limits this. A documented competing non-UCL injury is a known first-event competing outcome, distinct from observed event-free follow-up. Ambiguous excluded episodes remain censored. Source coverage through October 20 does not establish outcomes through December 31, and late injury reports may still revise ascertainment.

## Coverage and interpretation

Features with >10% missingness in eligible 2026 landmarks (all frozen comparison models, not only the primary):

{markdown(bad[['feature','missing_fraction','observed_pitchers']].head(20))}

Training-fitted median imputation, missing indicators and clipping are unchanged. Early-era structural absence, including arm angle, remains nullable in the source; no zero fill or test-derived imputation was introduced. Refer to `feature_coverage.csv` and `feature_distribution_shift.csv` for all features.

The endpoint is documented UCL injury under the existing refined adjudication rules; its event date is a defensible reported injury/IL proxy, not biological onset. Current-outing predictors are available after that outing. The same pitcher may appear in successive calendar partitions under the established longitudinal deployment design; whole-pitcher bootstrap intervals respect repeated observations, but this is not an unseen-pitcher test.

2026 was excluded from all current fits and decisions. Earlier stages had inspected descriptive 2026 event counts and source quality, so this is a held-out predictive evaluation, not a claim that every aspect of 2026 clinical ascertainment was previously unseen. Historical feature representations were carried forward; this revision selected among them using 2024–2025 rather than repeating a broad feature search.

Next: retain this freeze, mature the remaining 2026 outcome follow-up in a separately versioned evaluation, and test the same protocol prospectively. Do not choose a replacement model or threshold from these 2026 results. Frozen-score bootstrap intervals quantify cohort sampling variation, not full model-selection/refit uncertainty.

## Artifacts and validation

- Five figures in `figures/` (PNG and PDF): chronological discrimination, warning policy, follow-up censoring, training expansion and calibration.
- `tables/`: all candidate validation results, horizon-specific cohorts, model/ranking/calibration/warning metrics, event warning details, fixed-threshold operational burden, paired uncertainty, source coverage and QA.
- `../../../notebooks/ucl_revised_2026/`: three executed notebooks.
- `{q['checks']}` analysis checks passed; database validation passed. Unit-test results are recorded in `pytest.log`.
- Methods and exact rerun commands: `docs/ucl_revised_2026_methods.md`.
'''
    (OUT/'summary.md').write_text(text)


def run_reporting():
    if valid_checkpoint('reporting'):return
    run_qa();run_figures();run_summary()
    checkpoint('reporting',[OUT/'summary.md',OUT/'validation_summary.json',OUT/'tables/validation_checks.csv',
        OUT/'refresh_summary.json']+list((OUT/'figures').glob('*')))
