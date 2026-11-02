"""Endpoint-only adapter of frozen Kang pitcher-game cohorts.

Read saved adjudication, never import its chronological risk-set/model pipeline.
All appearance values and season roles come from the benchmark's frozen inputs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .kang_spec import ROOT, atomic_json, digest
from .kang_replication_data import config as benchmark_config, PREDICTORS

CONFIG = ROOT / 'config/kang_ucl_endpoint.yaml'
KEY = ['pitcher_id', 'season']
GRAIN = ['pitcher_id', 'game_id']


def settings(path=None):
    spec = yaml.safe_load(Path(path or CONFIG).read_text())
    base = benchmark_config(ROOT / spec['benchmark_config'])
    assert spec['endpoint_flag'] == 'endpoint_primary'
    assert spec['reference_date'] == 'prediction_reference_date'
    assert spec['cohort_policy'] == 'first_broad_shoulder_elbow_or_confirmed_ucl_episode'
    assert spec['control_policy'] == 'preserve_benchmark_controls_except_confirmed_ucl_promotions'
    assert (spec['study_start'], spec['study_end']) == (base['start_date'], base['end_date'])
    cfg = dict(base, output=spec['output'], cache=spec['cache'])
    assert cfg['output'] != base['output'] and cfg['cache'] != base['cache']
    assert Path(cfg['output']).parts[:2] == ('reports', 'kang_ucl_endpoint')
    assert Path(cfg['cache']).parts[:4] == ('data', 'injury', 'analysis', 'kang_ucl_endpoint')
    return spec, base, cfg


def benchmark_paths(base):
    """Hash the complete archived benchmark, including resampling checkpoints."""
    dirs = [base['cache'], base['output'], 'notebooks/kang_replication']
    paths = [p for d in dirs for p in (ROOT/d).rglob('*') if p.is_file() and p.name != '.DS_Store']
    files = ['config/kang_replication.yaml', 'docs/kang_replication_methods.md',
             'scripts/kang_replication.R', 'scripts/make_kang_replication_notebook.py',
             'src/injury_analysis/kang_replication.py', 'src/injury_analysis/kang_replication_data.py',
             'src/injury_analysis/kang_replication_validation.py', 'src/injury_analysis/kang_cohort.py',
             'src/injury_analysis/kang_spec.py', 'tests/test_kang_replication.py']
    return sorted(set(paths + [ROOT/f for f in files]))


def check_hashes(hashes, root=ROOT):
    changed = [f for f, value in hashes.items() if not (root/f).exists() or digest(root/f) != value]
    if changed:
        raise ValueError('Frozen source/artifact changed: ' + ', '.join(changed[:8]))


def freeze(spec, base, cfg):
    out = ROOT/cfg['output']; (out/'tables').mkdir(parents=True, exist_ok=True)
    (ROOT/cfg['cache']).mkdir(parents=True, exist_ok=True)
    path = out/'experiment_freeze.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['extension_spec'] != spec or saved['effective_model_config'] != cfg:
            raise ValueError('Changed frozen experiment specification; do not mix runs.')
        check_hashes(saved['benchmark_hashes']); check_hashes(saved['endpoint_source_hashes'])
        return saved
    ep_checkpoint = json.loads((ROOT/spec['adjudication_checkpoint']).read_text())
    check_hashes(ep_checkpoint['files'])
    sources = [spec['adjudicated_events'], spec['adjudication_checkpoint'], spec['endpoint_implementation'],
               'config/ucl_dynamic.yaml', 'src/injury_analysis/cohort.py', 'docs/ucl_dynamic_methods.md']
    db = (ROOT/base['database']).stat()
    value = dict(extension_spec=spec, effective_model_config=cfg,
        benchmark_hashes={str(p.relative_to(ROOT)):digest(p) for p in benchmark_paths(base)},
        endpoint_source_hashes={p:digest(ROOT/p) for p in sources},
        database_stat=dict(size=db.st_size,mtime_ns=db.st_mtime_ns),
        predictors=PREDICTORS, intentional_model_config_differences=['cache','output'],
        endpoint_changes=['documented UCL plus existing endpoint exclusions',
                          'confirmed forearm/UCL case promotions', 'existing earliest reference date'],
        split='Same R stratified random 67/33 game algorithm/seed; assignments can change with cohort/labels',
        no_chronological_split=True, no_additional_features=True,
        frozen_before_model_fitting=True)
    atomic_json(path, value)
    return value


def refine_cohorts(appearances, episodes, adjudicated, seasons, cfg):
    """Preserve broad first-event blockers; only accepted UCL episodes are cases.

    Non-UCL elbow case-seasons are excluded, not made healthy controls. Confirmed
    forearm episodes participate in the same first-event rule and can promote a
    former control. Later UCL episodes cannot replace an earlier broad target.
    """
    assert not appearances.duplicated(GRAIN).any()
    assert not adjudicated.injury_event_id.duplicated().any()
    e = episodes.copy().rename(columns={'player_id':'pitcher_id'})
    e['transaction_date'] = pd.to_datetime(e.transaction_date)
    e['season'] = e.transaction_date.dt.year
    fields = ['injury_event_id','player_id','endpoint_primary','prediction_reference_date',
              'evidence_category','endpoint_exclusions','initial_injury_region',
              'procedure_link_valid','linked_procedure_ids','ucl_confirmation_reported_date']
    u = adjudicated[fields].rename(columns={'player_id':'adjudicated_player_id',
                                          'initial_injury_region':'adjudicated_initial_region'})
    e = e.merge(u, on='injury_event_id', how='left', validate='one_to_one')
    matched = e.adjudicated_player_id.notna()
    assert e.loc[matched,'pitcher_id'].eq(e.loc[matched,'adjudicated_player_id']).all()
    e['ucl_accepted'] = e.endpoint_primary.eq(True)
    e['prediction_reference_date'] = pd.to_datetime(e.prediction_reference_date)
    bounds = appearances.groupby('season').game_date.agg(['min','max'])
    e = e.join(bounds,on='season')
    e['regular_placement'] = e.placement_transaction_id.notna() & e.transaction_date.between(e['min'],e['max'])
    e['broad_target'] = e.injury_region.isin(['Elbow / UCL','Shoulder'])
    first = e[e.regular_placement & (e.broad_target | e.ucl_accepted)].sort_values(
        ['pitcher_id','transaction_date','injury_event_id']).drop_duplicates(KEY).copy()
    first['case_season'] = first.ucl_accepted
    eligible = appearances.merge(seasons[KEY+['eligible','role']],on=KEY,validate='many_to_one')
    eligible = eligible[eligible.eligible].copy()
    merged = eligible.merge(first[KEY+['injury_event_id','case_season','prediction_reference_date',
                                      'transaction_date']],on=KEY,how='left',validate='many_to_one')
    merged['is_control'] = merged.injury_event_id.isna()
    merged['case_season'] = merged.case_season.eq(True)
    merged['lead_days'] = (merged.prediction_reference_date-merged.game_date).dt.days
    kept = merged[merged.is_control | (merged.case_season & merged.lead_days.gt(0))].copy()
    # Preserve the benchmark's supplemental prior-ELBOW-history field exactly.
    # It is not part of the 22-feature primary model or changed to UCL history.
    history = e[e.regular_placement & e.injury_region.eq('Elbow / UCL')].groupby('pitcher_id').season.min()
    kept['prior_region_il'] = kept.pitcher_id.map(history).lt(kept.season).astype(int)
    cohorts = {role:kept[kept.role.eq(role)].sort_values(['game_date','game_id','pitcher_id']).reset_index(drop=True)
               for role in ['starter','reliever']}
    return cohorts, e, first, merged


def labels(frame, window):
    return (frame.case_season & frame.lead_days.between(1,window)).astype(int)


def prepare(spec, base, cfg):
    out, cache, old = ROOT/cfg['output'], ROOT/cfg['cache'], ROOT/base['cache']
    a = pd.read_parquet(old/'appearances.parquet')
    e = pd.read_parquet(old/'replication_episodes.parquet')
    s = pd.read_csv(old/'season_eligibility.csv')
    u = pd.read_parquet(ROOT/spec['adjudicated_events'])
    cohorts, mapped, first, merged = refine_cohorts(a,e,u,s,cfg)
    benchmark = {r:pd.read_csv(old/('elbow_'+r)/'cohort.csv',parse_dates=['game_date'],low_memory=False) for r in cohorts}
    columns = ['game_id','pitcher_id','season','game_date','injury_event_id','lead_days','case_season','prior_region_il']+PREDICTORS
    records, support = [], []
    for outcome, frames in [('elbow',benchmark),('ucl',cohorts)]:
        for role, f in frames.items():
            case = f[f.case_season]
            records.append(dict(outcome=outcome,role=role,games=len(f),pitcher_seasons=len(f[KEY].drop_duplicates()),
                unique_pitchers=f.pitcher_id.nunique(),injury_events=f.injury_event_id.nunique(),
                injured_pitchers=case.pitcher_id.nunique(),control_games=int((~f.case_season).sum()),
                case_games=len(case)))
            for w in range(cfg['window_min'],cfg['window_max']+1,cfg['window_step']):
                y = labels(f,w)
                support.append(dict(outcome=outcome,role=role,window=w,games=len(f),positive_games=int(y.sum()),
                    negative_games=int((1-y).sum()),control_games=int((~f.case_season).sum()),
                    earlier_case_negative_games=int((f.case_season & y.eq(0)).sum()),
                    prevalence=float(y.mean()),represented_events=f.loc[y.eq(1),'injury_event_id'].nunique()))
            if outcome=='ucl':
                d=cache/('ucl_'+role);d.mkdir(exist_ok=True)
                f[columns].to_csv(d/'cohort.csv',index=False)
                # Preserve exact feature values and role assignment on every reused game.
                joined=f.merge(a[GRAIN+PREDICTORS],on=GRAIN,suffixes=('','_source'),validate='one_to_one')
                for col in PREDICTORS:
                    assert np.allclose(joined[col],joined[col+'_source'],equal_nan=True),col
    pd.DataFrame(records).to_csv(out/'tables/cohort_comparison.csv',index=False)
    pd.DataFrame(support).to_csv(out/'tables/window_support.csv',index=False)
    mapped.to_parquet(cache/'episode_mapping.parquet',index=False)
    first.to_parquet(cache/'first_target_events.parquet',index=False)
    # Explicit overlap and exclusions for every archived elbow case.
    overlaps=[]
    for role,f in benchmark.items():
        kept=set(cohorts[role].injury_event_id.dropna())
        b=f[f.case_season].drop_duplicates('injury_event_id')
        b=b[['injury_event_id','pitcher_id','season']].merge(mapped.drop(columns=['pitcher_id','season']),on='injury_event_id',validate='one_to_one')
        b['role']=role;b['retained_in_ucl_cohort']=b.injury_event_id.isin(kept)
        b['refinement_reason']=np.where(b.ucl_accepted,
            np.where(b.retained_in_ucl_cohort,'retained','accepted_but_not_first_or_no_pre_reference_games'),
            np.where(b.endpoint_exclusions.fillna('').ne(''),'existing_endpoint_QA_exclusion','no_confirmed_UCL_evidence'))
        b['reference_date_shift_days']=(b.transaction_date-b.prediction_reference_date).dt.days
        overlaps.append(b)
    overlap=pd.concat(overlaps,ignore_index=True)
    detail=['injury_event_id','pitcher_id','season','role','injury_region','initial_injury_region','injury_diagnosis',
            'evidence_category','endpoint_exclusions','ucl_accepted','retained_in_ucl_cohort','refinement_reason',
            'transaction_date','prediction_reference_date','reference_date_shift_days']
    overlap[detail].to_csv(out/'tables/endpoint_overlap_events.csv',index=False)
    overlap.groupby(['role','refinement_reason','evidence_category'],dropna=False).size().rename('events').reset_index().to_csv(out/'tables/endpoint_overlap_summary.csv',index=False)
    # All established UCL events in study years, including those not model eligible.
    accepted=u[u.endpoint_primary & u.prediction_reference_date.between(cfg['start_date'],cfg['end_date'])].copy()
    accepted['season']=accepted.prediction_reference_date.dt.year
    known=set(mapped.injury_event_id); first_ids=set(first.injury_event_id)
    broad_ids=set(overlap.injury_event_id);model_ids=set(pd.concat(cohorts.values()).injury_event_id.dropna())
    season_index=s.set_index(KEY);ep_index=mapped.set_index('injury_event_id')
    reasons=[]
    for r in accepted.itertuples():
        key=(r.player_id,r.season)
        if r.injury_event_id in model_ids:
            reason='retained_benchmark_elbow' if r.injury_event_id in broad_ids else 'added_confirmed_UCL_not_benchmark_elbow'
        elif r.injury_event_id not in known:reason='not_in_frozen_episode_source'
        elif key not in season_index.index or not season_index.loc[key,'eligible']:reason='benchmark_pitcher_season_ineligible'
        elif not ep_index.loc[r.injury_event_id,'regular_placement']:reason='outside_regular_season_placement_or_no_placement'
        elif r.injury_event_id not in first_ids:reason='earlier_broad_shoulder_elbow_or_UCL_event'
        else:reason='no_eligible_pre_reference_games'
        reasons.append(reason)
    accepted['benchmark_mapping_status']=reasons
    accepted[['injury_event_id','player_id','season','initial_injury_region','injury_region','evidence_category',
              'prediction_reference_date','transaction_date','ucl_confirmation_reported_date','benchmark_mapping_status']].to_csv(out/'tables/all_ucl_event_mapping.csv',index=False)
    accepted.groupby(['benchmark_mapping_status','initial_injury_region']).size().rename('events').reset_index().to_csv(out/'tables/ucl_source_coverage.csv',index=False)
    # Controls and moved dates are part of endpoint refinement, not a new risk set.
    changes=[]
    for role,f in cohorts.items():
        b=benchmark[role];old_control=set(map(tuple,b.loc[~b.case_season,GRAIN].to_numpy()))
        new_control=set(map(tuple,f.loc[~f.case_season,GRAIN].to_numpy()))
        assert new_control<=old_control
        cases=f[f.case_season]; retained=cases[cases.injury_event_id.isin(b.injury_event_id)]
        common=retained.merge(b[GRAIN+['lead_days']],on=GRAIN,suffixes=('','_benchmark'),validate='one_to_one')
        changes.append(dict(role=role,benchmark_control_games=len(old_control),ucl_control_games=len(new_control),
            newly_added_control_games=len(new_control-old_control),former_control_games_in_promoted_UCL_seasons=len(old_control-new_control),
            promoted_UCL_events=cases.loc[~cases.injury_event_id.isin(b.injury_event_id),'injury_event_id'].nunique(),
            retained_case_rows_with_changed_lead_days=int(common.lead_days.ne(common.lead_days_benchmark).sum())))
    pd.DataFrame(changes).to_csv(out/'tables/cohort_membership_changes.csv',index=False)
    (out/'tables/feature_crosswalk.csv').write_bytes((ROOT/base['output']/'tables/feature_crosswalk.csv').read_bytes())
    return cohorts, benchmark
