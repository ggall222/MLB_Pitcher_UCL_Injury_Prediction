"""Executed-data reports. Unknown dates, failed ranges and unresolved identities stay visible."""
from datetime import datetime, timezone
import json
import re
import numpy as np
import pandas as pd
import yaml
from .config import ROOT
from .build_labels import LABEL_COLUMNS
from .ingest_injuries import normalized_name
from .evaluation import temporal_partition


def save(frame,cfg,name):
    frame.to_csv(cfg['report_path']/name,index=False)
    return frame


def missing_reason(c):
    if 'arm_angle' in c:return 'Tracking field introduced later; history also requires prior measured outings.'
    if 'ivb' in c:return 'Supplemental MLB telemetry field and validated source alignment required.'
    if 'spin_axis' in c or 'spin_' in c or c.startswith('spin'):return 'Historical tracking coverage varies; pitch-type-specific metrics require that pitch.'
    if re.match(r'^(ff|si|fc|sl|st|cu|kc|ch|fs|sv)_',c) and not c.endswith(('usage_pct','pitch_count')):
        return 'Pitch type may be unused; derived history additionally needs measured prior outings.'
    if 'baseline' in c or c.endswith('_z'):return 'At least five valid prior measurements; z-score also needs nonzero variance.'
    if '_prev_' in c or '_slope_' in c:return 'Requires prior appearances and sufficient nonmissing measurements.'
    if c in ('velo_drop_game','first_n_fb_velo','last_n_fb_velo'):return 'Requires at least 40 measured fastballs in disjoint 20-pitch segments.'
    if c.endswith('_sd'):return 'Sample SD requires at least two measured pitches.'
    if 'previous_season' in c:return 'No observed prior MLB season or unavailable official prior-year totals.'
    if 'extension' in c or 'release_' in c or 'movement' in c or 'plate_' in c or 'location' in c:
        return 'Historical measurement coverage and tracking definitions vary.'
    if c=='rest_days':return 'First observed appearance has no previous observed rest interval.'
    return 'Inspect source or feature definition; no automatic zero fill.'


def missingness(db,cfg,features,model,starter):
    by=[];overall=[]
    raw=set(db.columns('pitches'));engineered=set(db.columns('mechanical_features')+db.columns('workload_features')+db.columns('pitcher_baselines'))
    groups=list(model.groupby('season'))
    for c in features:
        first=None;first_starter=None
        for season,g in groups:
            st=g[g.valid_starting_appearance.fillna(False)]
            available=float(g[c].notna().mean());sav=float(st[c].notna().mean()) if len(st) else np.nan
            if available>=.5 and first is None:first=int(season)
            if sav>=.5 and first_starter is None:first_starter=int(season)
            by.append(dict(feature=c,season=int(season),rows=len(g),missing_percentage=100*(1-available),
                           starter_rows=len(st),starter_missing_percentage=100*(1-sav)))
        overall.append(dict(feature=c,overall_missing_percentage=100*model[c].isna().mean(),
            starter_missing_percentage=100*starter[c].isna().mean() if len(starter) else np.nan,
            first_season_meaningful_availability=first,first_season_meaningful_starter_availability=first_starter,meaningful_threshold='at least 50% of appearances measured',
            feature_kind='engineered_history' if c in engineered else ('raw' if c in raw else 'appearance_aggregate_or_context'),
            structurally_expected=any(term in missing_reason(c).lower() for term in ('requires','introduced','unused','prior','coverage','variance')),
            missingness_explanation=missing_reason(c)))
    overall=save(pd.DataFrame(overall),cfg,'feature_missingness.csv')
    arm=overall[overall.feature.eq('arm_angle_avg')].first_season_meaningful_starter_availability
    modern=int(arm.iloc[0]) if len(arm) and pd.notna(arm.iloc[0]) else None
    eras=[dict(era='full_statcast_era',start_season=2015,end_season=int(model.season.max()),note='Broad features; retain tracking-definition sensitivity flags.'),
          dict(era='enhanced_tracking_era',start_season=2020,end_season=int(model.season.max()),note='Hawk-Eye era; 2026 location-definition change still needs sensitivity analysis.'),
          dict(era='modern_feature_era',start_season=modern,end_season=int(model.season.max()),note='First >=50% arm-angle starter coverage; check every subsequent season and use optional modern features.')]
    save(pd.DataFrame(eras),cfg,'modeling_era_definitions.csv')
    save(pd.DataFrame(by),cfg,'feature_availability_by_season.csv')
    rawfields=['arm_angle','spin_axis','release_spin_rate','pfx_x','pfx_z','release_extension','release_pos_x','release_pos_z','plate_x','plate_z','release_speed']
    expressions=','.join(f'100.0*count({c})/count(*) AS {c}_available_pct' for c in rawfields)
    save(db.query(f'SELECT year(game_date) AS season,count(*) AS physical_pitches,{expressions} FROM pitches WHERE is_actual_pitch GROUP BY 1 ORDER BY 1'),cfg,'raw_feature_availability_by_season.csv')
    history=[]
    majors=['avg_fb_velo','fb_velo_sd','release_x_avg','release_z_avg','release_x_sd','release_z_sd','extension_avg',
            'arm_angle_avg','spin_avg','horizontal_movement_avg','vertical_movement_avg','fastball_usage_pct','zone_pct','location_variability']
    eligible=starter[starter.training_eligible & starter.ucl_injury_next_30d.notna()]
    for metric in majors:
        out={'metric':metric,'model_rows':len(eligible)}
        for n in (3,5,10):
            col=f'{metric}_prev_{n}';out[f'rows_sufficient_previous_{n}']=int(eligible[col].notna().sum()) if col in eligible else None
        col=f'{metric}_baseline_mean';out['rows_sufficient_baseline']=int(eligible[col].notna().sum()) if col in eligible else None
        history.append(out)
    history.append(dict(metric='official_workload_appearance_history',model_rows=len(eligible),
        **{f'rows_sufficient_previous_{n}':int(eligible.workload_history_appearances.ge(n).sum()) for n in (3,5,10)}))
    for n in (7,14,28):
        history.append(dict(metric=f'workload_{n}d',model_rows=len(eligible),rows_complete_calendar_history=int(eligible[f'workload_{n}d_complete'].sum())))
    save(pd.DataFrame(history),cfg,'rolling_history_audit.csv')
    return overall


def event_audit(db,cfg,events,model,starter):
    allapps={pid:g for pid,g in model.groupby('pitcher_id')};starts={pid:g for pid,g in starter.groupby('pitcher_id')}
    rows=[]
    for event in events.to_dict('records'):
        pid=event['pitcher_id'];ref=event['label_reference_date'];g=allapps.get(pid,model.iloc[:0]);st=starts.get(pid,starter.iloc[:0])
        previous=g[(g.game_date<ref)&~g.timing_uncertain.fillna(True)] if pd.notna(ref) else g.iloc[:0]
        recent=previous[previous.game_date>=ref-pd.Timedelta(days=30)] if pd.notna(ref) else previous
        positives=st[st.training_eligible & st.ucl_injury_next_30d.fillna(False)&st.next_injury_event_id.eq(event['injury_event_id'])]
        event.update(has_statcast_history=len(g)>0,has_mlb_start_history=bool(len(g[g.starter_status.eq('starter')])),
            surgery_during_study=bool(pd.notna(event['surgery_date_lower']) and event['surgery_date_lower']<=pd.Timestamp(cfg['end_date']) and event['surgery_date_upper']>=pd.Timestamp(cfg['start_date'])),
            last_observed_pre_reference_appearance=previous.game_date.max(),
            number_of_appearances_previous_30d=len(recent),number_of_positive_30d_rows=len(positives),
            number_of_starts_previous_30d=int(recent.starter_status.eq('starter').sum()),
            warning_opportunities_7_to_30d=int((ref-positives.game_date).dt.days.between(7,30).sum()) if pd.notna(ref) else 0,
            qualifying_starter_event=bool(event['confirmed_ucl_surgery'] and pd.notna(ref) and pd.Timestamp(cfg['start_date'])<=ref<=pd.Timestamp(cfg['observation_end']) and len(st[st.game_date<ref])>0),
            positive_date_order_valid=bool((positives.game_date<ref).all()) if pd.notna(ref) else len(positives)==0,
            label_confidence='public_IL_proxy_requires_event_review' if pd.notna(ref) else 'low_surgery_only_unknown_onset')
        rows.append(event)
    audit=save(pd.DataFrame(rows),cfg,'injury_event_audit.csv')
    counts=[]
    for label in LABEL_COLUMNS:
        eligible=starter[starter.training_eligible & starter[label].notna()];positive=eligible[eligible[label].fillna(False)]
        counts.append(dict(label=label,eligible_rows=len(eligible),positive_rows=len(positive),negative_rows=len(eligible)-len(positive),
            positive_percentage=100*len(positive)/len(eligible) if len(eligible) else np.nan,
            unique_injury_events=positive.next_injury_event_id.nunique(),unique_injured_pitchers=positive.pitcher_id.nunique(),
            unknown_or_ineligible_rows=len(starter)-len(eligible)))
    labels=save(pd.DataFrame(counts),cfg,'injury_label_audit.csv')
    injuries=db.query('SELECT * FROM injuries');observed=set(model.pitcher_id);sp=set(model.loc[model.starter_status.eq('starter'),'pitcher_id'])
    metrics={
      'total_ucl_procedure_records':len(injuries),'matched_mlbam_records':int(injuries.pitcher_id.notna().sum()),
      'records_with_statcast_history':int(injuries.pitcher_id.isin(observed).sum()),
      'records_with_surgery_in_study_period':int((injuries.surgery_date_lower.le(pd.Timestamp(cfg['end_date']))&injuries.surgery_date_upper.ge(pd.Timestamp(cfg['start_date']))).sum()),
      'records_involving_pitchers_with_mlb_starts':int(injuries.pitcher_id.isin(sp).sum()),
      'qualifying_unique_starter_events':int(audit.qualifying_starter_event.sum()),
      'unique_injured_starters':audit.loc[audit.qualifying_starter_event,'pitcher_id'].nunique(),
      'repeat_procedures':int(injuries.repeat_procedure.fillna(False).sum()),
      'internal_brace_procedures':int(injuries.procedure_category.eq('ucl_internal_brace_repair').sum()),
      'tommy_john_reconstruction_procedures':int(injuries.procedure_category.eq('ucl_reconstruction').sum()),
      'uncertain_surgery_dates':int(injuries.surgery_date_precision.ne('day').sum()),
      'unconfirmed_or_out_of_scope_procedures':int((~injuries.confirmed_ucl_surgery.fillna(False)).sum()),
      'surgery_only_events_unknown_onset':int(audit.label_reference_date.isna().sum())}
    save(pd.DataFrame([dict(metric=k,count=v) for k,v in metrics.items()]),cfg,'injury_population_audit.csv')
    return audit,labels,metrics


def unmatched_audit(db,cfg):
    current=db.query('SELECT i.*,p.mlb_debut,p.raw_json AS person_json FROM injuries i LEFT JOIN pitchers p USING(pitcher_id)')
    original=pd.read_parquet(cfg['report_path']/'poc_2024/unmatched_original_55.parquet')
    candidates=json.loads((cfg['report_path']/'injury_identity_candidates.json').read_text())
    bykey={(normalized_name(c['player_name']),c['surgery_date_raw'],c['source_tab']):c for c in candidates}
    originalkeys={(normalized_name(r.player_name),r.surgery_date_raw,r.source_tab) for r in original.itertuples()}
    rows=[]
    for r in current.itertuples():
        key=(normalized_name(r.player_name),r.surgery_date_raw,r.source_tab)
        if pd.notna(r.pitcher_id) and key not in originalkeys:continue
        c=bykey.get(key,{})
        outside=pd.notna(r.surgery_date_upper) and (r.surgery_date_upper<pd.Timestamp(cfg['start_date']) or r.surgery_date_lower>pd.Timestamp(cfg['end_date']))
        nonpitcher=pd.notna(r.position) and r.position not in ('P','TWP')
        cand=c.get('candidates',[]);has_candidate=bool(cand)
        if pd.notna(r.pitcher_id):reason='reconciled_identifier';status='resolved'
        elif nonpitcher:reason='procedure_not_in_scope';status='unresolved'
        elif outside:reason='outside_statcast_period';status='unresolved'
        elif c.get('register_matches',0)>1:reason='name_matching_issue';status='unresolved'
        elif has_candidate:reason='incorrect_or_missing_player_id';status='unresolved'
        elif str(r.level).lower() in ('coll','hs'):reason='missing_mlbam_id';status='unresolved'
        elif str(r.level).lower() in ('rk','a','a+','a-','aa','aaa'):reason='missing_mlbam_id';status='unresolved'
        else:reason='missing_mlbam_id';status='unresolved'
        evidence='; '.join(f"MLBAM {p['mlb'].get('id')}, {p['mlb'].get('fullName')}, DOB {p['mlb'].get('birthDate')}, debut {p['mlb'].get('mlbDebutDate')}" for p in cand)
        rows.append(dict(injury_id=r.injury_id,player_name=r.player_name,surgery_date_raw=r.surgery_date_raw,
            source_tab=r.source_tab,original_unmatched_55=key in originalkeys,pitcher_id=r.pitcher_id,status=status,
            reason_category=reason,identity_method=r.identity_method,outside_statcast_period=outside,
            nonpitcher_procedure=nonpitcher,source_level=r.level,fangraphs_id=r.fangraphs_id,
            candidate_count=len(cand),candidate_evidence=evidence,
            review_note='Source level is not proof of no lifetime MLB history; unresolved candidates were not forced into a join.' if status=='unresolved' else 'Identity reconciled; cohort scope is assessed separately.',source_url=r.source_url))
    seen={(normalized_name(r['player_name']),r['surgery_date_raw'],r['source_tab']) for r in rows}
    for r in original.itertuples():
        key=(normalized_name(r.player_name),r.surgery_date_raw,r.source_tab)
        if key not in seen:
            rows.append(dict(injury_id=r.injury_id,player_name=r.player_name,surgery_date_raw=r.surgery_date_raw,
                source_tab=r.source_tab,original_unmatched_55=True,pitcher_id=None,status='unresolved',reason_category='other',
                review_note='Original record absent or date/name revised in current source; original raw row is preserved in poc_2024/unmatched_original_55.parquet.',source_url=r.source_url))
    audit=save(pd.DataFrame(rows),cfg,'unmatched_injury_audit.csv')
    save(audit.groupby(['status','reason_category']).size().reset_index(name='records'),cfg,'unmatched_injury_reason_counts.csv')
    return audit


def validate(db,cfg,features,model):
    checks=[]
    # Materialize once: several independent assertions inspect the same expensive alignment.
    from .build_appearances import alignment_view
    db.con.execute('SET preserve_insertion_order=false')
    db.con.execute('CREATE OR REPLACE TEMP TABLE _alignment_audit AS SELECT *,CAST(NULL AS INTEGER) AS season FROM aligned_pitch_tracking LIMIT 0')
    for year in sorted(model.season.dropna().unique()):
        view=alignment_view(db,int(year))
        db.con.execute('INSERT INTO _alignment_audit SELECT *,'+str(int(year))+' AS season FROM '+view)
    aligned=db.query('SELECT season,count(*) AS matched_physical_records,count(*) FILTER(WHERE alignment_valid) AS valid_alignment_records,count(*) FILTER(WHERE NOT alignment_valid) AS invalid_alignment_records,count(*) FILTER(WHERE NOT counts_match) AS physical_count_mismatch_records,count(*) FILTER(WHERE ivb_inches IS NOT NULL) AS usable_ivb_records,avg(velocity_difference) AS mean_source_velocity_difference FROM _alignment_audit GROUP BY season ORDER BY season')
    native=db.query('SELECT g.season,count(*) AS native_records,count(ivb_inches) AS native_measured_ivb_records FROM pitch_tracking t JOIN games g USING(game_id) GROUP BY g.season')
    save(aligned.merge(native,on='season',how='outer'),cfg,'pitch_tracking_alignment_by_season.csv')
    for sql in (ROOT/'sql/validation.sql').read_text().split(';'):
        if sql.strip():checks.extend(db.query(sql.replace('aligned_pitch_tracking','_alignment_audit')).to_dict('records'))
    def check(name,n,severity='error'):checks.append(dict(check_name=name,severity=severity,violations=int(n)))
    forbidden=['until_next','return_date','surgery_date','diagnosis','next_injury','next_known','label_','ucl_injury_next','risk_set','last_observed']
    check('predictor_temporal_allowlist',sum(any(t in c for t in forbidden) for c in features))
    check('nonregular_starter_cohort',int((model.valid_starting_appearance & model.game_type.ne('R')).sum()))
    for h in (14,30,60,90):
        p=model[model[f'ucl_injury_next_{h}d'].fillna(False)]
        check(f'positive_{h}d_strictly_pre_reference',(~((p.next_known_ucl_onset>p.game_date)&(p.next_known_ucl_onset<=p.game_date+pd.Timedelta(days=h)))).sum())
    check('injury_event_foreign_key',db.scalar('SELECT count(*) FROM injury_labels l LEFT JOIN injury_events e ON l.next_injury_event_id=e.injury_event_id WHERE l.next_injury_event_id IS NOT NULL AND e.injury_event_id IS NULL'))
    # Independent brute-force sample across every season, with last-five values and dates.
    sample=model.groupby('season',group_keys=False).apply(lambda x:x.sample(min(100,len(x)),random_state=17),include_groups=False)
    groups={pid:g.sort_values(['game_date','scheduled_start','game_id']) for pid,g in model.groupby('pitcher_id')}
    mismatches=0
    for r in sample.itertuples():
        g=groups[r.pitcher_id];previous=g[(g.game_date<r.game_date)&~g.timing_uncertain.fillna(True)].tail(5)
        vals=previous.avg_fb_velo.where(~previous.timing_uncertain.fillna(True))
        expected=vals.mean() if len(previous)==5 and vals.notna().sum()>=2 else np.nan
        mismatches+=not np.isclose(expected,r.avg_fb_velo_prev_5,equal_nan=True,atol=1e-7)
    check('independent_prior_5_recalculation_stratified_sample',mismatches)
    checks=[dict(r,violations=int(r['violations'])) for r in checks]
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),passed=not any(r['violations'] for r in checks if r['severity']=='error'),
                checks=checks,feature_count=len(features),independent_recalculation_rows=len(sample))
    (cfg['report_path']/'validation_report.json').write_text(json.dumps(report,indent=2))
    save(pd.DataFrame(checks),cfg,'leakage_and_integrity_audit.csv')
    (cfg['report_path']/'validation_report.md').write_text('# Historical database validation\n\n'+pd.DataFrame(checks).to_markdown(index=False)+'\n')
    db.con.execute('DROP TABLE _alignment_audit')
    return report


def generate(db,cfg,features,model,events):
    starter=model[model.valid_starting_appearance.fillna(False)]
    counts=db.query('SELECT year(game_date) AS season,count(*) AS pitches,count(*) FILTER(WHERE is_actual_pitch) AS physical_pitches,min(game_date) AS earliest_game_date,max(game_date) AS latest_game_date,count(DISTINCT game_id) AS pitch_games FROM pitches GROUP BY 1 ORDER BY 1')
    pop=model.groupby('season').agg(appearances=('game_id','size'),pitchers=('pitcher_id','nunique')).reset_index()
    for role in ('starter','relief','ambiguous'):
        pop=pop.merge(model[model.starter_status.eq(role)].groupby('season').size().rename(role+'_appearances').reset_index(),on='season',how='left')
    pop=pop.merge(starter.groupby('season').agg(valid_starting_appearances=('game_id','size'),unique_starting_pitchers=('pitcher_id','nunique')).reset_index(),on='season',how='left')
    for col in ['starter_appearances','relief_appearances','ambiguous_appearances','valid_starting_appearances','unique_starting_pitchers']:
        pop[col]=pop[col].fillna(0).astype(int)
    save(db.query('SELECT b.game_id,b.pitcher_id,g.game_date,b.is_starter,b.pitch_count,b.outs_recorded FROM boxscore_appearances b JOIN games g USING(game_id) LEFT JOIN appearances a USING(game_id,pitcher_id) WHERE a.game_id IS NULL ORDER BY g.game_date'),cfg,'official_appearances_without_statcast.csv')
    schedule=db.query("SELECT season,count(*) FILTER(WHERE status='Final') AS official_final_games FROM games WHERE game_type='R' AND game_date BETWEEN ? AND ? GROUP BY season",[cfg['start_date'],cfg['end_date']])
    coverage=save(counts.merge(pop,on='season').merge(schedule,on='season'),cfg,'season_coverage.csv')
    failed=save(db.query("""SELECT s.*,
        coalesce(g.game_date,try_cast(split_part(s.item_key,':',1) AS DATE)) AS range_start,
        coalesce(g.game_date,try_cast(split_part(s.item_key,':',2) AS DATE),try_cast(s.item_key AS DATE)) AS range_end
        FROM ingestion_state s LEFT JOIN games g ON s.source IN ('boxscore','pitch_tracking') AND try_cast(s.item_key AS BIGINT)=g.game_id
        WHERE s.status='failed' ORDER BY s.source,s.item_key"""),cfg,'failed_source_ranges.csv')
    save(db.query("SELECT game_id,game_date,season,status,json_extract_string(raw_json,'$.status.reason') AS reason FROM games WHERE status='Cancelled' ORDER BY game_date"),cfg,'nonplayed_games.csv')
    save(db.query('SELECT DISTINCT p.game_id,g.game_date FROM pitches p JOIN games g USING(game_id) LEFT JOIN (SELECT DISTINCT game_id FROM pitch_tracking) t USING(game_id) WHERE t.game_id IS NULL ORDER BY g.game_date'),cfg,'missing_supplemental_tracking_games.csv')
    gaps=save(db.query("SELECT g.game_id,g.game_date,g.season,s.status AS day_status,s.detail FROM games g LEFT JOIN (SELECT DISTINCT game_id FROM pitches) p USING(game_id) LEFT JOIN ingestion_state s ON s.source='statcast' AND s.item_key=CAST(g.game_date AS VARCHAR) WHERE g.status='Final' AND g.game_type='R' AND g.game_date BETWEEN ? AND ? AND p.game_id IS NULL ORDER BY game_date",[cfg['start_date'],cfg['end_date']]),cfg,'missing_statcast_games.csv')
    starter_audit=[]
    for role,g in model.groupby('starter_status'):
        starter_audit.append(dict(population=role,metric='appearance_count',count=len(g)))
        if role=='starter':
            for col in ['pitch_count','innings_pitched','batters_faced']:
                d=g[col].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_dict()
                starter_audit.append(dict(population=role,metric=col,**d))
    if not model.starter_status.eq('ambiguous').any():
        starter_audit.append(dict(population='ambiguous',metric='appearance_count',count=0))
    save(pd.DataFrame(starter_audit),cfg,'starter_population_audit.csv')
    if 'starter_crosscheck_disagrees' in model:
        save(model[model.starter_crosscheck_disagrees][['game_id','pitcher_id','game_date','starter_source','first_observed_pitcher_id','starter_status']],cfg,'starter_crosscheck.csv')
    save(db.query('SELECT game_id,team_id,count(*) FILTER(WHERE is_starter) AS starters FROM boxscore_appearances GROUP BY game_id,team_id HAVING count(*) FILTER(WHERE is_starter)<>1'),cfg,'starter_identification_exceptions.csv')
    audit,labels,injury_counts=event_audit(db,cfg,events,model,starter)
    unmatched=unmatched_audit(db,cfg)
    missing=missingness(db,cfg,features,model,starter)
    curated=yaml.safe_load((ROOT/'config/model_features_v1.yaml').read_text())['features']
    assert all(f['name'] in features for f in curated),'Curated feature missing from master allowlist'
    cf=pd.DataFrame(curated).merge(missing,left_on='name',right_on='feature')
    save(cf,cfg,'model_v1_feature_audit.csv')
    save(db.query('SELECT * FROM ingestion_issues'),cfg,'ingestion_issues.csv')
    save(starter[starter.training_eligible & starter.ucl_injury_next_30d.fillna(False)][['pitcher_id','pitcher_name','game_id','game_date','next_injury_event_id','next_known_ucl_onset','label_onset_basis']],cfg,'positive_cases_review.csv')
    # Evaluate candidate cut points against observed events, rather than choosing example dates blindly.
    eligible=starter[starter.training_eligible & starter.ucl_injury_next_30d.notna()].copy()
    splitrows=[]
    years=sorted(coverage.season.astype(int))
    for train_end,valid_end in [(2019,2021),(2020,2022),(2021,2023),(2022,2024)]:
        if not years or valid_end>=max(years):continue
        parts=temporal_partition(eligible,f'{train_end}-12-31',f'{valid_end}-12-31',90)
        for name in ('train','validation','test','purged'):
            g=eligible[parts.eq(name)];p=g[g.ucl_injury_next_30d.fillna(False)]
            splitrows.append(dict(option=f'{train_end}/{valid_end}',partition=name,rows=len(g),positive_rows=len(p),unique_events=p.next_injury_event_id.nunique(),unique_injured_pitchers=p.pitcher_id.nunique(),earliest=g.game_date.min(),latest=g.game_date.max()))
    splits=save(pd.DataFrame(splitrows),cfg,'temporal_split_audit.csv')
    recommendation='No split has sufficient independent event support; review event timing and coverage first.'
    if len(splits):
        viable=splits[splits.partition.ne('purged')].groupby('option').agg(min_events=('unique_events','min'),rows=('rows','sum'))
        choice=viable.sort_values(['min_events','rows'],ascending=False).index[0]
        train_year,valid_year=map(int,choice.split('/'))
        recommendation=f'Train {min(years)}–{train_year}; validation {train_year+1}–{valid_year}; test {valid_year+1}–{max(years)} (latest year partial). Chosen to maximize the smallest partition event count; inspect temporal_split_audit.csv. This compares event support without fitting a model.'
    report=validate(db,cfg,features,model)
    count30=labels.set_index('label').loc['ucl_injury_next_30d']
    major=missing[missing.starter_missing_percentage.ge(20)].sort_values('starter_missing_percentage',ascending=False)
    unresolved=unmatched[unmatched.status.eq('unresolved')]
    selected=[f['name'] for f in curated if f['era']=='long_history']
    summary=dict(successful_seasons=years,earliest_game_date=str(counts.earliest_game_date.min().date()),latest_game_date=str(counts.latest_game_date.max().date()),
        total_raw_pitches=int(counts.pitches.sum()),total_physical_pitches=int(counts.physical_pitches.sum()),
        total_appearances=len(model),unique_pitchers=model.pitcher_id.nunique(),starting_appearances=int(model.starter_status.eq('starter').sum()),
        relief_appearances=int(model.starter_status.eq('relief').sum()),ambiguous_appearances=int(model.starter_status.eq('ambiguous').sum()),
        valid_starting_appearances=len(starter),unique_starting_pitchers=starter.pitcher_id.nunique(),
        database_bytes=cfg['database_path'].stat().st_size,unmatched_injury_records=int(db.scalar('SELECT count(*) FROM injuries WHERE pitcher_id IS NULL')),
        positive_30d_rows=int(count30.positive_rows),positive_30d_percentage=float(count30.positive_percentage),
        positive_30d_unique_events=int(count30.unique_injury_events),failed_source_items=len(failed),missing_statcast_games=len(gaps),
        curated_feature_count=len(curated),recommended_long_history_feature_count=len(selected),**injury_counts)
    (cfg['report_path']/'historical_build_metrics.json').write_text(json.dumps(summary,indent=2,default=int))
    lines=['# Full historical build summary','',f"Executed coverage: {summary['earliest_game_date']} through {summary['latest_game_date']}; {len(years)} represented seasons.",'',
        f"{summary['total_raw_pitches']:,} raw pitch/event records; {len(model):,} appearances; {summary['unique_pitchers']:,} observed pitchers.",
        f"{summary['starting_appearances']:,} official starting appearances, {summary['relief_appearances']:,} relief, {summary['ambiguous_appearances']:,} ambiguous; {len(starter):,} quality-eligible starts from {starter.pitcher_id.nunique():,} pitchers.",
        f"Database: {cfg['database_path']} ({summary['database_bytes']/1024**3:.2f} GiB).",'',coverage.to_markdown(index=False),'',
        '## Injury population','',pd.DataFrame([dict(metric=k,count=v) for k,v in injury_counts.items()]).to_markdown(index=False),'',
        '## Starting appearance labels','',labels.to_markdown(index=False),'',
        f"Unmatched identities: {summary['unmatched_injury_records']}; original-55 reconciliation outcomes are individually recorded in unmatched_injury_audit.csv.",
        unmatched.groupby(['status','reason_category']).size().reset_index(name='records').to_markdown(index=False),'',
        f"Failed source items: {len(failed)}; finalized games without Statcast records: {len(gaps)}. Inspect failed_source_ranges.csv and missing_statcast_games.csv; represented seasons do not imply complete coverage.",
        f"Database integrity and temporal checks: {'PASS' if report['passed'] else 'FAIL'}.", '',
        'Primary population is regular-season MLB official starts, including openers. Official boxscore gamesStarted is preferred; first pitcher in official pitching order is fallback. Teams must have exactly one identified starter. Quality eligibility requires an official boxscore, reliable game timing, >=80% measured velocity, and physical-vs-official pitch-count difference <=2. No pitch-count threshold defines a starter.',
        'Prediction time is after appearance completion. Same-day and future observations are excluded from historical features; current-game metrics are available at this prediction time. Raw relief appearances, rejected quality rows and the full feature master are retained.',
        'No final model was fitted and no missing measurements were imputed.']
    (cfg['report_path']/'full_build_summary.md').write_text('\n\n'.join(lines)+'\n')
    readiness=['# Modeling readiness','',
        f"1. Seasons represented: {len(years)} ({', '.join(map(str,years))}); check source gaps before interpreting this as complete.",
        f"2. Raw pitch/event records: {summary['total_raw_pitches']:,}.",f"3. Total pitcher appearances: {len(model):,}.",
        f"4. Official starting appearances: {summary['starting_appearances']:,}; quality-eligible model starts: {len(starter):,}.",
        f"5. Unique starting pitchers in the model population: {starter.pitcher_id.nunique():,}.",
        f"6. Qualifying UCL episodes: {injury_counts['qualifying_unique_starter_events']}; represented by a positive 30-day model row: {int(count30.unique_injury_events)}.",
        f"7. Unique injured starters: {injury_counts['unique_injured_starters']}.",
        f"8. 30-day positive prevalence: {count30.positive_percentage:.4f}% ({int(count30.positive_rows)} / {int(count30.eligible_rows)} eligible rows).",
        f"9. {len(major)} master features exceed 20% missing among valid starts. Arm angle, pitch-specific absence, insufficient history and native IVB need separate consideration. See feature_missingness.csv and raw_feature_availability_by_season.csv.",
        f"10. {len(unresolved)} unresolved identities: {', '.join(unresolved.player_name)}. Candidate evidence and scope categories are in unmatched_injury_audit.csv.",
        '11. '+('The observed event count supports beginning an exploratory baseline after case-level label review and the checks below; it does not establish generalization.' if count30.unique_injury_events>=50 and report['passed'] and not len(failed) else 'Do not claim modeling readiness yet: inspect independent event support, validation failures and source gaps before fitting.'),
        '12. '+recommendation+' Use 90-day embargoes for cross-horizon comparisons. Purge events spanning retained partitions. Repeated pitchers may legitimately recur in future years; also report a separate unseen-pitcher sensitivity analysis.',
        f"13. model_features_v1.yaml selects {len(curated)} features with rationales: {len(selected)} long-history candidates plus {len(curated)-len(selected)} optional modern arm-angle features. Per-era missingness determines the final training-fold subset; no imputation performed.",
        '14. Remaining risks: public reporting can lag symptoms; generic forearm/elbow IL-to-surgery linkage is retrospective and requires review; surgery-only events have unknown onset and create censoring intervals; ascertainment is incomplete for minor leagues; missing follow-up is not a negative. Openers and role changes remain in the official-start population. Earlier data and 2026 location definitions differ. Cohort quality filters and incomplete telemetry can select healthier/measurable appearances.',
        '## Label reference and event review','',
        'A qualifying starter event has a confirmed procedure, a public reference inside the study/follow-up period, and at least one quality-eligible MLB start before that reference. This broad count can include former starters without a recent prediction opportunity; positive-window and observable-event counts are reported separately.',
        'The reference is the first public arm-IL announcement in the linked active episode. Effective/retroactive IL date is preserved separately. Clinical diagnosis, first symptom and last healthy appearance remain NULL unless explicitly sourced; last observed pre-reference appearance is not certified healthy. An official activation followed by an observed MLB appearance permits risk-set reentry. Source return dates never enter predictors. Multiple procedures within one active episode count as one event. Surgery-only records are lower confidence and do not generate fabricated precise targets.',
        'The next-3/5 appearance outcomes count all MLB pitching appearances, including relief, with a 90-day maximum. Calendar negatives need subsequent MLB observation through the horizon and no intervening arm episode or uncertain-surgery interval. Active arm-IL rows are excluded. This conservative observation rule must be examined for selection bias.',
        '## Feature eras','',
        'Keep 2015-present as the full tracking-history cohort. Compare broad features across 2015-2016 and 2017-2019 separately because source velocity definitions changed. Hawk-Eye began in 2020; use 2020-2025 as an enhanced-tracking sensitivity era, subject to measured coverage. Arm-angle modern-era boundaries are empirical: first_season_meaningful_availability uses >=50% nonmissing appearances and must be checked each subsequent year. Treat 2026 as a location-definition sensitivity era because plate reference and strike-zone definitions changed. Earlier seasons are preserved.',
        '[MLB Statcast overview](https://www.mlb.com/glossary/statcast); [MLB 2020 tracking update](https://www.mlb.com/news/2020-statcast-update-includes-pose-tracking-capabilities); [Savant CSV definitions](https://baseballsavant.mlb.com/csv-docs); [Chadwick identifier register](https://github.com/chadwickbureau/register).',
        '## Evaluation preparation','',
        'src/evaluation.py accepts held-out predictions and a predeclared event cohort. It computes AP, ROC-AUC, precision, recall, F2, alert-budget curves at 1%, 2.5%, 5%, 10%, event recall, first warning lead times, and false alerts per pitcher-season. Primary practical target: event recall with a 7-30-day warning at 5% appearance alerts. Report both all qualifying and observable event denominators. Retrospective top-score budgets are not deployable thresholds; choose deployment thresholds on validation data only.',
        'Any future imputer, scaler, feature selector or alert threshold must be fitted only on training/validation history. Retain structural NULLs and consider explicit era-specific feature subsets. Never fill historical arm angle with zero.']
    (cfg['report_path']/'modeling_readiness.md').write_text('\n\n'.join(readiness)+'\n')
    finalize_documentation(cfg,summary)
    return dict(report,row_counts=summary)


def finalize_documentation(cfg,summary):
    """Keep human-readable status synchronized with the executed database audit."""
    readme=ROOT/'README.md';text=readme.read_text()
    start='<!-- EXECUTED-BUILD-START -->';end='<!-- EXECUTED-BUILD-END -->'
    block=(start+'\n'+f"Executed coverage: **{summary['earliest_game_date']}–{summary['latest_game_date']}**. "
        f"The **{summary['database_bytes']/1024**3:.2f} GiB** database contains **{summary['total_raw_pitches']:,} raw records**, "
        f"**{summary['total_appearances']:,} appearances**, and **{summary['starting_appearances']:,} official starts**. "
        f"The 30-day starter target has **{summary['positive_30d_rows']} positives ({summary['positive_30d_percentage']:.3f}%)** "
        f"representing **{summary['positive_30d_unique_events']} distinct events**. "
        f"Unresolved source failures: **{summary['failed_source_items']}**; unmatched injury identities: **{summary['unmatched_injury_records']}**.\n"+end)
    if start in text:
        text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
    else:text=text.replace('## Where the data lives',block+'\n\n## Where the data lives',1)
    readme.write_text(text)
    alignment=pd.read_csv(cfg['report_path']/'pitch_tracking_alignment_by_season.csv')
    early=alignment[alignment.season.isin([2015,2016])]
    missing=pd.read_csv(cfg['report_path']/'feature_missingness.csv').set_index('feature')
    note=(f"\n\nHistorical measurement findings: arm_angle_avg is missing for {missing.loc['arm_angle_avg','starter_missing_percentage']:.2f}% of valid starts; "
        f"velo_drop_game is missing for {missing.loc['velo_drop_game','starter_missing_percentage']:.2f}% because many starts do not supply 40 measured fastballs. "
        f"Native IVB is sparsely supplied in 2015–2016 ({int(early.native_measured_ivb_records.sum()):,} measured native records). "
        f"The conservative cross-source speed gate rejects {int(early.invalid_alignment_records.sum()):,} earlier-era records, even though physical pitch counts match. "
        "This is a measurement-comparability limitation, not a reason to fill IVB with zero. IVB is excluded from the curated v1 list. See pitch_tracking_alignment_by_season.csv.\n")
    for name in ('modeling_readiness.md','full_build_summary.md'):
        path=cfg['report_path']/name;current=path.read_text()
        if 'Historical measurement findings:' not in current:path.write_text(current+note)
