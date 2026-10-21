"""Schema-backed concept/representation map and documented algebraic dependencies."""
import json
import re
import numpy as np
import pandas as pd
from .consolidation_common import ROOT,CACHE,CFG,table,inputs
from src.build_mechanical_features import CHANGE_NAMES
from src.build_baselines import Z_NAMES

GROUPS=['Velocity','Release position and variability','Extension','Arm angle','Spin','Pitch movement',
        'Location and strike fractions','Pitch mix and pitch-type counts','Workload and rest',
        'Role and recent starts','Appearance performance','Within-appearance velocity change',
        'Pitcher background and experience','Documented prior UCL history','Measurement coverage and history support']
PITCH_TYPES=['ff','si','fc','sl','st','cu','kc','ch','fs','sv']
CHANGE_BASE={v:k for k,v in CHANGE_NAMES.items()}
Z_BASE={v+'_z':k for k,v in Z_NAMES.items()}
SUPPORT={'velocity_measured_count','arm_angle_measured_count','velocity_coverage','arm_angle_coverage',
 'pitch_count_difference','flagged_pitch_count','prior_appearance_count','workload_history_appearances'}
BACKGROUND={'age_at_appearance','days_since_mlb_debut','throws_left','observed_prior_mlb_appearances',
 'observed_prior_mlb_starts','prior_reported_il_count','calendar_month'}
RAW_SOURCES={'avg_fb_velo':'release_speed;pitch_type','fb_velo_sd':'release_speed;pitch_type',
 'release_x_avg':'release_pos_x','release_z_avg':'release_pos_z','release_x_sd':'release_pos_x',
 'release_z_sd':'release_pos_z','extension_avg':'release_extension','extension_sd':'release_extension',
 'arm_angle_avg':'arm_angle','arm_angle_sd':'arm_angle','spin_avg':'release_spin_rate',
 'horizontal_movement_avg':'pfx_x','vertical_movement_avg':'pfx_z','zone_pct':'zone',
 'strike_pct':'pitch_result','first_pitch_strike_pct':'pitch_number;pitch_result',
 'ivb_inches':'validated MLB pitchData.breaks.breakVerticalInduced','spin_axis_avg':'spin_axis (circular mean)',
 'location_variability':'plate_x_sd;plate_z_sd','plate_x_sd':'plate_x','plate_z_sd':'plate_z',
 'avg_velocity':'release_speed','max_velocity':'release_speed','effective_speed_avg':'effective_speed',
 'max_fb_velo':'release_speed;pitch_type','median_fb_velo':'release_speed;pitch_type',
 'first_n_fb_velo':'release_speed;pitch_type;physical pitch order','last_n_fb_velo':'release_speed;pitch_type;physical pitch order',
 'velo_drop_game':'last_n_fb_velo;first_n_fb_velo','usage_pct':'pitch-type pitch_count;total_pitches',
 'pitch_count':'official boxscore pitch_count','fastball_usage_pct':'fastball_count;total_pitches',
 'fastball_count':'pitch_type;physical pitch count'}
RAW_SOURCES.update({
 'total_pitches':'physical Statcast pitch rows', 'statcast_batters_faced':'distinct at_bat_number',
 'high_velocity_pitch_count':'release_speed >= 95 mph', 'velocity_measured_count':'nonmissing cleaned release_speed',
 'arm_angle_measured_count':'nonmissing cleaned arm_angle','velocity_coverage':'velocity_measured_count;total_pitches',
 'arm_angle_coverage':'arm_angle_measured_count;total_pitches','pitch_count_difference':'total_pitches;official pitch_count',
 'flagged_pitch_count':'nonmissing pitch quality_flags','outs_recorded':'official boxscore outs',
 'innings_pitched':'official boxscore outs / 3','batters_faced':'official boxscore battersFaced',
 'walks':'official boxscore baseOnBalls','strikeouts':'official boxscore strikeOuts','hit_batters':'official boxscore hitBatsmen',
 'walk_rate':'walks;batters_faced','strikeout_rate':'strikeouts;batters_faced','hbp_count':'physical pitch is_hbp',
 'hbp_rate':'physical pitch is_hbp;physical pitch rows','age_at_appearance':'game_date;birth_date',
 'days_since_mlb_debut':'game_date;mlb_debut','throws_left':'consistent p_throws observed in appearance',
 'calendar_month':'game_date','prior_reported_il_count':'public_known_date of strictly earlier canonical IL episodes',
 'observed_prior_mlb_appearances':'strictly prior official MLB appearances in observed history',
 'observed_prior_mlb_starts':'strictly prior official MLB starts in observed history',
 'documented_prior_ucl':'confirmed prior procedure upper date bound','documented_prior_ucl_count':'confirmed prior procedure upper date bounds',
 'is_starter':'official boxscore gamesStarted or first pitching-order fallback',
 'prior_appearance_count':'strictly prior-day timing-reliable appearance rows'})


def classify(name,layer=''):
    """Naming rules are verified against the actual builder, not dictionary prose alone."""
    base=name;rep='current level';current=True;window='current appearance';note=''
    if name in Z_BASE:
        base=Z_BASE[name];rep='pitcher-relative z-score';window='current minus prior 20-outing mean / prior SD'
    elif '_baseline_' in name:
        base,kind=name.rsplit('_baseline_',1);rep='baseline '+kind;current=False;window='up to 20 strictly prior-day outings; >=5 valid for mean/SD'
    elif '_change_vs_prev_' in name:
        stem,n=name.rsplit('_change_vs_prev_',1);base=CHANGE_BASE.get(stem,stem)
        rep='change from recent history';window=f'current minus mean of previous {n} prior-day outings'
        note='Builder resolves precise window; generic dictionary descriptions may say previous five.'
    elif '_slope_prev_' in name:
        stem,n=name.rsplit('_slope_prev_',1);base=CHANGE_BASE.get(stem,stem)
        rep='slope/trend';current=False;window=f'OLS over previous {n} prior-day outings; >=3 valid'
        note='Builder defines OLS slope per outing; generic dictionary suffix rule can call this an average.'
    elif re.search(r'_prev_(3|5|10)$',name) and not name.startswith(('history_count','starter_')):
        base,n=name.rsplit('_prev_',1);rep='rolling historical level';current=False;window=f'previous {n} prior-day outings'
    pitch=next((p for p in PITCH_TYPES if base.startswith(p+'_')),'')
    core=base[len(pitch)+1:] if pitch else base
    support=name in SUPPORT or name.startswith('history_count_prev_') or name.endswith('_baseline_count') or bool(re.fullmatch(r'workload_\d+d_complete',name))
    if support:group=GROUPS[14]
    elif name.startswith('documented_prior_ucl'):group=GROUPS[13]
    elif name in BACKGROUND:group=GROUPS[12]
    elif name in ['first_n_fb_velo','last_n_fb_velo','velo_drop_game']:group=GROUPS[11]
    elif name in ['is_starter','starter_share_prev_5','starter_role_prev_5','starts_prev_30d','season_starts']:group=GROUPS[9]
    elif name in ['outs_recorded','innings_pitched','statcast_batters_faced','batters_faced','walks','strikeouts','hit_batters','hbp_count','hbp_rate','walk_rate','strikeout_rate']:group=GROUPS[10]
    elif name in ['total_pitches','pitch_count','high_velocity_pitch_count'] or layer=='workload_features':group=GROUPS[8]
    elif 'usage' in core or core in ['fastball_count','pitch_count']:group=GROUPS[7]
    elif core in ['plate_x_sd','plate_z_sd','location_variability','zone_pct','strike_pct','first_pitch_strike_pct']:group=GROUPS[6]
    elif 'movement' in core or core=='ivb_inches':group=GROUPS[5]
    elif 'spin' in core:group=GROUPS[4]
    elif core.startswith('arm_angle'):group=GROUPS[3]
    elif core.startswith('extension'):group=GROUPS[2]
    elif core.startswith('release_'):group=GROUPS[1]
    elif 'veloc' in core or 'velo' in core or core=='effective_speed_avg':group=GROUPS[0]
    else:group='Unclassified';note='Requires manual classification; excluded from predictive selection.'
    if layer=='workload_features':
        if name.startswith('season_'):rep='count';window='calendar year through prior day plus current outing';current=True
        elif name=='previous_season_innings':rep='background/context';window='completed previous calendar season';current=False
        elif name in ['acute_chronic_pitch_ratio','current_vs_previous_season_workload']:rep='ratio';current=True;window='current-inclusive workload ratio'
        elif name=='workload_acceleration_7d_vs_21d':rep='workload window';current=False;window='prior 7-day pitch rate minus prior days 8–28 rate'
        elif re.fullmatch(r'(pitches|innings|appearances)_\d+d',name):rep='workload window';current=True;window='[date−N days,date) plus current outing'
        else:rep='workload window';current=False;window='strictly before current date; same-day other outings excluded'
    if name in ['starter_role_prev_5','starter_share_prev_5']:rep='rolling historical level';current=False;window='previous five prior-day official appearances'
    if name in BACKGROUND:rep='background/context';current=name in ['throws_left','calendar_month'];window='known on appearance date; experience/history strictly prior'
    if name.startswith('documented_prior_ucl'):
        rep='background/context';current=False;window='procedure upper date bound precedes outing; public-known date not established'
        note='Registered source-safe, but excluded by the existing analysis knowledge-time policy.'
    if support:
        if not name.endswith('_baseline_count'):rep='coverage / support'
        if name.startswith('history_count') or name in ['prior_appearance_count','workload_history_appearances']:current=False;window='strictly prior-day appearance history'
    if rep=='current level' and 'usage' in name:rep='usage fraction'
    if rep=='current level' and (name.endswith('_count') or name in ['total_pitches','pitch_count']):rep='count'
    if name=='velo_drop_game':rep='within-appearance change';window='last minus first 20 measured FF/SI/FC pitches; >=40 required'
    elif name in ['first_n_fb_velo','last_n_fb_velo']:window='first/last 20 measured FF/SI/FC pitches; >=40 required'
    sources=RAW_SOURCES.get(core,core)
    if pitch:
        # A pitch-type count is a Statcast aggregation, not the similarly named
        # official appearance pitch_count. All type-specific fields filter types.
        if core=='pitch_count':sources='physical Statcast pitch rows'
        sources+='; pitch_type = '+pitch.upper()
    if layer=='workload_features':sources='official boxscore pitch_count/innings/is_starter/game_date; completed prior-season innings as applicable'
    if rep in ['rolling historical level','baseline mean','baseline sd','baseline count','change from recent history','slope/trend','pitcher-relative z-score']:
        sources=base+' (appearance aggregate); '+sources
    temporal=('Departure' if rep in ['change from recent history','pitcher-relative z-score','within-appearance change']
        else 'Trend' if rep=='slope/trend' else 'Profile/history' if rep in ['rolling historical level','baseline mean','baseline sd','background/context']
        else 'Availability' if support else 'Workload' if group==GROUPS[8] else 'Current state')
    availability='after completed outing' if current else 'available before outing; historical inputs exclude current date'
    if name.startswith('documented_prior_ucl'):availability='historical public availability unknown; excluded'
    if name in BACKGROUND and name!='throws_left':availability='available on prediction date before outing; reported history strictly prior'
    return dict(feature=name,feature_group=group,underlying_concept=base,concept_family=core,
        representation=rep,pitch_type=pitch,pitch_type_specific=bool(pitch),source_variables=sources,
        source_layer=layer,timing_availability=availability,
        current_appearance_included=current,history_window=window,signal_interpretation=temporal,
        classification_ambiguity=note)


def dependency_audit(train,features):
    """Verify algebra on co-observed training rows; never infer exactness from r alone."""
    rows=[]
    def check(name,formula,value,sources):
        if name not in features:return
        actual=train[name].astype(float).to_numpy();expected=np.asarray(value,dtype=float)
        valid=np.isfinite(actual)&np.isfinite(expected)
        error=np.abs(actual[valid]-expected[valid])
        rows.append(dict(feature=name,source_features=';'.join(sources),formula=formula,training_coobserved_rows=int(valid.sum()),
            max_absolute_error=float(error.max()) if len(error) else np.nan,
            verified_exact=bool(len(error) and np.allclose(actual[valid],expected[valid],rtol=1e-8,atol=1e-8)),
            note='Algebraic redundancy is conditional on source co-availability; missingness can differ.'))
    check('innings_pitched','outs_recorded / 3',train.outs_recorded/3,['outs_recorded'])
    check('location_variability','sqrt(plate_x_sd² + plate_z_sd²)',np.hypot(train.plate_x_sd,train.plate_z_sd),['plate_x_sd','plate_z_sd'])
    check('velo_drop_game','last_n_fb_velo − first_n_fb_velo',train.last_n_fb_velo-train.first_n_fb_velo,['last_n_fb_velo','first_n_fb_velo'])
    check('pitch_count_difference','total_pitches − pitch_count',train.total_pitches-train.pitch_count,['total_pitches','pitch_count'])
    for p in PITCH_TYPES+['fastball']:
        count=p+'_pitch_count' if p!='fastball' else 'fastball_count'
        check(p+'_usage_pct',f'{count} / total_pitches',train[count]/train.total_pitches,[count,'total_pitches'])
    check('fastball_count','ff_pitch_count + si_pitch_count + fc_pitch_count',train.ff_pitch_count+train.si_pitch_count+train.fc_pitch_count,['ff_pitch_count','si_pitch_count','fc_pitch_count'])
    for n in [7,14,28]:
        for stem,current in [('pitches','pitch_count'),('innings','innings_pitched')]:
            check(f'{stem}_{n}d',f'{stem}_prev_{n}d + {current}',train[f'{stem}_prev_{n}d']+train[current],[f'{stem}_prev_{n}d',current])
        check(f'appearances_{n}d',f'appearances_prev_{n}d + 1',train[f'appearances_prev_{n}d']+1,[f'appearances_prev_{n}d'])
    check('workload_acceleration_7d_vs_21d','prior7/7 − (prior28−prior7)/21',train.pitches_prev_7d/7-(train.pitches_prev_28d-train.pitches_prev_7d)/21,['pitches_prev_7d','pitches_prev_28d'])
    check('acute_chronic_pitch_ratio','4 × pitches_7d / pitches_28d',4*train.pitches_7d/train.pitches_28d,['pitches_7d','pitches_28d'])
    for name in features:
        if '_change_vs_prev_' in name:
            stem,n=name.rsplit('_change_vs_prev_',1);base=CHANGE_BASE.get(stem,stem)
            if base in train and base+'_prev_'+n in train:
                check(name,base+' − '+base+'_prev_'+n,train[base]-train[base+'_prev_'+n],[base,base+'_prev_'+n])
        if name in Z_BASE:
            base=Z_BASE[name];mean=base+'_baseline_mean';sd=base+'_baseline_sd'
            check(name,f'({base} − {mean}) / {sd}',(train[base]-train[mean])/train[sd].replace(0,np.nan),[base,mean,sd])
    return table(pd.DataFrame(rows),'deterministic_dependencies')


def run_inventory():
    f,events,spec=inputs();train=f.loc[f.partition.eq('train')]
    names=json.loads((ROOT/'data/exports/feature_columns.json').read_text())+['prior_reported_il_count','calendar_month']
    layers=json.loads((CACHE/'source_layers.json').read_text())
    dictionary=pd.read_csv(ROOT/'docs/data_dictionary.csv').query("table=='model_features'").set_index('variable_name')
    rows=[]
    for name in names:
        layer=next((k for k,v in layers.items() if name in v),'injury_analysis.cohort')
        r=classify(name,layer);x=train[name].astype(float).replace([np.inf,-np.inf],np.nan)
        valid=x.notna();counts=x.value_counts();pitchers=train.loc[valid,'pitcher_id'].nunique()
        r.update(training_missing_fraction=x.isna().mean(),training_unique_values=x.nunique(),
            training_observed_pitchers=pitchers,training_observed_rows=int(valid.sum()),
            training_modal_fraction=counts.iloc[0]/valid.sum() if valid.any() else np.nan,
            training_positive_rows_observed=int(train.loc[valid,'target'].sum()),
            dictionary_description=dictionary.loc[name,'description'] if name in dictionary.index else 'Existing injury cohort context field',
            dictionary_unit=dictionary.loc[name,'unit'] if name in dictionary.index else 'count',
            registry_model_safe=name in dictionary.index and bool(dictionary.loc[name,'safe_for_modeling']))
        reasons=[]
        if name.startswith('documented_prior_ucl'):reasons.append('publication timing not verified')
        if 'spin_axis' in name:reasons.append('circular degree coordinate needs sine/cosine encoding before linear selection')
        if r['feature_group']=='Unclassified':reasons.append('classification unresolved')
        if x.nunique()<=1:reasons.append('constant or unobserved in training')
        if x.isna().mean()>CFG['maximum_training_missing_fraction']:reasons.append('insufficient long-history training measurement support')
        if pitchers<CFG['minimum_observed_training_pitchers']:reasons.append('too few observed training pitchers')
        r['screen_exclusion']='; '.join(reasons);r['eligible_predictor']=not reasons
        r['near_zero_variance_flag']=bool(valid.any() and counts.iloc[0]/valid.sum()>.995)
        r['support_only_sensitivity']=r['feature_group']==GROUPS[14]
        rows.append(r)
    inventory=table(pd.DataFrame(rows),'feature_inventory')
    table(inventory.loc[inventory.classification_ambiguity.ne('')], 'dictionary_and_timing_caveats')
    table(inventory.groupby('feature_group').agg(features=('feature','size'),eligible=('eligible_predictor','sum'),
        near_constant=('near_zero_variance_flag','sum')).reindex(GROUPS).reset_index(),'feature_group_inventory')
    dependency_audit(train,names)
    missing=[]
    for (part,year),g in f.groupby(['partition','season']):
        miss=g[names].astype(float).replace([np.inf,-np.inf],np.nan).isna().mean()
        missing.extend(dict(partition=part,season=year,feature=c,missing_fraction=v,rows=len(g)) for c,v in miss.items())
    table(pd.DataFrame(missing),'feature_availability_by_season')
    counts=[]
    for part,g in f.groupby('partition'):
        ids=set(g.loc[g.target.eq(1),'next_event_id'])
        e=events.loc[events.injury_event_id.isin(ids)]
        counts.append(dict(partition=part,rows=len(g),positive_rows=int(g.target.sum()),
            represented_events=len(ids),initial_explicit_ucl_events=int(e.initial_explicit_ucl.sum()),
            later_supplemental_ucl_confirmation=int(e.supplemental_ucl_confirmed.sum()),
            first_appearance=g.game_date.min(),last_appearance=g.game_date.max()))
    table(pd.DataFrame(counts),'event_support')
    print(f'Inventoried {len(inventory)} fields; {inventory.eligible_predictor.sum()} pass train-only measurement/timing screening.',flush=True)
    return inventory
