"""Current-appearance metrics from physical pitches; raw outliers remain in pitches."""
import numpy as np
import pandas as pd

KEYS = ['game_id', 'pitcher_id']


def circular_mean(series):
    values = series.dropna().to_numpy(dtype=float)
    if not len(values):
        return np.nan
    radians = np.deg2rad(values)
    sine, cosine = np.sin(radians).mean(), np.cos(radians).mean()
    if np.hypot(sine, cosine) < 1e-8:
        return np.nan
    return float(np.rad2deg(np.arctan2(sine, cosine)) % 360)


def aggregate_pitches(pitches, cfg):
    p = pitches[pitches.is_actual_pitch.fillna(False)].copy()
    p = p.sort_values(KEYS + ['at_bat_number', 'pitch_number'])
    if 'ivb_inches' not in p:
        p['ivb_inches'] = np.nan
    for col, lo, hi in [('release_speed', 30, 110), ('release_pos_x', -7, 7),
                        ('release_pos_z', 0, 9), ('release_extension', 2, 9),
                        ('release_spin_rate', 0, 4500), ('arm_angle', -90, 180)]:
        p[col] = p[col].where(p[col].between(lo, hi))
    p['is_zone'] = p.zone.between(1, 9).astype(float).where(p.zone.notna())
    # Official-strike convention includes balls put in play.
    p['is_strike'] = p.pitch_result.isin(['S', 'X']).astype(float).where(p.pitch_result.notna())
    p['first_pitch_strike'] = p.is_strike.where(p.pitch_number.eq(1))
    p['high_velocity'] = p.release_speed.ge(cfg['high_velocity_mph'])
    groups = p.groupby(KEYS, sort=True)
    a = groups.agg(total_pitches=('pitch_number', 'size'), statcast_batters_faced=('at_bat_number', 'nunique'),
                   avg_velocity=('release_speed', 'mean'), max_velocity=('release_speed', 'max'),
                   effective_speed_avg=('effective_speed', 'mean'),
                   release_x_avg=('release_pos_x', 'mean'), release_z_avg=('release_pos_z', 'mean'),
                   release_x_sd=('release_pos_x', 'std'), release_z_sd=('release_pos_z', 'std'),
                   extension_avg=('release_extension', 'mean'), extension_sd=('release_extension', 'std'),
                   arm_angle_avg=('arm_angle', 'mean'), arm_angle_sd=('arm_angle', 'std'),
                   spin_avg=('release_spin_rate', 'mean'), ivb_inches=('ivb_inches', 'mean'),
                   horizontal_movement_avg=('pfx_x', 'mean'),
                   vertical_movement_avg=('pfx_z', 'mean'), zone_pct=('is_zone', 'mean'),
                   strike_pct=('is_strike', 'mean'), first_pitch_strike_pct=('first_pitch_strike', 'mean'),
                   hbp_count=('is_hbp', 'sum'), hbp_rate=('is_hbp', 'mean'),
                   plate_x_sd=('plate_x', 'std'), plate_z_sd=('plate_z', 'std'),
                   high_velocity_pitch_count=('high_velocity', 'sum'),
                   velocity_measured_count=('release_speed', 'count'), arm_angle_measured_count=('arm_angle', 'count'))
    a['location_variability'] = np.hypot(a.plate_x_sd, a.plate_z_sd)
    a['velocity_coverage'] = a.velocity_measured_count / a.total_pitches
    a['arm_angle_coverage'] = a.arm_angle_measured_count / a.total_pitches
    # IVB is supplemental native MLB telemetry; pfx_z remains separate in feet.
    fb = p[p.pitch_type.isin(cfg['fastball_types'])]
    b = fb.groupby(KEYS).agg(avg_fb_velo=('release_speed', 'mean'), max_fb_velo=('release_speed', 'max'),
                             median_fb_velo=('release_speed', 'median'), fb_velo_sd=('release_speed', 'std'),
                             fastball_count=('pitch_number', 'size'))
    a = a.join(b)
    a['fastball_count'] = a.fastball_count.fillna(0).astype(int)
    a['fastball_usage_pct'] = a.fastball_count / a.total_pitches
    n = cfg['velocity_segment_pitches']
    segments = []
    for key, g in fb.groupby(KEYS):
        v = g.release_speed.dropna()
        enough = len(v) >= 2 * n  # no overlap between first/last segments
        segments.append(dict(zip(KEYS, key), first_n_fb_velo=v.iloc[:n].mean() if enough else np.nan,
                             last_n_fb_velo=v.iloc[-n:].mean() if enough else np.nan,
                             velo_drop_game=(v.iloc[-n:].mean() - v.iloc[:n].mean()) if enough else np.nan))
    if segments:
        a = a.join(pd.DataFrame(segments).set_index(KEYS))
    else:
        a[['first_n_fb_velo', 'last_n_fb_velo', 'velo_drop_game']] = np.nan
    types = p.assign(pitch_type=p.pitch_type.fillna('UN')).groupby(KEYS + ['pitch_type']).agg(
        pitch_count=('pitch_number', 'size'), avg_velocity=('release_speed', 'mean'),
        ivb_inches=('ivb_inches', 'mean'),
        spin_avg=('release_spin_rate', 'mean'), horizontal_movement_avg=('pfx_x', 'mean'),
        vertical_movement_avg=('pfx_z', 'mean'), release_x_avg=('release_pos_x', 'mean'),
        release_z_avg=('release_pos_z', 'mean'), extension_avg=('release_extension', 'mean'),
        arm_angle_avg=('arm_angle', 'mean'), zone_pct=('is_zone', 'mean'),
        strike_pct=('is_strike', 'mean'), spin_axis_avg=('spin_axis', circular_mean))
    types['usage_pct'] = types.pitch_count / types.index.droplevel('pitch_type').map(a.total_pitches)
    types = types.reset_index()
    wide = {}
    for pt in cfg['major_pitch_types']:
        subset = types[types.pitch_type.eq(pt)].set_index(KEYS)
        for col in ['pitch_count', 'avg_velocity', 'spin_avg', 'horizontal_movement_avg',
                    'vertical_movement_avg', 'release_x_avg', 'release_z_avg', 'extension_avg',
                    'arm_angle_avg', 'zone_pct', 'strike_pct', 'spin_axis_avg', 'usage_pct', 'ivb_inches']:
            wide[pt.lower() + '_' + col] = subset[col].reindex(a.index)
            if col in ('pitch_count', 'usage_pct'):
                wide[pt.lower() + '_' + col] = wide[pt.lower() + '_' + col].fillna(0)
    a = pd.concat([a, pd.DataFrame(wide, index=a.index)], axis=1)
    return a.reset_index(), types


def build(db, cfg, season=None):
    alignment=alignment_view(db,season)
    where = '' if season is None else f' WHERE year(p.game_date)={int(season)}'
    p = db.query('SELECT p.* EXCLUDE(raw_json), t.ivb_inches FROM pitches p LEFT JOIN '+alignment+' t USING(game_id,at_bat_number,pitch_number)' + where)
    if p.empty:
        raise RuntimeError('No Statcast pitches available; cannot fabricate a modeling dataset')
    a, types = aggregate_pitches(p, cfg)
    metadata = db.query('''SELECT b.*, g.game_date, g.scheduled_start, g.season, g.game_type, g.timing_uncertain,
                           t.abbreviation AS team, o.abbreviation AS opponent,
                           p.pitcher_name, p.birth_date, p.mlb_debut, p.throws, p.position
                           FROM boxscore_appearances b JOIN games g USING(game_id)
                           LEFT JOIN team_seasons t ON t.team_id=b.team_id AND t.season=g.season
                           LEFT JOIN team_seasons o ON o.team_id=b.opponent_id AND o.season=g.season
                           JOIN pitchers p USING(pitcher_id)''' + ('' if season is None else f' WHERE g.season={int(season)}')).drop(columns='raw_json')
    result = a.merge(metadata, on=KEYS, how='left', validate='one_to_one')
    # Keep missing joins visible. Raw-pitch dates identify these rows for validation.
    dates = p.groupby(KEYS).game_date.first().rename('pitch_date').reset_index()
    result = result.merge(dates, on=KEYS, validate='one_to_one')
    result['game_date'] = result.game_date.fillna(result.pitch_date)
    result = result.drop(columns='pitch_date')
    hand = p.groupby(KEYS).p_throws.agg(lambda x: x.dropna().unique()[0] if x.dropna().nunique()==1 else None).rename('appearance_throws').reset_index()
    result = result.merge(hand,on=KEYS,validate='one_to_one')
    result['throws_left'] = result.appearance_throws.eq('L').where(result.appearance_throws.notna()).astype('boolean')
    result['boxscore_missing'] = result.pitch_count.isna()
    result['pitch_count_difference'] = result.total_pitches - result.pitch_count
    result['walk_rate'] = result.walks / result.batters_faced.replace(0, np.nan)
    result['strikeout_rate'] = result.strikeouts / result.batters_faced.replace(0, np.nan)
    result['age_at_appearance'] = (result.game_date - result.birth_date).dt.days / 365.2425
    result['days_since_mlb_debut'] = (result.game_date - result.mlb_debut).dt.days
    flags = p.groupby(KEYS).quality_flags.apply(lambda s: int(s.notna().sum())).rename('flagged_pitch_count').reset_index()
    result = result.merge(flags, on=KEYS, validate='one_to_one')
    result['quality_eligible'] = (~result.boxscore_missing & ~result.timing_uncertain.fillna(True) &
                                 result.velocity_coverage.ge(0.8) & result.pitch_count_difference.abs().le(2))
    result = result.sort_values(['pitcher_id', 'game_date', 'scheduled_start', 'game_id']).reset_index(drop=True)
    for col in ['is_starter', 'timing_uncertain', 'boxscore_missing', 'quality_eligible']:
        result[col] = result[col].astype('boolean')
    return result, types


def alignment_view(db,season=None):
    alignment='aligned_pitch_tracking'
    if season is not None:
        # Restrict before window functions, including native JSON parsing.
        from .config import ROOT
        definition=(ROOT/'sql/pitch_tracking_join.sql').read_text().split('AS\nWITH physical AS (',1)[1]
        definition='WITH physical AS ('+definition
        definition=definition.replace('FROM pitches WHERE is_actual_pitch',f'FROM pitches WHERE is_actual_pitch AND year(game_date)={int(season)}')
        definition=definition.replace('FROM pitch_tracking\n',f'FROM pitch_tracking WHERE game_id IN (SELECT game_id FROM games WHERE season={int(season)})\n')
        db.con.execute('CREATE OR REPLACE TEMP VIEW _season_aligned_pitch_tracking AS '+definition)
        alignment='_season_aligned_pitch_tracking'
    return alignment
