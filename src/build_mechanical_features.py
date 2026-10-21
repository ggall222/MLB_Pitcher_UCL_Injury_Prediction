"""Strictly prior-day histories also avoid ambiguous same-day/doubleheader ordering."""
import numpy as np
import pandas as pd

METRICS = ['avg_fb_velo', 'release_x_avg', 'release_z_avg', 'release_x_sd', 'release_z_sd',
           'extension_avg', 'arm_angle_avg', 'spin_avg', 'zone_pct']
CHANGE_NAMES = {'avg_fb_velo': 'fb_velo', 'release_x_avg': 'release_x', 'release_z_avg': 'release_z',
                'release_x_sd': 'release_x_sd', 'release_z_sd': 'release_z_sd',
                'extension_avg': 'extension', 'arm_angle_avg': 'arm_angle', 'spin_avg': 'spin', 'zone_pct': 'zone_pct'}


def prior_history(group, row):
    return group[group.game_date < row.game_date]


def slope(series):
    values = series.to_numpy(dtype=float)
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.arange(len(values))[ok], values[ok], 1)[0])


def build(appearances, cfg):
    from .history_math import prior_ends, moments, slopes
    metrics = METRICS + [pt.lower() + '_' + col for pt in cfg['major_pitch_types']
                        for col in ['avg_velocity', 'spin_avg', 'release_x_avg', 'release_z_avg', 'extension_avg']]
    extra=['fb_velo_sd','horizontal_movement_avg','vertical_movement_avg','location_variability',
           'strike_pct','first_pitch_strike_pct','fastball_usage_pct'] + [pt.lower()+'_usage_pct' for pt in cfg['major_pitch_types']]
    metrics += [c for c in extra if c in appearances]
    pieces=[]
    for pid,g in appearances.groupby('pitcher_id',sort=False):
        g=g.sort_values(['game_date','scheduled_start','game_id']).reset_index(drop=True)
        h=g[~g.timing_uncertain.fillna(True)] if 'timing_uncertain' in g else g
        dates=h.game_date.to_numpy();ends=np.searchsorted(dates,g.game_date.to_numpy(),side='left')
        values=h[metrics].to_numpy(dtype=float);previous=np.maximum(ends-1,0)
        maximum=np.full(len(g),np.datetime64('NaT'),dtype='datetime64[ns]')
        if len(dates):maximum[ends>0]=dates[previous[ends>0]]
        out={'game_id':g.game_id.to_numpy(),'pitcher_id':np.repeat(pid,len(g)),
             'prior_appearance_count':ends,'history_max_date':maximum}
        for w in (3,5,10):
            n,mean,_=moments(values,ends,w)
            mean[(n<max(2,w//2))|(ends[:,None]<w)]=np.nan
            out[f'history_count_prev_{w}']=np.minimum(ends,w)
            for j,c in enumerate(metrics):out[f'{c}_prev_{w}']=mean[:,j]
        slope_values=slopes(h[METRICS].to_numpy(dtype=float),ends)
        for j,c in enumerate(METRICS):
            name=CHANGE_NAMES[c]
            out[name+'_change_vs_prev_5']=g[c].to_numpy()-out[c+'_prev_5']
            out[name+'_slope_prev_5']=slope_values[:,j]
        out['fb_velo_change_vs_prev_3']=g.avg_fb_velo.to_numpy()-out['avg_fb_velo_prev_3']
        for c in extra:
            if c in g:out[c+'_change_vs_prev_5']=g[c].to_numpy()-out[c+'_prev_5']
        pieces.append(pd.DataFrame(out))
    return pd.concat(pieces,ignore_index=True)
