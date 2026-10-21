"""Within-pitcher z-scores: at most 20 prior outings, at least 5 valid observations."""
import numpy as np
import pandas as pd
from .build_mechanical_features import METRICS, prior_history

Z_NAMES = dict(zip(METRICS, ['velo', 'release_x', 'release_z', 'release_x_sd', 'release_z_sd',
                            'extension', 'arm_angle', 'spin', 'zone_pct']))


def build(appearances, cfg):
    from .history_math import prior_ends, moments
    pieces=[]
    for pid,g in appearances.groupby('pitcher_id',sort=False):
        g=g.sort_values(['game_date','scheduled_start','game_id']).reset_index(drop=True)
        h=g[~g.timing_uncertain.fillna(True)] if 'timing_uncertain' in g else g
        dates=h.game_date.to_numpy();ends=np.searchsorted(dates,g.game_date.to_numpy(),side='left')
        n,mean,sd=moments(h[METRICS].to_numpy(dtype=float),ends,cfg['baseline_appearances'])
        mean[n<cfg['baseline_min_history']]=np.nan
        sd[n<cfg['baseline_min_history']]=np.nan
        # Cancellation near a constant series must not create enormous artificial z-scores.
        sd[np.abs(sd)<1e-6]=0
        values=g[METRICS].to_numpy(dtype=float)
        z=np.divide(values-mean,sd,out=np.full_like(mean,np.nan),where=sd>1e-6)
        maximum=np.full(len(g),np.datetime64('NaT'),dtype='datetime64[ns]')
        if len(dates):maximum[ends>0]=dates[ends[ends>0]-1]
        out={'game_id':g.game_id.to_numpy(),'pitcher_id':np.repeat(pid,len(g)),
             'baseline_max_date':maximum}
        for j,c in enumerate(METRICS):
            out[c+'_baseline_count']=n[:,j].astype(int)
            out[c+'_baseline_mean']=mean[:,j];out[c+'_baseline_sd']=sd[:,j]
            out[Z_NAMES[c]+'_z']=z[:,j]
        pieces.append(pd.DataFrame(out))
    return pd.concat(pieces,ignore_index=True)
