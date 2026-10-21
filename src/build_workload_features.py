"""Official MLB workload, vectorized over strictly prior-day observations."""
import numpy as np
import pandas as pd


def build(appearances, box, seasons, cfg):
    previous={(r.pitcher_id,r.season):r.innings_pitched for r in seasons.itertuples()}
    boxes={pid:g.sort_values(['game_date','game_id']).reset_index(drop=True) for pid,g in box.groupby('pitcher_id')}
    pieces=[]
    for pid,a in appearances.groupby('pitcher_id',sort=False):
        a=a.reset_index(drop=True);b=boxes.get(pid,box.iloc[:0])
        days=a.game_date.to_numpy(dtype='datetime64[ns]');past=b.game_date.to_numpy(dtype='datetime64[ns]')
        ends=np.searchsorted(past,days,side='left')
        bp=b.pitch_count.to_numpy(dtype=float);bi=b.innings_pitched.to_numpy(dtype=float)
        bs=b.is_starter.fillna(False).to_numpy(dtype=float)
        cp=np.r_[0,np.nancumsum(bp)];ci=np.r_[0,np.nancumsum(bi)];cs=np.r_[0,np.cumsum(bs)]
        currentp=a.pitch_count.to_numpy(dtype=float);currenti=a.innings_pitched.to_numpy(dtype=float)
        last=np.full(len(a),np.datetime64('NaT'),dtype='datetime64[ns]')
        if len(past):last[ends>0]=past[ends[ends>0]-1]
        rest=(days-last)/np.timedelta64(1,'D')-1
        out={'game_id':a.game_id.to_numpy(),'pitcher_id':np.repeat(pid,len(a)),
             'workload_max_date':last,'rest_days':np.maximum(rest,0),'workload_history_appearances':ends}
        for horizon in (7,14,28):
            left=np.searchsorted(past,days-np.timedelta64(horizon,'D'),side='left')
            p=cp[ends]-cp[left];i=ci[ends]-ci[left]
            out[f'pitches_prev_{horizon}d']=p;out[f'pitches_{horizon}d']=p+currentp
            out[f'innings_prev_{horizon}d']=i;out[f'innings_{horizon}d']=i+currenti
            out[f'appearances_prev_{horizon}d']=ends-left
            out[f'appearances_{horizon}d']=ends-left+1
            out[f'workload_{horizon}d_complete']=(days-np.timedelta64(horizon,'D'))>=np.datetime64(cfg['workload_start_date'])
        left=np.searchsorted(past,days-np.timedelta64(30,'D'),side='left')
        out['starts_prev_30d']=(cs[ends]-cs[left]).astype(int)
        role=(cs[ends]-cs[np.maximum(ends-5,0)])/5
        role[ends<5]=np.nan
        out['starter_share_prev_5']=role
        out['starter_role_prev_5']=pd.array([bool(v>=.6) if np.isfinite(v) else None for v in role],dtype='boolean')
        season_left=np.searchsorted(past,a.game_date.dt.to_period('Y').dt.start_time.to_numpy(),side='left')
        out['season_pitch_count']=cp[ends]-cp[season_left]+currentp
        out['season_innings']=ci[ends]-ci[season_left]+currenti
        out['season_appearances']=ends-season_left+1
        out['season_starts']=cs[ends]-cs[season_left]+a.is_starter.to_numpy(dtype=float,na_value=np.nan)
        prev=np.array([previous.get((pid,year-1),np.nan) for year in a.game_date.dt.year],dtype=float)
        out['previous_season_innings']=prev
        out['current_vs_previous_season_workload']=np.divide(out['season_innings'],prev,out=np.full(len(a),np.nan),where=prev>0)
        older=out['pitches_prev_28d']-out['pitches_prev_7d']
        out['workload_acceleration_7d_vs_21d']=out['pitches_prev_7d']/7-older/21
        out['acute_chronic_pitch_ratio']=np.divide(out['pitches_7d']/7,out['pitches_28d']/28,out=np.full(len(a),np.nan),where=out['pitches_28d']>0)
        pieces.append(pd.DataFrame(out))
    return pd.concat(pieces,ignore_index=True)
