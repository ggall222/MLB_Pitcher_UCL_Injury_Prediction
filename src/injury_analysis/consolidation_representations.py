"""Small interpretable nested comparisons, using common rows and training-only fits."""
import numpy as np
import pandas as pd
from .consolidation_common import CACHE,table,read_table,inputs
from .consolidation_models import fit_named,summary_metrics
from .family_decomposition import paired_bootstrap_ap

CONTEXT=['age_at_appearance','throws_left','prior_reported_il_count','is_starter',
         'rest_days','pitches_prev_7d','pitches_prev_28d']
CONCEPTS={
 'Velocity': [('Prior baseline',['avg_fb_velo_baseline_mean']),('Departure',['velo_z']),
    ('Prior slope',['fb_velo_slope_prev_5']),('Recent level',['avg_fb_velo_prev_5']),('Current level',['avg_fb_velo']),('Velocity variability',['fb_velo_sd'])],
 'Release position':[('Prior baseline',['release_x_avg_baseline_mean','release_z_avg_baseline_mean']),
    ('Departure',['release_x_z','release_z_z']),('Prior slope',['release_x_slope_prev_5','release_z_slope_prev_5']),
    ('Recent level',['release_x_avg_prev_5','release_z_avg_prev_5']),('Current level',['release_x_avg','release_z_avg'])],
 'Release variability':[('Prior baseline',['release_x_sd_baseline_mean','release_z_sd_baseline_mean']),
    ('Departure',['release_x_sd_z','release_z_sd_z']),('Prior slope',['release_x_sd_slope_prev_5','release_z_sd_slope_prev_5']),
    ('Recent level',['release_x_sd_prev_5','release_z_sd_prev_5']),('Current variability',['release_x_sd','release_z_sd'])],
 'Extension':[('Prior baseline',['extension_avg_baseline_mean']),('Departure',['extension_z']),
    ('Prior slope',['extension_slope_prev_5']),('Recent level',['extension_avg_prev_5']),('Current level',['extension_avg'])],
 'Spin':[('Prior baseline',['spin_avg_baseline_mean']),('Departure',['spin_z']),
    ('Prior slope',['spin_slope_prev_5']),('Recent level',['spin_avg_prev_5']),('Current level',['spin_avg'])],
 'Movement':[('Prior ten-outing level',['horizontal_movement_avg_prev_10','vertical_movement_avg_prev_10']),
    ('Current-minus-recent',['horizontal_movement_avg_change_vs_prev_5','vertical_movement_avg_change_vs_prev_5']),
    ('Recent level',['horizontal_movement_avg_prev_5','vertical_movement_avg_prev_5']),
    ('Current level',['horizontal_movement_avg','vertical_movement_avg'])],
 'Command/location':[('Prior baseline',['zone_pct_baseline_mean']),('Departure',['zone_pct_z']),
    ('Prior slope',['zone_pct_slope_prev_5']),('Recent level',['zone_pct_prev_5']),
    ('Strike fractions',['strike_pct','first_pitch_strike_pct']),('Location variability',['location_variability'])],
 'Pitch mix':[('Prior ten-outing mix',['fastball_usage_pct_prev_10','sl_usage_pct_prev_10','ch_usage_pct_prev_10']),
    ('Current-minus-recent',['fastball_usage_pct_change_vs_prev_5','sl_usage_pct_change_vs_prev_5','ch_usage_pct_change_vs_prev_5']),
    ('Recent mix',['fastball_usage_pct_prev_5','sl_usage_pct_prev_5','ch_usage_pct_prev_5']),
    ('Current mix',['fastball_usage_pct','sl_usage_pct','ch_usage_pct']),('Pitch-type counts',['fastball_count','sl_pitch_count','ch_pitch_count'])],
 'Workload':[('Prior 28-day volume',['pitches_prev_28d']),('Prior 7-day volume',['pitches_prev_7d']),
    ('Workload acceleration',['workload_acceleration_7d_vs_21d']),('Acute/chronic ratio',['acute_chronic_pitch_ratio']),
    ('Rest',['rest_days']),('Current outing',['pitch_count']),('Season-to-date',['season_pitch_count'])],
}


def run_representations():
    f,events,_=inputs();train=f.loc[f.partition.eq('train')];valid=f.loc[f.partition.eq('validation')]
    rows=[];scores=[];allowed=set(read_table('feature_inventory').query('eligible_predictor').feature)
    for concept,steps in CONCEPTS.items():
        features=CONTEXT.copy() if concept!='Workload' else CONTEXT[:4]
        previous=None;scoremap={};namekeys=[]
        for i,(label,addition) in enumerate([('Shared context',[])]+steps):
            addition=[c for c in addition if c in allowed and c not in features];features+=addition
            name='representation_'+concept.replace('/','_').replace(' ','_')+f'_{i}'
            obj,score=fit_named(train,valid,features,name)
            row=dict(concept=concept,step=i,addition=label,added_features=';'.join(addition),features=len(features),
                     **summary_metrics(valid,score,events))
            row['ap_increment']=row['average_precision']-previous['average_precision'] if previous else 0.
            row['event_recall_increment']=row['injury_event_recall']-previous['injury_event_recall'] if previous else 0.
            rows.append(row);previous=row;scoremap[str(i)]=score
            scores.append(pd.DataFrame({'row_id':valid.row_id,'concept':concept,'step':i,'score':score}))
        bs,_=paired_bootstrap_ap(valid,scoremap)
        for row in rows:
            if row['concept']!=concept:continue
            i=row['step'];delta=bs[str(i)]-bs[str(max(0,i-1))]
            row['increment_low'],row['increment_high']=np.nanquantile(delta,[.025,.975])
        table(pd.DataFrame(rows),'representation_comparisons')
    pd.concat(scores,ignore_index=True).to_parquet(CACHE/'representation_validation_predictions.parquet',index=False)
    return pd.DataFrame(rows)


def run_supported_cohorts():
    """Do not call historically unsupported families useless; compare on common rows."""
    f,events,_=inputs();rows=[]
    definitions=[('Modern arm angle',f.game_date.ge('2020-01-01'),['arm_angle_avg','arm_angle_sd','arm_angle_change_vs_prev_5','arm_angle_z']),
        ('Starter within-outing velocity',f.is_starter.eq(True),['velo_drop_game'])]
    base=CONTEXT+['avg_fb_velo_baseline_mean','velo_z','release_z_sd_z','extension_z','spin_z']
    for label,mask,extra in definitions:
        common=f.loc[mask&f[extra].notna().all(axis=1)]
        train=common.loc[common.partition.eq('train')];valid=common.loc[common.partition.eq('validation')]
        counts={p:g.loc[g.target.eq(1),'next_event_id'].nunique() for p,g in [('train',train),('validation',valid)]}
        if counts['train']<30 or counts['validation']<10:
            rows.append(dict(analysis=label,status='insufficient common-cohort event support',train_events=counts['train'],validation_events=counts['validation']))
            continue
        scoremap={}
        for name,features in [('Base',base),('Base plus supported measurements',base+extra)]:
            obj,score=fit_named(train,valid,features,'supported_'+label.replace(' ','_')+'_'+name.replace(' ','_'))
            scoremap[name]=score
            rows.append(dict(analysis=label,variant=name,status='fit',train_events=counts['train'],validation_events=counts['validation'],
                             **summary_metrics(valid,score,events)))
        bs,_=paired_bootstrap_ap(valid,scoremap)
        lo,hi=np.nanquantile(bs['Base plus supported measurements']-bs['Base'],[.025,.975])
        rows[-1]['ap_increment_low']=lo;rows[-1]['ap_increment_high']=hi
    return table(pd.DataFrame(rows),'supported_cohort_sensitivity')
