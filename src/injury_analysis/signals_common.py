"""Separate artifacts for the feature decomposition / trajectory-shape extension."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import duckdb
from .common import ROOT, CACHE as ORIGINAL_CACHE, CFG as ORIGINAL_CFG, source_signature, sha
from .cohort import load

CONFIG_PATH = ROOT / 'config/injury_signals.yaml'
CFG = yaml.safe_load(CONFIG_PATH.read_text())
OUT = ROOT / CFG['output_path']
CACHE = ROOT / CFG['cache_path']
for p in [OUT, OUT/'tables', OUT/'figures', CACHE]: p.mkdir(parents=True, exist_ok=True)

BASELINE_COUNTS = {
    'velo_z':'avg_fb_velo_baseline_count', 'release_x_z':'release_x_avg_baseline_count',
    'release_z_z':'release_z_avg_baseline_count', 'release_x_sd_z':'release_x_sd_baseline_count',
    'release_z_sd_z':'release_z_sd_baseline_count', 'extension_z':'extension_avg_baseline_count',
    'arm_angle_z':'arm_angle_avg_baseline_count', 'spin_z':'spin_avg_baseline_count',
    'zone_pct_z':'zone_pct_baseline_count',
}
TRAJECTORY_FEATURES = [
    'velo_z','fb_velo_change_vs_prev_5','fb_velo_sd_change_vs_prev_5',
    'release_x_z','release_z_z','release_x_sd_z','release_z_sd_z',
    'extension_z','arm_angle_z','spin_z','horizontal_movement_avg_change_vs_prev_5',
    'vertical_movement_avg_change_vs_prev_5','zone_pct_z','location_variability_change_vs_prev_5',
    'fastball_usage_pct_change_vs_prev_5','rest_days','pitches_prev_28d',
    'workload_acceleration_7d_vs_21d','acute_chronic_pitch_ratio']
LABELS = {'velo_z':'Velocity (pitcher z)', 'fb_velo_change_vs_prev_5':'Velocity change (mph)',
    'fb_velo_sd_change_vs_prev_5':'Velocity SD change (mph)', 'release_x_z':'Release X (pitcher z)',
    'release_z_z':'Release Z (pitcher z)', 'release_x_sd_z':'Release X variability (z)',
    'release_z_sd_z':'Release Z variability (z)', 'extension_z':'Extension (pitcher z)',
    'arm_angle_z':'Arm angle (pitcher z)', 'spin_z':'Spin (pitcher z)', 'zone_pct_z':'Zone percentage (pitcher z)',
    'workload_acceleration_7d_vs_21d':'Workload acceleration (pitches/day)',
    'acute_chronic_pitch_ratio':'Acute/chronic pitch-rate ratio', 'rest_days':'Rest (days)',
    'pitches_prev_28d':'Prior 28-day pitches'}

def table(frame, name):
    frame.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    return frame

def read_table(name): return pd.read_csv(OUT/'tables'/f'{name}.csv')

def figure(fig, name):
    for suffix in ['png','pdf']: fig.savefig(OUT/'figures'/f'{name}.{suffix}',bbox_inches='tight',dpi=220)
    return fig

def inputs():
    """Reuse frozen cohort, labels and matching; project a few extra existing fields."""
    a, events, spec = load()
    before = source_signature()
    extra = sorted(set(BASELINE_COUNTS.values()) | set(TRAJECTORY_FEATURES))
    extra = [c for c in extra if c not in a]
    with duckdb.connect(str(ROOT/ORIGINAL_CFG['database_path']),read_only=True) as con:
        columns = set(con.execute('describe model_features').fetchdf().column_name)
        absent = set(extra)-columns
        if absent: raise ValueError(f'Required existing trajectory metrics missing: {absent}')
        projection = ','.join('"'+c+'"' for c in extra)
        supplement = con.execute(f'SELECT game_id,pitcher_id,{projection} FROM model_features').fetchdf()
    a = a.merge(supplement,on=['pitcher_id','game_id'],how='left',validate='one_to_one')
    assert source_signature() == before
    matches = pd.read_csv(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv',
        parse_dates=['anchor_date','control_anchor_date'])
    return a,events,spec,matches

def signature():
    return dict(source=source_signature(), config=sha(CONFIG_PATH),
        matching=sha(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv'),
        original_cohort=sha(ORIGINAL_CACHE/'manifest.json'))

def cluster_draws(ids, repetitions=None):
    unique, inverse = np.unique(ids,return_inverse=True)
    b = CFG['bootstrap_replicates'] if repetitions is None else repetitions
    rng = np.random.default_rng(CFG['seed'])
    counts = np.array([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(b)])
    return counts, inverse

def mean_ci(frame, value, cluster='pitcher_id'):
    f=frame[[cluster,value]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(f)==0: return dict(mean=np.nan,low=np.nan,high=np.nan,events=0,pitchers=0)
    groups=f.groupby(cluster)[value].agg(['sum','count'])
    draws,_=cluster_draws(groups.index.to_numpy())
    boot=draws @ groups['sum'].to_numpy() / (draws @ groups['count'].to_numpy())
    low,high=np.quantile(boot,[.025,.975]) if len(groups)>1 else [np.nan,np.nan]
    return dict(mean=f[value].mean(),low=low,high=high,events=len(f),pitchers=len(groups))

def save_manifest(extra=None):
    manifest=dict(signature=signature(),config=CFG,original_config=ORIGINAL_CFG,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/injury_analysis').glob('*.py')},
        test_status='2024–2025 has already been examined; these are exploratory temporal comparisons, not new untouched confirmation.')
    if extra: manifest.update(extra)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
