"""Shared paths, reproducibility metadata, tables, and publication-size figures."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'config/injury_analysis.yaml'
CFG = yaml.safe_load(CONFIG_PATH.read_text())
OUT = ROOT / CFG['output_path']
CACHE = ROOT / CFG['cache_path']
for folder in (OUT, OUT / 'tables', OUT / 'figures', CACHE):
    folder.mkdir(parents=True, exist_ok=True)

GROUPS = {
    'Elbow / UCL': 'Elbow / UCL', 'Shoulder': 'Shoulder', 'Forearm': 'Forearm',
    'Upper Arm': 'Other arm', 'Lat': 'Other arm',
    'Back / Spine': 'Trunk / neck', 'Core / Oblique / Abdomen': 'Trunk / neck', 'Neck': 'Trunk / neck',
    'Hip / Groin': 'Lower body', 'Hamstring': 'Lower body', 'Knee': 'Lower body',
    'Lower Leg': 'Lower body', 'Foot / Ankle': 'Lower body',
    'Hand / Wrist / Finger': 'Hand / wrist', 'Head / Concussion': 'Other',
    'Illness / non-musculoskeletal': 'Illness', 'Other': 'Other', 'Unknown / Unspecified': 'Unknown',
}
ARM = ['Elbow / UCL', 'Shoulder', 'Forearm', 'Other arm']
NONARM = ['Trunk / neck', 'Lower body']
CATEGORIES = ['Elbow / UCL', 'Shoulder', 'Forearm', 'Other arm', 'Trunk / neck', 'Lower body', 'Hand / wrist']
TRAJECTORY_FEATURES = ['velo_z', 'fb_velo_change_vs_prev_5', 'release_x_sd_z', 'release_z_sd_z',
    'extension_z', 'spin_z', 'zone_pct_z', 'workload_acceleration_7d_vs_21d',
    'acute_chronic_pitch_ratio', 'arm_angle_z']
LABELS = {'velo_z': 'Fastball velocity (pitcher z-score)',
    'fb_velo_change_vs_prev_5': 'Fastball change vs prior 5 (mph)',
    'release_x_sd_z': 'Lateral release variability (z-score)',
    'release_z_sd_z': 'Vertical release variability (z-score)',
    'extension_z': 'Extension (pitcher z-score)', 'spin_z': 'Spin (pitcher z-score)',
    'zone_pct_z': 'Zone percentage (pitcher z-score)',
    'workload_acceleration_7d_vs_21d': 'Recent workload change (pitches/day)',
    'acute_chronic_pitch_ratio': '7-day / 28-day pitch-rate ratio', 'arm_angle_z': 'Arm angle (pitcher z-score)'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_signature():
    path = ROOT / CFG['database_path']
    stat = path.stat()
    return {'database': str(path), 'database_bytes': stat.st_size, 'database_mtime_ns': stat.st_mtime_ns,
        'events_sha256': sha(ROOT / CFG['events_path']), 'config_sha256': sha(CONFIG_PATH),
        'feature_config_sha256': sha(ROOT / 'config/model_features_v1.yaml')}

def table(frame, name):
    frame.to_csv(OUT / 'tables' / f'{name}.csv', index=False)
    return frame

def read_table(name):
    return pd.read_csv(OUT / 'tables' / f'{name}.csv')

def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 220, 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': .15,
        'figure.constrained_layout.use': True, 'pdf.fonttype': 42})

def figure(fig, name):
    fig.savefig(OUT / 'figures' / f'{name}.png', bbox_inches='tight')
    fig.savefig(OUT / 'figures' / f'{name}.pdf', bbox_inches='tight')
    return fig

def cluster_mean_ci(frame, value, cluster='pitcher_id', seed=None):
    """Event-weighted mean, resampling whole pitchers (not individual appearances)."""
    f = frame[[cluster, value]].replace([np.inf, -np.inf], np.nan).dropna()
    if f.empty:
        return {'mean': np.nan, 'low': np.nan, 'high': np.nan, 'n': 0, 'pitchers': 0}
    sums = f.groupby(cluster)[value].agg(['sum', 'count'])
    rng = np.random.default_rng(CFG['seed'] if seed is None else seed)
    idx = rng.integers(0, len(sums), (CFG['bootstrap_replicates'], len(sums)))
    vals = sums['sum'].to_numpy()[idx].sum(axis=1) / sums['count'].to_numpy()[idx].sum(axis=1)
    lo, hi = np.quantile(vals, [.025, .975]) if len(sums) > 1 else [np.nan, np.nan]
    return {'mean': f[value].mean(), 'low': lo, 'high': hi, 'n': len(f), 'pitchers': len(sums)}

def conclusion(text):
    from IPython.display import Markdown, display
    display(Markdown('**Computed conclusion.** ' + text))
