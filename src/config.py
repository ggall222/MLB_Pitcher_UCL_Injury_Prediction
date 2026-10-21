from pathlib import Path
from datetime import date, timedelta
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    with open(path or ROOT / 'config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['start_date'] = date.fromisoformat(str(cfg.get('statcast_start_date', cfg.get('start_date'))))
    end = cfg.get('statcast_end_date', cfg.get('end_date'))
    cfg['end_date'] = date.today() - timedelta(days=1) if end in (None, 'latest') else date.fromisoformat(str(end))
    for key in ('database_path', 'cache_path', 'export_path', 'report_path'):
        cfg[key] = ROOT / cfg[key]
    return cfg
