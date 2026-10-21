"""Run/resume the additive UCL endpoint and dynamic-warning development stage."""
from .ucl_dynamic_common import snapshot
from .ucl_dynamic_endpoint import run_endpoint
from .ucl_dynamic_data import prepare_data
from .ucl_dynamic_models import run_models
from .ucl_dynamic_robustness import run_robustness
from .ucl_dynamic_trajectories import run_trajectories
from .ucl_dynamic_reporting import run_reporting

def main():
    snapshot()
    for name,fn in [('endpoint',run_endpoint),('landmarks',prepare_data),('models',run_models),
                    ('robustness',run_robustness),('trajectories',run_trajectories),('reporting',run_reporting)]:
        print('Stage:',name,flush=True);fn()
    print('Completed reports/ucl_dynamic/summary.md; source database preserved.',flush=True)

if __name__=='__main__':main()
