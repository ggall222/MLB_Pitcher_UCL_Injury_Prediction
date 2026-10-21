"""Execute/resume the additive nonlinear UCL model-form comparison."""
from .ucl_nonlinear_common import snapshot
from .ucl_nonlinear_models import run_models
from .ucl_nonlinear_analysis import run_analysis
from .ucl_nonlinear_diagnostics import run_diagnostics
from .ucl_nonlinear_probes import run_probes
from .ucl_nonlinear_reporting import run_reporting

def main():
    snapshot()
    for name,fn in [('models',run_models),('uncertainty and effects',run_analysis),('influence diagnostics',run_diagnostics),
                    ('profile/workload interpretation',run_probes),('reports and validation',run_reporting)]:
        print('Stage:',name,flush=True);fn()
    print('Completed reports/ucl_nonlinear/summary.md',flush=True)

if __name__=='__main__':main()
