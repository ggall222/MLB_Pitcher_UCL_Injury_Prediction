"""Run or resume the fixed first-pass temporal signal decomposition."""
from .ucl_temporal_models import run_development,run_followup_sensitivity,run_transfer
from .ucl_temporal_analysis import analyze_partition,run_refits,finalize_development
from .ucl_temporal_diagnostics import run_composition
from .ucl_temporal_trajectories import run_trajectories
from .ucl_temporal_reporting import run_reporting

def main():
    run_development()
    analyze_partition('validation')
    run_followup_sensitivity()
    run_refits()
    run_composition('validation')
    finalize_development()
    run_transfer()
    analyze_partition('test')
    run_composition('test')
    run_trajectories()
    run_reporting()

if __name__=='__main__':
    main()
