"""Resume locked temporal replication without refitting or modifying original outputs."""
from .ucl_replication_data import freeze_replication,prepare,score_replication
from .ucl_replication_analysis import run_analysis
from .ucl_replication_operations import run_operations
from .ucl_replication_workload import run_workload
from .ucl_replication_reporting import run_reporting

def main():
    freeze_replication();prepare();score_replication();run_analysis();run_operations();run_workload();run_reporting()

if __name__=='__main__':main()
