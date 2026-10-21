"""Execute an injury-analysis notebook suite, preserving outputs and source fingerprints."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.injury_analysis.common import OUT, source_signature


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-notebook',type=int,default=1,choices=range(1,9))
    parser.add_argument('--suite',choices=['injury-types','injury-signals','injury-reduction','injury-profiles','injury-supervised','ucl-visuals','feature-consolidation','ucl-dynamic','ucl-nonlinear','ucl-temporal'],default='injury-types')
    args=parser.parse_args()
    if args.suite!='injury-types' and args.from_notebook>3:
        parser.error('The extension suites each have three notebooks.')
    if args.suite=='injury-profiles' and args.from_notebook>2:
        parser.error('The profile/departure suite has two notebooks.')
    if args.suite=='ucl-visuals' and args.from_notebook!=1:
        parser.error('The focused UCL visual suite has one notebook.')
    output=OUT
    notebook_directory=ROOT/'notebooks/injury_type_analysis'
    if args.suite=='injury-signals':
        from src.injury_analysis.signals_common import OUT as signal_output
        output=signal_output
        notebook_directory=ROOT/'notebooks/injury_signal_analysis'
    elif args.suite=='injury-reduction':
        from src.injury_analysis.reduction import OUT as reduction_output
        output=reduction_output
        notebook_directory=ROOT/'notebooks/injury_dimensionality_reduction'
    elif args.suite=='injury-profiles':
        from src.injury_analysis.profile_departure import OUT as profile_output
        output=profile_output
        notebook_directory=ROOT/'notebooks/injury_profile_departure'
    elif args.suite=='injury-supervised':
        from src.injury_analysis.supervised_reduction import OUT as supervised_output
        output=supervised_output
        notebook_directory=ROOT/'notebooks/injury_supervised_reduction'
    elif args.suite=='ucl-visuals':
        from src.injury_analysis.ucl_latent_visuals import OUT as ucl_output
        output=ucl_output
        notebook_directory=ROOT/'notebooks/ucl_latent_visuals'
    elif args.suite=='feature-consolidation':
        from src.injury_analysis.consolidation_common import OUT as consolidation_output
        output=consolidation_output
        notebook_directory=ROOT/'notebooks/feature_consolidation'
    elif args.suite=='ucl-dynamic':
        from src.injury_analysis.ucl_dynamic_common import OUT as dynamic_output
        output=dynamic_output
        notebook_directory=ROOT/'notebooks/ucl_dynamic'
    elif args.suite=='ucl-nonlinear':
        from src.injury_analysis.ucl_nonlinear_common import OUT as nonlinear_output
        output=nonlinear_output
        notebook_directory=ROOT/'notebooks/ucl_nonlinear'
    elif args.suite=='ucl-temporal':
        from src.injury_analysis.ucl_temporal_common import OUT as temporal_output
        output=temporal_output
        notebook_directory=ROOT/'notebooks/ucl_temporal'
    # Install the kernel only inside this project's Python environment.
    subprocess.run([sys.executable,'-m','ipykernel','install','--prefix',sys.prefix,
        '--name','mlb-ucl-analysis','--display-name','MLB UCL analysis (.venv)'],check=True)
    os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
    before=source_signature()
    path=output/'notebook_execution.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    records=[r for r in previous.get('notebooks',[]) if int(r['name'][:2])<args.from_notebook]
    manifest={'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'python':sys.version,
        'executable':sys.executable,'source_signature':before,'notebooks':records,
        'package_versions':{p:importlib.metadata.version(p) for p in ['duckdb','pandas','numpy','scikit-learn',
            'scipy','matplotlib','nbformat','nbclient','ipykernel']}}
    for notebook in sorted(notebook_directory.glob('*.ipynb')):
        if int(notebook.name[:2])<args.from_notebook:
            continue
        print(f'EXECUTING {notebook.name}',flush=True)
        nb=nbformat.read(notebook,as_version=4)
        start=time.monotonic()
        record={'name':notebook.name,'success':False}
        try:
            NotebookClient(nb,timeout=1800,kernel_name='mlb-ucl-analysis',
                resources={'metadata':{'path':str(ROOT)}}).execute()
            record['success']=True
        finally:
            nbformat.write(nb,notebook)
            record.update(seconds=round(time.monotonic()-start,2),
                code_cells=sum(c.cell_type=='code' for c in nb.cells),
                executed_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
                errors=sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),
                sha256=hashlib.sha256(notebook.read_bytes()).hexdigest())
            records.append(record)
            manifest['notebooks']=records
            manifest['database_unchanged']=source_signature()==before
            manifest['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
            manifest['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'src/injury_analysis').glob('*.py'))}
            path.write_text(json.dumps(manifest,indent=2))
            print(json.dumps(record),flush=True)
    assert source_signature()==before,'Source changed during execution'
    print(f'Executed {len(records)} notebooks; sources unchanged. Manifest: {path}',flush=True)


if __name__=='__main__':
    main()
