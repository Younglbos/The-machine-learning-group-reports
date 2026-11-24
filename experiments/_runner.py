"""Run experiment modules from the repository root with the caller's interpreter."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run_module(module,arguments):
    completed=subprocess.run([sys.executable,'-m',module,*map(str,arguments)],cwd=ROOT)
    if completed.returncode: raise SystemExit(completed.returncode)


def baseline(folder,stage):
    folder=Path(folder)
    run_module('train_or_update.run_baselines',['--config',folder/'config.json','--stage',stage])
    if stage=='a':
        run_module('train_or_update.run_baselines',['--config',folder/'config.json','--stage','a_reference'])
    run_module('visualization.plot_baselines',['--config',folder/'config.json','--stage',stage])


def paper(folder,module):
    folder=Path(folder)
    parser=argparse.ArgumentParser(description='Run this experiment, optionally overriding its budget for a smoke test.')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if args.smoke:
        config['epochs']=2 if module.endswith('run_dsvc') else 5
        config['seeds']='11' if module.endswith('run_dsvc') else '1'
        config['output']=str(folder.relative_to(ROOT)/'smoke_results')
    command=[]
    for key,value in config.items(): command.extend(['--'+key.replace('_','-'),str(value)])
    run_module(module,command)
