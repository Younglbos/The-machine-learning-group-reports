"""Use an existing interpreter while keeping task caches and temporary files in the project.

Example: python scripts/run_in_environment.py experiments/04_hsacc_reproduction/run.py --smoke
Select an interpreter with --python, MVC_PYTHON, or the local .cache/runtime.json.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',dest='interpreter')
    parser.add_argument('--show-runtime',action='store_true')
    parser.add_argument('python_arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    local=ROOT/'.cache/runtime.json'
    settings=json.loads(local.read_text(encoding='utf-8')) if local.exists() else {}
    selected=args.interpreter or os.environ.get('MVC_PYTHON') or settings.get('python')
    if not selected:
        parser.error('Choose your existing interpreter with --python or MVC_PYTHON; no environment is downloaded.')
    interpreter=Path(selected).expanduser().resolve()
    if not interpreter.is_file():
        parser.error(f'Interpreter not found: {interpreter}')
    child_env=os.environ.copy()
    child_env['PYTHONUTF8']='1'
    child_env['PYTHONIOENCODING']='utf-8'
    child_env['TQDM_DISABLE']='1'
    cache_locations={'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'pip',
        'TORCH_HOME':'torch','MPLCONFIGDIR':'matplotlib','CUDA_CACHE_PATH':'cuda',
        'HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg','NUMBA_CACHE_DIR':'numba'}
    for key,folder in cache_locations.items():
        directory=ROOT/'.cache'/folder
        directory.mkdir(parents=True,exist_ok=True)
        child_env[key]=str(directory)
    extra=ROOT/'.cache/python_deps'
    extra.mkdir(parents=True,exist_ok=True)
    child_env['PYTHONPATH']=os.pathsep.join([str(extra),str(ROOT),child_env.get('PYTHONPATH','')])
    if args.show_runtime:
        print(json.dumps({'python':str(interpreter),'temporary_directory':child_env['TEMP'],
            'pip_cache':child_env['PIP_CACHE_DIR'],'torch_cache':child_env['TORCH_HOME'],
            'project_extra_packages':str(extra)},ensure_ascii=False,indent=2))
    command=args.python_arguments
    if command[:1]==['--']: command=command[1:]
    if not command:
        if args.show_runtime: return 0
        parser.error('Supply a Python script or use -- before Python options such as -m.')
    print(f'Using existing Python: {interpreter}',flush=True)
    return subprocess.run([str(interpreter),*command],cwd=ROOT,env=child_env).returncode


if __name__=='__main__': raise SystemExit(main())
