"""Continue the frozen course experiment plan sequentially, with resumable cases."""

import json
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
from evaluation.records import aggregate,write_csv,write_json,logger_at

ROOT=Path(__file__).resolve().parents[1]


def main():
    log=logger_at(ROOT/'experiments/06_course_comparison')
    status=ROOT/'experiments/06_course_comparison/progress.json'
    def phase(name):
        write_json(status,{'phase':name,'state':'running'})
        log.info('PHASE %s',name)
    def run(module,arguments,log_name):
        path=ROOT/'.cache/suite_logs'/log_name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8') as stream:
            process=subprocess.run([sys.executable,'-m',module,*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode: raise RuntimeError(f'Failed {module}; inspect {path}')
    phase('matched_A_running')
    deadline=time.monotonic()+3600
    while not (ROOT/'experiments/06_course_comparison/results_a/hsacc/summary.csv').exists():
        if time.monotonic()>deadline: raise TimeoutError('A comparison did not complete within one hour')
        time.sleep(5)
    a=pd.read_csv(ROOT/'experiments/06_course_comparison/results_a/hsacc/runs.csv')
    if len(a)!=27: raise ValueError('Incomplete A matrix')
    phase('matched_B_running')
    run('train_or_update.run_view_suite',[],'course_b.log')
    phase('ablations_running')
    settings=json.loads((ROOT/'experiments/07_ablations_robustness/config.json').read_text(encoding='utf-8'))
    base=settings['result_dir']
    shared=['--protocol','course','--seeds','1,2,3','--rates',str(settings['rate']),
            '--patterns',settings['pattern'],'--epochs',str(settings['epochs']),'--resume']
    for variant in settings['variants']:
        log.info('Ablation %s',variant)
        run('train_or_update.run_hsacc',shared+['--variant',variant,'--output',f'{base}/{variant}'],f'{variant}.log')
    phase('robustness_running')
    run('train_or_update.run_hsacc',shared+['--noise-std',str(settings['noise_std']),
        '--output',f'{base}/noise_hsacc'],'noise_hsacc.log')
    run('train_or_update.course_a_baselines',['--noise-std',str(settings['noise_std']),
        '--rates',str(settings['rate']),'--patterns',settings['pattern'],'--output',f'{base}/noise_baselines'],'noise_baselines.log')
    phase('isolated_A_efficiency_running')
    efficiency=[]
    for seed in [1,2,3]:
        for method in ['zero','mean','hsacc']:
            output=f'.cache/isolated_a/{method}_{seed}'
            log.info('Isolated efficiency %s seed=%d',method,seed)
            if method=='hsacc':
                run('train_or_update.run_hsacc',['--protocol','course','--seeds',str(seed),'--rates','0.4',
                    '--patterns','random','--epochs','500','--output',output,'--resume'],f'isolated_{method}_{seed}.log')
                row=pd.read_csv(ROOT/output/'runs.csv').iloc[0]
                elapsed=float(row['preprocess_s']+row['train_with_periodic_evaluation_s']+row['final_clustering_s'])
                gpu=float(row['peak_gpu_allocated_mb'])
            else:
                run('train_or_update.course_a_baselines',['--seeds',str(seed),'--rates','0.4','--patterns','random',
                    '--methods',method,'--output',output],f'isolated_{method}_{seed}.log')
                row=pd.read_csv(ROOT/output/'runs.csv').iloc[0]
                elapsed=float(row['total_algorithm_s'])
                gpu=0.0
            efficiency.append({'method':method,'seed':seed,'dataset':'Caltech101-20 HOG/GIST',
                'rate':0.4,'pattern':'random','elapsed_s':elapsed,'process_peak_rss_mb':float(row['process_peak_rss_mb']),
                'peak_gpu_mb':gpu,'acc':float(row['acc']),'timing_note':'HSACC includes periodic upstream evaluation; each case uses a fresh process.'})
            write_csv(ROOT/'experiments/07_ablations_robustness/results/isolated_efficiency.csv',efficiency)
    write_csv(ROOT/'experiments/07_ablations_robustness/results/isolated_efficiency_summary.csv',
        aggregate(efficiency,['method'],['elapsed_s','process_peak_rss_mb','peak_gpu_mb','acc']))
    write_json(status,{'phase':'experiments_complete','state':'complete','reporting_pending':True})
    log.info('All frozen comparison, ablation, robustness and isolated efficiency cases completed.')


if __name__=='__main__': main()
