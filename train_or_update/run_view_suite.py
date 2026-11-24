"""Run each course view-stream learner in its own process, with restartable records."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from evaluation.records import aggregate,write_csv,write_json,logger_at

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/06_course_comparison/config_b.json')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    out=ROOT/config['result_dir']
    out.mkdir(parents=True,exist_ok=True)
    log=logger_at(out)
    write_json(out/'frozen_protocol.json',config)
    rows=[]
    for condition in config['conditions']:
        for seed in config['seeds']:
            for method in config['methods']:
                folder=out/f'{condition}_{seed}_{method}'
                finished=folder/'completed.json'
                if finished.exists():
                    saved=json.loads(finished.read_text(encoding='utf-8'))
                    if saved['config']!=config or saved['smoke']: raise ValueError('Resume protocol mismatch')
                    log.info('Reuse completed %s',folder.name)
                else:
                    log.info('Run %s',folder.name)
                    command=[sys.executable,'-m','train_or_update.course_view_stream','--config',args.config,
                             '--method',method,'--seed',str(seed),'--condition',condition]
                    with (out/f'{condition}_{seed}_{method}.log').open('w',encoding='utf-8') as stream:
                        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode: raise RuntimeError(f'Failed {folder.name}; see its log')
                rows.extend(pd.read_csv(folder/'runs.csv').where(lambda frame: frame.notna(),None).to_dict('records'))
                write_csv(out/'runs.csv',rows)
    frame=pd.DataFrame(rows)
    # Verify exact same corrupted observations were supplied to all four methods.
    if not (frame.groupby(['condition','seed','view']).raw_view_sha256.nunique()==1).all():
        raise ValueError('Unmatched view inputs')
    summaries=aggregate(rows,['method','condition','view'],
        ['acc','nmi','ari','update_s','cumulative_s','peak_rss_mb','process_peak_rss_mb','peak_gpu_mb','retained_state_mb','probe_acc','forgetting','label_change'])
    # pandas represents missing first-view changes as NaN; use JSON only for finite aggregate fields.
    write_csv(out/'summary.csv',summaries)
    recovery=[]
    for (method,condition,seed), group in frame.groupby(['method','condition','seed']):
        group=group.sort_values('view')
        pre=float(group.loc[group.view==config['drift_view']-1,'acc'].iloc[0])
        post=group[group.view>=config['drift_view']]
        below=post[post.acc<0.9*pre]
        recovered=None
        batches=None
        if condition!='none':
            if below.empty: recovered,batches=True,0
            else:
                candidates=post[(post.view>below.iloc[0]['view'])&(post.acc>=0.9*pre)]
                recovered=not candidates.empty
                if recovered: batches=int(candidates.iloc[0]['view']-config['drift_view'])
        recovery.append({'method':method,'condition':condition,'seed':int(seed),'pre_drift_acc':pre,
            'threshold':0.9*pre,'drop':pre-float(post.acc.min()),'recovered':recovered,'recovery_batches':batches})
    write_csv(out/'recovery.csv',recovery)
    write_json(out/'verification.json',{'records':len(rows),'paired_inputs_match':True,
        'isolated_process_per_case':True,'budget_exceedances':int(frame.budget_exceeded.sum()),
        'expected_records':len(config['seeds'])*len(config['conditions'])*len(config['methods'])*6})


if __name__=='__main__': main()
