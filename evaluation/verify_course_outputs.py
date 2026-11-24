"""Recompute course metrics from saved labels and check experimental provenance."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.metrics import scores
from evaluation.records import file_hash, write_json
from data.course_protocols import mask_hash

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-only',action='store_true',help='Verify compact labels without raw datasets or local outputs')
    args=parser.parse_args()
    if args.public_only:
        manifest=json.loads((ROOT/'reports/prediction_verification.json').read_text(encoding='utf-8'))
        with gzip.open(ROOT/'reports/course_predictions.jsonl.gz','rt',encoding='utf-8') as source:
            rows=[json.loads(line) for line in source]
        if len(rows)!=manifest['verified_records']: raise ValueError('Public archive record count mismatch')
        for row,expected in zip(rows,manifest['checks']):
            if row['source']!=expected['path']: raise ValueError('Public archive ordering differs')
            for metric,value in scores(row['truth'],row['predicted']).items():
                if not np.isclose(value,expected[metric],rtol=0,atol=1e-12): raise ValueError('Public labels mismatch')
        print(f'Verified {len(rows)} public label records; no datasets, neural models or checkpoints required.')
        return
    verified=[]
    predictions=[]
    def check(row,path,probe=False):
        with np.load(path) as stored:
            truth,pred=stored['truth'],stored['predicted']
            if not np.array_equal(stored['sample_indices'],np.arange(len(truth))): raise ValueError(f'Object order {path}')
            recomputed=scores(truth,pred)
            for key,value in recomputed.items():
                if not np.isclose(float(row[key]),value,rtol=0,atol=1e-12): raise ValueError(f'{path}: {key}')
            if 'mask_sha256' in row and mask_hash(stored['mask'])!=row['mask_sha256']: raise ValueError('Mask hash mismatch')
            if probe:
                ids=stored['probe_indices']
                if not np.isclose(scores(truth[ids],stored['probe_predicted'])['acc'],row['probe_acc'],atol=1e-12):
                    raise ValueError('Probe prediction mismatch')
            identity={key:row[key] for key in ['method','seed','pattern','rate_parameter','condition','view','variant','noise_std'] if key in row}
            # Compact public labels only; latent features, checkpoints and source data stay local.
            predictions.append({'record':len(verified)+1,**identity,'source':str(path.relative_to(ROOT)),
                'truth':truth.astype(int).tolist(),'predicted':pred.astype(int).tolist()})
        verified.append({'path':str(path.relative_to(ROOT)),'sha256':file_hash(path),**recomputed})
    def baselines(folder,references=False):
        for row in pd.read_csv(folder/'runs.csv').to_dict('records'):
            fill=row['method'].split('_')[0]
            check(row,folder/'labels'/f"{row['pattern']}_{row['rate_parameter']:.2f}_{row['seed']}_{fill}.npz")
        if references:
            for row in pd.read_csv(folder/'complete_reference_runs.csv').to_dict('records'):
                check(row,folder/'labels'/f"reference_{row['method']}_{row['seed']}.npz")
    def learned(folder):
        for row in pd.read_csv(folder/'runs.csv').to_dict('records'):
            check(row,folder/f"{row['pattern']}_{row['rate_parameter']:.2f}_{row['seed']}_{row['variant']}"/'outputs.npz')
    base=ROOT/'experiments/06_course_comparison'
    baselines(base/'results_a/baselines',True)
    learned(base/'results_a/hsacc')
    for row in pd.read_csv(base/'results_b/runs.csv').to_dict('records'):
        check(row,base/'results_b'/f"{row['condition']}_{row['seed']}_{row['method']}"/f"view_{row['view']}.npz",True)
    extra=ROOT/'experiments/07_ablations_robustness/results'
    for variant in ['no_consistency','no_completion','equal_weights','noise_hsacc']: learned(extra/variant)
    baselines(extra/'noise_baselines')
    if len(verified)!=252: raise ValueError(f'Expected 252 matched course prediction records, found {len(verified)}')
    write_json(ROOT/'reports/prediction_verification.json',{'verified_records':len(verified),'tolerance':1e-12,
        'all_metrics_recomputed':True,'checks':verified})
    # A single compressed label archive permits third parties to recompute published quality values.
    with (ROOT/'reports/course_predictions.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as archive:
            for record in predictions:
                archive.write((json.dumps(record,separators=(',',':'),ensure_ascii=False)+'\n').encode('utf-8'))
    print(f'Verified {len(verified)} prediction records against saved arrays; public labels exported.')


if __name__=='__main__': main()
