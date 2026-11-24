"""Matched-environment, matched-mask K-means comparisons for the course HSACC runs."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from data.load_dataset import load_dataset
from data.preprocessing import dense_views,prepare_observed
from data.missing import make_mask
from data.course_protocols import mask_hash,noisy_observations
from evaluation.metrics import scores,silhouette
from evaluation.records import Measurement,aggregate,environment,logger_at,write_csv,write_json
from models.kmeans import fit_kmeans

ROOT=Path(__file__).resolve().parents[1]
OPTIONS={'n_init':10,'max_iter':300,'tol':1e-4}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/06_course_comparison/config_a.json')
    parser.add_argument('--output',default='experiments/06_course_comparison/results_a/baselines')
    parser.add_argument('--noise-std',type=float,default=0.0)
    parser.add_argument('--rates',default=None)
    parser.add_argument('--patterns',default=None)
    parser.add_argument('--seeds',default=None)
    parser.add_argument('--methods',default='zero,mean')
    parser.add_argument('--references',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if args.rates: config['rates']=list(map(float,args.rates.split(',')))
    if args.patterns: config['patterns']=args.patterns.split(',')
    if args.seeds: config['seeds']=list(map(int,args.seeds.split(',')))
    out=ROOT/args.output
    (out/'labels').mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',{**config,'noise_std':args.noise_std,'kmeans':OPTIONS})
    write_json(out/'environment.json',environment())
    log=logger_at(out)
    raw,truth=load_dataset(config['dataset'])
    raw=dense_views(raw)
    views=[raw[j] for j in config['view_indices']]
    rows=[]
    with threadpool_limits(limits=1):
        for pattern in config['patterns']:
            for rate in config['rates']:
                for seed in config['seeds']:
                    mask=make_mask(len(truth),len(views),rate,seed,pattern,truth)
                    current=noisy_observations(views,mask,seed,args.noise_std)
                    for fill in args.methods.split(','):
                        with Measurement() as measured:
                            start=time.perf_counter()
                            blocks,_=prepare_observed(current,mask,fill)
                            features=np.concatenate(blocks,axis=1)
                            preprocessing=time.perf_counter()-start
                            model,pred=fit_kmeans(features,len(np.unique(truth)),seed,OPTIONS)
                        row={'dataset':config['dataset'],'method':f'{fill}_fill_kmeans','pattern':pattern,
                            'rate_parameter':rate,'seed':seed,'noise_std':args.noise_std,'mask_sha256':mask_hash(mask),
                            'actual_block_rate':float((~mask).mean()),'actual_incomplete_rate':float((~mask.all(axis=1)).mean()),
                            **scores(truth,pred),'total_algorithm_s':measured.seconds,'preprocess_s':preprocessing,
                            'fit_predict_s':measured.seconds-preprocessing,'peak_rss_mb':measured.peak_rss_mb,
                            'process_peak_rss_mb':measured.process_lifetime_peak_mb,
                            'silhouette':silhouette(features,pred,seed),'iterations':model.n_iter_}
                        rows.append(row)
                        np.savez_compressed(out/'labels'/f'{pattern}_{rate:.2f}_{seed}_{fill}.npz',
                            predicted=pred,truth=truth,mask=mask,sample_indices=np.arange(len(truth)))
                        write_csv(out/'runs.csv',rows)
                        log.info('%s rate=%.2f seed=%d %s ACC=%.4f',pattern,rate,seed,fill,row['acc'])
    write_csv(out/'summary.csv',aggregate(rows,['dataset','method','pattern','rate_parameter','noise_std'],
        ['acc','nmi','ari','total_algorithm_s','peak_rss_mb','silhouette']))
    if args.references:
        references=[]
        complete_mask=np.ones((len(truth),len(views)),dtype=bool)
        with threadpool_limits(limits=1):
            blocks,_=prepare_observed(views,complete_mask,'zero')
            methods=[('concat_complete',np.concatenate(blocks,axis=1))]+[(f'single_complete_v{j+1}',v) for j,v in enumerate(blocks)]
            for method,features in methods:
                for seed in config['seeds']:
                    with Measurement() as measured:
                        _,pred=fit_kmeans(features,len(np.unique(truth)),seed,OPTIONS)
                    row={'method':method,'seed':seed,**scores(truth,pred),
                         'reference_uses_complete_data':True,'fit_predict_s':measured.seconds}
                    references.append(row)
                    np.savez_compressed(out/'labels'/f'reference_{method}_{seed}.npz',predicted=pred,
                        truth=truth,sample_indices=np.arange(len(truth)))
        write_csv(out/'complete_reference_runs.csv',references)
        write_csv(out/'complete_reference_summary.csv',aggregate(references,['method'],['acc','nmi','ari','fit_predict_s']))


if __name__=='__main__': main()
