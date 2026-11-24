"""One process per method/seed/condition for a matched six-view course comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits
from data.preprocessing import dense_views
from data.course_protocols import noisy_stream_view
from evaluation.metrics import scores,clustering_accuracy
from evaluation.records import Measurement,environment,logger_at,write_json,write_csv
from models.view_stream import ViewBaseline,DSVCCourse

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/06_course_comparison/config_b.json')
    parser.add_argument('--method',required=True,choices=['warm_start','sliding_window','full_retrain','dsvc_course'])
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--condition',choices=['none','view_noise'],required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if args.smoke: config['epochs']=2
    out=ROOT/config['result_dir']/f'{args.condition}_{args.seed}_{args.method}'
    if args.smoke: out=ROOT/'.cache/course_view_smoke'/f'{args.condition}_{args.seed}_{args.method}'
    out.mkdir(parents=True,exist_ok=True)
    log=logger_at(out)
    write_json(out/'config.json',{**config,**vars(args)})
    write_json(out/'environment.json',environment())
    raw=loadmat(ROOT/'third_party/DSVC/Datasets/HandWritten.mat',variable_names=['X','Y'])
    views=dense_views(raw['X'].ravel())
    truth=raw['Y'].ravel()
    if len(views)!=6 or len(np.unique(truth))!=config['clusters']: raise ValueError('Dataset schema changed')
    if args.method=='dsvc_course': model=DSVCCourse(config,args.seed)
    else: model=ViewBaseline(args.method,config['clusters'],args.seed,config['kmeans'],config['components'],config['window'])
    probe_ids=np.random.default_rng(args.seed).choice(len(truth),min(config['probe_samples'],len(truth)),replace=False)
    rows=[]
    previous_pred=None
    cumulative=0.0
    with threadpool_limits(limits=1):
        for v,clean in enumerate(views,start=1):
            current=noisy_stream_view(clean,args.seed,v,config['drift_view'],args.condition,config['noise_std'])
            digest=hashlib.sha256(current.tobytes()).hexdigest()
            if args.method=='dsvc_course':
                model.torch.cuda.reset_peak_memory_stats()
                model.torch.cuda.synchronize()
            with Measurement() as measured:
                scaled=MinMaxScaler().fit_transform(current)
                pred,representation=model.update(scaled)
                if args.method=='dsvc_course': model.torch.cuda.synchronize()
            cumulative+=measured.seconds
            if v==1:
                probe=representation[probe_ids].copy()
                initial_probe_acc=scores(truth[probe_ids],model.predict(probe))['acc']
            retained=model.predict(probe)
            probe_acc=scores(truth[probe_ids],retained)['acc']
            row={'method':args.method,'seed':args.seed,'condition':args.condition,'view':v,
                'raw_view_sha256':digest,**scores(truth,pred),'update_s':measured.seconds,
                'cumulative_s':cumulative,'peak_rss_mb':measured.peak_rss_mb,
                'process_peak_rss_mb':measured.process_lifetime_peak_mb,
                'peak_gpu_mb':model.torch.cuda.max_memory_allocated()/1024**2 if args.method=='dsvc_course' else 0,
                'retained_state_mb':model.state_bytes/1024**2,'budget_s':config['time_budget_s'],
                'budget_exceeded':measured.seconds>config['time_budget_s'],'probe_acc':probe_acc,
                'forgetting':initial_probe_acc-probe_acc,
                'label_change':None if previous_pred is None else 1-clustering_accuracy(previous_pred,pred),
                'epochs':config['epochs'] if args.method=='dsvc_course' else 0}
            previous_pred=pred.copy()
            rows.append(row)
            np.savez_compressed(out/f'view_{v}.npz',predicted=pred,truth=truth,sample_indices=np.arange(len(truth)),
                probe_indices=probe_ids,probe_predicted=retained)
            write_csv(out/'runs.csv',rows)
            log.info('%s %s seed=%d view=%d ACC=%.4f elapsed=%.2fs',args.method,args.condition,args.seed,v,row['acc'],measured.seconds)
    write_json(out/'completed.json',{'complete':True,'records':len(rows),'config':config,'smoke':args.smoke})
    if args.method=='dsvc_course': write_json(out/'loss_trace.json',model.loss_trace)


if __name__=='__main__': main()
