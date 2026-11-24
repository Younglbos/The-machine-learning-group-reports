"""Run pinned DSVC view-stream code, recording every view and final predictions.

Each invocation is a separate process to isolate upstream generic import names.
Default is the first three published-code seeds and the original 200 epochs/view.
"""

import argparse
import itertools
import os
from pathlib import Path
import random
from types import SimpleNamespace

os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch

from evaluation.metrics import scores, clustering_accuracy
from evaluation.records import Measurement, aggregate, environment, file_hash, logger_at, write_csv, write_json
from models.author_import import author_path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='HandWritten')
    parser.add_argument('--seeds',default='11,21,22')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--output',default='results/dsvc_author')
    args=parser.parse_args()
    author=author_path('DSVC')
    import data_loader
    from Knowledge_Generation import dual_attention
    from network import Autoencoder
    import Streaming_train
    # At 200 epochs this is identical; shorter smoke runs evaluate their actual final epoch.
    if args.epochs!=200:
        source=(author/'Streaming_train.py').read_text(encoding='utf-8')
        source=source.replace('if epoch+1 == 1 or (epoch + 1) % 200 == 0:',
                              'if epoch+1 == 1 or (epoch + 1) % 200 == 0 or epoch+1 == config.epochs:')
        exec(compile(source,str(author/'Streaming_train.py'),'exec'),Streaming_train.__dict__)
    if not torch.cuda.is_available():
        raise RuntimeError('Upstream DSVC uses .cuda() directly; verify CUDA before running')
    data_loader.path=str(author/'Datasets')+os.sep
    spec=data_loader.ALL_data[args.dataset]
    config=SimpleNamespace(datasets=spec,feature_dim=256,Cluster=spec['K'],view_num=spec['V'],
        Knowledge_size=spec['Knowledge_size'],lr=1e-4,beta1=spec['beta1'],beta2=spec['beta2'],
        batch_size=256,alpha=0.5,epochs=args.epochs,cuda_list=[0],running_time=False)
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    log=logger_at(out)
    write_json(out/'config.json',{**vars(config),'seeds':args.seeds,'run_type':'full_length' if args.epochs==200 else 'smoke',
        'protocol':'upstream_view_stream','note':'Retains author prototype buffers and per-sample historical embeddings; not a sample-stream experiment.'})
    write_json(out/'environment.json',environment())
    write_json(out/'source_hashes.json',{str(p.relative_to(ROOT)):file_hash(p) for p in author.glob('*.py')})
    views,labels=data_loader.load_data2(spec)
    truth=labels[0]
    rows=[]
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in map(int,args.seeds.split(',')):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.enabled=False
            buffer=[None,None]
            features=[]
            previous=None
            cumulative=0.0
            folder=out/f'seed_{seed}'
            folder.mkdir(exist_ok=True)
            for v,raw in enumerate(views):
                log.info('DSVC seed=%d view=%d/%d input_dim=%d epochs=%d',seed,v+1,len(views),raw.shape[1],args.epochs)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                with Measurement() as timing:
                    attention=dual_attention(dim=256,num=spec['Knowledge_size']).cuda()
                    decoder=Autoencoder(raw.shape[1],256,activation='relu',batchnorm=True).cuda()
                    optimizer=torch.optim.Adam(itertools.chain(decoder.parameters(),attention.parameters()),lr=1e-4)
                    trainer=Streaming_train.train_view_0 if v==0 else Streaming_train.train_view
                    embedded,buffer=trainer(decoder,attention,optimizer,raw,labels,v,str(folder)+os.sep,'unused.csv',
                        config,spec['K'],buffer,[spec['beta1'],spec['beta2']])
                    features.append(embedded.detach().cpu().numpy())
                    combined=np.concatenate(features,axis=1)
                    # Upstream uses KMeans defaults; preserve the installed-version default and record it.
                    estimator=KMeans(n_clusters=spec['K']).fit(combined)
                    pred=estimator.labels_
                    torch.cuda.synchronize()
                cumulative+=timing.seconds
                row={'dataset':args.dataset,'protocol':'upstream_view_stream','run_type':'full_length' if args.epochs==200 else 'smoke',
                    'seed':seed,'view':v+1,'epochs_per_view':args.epochs,**scores(truth,pred),
                    'view_update_with_periodic_evaluation_s':timing.seconds,'cumulative_s':cumulative,
                    'peak_rss_mb':timing.peak_rss_mb,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
                    'historical_embedding_mb':sum(x.nbytes for x in features)/1024**2,
                    'prototype_buffer_mb':sum(x.nelement()*x.element_size() for x in buffer)/1024**2,
                    'label_change':None if previous is None else 1-clustering_accuracy(previous,pred),
                    'kmeans_n_init':str(estimator.n_init)}
                previous=pred.copy()
                rows.append(row)
                np.savez_compressed(folder/f'view_{v+1}.npz',predicted=pred,truth=truth,
                    new_view_embedding=features[-1],sample_indices=np.arange(len(truth)))
                write_csv(out/'runs.csv',rows)
                log.info('DSVC view=%d ACC=%.4f NMI=%.4f ARI=%.4f update=%.2fs',v+1,row['acc'],row['nmi'],row['ari'],timing.seconds)
    write_csv(out/'summary.csv',aggregate(rows,['dataset','protocol','view','epochs_per_view'],
        ['acc','nmi','ari','view_update_with_periodic_evaluation_s','cumulative_s','peak_rss_mb','peak_gpu_allocated_mb','label_change']))


if __name__=='__main__': main()
