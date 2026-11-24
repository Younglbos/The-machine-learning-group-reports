"""HSACC author-code reproduction and separately marked course-protocol runs.

The pinned architecture, losses, optimizer and train method remain unmodified.
Wrappers add raw outputs and fixed-final-epoch evaluation. Author-best checkpoint
scores are retained separately and never substituted for final-epoch results.
"""

import argparse
import collections
import itertools
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from data.load_dataset import load_dataset
from data.preprocessing import dense_views, prepare_observed
from data.missing import make_mask, mask_summary
from data.course_protocols import mask_hash, noisy_observations
from evaluation.metrics import scores, silhouette
from evaluation.neighborhood import diagnose
from evaluation.records import Measurement, aggregate, environment, file_hash, logger_at, write_csv, write_json
from models.author_import import author_path
from models.kmeans import fit_kmeans

ROOT = Path(__file__).resolve().parents[1]


def export_representations(model, data, mask, complete_missing=True):
    device = data[0].device
    mask_tensor = torch.as_tensor(mask, device=device)
    model.autoencoder1.eval(), model.autoencoder2.eval()
    model.img2txt.eval(), model.txt2img.eval()
    encoders=[model.autoencoder1, model.autoencoder2]
    maps=[model.txt2img, model.img2txt]
    latents=[]
    with torch.no_grad():
        for j in range(2):
            observed=mask_tensor[:,j]
            latent=torch.zeros((len(mask), model._latent_dim),device=device)
            latent[observed]=encoders[j].encoder(data[j][observed])
            missing=~observed
            if complete_missing and missing.any():
                source=encoders[1-j].encoder(data[1-j][missing])
                latent[missing]=maps[j](source)[0]
            latents.append(latent.cpu().numpy())
    return latents, np.concatenate(latents,axis=1)


def one_run(args, seed, rate, pattern, out, author, config_fn, author_mask_fn):
    import model as author_model
    config=config_fn('Caltech101-20')
    config['training']['epoch']=args.epochs
    config['print_num']=min(50,args.epochs)
    if args.epochs % config['print_num']:
        raise ValueError('Epoch count must be a multiple of the evaluation interval')
    config['dataset']='Caltech101-20'
    config['missing_rate']=rate
    config['protocol']=args.protocol
    config['ablation']=args.variant
    config['data_seed']=seed
    config['noise_std']=args.noise_std
    tag=f'{pattern}_{rate:.2f}_{seed}_{args.variant}'
    folder=out/tag
    if args.resume and (folder/'metrics.json').exists() and (folder/'outputs.npz').exists() and (folder/'neighborhood.json').exists():
        old_config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        old_result=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if old_config!=config and any(old_config.get(key)!=config.get(key) for key in ['protocol','ablation','data_seed','noise_std']):
            raise ValueError(f'Resume configuration mismatch: {folder}')
        if old_result['epochs']!=args.epochs:
            raise ValueError('Resume epoch budget mismatch')
        print(f'Resuming: verified completed trial {tag}',flush=True)
        return old_result
    raw,truth=load_dataset('Caltech101-20')
    raw=dense_views(raw)
    raw=[raw[j] for j in [3,4]]
    preprocess_started=time.perf_counter()
    np.random.seed(seed)
    if args.protocol=='author':
        if args.variant!='full' or pattern!='random':
            raise ValueError('Author protocol has no course ablations or structured masks')
        mask=author_mask_fn(2,len(truth),rate).astype(bool)
        # The upstream loader uses one global min/max per view, not featurewise scaling.
        normalized=[((v-v.min())/(v.max()-v.min())).astype(np.float32) for v in raw]
        scaled_truth=normalized
        config['normalization']='upstream_global_minmax_each_view_before_masking'
        config['missing_rate_definition']='upstream_get_mask_argument_not_actual_block_fraction'
    else:
        mask=make_mask(len(truth),2,rate,seed,pattern,truth)
        corrupted=noisy_observations(raw,mask,seed,args.noise_std)
        normalized,scalers=prepare_observed(corrupted,mask,'zero')
        scaled_truth=[scaler.transform(v) for scaler,v in zip(scalers,raw)]
        config['normalization']='featurewise_minmax_fit_observed_rows_only'
        config['missing_rate_definition']='fraction_of_samples_missing_exactly_one_view'
    preprocess_seconds=time.perf_counter()-preprocess_started
    if args.variant=='no_consistency':
        config['training']['lambda3']=0.0
        config['training']['lambda4']=0.0
    if args.variant=='no_completion':
        config['training']['lambda2']=0.0
    config['mask_statistics']=mask_summary(mask)
    training_seed=config['seed'] if args.protocol=='author' else seed
    np.random.seed(training_seed)
    random.seed(training_seed+1)
    torch.manual_seed(training_seed+2)
    torch.cuda.manual_seed_all(training_seed+3)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device']=str(device)
    folder.mkdir(parents=True,exist_ok=True)
    log=logger_at(folder)
    log.info('HSACC protocol=%s seed=%d rate_parameter=%.2f epochs=%d',args.protocol,seed,rate,args.epochs)
    log.info('Actual mask statistics: %s',config['mask_statistics'])
    write_json(folder/'config.json',config)
    write_json(folder/'source_hashes.json',{str(p.relative_to(ROOT)):file_hash(p) for p in
        [author/'model.py',author/'loss.py',author/'utils/get_mask.py',author/'utils/next_batch.py',author/'configure/configure.py']})
    tensors=[torch.as_tensor(v*mask[:,j,None],device=device,dtype=torch.float32) for j,v in enumerate(normalized)]
    tensor_mask=torch.as_tensor(mask,device=device,dtype=torch.long)
    model=author_model.HSACC(config)
    original_weight_fn=author_model.compute_view_value
    if args.variant=='equal_weights':
        author_model.compute_view_value=lambda rs,H,view:torch.full((view,),1/view,device=H.device,dtype=H.dtype)
    optimizer=torch.optim.Adam(itertools.chain(model.autoencoder1.parameters(),model.autoencoder2.parameters(),
        model.img2txt.parameters(),model.txt2img.parameters()),lr=config['training']['lr'])
    accumulated=collections.defaultdict(list)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with Measurement() as measurement:
        original_best=model.train(config,log,accumulated,tensors[0],tensors[1],[truth,truth],tensor_mask,optimizer,device)
        if device.type=='cuda': torch.cuda.synchronize()
    completed,consensus=export_representations(model,tensors,mask,args.variant!='no_completion')
    options={'n_init':10,'max_iter':300,'tol':1e-4}
    final_cluster_started=time.perf_counter()
    _,pred=fit_kmeans(consensus,len(np.unique(truth)),seed,options)
    final_cluster_seconds=time.perf_counter()-final_cluster_started
    score=scores(truth,pred)
    result={'dataset':'Caltech101-20','views':'HOG+GIST','protocol':args.protocol,'variant':args.variant,
        'run_type':'full_length' if args.epochs==500 else 'smoke','seed':seed,'rate_parameter':rate,'pattern':pattern,
        'actual_block_rate':float((~mask).mean()),'actual_incomplete_rate':float((~mask.all(axis=1)).mean()),
        'epochs':args.epochs,**score,'silhouette':silhouette(consensus,pred,seed),
        'noise_std':args.noise_std,'mask_sha256':mask_hash(mask),
        'preprocess_s':preprocess_seconds,'final_clustering_s':final_cluster_seconds,
        'train_with_periodic_evaluation_s':measurement.seconds,'peak_rss_mb':measurement.peak_rss_mb,
        'process_peak_rss_mb':measurement.process_lifetime_peak_mb,
        'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        'upstream_best_acc':float(original_best[0]),'upstream_best_nmi':float(original_best[1]),'upstream_best_ari':float(original_best[2])}
    write_json(folder/'metrics.json',result)
    write_json(folder/'upstream_evaluation_trace.json',dict(accumulated))
    np.savez_compressed(folder/'outputs.npz',mask=mask,truth=truth,predicted=pred,consensus=consensus,
        completed_view1=completed[0],completed_view2=completed[1],sample_indices=np.arange(len(truth)))
    # Complete missing ground truth is accessed only now, for offline diagnostics.
    with torch.no_grad():
        true_latents=[encoder.encoder(torch.as_tensor(v,device=device)).cpu().numpy()
                      for encoder,v in zip([model.autoencoder1,model.autoencoder2],scaled_truth)]
    write_json(folder/'neighborhood.json',diagnose(true_latents,completed,consensus,mask,truth))
    np.savez_compressed(folder/'true_latents_for_evaluation_only.npz',view1=true_latents[0],view2=true_latents[1])
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/'final_state.pt')
    author_model.compute_view_value=original_weight_fn
    log.info('Fixed final epoch: ACC=%.4f NMI=%.4f ARI=%.4f',score['acc'],score['nmi'],score['ari'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=['author','course'],default='author')
    parser.add_argument('--seeds',default='1,2,3')
    parser.add_argument('--rates',default='0.5')
    parser.add_argument('--patterns',default='random')
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--variant',choices=['full','no_consistency','no_completion','equal_weights'],default='full')
    parser.add_argument('--output',default='results/hsacc_author')
    parser.add_argument('--noise-std',type=float,default=0.0)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    author=author_path('HSACC')
    from configure.configure import get_default_config
    # Compatibility change is made in memory; raw author files are not edited.
    source=(author/'utils/get_mask.py').read_text(encoding='utf-8').replace('np.int)','int)')
    namespace={}
    exec(compile(source,str(author/'utils/get_mask.py'),'exec'),namespace)
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'environment.json',environment())
    write_json(out/'invocation.json',vars(args))
    rows=[]
    with threadpool_limits(limits=1):
        for pattern in args.patterns.split(','):
            for rate in map(float,args.rates.split(',')):
                for seed in map(int,args.seeds.split(',')):
                    rows.append(one_run(args,seed,rate,pattern,out,author,get_default_config,namespace['get_mask']))
                    write_csv(out/'runs.csv',rows)
    write_csv(out/'summary.csv',aggregate(rows,['dataset','protocol','variant','pattern','rate_parameter','epochs'],
        ['acc','nmi','ari','train_with_periodic_evaluation_s','peak_rss_mb','peak_gpu_allocated_mb']))


if __name__=='__main__': main()
