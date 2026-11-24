"""Offline feature-space diagnostics; never feeds hidden features back to learning."""

import json
import os
from pathlib import Path
os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import torch

from data.load_dataset import load_dataset
from data.preprocessing import dense_views, prepare_observed
from evaluation.neighborhood import neighbors, overlap
from evaluation.records import file_hash, write_json
from models.author_import import author_path

ROOT=Path(__file__).resolve().parents[1]


def main():
    author_path('HSACC')
    from model import HSACC,compute_view_value
    torch.set_num_threads(1)
    raw,truth=load_dataset('Caltech101-20')
    raw=[dense_views(raw)[j] for j in [3,4]]
    out=ROOT/'reports'
    figures=out/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    rows=[]
    hashes={}
    weights=[]
    with threadpool_limits(limits=1):
        for seed in [1,2,3]:
            folder=ROOT/f'experiments/06_course_comparison/results_a/hsacc/random_0.40_{seed}_full'
            config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
            config['device']='cpu'
            model=HSACC(config)
            model.load_state_dict(torch.load(folder/'final_state.pt',map_location='cpu',weights_only=True))
            # HSACC overrides Module.train(); switch child modules individually.
            model.autoencoder1.eval()
            model.autoencoder2.eval()
            with np.load(folder/'outputs.npz') as outputs:
                mask=outputs['mask'].copy()
                consensus=outputs['consensus'].copy()
                latent=[outputs['completed_view1'].copy(),outputs['completed_view2'].copy()]
                if not np.array_equal(truth,outputs['truth']): raise ValueError('Object order differs')
            mean,scalers=prepare_observed(raw,mask,'mean')
            selected=np.flatnonzero(mask.all(axis=1))[:256]
            pair=[torch.as_tensor(z[selected]) for z in latent]
            with torch.no_grad(): value=compute_view_value(pair,(pair[0]+pair[1])/2,2).numpy()
            weights.append({'seed':seed,'view1':float(value[0]),'view2':float(value[1]),
                'max_difference_from_half':float(np.max(np.abs(value-.5)))})
            full=[scaler.transform(v).astype(np.float32) for scaler,v in zip(scalers,raw)]
            consensus_knn=neighbors(consensus)
            completed=[]
            for view,(real,filled,ae,z) in enumerate(zip(full,mean,[model.autoencoder1,model.autoencoder2],latent)):
                missing=~mask[:,view]
                restored=real.copy()
                with torch.no_grad(): decoded=ae.decoder(torch.as_tensor(z[missing])).numpy()
                restored[missing]=decoded
                completed.append(restored)
                true_knn=neighbors(real)
                for name,candidate in [('mean_fill',filled),('decoded_completion',restored)]:
                    found=neighbors(candidate)
                    rows.append({'seed':seed,'view':view+1,'representation':name,'missing_samples':int(missing.sum()),
                        'k':10,'neighbor_overlap_all':overlap(true_knn,found),
                        'neighbor_overlap_missing':overlap(true_knn,found,missing),
                        'neighbor_overlap_observed':overlap(true_knn,found,~missing),
                        'missing_mse':float(np.mean((real[missing]-candidate[missing])**2)),
                        'consensus_overlap_with_true_view':overlap(true_knn,consensus_knn),
                        'consensus_overlap_missing':overlap(true_knn,consensus_knn,missing)})
            if seed==1:
                # Fixed seed, all objects; labels color figures only. No selected best run.
                cmap=plt.get_cmap('tab20',20)
                norm=BoundaryNorm(np.arange(21)-0.5,20)
                labels=np.unique(truth,return_inverse=True)[1]
                fig,axes=plt.subplots(1,3,figsize=(14,4.5))
                representations=[np.concatenate(full,axis=1),np.concatenate(completed,axis=1),consensus]
                for axis,features,title in zip(axes,representations,['True complete raw features (offline only)',
                        'Observed raw + decoded missing features','HSACC completed consensus (latent)']):
                    embedding=PCA(n_components=2,svd_solver='randomized',random_state=1).fit_transform(features)
                    scatter=axis.scatter(embedding[:,0],embedding[:,1],c=labels,cmap=cmap,norm=norm,s=4,alpha=.6,rasterized=True)
                    axis.set(title=title,xlabel='PC1',ylabel='PC2')
                fig.suptitle('Caltech HOG/GIST | 40% incomplete samples, random | fixed seed 1, all 2386 objects\nIndependent PCA axes; colors = true classes for offline illustration only')
                fig.tight_layout(rect=(0,0,.94,.88))
                fig.colorbar(scatter,ax=axes.ravel().tolist(),ticks=np.arange(20),fraction=.015,pad=.02,label='Class index')
                fig.savefig(figures/'course_a_geometry_pca.png',dpi=160,bbox_inches='tight')
                plt.close(fig)
            hashes[str((folder/'final_state.pt').relative_to(ROOT))]=file_hash(folder/'final_state.pt')
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'geometry_runs.csv',index=False,encoding='utf-8-sig')
    summary=frame.groupby(['view','representation'])[['neighbor_overlap_missing','neighbor_overlap_all',
        'missing_mse','consensus_overlap_with_true_view','consensus_overlap_missing']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') for x in summary.columns]
    summary.to_csv(out/'geometry_summary.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for axis,view in zip(axes,[1,2]):
        current=frame[frame.view==view]
        stats=current.groupby('representation').neighbor_overlap_missing.agg(['mean','std']).reindex(['mean_fill','decoded_completion'])
        consensus=current.drop_duplicates('seed').consensus_overlap_missing
        axis.bar([0,1,2],[*stats['mean'],consensus.mean()],yerr=[*stats['std'],consensus.std(ddof=1)],capsize=4)
        axis.set(xticks=[0,1,2],xticklabels=['Mean fill','Decoded','Consensus'],ylabel='Recall of true raw-view 10-NN',
            title=f'View {view}: missing rows only',ylim=(0,.15))
    fig.suptitle('Caltech HOG/GIST | 40% incomplete samples, random | 3 seeds (mean +/- sample SD)')
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(figures/'course_a_geometry_neighbors.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    write_json(out/'geometry_verification.json',{'runs':len(frame),'seeds':[1,2,3],'checkpoint_sha256':hashes,
        'two_view_weight_check':weights,
        'evaluation_only':True,'geometry':'Euclidean 10-NN on observed-only MinMax features',
        'decode_note':'Decoder is an offline diagnostic; clustering uses completed latent representations.'})
    print(summary.to_string(index=False))


if __name__=='__main__': main()
