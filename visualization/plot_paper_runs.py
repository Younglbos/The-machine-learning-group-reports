"""Plot verified paper-code runs, keeping their protocols separate from course baselines."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def hsacc():
    root=ROOT/'experiments/04_hsacc_reproduction/results'
    frame=pd.read_csv(root/'runs.csv')
    if len(frame)!=3 or set(frame.epochs)!={500}:
        raise ValueError('HSACC figures require the three completed 500-epoch runs')
    dest=root/'figures'
    dest.mkdir(exist_ok=True)
    traces=[]
    neighborhoods=[]
    metadata=root/'reproduction_details.json'
    consolidated={entry['case']:entry for entry in json.loads(metadata.read_text(encoding='utf-8'))['trials']} if metadata.exists() else {}
    for row in frame.itertuples():
        folder=root/f'{row.pattern}_{row.rate_parameter:.2f}_{row.seed}_{row.variant}'
        if (folder/'upstream_evaluation_trace.json').is_file() and (folder/'neighborhood.json').is_file():
            trace=json.loads((folder/'upstream_evaluation_trace.json').read_text(encoding='utf-8'))
            neighborhood=json.loads((folder/'neighborhood.json').read_text(encoding='utf-8'))
        else:
            saved=consolidated[folder.name]
            trace,neighborhood=saved['periodic_evaluation'],saved['neighborhood']
        traces.append(trace)
        neighborhoods.extend({'seed':row.seed,**entry} for entry in neighborhood)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for axis,key,label in zip(axes,['acc','nmi','ARI'],['ACC','NMI','ARI']):
        values=np.asarray([trace[key] for trace in traces])*100
        epochs=np.arange(1,values.shape[1]+1)*50
        axis.errorbar(epochs,values.mean(axis=0),yerr=values.std(axis=0,ddof=1),capsize=3,marker='.')
        axis.set(xlabel='Training epoch',ylabel=f'Periodic {label} (%)')
    fig.suptitle('HSACC | Caltech HOG/GIST | author mask parameter 0.5 | 3 data seeds (mean +/- SD)')
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(dest/'training_metrics.png',bbox_inches='tight')
    plt.close(fig)
    table=pd.DataFrame(neighborhoods)
    table.to_csv(root/'neighborhood_runs.csv',index=False,encoding='utf-8-sig')
    columns=['completion_overlap_missing','completion_overlap_observed','consensus_overlap_with_true_view']
    summary=table.groupby('view')[columns].agg(['mean','std'])
    exported=summary.copy()
    exported.columns=['_'.join(column) for column in exported.columns]
    exported.reset_index().to_csv(root/'neighborhood_summary.csv',index=False,encoding='utf-8-sig')
    fig,axis=plt.subplots(figsize=(8,4))
    for i,(key,label) in enumerate(zip(columns,['Completed: missing rows','Completed: observed rows','Consensus vs. true view'])):
        axis.bar(np.arange(2)+(i-1)*0.25,summary[(key,'mean')],width=0.25,yerr=summary[(key,'std')],capsize=3,label=label)
    axis.set(xticks=[0,1],xticklabels=['HOG','GIST'],ylabel='10-neighbor overlap',ylim=(0,1))
    axis.legend(fontsize=8)
    axis.set_title('HSACC | Caltech | author mask 0.5 | 3 data seeds (mean +/- SD)')
    fig.tight_layout()
    fig.savefig(dest/'neighborhood_overlap.png',bbox_inches='tight')
    plt.close(fig)
    folder=root/'random_0.50_1_full'
    if not (folder/'outputs.npz').is_file() or not (folder/'true_latents_for_evaluation_only.npz').is_file():
        print('Retained saved PCA figure; generating a new embedding requires local training outputs.')
        return
    with np.load(folder/'outputs.npz') as output,np.load(folder/'true_latents_for_evaluation_only.npz') as reference:
        representations=[reference['view1'],output['completed_view1'],reference['view2'],output['consensus']]
        titles=['True HOG encoding (offline)','Completed HOG encoding','True GIST encoding (offline)','Concatenated consensus']
        truth=output['truth']
        fig,axes=plt.subplots(1,4,figsize=(14,3.8))
        for axis,features,title in zip(axes,representations,titles):
            points=PCA(n_components=2,svd_solver='full').fit_transform(features)
            scatter=axis.scatter(points[:,0],points[:,1],c=truth,cmap='tab20',s=4,alpha=0.6,
                norm=BoundaryNorm(np.arange(0.5,21.5,1),20))
            axis.set(title=title,xlabel='PC1',ylabel='PC2')
        fig.suptitle('HSACC | Caltech | author mask 0.5 | seed 1 | independent PCA axes; class colors for evaluation')
        fig.tight_layout(rect=(0,0,0.95,0.94))
        fig.colorbar(scatter,ax=axes.tolist(),fraction=0.015,pad=0.02,ticks=[1,5,10,15,20],label='True class')
        fig.savefig(dest/'representation_pca.png',bbox_inches='tight')
        plt.close(fig)


def dsvc():
    root=ROOT/'experiments/05_dsvc_reproduction/results'
    frame=pd.read_csv(root/'runs.csv')
    if len(frame)!=18 or set(frame.epochs_per_view)!={200}:
        raise ValueError('DSVC figures require 3 seeds x 6 completed 200-epoch views')
    dest=root/'figures'
    dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for axis,key,label in zip(axes,['acc','nmi','ari'],['ACC','NMI','ARI']):
        summary=frame.groupby('view')[key].agg(['mean','std'])*100
        axis.errorbar(summary.index,summary['mean'],yerr=summary['std'],capsize=3,marker='o')
        axis.set(xlabel='Arriving view (1-based)',ylabel=f'{label} (%)',xticks=range(1,7))
    fig.suptitle('DSVC | HandWritten | 200 epochs/view | 3 seeds (mean +/- SD) | author view stream')
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(dest/'view_stream_metrics.png',bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['hsacc','dsvc','all'],default='all')
    args=parser.parse_args()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    with threadpool_limits(limits=1):
        if args.experiment in ['hsacc','all']: hsacc()
        if args.experiment in ['dsvc','all']: dsvc()
    print('Paper-code figures saved; author and course protocols remain separate.')


if __name__=='__main__': main()
