"""Make missing-rate, per-batch, stability and efficiency figures from saved runs."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def plot_complete(out,dest):
    frame=pd.read_csv(out/'complete_runs.csv')
    fig,axes=plt.subplots(1,4,figsize=(14,4.2))
    for axis,(dataset,values) in zip(axes,frame.groupby('dataset',sort=False)):
        stats=values.groupby('method',sort=False).acc.agg(['mean','std'])
        axis.bar(range(len(stats)),stats['mean'],yerr=stats['std'],capsize=3)
        axis.set_xticks(range(len(stats)),[x.replace('_complete','').replace('single_','') for x in stats.index],rotation=60,ha='right')
        axis.set(title=dataset,ylabel='ACC (mean +/- sample SD)',ylim=(0,0.75))
    fig.suptitle('Complete-data K-means | per-view MinMax | 3 seeds')
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(dest/'complete_accuracy.png',bbox_inches='tight')
    plt.close(fig)


def plot_a(config,out,dest):
    missing = pd.read_csv(out/'scenario_a_runs.csv')
    fig, axes = plt.subplots(1,3,figsize=(13,4.0),sharey=True)
    for axis, pattern in zip(axes, config['scenario_a']['patterns']):
        for method, group in missing[missing.pattern==pattern].groupby('method'):
            stats=group.groupby('requested_rate').acc.agg(['mean','std'])
            axis.errorbar(stats.index*100, stats['mean'], yerr=stats['std'],marker='o',capsize=3,label=method)
        axis.set(title=pattern,xlabel='Incomplete samples (%)',xticks=[20,40,60])
        axis.legend(fontsize=8)
    axes[0].set_ylabel('ACC (mean +/- sample SD)')
    fig.suptitle('Caltech101-20 HOG/GIST | 3 seeds | one missing view per incomplete sample')
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(dest/'missing_rate.png',bbox_inches='tight')
    plt.close(fig)


def plot_b(config,out,dest):
    stream = pd.read_csv(out/'scenario_b_runs.csv')
    for metric, filename, ylabel in [('acc','stream_accuracy.png','Batch ACC'),
            ('cumulative_s','stream_time.png','Cumulative update + preprocessing (s)'),
            ('probe_acc','history_retention.png','Fixed historical probe ACC'),
            ('model_state_mb','stream_state.png','Retained model state (MiB)')]:
        fig,axes=plt.subplots(1,3,figsize=(13,4.0))
        for axis, condition in zip(axes,config['scenario_b']['drift_conditions']):
            for method,group in stream[stream.condition==condition].groupby('method'):
                stats=group.groupby('batch')[metric].agg(['mean','std'])
                axis.errorbar(stats.index,stats['mean'],yerr=stats['std'],marker='.',capsize=2,label=method)
            if condition!='none': axis.axvline(config['scenario_b']['drift_batch'],color='gray',ls='--',lw=1)
            axis.set(title=condition,xlabel='Arrival batch (1-based)',ylabel=ylabel)
            axis.legend(fontsize=8)
        fig.suptitle('NUS-WIDE 2400 | 10 sample batches | 3 seeds | first-batch frozen MinMax/PCA')
        fig.tight_layout(rect=(0,0,1,0.94))
        fig.savefig(dest/filename,bbox_inches='tight')
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/baselines.json')
    parser.add_argument('--stage',choices=['complete','a','b','all'],default='all')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    out=ROOT/config['result_dir']
    dest=out/'figures'
    dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    if args.stage in ['complete','all']: plot_complete(out,dest)
    if args.stage in ['a','all']: plot_a(config,out,dest)
    if args.stage in ['b','all']: plot_b(config,out,dest)
    print('Saved requested figures to',dest)


if __name__=='__main__':
    main()
