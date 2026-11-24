"""Validate paired course records and create analysis tables and publication figures."""

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
from evaluation.records import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
FIG=OUT/'figures'
NAMES={'hsacc_course':'HSACC course','mean_fill_kmeans':'Mean fill','zero_fill_kmeans':'Zero fill',
       'warm_start':'Warm start','sliding_window':'Window K-means','full_retrain':'Full retrain','dsvc_course':'DSVC course'}


def save_csv(name,frame): frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def grouped(frame,keys,metrics):
    value=frame.groupby(keys,dropna=False)[metrics].agg(['mean','std']).reset_index()
    value.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in value.columns]
    return value


def finish(fig,name):
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(FIG/name,bbox_inches='tight')
    plt.close(fig)


def scenario_a():
    base=pd.read_csv(ROOT/'experiments/06_course_comparison/results_a/baselines/runs.csv')
    learned=pd.read_csv(ROOT/'experiments/06_course_comparison/results_a/hsacc/runs.csv')
    if len(base)!=54 or len(learned)!=27: raise ValueError('Incomplete paired A matrix')
    learned['method']='hsacc_course'
    cols=['method','pattern','rate_parameter','seed','mask_sha256','acc','nmi','ari']
    all_runs=pd.concat([base[cols],learned[cols]],ignore_index=True)
    if not (all_runs.groupby(['pattern','rate_parameter','seed']).mask_sha256.nunique()==1).all():
        raise ValueError('A masks are not paired')
    summary=grouped(all_runs,['pattern','rate_parameter','method'],['acc','nmi','ari'])
    save_csv('scenario_a_runs.csv',all_runs)
    save_csv('scenario_a_summary.csv',summary)
    control=base[base.method=='mean_fill_kmeans']
    pairs=learned.merge(control,on=['pattern','rate_parameter','seed'],suffixes=('_model','_control'))
    pairs['acc_gain']=pairs.acc_model-pairs.acc_control
    gains=grouped(pairs,['pattern','rate_parameter'],['acc_gain'])
    gains['positive_seeds']=pairs.groupby(['pattern','rate_parameter']).acc_gain.apply(lambda x:int((x>0).sum())).values
    save_csv('scenario_a_paired_gains.csv',gains)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for axis,pattern in zip(axes,['random','single_view','class_related']):
        for method,group in all_runs[all_runs.pattern==pattern].groupby('method'):
            stats=group.groupby('rate_parameter').acc.agg(['mean','std'])
            axis.errorbar(stats.index*100,stats['mean'],yerr=stats['std'],capsize=3,marker='o',label=NAMES[method])
        axis.set(title=pattern,xlabel='Incomplete samples (%)',xticks=[20,40,60],ylim=(0,1))
        axis.legend(fontsize=8)
    axes[0].set_ylabel('ACC (mean +/- sample SD)')
    fig.suptitle('Paired Caltech HOG/GIST | observed-only MinMax | same masks | 3 seeds')
    finish(fig,'course_a_missingness.png')
    random_gains=gains[gains.pattern=='random']
    return all_runs,{'a_records':len(all_runs),'a_masks_matched':True,
        'random_rates_with_three_positive_acc_gains':int((random_gains.positive_seeds==3).sum())}


def scenario_b():
    frame=pd.read_csv(ROOT/'experiments/06_course_comparison/results_b/runs.csv')
    if len(frame)!=144: raise ValueError('Incomplete paired B matrix')
    if not (frame.groupby(['condition','seed','view']).raw_view_sha256.nunique()==1).all():
        raise ValueError('B observations are not paired')
    save_csv('scenario_b_runs.csv',frame)
    save_csv('scenario_b_summary.csv',grouped(frame,['condition','view','method'],['acc','nmi','ari','probe_acc','forgetting','label_change',
        'update_s','cumulative_s','process_peak_rss_mb','peak_gpu_mb','retained_state_mb']))
    save_csv('scenario_b_final.csv',grouped(frame[frame.view==6],['condition','method'],['acc','nmi','ari','probe_acc','forgetting',
        'cumulative_s','process_peak_rss_mb','peak_gpu_mb','retained_state_mb']))
    recovery=pd.read_csv(ROOT/'experiments/06_course_comparison/results_b/recovery.csv')
    save_csv('scenario_b_recovery.csv',recovery)
    clean=frame[frame.condition=='none']
    noisy=frame[frame.condition=='view_noise']
    pairs=noisy.merge(clean,on=['method','seed','view'],suffixes=('_noise','_clean'))
    pairs['noise_acc_effect']=pairs.acc_noise-pairs.acc_clean
    save_csv('scenario_b_noise_effect.csv',grouped(pairs,['method','view'],['noise_acc_effect']))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,condition in enumerate(['none','view_noise']):
        for col,metric in enumerate(['acc','nmi','ari']):
            axis=axes[row,col]
            for method,group in frame[frame.condition==condition].groupby('method'):
                stats=group.groupby('view')[metric].agg(['mean','std'])
                axis.errorbar(stats.index,stats['mean'],yerr=stats['std'],capsize=2,marker='.',label=NAMES[method])
            if condition!='none': axis.axvline(3,color='gray',ls='--',lw=1)
            axis.set(title=f'{condition}: {metric.upper()}',xlabel='Arriving view',xticks=range(1,7),ylim=(0,1.05))
            axis.legend(fontsize=7)
    fig.suptitle('Matched HandWritten view stream | 3 seeds | mean +/- sample SD | drift starts at view 3')
    finish(fig,'course_b_quality.png')
    for field,name,ylabel in [('probe_acc','course_b_retention.png','Fixed first-view probe ACC'),
            ('label_change','course_b_label_changes.png','Permutation-aligned label change'),
            ('cumulative_s','course_b_time.png','Cumulative update seconds'),
            ('retained_state_mb','course_b_state.png','Retained array state (MiB)')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for axis,condition in zip(axes,['none','view_noise']):
            for method,group in frame[frame.condition==condition].groupby('method'):
                stats=group.groupby('view')[field].agg(['mean','std']).dropna(subset=['mean'])
                axis.errorbar(stats.index,stats['mean'],yerr=stats['std'],capsize=2,marker='.',label=NAMES[method])
            axis.set(title=condition,xlabel='Arriving view',ylabel=ylabel,xticks=range(1,7))
            if field=='cumulative_s': axis.set_yscale('log')
            if condition!='none': axis.axvline(3,color='gray',ls='--',lw=1)
            axis.legend(fontsize=7)
        fig.suptitle('HandWritten | same raw views | 3 seeds (mean +/- sample SD)')
        finish(fig,name)
    return {'b_records':len(frame),'b_inputs_matched':True,'b_budget_exceedances':int(frame.budget_exceeded.sum())}


def extras(a_runs):
    root=ROOT/'experiments/07_ablations_robustness/results'
    full=pd.read_csv(ROOT/'experiments/06_course_comparison/results_a/hsacc/runs.csv')
    full=full[(full.pattern=='random')&(full.rate_parameter==0.4)].copy()
    ablations=[full]
    for variant in ['no_consistency','no_completion','equal_weights']:
        current=pd.read_csv(root/variant/'runs.csv')
        if len(current)!=3: raise ValueError('Incomplete ablation')
        ablations.append(current)
    ablation=pd.concat(ablations,ignore_index=True)
    if not (ablation.groupby('seed').mask_sha256.nunique()==1).all(): raise ValueError('Ablation masks differ')
    save_csv('ablation_runs.csv',ablation)
    save_csv('ablation_summary.csv',grouped(ablation,['variant'],['acc','nmi','ari']))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    order=['full','no_consistency','no_completion','equal_weights']
    labels=['Full','No consistency','No completion','Equal weights']
    for axis,metric in zip(axes,['acc','nmi','ari']):
        stats=ablation.groupby('variant')[metric].agg(['mean','std']).reindex(order)
        axis.bar(np.arange(4),stats['mean'],yerr=stats['std'],capsize=3)
        axis.set_xticks(np.arange(4),labels,rotation=35,ha='right')
        axis.set(ylabel=metric.upper(),ylim=(0,1))
    fig.suptitle('Caltech HOG/GIST | 40% incomplete samples, random | 3 seeds (mean +/- SD)')
    finish(fig,'course_a_ablations.png')
    noisy_base=pd.read_csv(root/'noise_baselines/runs.csv')
    noisy_model=pd.read_csv(root/'noise_hsacc/runs.csv')
    noisy_model['method']='hsacc_course'
    noisy=pd.concat([noisy_base,noisy_model],ignore_index=True)
    if len(noisy)!=9: raise ValueError('Incomplete robustness matrix')
    cols=['method','seed','mask_sha256','acc','nmi','ari']
    clean=a_runs[(a_runs.pattern=='random')&(a_runs.rate_parameter==0.4)][cols].copy()
    noisy=noisy[cols].copy()
    paired=noisy.merge(clean,on=['method','seed'],suffixes=('_noise','_clean'))
    if not (paired.mask_sha256_noise==paired.mask_sha256_clean).all(): raise ValueError('Robustness masks differ')
    paired['noise_acc_effect']=paired.acc_noise-paired.acc_clean
    save_csv('robustness_paired_effects.csv',paired)
    clean['condition']='clean'
    noisy['condition']='one_view_noise'
    combined=pd.concat([clean,noisy],ignore_index=True)
    save_csv('robustness_runs.csv',combined)
    save_csv('robustness_summary.csv',grouped(combined,['method','condition'],['acc','nmi','ari']))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    methods=['zero_fill_kmeans','mean_fill_kmeans','hsacc_course']
    for axis,metric in zip(axes,['acc','nmi']):
        for i,condition in enumerate(['clean','one_view_noise']):
            stats=combined[combined.condition==condition].groupby('method')[metric].agg(['mean','std']).reindex(methods)
            axis.bar(np.arange(3)+(i-0.5)*0.35,stats['mean'],width=0.35,yerr=stats['std'],capsize=3,label=condition)
        axis.set_xticks(np.arange(3),[NAMES[m] for m in methods])
        axis.set(ylabel=metric.upper(),ylim=(0,1))
        axis.legend(fontsize=8)
    fig.suptitle('Caltech | 40% incomplete samples, random | noise = observed view SD | 3 seeds')
    finish(fig,'course_a_robustness.png')
    efficiency=pd.read_csv(root/'isolated_efficiency.csv')
    if len(efficiency)!=9: raise ValueError('Incomplete isolated efficiency cases')
    save_csv('isolated_efficiency.csv',efficiency)
    save_csv('isolated_efficiency_summary.csv',grouped(efficiency,['method'],['elapsed_s','process_peak_rss_mb','peak_gpu_mb','acc']))
    return {'ablation_records':len(ablation),'robustness_records':len(combined),'isolated_a_cases':len(efficiency)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['a','b','all'],default='all')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    with threadpool_limits(limits=1):
        if args.stage=='a':
            _,validation=scenario_a()
        elif args.stage=='b':
            validation=scenario_b()
        else:
            a,validation=scenario_a()
            validation.update(scenario_b())
            validation.update(extras(a))
    suffix='' if args.stage=='all' else '_'+args.stage
    write_json(OUT/f'analysis{suffix}_verification.json',validation)
    print(json.dumps(validation,indent=2))


if __name__=='__main__': main()
