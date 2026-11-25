"""Small live baseline demonstration, or explicitly labeled replay of verified results."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data.load_dataset import load_dataset
from data.preprocessing import dense_views,prepare_observed
from data.missing import make_mask,mask_summary
from evaluation.metrics import scores
from models.kmeans import fit_kmeans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['live','replay'],default='replay')
    parser.add_argument('--scenario',choices=['a','b'],default='a')
    args=parser.parse_args()
    if args.mode=='replay':
        print('SAVED RESULT REPLAY: no neural training is performed by this command.')
        if args.scenario=='a':
            path=ROOT/'reports/tables/scenario_a_summary.csv'
            frame=pd.read_csv(path)
            print(frame[frame.pattern=='random'][['rate_parameter','method','acc_mean','acc_std','nmi_mean','ari_mean']].to_string(index=False))
        else:
            frame=pd.read_csv(ROOT/'reports/tables/scenario_b_summary.csv')
            print(frame[frame.method=='dsvc_course'][['condition','view','acc_mean','acc_std','cumulative_s_mean','retained_state_mb_mean']].to_string(index=False))
        print('All means use three seeds; SD uses ddof=1. See reports/多视图聚类课程设计_实验报告.docx for limitations.')
        return
    if args.scenario=='b':
        from scipy.io import loadmat
        from sklearn.preprocessing import MinMaxScaler
        from data.course_protocols import noisy_stream_view
        from models.view_stream import ViewBaseline
        raw=loadmat(ROOT/'third_party/DSVC/Datasets/HandWritten.mat',variable_names=['X','Y'])
        views=dense_views(raw['X'].ravel())
        truth=raw['Y'].ravel()
        model=ViewBaseline('warm_start',10,11,{'n_init':10,'max_iter':300,'tol':1e-4})
        print('LIVE BASELINE DEMO: HandWritten, 6 arriving views, noise starts at view 3, seed 11.')
        with threadpool_limits(limits=1):
            for v,view in enumerate(views,1):
                current=noisy_stream_view(view,11,v,3,'view_noise',1.0)
                pred,_=model.update(MinMaxScaler().fit_transform(current))
                print(json.dumps({'view':v,**scores(truth,pred),'state_mib':model.state_bytes/1024**2}))
    else:
        views,truth=load_dataset('Caltech101-20')
        views=dense_views(views)
        views=[views[j] for j in [3,4]]
        mask=make_mask(len(truth),2,.4,1,'random',truth)
        print('LIVE BASELINE DEMO: Caltech HOG/GIST, 40% incomplete samples, random, seed 1.')
        print(json.dumps(mask_summary(mask)))
        with threadpool_limits(limits=1):
            for fill in ['zero','mean']:
                blocks,_=prepare_observed(views,mask,fill)
                _,pred=fit_kmeans(np.concatenate(blocks,axis=1),20,1,{'n_init':10,'max_iter':300,'tol':1e-4})
                print(json.dumps({'method':fill,**scores(truth,pred)}))
    print('Demo output is not added to the formal result tables or counted as an extra seed.')


if __name__=='__main__': main()
