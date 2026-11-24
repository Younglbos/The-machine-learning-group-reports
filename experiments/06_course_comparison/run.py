"""Run paired course comparisons and optionally the remaining frozen experiment suite."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments._runner import run_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['a','b','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['a','all']:
        run_module('train_or_update.course_a_baselines',['--references'])
        run_module('train_or_update.run_hsacc',['--protocol','course','--rates','0.2,0.4,0.6',
            '--patterns','random,single_view,class_related','--seeds','1,2,3','--epochs','500','--resume',
            '--output','experiments/06_course_comparison/results_a/hsacc'])
    if args.stage=='b': run_module('train_or_update.run_view_suite',[])
    if args.stage=='all': run_module('train_or_update.run_followup_suite',[])


if __name__=='__main__': main()
