"""Validate experiment documentation, local Markdown links and recorded run counts."""

import argparse
import csv
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-paper-results',action='store_true')
    parser.add_argument('--include-course-results',action='store_true')
    args=parser.parse_args()
    experiments=sorted((ROOT/'experiments').glob('[0-9][0-9]_*'))
    if len(experiments)!=7:
        raise ValueError('Expected seven documented experiments')
    for folder in experiments:
        for name in ['README.md','run.py','config.json']:
            if not (folder/name).is_file(): raise FileNotFoundError(folder/name)
        json.loads((folder/'config.json').read_text(encoding='utf-8'))
    for doc in [ROOT/'README.md',ROOT/'THIRD_PARTY.md',*(x/'README.md' for x in experiments),ROOT/'reports/README.md']:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'): continue
            target=(doc.parent/link.split('#')[0]).resolve()
            if not target.exists(): raise FileNotFoundError(f'{doc.relative_to(ROOT)} -> {link}')
    expected={
        '01_complete_baselines/results/complete_runs.csv':63,
        '02_incomplete_multiview/results/scenario_a_runs.csv':54,
        '02_incomplete_multiview/results/scenario_a_complete_reference_runs.csv':9,
        '03_streaming_baselines/results/scenario_b_runs.csv':270,
    }
    for filename,count in expected.items():
        with (ROOT/'experiments'/filename).open(encoding='utf-8-sig',newline='') as source:
            rows=list(csv.DictReader(source))
        if len(rows)!=count or {int(row['seed']) for row in rows}!={1,2,3}:
            raise ValueError(f'Unexpected rows or seeds in {filename}')
        for row in rows:
            for field in ['acc','nmi','ari']:
                value=float(row[field])
                if not -1<=value<=1: raise ValueError(f'Invalid {field}: {filename}')
    paper_records=0
    if args.include_paper_results:
        checks=[('04_hsacc_reproduction/results/runs.csv',3,{1,2,3},'epochs',500),
                ('05_dsvc_reproduction/results/runs.csv',18,{11,21,22},'epochs_per_view',200)]
        for filename,count,seeds,epoch_key,epochs in checks:
            with (ROOT/'experiments'/filename).open(encoding='utf-8-sig',newline='') as source:
                rows=list(csv.DictReader(source))
            if len(rows)!=count or {int(row['seed']) for row in rows}!=seeds:
                raise ValueError(f'Incomplete paper-code results: {filename}')
            if any(row['run_type']!='full_length' or int(row[epoch_key])!=epochs for row in rows):
                raise ValueError(f'Smoke or unexpected budget in full results: {filename}')
            for row in rows:
                for metric in ['acc','nmi','ari']:
                    if not -1<=float(row[metric])<=1:
                        raise ValueError(f'Invalid {metric}: {filename}')
            paper_records+=len(rows)
    if args.include_course_results:
        expected_course={
            '06_course_comparison/results_a/baselines/runs.csv':54,
            '06_course_comparison/results_a/baselines/complete_reference_runs.csv':9,
            '06_course_comparison/results_a/hsacc/runs.csv':27,
            '06_course_comparison/results_b/runs.csv':144,
            '07_ablations_robustness/results/no_consistency/runs.csv':3,
            '07_ablations_robustness/results/no_completion/runs.csv':3,
            '07_ablations_robustness/results/equal_weights/runs.csv':3,
            '07_ablations_robustness/results/noise_hsacc/runs.csv':3,
            '07_ablations_robustness/results/noise_baselines/runs.csv':6,
            '07_ablations_robustness/results/isolated_efficiency.csv':9,
        }
        for filename,count in expected_course.items():
            with (ROOT/'experiments'/filename).open(encoding='utf-8-sig',newline='') as source:
                rows=list(csv.DictReader(source))
            if len(rows)!=count: raise ValueError(f'Incomplete course results: {filename}')
        analysis=json.loads((ROOT/'reports/analysis_verification.json').read_text(encoding='utf-8'))
        prediction=json.loads((ROOT/'reports/prediction_verification.json').read_text(encoding='utf-8'))
        if not (analysis['a_masks_matched'] and analysis['b_inputs_matched'] and prediction['all_metrics_recomputed']):
            raise ValueError('Course provenance verification failed')
        if prediction['verified_records']!=252: raise ValueError('Prediction count mismatch')
        for artifact in ['course_report.html','literature_review.html','course_predictions.jsonl.gz']:
            if not (ROOT/'reports'/artifact).is_file(): raise FileNotFoundError(artifact)
        print('Verified completed A/B course matrix, ablations, robustness, isolated efficiency and report artifacts.')
    print(f'Verified {len(experiments)} experiment directories, documentation links, 396 baseline records and {paper_records} requested paper-code records.')


if __name__=='__main__': main()
