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
    args=parser.parse_args()
    experiments=sorted((ROOT/'experiments').glob('[0-9][0-9]_*'))
    if len(experiments)!=5:
        raise ValueError('Expected five documented experiments')
    for folder in experiments:
        for name in ['README.md','run.py','config.json']:
            if not (folder/name).is_file(): raise FileNotFoundError(folder/name)
        json.loads((folder/'config.json').read_text(encoding='utf-8'))
    for doc in [ROOT/'README.md',ROOT/'THIRD_PARTY.md',*(x/'README.md' for x in experiments)]:
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
    print(f'Verified 5 experiment directories, documentation links, 396 baseline records and {paper_records} requested paper-code records.')


if __name__=='__main__': main()
