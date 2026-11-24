"""Validate experiment documentation, local Markdown links and recorded run counts."""

import csv
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
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
    print('Verified 5 experiment directories, documentation links and 396 recorded evaluations.')


if __name__=='__main__': main()
