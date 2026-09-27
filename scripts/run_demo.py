"""Run the distributed model on fictional business records."""
from pathlib import Path
import csv
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / 'scripts/match.py'),
                    '--data-dir', str(root / 'examples/data'),
                    '--work-dir', str(root / 'artifacts/demo'), '--workers', '2', '--save-features'], check=True)
    result = root / 'artifacts/demo/output/matching_results.tsv'
    with result.open() as stream:
        actual = {r['source1_entity_id']: set(filter(None, r['matched_entity_ids'].split(',')))
                  for r in csv.DictReader(stream, delimiter='\t')}
    expected = {'S1-1': {'S2-1', 'S3-1'}, 'S1-2': {'S2-2', 'S3-2'},
                'S1-3': {'S2-3', 'S3-3'}, 'S1-4': set()}
    if actual != expected:
        raise RuntimeError(f'Demo predictions differ from the expected fictional matches: {actual}')
    print('\nFictional example predictions (verified):')
    print(result.read_text())
    subprocess.run([sys.executable, str(root / 'scripts/review.py'), '--work-dir',
                    str(root / 'artifacts/demo')], check=True)


if __name__ == '__main__':
    main()
