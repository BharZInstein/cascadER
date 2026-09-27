"""Run cascadER on three local business-record TSV files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['entity_id', 'business_name', 'business_address', 'country']


def check_inputs(data):
    identities = []
    counts = []
    for source in (1, 2, 3):
        path = data / f'test_source{source}.tsv'
        count = 0
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            if reader.fieldnames != FIELDS:
                raise ValueError(f'{path}: expected columns {FIELDS}')
            for count, row in enumerate(reader, 1):
                eid = row['entity_id']
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'{path}: malformed row {count}')
                if not re.fullmatch(fr'S{source}-(0|[1-9][0-9]*)', eid):
                    raise ValueError(f'{path}: invalid ID {eid!r}; see README')
                if int(eid[3:]) >= 2**30:
                    raise ValueError(f'{path}: ID suffix is too large: {eid}')
        stat = path.stat()
        identities.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        counts.append(count)
    if not counts[0] or not sum(counts[1:]):
        raise ValueError('At least one reference and one target are required')
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'models')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    data, work, model = (p.resolve() for p in (args.data_dir, args.work_dir, args.model))
    identity = check_inputs(data)
    model_config = json.loads((model / 'model_config.json').read_text())
    if model_config.get('empty_candidate_threshold') is not None:
        raise ValueError('This entry point supports primary retrieval only; use the lower-level inference CLI for auxiliary retrieval')
    artifact_names = ['model_config.json', model_config.get('model_file', 'model.ubj'), 'base_model.ubj', 'text_weights.joblib']
    fingerprints = {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in artifact_names}
    source_fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'src').glob('*.py')}
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'run.json'
    expected = dict(inputs=identity, model=str(model), artifacts=fingerprints, source=source_fingerprints)
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(expected, completed=[])
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError('Inputs, model, or source changed; use a new work directory')

    def save():
        temporary = work / 'run.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    def run(name, script, *arguments):
        if name in state['completed']:
            return
        print(f'\n[{name}]', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'src' / script), *map(str, arguments)], check=True)
        state['completed'].append(name)
        save()

    save()
    db, refs = work / 'targets.sqlite', work / 'references.sqlite'
    competition, output = work / 'competition', work / 'output'
    run('index', 'pipeline.py', 'index', '--sources', data / 'test_source2.tsv', data / 'test_source3.tsv', '--db', db)
    run('extra_blocks', 'retrieval_extras.py', '--db', db, '--out', str(db) + '.extra.sqlite')
    run('references', 'reference_index.py', '--source', data / 'test_source1.tsv', '--out', refs)
    run('competition', 'precompute_competition.py', '--db', db, '--references', refs, '--out', competition, '--workers', args.workers)
    run('predict', 'predict_parallel.py', '--source1', data / 'test_source1.tsv', '--db', db,
        '--model', model, '--competition-cache', competition, '--out', output,
        '--workers', args.workers, '--save-scores')
    run('ownership', 'apply_unique_targets.py', '--output', output)
    run('validate', 'validate_outputs.py', '--test-dir', data, '--output', output,
        '--report', work / 'validation.json', '--unique-targets')
    print(f'\nMatches: {output / "matching_results.tsv"}')


if __name__ == '__main__':
    main()
