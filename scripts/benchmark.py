"""Reproducible fictional benchmark with fixed exact and fuzzy matching baselines."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import evaluate
from group_policy import chosen_pairs
from pipeline import record, rows, write_rows


def generate(destination, count, seed):
    """Create labeled records without consulting model scores or training examples."""
    rng = random.Random(seed)
    colors = 'Amber Azure Copper Coral Crimson Emerald Golden Indigo Ivory Jade Lilac Olive Opal Pearl Ruby Sage Scarlet Silver Teal Violet'.split()
    objects = 'Acorn Aspen Birch Cedar Clover Comet Crane Cypress Falcon Fern Finch Fox Heron Iris Juniper Kite Lantern Lark Laurel Lotus Maple Meadow Moon Oak Orchid Otter Pine Quartz Raven Reed Robin Sparrow Spruce Star Willow'.split()
    industries = ['Books', 'Textiles', 'Instruments', 'Bakery', 'Hardware', 'Design', 'Printing', 'Ceramics']
    combinations = [(a, b, c) for a in colors for b in objects for c in industries]
    if not 10 <= count <= len(combinations):
        raise ValueError(f'--references must be between 10 and {len(combinations)}')
    rng.shuffle(combinations)
    destination.mkdir(parents=True, exist_ok=False)
    sources = {1: [], 2: [], 3: []}
    truth, slices = [], {}
    modes = ['clean', 'case_and_punctuation', 'typo', 'reordered', 'missing_address', 'abbreviated']
    for i, words in enumerate(combinations[:count], 1):
        country = ['US', 'India', 'France'][(i - 1) % 3]
        mode = modes[((i - 1) // 3) % len(modes)]
        name = ' '.join(words)
        street = rng.choice(objects)
        number = rng.randrange(1, 900)
        suffix = {'US': ' LLC', 'India': ' Private Limited', 'France': ' SARL'}[country]
        address = {'US': f'{number} {street} Street Portland OR 97205',
                   'India': f'{number} {street} Road Pune Maharashtra 411001',
                   'France': f'{number} Rue {street} Lyon 69002'}[country]
        eid = f'S1-{i}'
        sources[1].append(dict(entity_id=eid, business_name=name + suffix, business_address=address, country=country))
        matches = []
        singleton = i % 7 == 0
        slices[eid] = dict(country=country, scenario=mode, singleton=singleton)
        if not singleton:
            for source in (2, 3):
                target_name, target_address = name + suffix, address
                if mode == 'case_and_punctuation':
                    target_name = name.upper().replace(' ', ' - ')
                    target_address = address.lower().replace(' ', ', ')
                elif mode == 'typo':
                    p = rng.randrange(1, len(words[0]) - 1)
                    noisy = words[0][:p] + words[0][p + 1:] + words[0][p]
                    target_name = ' '.join([noisy, words[1], words[2]])
                    target_address = address.replace(street, street[:-1])
                elif mode == 'reordered':
                    target_name = ' '.join(reversed(words))
                    target_address = ' '.join(reversed(address.split()))
                elif mode == 'missing_address':
                    target_name = name if source == 2 else name + suffix
                    target_address = ''
                elif mode == 'abbreviated':
                    target_name = name + {'US': ' Co', 'India': ' Pvt Ltd', 'France': ' SAS'}[country]
                    target_address = address.replace('Street', 'St').replace('Road', 'Rd').replace('Maharashtra', 'MH')
                target_id = f'S{source}-{i}'
                sources[source].append(dict(entity_id=target_id, business_name=target_name,
                                            business_address=target_address, country=country))
                matches.append(target_id)
        # A same-name business at a different address forces a false-merge decision.
        sources[2].append(dict(entity_id=f'S2-{count+i}', business_name=name + suffix,
                               business_address=address.replace(str(number), str(number + 3000), 1), country=country))
        truth.append(dict(source1_entity_id=eid, matched_entity_ids=','.join(matches)))
    for source, data in sources.items():
        write_rows(destination / f'test_source{source}.tsv', data)
    write_rows(destination / 'ground_truth.tsv', truth, ['source1_entity_id', 'matched_entity_ids'])
    (destination / 'slices.json').write_text(json.dumps(slices, indent=2) + '\n')


def read_links(path, column='matched_entity_ids'):
    return {r['source1_entity_id']: set(filter(None, r[column].split(','))) for r in rows(path)}


def score_baseline(references, targets, candidate_rows, exact=False):
    scores, groups, ids = [], [], []
    for group, (eid, candidates) in enumerate(candidate_rows):
        a = references[eid]
        for target in candidates:
            b = targets[target]
            # Both methods use the same retrieved candidates as the learned matcher.
            value = float(bool(a[1] and a[3]) and a[1] == b[1] and a[3] == b[3]) if exact else (
                .6 * fuzz.token_sort_ratio(a[1], b[1]) / 100 + .4 * fuzz.token_sort_ratio(a[3], b[3]) / 100)
            scores.append(value)
            groups.append(group)
            ids.append(target)
    scores, groups, ids = np.asarray(scores), np.asarray(groups, dtype=int), np.asarray(ids)
    selected = chosen_pairs(scores, 1.0 if exact else .85, groups, len(candidate_rows), targets=ids)
    predictions = {eid: set() for eid, _ in candidate_rows}
    for group, target in zip(groups[selected], ids[selected]):
        predictions[candidate_rows[group][0]].add(str(target))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'artifacts/benchmark')
    parser.add_argument('--references', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    data = work / 'data'
    generate(data, args.references, args.seed)
    started = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / 'scripts/match.py'), '--data-dir', str(data),
                    '--work-dir', str(work / 'run'), '--workers', str(args.workers), '--save-features'], check=True)
    elapsed = time.perf_counter() - started
    truth = read_links(data / 'ground_truth.tsv')
    candidate_rows = [(r['source1_entity_id'], list(filter(None, r['candidate_entity_ids'].split(','))))
                      for r in rows(work / 'run/output/candidate_pairs.tsv')]
    candidates = {eid: set(ids) for eid, ids in candidate_rows}
    references = {r['entity_id']: record(r) for r in rows(data / 'test_source1.tsv')}
    targets = {r['entity_id']: record(r) for source in (2, 3) for r in rows(data / f'test_source{source}.tsv')}
    countries = {eid: rec[4] for eid, rec in references.items()}
    methods = dict(normalized_exact=score_baseline(references, targets, candidate_rows, exact=True),
                   fuzzy_fixed_085=score_baseline(references, targets, candidate_rows),
                   cascader=read_links(work / 'run/output/matching_results.tsv'))
    slices = json.loads((data / 'slices.json').read_text())
    scores = {}
    for name, predictions in methods.items():
        result = evaluate(truth, predictions, countries, candidates)
        result['by_scenario'] = {}
        for scenario in sorted({r['scenario'] for r in slices.values()}):
            ids = [eid for eid in truth if slices[eid]['scenario'] == scenario]
            result['by_scenario'][scenario] = evaluate({eid: truth[eid] for eid in ids},
                {eid: predictions[eid] for eid in ids}, countries)
        scores[name] = result
    progress = json.loads((work / 'run/output/progress.json').read_text())
    report = dict(scope='Fictional generated stress test, not a real-world benchmark. No training or threshold selection on these records.',
                  generator=dict(seed=args.seed, references=args.references, same_name_distractors=True,
                                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                  data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob('*.tsv'))},
                  model_sha256=json.loads((ROOT / 'models/SHA256SUMS.json').read_text()),
                  environment=dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                                   dependencies={n: importlib.metadata.version(n) for n in ('numpy', 'xgboost', 'scikit-learn', 'rapidfuzz')}),
                  timing=dict(workers=args.workers, full_pipeline_seconds=elapsed,
                              prediction_seconds=progress['seconds'], references_per_second=args.references / elapsed,
                              scope='Single run; includes index construction, subprocess startup, ownership, and validation; excludes data generation and baseline scoring.'),
                  baseline_protocol='Same retrieved candidates and unique ownership. Exact requires both normalized full name and address. Fuzzy uses 0.6 name + 0.4 address token-sort similarity, fixed threshold 0.85; no tuning on this data.',
                  results=scores)
    (work / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, result in scores.items():
        print(f'{name}: macro F0.5={result["macro_f05"]:.5f}, precision={result["link_precision"]}, recall={result["link_recall"]}')
    print(f'Report: {work / "benchmark.json"}')


if __name__ == '__main__':
    main()
