"""Strict output validation without retaining all candidate lists in memory."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time

from pipeline import log


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_id_list(row, expected_id, label):
    if row is None or len(row) != 2 or row[0] != expected_id:
        raise ValueError(f'{label}: expected exactly one row for {expected_id} in source order; got {row}')
    ids = row[1].split(',') if row[1] else []
    if len(ids) != len(set(ids)):
        raise ValueError(f'{label}: duplicate target for {expected_id}')
    return set(ids)


def validate(test_dir, output, report, unique_targets=False):
    started = time.monotonic()
    test_dir, output = Path(test_dir), Path(output)
    valid = set()
    for source in (2, 3):
        with open(test_dir/f'test_source{source}.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                eid = line.partition('\t')[0]
                if not eid.startswith(f'S{source}-'):
                    raise ValueError(f'Unexpected ID in source {source}: {eid!r}')
                if eid in valid:
                    raise ValueError(f'Duplicate source ID {eid}')
                valid.add(eid)
        log(f'Loaded {len(valid):,} valid target IDs')
    matching, candidate = output/'matching_results.tsv', output/'candidate_pairs.tsv'
    seen, countries = set(), Counter()
    assigned=set() if unique_targets else None
    country_matches, country_candidates, country_empty = Counter(), Counter(), Counter()
    n = total_matches = total_candidates = empty_matches = empty_candidates = 0
    max_candidates = 0
    with open(test_dir/'test_source1.tsv', encoding='utf-8', newline='') as sf, \
            open(matching, encoding='utf-8', newline='') as mf, \
            open(candidate, encoding='utf-8', newline='') as cf:
        source_rows = csv.DictReader(sf, delimiter='\t')
        matches, candidates = csv.reader(mf, delimiter='\t'), csv.reader(cf, delimiter='\t')
        if next(matches, None) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Incorrect matching header')
        if next(candidates, None) != ['source1_entity_id', 'candidate_entity_ids']:
            raise ValueError('Incorrect candidate header')
        for row, mrow, crow in zip_longest(source_rows, matches, candidates):
            if row is None:
                raise ValueError('Extra output rows')
            eid = row['entity_id']
            if eid in seen:
                raise ValueError(f'Duplicate reference ID {eid}')
            seen.add(eid)
            mids, cids = read_id_list(mrow, eid, 'matching'), read_id_list(crow, eid, 'candidate')
            if not mids <= cids:
                raise ValueError(f'Matches outside candidate set for {eid}')
            if not cids <= valid:
                raise ValueError(f'Unknown/self/source1 target IDs for {eid}: {cids-valid}')
            if assigned is not None:
                if assigned&mids:raise ValueError(f'Target assigned to multiple references for {eid}')
                assigned.update(mids)
            n += 1
            countries[row['country']] += 1
            country_matches[row['country']] += len(mids)
            country_candidates[row['country']] += len(cids)
            country_empty[row['country']] += not mids
            total_matches += len(mids)
            total_candidates += len(cids)
            empty_matches += not mids
            empty_candidates += not cids
            max_candidates = max(max_candidates, len(cids))
            if n % 200000 == 0:
                log(f'Validated {n:,} rows')
    result = dict(status='PASS', rows=n, countries=dict(countries), valid_target_ids=len(valid),
                  by_country={country: dict(rows=count, predicted_links=country_matches[country],
                                            candidate_pairs=country_candidates[country],
                                            empty_match_rows=country_empty[country])
                              for country, count in countries.items()},
                  predicted_links=total_matches, candidate_pairs=total_candidates,
                  mean_candidates=total_candidates/max(1, n), max_candidates=max_candidates,
                  empty_match_rows=empty_matches, empty_candidate_rows=empty_candidates,
                  checks=['exact coverage and source order', 'headers', 'unique anchors',
                          'unique targets per row', 'valid target IDs', 'matches subset of candidates'],
                  sha256={matching.name: file_hash(matching), candidate.name: file_hash(candidate)},
                  seconds=time.monotonic()-started)
    if unique_targets:result['checks'].append('unique target ownership across all references')
    Path(report).write_text(json.dumps(result, indent=2))
    log(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--output', default='output')
    p.add_argument('--report', default='reports/submission_validation.json')
    p.add_argument('--unique-targets',action='store_true')
    args = p.parse_args()
    validate(args.test_dir, args.output, args.report,args.unique_targets)
