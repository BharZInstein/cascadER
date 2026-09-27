"""Export a standalone interactive report from a completed run with saved features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from group_policy import apply_policies, chosen_pairs
from pipeline import normalize, rows

EVIDENCE = ['name_sorted', 'address_sorted', 'first_stage_probability',
            'name_character_tfidf_cosine', 'address_character_tfidf_cosine',
            'first_number_equal', 'reference_score_margin']


def build_report(work, max_references=2000):
    state = json.loads((work / 'run.json').read_text())
    progress = json.loads((work / 'output/progress.json').read_text())
    if not progress.get('complete') or not progress.get('unique_assignment', {}).get('complete'):
        raise ValueError('Run prediction and unique assignment before exporting a report')
    if not progress['signature']['save_features']:
        raise ValueError('Run scripts/match.py in a new directory with --save-features')
    if progress['rows'] > max_references:
        raise ValueError(f'Review export is limited to {max_references} references; use a smaller catalog or explicitly raise --max-references')
    model_dir = Path(state['model'])
    for name, digest in state['artifacts'].items():
        if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Model artifacts changed since prediction')
    source_rows = []
    for original in state['inputs']:
        path = Path(original['path'])
        stat = path.stat()
        if stat.st_size != original['size'] or stat.st_mtime_ns != original['mtime_ns']:
            raise ValueError('Source records changed since prediction')
        source_rows.append(list(rows(path)))
    references = source_rows[0]
    targets = {r['entity_id']: r for group in source_rows[1:] for r in group}
    config = json.loads((model_dir / 'model_config.json').read_text())
    n_pairs, n_features = progress['pairs'], len(config['feature_names'])
    feature_path = work / 'output/pair_features.f32'
    if feature_path.stat().st_size != n_pairs * n_features * 4:
        raise ValueError('Feature file length does not match the prediction checkpoint')
    features = np.memmap(feature_path, mode='r', dtype='f4', shape=(n_pairs, n_features)) if n_pairs else np.empty((0, n_features), dtype='f4')
    model = XGBClassifier()
    model.load_model(model_dir / config['model_file'])
    model.set_params(n_jobs=2)
    scores = model.predict_proba(features)[:, 1] if n_pairs else np.empty(0)
    final = {r['source1_entity_id']: set(filter(None, r['matched_entity_ids'].split(',')))
             for r in rows(work / 'output/matching_results.tsv')}
    candidate_rows = list(rows(work / 'output/candidate_pairs.tsv'))
    ids = [list(filter(None, r['candidate_entity_ids'].split(','))) for r in candidate_rows]
    if [r['entity_id'] for r in references] != [r['source1_entity_id'] for r in candidate_rows] or set(final) != {r['entity_id'] for r in references}:
        raise ValueError('Output reference coverage or order differs from input')
    if sum(map(len, ids)) != n_pairs:
        raise ValueError('Candidate count differs from feature count')
    groups = np.repeat(np.arange(len(ids)), [len(group) for group in ids])
    flat_ids = np.array([eid for group in ids for eid in group])
    countries = np.array([normalize(r['country']) for r in references])
    chosen = apply_policies(scores, groups, countries, config, flat_ids)
    owners = {target: eid for eid, matches in final.items() for target in matches}
    if len(owners) != sum(map(len, final.values())):
        raise ValueError('Saved predictions assign a target to more than one reference')
    expected_owners = {str(target): references[int(group)]['entity_id'] for target, group in zip(flat_ids[chosen], groups[chosen])}
    if owners != expected_owners:
        raise ValueError('Rescored decisions differ from the saved predictions')
    items, offset = [], 0
    for row, candidates in zip(references, ids):
        eid = row['entity_id']
        values = scores[offset:offset + len(candidates)]
        policy = config.get('country_policies', {}).get(normalize(row['country']), config)
        before_ownership = chosen_pairs(values, policy['threshold'], np.zeros(len(values), dtype=int),
                                        1, rescue=policy.get('rescue_threshold'))
        records = []
        for i, target in enumerate(candidates):
            selected = target in final[eid]
            owner = owners.get(target)
            if selected:
                decision = 'accepted' if values[i] >= policy['threshold'] else 'accepted_by_fallback'
            elif before_ownership[i] and owner and owner != eid:
                decision = 'assigned_to_other_reference'
            else:
                decision = 'below_policy_threshold'
            records.append(dict(record=targets[target], score=float(values[i]), decision=decision,
                                owner=owner, selected=selected,
                                evidence={name: float(features[offset+i, config['feature_names'].index(name)])
                                          for name in EVIDENCE}))
        records.sort(key=lambda r: (-r['score'], r['record']['entity_id']))
        # Keep all accepted matches, even when outside the top ten candidates.
        displayed = [r for i, r in enumerate(records) if i < 10 or r['selected']]
        flags = []
        if not row['business_address'].strip():
            flags.append('Reference address is missing')
        if any(r['selected'] and not r['record']['business_address'].strip() for r in records):
            flags.append('Accepted target has no address')
        import re
        first_number = re.search(r'\d+', row['business_address'])
        if first_number and any(r['selected'] and (other := re.search(r'\d+', r['record']['business_address']))
                                and int(first_number.group()) != int(other.group()) for r in records):
            flags.append('Accepted pair has different first address numbers')
        if any(abs(r['score'] - policy['threshold']) <= .05 for r in records):
            flags.append('Candidate score is within 0.05 of the threshold')
        if any(r['decision'] == 'assigned_to_other_reference' for r in records):
            flags.append('A candidate was assigned to another reference')
        items.append(dict(reference=row, matches=len(final[eid]), candidate_count=len(candidates),
                          policy=policy, flags=flags, candidates=displayed))
        offset += len(candidates)
    return dict(schema_version=1, model=dict(trees=config['trees'], depth=config['depth'], features=n_features,
                                            sha256=state['artifacts'][config['model_file']]),
                summary=dict(references=len(items), candidates=n_pairs, matches=sum(len(v) for v in final.values()),
                             unmatched=sum(not item['matches'] for item in items), flagged=sum(bool(item['flags']) for item in items)),
                evidence_note='Feature values are inputs to the classifier, not causal explanations or calibrated probabilities.',
                items=items)


def render_html(report):
    # Escaping '<' prevents user-supplied text from closing the JSON script element.
    payload = json.dumps(report, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return (ROOT / 'web/review.html').read_text().replace('__REPORT_JSON__', payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--max-references', type=int, default=2000)
    args = parser.parse_args()
    report = build_report(args.work_dir.resolve(), args.max_references)
    destination = args.work_dir / 'review.html'
    destination.write_text(render_html(report))
    print(f'Open in your browser: {destination.resolve()}')
    print('The report embeds the supplied record text. Review decisions can be exported from the page.')


if __name__ == '__main__':
    main()
