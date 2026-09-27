"""Select a deterministic, reproducible range of anchor hashes without loading full data."""
import argparse
from pathlib import Path
from pipeline import bucket, rows, write_rows, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--hash-start', type=int, default=25)
    p.add_argument('--hash-stop', type=int, default=125)
    args = p.parse_args()
    if not 0 <= args.hash_start < args.hash_stop <= 10000:
        p.error('Require 0 <= hash-start < hash-stop <= 10000')
    source, out = Path(args.train_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [r for r in rows(source/'train_ground_truth.tsv')
              if args.hash_start <= bucket(r['source1_entity_id']) < args.hash_stop]
    ids = {r['source1_entity_id'] for r in labels}
    write_rows(out/'train_ground_truth.tsv', labels, ['source1_entity_id', 'matched_entity_ids'])
    write_rows(out/'train_source1.tsv', (r for r in rows(source/'train_source1.tsv') if r['entity_id'] in ids))
    log(f'Selected {len(ids):,} anchors in hash range [{args.hash_start}, {args.hash_stop})')


if __name__ == '__main__':
    main()
