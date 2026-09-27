"""Build the target ID lookup used by parallel training feature augmentation."""
import argparse
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    if Path(a.out).exists(): raise FileExistsError(a.out)
    with sqlite3.connect(a.out) as con:
        con.execute('ATTACH DATABASE ? AS source', (str(Path(a.db).resolve()),))
        con.execute('CREATE TABLE mapping AS SELECT eid,rid FROM source.records')
        con.execute('CREATE UNIQUE INDEX entity_ids ON mapping(eid)')


if __name__ == '__main__': main()
