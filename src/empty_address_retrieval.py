"""Retrieve name-only targets without letting addressed distractors crowd them out."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from rapidfuzz import fuzz
from robust_features import canonical_name
from pipeline import log


def name_keys(core,country):
    name=canonical_name(core,country);tokens=name.split()
    result={country+'|exact|'+hashlib.blake2b(' '.join(sorted(tokens)).encode(),digest_size=8).hexdigest()}
    result.update(country+'|word|'+token for token in tokens if len(token)>=3)
    compact=name.replace(' ','')
    if len(compact)>=6:result.add(country+'|prefix|'+compact[:12])
    return sorted(result)


def build(db,out):
    if Path(out).exists():raise FileExistsError(out)
    source=sqlite3.connect(db);con=sqlite3.connect(out);con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF')
    con.execute('CREATE TABLE blocks(key TEXT,rid INTEGER)');batch=[];count=0;started=time.monotonic()
    for rid,core,country in source.execute("SELECT rid,core,country FROM records WHERE address=''"):
        batch.extend((key,rid) for key in name_keys(core,country));count+=1
        if len(batch)>=20000:con.executemany('INSERT INTO blocks VALUES (?,?)',batch);batch.clear()
    con.executemany('INSERT INTO blocks VALUES (?,?)',batch);con.execute('CREATE INDEX block_keys ON blocks(key,rid)')
    con.execute('CREATE TABLE counts AS SELECT key,COUNT(*) AS n FROM blocks GROUP BY key');con.execute('CREATE UNIQUE INDEX count_keys ON counts(key)')
    con.execute('CREATE TABLE metadata(records INTEGER,complete INTEGER,source TEXT)');con.execute('INSERT INTO metadata VALUES (?,1,?)',(count,str(Path(db).resolve())))
    con.commit();con.close();source.close();log(f'Indexed {count:,} targets with empty addresses in {time.monotonic()-started:.1f}s')


class EmptyAddressRetrieval:
    def __init__(self,db,index):
        self.con=sqlite3.connect(db);self.con.execute('PRAGMA mmap_size=8589934592')
        self.con.execute('ATTACH DATABASE ? AS empty',(str(Path(index).resolve()),))
        row=self.con.execute('SELECT complete,source FROM empty.metadata').fetchone()
        if row!=(1,str(Path(db).resolve())):raise ValueError('Empty-address index is incomplete or belongs to another database')
        self.counts=dict(self.con.execute('SELECT key,n FROM empty.counts'))

    @lru_cache(maxsize=30000)
    def lookup(self,core,country):
        keys=[k for k in name_keys(core,country) if self.counts.get(k,0)<=1000]
        if not keys:return ()
        found=self.con.execute('SELECT eid,name,core,address,country FROM records WHERE rid IN (SELECT rid FROM empty.blocks WHERE key IN ('+','.join('?' for _ in keys)+'))',keys).fetchall()
        name=canonical_name(core,country)
        def score(row):
            other=canonical_name(row[2],country)
            return .6*fuzz.token_sort_ratio(name,other)+.25*fuzz.ratio(name.replace(' ',''),other.replace(' ',''))+.15*fuzz.token_set_ratio(name,other)
        return tuple(sorted(found,key=lambda r:(-score(r),r[0]))[:30])

    def candidates(self,a,exclude=(),limit=8):
        seen=set(exclude)
        return [r for r in self.lookup(a[2],a[4]) if r[0] not in seen][:limit]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.db,a.out)
