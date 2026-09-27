"""Supplementary name and address blocks; built only from supplied target records."""
import argparse
from functools import lru_cache
from pathlib import Path
import re
import sqlite3
import time
from anyascii import anyascii

VERSION = 1
STOP = set('null floor near india opposite building street road avenue block nagar delhi mumbai chennai kolkata bangalore private limited'.split())

@lru_cache(maxsize=100000)
def phonetic(text):
    text = anyascii(text).lower().replace('ph','f')
    text = text.translate(str.maketrans({'v':'b','w':'b','p':'f','d':'t','g':'k','q':'k','c':'k','z':'s'}))
    return re.sub(r'(.)\1+',r'\1',re.sub('[^b-df-hj-np-tv-z]','',text).replace('h',''))


def extra_keys(rec):
    _, name, core, address, country = rec
    prefix = country+'|'
    names=sorted(set(t for t in core.split() if len(t)>=3),key=lambda t:(-len(t),t))[:6]
    addresses=sorted(set(t for t in address.split() if len(t)>=4 and not t.isdigit() and t not in STOP),key=lambda t:(-len(t),t))[:7]
    result={prefix+'f|'+t for t in names}
    result.update(prefix+'w|'+t[:10] for t in addresses)
    # Consonant keys are only retrieval hints, never a rule declaring a match.
    roman=re.findall('[a-z0-9]+',anyascii(core).lower())
    for t in roman[:6]:
        sound=phonetic(t)
        if len(sound)>=3:
            result.add(prefix+'s|'+sound[:6])
    nums=list(dict.fromkeys(str(int(n)) for n in re.findall('[0-9]+',address)))[:3]
    for n in nums:
        for t in addresses[:4]: result.add(prefix+'d|'+n+'|'+t[:8])
    return sorted(result)


def build(db, destination):
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    source=sqlite3.connect(db)
    con=sqlite3.connect(destination)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA temp_store=FILE')
    con.execute('PRAGMA cache_size=-131072')
    con.execute('CREATE TABLE blocks(key TEXT, rid INTEGER)')
    batch=[]; start=time.monotonic()
    for n,row in enumerate(source.execute('SELECT rid,eid,name,core,address,country FROM records'),1):
        batch.extend((key,row[0]) for key in extra_keys(row[1:]))
        if n%10000==0:
            con.executemany('INSERT INTO blocks VALUES (?,?)',batch); con.commit(); batch.clear()
        if n%500000==0: print(f'{n:,} records, {time.monotonic()-start:.1f}s',flush=True)
    con.executemany('INSERT INTO blocks VALUES (?,?)',batch);con.commit()
    print('Building supplemental lookup index',flush=True)
    con.execute('CREATE INDEX blocks_key ON blocks(key,rid)')
    con.execute('CREATE TABLE large_blocks AS SELECT key,COUNT(*) AS n FROM blocks GROUP BY key HAVING COUNT(*)>100')
    con.execute('CREATE TABLE metadata(version INTEGER,records INTEGER)')
    con.execute('INSERT INTO metadata VALUES (?,?)',(VERSION,n));con.commit();con.close();source.close()
    print(f'COMPLETE {n:,} records in {time.monotonic()-start:.1f}s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();build(a.db,a.out)
