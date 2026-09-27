"""Index the supplied deduplicated reference source for competing-match checks."""
import argparse,hashlib,sqlite3,time
from pathlib import Path
from pipeline import rows,record,log

FRENCH_LEGAL={'sa','sarl','sas','sasu','eurl','sci'}

def comparison_name(text,country):
    return ' '.join(t for t in text.split() if t not in FRENCH_LEGAL) if country=='france' else text

def name_fingerprint(text,country):
    return fingerprint(comparison_name(text,country),country)

def fingerprint(text,country):
    return country+'|'+hashlib.blake2b(' '.join(sorted(text.split())).encode(),digest_size=12).hexdigest()

def build(source,destination):
    if Path(destination).exists():raise FileExistsError(destination)
    con=sqlite3.connect(destination);con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF')
    con.execute('CREATE TABLE refs(eid TEXT,name TEXT,core TEXT,address TEXT,country TEXT,namekey TEXT,addresskey TEXT)')
    batch=[];start=time.monotonic()
    for n,row in enumerate(rows(source),1):
        r=record(row);batch.append((*r,name_fingerprint(r[2],r[4]),fingerprint(r[3],r[4]) if r[3] else ''))
        if n%10000==0:con.executemany('INSERT INTO refs VALUES (?,?,?,?,?,?,?)',batch);con.commit();batch.clear()
    con.executemany('INSERT INTO refs VALUES (?,?,?,?,?,?,?)',batch);con.commit()
    con.execute('CREATE INDEX names ON refs(namekey)');con.execute('CREATE INDEX addresses ON refs(addresskey)')
    con.execute('CREATE TABLE metadata(records INTEGER,complete INTEGER)');con.execute('INSERT INTO metadata VALUES (?,1)',(n,));con.commit();con.close()
    log(f'Reference index complete: {n:,} in {time.monotonic()-start:.1f}s')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.source,a.out)
