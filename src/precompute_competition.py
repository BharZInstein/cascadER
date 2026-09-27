"""Cache each target's closest reference records for fast bounded-memory inference."""
import argparse,json,sqlite3,time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
import numpy as np
from competition_features import ReferenceCompetition
from pipeline import log

DTYPE=np.dtype([('ids','<u4',(2,)),('values','<f4',(2,4)),('counts','<u4',(2,)),('owner','<u4')])
MISSING=np.iinfo(np.uint32).max
COMP=None

def target_code(eid):
    if eid[:3] not in ('S2-','S3-'):raise ValueError(eid)
    value=int(eid[3:])
    if not 0<=value<2**30:raise ValueError(eid)
    return value+(2**30 if eid.startswith('S3-') else 0)

def initialize(path):
    global COMP
    COMP=ReferenceCompetition(path)

def process_batch(batch):
    keys=np.empty(len(batch),dtype=np.uint32);values=np.zeros(len(batch),dtype=DTYPE)
    values['ids']=MISSING;values['owner']=MISSING
    for i,b in enumerate(batch):
        keys[i]=target_code(b[0]);scored,nc,ac,owner=COMP.lookup(b)
        values['counts'][i]=(nc,ac)
        if owner:values['owner'][i]=int(owner[3:])
        for j,(score,eid,name,addr,num) in enumerate(scored):
            values['ids'][i,j]=int(eid[3:]);values['values'][i,j]=(score,name,addr,num)
    return keys.tobytes(),values.tobytes(),len(batch)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',required=True);p.add_argument('--references',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);con=sqlite3.connect(a.db)
    iterator=iter(islice(con.execute('SELECT eid,name,core,address,country FROM records'),a.limit));count=0;start=time.monotonic()
    with open(out/'keys.raw','wb') as kf,open(out/'values.raw','wb') as vf,ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.references,)) as executor:
        pending=deque()
        def submit():
            batch=list(islice(iterator,2048))
            if batch:pending.append(executor.submit(process_batch,batch))
        for _ in range(a.workers*2):submit()
        while pending:
            keys,values,n=pending.popleft().result();kf.write(keys);vf.write(values);count+=n;submit()
            if count//100000!=(count-n)//100000:log(f'Cached {count:,} targets; {count/(time.monotonic()-start):.1f}/sec')
    keys=np.memmap(out/'keys.raw',dtype=np.uint32,mode='r');values=np.memmap(out/'values.raw',dtype=DTYPE,mode='r')
    order=np.argsort(keys);sorted_keys=keys[order]
    if len(np.unique(sorted_keys))!=count:raise ValueError('Duplicate target IDs')
    np.save(out/'keys.npy',sorted_keys);np.save(out/'values.npy',values[order])
    def identity(path):
        path=Path(path).resolve();stat=path.stat();return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    (out/'metadata.json').write_text(json.dumps(dict(complete=True,records=count,db=identity(a.db),references=identity(a.references),limit=a.limit,seconds=time.monotonic()-start),indent=2))
    del keys,values
    (out/'keys.raw').unlink();(out/'values.raw').unlink();log(f'COMPLETE {count:,} targets in {time.monotonic()-start:.1f}s')

if __name__=='__main__':main()
