"""Append corpus-weighted text comparisons to the frozen candidate feature cache."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from pipeline import rows,record,log
from robust_features import RobustFeatures,ROBUST_NAMES

CON=X=IDS=META=ANCHORS=ROBUST=None


def initialize(cache,lookup,weights):
    global CON,X,IDS,META,ANCHORS,ROBUST
    cache=Path(cache);META=json.loads((cache/'metadata.json').read_text())
    CON=sqlite3.connect(META['db']);CON.execute('PRAGMA mmap_size=8589934592')
    CON.execute('ATTACH DATABASE ? AS lookup',(str(Path(lookup).resolve()),))
    X=np.memmap(cache/'features.f32',dtype='f4',mode='r',shape=(META['pairs'],len(META['feature_names'])))
    IDS=np.memmap(cache/'targets.s32',dtype='S32',mode='r')
    ANCHORS={r['entity_id']:record(r) for r in rows(Path(META['data'])/'train_source1.tsv')}
    ROBUST=RobustFeatures(weights)


def process(batch):
    start=batch[0][1];stop=batch[-1][2];ids=list(dict.fromkeys(v.decode() for v in IDS[start:stop]))
    lookup={r[0]:r for r in CON.execute('SELECT r.eid,r.name,r.core,r.address,r.country FROM lookup.mapping m JOIN records r ON r.rid=m.rid WHERE m.eid IN ('+','.join('?' for _ in ids)+')',ids)} if ids else {}
    anchors=[];found=[];probabilities=[];prob_index=META['feature_names'].index('first_stage_probability')
    for i,start,stop in batch:
        anchors.append(ANCHORS[META['anchors'][i]['id']]);found.append([lookup[v.decode()] for v in IDS[start:stop]])
        probabilities.append(X[start:stop,prob_index])
    extra=ROBUST.batch(anchors,found,probabilities)
    arrays=[np.concatenate((X[start:stop],features),axis=1) for (_,start,stop),features in zip(batch,extra)]
    return np.concatenate(arrays,axis=0).astype(np.float32).tobytes(),len(batch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','lookup','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((Path(a.cache)/'metadata.json').read_text());work=[];offset=0
    for i,row in enumerate(meta['anchors'][:a.limit]):work.append((i,offset,offset+row['candidates']));offset+=row['candidates']
    iterator=iter(work);started=time.monotonic();done=0
    with (out/'features.f32').open('wb') as f,ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.cache,a.lookup,a.weights)) as pool:
        pending=deque()
        def submit():
            batch=list(islice(iterator,64))
            if batch:pending.append(pool.submit(process,batch))
        for _ in range(a.workers*2):submit()
        while pending:
            blob,count=pending.popleft().result();f.write(blob);done+=count;submit()
            if done//2000!=(done-count)//2000:log(f'Augmented {done:,}/{len(work):,} anchors; {done/(time.monotonic()-started):.1f}/sec')
    for name,dtype in [('labels.i1','i1'),('groups.i4','i4'),('targets.s32','S32')]:
        if a.limit:np.memmap(Path(a.cache)/name,dtype=dtype,mode='r')[:offset].tofile(out/name)
        else:(out/name).symlink_to((Path(a.cache)/name).resolve())
    meta.update(anchors=meta['anchors'][:a.limit],pairs=offset,feature_names=meta['feature_names']+ROBUST_NAMES,
                robust_features=True,text_weights=str(Path(a.weights).resolve()),robust_augmentation_seconds=time.monotonic()-started)
    (out/'metadata.json').write_text(json.dumps(meta));log(f'COMPLETE {offset:,} pairs in {meta["robust_augmentation_seconds"]:.1f}s')


if __name__=='__main__':main()
