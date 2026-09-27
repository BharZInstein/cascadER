"""Build supervised pair features with bounded parallel batches and no label-based retrieval."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
from pathlib import Path
import time
import numpy as np
from pipeline import connect,candidates,features,record,rows,bucket,FEATURE_NAMES,RETRIEVAL_VERSION,log

CON=TOP_K=MAX_BLOCK=None

def initialize(db,top_k,max_block):
    global CON,TOP_K,MAX_BLOCK
    CON=connect(db)
    if CON.extra_blocks is None: raise ValueError('Supplemental index is required')
    TOP_K,MAX_BLOCK=top_k,max_block

def batch_features(batch):
    xs=[];ys=[];groups=[];targets=[];counts=[]
    for group,row,truth in batch:
        a=record(row);found=candidates(CON,a,TOP_K,MAX_BLOCK)
        counts.append((len(found),len(truth&{b[0] for b in found})))
        for b in found:
            xs.append(features(a,b,CON));ys.append(b[0] in truth);groups.append(group);targets.append(b[0])
    return np.asarray(xs,dtype=np.float32).tobytes(),np.asarray(ys,dtype=np.int8).tobytes(),np.asarray(groups,dtype=np.int32).tobytes(),np.asarray(targets,dtype='S32').tobytes(),counts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--db',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--top-k',type=int,default=60);p.add_argument('--max-block',type=int,default=1000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    truth={r['source1_entity_id']:set(filter(None,r['matched_entity_ids'].split(','))) for r in rows(Path(a.data)/'train_ground_truth.tsv')}
    seen=set()
    for links in truth.values():
        if seen&links: raise ValueError('Shared labeled target across anchors')
        seen.update(links)
    anchor_rows=list(rows(Path(a.data)/'train_source1.tsv'))
    assert {r['entity_id'] for r in anchor_rows}==set(truth)
    metadata=[dict(id=r['entity_id'],country=r['country'],true_count=len(truth[r['entity_id']]),split=bucket('split:'+r['entity_id'],10)) for r in anchor_rows]
    iterator=iter((i,r,truth[r['entity_id']]) for i,r in enumerate(anchor_rows))
    start=time.monotonic();done=pairs=0;counts=[]
    from contextlib import ExitStack
    with ExitStack() as stack:
        files=[stack.enter_context(open(out/name,'wb')) for name in ('features.f32','labels.i1','groups.i4','targets.s32')]
        executor=stack.enter_context(ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.db,a.top_k,a.max_block)))
        pending=deque()
        def submit():
            batch=list(islice(iterator,128))
            if batch:pending.append(executor.submit(batch_features,batch))
        for _ in range(a.workers*2):submit()
        while pending:
            result=pending.popleft().result()
            for f,blob in zip(files,result[:4]):f.write(blob)
            counts.extend(result[4]);done+=len(result[4]);pairs+=len(result[1]);submit()
            if done//2000!=(done-len(result[4]))//2000:log(f'{done:,}/{len(metadata):,} anchors; {done/(time.monotonic()-start):.1f}/sec; {pairs:,} pairs')
    for item,(n,recalled) in zip(metadata,counts):item.update(candidates=n,recalled=recalled)
    report=dict(complete=True,anchors=metadata,pairs=pairs,feature_names=FEATURE_NAMES,retrieval_version=RETRIEVAL_VERSION,
                top_k=a.top_k,max_block=a.max_block,data=str(Path(a.data).resolve()),db=str(Path(a.db).resolve()),seconds=time.monotonic()-start)
    (out/'metadata.json').write_text(json.dumps(report))
    log(f'COMPLETE: {pairs:,} pairs in {time.monotonic()-start:.1f}s')

if __name__=='__main__':main()
