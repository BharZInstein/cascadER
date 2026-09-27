"""Append text rarity, candidate context, and reference competition in bounded batches."""
import argparse,json,sqlite3,time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
import numpy as np
from pipeline import connect,rows,record,log
from rarity_features import rarity_features,RARITY_NAMES
from context_features import contextual_features,CONTEXT_NAMES,posterior_features,POSTERIOR_NAMES,candidate_similarities
from cached_competition import CachedCompetition
from competition_features import COMPETITION_NAMES

CON=X=IDS=META=ANCHORS=COMP=POST=None

def initialize(cache,lookup,competition,posterior):
    global CON,X,IDS,META,ANCHORS,COMP,POST
    cache=Path(cache);META=json.loads((cache/'metadata.json').read_text())
    CON=connect(META['db']);CON.execute('ATTACH DATABASE ? AS lookup',(str(Path(lookup).resolve()),))
    X=np.memmap(cache/'features.f32',dtype=np.float32,mode='r',shape=(META['pairs'],len(META['feature_names'])))
    IDS=np.memmap(cache/'targets.s32',dtype='S32',mode='r')
    ANCHORS={r['entity_id']:record(r) for r in rows(Path(META['data'])/'train_source1.tsv')}
    COMP=CachedCompetition(competition) if competition else None
    POST=np.load(Path(posterior)/'crossfit_scores.npy',mmap_mode='r') if posterior else None

def augment_batch(batch):
    start=batch[0][1];stop=batch[-1][2];ids=list(dict.fromkeys(v.decode() for v in IDS[start:stop]))
    lookup={r[0]:r for r in CON.execute('SELECT r.eid,r.name,r.core,r.address,r.country FROM lookup.mapping m JOIN records r ON r.rid=m.rid WHERE m.eid IN ('+','.join('?' for _ in ids)+')',ids)} if ids else {}
    output=[]
    for i,start,stop in batch:
        a=ANCHORS[META['anchors'][i]['id']];found=[lookup[v.decode()] for v in IDS[start:stop]];base=X[start:stop]
        rare=np.asarray([rarity_features(a,b,CON) for b in found],dtype=np.float32).reshape(len(found),len(RARITY_NAMES))
        similarities=candidate_similarities(found) if len(found)>1 else None
        blocks=[base,rare,contextual_features(a,found,base,similarities)]
        if POST is not None:blocks.append(posterior_features(found,POST[start:stop],similarities))
        if COMP is not None:blocks.append(np.asarray([COMP.features(a,b) for b in found],dtype=np.float32).reshape(len(found),len(COMPETITION_NAMES)))
        output.append(np.concatenate(blocks,axis=1))
    return np.concatenate(output,axis=0).astype(np.float32).tobytes(),len(batch)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--lookup',required=True);p.add_argument('--competition');p.add_argument('--posterior-dir');p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);meta=json.loads((Path(a.cache)/'metadata.json').read_text())
    work=[];offset=0
    for i,row in enumerate(meta['anchors']):work.append((i,offset,offset+row['candidates']));offset+=row['candidates']
    iterator=iter(work);start=time.monotonic();done=0
    with open(out/'features.f32','wb') as f,ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.cache,a.lookup,a.competition,a.posterior_dir)) as executor:
        pending=deque()
        def submit():
            batch=list(islice(iterator,64))
            if batch:pending.append(executor.submit(augment_batch,batch))
        for _ in range(a.workers*2):submit()
        while pending:
            blob,n=pending.popleft().result();f.write(blob);done+=n;submit()
            if done//2000!=(done-n)//2000:log(f'Augmented {done:,}/{len(work):,}; {done/(time.monotonic()-start):.1f}/sec')
    for name in ('labels.i1','groups.i4','targets.s32'):(out/name).symlink_to((Path(a.cache)/name).resolve())
    meta['feature_names']+=RARITY_NAMES+CONTEXT_NAMES;meta.update(rarity_features=True,contextual_features=True)
    if a.posterior_dir:meta['feature_names']+=POSTERIOR_NAMES;meta['posterior_dir']=str(Path(a.posterior_dir).resolve())
    if a.competition:meta['feature_names']+=COMPETITION_NAMES;meta['competition_features']=True;meta['competition_cache']=str(Path(a.competition).resolve())
    meta['augmentation_seconds']=time.monotonic()-start;(out/'metadata.json').write_text(json.dumps(meta));log('Augmented cache COMPLETE')

if __name__=='__main__':main()
