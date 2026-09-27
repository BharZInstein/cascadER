"""Merge disjoint anchor caches with weighted sampling of easy training negatives."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import numpy as np
from pipeline import log


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    metas=[json.loads((Path(path)/'metadata.json').read_text()) for path in a.cache]
    names=metas[0]['feature_names'];assert all(m['complete'] and m['feature_names']==names for m in metas)
    seen=set();anchors=[];pairs=0;rng=np.random.default_rng(20260927);prob_index=names.index('first_stage_probability')
    data=out/'data';data.mkdir()
    with ExitStack() as stack:
        fs={name:stack.enter_context((out/name).open('wb')) for name in ('features.f32','labels.i1','groups.i4','targets.s32','weights.f32')}
        source_files={name:stack.enter_context((data/name).open('wb')) for name in ('train_source1.tsv','train_ground_truth.tsv')}
        for cache,meta in zip(map(Path,a.cache),metas):
            for name,f in source_files.items():
                with (Path(meta['data'])/name).open('rb') as source:
                    header=source.readline()
                    if f.tell()==0:f.write(header)
                    for chunk in iter(lambda:source.read(1024*1024),b''):f.write(chunk)
            x=np.memmap(cache/'features.f32',dtype='f4',mode='r',shape=(meta['pairs'],len(names)))
            y=np.memmap(cache/'labels.i1',dtype='i1',mode='r');t=np.memmap(cache/'targets.s32',dtype='S32',mode='r')
            offset=0
            for row in meta['anchors']:
                if row['id'] in seen:raise ValueError('Overlapping anchor samples')
                seen.add(row['id']);stop=offset+row['candidates'];xx=x[offset:stop];yy=y[offset:stop]
                weight=np.ones(len(yy),dtype=np.float32)
                if row['split']<6:
                    required=(yy==1)|(xx[:,prob_index]>=.001)
                    take=required|(rng.random(len(yy))<.05)
                    weight[~required]=20
                else:take=np.ones(len(yy),dtype=bool)
                count=int(take.sum());group=len(anchors)
                xx[take].tofile(fs['features.f32']);yy[take].tofile(fs['labels.i1'])
                np.full(count,group,dtype=np.int32).tofile(fs['groups.i4']);t[offset:stop][take].tofile(fs['targets.s32']);weight[take].tofile(fs['weights.f32'])
                anchors.append(dict(row,candidates=count,original_candidates=row['candidates']));pairs+=count;offset=stop
            log(f'Merged {cache}: {len(anchors):,} anchors, {pairs:,} retained pairs')
    meta=dict(metas[0],anchors=anchors,pairs=pairs,data=str(data.resolve()),
              merged_caches=list(map(str,map(Path,a.cache))),
              negative_sampling=dict(training_only=True,retain_all_retrieved_positives=True,first_stage_probability_minimum=.001,
                                     easy_negative_sample_probability=.05,easy_negative_weight=20,random_seed=20260927,
                                     evaluation_pairs_unchanged=True))
    (out/'metadata.json').write_text(json.dumps(meta));log('Merged training cache complete')


if __name__=='__main__':main()
