"""Assign each predicted target to its highest-scoring reference; retain raw outputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from pipeline import log


def target_code(eid):
    if eid[:3] not in ('S2-', 'S3-'):
        raise ValueError(eid)
    value = int(eid[3:])
    if not 0 <= value < 2**30:
        raise ValueError(eid)
    return value + (2**30 if eid.startswith('S3-') else 0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    progress=json.loads((out/'progress.json').read_text())
    if progress.get('unique_assignment',{}).get('complete'):
        log('Unique target assignment already complete');return
    if not progress.get('complete') or not progress['signature'].get('save_scores'):raise ValueError('Completed prediction with saved scores required')
    original=out/'matching_results.raw.tsv';matching=out/'matching_results.tsv'
    if not original.exists():matching.rename(original)
    raw_prob=out/'match_probabilities.raw.f32';score_path=out/'match_probabilities.f32'
    if not raw_prob.exists():score_path.rename(raw_prob)
    n=progress['matches'];scores=np.memmap(raw_prob,dtype=np.float32,mode='r') if raw_prob.stat().st_size else np.empty(0,dtype=np.float32)
    if len(scores)!=n:raise ValueError('Score count disagrees with prediction checkpoint')
    codes=np.empty(n,dtype=np.uint32);groups=np.empty(n,dtype=np.int32);offset=0;row_count=0
    with original.open() as f:
        for row_count,row in enumerate(csv.DictReader(f,delimiter='\t'),1):
            ids=list(filter(None,row['matched_entity_ids'].split(',')));stop=offset+len(ids)
            codes[offset:stop]=[target_code(eid) for eid in ids];groups[offset:stop]=row_count-1;offset=stop
    if offset!=n or row_count!=progress['rows']:raise ValueError('Raw output counts changed')
    order=np.lexsort((groups,-scores,codes));ordered=codes[order]
    first=np.r_[True,ordered[1:]!=ordered[:-1]] if n else np.zeros(0,dtype=bool)
    keep=np.zeros(n,dtype=bool);keep[order[first]]=True;offset=0
    temporary=out/'matching_results.unique.partial'
    with original.open() as f,temporary.open('w') as g:
        g.write('source1_entity_id\tmatched_entity_ids\n')
        for row in csv.DictReader(f,delimiter='\t'):
            ids=list(filter(None,row['matched_entity_ids'].split(',')));stop=offset+len(ids)
            selected=[eid for eid,take in zip(ids,keep[offset:stop]) if take]
            g.write(row['source1_entity_id']+'\t'+','.join(selected)+'\n');offset=stop
    scores[keep].tofile(score_path);temporary.replace(matching)
    report=dict(complete=True,raw_links=n,unique_links=int(keep.sum()),removed_duplicate_assignments=int(n-keep.sum()),
                policy='Highest model probability per target; ties use earliest source1 input row. Raw predictions preserved.')
    progress.update(raw_matches=n,matches=int(keep.sum()),unique_assignment=report)
    progress['bytes'][0]=matching.stat().st_size
    progress['bytes'][3 if progress['signature']['save_features'] else 2]=score_path.stat().st_size
    temporary=out/'progress.json.tmp';temporary.write_text(json.dumps(progress,indent=2));temporary.replace(out/'progress.json')
    (out/'unique_assignment.json').write_text(json.dumps(report,indent=2));log(str(report))

if __name__=='__main__':main()
