"""Bounded-memory, resumable prediction using CPU worker processes."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import time

import numpy as np
from pipeline import (connect, candidates, features, record, rows, log,
                      FEATURE_NAMES, RETRIEVAL_VERSION, INDEX_VERSION)
from context_features import contextual_features,posterior_features,CONTEXT_NAMES,POSTERIOR_NAMES,candidate_similarities
from rarity_features import rarity_features,RARITY_NAMES
from cached_competition import CachedCompetition
from competition_features import COMPETITION_NAMES
from robust_features import RobustFeatures,ROBUST_NAMES
from empty_address_retrieval import EmptyAddressRetrieval
from supplementary_matching import additional_features

CON = MODEL = CONFIG = None
BASE_MODEL = None
COMPETITION = None
ROBUST = None
EXTRA = None
SAVE_FEATURES = False
SAVE_SCORES = False


def initialize(db, model_dir, save_features=False, competition_cache=None, save_scores=False, empty_index=None):
    global CON, MODEL, CONFIG, SAVE_FEATURES, BASE_MODEL, COMPETITION, SAVE_SCORES, ROBUST, EXTRA
    SAVE_FEATURES = save_features
    SAVE_SCORES = save_scores
    from xgboost import XGBClassifier
    CON = connect(db)
    metadata = CON.execute('SELECT complete,retrieval_version FROM metadata').fetchone()
    if metadata != (1, INDEX_VERSION):
        raise ValueError('Incomplete or incompatible index')
    CONFIG = json.loads((Path(model_dir)/'model_config.json').read_text())
    expected=list(FEATURE_NAMES)
    if CONFIG.get('rarity_features'):expected+=RARITY_NAMES
    if CONFIG.get('contextual_features'):expected+=CONTEXT_NAMES
    if CONFIG.get('posterior_features'):expected+=POSTERIOR_NAMES
    if CONFIG.get('competition_features'):expected+=COMPETITION_NAMES
    if CONFIG.get('robust_features'):expected+=ROBUST_NAMES
    if CONFIG['feature_names'] != expected or CONFIG['retrieval_version'] != RETRIEVAL_VERSION:
        raise ValueError('Incompatible model')
    if CONFIG.get('supplemental_index_required') and CON.extra_blocks is None:
        raise ValueError('Required supplemental index is missing or incomplete')
    if CONFIG.get('model_kind') == 'linear_svm':
        MODEL = json.loads((Path(model_dir)/'linear_svm.json').read_text())
    elif CONFIG.get('model_kind')=='lightgbm':
        from lightgbm import Booster
        MODEL=Booster(model_file=str(Path(model_dir)/CONFIG.get('model_file','model.txt')))
    else:
        MODEL = XGBClassifier()
        MODEL.load_model(Path(model_dir)/'model.ubj')
        MODEL.set_params(n_jobs=1)
    BASE_MODEL=None
    if CONFIG.get('posterior_features'):
        BASE_MODEL=XGBClassifier()
        BASE_MODEL.load_model(Path(model_dir)/'base_model.ubj')
        BASE_MODEL.set_params(n_jobs=1)
    COMPETITION=None
    if CONFIG.get('competition_features'):
        if not competition_cache:raise ValueError('Competition cache is required')
        COMPETITION=CachedCompetition(competition_cache)
        if COMPETITION.metadata['limit'] is not None:raise ValueError('Limited competition cache')
        if COMPETITION.metadata['db']!=identity(db):raise ValueError('Competition cache belongs to different targets')
    ROBUST=RobustFeatures(Path(model_dir)/CONFIG['text_weights_file']) if CONFIG.get('robust_features') else None
    EXTRA=None
    if CONFIG.get('empty_candidate_threshold') is not None:
        if not empty_index:raise ValueError('Auxiliary empty-address index required')
        EXTRA=EmptyAddressRetrieval(db,empty_index)


def process_batch(batch):
    all_features, groups, raw_groups = [], [], []
    for row in batch:
        rec = record(row)
        found = candidates(CON, rec, CONFIG['top_k'], CONFIG['max_block'])
        groups.append((rec[0], [b[0] for b in found]))
        raw_groups.append((rec,found))
        all_features.extend(features(rec, other, CON) for other in found)
    if all_features:
        x = np.asarray(all_features, dtype=np.float32)
        first_scores=BASE_MODEL.predict_proba(x)[:,1] if BASE_MODEL is not None else None
        if CONFIG.get('rarity_features') or CONFIG.get('contextual_features') or CONFIG.get('posterior_features') or CONFIG.get('competition_features'):
            expanded=[];offset=0
            for rec,found in raw_groups:
                stop=offset+len(found);base=x[offset:stop];blocks=[base]
                similarities=candidate_similarities(found) if len(found)>1 and (CONFIG.get('contextual_features') or CONFIG.get('posterior_features')) else None
                if CONFIG.get('rarity_features'):
                    blocks.append(np.asarray([rarity_features(rec,b,CON) for b in found],dtype=np.float32).reshape(len(found),len(RARITY_NAMES)))
                if CONFIG.get('contextual_features'):blocks.append(contextual_features(rec,found,base,similarities))
                if CONFIG.get('posterior_features'):blocks.append(posterior_features(found,first_scores[offset:stop],similarities))
                if COMPETITION is not None:blocks.append(np.asarray([COMPETITION.features(rec,b) for b in found],dtype=np.float32).reshape(len(found),len(COMPETITION_NAMES)))
                expanded.append(np.concatenate(blocks,axis=1));offset=stop
            x=np.concatenate(expanded,axis=0)
        if ROBUST is not None:
            probabilities=[];offset=0
            for rec,found in raw_groups:
                probabilities.append(first_scores[offset:offset+len(found)]);offset+=len(found)
            blocks=ROBUST.batch([a for a,b in raw_groups],[b for a,b in raw_groups],probabilities)
            x=np.concatenate((x,np.concatenate(blocks,axis=0)),axis=1)
        if CONFIG.get('model_kind') == 'linear_svm':
            scores = x @ np.asarray(MODEL['coefficient']) + MODEL['intercept']
        elif CONFIG.get('model_kind')=='lightgbm':
            scores=np.asarray(MODEL.predict(x,num_threads=1),dtype=np.float32)
        else:
            scores = MODEL.predict_proba(x)[:, 1]
    else:
        scores = []
        x=np.zeros((0,len(CONFIG['feature_names'])),dtype=np.float32);first_scores=np.empty(0,dtype=np.float32)
    if EXTRA is not None:
        base_blocks=[];probabilities=[];offset=0
        for rec,found in raw_groups:
            base_blocks.append(x[offset:offset+len(found)]);probabilities.append(first_scores[offset:offset+len(found)]);offset+=len(found)
        extra_groups,extra_features=additional_features([r for r,f in raw_groups],[f for r,f in raw_groups],base_blocks,probabilities,CON,BASE_MODEL,COMPETITION,ROBUST,EXTRA)
        matrix=np.concatenate(extra_features,axis=0)
        extra_scores=MODEL.predict_proba(matrix)[:,1] if len(matrix) else np.empty(0,dtype=np.float32)
        output_features=[];output_scores=[];updated_groups=[];offset=extra_offset=0
        for (eid,ids),extra,added,base in zip(groups,extra_groups,extra_features,base_blocks):
            ps=extra_scores[extra_offset:extra_offset+len(extra)];extra_offset+=len(extra)
            eligible=ps>=CONFIG['empty_candidate_threshold']
            if CONFIG.get('empty_candidate_require_best_reference'):eligible &= added[:,CONFIG['feature_names'].index('best_reference_is_anchor')]==1
            output_scores.append(np.r_[scores[offset:offset+len(ids)],np.where(eligible,ps,-1).astype(np.float32)]);offset+=len(ids)
            output_features.append(np.concatenate((base,added),axis=0));updated_groups.append((eid,ids+[r[0] for r in extra]))
        x=np.concatenate(output_features,axis=0);scores=np.concatenate(output_scores);groups=updated_groups
    matches, candidate_lines, selected_scores = [], [], []
    offset = n_matches = 0
    for (eid, ids),(rec,_) in zip(groups,raw_groups):
        group_scores=scores[offset:offset+len(ids)]
        policy=CONFIG.get('country_policies',{}).get(rec[4],CONFIG)
        positions=[i for i,score in enumerate(group_scores) if score>=policy['threshold']]
        rescue=policy.get('rescue_threshold')
        if not positions and len(ids) and rescue is not None and max(group_scores)>=rescue:
            positions=[int(np.argmax(group_scores))]
        selected=[ids[i] for i in positions]
        if SAVE_SCORES:selected_scores.extend(float(group_scores[i]) for i in positions)
        offset += len(ids)
        n_matches += len(selected)
        matches.append(eid+'\t'+','.join(selected)+'\n')
        candidate_lines.append(eid+'\t'+','.join(ids)+'\n')
    feature_bytes = x.tobytes() if SAVE_FEATURES and len(x) else b''
    score_bytes=np.asarray(selected_scores,dtype=np.float32).tobytes() if SAVE_SCORES else b''
    return ''.join(matches).encode(), ''.join(candidate_lines).encode(), len(batch), offset, n_matches, feature_bytes, score_bytes


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source1', required=True)
    p.add_argument('--db', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='output')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--limit', type=int, help='Benchmark only; outputs cover only the requested reference rows')
    p.add_argument('--save-features', action='store_true', help='Cache float32 features for fast future model rescoring')
    p.add_argument('--save-scores',action='store_true',help='Save matched-link probabilities for unique target assignment')
    p.add_argument('--competition-cache')
    p.add_argument('--empty-index')
    args = p.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        p.error('workers and batch-size must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out/'matching_results.tsv.partial', out/'candidate_pairs.tsv.partial']
    if args.save_features:
        paths.append(out/'pair_features.f32.partial')
    if args.save_scores:paths.append(out/'match_probabilities.f32.partial')
    if any(path.with_suffix('').exists() for path in paths):
        raise FileExistsError('Completed outputs already exist; select a new output directory')
    model_config = Path(args.model)/'model_config.json'
    config=json.loads(model_config.read_text())
    signature = dict(source=identity(args.source1), index=identity(args.db),
                     model=identity(Path(args.model)/config.get('model_file','model.ubj')),
                     pipeline=hashlib.sha256(Path(__file__).with_name('pipeline.py').read_bytes()).hexdigest(),
                     config=hashlib.sha256(model_config.read_bytes()).hexdigest(), limit=args.limit,
                     save_features=args.save_features,save_scores=args.save_scores)
    signature['helpers']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('predict_parallel.py','retrieval_extras.py','context_features.py','rarity_features.py',
                                       'competition_features.py','cached_competition.py','reference_index.py')}
    extra=Path(str(args.db)+'.extra.sqlite')
    if extra.exists():signature['supplemental_index']=identity(extra)
    if (Path(args.model)/'base_model.ubj').exists():signature['base_model']=identity(Path(args.model)/'base_model.ubj')
    if config.get('robust_features'):
        signature['text_weights']=identity(Path(args.model)/config['text_weights_file'])
        signature['helpers']['robust_features.py']=hashlib.sha256(Path(__file__).with_name('robust_features.py').read_bytes()).hexdigest()
    if args.empty_index:
        signature['empty_index']=identity(args.empty_index)
        for name in ('empty_address_retrieval.py','supplementary_matching.py'):signature['helpers'][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    if args.competition_cache:
        signature['competition_cache']={name:identity(Path(args.competition_cache)/name) for name in ('metadata.json','keys.npy','values.npy')}
    if (Path(args.model)/'linear_svm.json').exists():
        signature['linear_svm'] = identity(Path(args.model)/'linear_svm.json')
    checkpoint = out/'progress.json'
    if checkpoint.exists():
        progress = json.loads(checkpoint.read_text())
        if signature != progress['signature']:
            raise ValueError('Resume inputs differ from checkpoint')
        for path, nbytes in zip(paths, progress['bytes']):
            with open(path, 'r+b') as f:
                if f.seek(0, 2) < nbytes:
                    raise ValueError('Output shorter than checkpoint')
                f.truncate(nbytes)
        log(f'Resuming from {progress["rows"]:,} completed anchors')
    else:
        if any(path.exists() for path in paths):
            raise FileExistsError('Partial files without a checkpoint; select a new output directory')
        headers = [b'source1_entity_id\tmatched_entity_ids\n', b'source1_entity_id\tcandidate_entity_ids\n']
        if args.save_features:
            headers.append(b'')
        if args.save_scores:headers.append(b'')
        for path, header in zip(paths, headers):
            path.write_bytes(header)
        progress = dict(signature=signature, rows=0, pairs=0, matches=0,
                        bytes=list(map(len, headers)), seconds=0)
        checkpoint.write_text(json.dumps(progress, indent=2))
    iterator = iter(islice(rows(args.source1), progress['rows'], args.limit))
    began = time.monotonic()
    prior_seconds, prior_rows = progress['seconds'], progress['rows']
    next_report = progress['rows'] + 10000
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = [stack.enter_context(open(path, 'ab')) for path in paths]
        mf, cf = streams[:2]
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers,
                    initializer=initialize, initargs=(args.db, args.model, args.save_features,args.competition_cache,args.save_scores,args.empty_index)))
        pending = deque()

        def submit():
            batch = list(islice(iterator, args.batch_size))
            if batch:
                pending.append(executor.submit(process_batch, batch))

        for _ in range(args.workers*2):
            submit()
        while pending:
            match_data, candidate_data, nrows, npairs, nmatch, feature_data, score_data = pending.popleft().result()
            mf.write(match_data)
            cf.write(candidate_data)
            if args.save_features:
                streams[2].write(feature_data)
            if args.save_scores:streams[3 if args.save_features else 2].write(score_data)
            for f in streams:
                f.flush()
                os.fsync(f.fileno())
            progress['rows'] += nrows
            progress['pairs'] += npairs
            progress['matches'] += nmatch
            progress['bytes'] = [f.tell() for f in streams]
            progress['seconds'] = prior_seconds + time.monotonic()-began
            temporary = checkpoint.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(checkpoint)
            if progress['rows'] >= next_report:
                rate = (progress['rows']-prior_rows)/(time.monotonic()-began)
                log(f'{progress["rows"]:,} anchors; {rate:.1f} anchors/sec; {progress["pairs"]:,} candidate pairs')
                next_report = progress['rows'] + 10000
            submit()
    for path in paths:
        path.rename(path.with_suffix(''))
    progress['complete'] = True
    checkpoint.write_text(json.dumps(progress, indent=2))
    log(f'Completed {progress["rows"]:,} anchors in {progress["seconds"]:.1f} seconds')


if __name__ == '__main__':
    main()
