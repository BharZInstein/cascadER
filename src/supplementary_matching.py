"""Compute features for auxiliary candidates using the original set as context."""
import numpy as np
from pipeline import features,FEATURE_NAMES
from context_features import candidate_similarities,contextual_features,posterior_features
from rarity_features import rarity_features


def additional_features(anchors,found,base_blocks,first_probabilities,con,base_model,competition,robust,retriever):
    extras=[retriever.candidates(a,[r[0] for r in original],8) for a,original in zip(anchors,found)]
    new_base=[np.asarray([features(a,b,con) for b in extra],dtype=np.float32).reshape(len(extra),len(FEATURE_NAMES)) for a,extra in zip(anchors,extras)]
    together=np.concatenate(new_base,axis=0)
    new_prob=base_model.predict_proba(together)[:,1] if len(together) else np.empty(0,dtype=np.float32)
    all_found=[original+extra for original,extra in zip(found,extras)];probabilities=[];offset=0
    for old,extra in zip(first_probabilities,extras):
        probabilities.append(np.r_[old,new_prob[offset:offset+len(extra)]]);offset+=len(extra)
    text=robust.batch(anchors,all_found,probabilities);result=[]
    for a,original,extra,old_base,base,records,prob,new_text in zip(anchors,found,extras,base_blocks,new_base,all_found,probabilities,text):
        if not extra:result.append(np.zeros((0,108),dtype=np.float32));continue
        similarity=candidate_similarities(records);whole_base=np.concatenate((old_base[:,:len(FEATURE_NAMES)],base),axis=0)
        context=contextual_features(a,records,whole_base,similarity)[len(original):]
        posterior=posterior_features(records,prob,similarity)[len(original):]
        rare=np.asarray([rarity_features(a,b,con) for b in extra],dtype=np.float32)
        comp=np.asarray([competition.features(a,b) for b in extra],dtype=np.float32)
        result.append(np.concatenate((base,rare,context,posterior,comp,new_text[len(original):]),axis=1))
    return extras,result
