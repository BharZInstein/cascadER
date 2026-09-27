"""Label-free comparison of candidate records to other candidates for the same anchor."""
import re
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz,process
from anyascii import anyascii

CONTEXT_NAMES = [f'{kind}_{measure}' for kind in ('name_bridge','address_bridge') for measure in
                 ('target_name','target_address','target_numbers','anchor_name','anchor_address','anchor_numbers')]
CONTEXT_NAMES += ['supported_name_neighbors','supported_address_neighbors']


@lru_cache(maxsize=100000)
def context_name(core):
    return anyascii(core).lower().replace(' ','')

def candidate_similarities(found):
    names=[context_name(r[2]) for r in found]
    addresses=[r[3] for r in found]
    def unique(values):
        mapping={v:i for i,v in enumerate(dict.fromkeys(values))}
        return list(mapping),np.asarray([mapping[v] for v in values])
    un,ni=unique(names);ua,ai=unique(addresses)
    nm=process.cdist(un,un,scorer=fuzz.ratio,dtype=np.float32)/100
    nm=nm[ni[:,None],ni[None,:]]
    am=.6*process.cdist(ua,ua,scorer=fuzz.token_sort_ratio,dtype=np.float32)/100
    am+=.4*process.cdist(ua,ua,scorer=fuzz.token_set_ratio,dtype=np.float32)/100
    am=am[ai[:,None],ai[None,:]]
    present=np.asarray([bool(a) for a in addresses]);am*=present[:,None]&present[None,:]
    np.fill_diagonal(nm,0);np.fill_diagonal(am,0)
    nums=[set(str(int(v)) for v in re.findall('[0-9]+',a)) for a in addresses]
    return nm,am,nums

def contextual_features(anchor, found, base, similarities=None):
    n=len(found)
    if n<2:return np.zeros((n,len(CONTEXT_NAMES)),dtype=np.float32)
    name_matrix,address_matrix,nums=similarities if similarities is not None else candidate_similarities(found)
    name_quality=np.maximum.reduce([base[:,0],base[:,4],base[:,20],base[:,25]])
    address_quality=.6*base[:,6]+.4*base[:,7]
    number_quality=base[:,23]
    # These support scores use only input text, not labels or classifier predictions.
    name_support=(address_quality**3)*(.4+.6*number_quality)*(1-base[:,18])
    address_support=(name_quality**3)*(.5+.5*number_quality)
    ni=np.argmax(name_matrix*name_support[None,:],axis=1)
    ai=np.argmax(address_matrix*address_support[None,:],axis=1)
    result=np.empty((n,len(CONTEXT_NAMES)),dtype=np.float32)
    for offset,indices in ((0,ni),(6,ai)):
        for i,j in enumerate(indices):
            denominator=len(nums[i]|nums[j]);overlap=len(nums[i]&nums[j])/denominator if denominator else 0
            result[i,offset:offset+6]=[name_matrix[i,j],address_matrix[i,j],overlap,name_quality[j],address_quality[j],number_quality[j]]
    result[:,12]=np.log1p(((name_matrix>.9)&(name_support[None,:]>.7)).sum(axis=1))
    result[:,13]=np.log1p(((address_matrix>.9)&(address_support[None,:]>.7)).sum(axis=1))
    return result

POSTERIOR_NAMES=['first_stage_probability']+[f'{kind}_seed_{measure}' for kind in ('name','address','joint') for measure in ('name_similarity','address_similarity','number_similarity','probability')]
POSTERIOR_NAMES+=['confident_name_neighbors','confident_address_neighbors']

def posterior_features(found,probabilities,similarities=None):
    n=len(found);result=np.zeros((n,len(POSTERIOR_NAMES)),dtype=np.float32)
    if not n:return result
    result[:,0]=probabilities
    if n<2:return result
    nm,am,nums=similarities if similarities is not None else candidate_similarities(found)
    for offset,sim in ((1,nm),(5,am),(9,.6*nm+.4*am)):
        indices=np.argmax(sim*probabilities[None,:],axis=1)
        for i,j in enumerate(indices):
            denominator=len(nums[i]|nums[j]);overlap=len(nums[i]&nums[j])/denominator if denominator else 0
            result[i,offset:offset+4]=[nm[i,j],am[i,j],overlap,probabilities[j]]
    result[:,13]=np.log1p(((nm>.9)&(probabilities[None,:]>.95)).sum(axis=1))
    result[:,14]=np.log1p(((am>.9)&(probabilities[None,:]>.95)).sum(axis=1))
    return result
