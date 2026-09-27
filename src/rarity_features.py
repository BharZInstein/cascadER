"""Text shape and corpus frequency features derived only from supplied records."""
import hashlib,re,math
from functools import lru_cache
import numpy as np
from retrieval_extras import STOP

RARITY_NAMES=['anchor_name_words','target_name_words','target_name_digits','target_name_letters','target_name_vowels',
              'anchor_address_frequency','target_address_frequency','rarest_shared_address_word',
              'weighted_address_overlap','weighted_name_overlap','joined_number_ratio']


def rarity_features(a,b,con):
    from rapidfuzz.fuzz import ratio
    _,an,ac,aa,country=a;_,bn,bc,ba,_=b
    letters=re.sub('[^a-z]','',bc)
    ak=address_keys(a);bk=address_keys(b)
    common=set(ak)&set(bk)
    shared=[con.extra_frequency(k) for k in common]
    address_freq=lambda s:con.key_frequency(country+'|a|'+hashlib.blake2b(s.encode(),digest_size=8).hexdigest()) if s else 0
    ank={country+'|f|'+t for t in ac.split() if len(t)>=3};bnk={country+'|f|'+t for t in bc.split() if len(t)>=3}
    def weighted_overlap(left,right):
        union=left|right
        weights={k:1/math.sqrt(max(1,con.extra_frequency(k))) for k in union}
        return sum(weights[k] for k in left&right)/max(1e-9,sum(weights.values()))
    return [len(ac.split()),len(bc.split()),sum(c.isdigit() for c in bc),len(letters),
            sum(c in 'aeiou' for c in letters)/max(1,len(letters)),np.log1p(address_freq(aa)),np.log1p(address_freq(ba)),
            np.log1p(min(shared)) if shared else 20,weighted_overlap(set(ak),set(bk)),weighted_overlap(ank,bnk),
            ratio(''.join(re.findall('[0-9]+',aa)),''.join(re.findall('[0-9]+',ba)))/100]

def address_keys(rec):
    return cached_address_keys(rec[3],rec[4])

@lru_cache(maxsize=100000)
def cached_address_keys(address,country):
    tokens=sorted({t for t in address.split() if len(t)>=4 and not t.isdigit() and t not in STOP},key=lambda t:(-len(t),t))[:7]
    return tuple(sorted({country+'|w|'+t[:10] for t in tokens}))
