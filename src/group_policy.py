"""Optional top-candidate fallback, selected on tuning groups."""
import numpy as np

def best_indices(scores,groups,n):
    maxima=np.full(n,-np.inf);np.maximum.at(maxima,groups,scores)
    ties=scores==maxima[groups];best=np.full(n,len(scores),dtype=np.int64)
    np.minimum.at(best,groups[ties],np.flatnonzero(ties));best[best==len(scores)]=-1
    return best

def unique_assignments(chosen,scores,targets,groups):
    selected=np.flatnonzero(chosen)
    if not len(selected):return chosen
    order=np.lexsort((groups[selected],-scores[selected],targets[selected]))
    ordered=selected[order];ids=targets[ordered]
    first=np.r_[True,ids[1:]!=ids[:-1]]
    result=np.zeros(len(chosen),dtype=bool);result[ordered[first]]=True
    return result

def chosen_pairs(scores,threshold,groups,n,rescue=None,best=None,targets=None):
    chosen=scores>=threshold
    if rescue is not None:
        if best is None:best=best_indices(scores,groups,n)
        count=np.bincount(groups[chosen],minlength=n)
        empty=np.flatnonzero((count==0)&(best>=0));top=best[empty]
        chosen[top[scores[top]>=rescue]]=True
    return unique_assignments(chosen,scores,targets,groups) if targets is not None else chosen

def apply_policies(scores,groups,countries,config,targets=None):
    n=len(countries);policies=config.get('country_policies',{})
    if not policies:return chosen_pairs(scores,config['threshold'],groups,n,config.get('rescue_threshold'),targets=targets)
    chosen=np.zeros(len(scores),dtype=bool)
    for country in np.unique(countries[groups]):
        mask=countries[groups]==country;policy=policies.get(str(country),config)
        chosen[mask]=chosen_pairs(scores[mask],policy['threshold'],groups[mask],n,policy.get('rescue_threshold'))
    return unique_assignments(chosen,scores,targets,groups) if targets is not None else chosen
