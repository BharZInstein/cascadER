"""Read precomputed competition records without repeating reference searches."""
import json
from pathlib import Path
import numpy as np
from competition_features import ReferenceCompetition

MISSING=np.iinfo(np.uint32).max
class CachedCompetition(ReferenceCompetition):
    def __init__(self,path):
        path=Path(path);self.metadata=json.loads((path/'metadata.json').read_text())
        if not self.metadata['complete']:raise ValueError('Incomplete competition cache')
        self.keys=np.load(path/'keys.npy',mmap_mode='r');self.values=np.load(path/'values.npy',mmap_mode='r')
    def lookup(self,b):
        value=int(b[0][3:])+(2**30 if b[0].startswith('S3-') else 0)
        # Match the uint32 index type; a Python int makes NumPy copy the full index.
        i=int(np.searchsorted(self.keys,np.uint32(value)))
        if i==len(self.keys) or int(self.keys[i])!=value:raise ValueError('Target missing from competition cache: '+b[0])
        row=self.values[i];scored=[]
        for eid,values in zip(row['ids'],row['values']):
            if eid!=MISSING:scored.append((float(values[0]),'S1-'+str(int(eid)),*map(float,values[1:])))
        return scored,int(row['counts'][0]),int(row['counts'][1]),None if row['owner']==MISSING else 'S1-'+str(int(row['owner']))
