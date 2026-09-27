"""Unlabeled reference competition features; IDs are used only for joins."""
from functools import lru_cache
import math,sqlite3
from rapidfuzz import fuzz
from reference_index import fingerprint,name_fingerprint,comparison_name
from pipeline import roman_core,numeric_tokens

COMPETITION_NAMES=['best_reference_is_anchor','reference_score_margin','other_reference_name','other_reference_address',
                   'other_reference_numbers','same_name_reference_count','same_address_reference_count',
                   'unique_address_reference_is_anchor','best_reference_score']

class ReferenceCompetition:
    def __init__(self,path):
        self.con=sqlite3.connect(path)
        self.con.execute('PRAGMA mmap_size=4294967296')
        self.con.execute('PRAGMA cache_size=-32768')
        if self.con.execute('SELECT complete FROM metadata').fetchone()!=(1,):raise ValueError('Incomplete reference index')

    @lru_cache(maxsize=100000)
    def lookup(self,b):
        namekey=name_fingerprint(b[2],b[4]);addresskey=fingerprint(b[3],b[4]) if b[3] else None
        if addresskey:
            found=self.con.execute('SELECT eid,core,address,namekey,addresskey FROM refs WHERE namekey=? OR addresskey=? LIMIT 1001',(namekey,addresskey)).fetchall()
        else:
            found=self.con.execute('SELECT eid,core,address,namekey,addresskey FROM refs WHERE namekey=? LIMIT 1001',(namekey,)).fetchall()
        scored=[];name_count=address_count=0;address_owner=None
        for eid,core,address,nk,ak in found:
            name_count+=nk==namekey
            if addresskey and ak==addresskey:address_count+=1;address_owner=eid
            name=fuzz.token_sort_ratio(roman_core(comparison_name(core,b[4])),roman_core(comparison_name(b[2],b[4])))/100
            addr=(.6*fuzz.token_sort_ratio(address,b[3])+.4*fuzz.token_set_ratio(address,b[3]))/100 if address and b[3] else 0
            left,right=numeric_tokens(address),numeric_tokens(b[3]);num=len(left&right)/max(1,len(left|right))
            score=.45*name+.4*addr+.15*num
            scored.append((score,eid,name,addr,num))
        scored.sort(key=lambda r:(-r[0],r[1]))
        return scored[:2],name_count,address_count,address_owner

    def features(self,a,b):
        scored,nc,ac,owner=self.lookup(b)
        other=next((r for r in scored if r[1]!=a[0]),None)
        left,right=numeric_tokens(a[3]),numeric_tokens(b[3]);num=len(left&right)/max(1,len(left|right))
        name=fuzz.token_sort_ratio(roman_core(comparison_name(a[2],a[4])),roman_core(comparison_name(b[2],b[4])))/100
        addr=(.6*fuzz.token_sort_ratio(a[3],b[3])+.4*fuzz.token_set_ratio(a[3],b[3]))/100 if a[3] and b[3] else 0
        own=.45*name+.4*addr+.15*num
        return [float(bool(scored) and scored[0][1]==a[0]),own-(other[0] if other else 0),
                other[2] if other else 0,other[3] if other else 0,other[4] if other else 0,
                math.log1p(nc),math.log1p(ac),float(ac==1 and owner==a[0]),scored[0][0] if scored else 0]
