#!/usr/bin/env python3
"""Independent native geometry/support work and cycle identities.

No DUT state trace or precomputed support is supplied to RTL. This checker
reads the ordinary configured source/mask/origin after simulation and derives
expected transactions plus closed-form costs for the three fixed schedules.
"""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent

def read_hex(path):
    return np.array([int(x,16) for x in path.read_text().split()],dtype=np.int64)

def expected(path):
    raw=read_hex(path/'source.hex').reshape(96,4,4)
    live=read_hex(path/'mask.hex').reshape(12,24).astype(bool)
    origin=read_hex(path/'origin.hex');origin=np.where(origin>=2**31,origin-2**32,origin)
    valid=((np.arange(4)+origin[0]>=0)&(np.arange(4)+origin[0]<240))[:,None]&((np.arange(4)+origin[1]>=0)&(np.arange(4)+origin[1]<320))[None,:]
    source=raw*valid[None,:,:]
    counts={m:dict(cycles=1442,source_words=0,weight_words=0,psum_reads=480,psum_writes=480,sum_issues=0,update_issues=0,pattern_copies=0,masked_contexts=0,zero_contexts=0) for m in (3,4,5)}
    all_codes=set();overlap=0;destination_groups=0;nonzero_pairs=0;demand_patterns=0
    for cg in range(24):
        g=int(live[:,cg].sum())
        if not g:
            for v in counts.values():v['cycles']+=2;v['masked_contexts']+=1
            continue
        for m,v in counts.items():
            v['source_words']+=int(valid.sum())*4
            v['cycles']+=(160 if m==3 else 112)
        for y in range(4):
            for x in range(4):
                a=[int(v) for v in source[cg*4:cg*4+4,y,x]]
                consumers=g*sum(0<=y-py<3 and 0<=x-px<3 for py in range(2) for px in range(2))
                pair_u=[(a[k]|a[k+1]).bit_count() for k in (0,2)]
                pair_active=sum(u!=0 for u in pair_u)
                pair_sum=sum(bool(a[k]&a[k+1]) for k in (0,2))
                needed=sum(v!=0 for v in a)
                codes=[sum(((a[c]>>t)&1)<<c for c in range(4)) for t in range(10)]
                all_codes.update(codes)
                patterns=set(codes)-{0}
                u=sum(c!=0 for c in codes)
                builds=sum(c.bit_count()-1 for c in patterns)
                if not patterns:
                    counts[3]['zero_contexts']+=2
                    counts[4]['zero_contexts']+=1
                    counts[5]['zero_contexts']+=1
                    continue
                counts[3]['zero_contexts']+=2-pair_active
                destination_groups+=consumers
                nonzero_pairs+=pair_active*consumers
                demand_patterns+=len(patterns)*consumers
                overlap+=(sum(pair_u)-u)*consumers
                for m,v in counts.items():
                    v['weight_words']+=needed*consumers
                    updates=(u if m==4 else sum(pair_u))*consumers
                    v['update_issues']+=updates;v['psum_reads']+=updates;v['psum_writes']+=updates
                    v['sum_issues']+=(builds if m==4 else pair_sum)*consumers
                counts[3]['cycles']+=(needed+pair_sum+3*sum(pair_u)+3*pair_active)*consumers
                counts[5]['cycles']+=(needed+pair_sum+3*sum(pair_u)+pair_active+2)*consumers
                counts[4]['cycles']+=(needed+builds+3*u+2*len(patterns)+2)*consumers
                counts[4]['pattern_copies']+=len(patterns)*consumers
    return counts,dict(support_codes=sorted(all_codes),saved_psum_update_beats=overlap,
        c4_destination_services=destination_groups,nonzero_pair_services=nonzero_pairs,demanded_pattern_services=demand_patterns)

def main():
    rows=json.loads((HERE/'results.json').read_text());cache={};checked=0;facts={}
    for r in rows:
        path=Path(r['fixture_path'])
        if path not in cache:cache[path]=expected(path)
        counts,detail=cache[path];e=counts[r['mode']]
        for field,value in e.items():
            actual=r[field]
            if field=='cycles':actual-=r['source_stalls']+r['weight_stalls']+r['output_stalls']
            assert actual==value,(r['fixture'],r['mode'],field,actual,value)
            checked+=1
        facts[r['fixture']]=detail
    control=facts['all_supports_extreme'];assert control['support_codes']==list(range(16))
    result=dict(passed=True,runs=len(rows),scalar_counter_checks=checked,
        all_support_codes_covered=True,independent_formula_scope='Configured native source/valid geometry/live mask; no RTL state trace; stalls checked additively.',fixtures=facts)
    (HERE/'ledger_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fixtures'},indent=2))
if __name__=='__main__':main()
