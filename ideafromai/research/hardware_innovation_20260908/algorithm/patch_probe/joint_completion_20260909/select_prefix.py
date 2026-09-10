"""Give both fixed graphs the same ordinary train-only three-column choice.

This is a stronger compilation control, not another new mechanism. Enumerate
all 120 sets at the untrained fitted A/b and fixed gamma=3; select least group
column work subject to <=0.1% changed gates relative to that graph's full
function. No validation input enters selection and no gamma search is used.
"""
from itertools import combinations
import json
import time

import torch

from train_local import HERE, Completion, load_inputs


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    variants,splits,mean,cov,theta,_=load_inputs()
    data=splits['train']
    result=dict(scope='ordinary train-only prefix compilation on real FP norm1 capture',
        selection='minimum group-column visits among all120 triples with own-full gate change <=0.001; fallback least error if none',
        fixed_gamma=3.,train=data['files'],uses_validation=False,selected={},all={})
    started=time.monotonic()
    for name,params in variants.items():
        rows=[]
        for prefix in combinations(range(10),3):
            model=Completion(params,mean,cov,theta,prefix)
            gates=wrong=0
            work=base=0.
            for start in range(0,len(data['y']),256):
                sl=slice(start,start+256)
                full,predicted,_,accept=model(data['y'][sl],data['channels'][sl])
                wrong+=int((accept&((predicted>=0)!=(full>=0))).sum())
                gates+=full.numel()
                _,need=model.demands((~accept).float())
                work+=float((need*data['visits'][sl]).sum())
                base+=float(data['visits'][sl].sum())
            rows.append(dict(prefix=list(prefix),column_ratio=work/base,own_full_error=wrong/gates))
        feasible=[r for r in rows if r['own_full_error']<=0.001]
        chosen=min(feasible,key=lambda r:(r['column_ratio'],r['own_full_error'],r['prefix'])) if feasible else min(rows,key=lambda r:(r['own_full_error'],r['column_ratio'],r['prefix']))
        result['all'][name]=rows
        result['selected'][name]=dict(**chosen,feasible_count=len(feasible))
        print(name,json.dumps(result['selected'][name]),flush=True)
    result['wall_seconds']=time.monotonic()-started
    (HERE/'prefix_selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
