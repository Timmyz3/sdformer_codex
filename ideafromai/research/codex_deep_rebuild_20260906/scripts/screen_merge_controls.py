#!/usr/bin/env python3.12
"""Control experiments: fixed wide input positions and deferred single sources.
CPU intent bounds only. Single-source baseline covers singleton signatures,
not every possible individual source from a repeated signature.
"""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import screen_reuse as s
from screen_small_state import merged_cost


def verify(x,selection,codes,fixed):
    rng=np.random.default_rng(261906+x.shape[1]);w=rng.integers(-128,128,(x.shape[1],96),dtype=np.int64)
    w[:,0]=-128;w[:,1]=127
    last={m:max(i//8 for i,c in enumerate(codes) if c==m) for m in selection}
    pending={m:set(d for d in range(4) if m>>d&1) for m in selection}
    partial={m:np.zeros(96,dtype=np.int64) for m in selection}
    output=np.zeros((4,96),dtype=np.int64);intents=0;selected=set(selection)
    for g in range(x.shape[1]//8):
        cg=codes[g*8:g*8+8]
        for m in selection:
            members=[g*8+b for b,c in enumerate(cg) if c==m]
            if members:
                partial[m]+=w[members].sum(0);intents+=1
        for d in range(4):
            members=[g*8+b for b,c in enumerate(cg) if int(c)>>d&1 and int(c) not in selected]
            if not members:continue
            value=w[members].sum(0);free=(sum(g*8+b not in members for b in (0,1)) if fixed else 8-len(members))
            ready=[m for m in selection if d in pending[m] and last[m]<=g]
            for m in ready[:free]:value+=partial[m];pending[m].remove(d)
            output[d]+=value;intents+=1
    for m in selection:
        for d in pending[m]:output[d]+=partial[m];intents+=1
    assert np.array_equal(output,x.astype(np.int64)@w)
    assert intents==merged_cost(x,selection,codes,fixed)[0]
    return output.size


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fixtures=s.HW/'tb_m2018/fixtures'
    lp=s.HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json'
    layers={r['layer_id']:r for r in json.loads(lp.read_text())['layers']}
    totals={};rows=[];checked=0;count=0
    for prefix,extent in (('m2051_ep34_tsbg_full40_s1920',48),('m2067_ep34_fc2_exact_continuation_s960',192)):
        meta=json.loads((fixtures/(prefix+'.json')).read_text())
        words=np.array([int(v,16)&65535 for v in (fixtures/(prefix+'.memh')).read_text().split()],dtype=np.uint16).reshape(len(meta['rows']),4,extent)
        for row in meta['rows']:
            assert row['negative_codes']==0
            target=row['target'] if extent==48 else 'FC2';mult=layers[row['layer_id']]['weight_layout']['output_tile_count']
            x=s.prior.bits(words[row['slot'],:,:int(row['source_groups'])])
            for window in (16,64,768):
                c=Counter();key=f'{target}_window{window}'
                for start in range(0,x.shape[1],window):
                    xx=x[:,start:start+window];codes=(xx.astype(np.int64)*np.array([1,2,4,8])[:,None]).sum(0)
                    hist=np.bincount(codes,minlength=16)
                    ordinary=int((xx.reshape(4,-1,8).sum(2)>0).sum())
                    c.update(ordinary=ordinary,windows=1)
                    selections={'partial':[m for m in range(1,16) if m.bit_count()>=2 and hist[m]>=2],
                                'singleton':[m for m in range(1,16) if m.bit_count()>=2 and hist[m]==1],
                                'combined':[m for m in range(1,16) if m.bit_count()>=2 and hist[m]>=1]}
                    for label,eligible in selections.items():
                        for fixed in (False,True):
                            name=label+('_fixed01' if fixed else '_flexible')
                            best=(ordinary,(),0)
                            for n in (1,2):
                                for sel in combinations(eligible,n):
                                    cost,merged,_=merged_cost(xx,sel,codes,fixed)
                                    if (cost,len(sel),sel)<(best[0],len(best[1]),best[1]):best=(cost,sel,merged)
                            c.update({name:best[0],name+'_merged':best[2],name+'_selected':len(best[1])})
                            if row['slot']<1:checked+=verify(xx,best[1],codes,fixed)
                totals.setdefault(key,Counter()).update({k:v*mult for k,v in c.items()})
                rows.append(dict(target=target,window=window,sequence=row['sequence'],slot=row['slot'],output_tiles=mult,counts=c))
            count+=1
            if count%480==0:print(f'controls {count}/2880',flush=True)
    for c in totals.values():
        for scheme in ('partial','singleton','combined'):
            for ports in ('fixed01','flexible'):
                name=scheme+'_'+ports;c[name+'_reduction']=1-c[name]/max(1,c['ordinary'])
    out=dict(scope='C2 controlled fusion intent bounds, not cycles/PPA',workloads=count,weighted=totals,rows=rows,
        checked_event_replay_outputs=checked,event_replay_mismatches=0,
        boundaries=['Fixed01 permits wide injection only at source-bank0/1 positions unused by that consumer bypass',
          'Flexible variant needs arbitrary input packing/selection logic; both variants need a charged mixed-width reduction tree',
          'Singleton control stages signatures containing exactly one source; it is weaker than unrestricted individual-source staging',
          'Combined selection may select either a singleton or a shared partial, with at most two total slots',
          'All choices see arrived full-window descriptors; exhaustive selector, buffering and stalls remain unpriced',
          'Class updates, producer completion, per-consumer merge and leftover scatter undergo directed numeric event replay',
          'No additional TSBG weight-fetch savings, network speedup or accepted RTL result is claimed'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
