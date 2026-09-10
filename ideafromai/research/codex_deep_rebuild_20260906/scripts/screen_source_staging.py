#!/usr/bin/env python3.12
"""Strong C2 control: up to two arbitrary individual sources, fixed inputs 0/1.
An optimistic input latch captures an already fetched weight without an ALU
update. Also report a variant charging one extra service intent per capture.
Descriptor selection, latch port timing and pipeline stalls remain unpriced.
"""
import sys
sys.dont_write_bytecode=True
import argparse,json
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import screen_reuse as s


def costs(x):
    n=x.shape[1];groups=n//8
    codes=(x.astype(np.int64)*np.array([1,2,4,8])[:,None]).sum(0)
    eligible=np.flatnonzero(codes).tolist()
    choices=[()]+[(i,) for i in eligible]+list(combinations(eligible,2))
    pos=np.full((len(choices),2),-1,dtype=np.int64)
    for c,sel in enumerate(choices):pos[c,:len(sel)]=sel
    ordinary_mask=(x.reshape(4,groups,8).transpose(1,0,2)*np.array([1,2,4,8,16,32,64,128])).sum(2)
    masks=np.broadcast_to(ordinary_mask,(len(choices),groups,4)).copy()
    pending=np.zeros((len(choices),2),dtype=np.int64)
    for slot in range(2):
        valid=np.flatnonzero(pos[:,slot]>=0);p=pos[valid,slot]
        pending[valid,slot]=codes[p]
        for d in range(4):
            masks[valid,p//8,d]&=~(1<<(p%8))
    base=(masks!=0).sum((1,2)).astype(np.int64)
    for g in range(groups):
        for d in range(4):
            free=((masks[:,g,d]&1)==0).astype(np.int64)+((masks[:,g,d]&2)==0)
            for slot in range(2):
                ready=(pos[:,slot]>=0)&(pos[:,slot]//8<=g)&((pending[:,slot]&(1<<d))!=0)&(masks[:,g,d]!=0)&(free>0)
                pending[ready,slot]&=~(1<<d);free-=ready
    pc=np.array([i.bit_count() for i in range(16)])
    base+=pc[pending].sum(1)
    selections=(pos>=0).sum(1)
    picks=[]
    for capture_cost in (0,1):
        charge=base+capture_cost*selections
        # Fewer stored sources on equal service count.
        best=int(np.argmin(charge*3+selections))
        picks.append((int(charge[best]),choices[best]))
    return int((ordinary_mask!=0).sum()),picks


def verify(x,sel,capture_cost,expected):
    rng=np.random.default_rng(260926);w=rng.integers(-128,128,(x.shape[1],96),dtype=np.int64)
    w[:,0]=-128;w[:,1]=127
    y=np.zeros((4,96),dtype=np.int64);pending={i:{d for d in range(4) if x[d,i]} for i in sel}
    slots={};intents=0
    for g in range(x.shape[1]//8):
        for i in sel:
            if i//8==g:slots[i]=w[i].copy();intents+=capture_cost
        for d in range(4):
            members=[i for i in range(g*8,g*8+8) if x[d,i] and i not in sel]
            if not members:continue
            v=w[members].sum(0);free=sum(g*8+b not in members for b in (0,1))
            ready=[i for i in sel if i in slots and d in pending[i]]
            for i in ready[:free]:v+=slots[i];pending[i].remove(d)
            y[d]+=v;intents+=1
    for i in sel:
        for d in pending[i]:y[d]+=slots[i];intents+=1
    assert np.array_equal(y,x.astype(np.int64)@w)
    assert intents==expected,(intents,expected)
    return y.size


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fixtures=s.HW/'tb_m2018/fixtures'
    layers={r['layer_id']:r for r in json.loads((s.HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json').read_text())['layers']}
    totals={};rows=[];checked=0;count=0
    for prefix,extent in (('m2051_ep34_tsbg_full40_s1920',48),('m2067_ep34_fc2_exact_continuation_s960',192)):
        meta=json.loads((fixtures/(prefix+'.json')).read_text())
        words=np.array([int(v,16)&65535 for v in (fixtures/(prefix+'.memh')).read_text().split()],dtype=np.uint16).reshape(len(meta['rows']),4,extent)
        for row in meta['rows']:
            assert row['negative_codes']==0
            target=row['target'] if extent==48 else 'FC2';mult=layers[row['layer_id']]['weight_layout']['output_tile_count']
            x=s.prior.bits(words[row['slot'],:,:int(row['source_groups'])]);c=Counter()
            for start in range(0,x.shape[1],64):
                xx=x[:,start:start+64];ordinary,picks=costs(xx);c.update(ordinary=ordinary,windows=1)
                for price,(cost,sel) in enumerate(picks):
                    c.update({f'capture{price}':cost,f'capture{price}_slots':len(sel)})
                    if row['slot']<1:checked+=verify(xx,sel,price,cost)
            totals.setdefault(target,Counter()).update({k:v*mult for k,v in c.items()})
            rows.append(dict(target=target,slot=row['slot'],sequence=row['sequence'],output_tiles=mult,counts=c))
            count+=1
            if count%480==0:print(f'staging {count}/2880',flush=True)
    for c in totals.values():
        for price in (0,1):c[f'capture{price}_reduction']=1-c[f'capture{price}']/c['ordinary']
    out=dict(scope='C2 strong individual-source staging CPU intent bound; window=64, up to two raw INT8 vector latches, fixed tree inputs 0/1',
        workloads=count,weighted=totals,rows=rows,checked_outputs=checked,mismatches=0,
        boundaries=['Any active individual source can be chosen, including repeated destination signatures and unicast sources',
          'Capture0 optimistically overlaps up to two vector-latch writes with the already accepted bank payload; capture1 charges one service intent per staged source',
          'Fixed inputs 0/1 still need a two-slot to two-input selection network; slot0 is not hardwired exclusively to input0',
          'Capture-only bank requests must remain live after their ALU bypass disappears; up to two same-group latch writes must be supported',
          'No extra SRAM-request saving, no implemented selector/ports/stall timing, and no RTL or PPA claim',
          'Each full window is arrived before exhaustive choice; this is a strong upper bound, not a ready online controller',
          'Compare combined partial+singleton scheme against capture1 to isolate choice space, and capture0 to expose cheaper raw staging'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
