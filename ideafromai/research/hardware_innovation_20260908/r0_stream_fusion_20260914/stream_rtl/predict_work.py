#!/usr/bin/env python3
"""Input-derived full-frame native geometry ledger, never supplied to RTL."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;D=H.parent/'data'
source=np.load(D/'source_words.npy',mmap_mode='r')
a=np.pad(source,((0,0),(1,1),(1,1)))
view=np.lib.stride_tricks.sliding_window_view(a,(4,4),axis=(1,2))[:,::2,::2,:,:]
assert view.shape==(96,120,160,4,4)
pc=np.array([i.bit_count() for i in range(1024)],dtype=np.int32)
geom=np.outer([1,2,2,1],[1,2,2,1])[None,None,:,:]
validy=np.full(120,4,dtype=np.int64);validy[[0,-1]]=3
validx=np.full(160,4,dtype=np.int64);validx[[0,-1]]=3
valid=(validy[:,None]*validx[None,:]).reshape(-1)
profiles=[]
for cg in range(24):
 v=[view[cg*4+i] for i in range(4)]
 u0=v[0]|v[1];u1=v[2]|v[3];u=u0|u1
 q=sum((x!=0).astype(np.int32) for x in v)
 j=(v[0]&v[1]!=0).astype(np.int32)+(v[2]&v[3]!=0).astype(np.int32)
 k=(u0!=0).astype(np.int32)+(u1!=0).astype(np.int32)
 active=(u!=0).astype(np.int32);h=pc[u0&u1];l=pc[u]
 def reduce(x):return np.sum(x*geom,axis=(2,3),dtype=np.int64).reshape(-1)
 profiles.append(dict(weight=reduce(q),parent=reduce(j),merge=reduce(h),update5=reduce(l+h),update6=reduce(l),
  service5=reduce(q+j+3*(l+h)+k+2*active),service6=reduce(q+j+3*l+h+3*active),terminal=reduce(k-active)))
metrics={}
for arm in ('dense_q16','block_magnitude25','cin_fullcost25'):
 live=np.load(D/f'{arm}_weights.npz')['live'];groups=np.sum(live,axis=0,dtype=np.int64);alive=int(np.count_nonzero(groups))
 metrics[arm]={}
 for mode in (5,6):
  core=np.full(19200,1442+112*alive+2*(24-alive),dtype=np.int64)
  fields={f:np.zeros(19200,dtype=np.int64) for f in ('core_weight_words','core_sum_issues','core_update_issues','core_merge_issues','terminal_savings')}
  for g,p in zip(groups,profiles):
   core+=g*p[f'service{mode}'];fields['core_weight_words']+=g*p['weight']
   fields['core_update_issues']+=g*p[f'update{mode}'];fields['core_merge_issues']+=g*p['merge']*(mode==6)
   fields['core_sum_issues']+=g*(p['parent']+(p['merge'] if mode==6 else 0));fields['terminal_savings']+=g*p['terminal']
  fields.update(core_cycles=core,core_source_words=alive*4*valid,
   core_psum_reads=fields['core_update_issues']+480,core_psum_writes=fields['core_update_issues']+480,
   external_source_words=96*valid,source_load_words=np.full(19200,1536,dtype=np.int64),origin_words=np.ones(19200,dtype=np.int64))
  np.savez(H/f'expected_{arm}_m{mode}.npz',**fields)
  metrics[arm][str(mode)]={}
  for name,sel in [('first64',slice(128,192)),('full',slice(None))]:
   n=64 if name=='first64' else 19200;r={f:int(v[sel].sum()) for f,v in fields.items()}
   r['tiles']=n;r['total_cycles_cold']=r['core_cycles']+1539*n+10657
   metrics[arm][str(mode)][name]=r
(H/'predicted_work.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({a:{m:v['full']['total_cycles_cold'] for m,v in ms.items()} for a,ms in metrics.items()}))
