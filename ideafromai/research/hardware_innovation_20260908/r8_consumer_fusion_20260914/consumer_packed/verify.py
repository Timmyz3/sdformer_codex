import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;rows=json.loads((H/'results.json').read_text());cases=json.loads((H/'fixtures.json').read_text())
checks=0;values=0;models={}
for case in cases:
 d=H/'fixtures'/case['name'];w=np.fromfile(d/'source.bin','<u2').reshape(96,4,4)
 s=((w[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
 oy,ox=case['input_origin']
 for y in range(4):
  for x in range(4):
   if not(0<=oy+y<240 and 0<=ox+x<320):s[:,:,y,x]=0
 q=np.fromfile(d/'param4.bin','<i4').reshape(864,8).T.astype(np.int64)
 v=np.fromfile(d/'param5.bin','<i4').reshape(12,8,8).transpose(0,2,1).reshape(96,8).astype(np.int64)
 patches=np.stack([s[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)])
 z=patches@q.T;raw=z@v.T
 # Every scalar gold is independently recomputed from native 3x3 geometry.
 goldraw=raw.reshape(4,10,12,8).transpose(2,0,1,3).reshape(480,8)
 assert np.array_equal(goldraw.ravel(),np.fromfile(d/'raw.bin','<i4'));values+=3840
 fp=np.fromfile(d/'identity_fp32.bin','<f4').astype(np.float64)
 j=np.clip(np.rint(fp*(1<<20)),-2**31,2**31-1).astype(np.int64).reshape(480,8)
 assert np.array_equal(j.ravel(),np.fromfile(d/'identity.bin','<i4'));values+=3840
 ab=np.fromfile(d/'param7.bin','<i4').astype(np.int64).reshape(12,2,8)
 gold=np.fromfile(d/'gold.bin','<i4').reshape(480,8)
 for row in range(480):
  for l in range(8):
   wide=int(goldraw[row,l])*int(ab[row//40,0,l])+(int(j[row,l])+int(ab[row//40,1,l]))*(1<<20)
   quotient,rem=divmod(wide,1<<26);quotient+=int(rem>(1<<25) or rem==(1<<25) and quotient%2)
   assert gold[row,l]==max(-2**23,min(2**23-1,quotient));values+=1
 klive=np.any(q!=0,axis=0);effective=patches*klive[None,None,:]
 singles=int(effective.sum());pairs=int(np.count_nonzero(effective[0]|effective[1])+np.count_nonzero(effective[2]|effective[3]))
 dual=int(np.count_nonzero(effective[0]&effective[1])+np.count_nonzero(effective[2]&effective[3]));assert singles-pairs==dual
 active_k=int(np.count_nonzero(np.any(effective!=0,axis=(0,1))));nklive=int(klive.sum())
 mac=0;vwords=0
 for og in range(12):
  for rank in range(8):
   if np.any(v[og*8:og*8+8,rank]!=0) and np.any(z[:,:,rank]!=0):
    vwords+=1;mac+=int(np.count_nonzero(z[:,:,rank]))
 valid=96*sum(0<=oy+y<240 for y in range(4))*sum(0<=ox+x<320 for x in range(4))
 for mode,first in [(14,singles),(15,pairs)]:
  models[(case['name'],mode)]=dict(core_source_words=valid,core_local_source_reads=nklive,core_weight_words=active_k+vwords,core_second_weight_words=vwords,core_first_issues=first,core_dual_updates=dual if mode==15 else 0,core_z_vector_reads=first+40,core_z_scalar_reads=mac,core_z_writes=first+20,core_psum_reads=480,core_psum_writes=480,core_mac_issues=mac,
   core_base_cycles=5437+nklive+2*active_k+3*first+mac)
for x in rows:
 model=models[(x['fixture'],x['mode'])]
 for k,v in model.items():
  if k=='core_base_cycles':assert x['core_cycles']==v+x['core_source_stalls']+x['core_weight_stalls']+x['core_output_stalls'],(k,x,v)
  else:assert x[k]==v,(k,x,v)
  checks+=1
 fixed={'consumer_raw_words':480,'consumer_identity_words':480,'consumer_conversion_issues':480,'consumer_coefficient_words':24,'consumer_mul_issues':480,'consumer_add_issues':960,'consumer_round_issues':480,'consumer_output_words':480,'outputs':3840,'raw_outputs':3840,'J_outputs':3840,'source_load_words':1536,'origin_words':1,'static_words':1848 if x['command']==0 else 0}
 for k,v in fixed.items():assert x[k]==v,(k,x,v);checks+=1
 assert x['consumer_cycles']==3385+x['consumer_join_wait_cycles']+x['consumer_output_stalls'];checks+=1
 assert x['total_cycles']==x['consumer_cycles']+x['static_words']+x['parameter_stalls']+1537+x['source_load_stalls']+3;checks+=1
assert (H/'i24_consumer.sv').read_text()==(H.parent/'consumer_rtl/i24_consumer.sv').read_text()
summary=dict(runs=len(rows),values_checked_each_checkpoint=sum(x['outputs'] for x in rows),independent_scalar_gold_values=values,counter_and_cycle_checks=checks,consumer_text_equal=True)
(H/'checks.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
