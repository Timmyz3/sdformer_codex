import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
src=np.load(H.parent/'data/first_source_words.npy');q=np.load(H.parent/'data/consumer_coefficients.npz')['q1'];live=np.any(q!=0,axis=0).reshape(96,3,3)
pad=np.pad(src,((0,0),(1,1),(1,1)));pop=np.array([i.bit_count() for i in range(1024)],np.uint8)
counts={}
for name,indices in [('64',np.arange(128,192)),('full',np.arange(19200))]:
 singles=dual=pairs=0
 for py in range(2):
  for dy in range(3):
   for dx in range(3):
    a=pad[:,dy+py:dy+py+240:2,dx:dx+320:2].reshape(96,19200)[:,indices]
    b=pad[:,dy+py:dy+py+240:2,dx+1:dx+321:2].reshape(96,19200)[:,indices]
    mask=live[:,dy,dx,None]
    singles+=int((pop[a]*mask).sum())+int((pop[b]*mask).sum())
    pairs+=int((pop[a|b]*mask).sum());dual+=int((pop[a&b]*mask).sum())
 assert singles==pairs+dual
 counts[name]=dict(singles=singles,pairs=pairs,dual=dual)
summary={}
for suffix in ['64','full']:
 path=H/f'results_{suffix}.json'
 if not path.exists():continue
 rows=json.loads(path.read_text());checks=0
 for x in rows:
  n=x['tiles'];ext=0
  for tile in range(x['first_tile'],x['first_tile']+n):
   oy=2*(tile//160)-1;ox=2*(tile%160)-1
   ext+=96*sum(0<=oy+y<240 for y in range(4))*sum(0<=ox+z<320 for z in range(4))
  first=counts[suffix]['singles' if x['mode']==14 else 'pairs']
  target={'source_load_words':1536*n,'origin_words':n,'external_source_words':ext,'core_source_words':ext,'padding_words':1536*n-ext,'static_words':1848 if x['command']==0 else 0,'core_first_issues':first,'core_dual_updates':counts[suffix]['dual'] if x['mode']==15 else 0,'core_z_vector_reads':first+40*n,'core_z_writes':first+20*n,'core_psum_reads':480*n,'core_psum_writes':480*n,'consumer_identity_words':480*n,'consumer_raw_words':480*n,'consumer_coefficient_words':24*n,'consumer_conversion_issues':480*n,'consumer_mul_issues':480*n,'consumer_add_issues':960*n,'consumer_round_issues':480*n,'consumer_output_words':480*n,'consumer_saturations':0,'consumer_conversion_saturations':0,'outputs':3840*n,'raw_outputs':3840*n,'J_outputs':3840*n,'retired_tiles':n,'output_beats':480*n}
  for k,v in target.items():assert x[k]==v,(suffix,k,x[k],v);checks+=1
  assert x['consumer_cycles']==3385*n+x['consumer_join_wait_cycles']+x['consumer_output_stalls'];checks+=1
  assert x['total_cycles']==x['consumer_cycles']+x['static_words']+x['parameter_stalls']+1537*n+x['source_load_stalls']+2*n+1;checks+=1
  # Actual core absolute cycle law, Q2 words separate from Q1 demand words.
  nlive=int(live.sum())*n;active=x['core_weight_words']-x['core_second_weight_words']
  base=5437*n+nlive+2*active+3*first+x['core_mac_issues']
  assert x['core_cycles']==base+x['core_source_stalls']+x['core_weight_stalls']+x['core_output_stalls'];checks+=1
 ix={(x['mode'],x['stall'],x['command']):x for x in rows}
 for key,x in ix.items():
  if key[0]!=14 or key[1]!=0:continue
  y=ix[(15,key[1],key[2])]
  assert x['total_cycles']-y['total_cycles']==3*counts[suffix]['dual'];checks+=1
  for k in ['core_weight_words','core_mac_issues','core_psum_reads','core_psum_writes','core_z_scalar_reads','consumer_identity_words','consumer_conversion_issues']:
   assert x[k]==y[k];checks+=1
 summary[suffix]=dict(jobs=len(rows),values_each_checkpoint=sum(x['outputs'] for x in rows),geometry_state_and_delta_checks=checks,independent_source_coactivity=counts[suffix])
(H/'stream_checks.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
