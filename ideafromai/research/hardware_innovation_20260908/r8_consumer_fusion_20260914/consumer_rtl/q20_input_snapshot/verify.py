import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
r=json.loads((H/'results.json').read_text());old=json.loads((H/'baseline_results.json').read_text())
index={(x['fixture'],x['mode'],x['stall'],x['command']):x for x in r};count=0
for o in old:
 n=index[(o['fixture'],o['mode'],o['stall'],o['command'])]
 for k,v in o.items():
  if k!='wall_seconds_so_far':assert n[k]==v,(k,o,n);count+=1
same_count=count;ledger=0
for x in r:
 target={'consumer_raw_words':480,'consumer_identity_words':480,'consumer_coefficient_words':24,
         'consumer_mul_issues':480,'consumer_add_issues':960,'consumer_round_issues':480,'consumer_output_words':480,
         'outputs':3840,'raw_outputs':3840,'source_load_words':1536,'origin_words':1,
         'static_words':12504 if x['command']==0 else 0,'retired_tiles':1,'output_beats':480}
 for k,v in target.items():assert x[k]==v,(k,x);ledger+=1
 assert x['external_source_words']+x['padding_words']==1536;ledger+=1
 assert x['consumer_cycles']==2905+x['consumer_join_wait_cycles']+x['consumer_output_stalls'];ledger+=1
 assert x['total_cycles']==x['consumer_cycles']+x['static_words']+x['parameter_stalls']+1537+x['source_load_stalls']+3;ledger+=1
case=json.loads((H/'fixtures.json').read_text());checks=0
for c in case:
 d=H/'fixtures'/c['name'];p=np.fromfile(d/'raw.bin','<i4').astype(np.int64).reshape(480,8)
 j=np.fromfile(d/'identity.bin','<i4').astype(np.int64).reshape(480,8)
 ab=np.fromfile(d/'param7.bin','<i4').astype(np.int64).reshape(12,2,8)
 actual=np.fromfile(d/'gold.bin','<i4').reshape(480,8)
 # Python arbitrary precision and divmod constitute a separate scalar oracle.
 for row in range(480):
  for l in range(8):
   wide=int(p[row,l])*int(ab[row//40,0,l])+(int(j[row,l])+int(ab[row//40,1,l]))*(1<<20)
   q,rem=divmod(wide,1<<26)
   q+=int(rem>(1<<25) or rem==(1<<25) and q%2)
   assert int(actual[row,l])==max(-2**23,min(2**23-1,q));checks+=1
 for x in r:
  if x['fixture']==c['name']:assert x['consumer_saturations']==c['saturations'];ledger+=1
x=np.load(H.parent/'data/consumer_integer_first8.npz');independent=0
for k in range(8):
 for fn,key in [('raw','p_int'),('identity','identity_q20'),('gold','i24_new')]:
  v=x[key][k].reshape(10,12,8,2,2).transpose(1,3,4,0,2).reshape(-1)
  assert np.array_equal(v,np.fromfile(H/'fixtures'/f'real_{k}'/(fn+'.bin'),'<i4'));independent+=len(v)
# The unchanged old leaf remains a traceable source, while the evaluated top
# instantiates root-owned r8_fusion to budget candidate state in all modes.
oldsv=H.parents[1]/'r0_stream_fusion_20260914/integer_factor/integer_factor.sv'
assert (H/'integer_factor.sv').read_text()==oldsv.read_text()
summary=dict(runs=len(r),I24_outputs=sum(x['outputs'] for x in r),raw_outputs=sum(x['raw_outputs'] for x in r),
 baseline_same_fields=same_count,ledger_assertions=ledger,python_scalar_oracle_values=checks,
 independent_data_values=independent,old_leaf_text_equal=True)
(H/'checks.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
