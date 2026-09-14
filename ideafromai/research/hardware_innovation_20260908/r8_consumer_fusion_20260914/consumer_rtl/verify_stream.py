import json
from pathlib import Path
H=Path(__file__).resolve().parent
out={}
for suffix in ['64','full']:
 p=H/f'results_{suffix}.json'
 if not p.exists():continue
 rows=json.loads(p.read_text())
 if not all(x.get('identity_input')=='IEEE_binary32' for x in rows):continue
 checks=0
 for x in rows:
  n=x['tiles'];first=x['first_tile'];ext=0
  for tile in range(first,first+n):
   oy=2*(tile//160)-1;ox=2*(tile%160)-1
   ext+=96*sum(0<=oy+y<240 for y in range(4))*sum(0<=ox+z<320 for z in range(4))
  target={'source_load_words':1536*n,'origin_words':n,'external_source_words':ext,'padding_words':1536*n-ext,'static_words':12504 if x['command']==0 else 0,'consumer_identity_words':480*n,'consumer_raw_words':480*n,'consumer_coefficient_words':24*n,'consumer_conversion_issues':480*n,'consumer_mul_issues':480*n,'consumer_add_issues':960*n,'consumer_round_issues':480*n,'consumer_output_words':480*n,'consumer_saturations':0,'consumer_conversion_saturations':0,'outputs':3840*n,'raw_outputs':3840*n,'J_outputs':3840*n,'retired_tiles':n,'output_beats':480*n}
  for k,v in target.items():assert x[k]==v,(k,x[k],v);checks+=1
  assert x['consumer_cycles']==3385*n+x['consumer_join_wait_cycles']+x['consumer_output_stalls'];checks+=1
  assert x['total_cycles']==x['consumer_cycles']+x['static_words']+x['parameter_stalls']+1537*n+x['source_load_stalls']+2*n+1;checks+=1
 # For unblocked jobs the only function-preserving change from archived Q20
 # endpoint is one conversion vector state per result. Source/W work unchanged.
 old=json.loads((H/'q20_input_snapshot'/f'results_{suffix}.json').read_text())
 ix={(x['mode'],x['stall'],x['command']):x for x in old}
 compared=0
 for x in rows:
  if x['stall']:continue
  y=ix[(x['mode'],x['stall'],x['command'])]
  assert x['total_cycles']-y['total_cycles']==480*x['tiles'];compared+=1
  for k in ['core_source_words','core_weight_words','core_z_reads','core_z_writes','core_psum_reads','core_psum_writes','external_source_words','static_words','consumer_identity_words','consumer_mul_issues','consumer_add_issues']:
   assert x[k]==y[k],(k,x,y);compared+=1
 out[suffix]=dict(jobs=len(rows),I24_values=sum(x['outputs'] for x in rows),raw_values=sum(x['raw_outputs'] for x in rows),J_values=sum(x['J_outputs'] for x in rows),ledger_checks=checks,Q20_snapshot_delta_checks=compared)
(H/'stream_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
