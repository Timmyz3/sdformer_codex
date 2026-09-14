from pathlib import Path
from collections import defaultdict
import json,re
H=Path(__file__).resolve().parent
orig=[json.loads(x) for x in (H/'results.jsonl').read_text().splitlines()];rows=[json.loads(x) for x in (H/'lazy_results.jsonl').read_text().splitlines()]
ix={(r['case'],r['frontend'],r['reduce'],r['stress']):r for r in orig}
meta={r['name']:r for r in json.loads((H/'cases.json').read_text())['cases']};checks=matched=0;groups=defaultdict(list)
def eq(a,b):
 global checks
 checks+=1;assert a==b,(a,b)
eq(json.loads((H/'verification.json').read_text())['passed'],True)
for r in rows:
 k=r['case'],3,r['reduce'],r['stress'];b=ix[k];direct=ix[r['case'],0,r['reduce'],r['stress']];empty=direct['w_value_reads']==0
 if r['frontend']==3:
  for key,val in b.items():eq(r[key],val)
  eq(r['rank_catchup_pe_beats'],0);matched+=1
 else:
  eq(r['frontend'],4)
  for key in ['real','time','reduce','stress','source_reads','w_zero','packets','events','commits','psn_issues','w_value_reads','nnz_tile0','nnz_tile1','image_bytes_tile0','image_bytes_tile1','configuration_beats']:eq(r[key],b[key])
  eq(r['w_metadata_reads'],0 if empty else b['w_metadata_reads'])
  eq(r['w_reads'],r['w_metadata_reads']+r['w_value_reads'])
  eq(r['rank_catchup_pe_beats'],0 if empty else 1536*((r['nnz_tile0']<334)+(r['nnz_tile1']<334)))
  if empty:
   eq(r['frontend_load_beats'],0);eq(r['last_gate'],direct['last_gate']);eq(r['w_reads'],0)
 eq(r['service_including_configuration'],r['last_gate']+r['configuration_beats']+1)
 family=meta[r['case']].get('weight_variant','original_integer' if r['real'] else 'directed')
 groups[family,'time' if r['time'] else 'class','member' if r['reduce'] else 'scalar','stalled' if r['stress'] else 'ready',r['frontend']].append(r)
out=[]
for key,rr in groups.items():
 family,route,alu,pressure,front=key;a=dict(family=family,route=route,alu=alu,pressure=pressure,frontend=front,tasks=len(rr))
 for f in rr[0]:
  if f not in ['case','real','time','frontend','reduce','stress']:
   a[f]=max(r[f] for r in rr) if f.endswith('_peak') else sum(r[f] for r in rr)
 for f in ['last_gate','w_reads','w_metadata_reads','w_value_reads','service_including_configuration','frontend_load_beats','rank_catchup_pe_beats']:a['mean_'+f]=a[f]/len(rr)
 out.append(a)
(H/'lazy_group_results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in out))
m=re.search(r'PASS tasks=(\d+) gate_bits=(\d+) cycles=(\d+) response_backpressure=(\d+) nr_peak=(\d+) W_accept_peak=(\d+)',(H/'run_lazy.log').read_text())
eq(bool(m),True);eq(int(m[1]),len(rows));eq(int(m[2]),len(rows)*320)
result=dict(passed=True,checks=checks,rtl_tasks=len(rows),gate_bits_compared=len(rows)*320,prior_mode3_all_fields_reproduced=matched,empty_source_no_weight_or_metadata_requests=True,full_bitmap_cursor_advances=1536,unit='PE cycles per sparse tile (4PE x384); parallel/overlapped, not additional service cycles',all_rank_addresses_and_last_NR4_outputs_checked_in_CPP=True,NR4_peak=int(m[5]),W_accept_peak=int(m[6]),response_backpressure_beats=int(m[4]),simulation_beats=int(m[3]))
(H/'lazy_verification.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps(result,indent=2))
