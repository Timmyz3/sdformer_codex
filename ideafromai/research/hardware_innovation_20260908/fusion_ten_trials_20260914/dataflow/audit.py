"""Author-side receipt/port/geometry audit, not a fresh RTL run or independent review."""
import json,csv
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
O=H.parents[1]/'r8_consumer_fusion_20260914'
checks=0
def eq(a,b):
 global checks
 checks+=1
 assert a==b,(a,b)
rows_by={};summary={}
for name in ['d1_forward','d2_halo','d3_interleave']:
 d=H/name; rows=[]
 for f in sorted(d.glob('results*.json')):
  for rr in json.loads(f.read_text()):
   r=dict(rr);r['receipt']=f.name;rows.append(r)
   n=r.get('tiles',1); beats=480*n
   for k in ['outputs','raw_outputs','J_outputs']:eq(r[k],3840*n)
   eq(r['retired_tiles'],n);eq(r['output_beats'],beats)
   eq(r['origin_words'],n);eq(r['static_words'],0 if r['command'] else 1848)
   eq(r['source_load_words'],r['external_source_words']+r['padding_words'])
   for k in ['consumer_raw_words','consumer_identity_words','consumer_mul_issues','consumer_round_issues','consumer_output_words','consumer_conversion_issues']:eq(r[k],beats)
   eq(r['consumer_coefficient_words'],24*n);eq(r['consumer_add_issues'],2*beats)
   eq(r['consumer_cycles'],3385*n+r['consumer_join_wait_cycles']+r['consumer_output_stalls'])
   overhead=sum(r[k] for k in ['static_words','parameter_stalls','source_load_words','origin_words','source_load_stalls'])
   if name=='d3_interleave':
    eq(r['total_cycles'],r['window_cycles']+r['launch_cycles']+overhead+1)
    eq(r['batches'],(n+1)//2);eq(r['launch_cycles'],r['batches'])
    eq(r['window_cycles'],r['consumer_cycles']+n+n//2)
    eq(r['shared_source_grants'],r['core_source_words']);eq(r['shared_weight_grants'],r['core_weight_words'])
    eq(r['shared_z_grants'],r['core_z_vector_reads']+r['core_z_writes']+r['core_z_scalar_reads'])
    eq(r['shared_psum_grants'],r['core_psum_reads']+r['core_psum_writes'])
    eq(r['shared_alu_grants'],r['core_first_issues']+r['core_mac_issues'])
    eq(r['conflict_cycles'],r['core_arbitration_stalls'])
   else:eq(r['total_cycles'],r['consumer_cycles']+overhead+2*n+1)
   if name=='d1_forward':
    eq(r['core_psum_reads'],beats if r['mode']==0 else 0)
    eq(r['core_psum_writes'],0 if r['mode']==2 else beats)
   else:eq(r['core_psum_reads'],beats);eq(r['core_psum_writes'],beats)
   if name=='d2_halo':
    first=r.get('first_tile',0); eligible=sum(t%160!=0 for t in range(first+1,first+n)) if r['mode'] else 0
    eq(r['halo_tiles'],eligible);eq(r['retained_source_words'],768*eligible)
    eq(r['source_load_words'],1536*n-768*eligible)
   else:eq(r['source_load_words'],1536*n)
 rows_by[name]=rows
 summary[name]={'jobs':len(rows),'values_each_checkpoint':sum(r['outputs'] for r in rows),'files':sorted(set(r['receipt'] for r in rows))}
 groups={}
 for r in rows:
  key=(r['receipt'],r.get('fixture'),r.get('first_tile'),r.get('tiles'),r['stall'],r['command'])
  groups.setdefault(key,{})[r['mode']]=r
 samework=['core_source_words','core_weight_words','core_second_weight_words','core_local_source_reads','core_z_vector_reads','core_z_scalar_reads','core_z_writes','core_first_issues','core_dual_updates','core_mac_issues']
 for group in groups.values():
  base=group[min(group)]
  for r in group.values():
   for k in samework:eq(r[k],base[k])
  if name=='d1_forward' and 1 in group and 2 in group:
   for k in group[1]:
    if isinstance(group[1][k],int) and k not in ['mode','core_psum_writes']:eq(group[1][k],group[2][k])
  if name=='d2_halo' and not base['stall']:
   eq(group[0]['total_cycles']-group[1]['total_cycles'],group[1]['retained_source_words'])
 # Trace common functional counts against the old frozen strongest producer on single-tile fixtures.
 old=json.loads((O/'consumer_packed/results.json').read_text())
 old={(r['fixture'],r['stall'],r['command']):r for r in old if r['mode']==15}
 for r in rows:
  if 'fixture' in r:
   b=old[r['fixture'],r['stall'],r['command']]
   for k in samework:eq(r[k],b[k])

# Final three-mode wrapper must preserve the original two-mode receipts.
D=H/'d3_interleave'
for p in (D/'prior_two_mode').glob('results*.json'):
 old=json.loads(p.read_text());new=json.loads((D/p.name).read_text())
 def key(r):return (r.get('fixture'),r.get('first_tile'),r.get('tiles'),r['mode'],r['stall'],r['command'])
 lookup={key(r):r for r in new}
 for r in old:
  for k,v in r.items():
   if isinstance(v,int):eq(v,lookup[key(r)][k])

# A separate address/state simulation of D2's finite ring for the complete real frame.
source=np.load(O/'data/first_source_words.npy',mmap_mode='r')
mem=np.zeros((96,4,4),np.uint16);phase=0;ext=pad=retained=0
for tid in range(19200):
 y,x=2*(tid//160)-1,2*(tid%160)-1
 reuse=tid>0 and tid%160!=0
 phase=(phase^2) if reuse else 0
 for yy in range(4):
  for xx in range(2 if reuse else 0,4):
   valid=0<=y+yy<240 and 0<=x+xx<320
   mem[:,yy,xx^phase]=source[:,y+yy,x+xx] if valid else 0
   ext+=96*valid;pad+=96*(not valid)
 for yy in range(4):
  for xx in range(4):
   expected=source[:,y+yy,x+xx] if 0<=y+yy<240 and 0<=x+xx<320 else np.zeros(96,np.uint16)
   eq(bool(np.array_equal(mem[:,yy,xx^phase],expected)),True)
 retained+=768*reuse
full=next(r for r in rows_by['d2_halo'] if r['receipt']=='results_full.json' and r['mode']==1)
eq(full['external_source_words'],ext);eq(full['padding_words'],pad);eq(full['retained_source_words'],retained)
summary['author_receipt_and_geometry_checks']=checks
summary['not_independent_review']=True
(H/'audit_results.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,rows in rows_by.items():
 with (H/name/'benefits.csv').open('w') as f:
  keys=['receipt','fixture','first_tile','tiles','mode','stall','command','total_cycles','static_words','source_load_words','external_source_words','padding_words','core_weight_words','core_z_vector_reads','core_z_writes','core_psum_reads','core_psum_writes','core_first_issues','core_mac_issues','core_output_stalls','consumer_cycles','consumer_join_wait_cycles','consumer_output_stalls','output_beats']
  if name=='d2_halo':keys+=['halo_tiles','retained_source_words']
  if name=='d3_interleave':keys+=['window_cycles','conflict_cycles','both_compute_cycles','shared_alu_grants']
  w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
